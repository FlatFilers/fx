//! L2: one session and its lifecycle.
//!
//! - `openNew` touches nothing on disk. `set` lines are held in memory until
//!   the first `turn_started`, which publishes the session (D2).
//! - `append` assigns seq and ts under the Session mutex, checks the batch
//!   with the pure rules in fold.zig, writes it with one call, and syncs
//!   after the unlock when the batch holds a durable-class event (D3).
//! - `openResume` takes the flock, cuts a torn tail, folds from the newest
//!   snapshot, and repairs what a crash left open (D1).
//! - `close` interrupts an open turn, appends `closed`, syncs, unlocks.
//! - `readPage` reads any session without a lock.
//!
//! Locks, always in this order: the flock on `{id}/lock` (one writing
//! process), then `Session.mutex` (one call at a time), then
//! `Session.sync_mutex` (one fsync at a time). Hosts never lock anything.

const std = @import("std");
const schema = @import("schema.zig");
const storage = @import("storage.zig");
const log_mod = @import("log.zig");
const fold = @import("fold.zig");
const diag = @import("diag.zig");
const trace = if (storage.hooks) @import("trace.zig") else struct {};

const hooks = storage.hooks;
const Log = log_mod.Log;

/// D7, tuned by `zig build bench`: at 10K turns resume takes 1.2 ms at
/// 256 KB and 0.4 ms at 64 KB. Smaller distances gain little and cost more
/// log bytes when the folded state (settings, children) is large.
pub const default_snapshot_every_bytes: u64 = 64 << 10;

pub const Options = struct {
    /// How long `openResume` retries a held flock before Busy.
    lock_wait_ms: u64 = 2000,
    /// A snapshot follows once this many bytes were appended since the last
    /// one (D7). A `compacted` line always gets one.
    snapshot_every_bytes: u64 = default_snapshot_every_bytes,
    /// The index is rewritten with one line per id past this size.
    index_compact_bytes: u64 = 1 << 20,
};

/// What every Session of one manager shares. Owned by L4; outlives them.
pub const Env = struct {
    /// Thread-safe when Sessions are used from several threads.
    gpa: std.mem.Allocator,
    s: storage.Storage,
    root: storage.Dir,
    options: Options = .{},
    diagnostics: ?diag.Sink = null,
    observer: if (hooks) ?Observer else void = if (hooks) null else {},
    catalog_observer: if (hooks) ?CatalogObserver else void = if (hooks) null else {},
    planted: if (hooks) trace.Planted else void = if (hooks) .none else {},

    /// Hooks only: reports a catalog-level step (`tla/Catalog.tla`).
    pub fn observeCatalog(env: *const Env, id: []const u8, what: CatalogStep) void {
        if (hooks) {
            const observer = env.catalog_observer orelse return;
            observer.notify(observer.context, id, what);
        }
    }

    fn nowMs(env: *const Env) u64 {
        return std.math.cast(u64, std.Io.Timestamp.now(env.s.io, .real).toMilliseconds()) orelse 0;
    }

    pub fn isPlanted(env: *const Env, bug: anytype) bool {
        return if (hooks) env.planted == bug else false;
    }
};

/// Hooks only: what the Session just did, for the spec tracers in tests.
pub const Observed = union(enum) {
    opened_new,
    made_tmp,
    /// One line reached the file (observed mode writes line by line).
    wrote_line: struct { seq: u64, kind: schema.Kind, cause: Cause },
    publish_synced,
    renamed,
    published,
    /// Resume took the lock and repaired an open turn (if any).
    reopened,
    closed,
    /// `append` holds the mutex / is about to release it after writing.
    mutex_acquired,
    mutex_releasing,
    /// The flock was released by `close`.
    unlocked,
};

/// Hooks only: catalog-level steps, for the Catalog tracer in tests.
pub const CatalogStep = enum { index_put, index_del, trashed, purged, healed, rebuilt };

pub const CatalogObserver = struct {
    context: *anyopaque,
    notify: *const fn (context: *anyopaque, id: []const u8, what: CatalogStep) void,
};

pub const Cause = enum { header, host, snapshot, close, interrupt_repair, child_repair, workspace_repair };

pub const Observer = struct {
    context: *anyopaque,
    notify: *const fn (context: *anyopaque, session: *Session, what: Observed) void,
};

pub const AppendError = error{ InvalidTransition, SessionClosed, TooLarge, OutOfMemory } || storage.IoFault;
pub const OpenError = error{ NotFound, Busy, ChildSession, Corrupt, UnsupportedVersion, OutOfMemory } || storage.IoFault;

pub const Identity = struct {
    id: []u8,
    workspace: []u8,
    role: schema.Role,
    host: schema.Host,
    parent: ?[]u8 = null,
    forked_from: ?Origin = null,

    pub const Origin = struct { id: []u8, seq: u64 };

    fn fromCreated(gpa: std.mem.Allocator, c: schema.Body.Created) error{OutOfMemory}!Identity {
        var identity: Identity = .{
            .id = try gpa.dupe(u8, c.id),
            .workspace = &.{},
            .role = c.role,
            .host = c.host,
        };
        errdefer identity.deinit(gpa);
        identity.workspace = try gpa.dupe(u8, c.workspace);
        if (c.parent) |parent| identity.parent = try gpa.dupe(u8, parent);
        if (c.forked_from) |origin| identity.forked_from = .{ .id = try gpa.dupe(u8, origin.id), .seq = origin.seq };
        return identity;
    }

    fn created(identity: *const Identity) schema.Body.Created {
        return .{
            .id = identity.id,
            .workspace = identity.workspace,
            .role = identity.role,
            .host = identity.host,
            .parent = identity.parent,
            .forked_from = if (identity.forked_from) |o| .{ .id = o.id, .seq = o.seq } else null,
        };
    }

    /// Child lines at or before this seq belong to a fork's source.
    fn forkSeq(identity: *const Identity) u64 {
        return if (identity.forked_from) |o| o.seq else 0;
    }

    fn deinit(identity: *Identity, gpa: std.mem.Allocator) void {
        gpa.free(identity.id);
        gpa.free(identity.workspace);
        if (identity.parent) |parent| gpa.free(parent);
        if (identity.forked_from) |origin| gpa.free(origin.id);
    }
};

const Live = struct {
    dir: storage.Dir,
    log: Log,
    lock: storage.File,
};

const Phase = union(enum) {
    /// Before the first turn: nothing on disk.
    held,
    live: Live,
    /// A write or sync failed; durability is unknown until a reopen.
    failed: ?Live,
    closed,
};

/// The last written seq, which `syncThrough` reads without `mutex` so that
/// appends go on during an fsync. A 64-bit atomic where the target has one;
/// on 32-bit targets such as fx's wasm build, a value behind its own brief
/// lock (never held across an fsync).
const SeqCell = if (@bitSizeOf(usize) >= 64) struct {
    value: std.atomic.Value(u64) = .init(0),

    fn store(cell: *@This(), _: std.Io, seq: u64) void {
        cell.value.store(seq, .release);
    }

    fn load(cell: *@This(), _: std.Io) u64 {
        return cell.value.load(.acquire);
    }
} else struct {
    mutex: std.Io.Mutex = .init,
    value: u64 = 0,

    fn store(cell: *@This(), io: std.Io, seq: u64) void {
        cell.mutex.lockUncancelable(io);
        defer cell.mutex.unlock(io);
        cell.value = seq;
    }

    fn load(cell: *@This(), io: std.Io) u64 {
        cell.mutex.lockUncancelable(io);
        defer cell.mutex.unlock(io);
        return cell.value;
    }
};

pub const Session = struct {
    env: *const Env,
    mutex: std.Io.Mutex = .init,
    sync_mutex: std.Io.Mutex = .init,

    identity: Identity,
    /// Under `mutex`.
    state: fold.State = .{},
    phase: Phase,
    /// Import only (D8): keep the original timestamp of each event.
    import_ts: ?u64 = null,
    /// `ts` of line 1: set by an import, else by the first publish or fork,
    /// or read back on resume.
    created_ms: ?u64 = null,
    /// `ts` of the newest line, never before `created_ms`; 0 until the
    /// first write (D20). Under `mutex`.
    updated_ms: u64 = 0,

    // Scratch and held lines, under `mutex`.
    held: std.ArrayList(u8) = .empty,
    held_lines: u64 = 0,
    batch: std.ArrayList(u8) = .empty,
    bounds: std.ArrayList(usize) = .empty,
    bodies: std.ArrayList(schema.Body) = .empty,
    /// Log offset just past the newest snapshot, or 0.
    snapshot_base: u64 = 0,

    // Durability, under `sync_mutex`.
    sync_file: ?storage.File = null,
    synced_seq: u64 = 0,
    /// Written under `mutex`, read under `sync_mutex`.
    written_seq: SeqCell = .{},
    /// A sync that failed outside `mutex`, as a `FaultCode`; `none` until then.
    sync_fault: std.atomic.Value(u8) = .init(@intFromEnum(FaultCode.none)),
    /// Under `mutex`: the cause of the write or sync that failed the
    /// session. Every later call reports it (D40).
    fault: ?storage.IoFault = null,

    pub fn id(session: *const Session) []const u8 {
        return session.identity.id;
    }

    /// Whether appending `events` leaves this session held in memory, so it
    /// needs no disk, not even the root folder (D2). A publishing thread
    /// always opens the root first, so a stale answer is harmless.
    pub fn staysHeld(session: *Session, events: []const fold.Event) bool {
        const io = session.env.s.io;
        session.mutex.lockUncancelable(io);
        defer session.mutex.unlock(io);
        if (session.phase != .held) return false;
        for (events) |event| {
            if (event == .turn_started) return false;
        }
        return true;
    }

    /// A page of this session's own lines through its open log: no open and
    /// no scan for the end, so paging the current session's transcript
    /// costs only the reads. Holds `mutex` while it reads one page, so a
    /// concurrent `close` cannot take the file away. Before the first turn
    /// nothing is on disk: an empty page.
    pub fn readPage(
        session: *Session,
        gpa: std.mem.Allocator,
        from: From,
        direction: Direction,
        limit: usize,
    ) (error{ SessionClosed, OutOfMemory } || storage.IoFault)!Page {
        const io = session.env.s.io;
        session.mutex.lockUncancelable(io);
        defer session.mutex.unlock(io);
        const log: *const Log = switch (session.phase) {
            .held => return .{ .arena = .init(gpa), .entries = &.{}, .next = null, .damaged = false },
            .live => |*live| &live.log,
            // After a failed write, the lines up to the tracked end are whole.
            .failed => |*maybe| if (maybe.*) |*live| &live.log else return error.SessionClosed,
            .closed => return error.SessionClosed,
        };
        return readLines(session.env.s, gpa, log.file, log.end, from, direction, limit);
    }

    /// An owned copy of the folded state, taken under the mutex.
    pub fn stateCopy(session: *Session, gpa: std.mem.Allocator) error{OutOfMemory}!fold.State {
        const io = session.env.s.io;
        session.mutex.lockUncancelable(io);
        defer session.mutex.unlock(io);
        var copy = try session.state.clone(gpa);
        copy.created_ms = session.created_ms orelse 0;
        copy.updated_ms = session.updated_ms;
        return copy;
    }

    /// Lines stamped `ts` reached the log.
    fn wrote(session: *Session, ts: u64) void {
        session.updated_ms = @max(ts, session.created_ms orelse 0);
    }

    /// Appends a batch and returns the seq of its last line. The batch is
    /// written with one call; it is checked as a whole, so a refused batch
    /// writes nothing. Returns after the fsync if the batch holds a
    /// durable-class event.
    pub fn append(session: *Session, events: []const fold.Event) AppendError!u64 {
        return (try session.appendReport(events, null)).last_seq;
    }

    pub const Appended = struct {
        last_seq: u64,
        /// This call published the session (its first turn).
        published: bool,
        /// The batch set the title or the workspace.
        listing_changed: bool,
    };

    /// `append`, plus what the catalog needs to know. `ts_ms` stamps the
    /// batch with an original time (import only).
    pub fn appendReport(session: *Session, events: []const fold.Event, ts_ms: ?u64) AppendError!Appended {
        const io = session.env.s.io;
        var sync_through: ?u64 = null;
        const result = blk: {
            session.mutex.lockUncancelable(io);
            defer session.mutex.unlock(io);
            session.observe(.mutex_acquired);
            const was_held = session.phase == .held;
            if (ts_ms) |ts| session.import_ts = ts;
            const last = try session.appendLocked(events, &sync_through);
            session.observe(.mutex_releasing);
            break :blk Appended{
                .last_seq = last,
                .published = was_held and session.phase == .live,
                // Only a published session is in the index.
                .listing_changed = session.phase == .live and changesListing(events),
            };
        };
        if (sync_through) |seq| try session.syncThrough(seq);
        return result;
    }

    fn appendLocked(session: *Session, events: []const fold.Event, sync_through: *?u64) AppendError!u64 {
        switch (session.phase) {
            .closed => return error.SessionClosed,
            .failed => return session.fault orelse error.Io,
            .held, .live => {},
        }
        if (faultFromCode(session.sync_fault.load(.acquire))) |cause| {
            session.markFailed(cause);
            return cause;
        }
        if (events.len == 0) return session.state.last_seq;
        const gpa = session.env.gpa;
        try session.bodies.resize(gpa, events.len);
        const bodies = session.bodies.items;
        fold.plan(&session.state, events, bodies) catch |err| {
            if (!session.plantedItemOutsideTurn(events, bodies)) return err;
        };
        const ts = session.timestamp();
        switch (session.phase) {
            .held => {
                // Nothing exists on disk yet, so no blob can be referenced.
                if (hasBlobRefs(bodies)) return error.InvalidTransition;
                if (!hasTurnStart(bodies)) {
                    try session.hold(bodies, ts);
                } else {
                    // The publish syncs every line it writes.
                    try session.publish(bodies, ts);
                    try session.maybeSnapshot(&session.phase.live, bodies, ts);
                }
            },
            .live => |*live| {
                try session.checkBlobRefs(live, bodies);
                try session.writeBodies(live, bodies, ts, .host);
                try session.maybeSnapshot(live, bodies, ts);
                if (needsSync(bodies)) sync_through.* = session.state.last_seq;
            },
            .failed, .closed => unreachable,
        }
        return session.state.last_seq;
    }

    /// Hooks only: the planted TurnLifecycle bug writes an item anyway.
    fn plantedItemOutsideTurn(session: *Session, events: []const fold.Event, bodies: []schema.Body) bool {
        if (!session.env.isPlanted(.accept_item_outside_turn)) return false;
        if (events.len != 1 or events[0] != .item or session.phase != .live) return false;
        bodies[0] = .{ .item = .{ .turn = session.state.last_turn, .type = events[0].item.type, .data = events[0].item.data } };
        return true;
    }

    /// Before the first turn, a batch without `turn_started` may only hold
    /// settings; they wait in memory, already framed (seq 2 onward).
    fn hold(session: *Session, bodies: []const schema.Body, ts: u64) AppendError!void {
        for (bodies) |body| if (body != .set) return error.InvalidTransition;
        const gpa = session.env.gpa;
        const start = session.held.items.len;
        errdefer session.held.shrinkRetainingCapacity(start);
        for (bodies, 0..) |body, i| {
            try session.frame(&session.held, 2 + session.held_lines + i, ts, body);
        }
        for (bodies, 0..) |body, i| {
            try fold.apply(gpa, &session.state, 0, .{ .seq = 2 + session.held_lines + i, .offset = 0, .body = body });
        }
        session.held_lines += bodies.len;
    }

    /// The first turn: stage the session in `.tmp/{id}`, make line 1
    /// durable, then rename it into place and sync the root, so a visible
    /// session always has a durable first line (`tla/Lifecycle.tla`).
    fn publish(session: *Session, bodies: []const schema.Body, ts: u64) AppendError!void {
        const gpa = session.env.gpa;
        // Nothing is visible; later calls name the cause (D40).
        errdefer |err| {
            session.phase = .{ .failed = null };
            if (session.fault == null) session.fault = asIoFault(err);
        }

        // Frame everything first: line 1, the held lines, then the batch.
        session.batch.clearRetainingCapacity();
        session.bounds.clearRetainingCapacity();
        const created = session.created_ms orelse ts;
        try session.frameMarked(1, created, .{ .session_created = session.identity.created() });
        session.created_ms = created;
        var at: usize = 0;
        while (at < session.held.items.len) {
            const nl = std.mem.findScalarPos(u8, session.held.items, at, '\n').?;
            try session.batch.appendSlice(gpa, session.held.items[at .. nl + 1]);
            try session.bounds.append(gpa, session.batch.items.len);
            at = nl + 1;
        }
        const first_batch_seq = 2 + session.held_lines;
        for (bodies, 0..) |body, i| try session.frameMarked(first_batch_seq + i, ts, body);

        const live = try session.stage(null);

        // Fold the batch; line 1 has no state and the held lines are folded.
        for (bodies, 0..) |body, i| {
            const line_index: usize = @intCast(1 + session.held_lines + i);
            try fold.apply(gpa, &session.state, 0, .{
                .seq = first_batch_seq + i,
                .offset = session.bounds.items[line_index - 1],
                .body = body,
            });
        }
        session.phase = .{ .live = live };
        session.wrote(ts);
        session.held.clearAndFree(gpa);
        session.held_lines = 0;
        session.written_seq.store(session.env.s.io, session.state.last_seq);
        session.setSyncFile(live.log.file, session.state.last_seq);
    }

    /// The staged publish shared by the first turn and fork: `.tmp/{id}`,
    /// the lines framed in `session.batch`, blobs linked from a fork's
    /// source, a synced log, the lock, a synced folder, the rename into
    /// place, and a synced root. Only then is the session visible, always
    /// with a durable first line (`tla/Lifecycle.tla`).
    fn stage(session: *Session, source_blobs: ?storage.Dir) AppendError!Live {
        const env = session.env;
        const s = env.s;
        const id_ = session.identity.id;
        const tmp = s.ensureDir(env.root, ".tmp") catch |io_err| return storage.ioFault(io_err);
        defer s.closeDir(tmp);
        s.makeDir(tmp, id_) catch |err| switch (err) {
            // A leftover from a crashed attempt with the same id (an import).
            error.AlreadyExists => {
                s.deleteTree(tmp, id_) catch |io_err| return storage.ioFault(io_err);
                s.makeDir(tmp, id_) catch |io_err| return storage.ioFault(io_err);
            },
            else => |io_err| return storage.ioFault(io_err),
        };
        session.observe(.made_tmp);
        const dir = s.openDir(tmp, id_) catch |io_err| return storage.ioFault(io_err);
        errdefer s.closeDir(dir);
        {
            const blobs = s.ensureDir(dir, "blobs") catch |io_err| return storage.ioFault(io_err);
            defer s.closeDir(blobs);
            if (source_blobs) |source| {
                try session.linkBlobs(source, blobs);
                s.syncDir(blobs) catch |io_err| return storage.ioFault(io_err);
            }
        }
        var log = Log.create(s, dir, "log.jsonl", .{}) catch |io_err| return storage.ioFault(io_err);
        errdefer log.close();
        session.writeFramed(&log, 1, .header) catch |io_err| return storage.ioFault(io_err);

        const rename_first = env.isPlanted(.rename_before_fsync);
        if (rename_first) try session.publishRename(tmp);
        s.sync(log.file) catch |io_err| return storage.ioFault(io_err);
        session.observe(.publish_synced);
        const lock = s.createFile(dir, "lock") catch |io_err| return storage.ioFault(io_err);
        errdefer s.closeFile(lock);
        if (!(s.tryLock(lock) catch |io_err| return storage.ioFault(io_err))) return error.Io;
        s.syncDir(dir) catch |io_err| return storage.ioFault(io_err);
        if (!rename_first) try session.publishRename(tmp);
        s.syncDir(env.root) catch |io_err| return storage.ioFault(io_err);
        session.observe(.published);
        return .{ .dir = dir, .log = log, .lock = lock };
    }

    /// Hard-links every blob of a fork's source (D6), copying when a link
    /// is refused, for example across file systems.
    fn linkBlobs(session: *Session, source: storage.Dir, target: storage.Dir) AppendError!void {
        const s = session.env.s;
        var skipped_one = false;
        var listing = s.list(source);
        while (listing.next() catch |io_err| return storage.ioFault(io_err)) |entry| {
            if (entry.kind != .file or !schema.validBlobHash(entry.name)) continue;
            if (session.env.isPlanted(.fork_skips_blob_link) and !skipped_one) {
                skipped_one = true;
                continue;
            }
            s.link(source, entry.name, target, entry.name) catch |err| switch (err) {
                error.AlreadyExists => {},
                error.NoSpace => return error.NoSpaceLeft,
                else => try copyBlob(session.env, source, target, entry.name),
            };
        }
    }

    /// Stores `bytes` as a blob of this session and returns its name, the
    /// SHA-256 hash (D6). Returns only once the blob and its name are
    /// durable: a line referring to it may become durable through any later
    /// fsync, including one another thread starts. Storing the same bytes
    /// again costs one stat.
    pub fn putBlob(session: *Session, bytes: []const u8) AppendError![schema.blob_hash_len]u8 {
        if (bytes.len > max_blob_bytes) return error.TooLarge;
        const hash = schema.blobHash(bytes);
        const env = session.env;
        const s = env.s;
        const io = s.io;
        // Hold the mutex only to check the phase and borrow the folder.
        const dir = blk: {
            session.mutex.lockUncancelable(io);
            defer session.mutex.unlock(io);
            switch (session.phase) {
                .live => |live| break :blk s.openDir(live.dir, "blobs") catch |io_err| return storage.ioFault(io_err),
                .held => return error.InvalidTransition,
                .failed => return session.fault orelse error.Io,
                .closed => return error.SessionClosed,
            }
        };
        defer s.closeDir(dir);
        if (s.stat(dir, &hash)) |_| return hash else |_| {}
        var tmp_name: [1 + schema.blob_hash_len + 1 + 16 + 4]u8 = undefined;
        var suffix: [8]u8 = undefined;
        io.random(&suffix);
        const name = std.fmt.bufPrint(&tmp_name, ".{s}.{x}.tmp", .{ &hash, &suffix }) catch unreachable;
        const file = s.createFile(dir, name) catch |io_err| return storage.ioFault(io_err);
        var file_open = true;
        defer if (file_open) s.closeFile(file);
        errdefer s.deleteFile(dir, name) catch {};
        s.writeAt(file, bytes, 0) catch |io_err| return storage.ioFault(io_err);
        s.sync(file) catch |io_err| return storage.ioFault(io_err);
        s.closeFile(file);
        file_open = false;
        s.rename(dir, name, dir, &hash) catch |io_err| return storage.ioFault(io_err);
        s.syncDir(dir) catch |io_err| return storage.ioFault(io_err);
        return hash;
    }

    /// Every blob an item refers to must already exist in this session
    /// (`tla/Fork.tla` `RefsExist`).
    fn checkBlobRefs(session: *Session, live: *Live, bodies: []const schema.Body) AppendError!void {
        if (!hasBlobRefs(bodies)) return;
        const s = session.env.s;
        const dir = s.openDir(live.dir, "blobs") catch |io_err| return storage.ioFault(io_err);
        defer s.closeDir(dir);
        for (bodies) |body| switch (body) {
            .item => |piece| for (piece.blobs) |hash| {
                if (!schema.validBlobHash(hash)) return error.InvalidTransition;
                _ = s.stat(dir, hash) catch return error.InvalidTransition;
            },
            else => {},
        };
    }

    fn publishRename(session: *Session, tmp: storage.Dir) AppendError!void {
        session.env.s.rename(tmp, session.identity.id, session.env.root, session.identity.id) catch |io_err| return storage.ioFault(io_err);
        session.observe(.renamed);
    }

    /// Frames `bodies` at the end of the live log, writes them, and folds them.
    fn writeBodies(session: *Session, live: *Live, bodies: []const schema.Body, ts: u64, cause: Cause) AppendError!void {
        const gpa = session.env.gpa;
        const first_seq = live.log.next_seq;
        const base = live.log.end;
        session.batch.clearRetainingCapacity();
        session.bounds.clearRetainingCapacity();
        for (bodies, 0..) |body, i| try session.frameMarked(first_seq + i, ts, body);
        session.writeFramed(&live.log, first_seq, cause) catch |err| {
            const fault = storage.ioFault(err);
            session.markFailed(fault);
            return fault;
        };
        const fork_seq = session.identity.forkSeq();
        for (bodies, 0..) |body, i| {
            const offset = base + if (i == 0) 0 else session.bounds.items[i - 1];
            try fold.apply(gpa, &session.state, fork_seq, .{ .seq = first_seq + i, .offset = offset, .body = body });
        }
        session.wrote(ts);
        session.written_seq.store(session.env.s.io, session.state.last_seq);
    }

    /// Writes `session.batch` (lines ending at `session.bounds`) in one
    /// append, observed or not, so traced runs and crash matrices see the
    /// write pattern production has. An observer then sees each line; a
    /// tracer reading the disk bounds its view by the line's seq.
    fn writeFramed(session: *Session, log: *Log, first_seq: u64, cause: Cause) log_mod.AppendError!void {
        try log.append(session.batch.items, session.bounds.items.len);
        if (!session.observing()) return;
        var start: usize = 0;
        for (session.bounds.items, 0..) |end, i| {
            const line = session.batch.items[start..end];
            const header = log_mod.checkLine(line) catch unreachable;
            // In a publish only line 1 is the header; the held and batch
            // lines after it come from the host.
            const line_cause: Cause = if (cause != .header) cause else if (i == 0) .header else .host;
            session.observe(.{ .wrote_line = .{ .seq = first_seq + i, .kind = header.kind.?, .cause = line_cause } });
            start = end;
        }
    }

    fn maybeSnapshot(session: *Session, live: *Live, bodies: []const schema.Body, ts: u64) AppendError!void {
        var compacted = false;
        for (bodies) |body| {
            if (body == .compacted) compacted = true;
        }
        if (!compacted and live.log.end - session.snapshot_base < session.env.options.snapshot_every_bytes) return;
        try session.writeSnapshot(live, ts);
    }

    /// A snapshot is a cache: one that does not fit in a line is skipped and
    /// reported, never failing the append whose batch is already written.
    fn writeSnapshot(session: *Session, live: *Live, ts: u64) AppendError!void {
        const gpa = session.env.gpa;
        var encoded: std.ArrayList(u8) = .empty;
        defer encoded.deinit(gpa);
        const body = try session.snapshotBody(&encoded);
        session.writeBodies(live, &.{body}, ts, .snapshot) catch |err| switch (err) {
            error.TooLarge => diag.report(session.env.diagnostics, .{
                .kind = .snapshot_skipped,
                .session_id = session.identity.id,
                .count = encoded.items.len,
            }),
            else => return err,
        };
        session.snapshot_base = live.log.end;
    }

    /// A snapshot of the state folded so far. The body borrows `encoded`.
    fn snapshotBody(session: *Session, encoded: *std.ArrayList(u8)) error{OutOfMemory}!schema.Body {
        const gpa = session.env.gpa;
        if (session.env.isPlanted(.snapshot_drops_field)) {
            var copy = try session.state.clone(gpa);
            defer copy.deinit(gpa);
            if (copy.prefs) |p| gpa.free(p);
            copy.prefs = null;
            try fold.encodeState(gpa, encoded, &copy);
        } else {
            try fold.encodeState(gpa, encoded, &session.state);
        }
        return .{ .snapshot = .{
            .covers_seq = session.state.last_seq,
            .state = encoded.items,
            .compaction_offset = session.state.compaction_offset,
        } };
    }

    fn frame(session: *Session, out: *std.ArrayList(u8), seq: u64, ts: u64, body: schema.Body) AppendError!void {
        const gpa = session.env.gpa;
        var fields: std.ArrayList(u8) = .empty;
        defer fields.deinit(gpa);
        try schema.appendBody(gpa, &fields, body);
        log_mod.appendLine(gpa, out, seq, ts, std.meta.activeTag(body), fields.items) catch |err| switch (err) {
            error.TooLarge => return error.TooLarge,
            error.OutOfMemory => return error.OutOfMemory,
        };
    }

    fn frameMarked(session: *Session, seq: u64, ts: u64, body: schema.Body) AppendError!void {
        try session.frame(&session.batch, seq, ts, body);
        try session.bounds.append(session.env.gpa, session.batch.items.len);
    }

    /// Returns once every line through `seq` is synced. Runs outside
    /// `mutex`, so appends from other threads continue during the fsync;
    /// one fsync covers every line written before it started.
    fn syncThrough(session: *Session, seq: u64) AppendError!void {
        const io = session.env.s.io;
        session.sync_mutex.lockUncancelable(io);
        defer session.sync_mutex.unlock(io);
        if (session.synced_seq >= seq) return;
        const file = session.sync_file orelse return error.SessionClosed;
        const target = session.written_seq.load(io);
        std.debug.assert(target >= seq);
        session.env.s.sync(file) catch |err| {
            const fault = storage.ioFault(err);
            // Only the first cause is kept; a later failure changes nothing.
            _ = session.sync_fault.cmpxchgStrong(@intFromEnum(FaultCode.none), @intFromEnum(faultCode(fault)), .release, .monotonic);
            return fault;
        };
        session.synced_seq = target;
    }

    fn setSyncFile(session: *Session, file: ?storage.File, synced: u64) void {
        const io = session.env.s.io;
        session.sync_mutex.lockUncancelable(io);
        defer session.sync_mutex.unlock(io);
        session.sync_file = file;
        session.synced_seq = synced;
    }

    /// Fails a live session with `cause`; the first cause is the one kept.
    fn markFailed(session: *Session, cause: storage.IoFault) void {
        switch (session.phase) {
            .live => |live| session.phase = .{ .failed = live },
            else => {},
        }
        if (session.fault == null) session.fault = cause;
    }

    fn timestamp(session: *const Session) u64 {
        return session.import_ts orelse session.env.nowMs();
    }

    fn observing(session: *const Session) bool {
        return if (hooks) session.env.observer != null else false;
    }

    fn observe(session: *Session, what: Observed) void {
        if (hooks) {
            const observer = session.env.observer orelse return;
            observer.notify(observer.context, session, what);
        }
    }

    /// Ends the session: an open turn is interrupted, `closed` is appended
    /// and synced, and the flock is released. Later calls get SessionClosed.
    /// Calling it again does nothing. The handle stays valid until `destroy`.
    pub fn close(session: *Session) AppendError!void {
        _ = try session.closeReport();
    }

    /// `close`, returning whether the session exists on disk (it was
    /// published) and this call closed it.
    pub fn closeReport(session: *Session) AppendError!bool {
        const io = session.env.s.io;
        session.mutex.lockUncancelable(io);
        defer session.mutex.unlock(io);
        const was_live = session.phase == .live;
        try session.closeLocked();
        return was_live;
    }

    fn closeLocked(session: *Session) AppendError!void {
        switch (session.phase) {
            .closed => return,
            .held => {
                if (session.held_lines > 0) diag.report(session.env.diagnostics, .{
                    .kind = .held_lines_dropped,
                    .session_id = session.identity.id,
                    .count = session.held_lines,
                });
                session.held.clearAndFree(session.env.gpa);
                session.held_lines = 0;
                session.phase = .closed;
            },
            .failed => |maybe_live| {
                if (maybe_live) |live| session.release(live);
                session.phase = .closed;
            },
            .live => |*live| {
                var result: AppendError!void = {};
                var bodies_buffer: [2]schema.Body = undefined;
                var n: usize = 0;
                if (session.state.open_turn) |turn| {
                    bodies_buffer[n] = .{ .turn_interrupted = .{ .turn = turn, .reason = .closed } };
                    n += 1;
                }
                bodies_buffer[n] = .closed;
                n += 1;
                if (session.writeBodies(live, bodies_buffer[0..n], session.timestamp(), .close)) {
                    session.observe(.closed);
                    result = session.syncThrough(session.state.last_seq);
                } else |err| result = err;
                const resources = switch (session.phase) {
                    .live => |l| l,
                    .failed => |l| l.?,
                    else => unreachable,
                };
                session.release(resources);
                session.phase = .closed;
                return result;
            },
        }
    }

    fn release(session: *Session, live: Live) void {
        const s = session.env.s;
        session.setSyncFile(null, session.synced_seq);
        var log = live.log;
        s.unlock(live.lock);
        session.observe(.unlocked);
        s.closeFile(live.lock);
        log.close();
        s.closeDir(live.dir);
    }

    /// Releases the lock and files without writing anything, as a process
    /// crash would, then frees the handle. Used when `openResume` fails
    /// midway, and by tests to simulate a crash.
    pub fn abandon(session: *Session) void {
        switch (session.phase) {
            .live => |live| session.release(live),
            .failed => |maybe_live| if (maybe_live) |live| session.release(live),
            .held, .closed => {},
        }
        session.phase = .closed;
        session.destroy();
    }

    /// Frees the handle. The session must be closed, and no other thread
    /// may still be using it.
    pub fn destroy(session: *Session) void {
        std.debug.assert(session.phase == .closed);
        const gpa = session.env.gpa;
        session.identity.deinit(gpa);
        session.state.deinit(gpa);
        session.held.deinit(gpa);
        session.batch.deinit(gpa);
        session.bounds.deinit(gpa);
        session.bodies.deinit(gpa);
        gpa.destroy(session);
    }
};

/// Blobs above this size are refused; the adapter keeps fx's own, smaller limits.
pub const max_blob_bytes: usize = 512 << 20;

/// An I/O fault as one byte, so a sync on another thread can hand its cause
/// to the next append through an atomic.
const FaultCode = enum(u8) { none, io, no_space, access_denied, read_only, too_big };

fn faultCode(fault: storage.IoFault) FaultCode {
    return switch (fault) {
        error.Io => .io,
        error.NoSpaceLeft => .no_space,
        error.AccessDenied => .access_denied,
        error.ReadOnlyFileSystem => .read_only,
        error.FileTooBig => .too_big,
    };
}

/// The I/O fault an append failed with, if it was one.
fn asIoFault(err: AppendError) ?storage.IoFault {
    return switch (err) {
        error.Io => error.Io,
        error.NoSpaceLeft => error.NoSpaceLeft,
        error.AccessDenied => error.AccessDenied,
        error.ReadOnlyFileSystem => error.ReadOnlyFileSystem,
        error.FileTooBig => error.FileTooBig,
        else => null,
    };
}

fn faultFromCode(code: u8) ?storage.IoFault {
    return switch (std.enums.fromInt(FaultCode, code) orelse .io) {
        .none => null,
        .io => error.Io,
        .no_space => error.NoSpaceLeft,
        .access_denied => error.AccessDenied,
        .read_only => error.ReadOnlyFileSystem,
        .too_big => error.FileTooBig,
    };
}

test "a fault code round trips every I/O fault, and none is no fault" {
    const faults = [_]storage.IoFault{ error.Io, error.NoSpaceLeft, error.AccessDenied, error.ReadOnlyFileSystem, error.FileTooBig };
    for (faults) |fault| try std.testing.expectEqual(fault, faultFromCode(@intFromEnum(faultCode(fault))).?);
    try std.testing.expectEqual(@as(?storage.IoFault, null), faultFromCode(@intFromEnum(FaultCode.none)));
}

fn hasBlobRefs(bodies: []const schema.Body) bool {
    for (bodies) |body| switch (body) {
        .item => |piece| if (piece.blobs.len > 0) return true,
        else => {},
    };
    return false;
}

/// The copy fallback when a hard link is refused: the copy is synced
/// before the folder is.
fn copyBlob(env: *const Env, source: storage.Dir, target: storage.Dir, name: []const u8) AppendError!void {
    const s = env.s;
    const gpa = env.gpa;
    const from = s.openFile(source, name, .read_only) catch |io_err| return storage.ioFault(io_err);
    defer s.closeFile(from);
    const len = s.length(from) catch |io_err| return storage.ioFault(io_err);
    if (len > max_blob_bytes) return error.Io;
    const bytes = try gpa.alloc(u8, @intCast(len));
    defer gpa.free(bytes);
    if ((s.readAt(from, bytes, 0) catch |io_err| return storage.ioFault(io_err)) != bytes.len) return error.Io;
    const to = s.createFile(target, name) catch |io_err| return storage.ioFault(io_err);
    defer s.closeFile(to);
    s.writeAt(to, bytes, 0) catch |io_err| return storage.ioFault(io_err);
    s.sync(to) catch |io_err| return storage.ioFault(io_err);
}

fn changesListing(events: []const fold.Event) bool {
    for (events) |event| switch (event) {
        .set => |s| if (s.key == .title or s.key == .workspace or s.key == .language) return true,
        else => {},
    };
    return false;
}

fn hasTurnStart(bodies: []const schema.Body) bool {
    for (bodies) |body| {
        if (body == .turn_started) return true;
    }
    return false;
}

/// Durability points (system-design.md "Durable State And Schema").
fn needsSync(bodies: []const schema.Body) bool {
    for (bodies) |body| switch (body) {
        .turn_committed, .turn_interrupted, .child_spawned, .child_finished => return true,
        // Usage is durable before fx clears its usage-recovery marker
        // (`tla/Wiring.tla` UsageNeverSilent).
        .set => |s| switch (s.key) {
            .permissions, .usage => return true,
            .prefs, .title, .workspace, .language => {},
        },
        else => {},
    };
    return false;
}

// ---------------------------------------------------------------------------
// Opening

pub const NewOptions = struct {
    workspace: []const u8,
    role: schema.Role = .root,
    host: schema.Host,
    parent: ?[]const u8 = null,
    forked_from: ?schema.ForkOrigin = null,
    /// Import (D8) keeps a v1 id; otherwise a fresh id is drawn.
    id: ?[]const u8 = null,
    /// Import only: the original creation time, stamped on line 1.
    created_ms: ?u64 = null,
};

/// A new session in memory. Nothing touches the disk until the first turn.
pub fn openNew(env: *const Env, options: NewOptions) error{OutOfMemory}!*Session {
    const gpa = env.gpa;
    var fresh_id: [schema.new_id_len]u8 = undefined;
    const id_ = options.id orelse blk: {
        schema.newId(env.s.io, &fresh_id);
        break :blk &fresh_id;
    };
    var identity = try Identity.fromCreated(gpa, .{
        .id = id_,
        .workspace = options.workspace,
        .role = options.role,
        .host = options.host,
        .parent = options.parent,
        .forked_from = options.forked_from,
    });
    errdefer identity.deinit(gpa);
    const session = try gpa.create(Session);
    session.* = .{ .env = env, .identity = identity, .phase = .held, .created_ms = options.created_ms };
    session.observe(.opened_new);
    return session;
}

pub const ResumeOptions = struct {
    id: []const u8,
    workspace: []const u8,
    host: schema.Host,
    /// Required to resume a child session.
    parent: ?[]const u8 = null,
    /// How long to wait for the flock before Busy; null uses the
    /// environment's `lock_wait_ms` (D38).
    lock_wait_ms: ?u64 = null,
};

/// Opens a published session for writing: the flock (else Busy after the
/// retry window), the tail cut, line 1, the newest snapshot and the tail
/// after it, then the crash repair and one sync. Cost does not grow with
/// the session's age.
pub fn openResume(env: *const Env, options: ResumeOptions) OpenError!*Session {
    var parts = try openParts(env, options);
    const session = env.gpa.create(Session) catch {
        parts.deinit(env);
        return error.OutOfMemory;
    };
    // Ownership of every part moves into the Session here.
    session.* = .{
        .env = env,
        .identity = parts.loaded.identity,
        .state = parts.loaded.state,
        .phase = .{ .live = .{ .dir = parts.dir, .log = parts.log, .lock = parts.lock } },
        .snapshot_base = parts.loaded.snapshot_end,
        .created_ms = parts.loaded.created_ms,
        .updated_ms = parts.loaded.updated_ms,
    };
    session.written_seq.store(session.env.s.io, session.state.last_seq);
    session.setSyncFile(parts.log.file, session.state.last_seq);
    repair(session, options.workspace) catch |err| {
        session.abandon();
        return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => |io_err| storage.ioFault(io_err),
        };
    };
    return session;
}

const Parts = struct {
    dir: storage.Dir,
    lock: storage.File,
    log: Log,
    loaded: Loaded,

    fn deinit(p: *Parts, env: *const Env) void {
        p.loaded.deinit(env.gpa);
        p.log.close();
        env.s.closeFile(p.lock);
        env.s.closeDir(p.dir);
    }
};

/// Everything `openResume` needs before the Session exists; on error,
/// everything acquired so far is released here.
fn openParts(env: *const Env, options: ResumeOptions) OpenError!Parts {
    const s = env.s;
    const gpa = env.gpa;
    if (!schema.validId(options.id)) return error.NotFound;
    const dir = s.openDir(env.root, options.id) catch |err| return switch (err) {
        error.NotFound => error.NotFound,
        else => |io_err| storage.ioFault(io_err),
    };
    errdefer s.closeDir(dir);
    const lock = s.openFile(dir, "lock", .read_write) catch |err| return switch (err) {
        error.NotFound => error.Corrupt,
        else => |io_err| storage.ioFault(io_err),
    };
    errdefer s.closeFile(lock);
    if (!env.isPlanted(.skip_flock)) try acquireLock(env, lock, options.lock_wait_ms orelse env.options.lock_wait_ms);

    var opened = Log.open(gpa, s, dir, "log.jsonl", .read_write, .{}) catch |err| return switch (err) {
        error.NotFound => error.Corrupt,
        error.OutOfMemory => error.OutOfMemory,
        else => |io_err| storage.ioFault(io_err),
    };
    errdefer opened.log.close();
    if (opened.cut_bytes > 0) diag.report(env.diagnostics, .{
        .kind = .torn_tail_cut,
        .session_id = options.id,
        .count = opened.cut_bytes,
        .offset = opened.log.end,
    });
    switch (opened.verdict) {
        .clean, .torn => {},
        .corrupt => |c| {
            diag.report(env.diagnostics, .{ .kind = .opened_read_only, .session_id = options.id, .offset = c.at });
            return error.Corrupt;
        },
        .newer_version => |n| {
            diag.report(env.diagnostics, .{ .kind = .opened_read_only, .session_id = options.id, .offset = n.at });
            return error.UnsupportedVersion;
        },
    }
    if (opened.log.lineCount() == 0) return error.Corrupt;

    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    var damaged_at: u64 = 0;
    var loaded = loadFolded(env, &arena_state, opened.log.file, opened.log.end, &damaged_at) catch |err| {
        // Damage behind the tail, found while folding: reported once here.
        if (err == error.Corrupt) diag.report(env.diagnostics, .{
            .kind = .opened_read_only,
            .session_id = options.id,
            .offset = damaged_at,
        });
        return err;
    };
    errdefer loaded.deinit(gpa);
    if (!std.mem.eql(u8, loaded.identity.id, options.id)) return error.Corrupt;
    if (loaded.identity.role == .child) {
        const parent = options.parent orelse return error.ChildSession;
        const own = loaded.identity.parent orelse return error.ChildSession;
        if (!std.mem.eql(u8, parent, own)) return error.ChildSession;
    }
    return .{ .dir = dir, .lock = lock, .log = opened.log, .loaded = loaded };
}

fn acquireLock(env: *const Env, lock: storage.File, wait_ms: u64) OpenError!void {
    const io = env.s.io;
    const step_ms: u64 = 10;
    var waited: u64 = 0;
    while (true) {
        if (env.s.tryLock(lock) catch |io_err| return storage.ioFault(io_err)) return;
        if (waited >= wait_ms) return error.Busy;
        io.sleep(.fromMilliseconds(@intCast(step_ms)), .awake) catch return error.Busy;
        waited += step_ms;
    }
}

const Loaded = struct {
    identity: Identity,
    state: fold.State,
    /// Log offset just past the newest snapshot, or 0.
    snapshot_end: u64,
    /// `ts` of line 1 and of the last line.
    created_ms: u64,
    updated_ms: u64,

    fn deinit(l: *Loaded, gpa: std.mem.Allocator) void {
        l.identity.deinit(gpa);
        l.state.deinit(gpa);
    }
};

/// Line 1, then the newest snapshot found scanning backward, then the tail
/// after it. Without a snapshot, the whole log is folded.
fn loadFolded(env: *const Env, arena_state: *std.heap.ArenaAllocator, file: storage.File, end: u64, damaged_at: *u64) OpenError!Loaded {
    const gpa = env.gpa;
    const arena = arena_state.allocator();
    var first = log_mod.ForwardReader.init(gpa, env.s, file, 0, end, 1);
    defer first.deinit();
    const line1 = (first.next() catch |err| {
        damaged_at.* = first.damaged_at orelse 0;
        return mapRead(err);
    }) orelse return error.Corrupt;
    if (line1.header.kind != .session_created) return error.Corrupt;
    const created = schema.parseBody(arena, .session_created, line1.body()) catch return error.Corrupt;
    var identity = try Identity.fromCreated(gpa, created.session_created);
    errdefer identity.deinit(gpa);
    const line1_end = line1.offset + line1.bytes.len;

    // The newest snapshot, scanning back from the end.
    var snapshot: ?struct { seq: u64, end: u64, state: []const u8, ts_ms: u64 } = null;
    {
        var back = log_mod.BackwardReader.init(gpa, env.s, file, end);
        defer back.deinit();
        while (back.next() catch |err| {
            damaged_at.* = back.damaged_at orelse 0;
            return mapRead(err);
        }) |line| {
            if (line.offset < line1_end) break;
            if (line.header.kind != .snapshot) continue;
            const body = schema.parseBody(arena, .snapshot, line.body()) catch return error.Corrupt;
            snapshot = .{ .seq = line.header.seq, .end = line.offset + line.bytes.len, .state = body.snapshot.state, .ts_ms = line.header.ts_ms };
            break;
        }
    }

    var state: fold.State = .{};
    errdefer state.deinit(gpa);
    const fork_seq = identity.forkSeq();
    var from = line1_end;
    var expected: u64 = 2;
    state.last_seq = 1;
    if (snapshot) |snap| {
        state = fold.decodeState(gpa, arena, snap.state) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            error.BadBody => error.Corrupt,
        };
        // A snapshot copied from a fork's source carries the source's children.
        if (snap.seq <= fork_seq) state.clearChildren(gpa);
        state.last_seq = snap.seq;
        from = snap.end;
        expected = snap.seq + 1;
    }
    var updated_ms = if (snapshot) |snap| snap.ts_ms else line1.header.ts_ms;
    const created_ms = line1.header.ts_ms;
    var tail = log_mod.ForwardReader.init(gpa, env.s, file, from, end, expected);
    defer tail.deinit();
    var line_arena = std.heap.ArenaAllocator.init(gpa);
    defer line_arena.deinit();
    while (tail.next() catch |err| {
        damaged_at.* = tail.damaged_at orelse 0;
        return mapRead(err);
    }) |line| {
        _ = line_arena.reset(.retain_capacity);
        updated_ms = line.header.ts_ms;
        const kind = line.header.kind orelse {
            state.last_seq = line.header.seq; // unknown kinds are skipped
            state.clean_exit = false;
            continue;
        };
        const body = schema.parseBody(line_arena.allocator(), kind, line.body()) catch return error.Corrupt;
        try fold.apply(gpa, &state, fork_seq, .{ .seq = line.header.seq, .offset = line.offset, .body = body });
    }
    return .{
        .identity = identity,
        .state = state,
        .snapshot_end = if (snapshot) |snap| snap.end else 0,
        .created_ms = created_ms,
        // A fork's copied lines keep the source's older times (D20).
        .updated_ms = @max(updated_ms, created_ms),
    };
}

fn mapRead(err: log_mod.ReadError) OpenError {
    return switch (err) {
        error.Corrupt => error.Corrupt,
        error.OutOfMemory => error.OutOfMemory,
        else => |io_err| storage.ioFault(io_err),
    };
}

/// Crash repair at the end of `openResume` (D1, `tla/Subagents.tla`).
fn repair(session: *Session, workspace: []const u8) AppendError!void {
    const gpa = session.env.gpa;
    const live = &session.phase.live;
    const ts = session.timestamp();
    // An open turn from the previous owner is interrupted (D1).
    if (session.state.open_turn) |turn| {
        try session.writeBodies(live, &.{.{ .turn_interrupted = .{ .turn = turn, .reason = .crash } }}, ts, .interrupt_repair);
    }
    session.observe(.reopened);
    // Every unfinished work item gets a known outcome (tla/Subagents.tla).
    var finished: std.ArrayList(schema.Body) = .empty;
    defer finished.deinit(gpa);
    for (session.state.children.items) |child| {
        if (!child.open) continue;
        const has_log = childHasLog(session.env, child.id);
        try finished.append(gpa, .{ .child_finished = .{
            .child = child.id,
            .work_id = child.work_id,
            .outcome = if (has_log) .interrupted else .lost,
        } });
    }
    if (finished.items.len > 0) {
        // `writeBodies` folds the lines, which updates the children the
        // bodies borrow from; frame from copies.
        var copies = std.heap.ArenaAllocator.init(gpa);
        defer copies.deinit();
        for (finished.items) |*body| {
            body.child_finished.child = try copies.allocator().dupe(u8, body.child_finished.child);
            body.child_finished.work_id = try copies.allocator().dupe(u8, body.child_finished.work_id);
        }
        if (session.env.isPlanted(.repair_marks_child_twice)) {
            // Copied first: appending may move the list it came from.
            const twice = finished.items[0];
            try finished.append(gpa, twice);
        }
        try session.writeBodies(live, finished.items, ts, .child_repair);
    }
    // Resumed from another workspace: record it.
    var scratch = std.heap.ArenaAllocator.init(gpa);
    defer scratch.deinit();
    if (!std.mem.eql(u8, try currentWorkspace(session, scratch.allocator()), workspace)) {
        var value: std.ArrayList(u8) = .empty;
        defer value.deinit(gpa);
        try schema.appendJsonString(gpa, &value, workspace);
        try session.writeBodies(live, &.{.{ .set = .{ .key = .workspace, .value = value.items } }}, ts, .workspace_repair);
    }
    try session.syncThrough(session.state.last_seq);
}

/// The workspace in effect: the newest `set workspace`, else line 1's.
/// The result may point into `arena` or into the session.
pub fn currentWorkspace(session: *const Session, arena: std.mem.Allocator) error{OutOfMemory}![]const u8 {
    const raw = session.state.workspace orelse return session.identity.workspace;
    return std.json.parseFromSliceLeaky([]const u8, arena, raw, .{}) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        // Not a JSON string: compare the raw bytes, which never match a path.
        else => raw,
    };
}

fn childHasLog(env: *const Env, child: []const u8) bool {
    if (!schema.validId(child)) return false;
    const dir = env.s.openDir(env.root, child) catch return false;
    defer env.s.closeDir(dir);
    _ = env.s.stat(dir, "log.jsonl") catch return false;
    return true;
}

// ---------------------------------------------------------------------------
// Reading (lock-free)

/// A position between two lines.
pub const Cursor = struct {
    /// A line boundary in the log.
    offset: u64,
    /// Seq of the line that starts at `offset`.
    seq: u64,
};

pub const From = union(enum) { start, end, at: Cursor };
pub const Direction = enum { forward, backward };

pub const Entry = struct {
    seq: u64,
    /// Byte offset of the line: a `Cursor` for reading from it.
    offset: u64,
    ts_ms: u64,
    kind: ?schema.Kind,
    kind_name: []const u8,
    /// Parsed fields; null for a kind this version does not know.
    body: ?schema.Body,
};

/// One page of lines. Owns everything it points to; free with `deinit`.
pub const Page = struct {
    arena: std.heap.ArenaAllocator,
    /// In reading order: oldest first forward, newest first backward.
    entries: []Entry,
    /// Where the next page in the same direction starts; null at the end.
    next: ?Cursor,
    /// Reading stopped at a damaged line: nothing past it can be trusted.
    damaged: bool,

    pub fn deinit(page: *Page) void {
        page.arena.deinit();
    }
};

pub const ReadError = error{ NotFound, OutOfMemory } || storage.IoFault;

/// Reads up to `limit` lines of any session without taking its lock. Only
/// complete, valid lines are returned; written lines never change, so a
/// concurrent writer cannot disturb a reader.
pub fn readPage(
    env: *const Env,
    gpa: std.mem.Allocator,
    id_: []const u8,
    from: From,
    direction: Direction,
    limit: usize,
) ReadError!Page {
    const s = env.s;
    if (!schema.validId(id_)) return error.NotFound;
    const dir = s.openDir(env.root, id_) catch |err| return switch (err) {
        error.NotFound => error.NotFound,
        else => |io_err| storage.ioFault(io_err),
    };
    defer s.closeDir(dir);
    const file = s.openFile(dir, "log.jsonl", .read_only) catch |err| return switch (err) {
        error.NotFound => error.NotFound,
        else => |io_err| storage.ioFault(io_err),
    };
    defer s.closeFile(file);
    const len = s.length(file) catch |io_err| return storage.ioFault(io_err);
    const end = log_mod.lastLineEnd(s, file, len) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => |io_err| storage.ioFault(io_err),
    };
    return readLines(s, gpa, file, end, from, direction, limit);
}

/// A page of the lines before `end` in an open log, shared by the read by
/// id and `Session.readPage`. The caller keeps `file` open throughout.
/// Entries of a page allocated up front, without the safety fill.
const max_raw_page_entries = 1024;

fn readLines(
    s: storage.Storage,
    gpa: std.mem.Allocator,
    file: storage.File,
    end: u64,
    from: From,
    direction: Direction,
    limit: usize,
) (error{OutOfMemory} || storage.IoFault)!Page {
    var page: Page = .{ .arena = .init(gpa), .entries = &.{}, .next = null, .damaged = false };
    errdefer page.arena.deinit();
    const arena = page.arena.allocator();
    // Raw, as `log.ReadBuffer` explains: every entry is written in full
    // before it is read. A longer page grows the list as usual.
    const first_capacity: usize = @min(limit, max_raw_page_entries);
    const first_memory = arena.rawAlloc(first_capacity * @sizeOf(Entry), .of(Entry), @returnAddress()) orelse return error.OutOfMemory;
    const first_entries: [*]Entry = @ptrCast(@alignCast(first_memory));
    var entries: std.ArrayList(Entry) = .initBuffer(first_entries[0..first_capacity]);
    switch (direction) {
        .forward => {
            const at: Cursor = switch (from) {
                .start => .{ .offset = 0, .seq = 1 },
                .end => return page,
                .at => |c| c,
            };
            var reader = log_mod.ForwardReader.init(gpa, s, file, at.offset, end, at.seq);
            defer reader.deinit();
            var last: ?Cursor = null;
            while (entries.items.len < limit) {
                const line = reader.next() catch |err| switch (err) {
                    error.Corrupt => {
                        page.damaged = true;
                        break;
                    },
                    error.OutOfMemory => return error.OutOfMemory,
                    else => |io_err| return storage.ioFault(io_err),
                } orelse break;
                try entries.append(arena, try copyEntry(arena, line));
                last = .{ .offset = line.offset + line.bytes.len, .seq = line.header.seq + 1 };
            }
            if (!page.damaged and entries.items.len == limit) {
                if (last) |c| if (c.offset < end) {
                    page.next = c;
                };
            }
        },
        .backward => {
            var reader = switch (from) {
                .start => return page,
                .end => log_mod.BackwardReader.init(gpa, s, file, end),
                .at => |c| blk: {
                    var r = log_mod.BackwardReader.init(gpa, s, file, c.offset);
                    r.expected_seq = if (c.seq > 1) c.seq - 1 else null;
                    break :blk r;
                },
            };
            defer reader.deinit();
            var oldest: ?Cursor = null;
            while (entries.items.len < limit) {
                const line = reader.next() catch |err| switch (err) {
                    error.Corrupt => {
                        page.damaged = true;
                        break;
                    },
                    error.OutOfMemory => return error.OutOfMemory,
                    else => |io_err| return storage.ioFault(io_err),
                } orelse break;
                try entries.append(arena, try copyEntry(arena, line));
                oldest = .{ .offset = line.offset, .seq = line.header.seq };
            }
            if (!page.damaged and entries.items.len == limit) {
                if (oldest) |c| if (c.offset > 0) {
                    page.next = c;
                };
            }
        },
    }
    page.entries = entries.items;
    return page;
}

fn copyEntry(arena: std.mem.Allocator, line: log_mod.Line) error{OutOfMemory}!Entry {
    // The reader's buffer is reused on the next line: copy before parsing.
    // Raw, as `log.ReadBuffer` explains: the copy writes every byte.
    const bytes = (arena.rawAlloc(line.bytes.len, .@"1", @returnAddress()) orelse return error.OutOfMemory)[0..line.bytes.len];
    @memcpy(bytes, line.bytes);
    // The reader checked the line; only the header's name moves.
    var header = line.header;
    header.kind_name = bytes[@intFromPtr(header.kind_name.ptr) - @intFromPtr(line.bytes.ptr) ..][0..header.kind_name.len];
    const kind = header.kind;
    const body: ?schema.Body = if (kind) |k|
        schema.parseLineBody(arena, k, bytes) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.BadBody => null,
        }
    else
        null;
    return .{ .seq = header.seq, .offset = line.offset, .ts_ms = header.ts_ms, .kind = kind, .kind_name = header.kind_name, .body = body };
}

// ---------------------------------------------------------------------------
// Blobs, fork and delete

pub const BlobError = error{ NotFound, Corrupt, OutOfMemory } || storage.IoFault;

/// Reads a blob of any session without a lock. The bytes are checked
/// against their name, so a damaged blob is reported, never returned.
/// The caller owns the result.
pub fn readBlob(env: *const Env, gpa: std.mem.Allocator, id_: []const u8, hash: []const u8) BlobError![]u8 {
    const s = env.s;
    if (!schema.validId(id_) or !schema.validBlobHash(hash)) return error.NotFound;
    const dir = s.openDir(env.root, id_) catch |err| return notFoundOr(err);
    defer s.closeDir(dir);
    const blobs = s.openDir(dir, "blobs") catch |err| return notFoundOr(err);
    defer s.closeDir(blobs);
    const file = s.openFile(blobs, hash, .read_only) catch |err| return notFoundOr(err);
    defer s.closeFile(file);
    const len = s.length(file) catch |io_err| return storage.ioFault(io_err);
    if (len > max_blob_bytes) return error.Corrupt;
    const bytes = try gpa.alloc(u8, @intCast(len));
    errdefer gpa.free(bytes);
    if ((s.readAt(file, bytes, 0) catch |io_err| return storage.ioFault(io_err)) != bytes.len) return error.Io;
    const actual = schema.blobHash(bytes);
    if (!std.mem.eql(u8, &actual, hash)) return error.Corrupt;
    return bytes;
}

fn notFoundOr(err: storage.Error) (error{NotFound} || storage.IoFault) {
    return if (err == error.NotFound) error.NotFound else error.Io;
}

pub const ForkPoint = union(enum) {
    /// The end of this turn; 0 means before the first turn.
    turn: u64,
    /// The last turn that ended before any damage (`fx session recover`, D15).
    last_good,
};

pub const ForkOptions = struct {
    source: []const u8,
    at: ForkPoint,
    workspace: []const u8,
    host: schema.Host,
};

pub const ForkError = OpenError || error{InvalidForkPoint};

/// A new root session whose lines 2..S are the source's, unchanged, where
/// S ends a turn (D5). The source is read without its lock and never
/// changed; a damaged source forks up to its last good turn. Every source
/// blob is hard-linked (D6), so the fork stays whole if the source goes.
pub fn openFork(env: *const Env, options: ForkOptions) ForkError!*Session {
    const s = env.s;
    const gpa = env.gpa;
    if (!schema.validId(options.source)) return error.NotFound;
    const src_dir = s.openDir(env.root, options.source) catch |err| return notFoundOr(err);
    defer s.closeDir(src_dir);
    const src = s.openFile(src_dir, "log.jsonl", .read_only) catch |err| return switch (err) {
        error.NotFound => error.Corrupt,
        else => |io_err| storage.ioFault(io_err),
    };
    defer s.closeFile(src);
    const len = s.length(src) catch |io_err| return storage.ioFault(io_err);
    const end = log_mod.lastLineEnd(s, src, len) catch return error.Corrupt;
    const point = try findForkPoint(env, src, end, options.at);

    const session = try openNew(env, .{
        .workspace = options.workspace,
        .host = options.host,
        .forked_from = .{ .id = options.source, .seq = point.seq },
    });
    errdefer {
        session.phase = .closed;
        session.destroy();
    }
    // Line 1 is new; lines 2..S are copied (see `copyFolded`).
    session.batch.clearRetainingCapacity();
    session.bounds.clearRetainingCapacity();
    const created = env.nowMs();
    session.frameMarked(1, created, .{ .session_created = session.identity.created() }) catch |err| return forkError(err);
    session.created_ms = created;
    const copied = try gpa.alloc(u8, @intCast(point.end - point.line1_end));
    defer gpa.free(copied);
    if ((s.readAt(src, copied, point.line1_end) catch |io_err| return storage.ioFault(io_err)) != copied.len) return error.Io;
    copyFolded(session, copied, point.seq) catch |err| return forkError(err);
    const blobs = s.openDir(src_dir, "blobs") catch |err| switch (err) {
        error.NotFound => null,
        else => |io_err| return storage.ioFault(io_err),
    };
    defer if (blobs) |b| s.closeDir(b);
    const live = session.stage(blobs) catch |err| return forkError(err);
    session.phase = .{ .live = live };
    session.wrote(created);
    session.written_seq.store(session.env.s.io, point.seq);
    session.setSyncFile(live.log.file, point.seq);
    return session;
}

fn forkError(err: AppendError) ForkError {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        // Copied lines were checked on read; anything else is an I/O failure.
        error.InvalidTransition, error.SessionClosed, error.TooLarge => error.Io,
        else => |io_err| storage.ioFault(io_err),
    };
}

const ForkPointFound = struct {
    seq: u64,
    /// Offset just past line S in the source.
    end: u64,
    /// Offset just past line 1 in the source.
    line1_end: u64,
};

/// Pass 1 of a fork: where S is. Reading stops at the first damage.
fn findForkPoint(env: *const Env, src: storage.File, end: u64, at: ForkPoint) ForkError!ForkPointFound {
    const gpa = env.gpa;
    var reader = log_mod.ForwardReader.init(gpa, env.s, src, 0, end, 1);
    defer reader.deinit();
    const line1 = (reader.next() catch |err| return mapRead(err)) orelse return error.Corrupt;
    if (line1.header.kind != .session_created) return error.Corrupt;
    const line1_end = line1.offset + line1.bytes.len;
    var before_first_turn: ForkPointFound = .{ .seq = 1, .end = line1_end, .line1_end = line1_end };
    var turn_seen = false;
    var last_end: ?ForkPointFound = null;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    while (true) {
        const line = (reader.next() catch |err| switch (err) {
            error.Corrupt => break, // the good part ends here (D15)
            error.OutOfMemory => return error.OutOfMemory,
            else => |io_err| return storage.ioFault(io_err),
        }) orelse break;
        const here: ForkPointFound = .{ .seq = line.header.seq, .end = line.offset + line.bytes.len, .line1_end = line1_end };
        const kind = line.header.kind orelse continue;
        switch (kind) {
            .turn_started => turn_seen = true,
            .turn_committed, .turn_interrupted => {
                _ = arena.reset(.retain_capacity);
                const body = schema.parseBody(arena.allocator(), kind, line.body()) catch return error.Corrupt;
                const turn = switch (body) {
                    .turn_committed => |t| t.turn,
                    .turn_interrupted => |t| t.turn,
                    else => unreachable,
                };
                last_end = here;
                if (at == .turn and at.turn == turn) return here;
            },
            else => {},
        }
        if (!turn_seen) before_first_turn = here;
    }
    return switch (at) {
        .turn => |turn| if (turn == 0) before_first_turn else error.InvalidForkPoint,
        .last_good => last_end orelse error.InvalidForkPoint,
    };
}

/// Pass 2 of a fork: appends the source's lines 2..S to the batch and folds
/// each at its offset in the fork. Lines are copied byte for byte, so their
/// crc holds, except snapshots: a snapshot records byte offsets, and the
/// fork's line 1 has a different length, so each snapshot is encoded again
/// from the fork's own fold at that point, with the same seq and time.
fn copyFolded(session: *Session, copied: []const u8, fork_seq: u64) AppendError!void {
    const gpa = session.env.gpa;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var encoded: std.ArrayList(u8) = .empty;
    defer encoded.deinit(gpa);
    var at: usize = 0;
    while (std.mem.findScalarPos(u8, copied, at, '\n')) |nl| : (at = nl + 1) {
        _ = arena.reset(.retain_capacity);
        const source_line = copied[at .. nl + 1];
        const source_header = log_mod.checkLine(source_line) catch |io_err| return storage.ioFault(io_err);
        const start = session.batch.items.len;
        if (source_header.kind == .snapshot) {
            encoded.clearRetainingCapacity();
            try session.frameMarked(source_header.seq, source_header.ts_ms, try session.snapshotBody(&encoded));
        } else {
            try session.batch.appendSlice(gpa, source_line);
            try session.bounds.append(gpa, session.batch.items.len);
        }
        const line = session.batch.items[start..];
        const header = log_mod.checkLine(line) catch unreachable;
        if (header.kind) |kind| {
            const body = schema.parseBody(arena.allocator(), kind, log_mod.lineBody(line, header)) catch |io_err| return storage.ioFault(io_err);
            try fold.apply(gpa, &session.state, fork_seq, .{ .seq = header.seq, .offset = start, .body = body });
            if (kind == .snapshot) session.snapshot_base = session.batch.items.len;
        } else {
            session.state.last_seq = header.seq;
        }
    }
    // The fork owns no children spawned before its fork point.
    std.debug.assert(session.state.children.items.len == 0);
}

pub const DeleteError = error{ NotFound, Busy, OutOfMemory } || storage.IoFault;

/// The child ids a session owns, from its `child_spawned` lines. The
/// caller frees each id and the list.
pub fn childrenOf(env: *const Env, id_: []const u8) DeleteError![][]u8 {
    const s = env.s;
    const gpa = env.gpa;
    if (!schema.validId(id_)) return error.NotFound;
    const dir = s.openDir(env.root, id_) catch |err| return notFoundOr(err);
    defer s.closeDir(dir);
    var children: std.ArrayList([]u8) = .empty;
    errdefer {
        for (children.items) |c| gpa.free(c);
        children.deinit(gpa);
    }
    try collectChildren(env, dir, &children);
    return children.toOwnedSlice(gpa);
}

/// Step 1 of a delete (`tla/Catalog.tla` `Trash`): Busy if the session is
/// open anywhere, else it is renamed to `.trash/{id}`, which makes it gone
/// for everyone at once. The caller then appends the index tombstone and
/// calls `purgeTrashed`; owned children are deleted first (`childrenOf`),
/// so a Busy child never leaves an orphan.
pub fn trashSession(env: *const Env, id_: []const u8) DeleteError!void {
    const s = env.s;
    if (!schema.validId(id_)) return error.NotFound;
    const dir = s.openDir(env.root, id_) catch |err| return notFoundOr(err);
    var dir_open = true;
    defer if (dir_open) s.closeDir(dir);
    const lock = s.openFile(dir, "lock", .read_write) catch |err| switch (err) {
        error.NotFound => null,
        else => |io_err| return storage.ioFault(io_err),
    };
    defer if (lock) |l| s.closeFile(l);
    if (lock) |l| if (!(s.tryLock(l) catch |io_err| return storage.ioFault(io_err))) return error.Busy;
    const trash = s.ensureDir(env.root, ".trash") catch |io_err| return storage.ioFault(io_err);
    defer s.closeDir(trash);
    s.deleteTree(trash, id_) catch |io_err| return storage.ioFault(io_err);
    s.closeDir(dir);
    dir_open = false;
    s.rename(env.root, id_, trash, id_) catch |err| return notFoundOr(err);
    s.syncDir(env.root) catch |io_err| return storage.ioFault(io_err);
    env.observeCatalog(id_, .trashed);
}

/// Step 3 of a delete (`Purge`): removes `.trash/{id}` and everything in it.
pub fn purgeTrashed(env: *const Env, id_: []const u8) DeleteError!void {
    const s = env.s;
    const trash = s.openDir(env.root, ".trash") catch |err| return notFoundOr(err);
    defer s.closeDir(trash);
    s.deleteTree(trash, id_) catch |io_err| return storage.ioFault(io_err);
    env.observeCatalog(id_, .purged);
}

/// Whether a session with this id is open for writing anywhere.
pub fn isBusy(env: *const Env, id_: []const u8) DeleteError!bool {
    const s = env.s;
    if (!schema.validId(id_)) return error.NotFound;
    const dir = s.openDir(env.root, id_) catch |err| return notFoundOr(err);
    defer s.closeDir(dir);
    const lock = s.openFile(dir, "lock", .read_write) catch |err| return switch (err) {
        error.NotFound => false,
        else => |io_err| storage.ioFault(io_err),
    };
    defer s.closeFile(lock);
    if (!(s.tryLock(lock) catch |io_err| return storage.ioFault(io_err))) return true;
    s.unlock(lock);
    return false;
}

/// The distinct child ids named by `child_spawned` lines.
fn collectChildren(env: *const Env, dir: storage.Dir, out: *std.ArrayList([]u8)) DeleteError!void {
    const s = env.s;
    const gpa = env.gpa;
    const file = s.openFile(dir, "log.jsonl", .read_only) catch |err| return switch (err) {
        error.NotFound => {},
        else => |io_err| storage.ioFault(io_err),
    };
    defer s.closeFile(file);
    const len = s.length(file) catch |io_err| return storage.ioFault(io_err);
    var reader = log_mod.ForwardReader.init(gpa, s, file, 0, len, 1);
    defer reader.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    while (true) {
        const line = (reader.next() catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => break, // a damaged log still deletes; its later children are not found
        }) orelse break;
        if (line.header.kind != .child_spawned) continue;
        _ = arena.reset(.retain_capacity);
        const body = schema.parseBody(arena.allocator(), .child_spawned, line.body()) catch continue;
        const child = body.child_spawned.child;
        for (out.items) |known| {
            if (std.mem.eql(u8, known, child)) break;
        } else try out.append(gpa, try gpa.dupe(u8, child));
    }
}

// ---------------------------------------------------------------------------
// Doctor

pub const Verified = struct {
    lines: u64,
    /// Offset of the first damaged line, if any; lines past it were not checked.
    damaged_at: ?u64,
    /// Snapshots whose state differs from the fold at their position.
    bad_snapshots: u64,
};

/// Reads a whole log without its lock: every line's frame, checksum and
/// seq, and every snapshot against the fold at its position (a snapshot is
/// a cache that must equal what it replaces).
pub fn verifySession(env: *const Env, id_: []const u8) OpenError!Verified {
    const s = env.s;
    const gpa = env.gpa;
    if (!schema.validId(id_)) return error.NotFound;
    const dir = s.openDir(env.root, id_) catch |err| return notFoundOr(err);
    defer s.closeDir(dir);
    const file = s.openFile(dir, "log.jsonl", .read_only) catch |err| return notFoundOr(err);
    defer s.closeFile(file);
    const len = s.length(file) catch |io_err| return storage.ioFault(io_err);
    const end = log_mod.lastLineEnd(s, file, len) catch return error.Corrupt;
    var reader = log_mod.ForwardReader.init(gpa, s, file, 0, end, 1);
    defer reader.deinit();
    var state: fold.State = .{};
    defer state.deinit(gpa);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var result: Verified = .{ .lines = 0, .damaged_at = null, .bad_snapshots = 0 };
    var fork_seq: u64 = 0;
    while (true) {
        const line = (reader.next() catch |err| switch (err) {
            error.Corrupt => {
                result.damaged_at = reader.damaged_at;
                break;
            },
            error.OutOfMemory => return error.OutOfMemory,
            else => |io_err| return storage.ioFault(io_err),
        }) orelse break;
        result.lines += 1;
        _ = arena.reset(.retain_capacity);
        const kind = line.header.kind orelse continue;
        const body = schema.parseBody(arena.allocator(), kind, line.body()) catch {
            result.damaged_at = line.offset;
            break;
        };
        switch (body) {
            .session_created => |c| fork_seq = if (c.forked_from) |o| o.seq else 0,
            .snapshot => |snap| {
                var decoded = fold.decodeState(gpa, arena.allocator(), snap.state) catch {
                    result.bad_snapshots += 1;
                    continue;
                };
                defer decoded.deinit(gpa);
                decoded.last_seq = state.last_seq;
                decoded.clean_exit = state.clean_exit;
                // A snapshot copied from a fork's source carries its children.
                if (line.header.seq <= fork_seq) decoded.clearChildren(gpa);
                if (!decoded.eql(&state)) result.bad_snapshots += 1;
            },
            else => {},
        }
        try fold.apply(gpa, &state, fork_seq, .{ .seq = line.header.seq, .offset = line.offset, .body = body });
    }
    return result;
}

/// What the catalog records about one session, read without its lock.
pub const Summary = struct {
    identity: Identity,
    state: fold.State,
    created_ms: u64,
    updated_ms: u64,

    pub fn deinit(summary: *Summary, gpa: std.mem.Allocator) void {
        summary.identity.deinit(gpa);
        summary.state.deinit(gpa);
    }
};

/// Line 1, the newest snapshot and the tail of a session, read-only.
pub fn readSummary(env: *const Env, id_: []const u8) OpenError!Summary {
    const s = env.s;
    const gpa = env.gpa;
    if (!schema.validId(id_)) return error.NotFound;
    const dir = s.openDir(env.root, id_) catch |err| return notFoundOr(err);
    defer s.closeDir(dir);
    const file = s.openFile(dir, "log.jsonl", .read_only) catch |err| return notFoundOr(err);
    defer s.closeFile(file);
    const len = s.length(file) catch |io_err| return storage.ioFault(io_err);
    const end = log_mod.lastLineEnd(s, file, len) catch return error.Corrupt;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    var damaged_at: u64 = 0;
    const loaded = try loadFolded(env, &arena_state, file, end, &damaged_at);
    return .{ .identity = loaded.identity, .state = loaded.state, .created_ms = loaded.created_ms, .updated_ms = loaded.updated_ms };
}

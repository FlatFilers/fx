//! The session manager: one append-only log per session, one owner for
//! session storage and the live-session lifecycle.
//!
//! This file (L4) is the only public root; fx imports it and nothing else.
//! A process creates one `Manager` with `init`, which does no disk I/O.
//! Thirteen operations cover every host (docs/session-manager/system-design.md
//! "API"), plus `verify` and `rebuild` for doctor:
//!
//!   openNew  openResume  openFork  openImport        -> Session
//!   Session.append  Session.putBlob  Session.state  Session.read  Session.close
//!   read  getBlob  list  delete                       (by id, no lock)
//!
//! Every input is checked here; the levels below trust it. A Session is
//! safe to use from any thread, and each call on it is atomic relative to
//! the others. Index (catalog) failures never fail a session change: the
//! session data is already durable, and the failure is reported through
//! the diagnostics callback (`index_stale`).

const std = @import("std");
const schema = @import("schema.zig");
const storage = @import("storage.zig");
const fold = @import("fold.zig");
const log_mod = @import("log.zig");
const session_mod = @import("session.zig");
const catalog_mod = @import("catalog.zig");
const diag = @import("diag.zig");

pub const Role = schema.Role;
pub const Host = schema.Host;
pub const Reason = schema.Reason;
pub const Outcome = schema.Outcome;
pub const SetKey = schema.SetKey;
pub const Kind = schema.Kind;
pub const Body = schema.Body;
pub const Event = fold.Event;
pub const Piece = fold.Piece;
pub const State = fold.State;
pub const Child = fold.Child;
pub const Diagnostic = diag.Event;
pub const DiagnosticKind = diag.Kind;
pub const Diagnostics = diag.Sink;
pub const Page = session_mod.Page;
pub const Entry = session_mod.Entry;
pub const Cursor = session_mod.Cursor;
pub const From = session_mod.From;
pub const Direction = session_mod.Direction;
pub const ForkPoint = session_mod.ForkPoint;
pub const Verified = session_mod.Verified;
pub const Summary = catalog_mod.Summary;
pub const ListPage = catalog_mod.Page;
pub const ListCursor = catalog_mod.Cursor;
pub const Filter = catalog_mod.Filter;
pub const Rebuilt = catalog_mod.Rebuilt;
pub const BlobHash = [schema.blob_hash_len]u8;
pub const max_blob_bytes = session_mod.max_blob_bytes;
/// Largest `data` or `value` accepted in one event; the adapter moves
/// bodies above its own, smaller threshold into blobs.
pub const max_value_bytes: usize = log_mod.max_line_bytes - 64 * 1024;
const max_text_bytes = 4096;

/// An I/O failure with its OS cause when known, `Io` otherwise (D29).
pub const IoFault = storage.IoFault;
pub const OpenError = error{ InvalidArgument, NotFound, Busy, ChildSession, Corrupt, UnsupportedVersion, InvalidForkPoint, Exists, OutOfMemory } || storage.IoFault;
pub const AppendError = error{ InvalidArgument, InvalidTransition, SessionClosed, TooLarge, OutOfMemory } || storage.IoFault;
pub const CloseError = error{OutOfMemory} || storage.IoFault;
pub const ReadError = error{ InvalidArgument, NotFound, OutOfMemory } || storage.IoFault;
pub const SessionReadError = error{ InvalidArgument, SessionClosed, OutOfMemory } || storage.IoFault;
pub const BlobError = error{ InvalidArgument, NotFound, Corrupt, OutOfMemory } || storage.IoFault;
pub const ListError = error{ Busy, OutOfMemory } || storage.IoFault;
pub const DeleteError = error{ InvalidArgument, NotFound, Busy, OutOfMemory } || storage.IoFault;
pub const VerifyError = error{ InvalidArgument, NotFound, OutOfMemory } || storage.IoFault;
pub const RebuildError = error{ Busy, OutOfMemory } || storage.IoFault;

pub const Backend = enum { posix };

pub const InitOptions = struct {
    /// The sessions root, `~/.fx/sessions/v2` in fx (D19); created on first
    /// use, with any missing parent folders, all `0700`.
    root: []const u8,
    backend: Backend = .posix,
    /// Called once for every repair or drop (D14).
    diagnostics: ?Diagnostics = null,
    /// How long an open waits for a session or index lock before Busy.
    lock_wait_ms: u64 = 2000,
    /// Snapshot distance (D7).
    snapshot_every_bytes: u64 = session_mod.default_snapshot_every_bytes,
    /// Index size that triggers a rewrite with one line per id.
    index_compact_bytes: u64 = 1 << 20,
};

pub const NewOptions = struct {
    workspace: []const u8,
    host: Host,
    role: Role = .root,
    /// Required for a child session.
    parent: ?[]const u8 = null,
    /// A child's id, already named in its parent's `child_spawned` (D34); a
    /// root always gets a fresh one. A taken id fails the first turn's
    /// append and leaves the existing session as it was.
    id: ?[]const u8 = null,
};

pub const ResumeTarget = union(enum) {
    id: []const u8,
    /// The newest updated root session in the workspace (`--resume last`).
    last,
    /// The root session this host opened last in the workspace (`-c` is
    /// `.last_opened = .app`).
    last_opened: Host,
};

pub const ResumeOptions = struct {
    target: ResumeTarget,
    workspace: []const u8,
    host: Host,
    parent: ?[]const u8 = null,
    /// How long this open waits for the session's writer lock before Busy;
    /// null uses the manager's `lock_wait_ms`. A picker passes 0 so a
    /// session open elsewhere shows as busy at once (D38).
    lock_wait_ms: ?u64 = null,
};

pub const ForkOptions = struct {
    source: []const u8,
    at: ForkPoint,
    workspace: []const u8,
    host: Host,
};

pub const ImportOptions = struct {
    /// The v1 id, kept (D8).
    id: []const u8,
    workspace: []const u8,
    host: Host,
    role: Role = .root,
    parent: ?[]const u8 = null,
    created_ms: u64,
};

pub const Manager = struct {
    gpa: std.mem.Allocator,
    root_path: []u8,
    env: session_mod.Env,
    root_mutex: std.Io.Mutex = .init,
    root_open: bool = false,
    index_lock: catalog_mod.IndexLock = .{},

    /// No disk I/O happens here; the root opens on first use. `gpa` must be
    /// thread-safe when Sessions are used from several threads. The Manager
    /// must outlive every Session.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, options: InitOptions) error{ InvalidArgument, OutOfMemory }!*Manager {
        if (options.root.len == 0 or !std.unicode.utf8ValidateSlice(options.root)) return error.InvalidArgument;
        const m = try gpa.create(Manager);
        errdefer gpa.destroy(m);
        m.* = .{
            .gpa = gpa,
            .root_path = try gpa.dupe(u8, options.root),
            .env = .{
                .gpa = gpa,
                .s = .{ .io = io },
                .root = undefined,
                .options = .{
                    .lock_wait_ms = options.lock_wait_ms,
                    .snapshot_every_bytes = options.snapshot_every_bytes,
                    .index_compact_bytes = options.index_compact_bytes,
                },
                .diagnostics = options.diagnostics,
            },
        };
        return m;
    }

    /// Every Session must be closed and released first.
    pub fn deinit(m: *Manager) void {
        m.index_lock.close(m.env.s);
        if (m.root_open) m.env.s.closeDir(m.env.root);
        m.gpa.free(m.root_path);
        m.gpa.destroy(m);
    }

    /// Test and driver builds only: report internal steps to a tracer, and
    /// optionally plant a known bug the trace check must reject.
    pub fn setTracing(
        m: *Manager,
        observer: if (storage.hooks) session_mod.Observer else void,
        planted: if (storage.hooks) hooks.trace.Planted else void,
    ) void {
        if (storage.hooks) {
            m.env.observer = observer;
            m.env.planted = planted;
        }
    }

    /// Test and driver builds only: route storage through a fault injector.
    pub fn setFault(m: *Manager, fault: if (storage.hooks) *storage.Fault else void) void {
        if (storage.hooks) m.env.s.fault = fault;
    }

    fn ready(m: *Manager) storage.IoFault!void {
        const io = m.env.s.io;
        m.root_mutex.lockUncancelable(io);
        defer m.root_mutex.unlock(io);
        if (m.root_open) return;
        m.env.root = m.env.s.openRoot(m.root_path) catch |io_err| return storage.ioFault(io_err);
        m.root_open = true;
    }

    /// `ready` for calls that only find or read: a missing root is not
    /// created, so reading on a machine that never saved a session writes
    /// nothing. False means there is nothing to find.
    fn readyToRead(m: *Manager) storage.IoFault!bool {
        {
            const io = m.env.s.io;
            m.root_mutex.lockUncancelable(io);
            defer m.root_mutex.unlock(io);
            if (m.root_open) return true;
        }
        const present = m.env.s.rootExists(m.root_path) catch |io_err| return storage.ioFault(io_err);
        if (!present) return false;
        try ready(m);
        return true;
    }

    fn catalog(m: *Manager) catalog_mod.Catalog {
        return .{ .env = &m.env, .lock = &m.index_lock };
    }

    fn nowMs(m: *Manager) u64 {
        return std.math.cast(u64, std.Io.Timestamp.now(m.env.s.io, .real).toMilliseconds()) orelse 0;
    }

    /// Records an index failure that did not fail the session change.
    fn indexStale(m: *Manager, id: []const u8) void {
        diag.report(m.env.diagnostics, .{ .kind = .index_stale, .session_id = id });
    }

    // -- open -----------------------------------------------------------------

    /// A new session in memory. Nothing touches the disk until its first
    /// turn (D2).
    pub fn openNew(m: *Manager, options: NewOptions) OpenError!Session {
        try checkText(options.workspace);
        try checkRoleParent(options.role, options.parent);
        if (options.id) |id| {
            if (options.role != .child) return error.InvalidArgument;
            try checkId(id);
        }
        const inner = try session_mod.openNew(&m.env, .{
            .workspace = options.workspace,
            .host = options.host,
            .role = options.role,
            .parent = options.parent,
            .id = options.id,
        });
        return .{ .manager = m, .inner = inner };
    }

    /// Opens a saved session for writing: Busy if another process has it.
    pub fn openResume(m: *Manager, options: ResumeOptions) OpenError!Session {
        try checkText(options.workspace);
        if (options.parent) |p| try checkId(p);
        // Arguments are checked before the disk is asked anything.
        switch (options.target) {
            .id => |id| try checkId(id),
            .last, .last_opened => {},
        }
        if (!try readyToRead(m)) return error.NotFound;
        // An id resolved from the index is owned here.
        var resolved: ?[]u8 = null;
        defer if (resolved) |r| m.gpa.free(r);
        const id: []const u8 = switch (options.target) {
            .id => |id| id,
            .last, .last_opened => blk: {
                const target: catalog_mod.Target = switch (options.target) {
                    .last => .last,
                    .last_opened => |host| .{ .last_opened = host },
                    .id => unreachable,
                };
                resolved = (try m.resolve(options.workspace, target)) orelse return error.NotFound;
                break :blk resolved.?;
            },
        };
        const inner = session_mod.openResume(&m.env, .{
            .id = id,
            .workspace = options.workspace,
            .host = options.host,
            .parent = options.parent,
            .lock_wait_ms = options.lock_wait_ms,
        }) catch |err| return switch (err) {
            error.NotFound, error.Busy, error.ChildSession, error.Corrupt, error.UnsupportedVersion, error.OutOfMemory => |e| e,
            else => |io_err| storage.ioFault(io_err),
        };
        m.catalog().opened(inner.id(), options.host, m.nowMs()) catch m.indexStale(inner.id());
        return .{ .manager = m, .inner = inner };
    }

    fn resolve(m: *Manager, workspace: []const u8, target: catalog_mod.Target) OpenError!?[]u8 {
        return m.catalog().resolve(m.gpa, workspace, target) catch |err| switch (err) {
            error.Busy => error.Busy,
            error.OutOfMemory => error.OutOfMemory,
            else => |io_err| storage.ioFault(io_err),
        };
    }

    /// A new root session that starts as a copy of `source` up to a turn
    /// boundary (D5). The source is never changed, and may be damaged
    /// (`.last_good`, D15).
    pub fn openFork(m: *Manager, options: ForkOptions) OpenError!Session {
        try checkId(options.source);
        try checkText(options.workspace);
        try ready(m);
        const inner = session_mod.openFork(&m.env, .{
            .source = options.source,
            .at = options.at,
            .workspace = options.workspace,
            .host = options.host,
        }) catch |err| return switch (err) {
            error.NotFound, error.Busy, error.ChildSession, error.Corrupt, error.UnsupportedVersion, error.InvalidForkPoint, error.OutOfMemory => |e| e,
            else => |io_err| storage.ioFault(io_err),
        };
        const session: Session = .{ .manager = m, .inner = inner };
        session.updateIndexAs(.published);
        return session;
    }

    /// The v1 conversion only (D8): a new session under an existing id,
    /// whose batches keep their original times (`Session.appendAt`). Exists
    /// if the id is taken now or was ever deleted.
    pub fn openImport(m: *Manager, options: ImportOptions) OpenError!Session {
        try checkId(options.id);
        try checkText(options.workspace);
        try checkRoleParent(options.role, options.parent);
        try ready(m);
        if (m.env.s.stat(m.env.root, options.id)) |_| return error.Exists else |err| switch (err) {
            error.NotFound => {},
            else => |io_err| return storage.ioFault(io_err),
        }
        const deleted = m.catalog().isDeleted(m.gpa, options.id) catch |err| return switch (err) {
            error.Busy => error.Busy,
            error.OutOfMemory => error.OutOfMemory,
            else => |io_err| storage.ioFault(io_err),
        };
        if (deleted) return error.Exists;
        const inner = try session_mod.openNew(&m.env, .{
            .workspace = options.workspace,
            .host = options.host,
            .role = options.role,
            .parent = options.parent,
            .id = options.id,
            .created_ms = options.created_ms,
        });
        return .{ .manager = m, .inner = inner, .import = true };
    }

    // -- by id, without a lock ------------------------------------------------

    /// A page of lines: forward from `.start` or a cursor, or backward from
    /// `.end` or a cursor. The caller frees the page.
    pub fn read(m: *Manager, gpa: std.mem.Allocator, id: []const u8, from: From, direction: Direction, limit: usize) ReadError!Page {
        try checkId(id);
        if (limit == 0) return error.InvalidArgument;
        if (!try readyToRead(m)) return error.NotFound;
        return session_mod.readPage(&m.env, gpa, id, from, direction, limit);
    }

    /// A blob's bytes, checked against its hash. The caller frees them.
    pub fn getBlob(m: *Manager, gpa: std.mem.Allocator, id: []const u8, hash: []const u8) BlobError![]u8 {
        try checkId(id);
        if (!schema.validBlobHash(hash)) return error.InvalidArgument;
        if (!try readyToRead(m)) return error.NotFound;
        return session_mod.readBlob(&m.env, gpa, id, hash);
    }

    /// Root sessions, newest first, from the index alone.
    pub fn list(m: *Manager, gpa: std.mem.Allocator, filter: Filter, cursor: ?ListCursor, limit: usize) ListError!ListPage {
        if (!try readyToRead(m)) return .{ .arena = .init(gpa), .items = &.{}, .next = null };
        return m.catalog().list(gpa, filter, cursor, @max(limit, 1));
    }

    /// Deletes a session and the children it owns. Busy if any is open.
    /// Order per session (`tla/Catalog.tla`): children first, then the
    /// rename to `.trash`, the index tombstone, and the purge. fx removes
    /// the matching terminal folders (D13).
    pub fn delete(m: *Manager, id: []const u8) DeleteError!void {
        try checkId(id);
        if (!try readyToRead(m)) return error.NotFound;
        return m.deleteDepth(id, 0);
    }

    fn deleteDepth(m: *Manager, id: []const u8, depth: usize) DeleteError!void {
        const children = try session_mod.childrenOf(&m.env, id);
        defer {
            for (children) |c| m.gpa.free(c);
            m.gpa.free(children);
        }
        if (depth < 8) for (children) |child| {
            m.deleteDepth(child, depth + 1) catch |err| switch (err) {
                error.NotFound => {},
                else => return err,
            };
        };
        try session_mod.trashSession(&m.env, id);
        m.catalog().del(id, m.nowMs()) catch m.indexStale(id);
        try session_mod.purgeTrashed(&m.env, id);
    }

    /// Doctor: checks every line and snapshot of one session.
    pub fn verify(m: *Manager, id: []const u8) VerifyError!Verified {
        try checkId(id);
        if (!try readyToRead(m)) return error.NotFound;
        return session_mod.verifySession(&m.env, id) catch |err| switch (err) {
            error.NotFound, error.OutOfMemory => |e| e,
            error.Busy, error.ChildSession, error.Corrupt, error.UnsupportedVersion => error.Io,
            else => |io_err| storage.ioFault(io_err),
        };
    }

    /// Doctor: re-derives the index and sweeps leftovers.
    pub fn rebuild(m: *Manager) RebuildError!Rebuilt {
        try ready(m);
        return m.catalog().rebuild(m.gpa);
    }
};

pub const Session = struct {
    manager: *Manager,
    inner: *session_mod.Session,
    import: bool = false,

    pub fn id(s: Session) []const u8 {
        return s.inner.id();
    }

    /// Appends a batch and returns the seq of its last line. Returns after
    /// the fsync when the batch holds a durable-class event (D3).
    pub fn append(s: Session, events: []const Event) AppendError!u64 {
        if (s.import) return error.InvalidArgument;
        return s.appendChecked(events, null);
    }

    /// Import only: appends a batch stamped with its original time.
    pub fn appendAt(s: Session, events: []const Event, ts_ms: u64) AppendError!u64 {
        if (!s.import) return error.InvalidArgument;
        return s.appendChecked(events, ts_ms);
    }

    fn appendChecked(s: Session, events: []const Event, ts_ms: ?u64) AppendError!u64 {
        try checkEvents(s.manager.gpa, events, s.import);
        // The root opens on the first publish at the latest, and an append
        // that stays held in memory does no disk I/O at all (D2).
        if (!s.inner.staysHeld(events)) try s.manager.ready();
        const result = try s.inner.appendReport(events, ts_ms);
        if (result.published) s.updateIndexAs(.published) else if (result.listing_changed) s.updateIndex();
        return result.last_seq;
    }

    /// Stores a large body and returns its hash, once it is durable (D6).
    pub fn putBlob(s: Session, bytes: []const u8) AppendError!BlobHash {
        return s.inner.putBlob(bytes);
    }

    /// A page of this session's lines through its own open log: the current
    /// session's scrollback, with no open per page. `Manager.read` is for a
    /// session that is not open here. The caller frees the page.
    pub fn read(s: Session, gpa: std.mem.Allocator, from: From, direction: Direction, limit: usize) SessionReadError!Page {
        if (limit == 0) return error.InvalidArgument;
        return s.inner.readPage(gpa, from, direction, limit);
    }

    /// An owned copy of the folded state; free with `State.deinit`.
    pub fn state(s: Session, gpa: std.mem.Allocator) error{OutOfMemory}!State {
        return s.inner.stateCopy(gpa);
    }

    /// Interrupts an open turn, appends `closed`, syncs, updates the index
    /// and releases the lock. Later calls get SessionClosed.
    pub fn close(s: Session) CloseError!void {
        const was_live = s.inner.closeReport() catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => |io_err| storage.ioFault(io_err),
        };
        if (was_live) s.updateIndex();
    }

    /// Frees the handle, closing it first if needed. No other thread may
    /// still use it.
    pub fn release(s: Session) void {
        s.close() catch {};
        s.inner.destroy();
    }

    fn updateIndex(s: Session) void {
        s.updateIndexAs(.changed);
    }

    /// A new session counts as opened by the host that created it, so `-c`
    /// finds it; later updates leave the open times alone.
    fn updateIndexAs(s: Session, why: enum { published, changed }) void {
        const m = s.manager;
        var arena = std.heap.ArenaAllocator.init(m.gpa);
        defer arena.deinit();
        var summary = s.indexSummary(arena.allocator()) catch return m.indexStale(s.id());
        if (why == .published) summary.opened_ms[@intFromEnum(summary.host)] = summary.updated_ms;
        m.catalog().put(summary) catch m.indexStale(s.id());
    }

    fn indexSummary(s: Session, arena: std.mem.Allocator) error{OutOfMemory}!Summary {
        const st = try s.inner.stateCopy(arena);
        const identity = &s.inner.identity;
        const workspace = try session_mod.currentWorkspace(s.inner, arena);
        return .{
            .id = identity.id,
            .role = identity.role,
            .host = identity.host,
            .workspace = workspace,
            .title = st.title,
            .language = st.language,
            .parent = identity.parent,
            // The newest line's time, as a rebuild computes it (D20).
            .created_ms = if (st.created_ms != 0) st.created_ms else s.manager.nowMs(),
            .updated_ms = if (st.updated_ms != 0) st.updated_ms else s.manager.nowMs(),
            .turns = st.committed + st.interrupted,
        };
    }
};

/// Test and driver builds only (`-Dhooks=true`): fault injection, the trace
/// writer and the internals a tracer reads. Empty in the build fx uses.
pub const hooks = if (storage.hooks) struct {
    pub const Fault = storage.Fault;
    pub const flipBit = @import("storage_fault.zig").flipBit;
    pub const Dir = storage.Dir;
    pub const Recorder = diag.Recorder;
    pub const trace = @import("trace.zig");
    pub const session = session_mod;
    pub const log = log_mod;
    pub const schema = @import("schema.zig");
} else struct {};

// ---------------------------------------------------------------------------
// Input checks (pure)

fn checkText(text: []const u8) error{InvalidArgument}!void {
    if (text.len == 0 or text.len > max_text_bytes) return error.InvalidArgument;
    if (!std.unicode.utf8ValidateSlice(text)) return error.InvalidArgument;
}

fn checkId(id: []const u8) error{InvalidArgument}!void {
    if (!schema.validId(id)) return error.InvalidArgument;
}

fn checkRoleParent(role: Role, parent: ?[]const u8) error{InvalidArgument}!void {
    switch (role) {
        .root => if (parent != null) return error.InvalidArgument,
        .child => try checkId(parent orelse return error.InvalidArgument),
    }
}

/// Valid JSON with no surrounding whitespace: a reader gets exactly these
/// bytes back, and a resumed state equals the live one.
fn checkJson(gpa: std.mem.Allocator, raw: []const u8) AppendError!void {
    if (raw.len > max_value_bytes) return error.TooLarge;
    if (raw.len == 0 or isJsonSpace(raw[0]) or isJsonSpace(raw[raw.len - 1])) return error.InvalidArgument;
    if (!(std.json.validate(gpa, raw) catch return error.OutOfMemory)) return error.InvalidArgument;
}

fn isJsonSpace(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\r' or c == '\n';
}

fn checkJsonString(gpa: std.mem.Allocator, raw: []const u8) AppendError!void {
    try checkJson(gpa, raw);
    if (raw.len < 2 or raw[0] != '"') return error.InvalidArgument;
}

/// A JSON string of 1 to 24 bytes, as v1 stores a conversation language
/// (D18); any bytes, so every v1 value converts.
fn checkLanguage(gpa: std.mem.Allocator, raw: []const u8) AppendError!void {
    try checkJsonString(gpa, raw);
    const parsed = std.json.parseFromSlice([]const u8, gpa, raw, .{}) catch return error.InvalidArgument;
    defer parsed.deinit();
    if (parsed.value.len == 0 or parsed.value.len > max_language_bytes) return error.InvalidArgument;
}

const max_language_bytes = 24;

/// What a host may send: valid JSON for fx's content, and only the kinds
/// and reasons that belong to hosts. The manager writes the rest itself.
fn checkEvents(gpa: std.mem.Allocator, events: []const Event, import: bool) AppendError!void {
    if (events.len == 0) return error.InvalidArgument;
    for (events) |event| switch (event) {
        .turn_started, .turn_committed => {},
        .item => |piece| {
            if (!schema.validItemType(piece.type)) return error.InvalidArgument;
            try checkJson(gpa, piece.data);
            for (piece.blobs) |hash| if (!schema.validBlobHash(hash)) return error.InvalidArgument;
        },
        .compacted => |data| try checkJson(gpa, data),
        // Crash and close interruptions are the manager's; an import
        // replays whatever v1 recorded.
        .turn_interrupted => |reason| switch (reason) {
            .cancel, .failed => {},
            .closed, .crash => if (!import) return error.InvalidArgument,
        },
        .set => |s| switch (s.key) {
            .title, .workspace => try checkJsonString(gpa, s.value),
            .language => try checkLanguage(gpa, s.value),
            .prefs, .permissions, .usage => try checkJson(gpa, s.value),
        },
        .child_spawned => |c| {
            try checkId(c.child);
            try checkWorkId(c.work_id);
            if (c.data) |data| try checkChildData(gpa, data);
        },
        .child_finished => |c| {
            try checkId(c.child);
            try checkWorkId(c.work_id);
            if (c.outcome == .lost and !import) return error.InvalidArgument;
            if (c.data) |data| try checkChildData(gpa, data);
        },
    };
}

/// Folded into every snapshot for each child, so kept small (D22).
fn checkChildData(gpa: std.mem.Allocator, raw: []const u8) AppendError!void {
    if (raw.len > max_child_data_bytes) return error.TooLarge;
    try checkJson(gpa, raw);
}

pub const max_child_data_bytes = 4096;

fn checkWorkId(work_id: []const u8) error{InvalidArgument}!void {
    if (work_id.len == 0 or work_id.len > 255) return error.InvalidArgument;
    if (!std.unicode.utf8ValidateSlice(work_id)) return error.InvalidArgument;
}

test {
    _ = @import("boundary_test.zig");
    _ = @import("storage.zig");
    _ = @import("schema.zig");
    _ = @import("log.zig");
    _ = @import("fold.zig");
    _ = @import("session.zig");
    _ = @import("session_test.zig");
    _ = @import("catalog.zig");
    _ = @import("api_test.zig");
    // Fault injection and traces exist only with -Dhooks=true.
    if (storage.hooks) {
        _ = @import("storage_fault.zig");
        _ = @import("trace.zig");
        _ = @import("log_model_test.zig");
        _ = @import("session_model_test.zig");
        _ = @import("catalog_model_test.zig");
    }
}

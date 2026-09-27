//! L3: the catalog, `index.jsonl`.
//!
//! The index is derived and rebuildable. It holds one checksummed line per
//! change: `put` (a session's summary), `opened` (a host opened it) and
//! `del` (a tombstone), appended under `index.lock` and never fsynced.
//! Folding it keeps the newest record per id; a tombstone is final, so a
//! deleted id never comes back (`tla/Catalog.tla` `NoResurrection`).
//! `list` reads only the index and self-heals entries whose folder is gone;
//! `rebuild` re-derives it from the folders and sweeps `.tmp` and `.trash`.
//!
//! Pure core: `encodeRecord`, `decodeRecord` and `Index.fold`.

const std = @import("std");
const schema = @import("schema.zig");
const storage = @import("storage.zig");
const log_mod = @import("log.zig");
const session_mod = @import("session.zig");
const diag = @import("diag.zig");

const index_name = "index.jsonl";
const index_tmp_name = "index.jsonl.tmp";
const lock_name = "index.lock";

/// A `.tmp` entry younger than this may be a publish in progress.
const sweep_after_ms: i64 = 60 * 60 * 1000;

pub const host_count = @typeInfo(schema.Host).@"enum".fields.len;

pub const Summary = struct {
    id: []const u8,
    role: schema.Role,
    /// The host that created the session.
    host: schema.Host,
    /// The workspace in effect.
    workspace: []const u8,
    /// Raw JSON string, as set.
    title: ?[]const u8 = null,
    /// Raw JSON string, as set (D18).
    language: ?[]const u8 = null,
    parent: ?[]const u8 = null,
    created_ms: u64,
    updated_ms: u64,
    /// Turns that ended, committed or interrupted.
    turns: u64,
    /// When each host last opened the session; 0 means never.
    opened_ms: [host_count]u64 = @splat(0),
};

pub const Record = union(enum) {
    put: Summary,
    opened: struct { id: []const u8, host: schema.Host, ts_ms: u64 },
    del: struct { id: []const u8, ts_ms: u64 },
};

// ---------------------------------------------------------------------------
// Pure core

/// Appends one framed index line.
pub fn encodeRecord(gpa: std.mem.Allocator, out: *std.ArrayList(u8), record: Record) log_mod.FrameError!void {
    var payload: std.ArrayList(u8) = .empty;
    defer payload.deinit(gpa);
    switch (record) {
        .put => |p| {
            try payload.appendSlice(gpa, "{\"op\":\"put\",\"id\":");
            try schema.appendJsonString(gpa, &payload, p.id);
            try payload.print(gpa, ",\"role\":\"{s}\",\"host\":\"{s}\",\"workspace\":", .{ @tagName(p.role), @tagName(p.host) });
            try schema.appendJsonString(gpa, &payload, p.workspace);
            if (p.title) |title| {
                try payload.appendSlice(gpa, ",\"title\":");
                try payload.appendSlice(gpa, title);
            }
            if (p.language) |language| {
                try payload.appendSlice(gpa, ",\"language\":");
                try payload.appendSlice(gpa, language);
            }
            if (p.parent) |parent| {
                try payload.appendSlice(gpa, ",\"parent\":");
                try schema.appendJsonString(gpa, &payload, parent);
            }
            try payload.print(gpa, ",\"created\":{d},\"updated\":{d},\"turns\":{d},\"opened\":[", .{ p.created_ms, p.updated_ms, p.turns });
            for (p.opened_ms, 0..) |ts, i| {
                if (i > 0) try payload.append(gpa, ',');
                try payload.print(gpa, "{d}", .{ts});
            }
            try payload.append(gpa, ']');
        },
        .opened => |o| {
            try payload.appendSlice(gpa, "{\"op\":\"opened\",\"id\":");
            try schema.appendJsonString(gpa, &payload, o.id);
            try payload.print(gpa, ",\"host\":\"{s}\",\"ts\":{d}", .{ @tagName(o.host), o.ts_ms });
        },
        .del => |d| {
            try payload.appendSlice(gpa, "{\"op\":\"del\",\"id\":");
            try schema.appendJsonString(gpa, &payload, d.id);
            try payload.print(gpa, ",\"ts\":{d}", .{d.ts_ms});
        },
    }
    try log_mod.appendFramed(gpa, out, payload.items);
}

/// Parses one index line, newline included. Null for a line that fails
/// its checksum or does not parse.
pub fn decodeRecord(arena: std.mem.Allocator, line: []const u8) error{OutOfMemory}!?Record {
    const payload = log_mod.checkFrame(line) catch return null;
    const object = try std.mem.concat(arena, u8, &.{ payload, "}" });
    const f = schema.Fields.parseObject(arena, object) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.BadBody => null,
    };
    const op = f.req([]const u8, "op") catch return null;
    const id = f.req([]const u8, "id") catch return null;
    if (std.mem.eql(u8, op, "put")) {
        var summary: Summary = .{
            .id = id,
            .role = f.req(schema.Role, "role") catch return null,
            .host = f.req(schema.Host, "host") catch return null,
            .workspace = f.req([]const u8, "workspace") catch return null,
            .title = f.raw("title"),
            .language = f.raw("language"),
            .parent = f.opt([]const u8, "parent") catch return null,
            .created_ms = f.req(u64, "created") catch return null,
            .updated_ms = f.req(u64, "updated") catch return null,
            .turns = f.req(u64, "turns") catch return null,
        };
        const opened = f.req([]const u64, "opened") catch return null;
        if (opened.len != host_count) return null;
        @memcpy(&summary.opened_ms, opened);
        return .{ .put = summary };
    }
    if (std.mem.eql(u8, op, "opened")) {
        return .{ .opened = .{
            .id = id,
            .host = f.req(schema.Host, "host") catch return null,
            .ts_ms = f.req(u64, "ts") catch return null,
        } };
    }
    if (std.mem.eql(u8, op, "del")) {
        return .{ .del = .{ .id = id, .ts_ms = f.req(u64, "ts") catch return null } };
    }
    return null;
}

pub const Entry = struct {
    summary: Summary,
    deleted: bool,
};

/// The folded index: the newest summary per id, and every tombstone.
/// Owns everything through its arena.
pub const Index = struct {
    arena: std.heap.ArenaAllocator,
    entries: std.StringArrayHashMapUnmanaged(Entry) = .empty,
    /// Lines that failed their checksum or did not parse.
    damaged_lines: u64 = 0,

    pub fn deinit(index: *Index) void {
        index.arena.deinit();
    }

    /// Pure: folds index bytes. A bad line is counted and skipped; a torn
    /// final line is ignored.
    pub fn fold(gpa: std.mem.Allocator, bytes: []const u8) error{OutOfMemory}!Index {
        var index: Index = .{ .arena = .init(gpa) };
        errdefer index.deinit();
        const arena = index.arena.allocator();
        var at: usize = 0;
        while (std.mem.findScalarPos(u8, bytes, at, '\n')) |nl| : (at = nl + 1) {
            const record = (try decodeRecord(arena, bytes[at .. nl + 1])) orelse {
                index.damaged_lines += 1;
                continue;
            };
            try index.apply(record);
        }
        return index;
    }

    pub fn apply(index: *Index, record: Record) error{OutOfMemory}!void {
        const arena = index.arena.allocator();
        switch (record) {
            .put => |summary| {
                const slot = try index.entries.getOrPut(arena, summary.id);
                if (slot.found_existing) {
                    // A tombstone is final.
                    if (slot.value_ptr.deleted) return;
                    var merged = summary;
                    for (&merged.opened_ms, slot.value_ptr.summary.opened_ms) |*ts, old| ts.* = @max(ts.*, old);
                    slot.value_ptr.summary = merged;
                } else {
                    slot.value_ptr.* = .{ .summary = summary, .deleted = false };
                }
            },
            .opened => |o| if (index.entries.getPtr(o.id)) |entry| {
                const i = @intFromEnum(o.host);
                entry.summary.opened_ms[i] = @max(entry.summary.opened_ms[i], o.ts_ms);
            },
            .del => |d| {
                const slot = try index.entries.getOrPut(arena, d.id);
                if (!slot.found_existing) {
                    slot.value_ptr.summary = .{ .id = d.id, .role = .root, .host = .app, .workspace = "", .created_ms = d.ts_ms, .updated_ms = d.ts_ms, .turns = 0 };
                }
                slot.value_ptr.deleted = true;
            },
        }
    }

    pub fn isDeleted(index: *const Index, id: []const u8) bool {
        const entry = index.entries.get(id) orelse return false;
        return entry.deleted;
    }
};

// ---------------------------------------------------------------------------
// Effectful shell

pub const Error = error{ Busy, Io, OutOfMemory };

pub const Filter = union(enum) { all, workspace: []const u8 };

/// Where the next page of `list` starts.
pub const Cursor = struct {
    updated_ms: u64,
    id_buffer: [255]u8 = undefined,
    id_len: usize = 0,

    pub fn id(c: *const Cursor) []const u8 {
        return c.id_buffer[0..c.id_len];
    }
};

pub const Page = struct {
    arena: std.heap.ArenaAllocator,
    /// Newest first.
    items: []Summary,
    next: ?Cursor,

    pub fn deinit(page: *Page) void {
        page.arena.deinit();
    }
};

pub const Target = union(enum) {
    /// The newest updated root session in the workspace, from any host.
    last,
    /// The root session this host opened last in the workspace (`-c`).
    last_opened: schema.Host,
};

pub const Rebuilt = struct { sessions: u64, swept: u64 };

/// `index.lock`, opened once and kept for the manager's life (it is never
/// renamed or removed). The flock excludes other processes; `mutex`
/// excludes this process's threads, which share one open file and so one
/// flock. A crash drops both, as before.
pub const IndexLock = struct {
    mutex: std.Io.Mutex = .init,
    file: ?storage.File = null,

    pub fn close(lock: *IndexLock, s: storage.Storage) void {
        if (lock.file) |file| s.closeFile(file);
        lock.file = null;
    }
};

pub const Catalog = struct {
    env: *const session_mod.Env,
    lock: *IndexLock,

    pub fn put(cat: Catalog, summary: Summary) Error!void {
        try cat.append(.{ .put = summary });
        cat.env.observeCatalog(summary.id, .index_put);
    }

    pub fn opened(cat: Catalog, id: []const u8, host: schema.Host, ts_ms: u64) Error!void {
        try cat.append(.{ .opened = .{ .id = id, .host = host, .ts_ms = ts_ms } });
    }

    pub fn del(cat: Catalog, id: []const u8, ts_ms: u64) Error!void {
        try cat.append(.{ .del = .{ .id = id, .ts_ms = ts_ms } });
        cat.env.observeCatalog(id, .index_del);
    }

    /// Reads and folds the index without its lock. The file only grows,
    /// or is replaced whole by a rename, so a reader always sees a prefix
    /// of whole lines plus at most one torn line, which the fold ignores.
    /// Damaged lines are dropped by rewriting the index, reported once.
    pub fn load(cat: Catalog, gpa: std.mem.Allocator) Error!Index {
        const bytes = try cat.readIndex(gpa);
        defer gpa.free(bytes);
        var index = try Index.fold(gpa, bytes);
        errdefer index.deinit();
        if (index.damaged_lines > 0) {
            const lock = try cat.acquireIndexLock();
            defer cat.releaseIndexLock(lock);
            try cat.writeFresh(&index);
            diag.report(cat.env.diagnostics, .{ .kind = .index_healed, .session_id = "", .count = index.damaged_lines });
            index.damaged_lines = 0;
        }
        return index;
    }

    /// Root sessions, newest updated first, `limit` per page. Entries whose
    /// folder is gone are tombstoned and skipped (self-heal).
    pub fn list(cat: Catalog, gpa: std.mem.Allocator, filter: Filter, cursor: ?Cursor, limit: usize) Error!Page {
        var index = try cat.load(gpa);
        defer index.deinit();
        var page: Page = .{ .arena = .init(gpa), .items = &.{}, .next = null };
        errdefer page.deinit();
        const arena = page.arena.allocator();
        const candidates = try cat.sorted(arena, &index, filter);
        var start: usize = 0;
        if (cursor) |c| {
            while (start < candidates.len and !isAfter(candidates[start], c)) start += 1;
        }
        var items: std.ArrayList(Summary) = .empty;
        var i = start;
        while (i < candidates.len and items.items.len < limit) : (i += 1) {
            if (!try cat.healIfGone(candidates[i].id)) continue;
            try items.append(arena, try copySummary(arena, candidates[i]));
        }
        if (i < candidates.len and items.items.len == limit) {
            const last = items.items[items.items.len - 1];
            var next: Cursor = .{ .updated_ms = last.updated_ms, .id_len = last.id.len };
            @memcpy(next.id_buffer[0..last.id.len], last.id);
            page.next = next;
        }
        page.items = items.items;
        return page;
    }

    /// Resolves `.last` or `.last_opened{host}` to an id the caller owns.
    pub fn resolve(cat: Catalog, gpa: std.mem.Allocator, workspace: []const u8, target: Target) Error!?[]u8 {
        var index = try cat.load(gpa);
        defer index.deinit();
        while (true) {
            var best: ?*const Summary = null;
            var best_key: u64 = 0;
            var it = index.entries.iterator();
            while (it.next()) |kv| {
                const entry = kv.value_ptr;
                if (entry.deleted or entry.summary.role != .root) continue;
                if (!std.mem.eql(u8, entry.summary.workspace, workspace)) continue;
                const key = switch (target) {
                    .last => entry.summary.updated_ms,
                    .last_opened => |host| entry.summary.opened_ms[@intFromEnum(host)],
                };
                if (key == 0 and target == .last_opened) continue;
                if (best == null or key > best_key) {
                    best = &entry.summary;
                    best_key = key;
                }
            }
            const found = best orelse return null;
            if (try cat.healIfGone(found.id)) return try gpa.dupe(u8, found.id);
            // Gone: its tombstone is in the file now; forget it here too.
            index.entries.getPtr(found.id).?.deleted = true;
        }
    }

    pub fn isDeleted(cat: Catalog, gpa: std.mem.Allocator, id: []const u8) Error!bool {
        var index = try cat.load(gpa);
        defer index.deinit();
        return index.isDeleted(id);
    }

    /// Re-derives the index from the session folders under `index.lock`,
    /// keeping tombstones and each host's open times from the old index,
    /// then sweeps `.trash` and old `.tmp` entries.
    pub fn rebuild(cat: Catalog, gpa: std.mem.Allocator) Error!Rebuilt {
        const env = cat.env;
        const s = env.s;
        const lock = try cat.acquireIndexLock();
        defer cat.releaseIndexLock(lock);
        const old_bytes = try cat.readIndex(gpa);
        defer gpa.free(old_bytes);
        var old = try Index.fold(gpa, old_bytes);
        defer old.deinit();

        var fresh: Index = .{ .arena = .init(gpa) };
        defer fresh.deinit();
        const arena = fresh.arena.allocator();
        var sessions: u64 = 0;
        var listing = s.list(env.root);
        while (listing.next() catch return error.Io) |entry| {
            if (entry.kind != .directory or !schema.validId(entry.name)) continue;
            const id = try arena.dupe(u8, entry.name);
            var summary = session_mod.readSummary(env, id) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                // Damaged or unreadable: listed by id so it can be recovered.
                else => {
                    try fresh.apply(.{ .put = .{ .id = id, .role = .root, .host = .app, .workspace = "", .created_ms = 0, .updated_ms = 0, .turns = 0 } });
                    sessions += 1;
                    continue;
                },
            };
            defer summary.deinit(gpa);
            var derived = try summaryFrom(arena, &summary);
            if (old.entries.get(id)) |known| derived.opened_ms = known.summary.opened_ms;
            if (old.isDeleted(id)) continue;
            try fresh.apply(.{ .put = derived });
            sessions += 1;
        }
        if (env.isPlanted(.rebuild_resurrects)) try cat.plantedResurrect(&fresh);
        var it = old.entries.iterator();
        while (it.next()) |kv| {
            if (kv.value_ptr.deleted) try fresh.apply(.{ .del = .{ .id = try arena.dupe(u8, kv.key_ptr.*), .ts_ms = kv.value_ptr.summary.updated_ms } });
        }
        try cat.writeFresh(&fresh);
        const swept = try cat.sweep();
        if (swept > 0) diag.report(env.diagnostics, .{ .kind = .rebuild_swept, .session_id = "", .count = swept });
        env.observeCatalog("", .rebuilt);
        return .{ .sessions = sessions, .swept = swept };
    }

    /// Hooks only: the planted Catalog bug lists `.trash` entries as live.
    fn plantedResurrect(cat: Catalog, fresh: *Index) Error!void {
        const s = cat.env.s;
        const trash = s.openDir(cat.env.root, ".trash") catch return;
        defer s.closeDir(trash);
        var listing = s.list(trash);
        while (listing.next() catch return error.Io) |entry| {
            if (!schema.validId(entry.name)) continue;
            const id = try fresh.arena.allocator().dupe(u8, entry.name);
            try fresh.apply(.{ .put = .{ .id = id, .role = .root, .host = .app, .workspace = "/w", .created_ms = 0, .updated_ms = 0, .turns = 0 } });
        }
    }

    // -- internals -----------------------------------------------------------

    fn append(cat: Catalog, record: Record) Error!void {
        const env = cat.env;
        const s = env.s;
        const gpa = env.gpa;
        var line: std.ArrayList(u8) = .empty;
        defer line.deinit(gpa);
        encodeRecord(gpa, &line, record) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            error.TooLarge => error.Io,
        };
        const lock = try cat.acquireIndexLock();
        defer cat.releaseIndexLock(lock);
        const file = s.openFile(env.root, index_name, .read_write) catch |err| switch (err) {
            error.NotFound => s.createFile(env.root, index_name) catch return error.Io,
            else => return error.Io,
        };
        defer s.closeDerived(file);
        const len = s.length(file) catch return error.Io;
        // A crash may have left half a line; never glue a record onto it.
        const end = log_mod.lastLineEnd(s, file, len) catch return error.Io;
        if (end != len) s.setLength(file, end) catch return error.Io;
        s.writeAt(file, line.items, end) catch return error.Io;
        if (end + line.items.len > env.options.index_compact_bytes) {
            const bytes = try cat.readIndex(gpa);
            defer gpa.free(bytes);
            var index = try Index.fold(gpa, bytes);
            defer index.deinit();
            try cat.writeFresh(&index);
        }
    }

    /// Replaces the index with one line per id, atomically: a temporary
    /// file, synced, renamed over the old one. Caller holds `index.lock`.
    fn writeFresh(cat: Catalog, index: *const Index) Error!void {
        const env = cat.env;
        const s = env.s;
        const gpa = env.gpa;
        var bytes: std.ArrayList(u8) = .empty;
        defer bytes.deinit(gpa);
        var it = index.entries.iterator();
        while (it.next()) |kv| {
            const entry = kv.value_ptr;
            const record: Record = if (entry.deleted)
                .{ .del = .{ .id = entry.summary.id, .ts_ms = entry.summary.updated_ms } }
            else
                .{ .put = entry.summary };
            encodeRecord(gpa, &bytes, record) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                error.TooLarge => error.Io,
            };
        }
        s.deleteFile(env.root, index_tmp_name) catch |err| switch (err) {
            error.NotFound => {},
            else => return error.Io,
        };
        const file = s.createFile(env.root, index_tmp_name) catch return error.Io;
        defer s.closeFile(file);
        if (bytes.items.len > 0) s.writeAt(file, bytes.items, 0) catch return error.Io;
        s.sync(file) catch return error.Io;
        s.rename(env.root, index_tmp_name, env.root, index_name) catch return error.Io;
        s.syncDir(env.root) catch return error.Io;
    }

    /// The whole index file; empty when there is none. Caller owns it.
    fn readIndex(cat: Catalog, gpa: std.mem.Allocator) Error![]u8 {
        const s = cat.env.s;
        const file = s.openFile(cat.env.root, index_name, .read_only) catch |err| return switch (err) {
            error.NotFound => try gpa.alloc(u8, 0),
            else => error.Io,
        };
        defer s.closeFile(file);
        const len = s.length(file) catch return error.Io;
        var bytes: std.ArrayList(u8) = .empty;
        errdefer bytes.deinit(gpa);
        try bytes.resize(gpa, @intCast(len));
        const n = s.readAt(file, bytes.items, 0) catch return error.Io;
        // Shorter only if a writer cut a torn tail meanwhile.
        bytes.shrinkRetainingCapacity(n);
        return bytes.toOwnedSlice(gpa);
    }

    fn acquireIndexLock(cat: Catalog) Error!storage.File {
        const env = cat.env;
        const s = env.s;
        cat.lock.mutex.lockUncancelable(s.io);
        errdefer cat.lock.mutex.unlock(s.io);
        const file = cat.lock.file orelse blk: {
            const opened_file = s.openFile(env.root, lock_name, .read_write) catch |err| switch (err) {
                error.NotFound => s.createFile(env.root, lock_name) catch |create_err| switch (create_err) {
                    // Another process created it first.
                    error.AlreadyExists => s.openFile(env.root, lock_name, .read_write) catch return error.Io,
                    else => return error.Io,
                },
                else => return error.Io,
            };
            cat.lock.file = opened_file;
            break :blk opened_file;
        };
        var waited: u64 = 0;
        while (!(s.tryLock(file) catch return error.Io)) {
            if (waited >= env.options.lock_wait_ms) return error.Busy;
            s.io.sleep(.fromMilliseconds(5), .awake) catch return error.Busy;
            waited += 5;
        }
        return file;
    }

    fn releaseIndexLock(cat: Catalog, file: storage.File) void {
        cat.env.s.unlock(file);
        cat.lock.mutex.unlock(cat.env.s.io);
    }

    /// Whether the session's folder still exists; if not, it is tombstoned
    /// (`tla/Catalog.tla` `Heal`) and reported once.
    /// One `stat` of the folder, no open: a published folder always holds
    /// its synced log (publish renames it in whole, delete renames it away),
    /// so the folder alone decides. `rebuild` checks the logs themselves.
    fn healIfGone(cat: Catalog, id: []const u8) Error!bool {
        const s = cat.env.s;
        const present = if (s.stat(cat.env.root, id)) |st| st.kind == .directory else |err| switch (err) {
            error.NotFound => false,
            else => return error.Io,
        };
        if (present) return true;
        try cat.del(id, nowMs(cat.env));
        diag.report(cat.env.diagnostics, .{ .kind = .index_healed, .session_id = id, .count = 1 });
        cat.env.observeCatalog(id, .healed);
        return false;
    }

    fn sorted(cat: Catalog, arena: std.mem.Allocator, index: *const Index, filter: Filter) Error![]Summary {
        _ = cat;
        var out: std.ArrayList(Summary) = .empty;
        var it = index.entries.iterator();
        while (it.next()) |kv| {
            const entry = kv.value_ptr;
            if (entry.deleted or entry.summary.role != .root) continue;
            switch (filter) {
                .all => {},
                .workspace => |w| if (!std.mem.eql(u8, entry.summary.workspace, w)) continue,
            }
            try out.append(arena, entry.summary);
        }
        std.mem.sort(Summary, out.items, {}, newerFirst);
        return out.items;
    }

    /// Removes `.trash` entries, and `.tmp` entries old enough that no
    /// publish can still be writing them. Caller holds `index.lock`.
    fn sweep(cat: Catalog) Error!u64 {
        const env = cat.env;
        const s = env.s;
        var swept: u64 = 0;
        const now = std.math.cast(i64, nowMs(env)) orelse std.math.maxInt(i64);
        for ([_][]const u8{ ".trash", ".tmp" }) |folder| {
            const dir = s.openDir(env.root, folder) catch |err| switch (err) {
                error.NotFound => continue,
                else => return error.Io,
            };
            defer s.closeDir(dir);
            var names: std.ArrayList([]u8) = .empty;
            defer {
                for (names.items) |n| env.gpa.free(n);
                names.deinit(env.gpa);
            }
            var listing = s.list(dir);
            while (listing.next() catch return error.Io) |entry| {
                try names.append(env.gpa, try env.gpa.dupe(u8, entry.name));
            }
            for (names.items) |name| {
                if (!schema.validId(name)) continue;
                if (std.mem.eql(u8, folder, ".tmp")) {
                    const st = s.stat(dir, name) catch continue;
                    if (now - st.mtime_ms < sweep_after_ms) continue;
                }
                s.deleteTree(dir, name) catch return error.Io;
                swept += 1;
            }
        }
        return swept;
    }
};

fn nowMs(env: *const session_mod.Env) u64 {
    return std.math.cast(u64, std.Io.Timestamp.now(env.s.io, .real).toMilliseconds()) orelse 0;
}

fn newerFirst(_: void, a: Summary, b: Summary) bool {
    if (a.updated_ms != b.updated_ms) return a.updated_ms > b.updated_ms;
    return std.mem.lessThan(u8, a.id, b.id);
}

/// Whether `s` sorts after the cursor's entry.
fn isAfter(s: Summary, c: Cursor) bool {
    if (s.updated_ms != c.updated_ms) return s.updated_ms < c.updated_ms;
    return std.mem.lessThan(u8, c.id(), s.id);
}

fn copySummary(arena: std.mem.Allocator, s: Summary) error{OutOfMemory}!Summary {
    var copy = s;
    copy.id = try arena.dupe(u8, s.id);
    copy.workspace = try arena.dupe(u8, s.workspace);
    if (s.title) |t| copy.title = try arena.dupe(u8, t);
    if (s.language) |l| copy.language = try arena.dupe(u8, l);
    if (s.parent) |p| copy.parent = try arena.dupe(u8, p);
    return copy;
}

/// The index record for a session read from its folder.
pub fn summaryFrom(arena: std.mem.Allocator, summary: *const session_mod.Summary) error{OutOfMemory}!Summary {
    const identity = &summary.identity;
    const state = &summary.state;
    const workspace = if (state.workspace) |raw|
        std.json.parseFromSliceLeaky([]const u8, arena, raw, .{}) catch identity.workspace
    else
        identity.workspace;
    return .{
        .id = try arena.dupe(u8, identity.id),
        .role = identity.role,
        .host = identity.host,
        .workspace = try arena.dupe(u8, workspace),
        .title = if (state.title) |t| try arena.dupe(u8, t) else null,
        .language = if (state.language) |l| try arena.dupe(u8, l) else null,
        .parent = if (identity.parent) |p| try arena.dupe(u8, p) else null,
        .created_ms = summary.created_ms,
        .updated_ms = summary.updated_ms,
        .turns = state.committed + state.interrupted,
    };
}

// ---------------------------------------------------------------------------
// Tests (pure)

const testing = std.testing;

fn sample(id: []const u8, updated: u64) Summary {
    return .{ .id = id, .role = .root, .host = .app, .workspace = "/w", .created_ms = 1, .updated_ms = updated, .turns = 2 };
}

test "records round trip" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var out: std.ArrayList(u8) = .empty;
    var summary = sample("abc", 7);
    summary.title = "\"a \\\"title\\\"\"";
    summary.parent = "p";
    summary.opened_ms[@intFromEnum(schema.Host.ask)] = 9;
    try encodeRecord(arena.allocator(), &out, .{ .put = summary });
    try encodeRecord(arena.allocator(), &out, .{ .opened = .{ .id = "abc", .host = .acp, .ts_ms = 10 } });
    try encodeRecord(arena.allocator(), &out, .{ .del = .{ .id = "abc", .ts_ms = 11 } });
    var lines = std.mem.splitScalar(u8, out.items, '\n');
    const put = (try decodeRecord(arena.allocator(), try std.mem.concat(arena.allocator(), u8, &.{ lines.next().?, "\n" }))).?.put;
    try testing.expectEqualStrings("\"a \\\"title\\\"\"", put.title.?);
    try testing.expectEqual(@as(u64, 9), put.opened_ms[@intFromEnum(schema.Host.ask)]);
    const opened = (try decodeRecord(arena.allocator(), try std.mem.concat(arena.allocator(), u8, &.{ lines.next().?, "\n" }))).?.opened;
    try testing.expectEqual(schema.Host.acp, opened.host);
    const del = (try decodeRecord(arena.allocator(), try std.mem.concat(arena.allocator(), u8, &.{ lines.next().?, "\n" }))).?.del;
    try testing.expectEqual(@as(u64, 11), del.ts_ms);
}

test "the fold keeps the newest put, merges open times, and a tombstone is final" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var out: std.ArrayList(u8) = .empty;
    const a = arena.allocator();
    try encodeRecord(a, &out, .{ .put = sample("s1", 1) });
    try encodeRecord(a, &out, .{ .opened = .{ .id = "s1", .host = .app, .ts_ms = 5 } });
    var newer = sample("s1", 3);
    newer.turns = 4;
    try encodeRecord(a, &out, .{ .put = newer });
    try encodeRecord(a, &out, .{ .put = sample("s2", 2) });
    try encodeRecord(a, &out, .{ .del = .{ .id = "s2", .ts_ms = 4 } });
    try encodeRecord(a, &out, .{ .put = sample("s2", 9) }); // stale writer: ignored
    try out.appendSlice(a, "{\"op\":\"put\",\"id\":\"torn"); // torn tail
    var index = try Index.fold(testing.allocator, out.items);
    defer index.deinit();
    try testing.expectEqual(@as(u64, 0), index.damaged_lines);
    const s1 = index.entries.get("s1").?.summary;
    try testing.expectEqual(@as(u64, 4), s1.turns);
    try testing.expectEqual(@as(u64, 5), s1.opened_ms[@intFromEnum(schema.Host.app)]);
    try testing.expect(index.isDeleted("s2"));

    // A flipped byte in a middle line is counted, never folded.
    out.items[20] ^= 1;
    var damaged = try Index.fold(testing.allocator, out.items);
    defer damaged.deinit();
    try testing.expectEqual(@as(u64, 1), damaged.damaged_lines);
}

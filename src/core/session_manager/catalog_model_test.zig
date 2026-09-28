//! L3 and L4 against `tla/Catalog.tla`: publish, delete, crashes between
//! the folder change and the index write, self-heal and rebuild, driven
//! through the public API. Every run writes a trace for TLC.

const std = @import("std");
const api = @import("api.zig");
const catalog_mod = @import("catalog.zig");
const session_mod = @import("session.zig");
const trace = @import("trace.zig");
const Fault = @import("storage_fault.zig").Fault;

const testing = std.testing;
const gpa = testing.allocator;
const io = testing.io;
const Io = std.Io;

const CatTracer = struct {
    trace: trace.Trace,
    root: Io.Dir,
    fault: *Fault,
    ids: [3]?[]u8 = .{ null, null, null },
    published: [3]bool = .{ false, false, false },
    pending: [3][]const u8 = .{ "none", "none", "none" },
    ever_deleted: [3]bool = .{ false, false, false },
    /// Kill the process right after this step for this session.
    kill_after: ?struct { what: enum { published, trashed }, id: []const u8 } = null,

    const labels = [3][]const u8{ "s1", "s2", "s3" };

    fn deinit(t: *CatTracer) void {
        for (t.ids) |maybe| if (maybe) |id| gpa.free(id);
    }

    fn slot(t: *CatTracer, id: []const u8) usize {
        for (t.ids, 0..) |maybe, i| {
            if (maybe) |known| if (std.mem.eql(u8, known, id)) return i;
        }
        for (&t.ids, 0..) |*maybe, i| if (maybe.* == null) {
            maybe.* = gpa.dupe(u8, id) catch @panic("oom");
            return i;
        };
        @panic("more than three traced sessions");
    }

    fn observer(t: *CatTracer) session_mod.Observer {
        return .{ .context = t, .notify = notify };
    }

    fn catalogObserver(t: *CatTracer) session_mod.CatalogObserver {
        return .{ .context = t, .notify = notifyCatalog };
    }

    fn notify(context: *anyopaque, session: *session_mod.Session, what: session_mod.Observed) void {
        const t: *CatTracer = @ptrCast(@alignCast(context));
        if (what != .published) return;
        const i = t.slot(session.id());
        t.published[i] = true;
        t.pending[i] = "put";
        t.emit("Publish", labels[i]);
        t.maybeKill(.published, session.id());
    }

    fn notifyCatalog(context: *anyopaque, id: []const u8, what: session_mod.CatalogStep) void {
        const t: *CatTracer = @ptrCast(@alignCast(context));
        if (what == .rebuilt) {
            t.pending = .{ "none", "none", "none" };
            t.emit("Rebuild", null);
            return;
        }
        const i = t.slot(id);
        switch (what) {
            // Updates of an indexed session are not spec actions.
            .index_put => if (std.mem.eql(u8, t.pending[i], "put")) {
                t.pending[i] = "none";
                t.emit("IndexPut", labels[i]);
            },
            // A heal's tombstone is reported by `.healed`.
            .index_del => if (std.mem.eql(u8, t.pending[i], "del")) {
                t.pending[i] = "none";
                t.emit("IndexDel", labels[i]);
            },
            .trashed => {
                t.pending[i] = "del";
                t.ever_deleted[i] = true;
                t.emit("Trash", labels[i]);
                t.maybeKill(.trashed, id);
            },
            .purged => t.emit("Purge", labels[i]),
            .healed => t.emit("Heal", labels[i]),
            .rebuilt => unreachable,
        }
    }

    fn maybeKill(t: *CatTracer, what: @TypeOf(t.kill_after.?.what), id: []const u8) void {
        const k = t.kill_after orelse return;
        if (k.what == what and std.mem.eql(u8, k.id, id)) t.fault.kill();
    }

    /// fx dies: owed index writes are forgotten.
    fn crash(t: *CatTracer) void {
        t.pending = .{ "none", "none", "none" };
        t.emit("Crash", null);
        t.kill_after = null;
        t.fault.restart();
    }

    fn dirOf(t: *CatTracer, i: usize) []const u8 {
        const id = t.ids[i] orelse return "none";
        var path: [300]u8 = undefined;
        if (exists(t.root, id)) return "live";
        if (exists(t.root, std.fmt.bufPrint(&path, ".trash/{s}", .{id}) catch return "none")) return "trash";
        return if (t.published[i]) "gone" else "none";
    }

    const IndexLine = struct { id: []const u8, op: []const u8 };

    fn indexOf(t: *CatTracer, arena: std.mem.Allocator) ![]IndexLine {
        var out: std.ArrayList(IndexLine) = .empty;
        const bytes = t.root.readFileAlloc(io, "index.jsonl", gpa, .limited(8 << 20)) catch return out.items;
        defer gpa.free(bytes);
        var last_op: [3][]const u8 = .{ "none", "none", "none" };
        var at: usize = 0;
        while (std.mem.findScalarPos(u8, bytes, at, '\n')) |nl| : (at = nl + 1) {
            const record = (try catalog_mod.decodeRecord(arena, bytes[at .. nl + 1])) orelse continue;
            const id, const op = switch (record) {
                .put => |p| .{ p.id, "put" },
                .del => |d| .{ d.id, "del" },
                .opened => continue,
            };
            const i = t.slot(id);
            if (std.mem.eql(u8, op, "put") and std.mem.eql(u8, last_op[i], "put")) continue;
            last_op[i] = op;
            try out.append(arena, .{ .id = labels[i], .op = op });
        }
        return out.items;
    }

    fn emit(t: *CatTracer, event: []const u8, s: ?[]const u8) void {
        var arena_state = std.heap.ArenaAllocator.init(gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        const index = t.indexOf(arena) catch return;
        const Names = struct { s1: []const u8, s2: []const u8, s3: []const u8 };
        var deleted: std.ArrayList([]const u8) = .empty;
        for (t.ever_deleted, labels) |d, label| if (d) deleted.append(arena, label) catch return;
        const Set = struct { @"$set": []const []const u8 };
        const dir: Names = .{ .s1 = t.dirOf(0), .s2 = t.dirOf(1), .s3 = t.dirOf(2) };
        const pending: Names = .{ .s1 = t.pending[0], .s2 = t.pending[1], .s3 = t.pending[2] };
        const ever: Set = .{ .@"$set" = deleted.items };
        if (s) |label| {
            t.trace.write(.{ .event = event, .s = label, .dir = dir, .index = index, .pending = pending, .everDeleted = ever });
        } else {
            t.trace.write(.{ .event = event, .dir = dir, .index = index, .pending = pending, .everDeleted = ever });
        }
    }
};

fn exists(root: Io.Dir, path: []const u8) bool {
    _ = root.statFile(io, path, .{ .follow_symlinks = false }) catch return false;
    return true;
}

fn publish(m: *api.Manager) ![]u8 {
    const s = try m.openNew(.{ .workspace = "/w", .host = .app });
    _ = try s.append(&.{ .turn_started, .turn_committed });
    const id = try gpa.dupe(u8, s.id());
    s.release();
    return id;
}

fn listCount(m: *api.Manager) !usize {
    var page = try m.list(gpa, .all, null, 10);
    defer page.deinit();
    return page.items.len;
}

fn runCatalogTrace(case: []const u8, planted: trace.Planted) !void {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const base = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(base);
    const root_path = try std.fs.path.join(gpa, &.{ base, "sessions", "v2" });
    defer gpa.free(root_path);
    var fault = Fault.init(gpa, io, 1);
    defer fault.deinit();
    const m = try api.Manager.init(gpa, io, .{ .root = root_path, .lock_wait_ms = 50 });
    defer m.deinit();
    m.setFault(&fault);
    try tmp.dir.createDirPath(io, "sessions/v2");
    var root = try tmp.dir.openDir(io, "sessions/v2", .{});
    defer root.close(io);
    var tracer: CatTracer = .{ .trace = try trace.Trace.create(gpa, io, "Catalog", case), .root = root, .fault = &fault };
    defer tracer.deinit();
    m.env.observer = tracer.observer();
    m.env.catalog_observer = tracer.catalogObserver();
    m.env.planted = planted;

    const s1 = try publish(m);
    defer gpa.free(s1);
    const s2 = try publish(m);
    defer gpa.free(s2);

    // s3: the process dies between its publish and its index put.
    {
        const s = try m.openNew(.{ .workspace = "/w", .host = .app });
        tracer.kill_after = .{ .what = .published, .id = s.id() };
        _ = try s.append(&.{ .turn_started, .turn_committed });
        s.release();
        tracer.crash();
    }
    try testing.expectEqual(@as(usize, 2), try listCount(m));

    // A clean delete, then one that dies between trash and tombstone.
    try m.delete(s1);
    tracer.kill_after = .{ .what = .trashed, .id = s2 };
    try testing.expectError(error.Io, m.delete(s2));
    tracer.crash();
    // list heals the stale entry, then rebuild purges .trash and finds s3.
    // The planted run rebuilds straight away: a tombstone from the heal
    // would otherwise stop the planted bug (tombstones survive rebuild).
    if (planted == .none) try testing.expectEqual(@as(usize, 0), try listCount(m));
    _ = try m.rebuild();
    if (planted == .none) {
        try testing.expectEqual(@as(usize, 1), try listCount(m));
        try testing.expectError(error.NotFound, m.read(gpa, s2, .start, .forward, 1));
    }
    try tracer.trace.finish();
}

test "Catalog trace: publish, crash before the index, delete, crash after trash, heal, rebuild" {
    try runCatalogTrace("crashes-heal-rebuild", .none);
}

test "Catalog trace: planted bug, rebuild resurrects a trashed id" {
    try runCatalogTrace("planted-rebuild_resurrects", .rebuild_resurrects);
}

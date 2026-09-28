//! The API as fx will use it: this file imports only `api.zig`.

const std = @import("std");
const api = @import("api.zig");

const testing = std.testing;
const gpa = testing.allocator;
const io = testing.io;

const Recorder = struct {
    kinds: std.ArrayList(api.DiagnosticKind) = .empty,
    mutex: std.Io.Mutex = .init,

    fn sink(r: *Recorder) api.Diagnostics {
        return .{ .context = r, .emit = emit };
    }

    fn emit(context: ?*anyopaque, event: api.Diagnostic) void {
        const r: *Recorder = @ptrCast(@alignCast(context.?));
        r.mutex.lockUncancelable(io);
        defer r.mutex.unlock(io);
        r.kinds.append(gpa, event.kind) catch {};
    }

    fn count(r: *Recorder, kind: api.DiagnosticKind) usize {
        var n: usize = 0;
        for (r.kinds.items) |k| {
            if (k == kind) n += 1;
        }
        return n;
    }
};

const Fixture = struct {
    tmp: testing.TmpDir,
    root: []u8,
    recorder: Recorder = .{},
    manager: *api.Manager,

    fn init(f: *Fixture) !void {
        return f.initWith(1 << 20);
    }

    fn initWith(f: *Fixture, index_compact_bytes: u64) !void {
        f.tmp = testing.tmpDir(.{ .iterate = true });
        const base = try f.tmp.dir.realPathFileAlloc(io, ".", gpa);
        defer gpa.free(base);
        f.root = try std.fs.path.join(gpa, &.{ base, "sessions", "v2" });
        f.recorder = .{};
        f.manager = try api.Manager.init(gpa, io, .{
            .root = f.root,
            .diagnostics = f.recorder.sink(),
            .lock_wait_ms = 50,
            .index_compact_bytes = index_compact_bytes,
        });
    }

    fn deinit(f: *Fixture) void {
        f.manager.deinit();
        f.recorder.kinds.deinit(gpa);
        gpa.free(f.root);
        f.tmp.cleanup();
    }

    fn dir(f: *Fixture) !std.Io.Dir {
        return std.Io.Dir.cwd().openDir(io, f.root, .{});
    }
};

const piece: api.Event = .{ .item = .{ .type = "assistant", .data = "{\"text\":\"hi\"}" } };

fn listIds(m: *api.Manager, filter: api.Filter) ![][]const u8 {
    var page = try m.list(gpa, filter, null, 100);
    defer page.deinit();
    const ids = try gpa.alloc([]const u8, page.items.len);
    for (page.items, ids) |item, *id| id.* = try gpa.dupe(u8, item.id);
    return ids;
}

fn freeIds(ids: [][]const u8) void {
    for (ids) |id| gpa.free(id);
    gpa.free(ids);
}

test "init touches nothing on disk" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    try testing.expectError(error.FileNotFound, f.dir());
    // Held settings and a close before any turn: still nothing, not even the
    // root folder, so fx starting up does no session I/O (D2).
    const s = try f.manager.openNew(.{ .workspace = "/w", .host = .app });
    _ = try s.append(&.{.{ .set = .{ .key = .title, .value = "\"draft\"" } }});
    s.release();
    try testing.expectError(error.FileNotFound, f.dir());
    // The first turn creates the root and publishes the session.
    const t = try f.manager.openNew(.{ .workspace = "/w", .host = .app });
    defer t.release();
    _ = try t.append(&.{ .turn_started, piece, .turn_committed });
    var root = try f.dir();
    root.close(io);
}

test "a session's whole life through the API" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    const m = f.manager;
    const s = try m.openNew(.{ .workspace = "/w", .host = .app });
    _ = try s.append(&.{.{ .set = .{ .key = .title, .value = "\"first\"" } }});
    _ = try s.append(&.{ .turn_started, piece });
    const hash = try s.putBlob("big tool output");
    const refs = [_][]const u8{&hash};
    _ = try s.append(&.{ .{ .item = .{ .type = "tool_result", .data = "{\"tool\":1}", .blobs = &refs } }, .turn_committed });
    const id = try gpa.dupe(u8, s.id());
    defer gpa.free(id);

    // Listed from the first turn on, with its title.
    {
        var page = try m.list(gpa, .{ .workspace = "/w" }, null, 10);
        defer page.deinit();
        try testing.expectEqual(@as(usize, 1), page.items.len);
        try testing.expectEqualStrings("\"first\"", page.items[0].title.?);
    }
    s.release();
    {
        var page = try m.list(gpa, .all, null, 10);
        defer page.deinit();
        try testing.expectEqual(@as(u64, 1), page.items[0].turns);
    }

    // Scrollback: newest first.
    var back = try m.read(gpa, id, .end, .backward, 2);
    defer back.deinit();
    try testing.expectEqual(api.Kind.closed, back.entries[0].kind.?);
    try testing.expectEqual(api.Kind.turn_committed, back.entries[1].kind.?);
    const bytes = try m.getBlob(gpa, id, &hash);
    defer gpa.free(bytes);
    try testing.expectEqualStrings("big tool output", bytes);

    // Resume, state, and verify.
    const r = try m.openResume(.{ .target = .{ .id = id }, .workspace = "/w", .host = .ask });
    var state = try r.state(gpa);
    defer state.deinit(gpa);
    try testing.expectEqual(@as(u64, 1), state.committed);
    try testing.expect(state.clean_exit);
    r.release();
    const verified = try m.verify(id);
    try testing.expectEqual(@as(?u64, null), verified.damaged_at);
    try testing.expectEqual(@as(u64, 0), verified.bad_snapshots);
}

test "-c and --resume last stay distinct (D11)" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    const m = f.manager;
    const a = try m.openNew(.{ .workspace = "/w", .host = .app });
    _ = try a.append(&.{ .turn_started, .turn_committed });
    const a_id = try gpa.dupe(u8, a.id());
    defer gpa.free(a_id);
    a.release();
    const b = try m.openNew(.{ .workspace = "/w", .host = .ask });
    _ = try b.append(&.{ .turn_started, .turn_committed });
    const b_id = try gpa.dupe(u8, b.id());
    defer gpa.free(b_id);
    b.release();
    const other = try m.openNew(.{ .workspace = "/elsewhere", .host = .app });
    _ = try other.append(&.{ .turn_started, .turn_committed });
    other.release();

    // The app opens a; then ask and acp open b.
    (try m.openResume(.{ .target = .{ .id = a_id }, .workspace = "/w", .host = .app })).release();
    (try m.openResume(.{ .target = .{ .id = b_id }, .workspace = "/w", .host = .ask })).release();
    (try m.openResume(.{ .target = .{ .id = b_id }, .workspace = "/w", .host = .acp })).release();

    // b was updated last (closed by acp); the app last opened a. Resuming
    // is itself an open, so this check runs as `fx ask --resume last`.
    const last = try m.openResume(.{ .target = .last, .workspace = "/w", .host = .ask });
    try testing.expectEqualStrings(b_id, last.id());
    last.release();
    const c = try m.openResume(.{ .target = .{ .last_opened = .app }, .workspace = "/w", .host = .app });
    try testing.expectEqualStrings(a_id, c.id());
    c.release();
    try testing.expectError(error.NotFound, m.openResume(.{ .target = .{ .last_opened = .sdk }, .workspace = "/w", .host = .sdk }));
    try testing.expectError(error.NotFound, m.openResume(.{ .target = .last, .workspace = "/nowhere", .host = .app }));
}

test "fork, children, delete: listing and ids" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    const m = f.manager;
    const p = try m.openNew(.{ .workspace = "/w", .host = .app });
    _ = try p.append(&.{ .turn_started, piece, .turn_committed });
    const p_id = try gpa.dupe(u8, p.id());
    defer gpa.free(p_id);
    const c = try m.openNew(.{ .workspace = "/w", .host = .child, .role = .child, .parent = p_id });
    _ = try p.append(&.{ .turn_started, .{ .child_spawned = .{ .child = c.id(), .work_id = "w" } } });
    _ = try c.append(&.{ .turn_started, .turn_committed });
    _ = try p.append(&.{ .{ .child_finished = .{ .child = c.id(), .work_id = "w", .outcome = .ok } }, .turn_committed });
    const c_id = try gpa.dupe(u8, c.id());
    defer gpa.free(c_id);
    c.release();

    const fork = try m.openFork(.{ .source = p_id, .at = .{ .turn = 1 }, .workspace = "/w", .host = .app });
    const fork_id = try gpa.dupe(u8, fork.id());
    defer gpa.free(fork_id);
    fork.release();

    // Children are never listed.
    const ids = try listIds(m, .all);
    defer freeIds(ids);
    try testing.expectEqual(@as(usize, 2), ids.len);
    for (ids) |id| try testing.expect(!std.mem.eql(u8, id, c_id));

    try testing.expectError(error.Busy, m.delete(p_id));
    p.release();
    try m.delete(p_id);
    try testing.expectError(error.NotFound, m.read(gpa, c_id, .start, .forward, 1));
    const after = try listIds(m, .all);
    defer freeIds(after);
    try testing.expectEqual(@as(usize, 1), after.len);
    try testing.expectEqualStrings(fork_id, after[0]);
    // A deleted id never comes back, not even through an import.
    try testing.expectError(error.Exists, m.openImport(.{ .id = p_id, .workspace = "/w", .host = .app, .created_ms = 1 }));
    try testing.expectError(error.NotFound, m.delete(p_id));
}

test "import keeps the v1 id and the original times" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    const m = f.manager;
    const s = try m.openImport(.{ .id = "1786460757753-v1", .workspace = "/w", .host = .app, .created_ms = 1000 });
    try testing.expectError(error.InvalidArgument, s.append(&.{.turn_started}));
    _ = try s.appendAt(&.{ .turn_started, piece, .turn_committed }, 2000);
    _ = try s.appendAt(&.{ .turn_started, .{ .turn_interrupted = .crash } }, 3000);
    s.release();
    var page = try m.read(gpa, "1786460757753-v1", .start, .forward, 10);
    defer page.deinit();
    try testing.expectEqual(@as(u64, 1000), page.entries[0].ts_ms);
    try testing.expectEqual(@as(u64, 2000), page.entries[1].ts_ms);
    try testing.expectEqual(@as(u64, 3000), page.entries[5].ts_ms);
    try testing.expectError(error.Exists, m.openImport(.{ .id = "1786460757753-v1", .workspace = "/w", .host = .app, .created_ms = 1 }));
    // The list shows the original times, not the time of the conversion (D20).
    var listed = try m.list(gpa, .all, null, 10);
    defer listed.deinit();
    try testing.expectEqual(@as(u64, 1000), listed.items[0].created_ms);
    try testing.expectEqual(@as(u64, 3000), listed.items[0].updated_ms);
}

/// `ts` of the newest line in a session's log.
fn newestTs(m: *api.Manager, id: []const u8) !u64 {
    var page = try m.read(gpa, id, .end, .backward, 1);
    defer page.deinit();
    return page.entries[0].ts_ms;
}

test "state carries the created and updated times, and the list agrees (D20)" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    const m = f.manager;
    const s = try m.openNew(.{ .workspace = "/w", .host = .app });
    {
        var st = try s.state(gpa);
        defer st.deinit(gpa);
        try testing.expectEqual(@as(u64, 0), st.created_ms);
        try testing.expectEqual(@as(u64, 0), st.updated_ms);
    }
    _ = try s.append(&.{ .turn_started, piece, .turn_committed });
    const id = try gpa.dupe(u8, s.id());
    defer gpa.free(id);
    var first = try s.state(gpa);
    defer first.deinit(gpa);
    try testing.expect(first.created_ms > 0);
    try testing.expectEqual(try newestTs(m, id), first.updated_ms);
    s.release();

    // Resume reads both back from the log; the list shows the same times.
    const r = try m.openResume(.{ .target = .{ .id = id }, .workspace = "/w", .host = .app });
    var st = try r.state(gpa);
    defer st.deinit(gpa);
    try testing.expectEqual(first.created_ms, st.created_ms);
    try testing.expectEqual(try newestTs(m, id), st.updated_ms);
    r.release();
    var listed = try m.list(gpa, .all, null, 10);
    defer listed.deinit();
    try testing.expectEqual(first.created_ms, listed.items[0].created_ms);
    try testing.expectEqual(try newestTs(m, id), listed.items[0].updated_ms);

    // A fork is created now; its copied lines keep older times.
    const fk = try m.openFork(.{ .source = id, .at = .{ .turn = 1 }, .workspace = "/w", .host = .app });
    defer fk.release();
    var forked = try fk.state(gpa);
    defer forked.deinit(gpa);
    try testing.expect(forked.created_ms >= st.updated_ms);
    try testing.expectEqual(forked.created_ms, forked.updated_ms);
}

test "hosts end turns as cancel or failed; closed and crash stay the manager's (D21)" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    const s = try f.manager.openNew(.{ .workspace = "/w", .host = .app });
    defer s.release();
    _ = try s.append(&.{.turn_started});
    try testing.expectError(error.InvalidArgument, s.append(&.{.{ .turn_interrupted = .closed }}));
    try testing.expectError(error.InvalidArgument, s.append(&.{.{ .turn_interrupted = .crash }}));
    _ = try s.append(&.{.{ .turn_interrupted = .failed }});
    var st = try s.state(gpa);
    defer st.deinit(gpa);
    try testing.expectEqual(api.Reason.failed, st.last_interrupted.?.reason);
}

test "child data is small, exact, and never lost; outcomes include cancelled (D22)" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    const m = f.manager;
    const p = try m.openNew(.{ .workspace = "/w", .host = .app });
    defer p.release();
    _ = try p.append(&.{.turn_started});
    const c = try m.openNew(.{ .workspace = "/w", .host = .child, .role = .child, .parent = p.id() });
    defer c.release();

    const big = try gpa.alloc(u8, api.max_child_data_bytes + 1);
    defer gpa.free(big);
    @memset(big, 'a');
    big[0] = '"';
    big[big.len - 1] = '"';
    try testing.expectError(error.TooLarge, p.append(&.{.{ .child_spawned = .{ .child = c.id(), .work_id = "w1", .data = big } }}));
    try testing.expectError(error.InvalidArgument, p.append(&.{.{ .child_spawned = .{ .child = c.id(), .work_id = "w1", .data = " {}" } }}));
    const data = "{\"name\": \"reviewer\", \"agent\": \"general\"}";
    _ = try p.append(&.{.{ .child_spawned = .{ .child = c.id(), .work_id = "w1", .data = data } }});
    try testing.expectError(error.InvalidArgument, p.append(&.{.{ .child_finished = .{ .child = c.id(), .work_id = "w1", .outcome = .lost } }}));
    try testing.expectError(error.InvalidTransition, p.append(&.{.{ .child_spawned = .{ .child = c.id(), .work_id = "w2" } }}));
    _ = try p.append(&.{.{ .child_finished = .{ .child = c.id(), .work_id = "w1", .outcome = .cancelled } }});

    var st = try p.state(gpa);
    defer st.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), st.children.items.len);
    const child: api.Child = st.children.items[0];
    try testing.expectEqualStrings(c.id(), child.id);
    try testing.expectEqualStrings(data, child.spawn_data.?);
    try testing.expectEqual(@as(?api.Outcome, .cancelled), child.outcome);
    try testing.expect(!child.open);
}

test "raw values with surrounding whitespace are refused, so readers get the exact bytes" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    const s = try f.manager.openNew(.{ .workspace = "/w", .host = .app });
    defer s.release();
    try testing.expectError(error.InvalidArgument, s.append(&.{.{ .set = .{ .key = .prefs, .value = "{} " } }}));
    try testing.expectError(error.InvalidArgument, s.append(&.{.{ .set = .{ .key = .title, .value = " \"t\"" } }}));
    _ = try s.append(&.{ .turn_started, .{ .set = .{ .key = .usage, .value = "null" } } });
    try testing.expectError(error.InvalidArgument, s.append(&.{.{ .item = .{ .type = "user", .data = "\n{}" } }}));
}

test "a state too large for one snapshot is skipped and reported; the append and resume stand" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    const m = f.manager;
    // Two settings of 3 MiB each fit one line apiece, but not one snapshot.
    const big = try gpa.alloc(u8, 3 << 20);
    defer gpa.free(big);
    @memset(big, 'a');
    big[0] = '"';
    big[big.len - 1] = '"';
    const s = try m.openNew(.{ .workspace = "/w", .host = .app });
    _ = try s.append(&.{ .turn_started, .{ .set = .{ .key = .prefs, .value = big } } });
    _ = try s.append(&.{.{ .set = .{ .key = .usage, .value = big } }});
    _ = try s.append(&.{ .{ .compacted = "{}" }, .turn_committed });
    try testing.expect(f.recorder.count(.snapshot_skipped) >= 1);
    const id = try gpa.dupe(u8, s.id());
    defer gpa.free(id);
    s.release();

    const r = try m.openResume(.{ .target = .{ .id = id }, .workspace = "/w", .host = .app });
    defer r.release();
    var st = try r.state(gpa);
    defer st.deinit(gpa);
    try testing.expectEqual(@as(u64, 1), st.committed);
    try testing.expectEqualStrings(big, st.prefs.?);
    try testing.expectEqualStrings(big, st.usage.?);
    const verified = try m.verify(id);
    try testing.expectEqual(@as(?u64, null), verified.damaged_at);
    try testing.expectEqual(@as(u64, 0), verified.bad_snapshots);
}

test "list self-heals a missing folder and rebuild restores the index, each reported once" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    const m = f.manager;
    var ids: [3][]u8 = undefined;
    for (&ids) |*id| {
        const s = try m.openNew(.{ .workspace = "/w", .host = .app });
        _ = try s.append(&.{ .turn_started, .turn_committed });
        id.* = try gpa.dupe(u8, s.id());
        s.release();
    }
    defer for (ids) |id| gpa.free(id);
    var root = try f.dir();
    defer root.close(io);

    // A folder vanishes behind the manager's back.
    try root.deleteTree(io, ids[0]);
    const healed = try listIds(m, .all);
    defer freeIds(healed);
    try testing.expectEqual(@as(usize, 2), healed.len);
    try testing.expectEqual(@as(usize, 1), f.recorder.count(.index_healed));
    const again = try listIds(m, .all);
    defer freeIds(again);
    try testing.expectEqual(@as(usize, 1), f.recorder.count(.index_healed));

    // The index is lost, and a leftover sits in .trash.
    try root.deleteFile(io, "index.jsonl");
    try root.createDirPath(io, ".trash/leftover");
    const empty = try listIds(m, .all);
    defer freeIds(empty);
    try testing.expectEqual(@as(usize, 0), empty.len);
    const rebuilt = try m.rebuild();
    try testing.expectEqual(@as(u64, 2), rebuilt.sessions);
    try testing.expectEqual(@as(u64, 1), rebuilt.swept);
    try testing.expectEqual(@as(usize, 1), f.recorder.count(.rebuild_swept));
    const restored = try listIds(m, .all);
    defer freeIds(restored);
    try testing.expectEqual(@as(usize, 2), restored.len);
}

test "a torn index tail is cut before the next record, and a big index compacts" {
    var f: Fixture = undefined;
    try f.initWith(4096);
    defer f.deinit();
    const m = f.manager;
    const s = try m.openNew(.{ .workspace = "/w", .host = .app });
    _ = try s.append(&.{ .turn_started, .turn_committed });
    const id = try gpa.dupe(u8, s.id());
    defer gpa.free(id);
    s.release();
    var root = try f.dir();
    defer root.close(io);
    {
        var index = try root.openFile(io, "index.jsonl", .{ .mode = .read_write });
        defer index.close(io);
        const len = try index.length(io);
        try index.writePositionalAll(io, "{\"op\":\"put\",\"id\":\"tor", len);
    }
    for (0..40) |_| (try m.openResume(.{ .target = .{ .id = id }, .workspace = "/w", .host = .app })).release();
    const ids = try listIds(m, .all);
    defer freeIds(ids);
    try testing.expectEqual(@as(usize, 1), ids.len);
    try testing.expectEqual(@as(usize, 0), f.recorder.count(.index_healed));
    const st = try root.statFile(io, "index.jsonl", .{});
    try testing.expect(st.size < 4096);
}

test "inputs are checked at the boundary" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    const m = f.manager;
    try testing.expectError(error.InvalidArgument, m.openNew(.{ .workspace = "", .host = .app }));
    try testing.expectError(error.InvalidArgument, m.openNew(.{ .workspace = "/w", .host = .child, .role = .child }));
    try testing.expectError(error.InvalidArgument, m.openResume(.{ .target = .{ .id = "../etc" }, .workspace = "/w", .host = .app }));
    const s = try m.openNew(.{ .workspace = "/w", .host = .app });
    defer s.release();
    _ = try s.append(&.{.turn_started});
    try testing.expectError(error.InvalidArgument, s.append(&.{.{ .item = .{ .type = "assistant", .data = "{not json" } }}));
    for ([_][]const u8{ "", "Steering", "tool-call", "x" ** 33 }) |bad_type| {
        try testing.expectError(error.InvalidArgument, s.append(&.{.{ .item = .{ .type = bad_type, .data = "{}" } }}));
    }
    try testing.expectError(error.InvalidArgument, s.append(&.{.{ .set = .{ .key = .title, .value = "42" } }}));
    try testing.expectError(error.InvalidArgument, s.append(&.{.{ .turn_interrupted = .crash }}));
    try testing.expectError(error.InvalidArgument, s.append(&.{}));
    try testing.expectError(error.InvalidArgument, m.getBlob(gpa, s.id(), "nothex"));
    // Nothing was written by any refused call.
    var page = try m.read(gpa, s.id(), .start, .forward, 10);
    defer page.deinit();
    try testing.expectEqual(@as(usize, 2), page.entries.len);
}

test "item types: stored on the line, returned by read, never interpreted" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    const m = f.manager;
    const s = try m.openNew(.{ .workspace = "/w", .host = .app });
    const id = try gpa.dupe(u8, s.id());
    defer gpa.free(id);
    const types = [_][]const u8{ "user", "assistant", "tool_call", "tool_result", "steering", "a_name_fx_adds_later" };
    _ = try s.append(&.{.turn_started});
    for (types) |t| _ = try s.append(&.{.{ .item = .{ .type = t, .data = "{}" } }});
    _ = try s.append(&.{.turn_committed});
    s.release();

    // `read` gives every type back, in order.
    var page = try m.read(gpa, id, .start, .forward, 100);
    defer page.deinit();
    var seen: usize = 0;
    for (page.entries) |entry| {
        const body = entry.body orelse continue;
        if (body != .item) continue;
        try testing.expectEqualStrings(types[seen], body.item.type);
        seen += 1;
    }
    try testing.expectEqual(types.len, seen);

    // On disk the type is a plain field, so `grep` and `jq` find a steer.
    var dir = try f.dir();
    defer dir.close(io);
    var path: [300]u8 = undefined;
    const bytes = try dir.readFileAlloc(io, try std.fmt.bufPrint(&path, "{s}/log.jsonl", .{id}), gpa, .limited(1 << 20));
    defer gpa.free(bytes);
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, bytes, "\"kind\":\"item\",\"turn\":1,\"type\":\"steering\""));
}

test "the conversation language is listed, follows changes, survives rebuild and resume" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    const m = f.manager;
    const s = try m.openNew(.{ .workspace = "/w", .host = .app });
    const id = try gpa.dupe(u8, s.id());
    defer gpa.free(id);
    // Held before the first turn, like any setting (D2).
    _ = try s.append(&.{.{ .set = .{ .key = .language, .value = "\"es\"" } }});
    _ = try s.append(&.{ .turn_started, piece, .turn_committed });
    try expectListedLanguage(m, "\"es\"");

    _ = try s.append(&.{.{ .set = .{ .key = .language, .value = "\"und-Latn\"" } }});
    try expectListedLanguage(m, "\"und-Latn\"");

    // Refused: not a string, empty, longer than 24 bytes. Nothing changes.
    for ([_][]const u8{ "42", "\"\"", "\"" ++ "x" ** 25 ++ "\"" }) |bad| {
        try testing.expectError(error.InvalidArgument, s.append(&.{.{ .set = .{ .key = .language, .value = bad } }}));
    }
    s.release();
    try expectListedLanguage(m, "\"und-Latn\"");

    // The index is derived: a rebuild finds the language in the log.
    _ = try m.rebuild();
    try expectListedLanguage(m, "\"und-Latn\"");

    const r = try m.openResume(.{ .target = .{ .id = id }, .workspace = "/w", .host = .app });
    defer r.release();
    var st = try r.state(gpa);
    defer st.deinit(gpa);
    try testing.expectEqualStrings("\"und-Latn\"", st.language.?);
}

fn expectListedLanguage(m: *api.Manager, want: []const u8) !void {
    var page = try m.list(gpa, .all, null, 10);
    defer page.deinit();
    try testing.expectEqual(@as(usize, 1), page.items.len);
    try testing.expectEqualStrings(want, page.items[0].language.?);
}

test "Session.read pages the open session exactly like read by id" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    const m = f.manager;
    const s = try m.openNew(.{ .workspace = "/w", .host = .app });
    defer s.release();
    // Nothing on disk before the first turn: an empty page, not an error.
    var empty = try s.read(gpa, .end, .backward, 10);
    try testing.expectEqual(@as(usize, 0), empty.entries.len);
    empty.deinit();
    try testing.expectError(error.InvalidArgument, s.read(gpa, .end, .backward, 0));

    for (0..12) |_| _ = try s.append(&.{ .turn_started, piece, piece, .turn_committed });
    for ([_]api.Direction{ .backward, .forward }) |direction| {
        var from_open: api.From = if (direction == .backward) .end else .start;
        var from_id = from_open;
        var pages: usize = 0;
        while (true) : (pages += 1) {
            var a = try s.read(gpa, from_open, direction, 7);
            defer a.deinit();
            var b = try m.read(gpa, s.id(), from_id, direction, 7);
            defer b.deinit();
            try testing.expectEqual(b.entries.len, a.entries.len);
            for (a.entries, b.entries) |x, y| {
                try testing.expectEqual(y.seq, x.seq);
                try testing.expectEqual(y.offset, x.offset);
            }
            try testing.expectEqual(b.next == null, a.next == null);
            from_open = .{ .at = a.next orelse break };
            from_id = .{ .at = b.next.? };
        }
        try testing.expect(pages > 1);
    }
}

test "Session.read while another thread appends, and after close" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    const m = f.manager;
    const s = try m.openNew(.{ .workspace = "/w", .host = .app });
    _ = try s.append(&.{ .turn_started, piece, .turn_committed });
    const Writer = struct {
        fn run(session: api.Session) void {
            for (0..200) |_| _ = session.append(&.{ .turn_started, piece, .turn_committed }) catch return;
        }
    };
    const writer = try std.Thread.spawn(.{}, Writer.run, .{s});
    for (0..200) |_| {
        var page = try s.read(gpa, .end, .backward, 20);
        defer page.deinit();
        // Newest first, one line after another, never a torn line.
        for (page.entries[1..], page.entries[0 .. page.entries.len - 1]) |older, newer| {
            try testing.expectEqual(newer.seq - 1, older.seq);
        }
    }
    writer.join();
    try s.close();
    try testing.expectError(error.SessionClosed, s.read(gpa, .end, .backward, 10));
    s.release();
}

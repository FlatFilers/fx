//! L2 behavior, through the Session only: data-flow.md scenarios 1 to 3 and
//! 5 to 11. Runs with and without hooks; fault-only cases skip without.

const std = @import("std");
const schema = @import("schema.zig");
const storage = @import("storage.zig");
const log_mod = @import("log.zig");
const fold = @import("fold.zig");
const diag = @import("diag.zig");
const session_mod = @import("session.zig");

const testing = std.testing;
const gpa = testing.allocator;
const io = testing.io;
const hooks = storage.hooks;
const Session = session_mod.Session;
const Fault = storage.Fault;

pub const TestEnv = struct {
    tmp: testing.TmpDir,
    fault: if (hooks) Fault else void,
    recorder: diag.Recorder,
    env: session_mod.Env,
    /// Set by `crash`: a crash closes written files unsynced by definition.
    crashed: bool = false,

    /// In place: `env` points into `t`, so `t` must not move afterwards.
    pub fn init(t: *TestEnv, options: session_mod.Options) void {
        t.crashed = false;
        t.tmp = testing.tmpDir(.{ .iterate = true });
        t.recorder = .{ .gpa = gpa, .io = io };
        if (hooks) t.fault = .init(gpa, io, 1);
        t.env = .{
            .gpa = gpa,
            .s = if (hooks) .{ .io = io, .fault = &t.fault } else .{ .io = io },
            .root = .{ .handle = t.tmp.dir },
            .options = options,
            .diagnostics = t.recorder.sink(),
        };
    }

    pub fn deinit(t: *TestEnv) void {
        if (hooks) {
            // Every written file was closed only after a sync.
            if (!t.crashed) testing.expectEqual(@as(usize, 0), t.fault.closed_unsynced) catch @panic("a written file was closed unsynced");
            t.fault.deinit();
        }
        t.recorder.deinit();
        t.tmp.cleanup();
    }

    pub fn kinds(t: *TestEnv, id: []const u8) ![]schema.Kind {
        var page = try session_mod.readPage(&t.env, gpa, id, .start, .forward, 1 << 20);
        defer page.deinit();
        const out = try gpa.alloc(schema.Kind, page.entries.len);
        for (page.entries, 0..) |entry, i| {
            try testing.expectEqual(@as(u64, i + 1), entry.seq);
            out[i] = entry.kind.?;
        }
        return out;
    }
};

/// fx dies: nothing more is written, and the kernel drops the flock.
fn crash(t: *TestEnv, s: *Session) void {
    t.crashed = true;
    s.abandon();
}

fn expectKinds(t: *TestEnv, id: []const u8, expected: []const schema.Kind) !void {
    const got = try t.kinds(id);
    defer gpa.free(got);
    try testing.expectEqualSlices(schema.Kind, expected, got);
}

fn newRoot(t: *TestEnv) !*Session {
    return session_mod.openNew(&t.env, .{ .workspace = "/w", .host = .app });
}

fn resumeRoot(t: *TestEnv, id: []const u8) !*Session {
    return session_mod.openResume(&t.env, .{ .id = id, .workspace = "/w", .host = .app });
}

fn closeAndDestroy(s: *Session) !void {
    try s.close();
    s.destroy();
}

/// Delete as L3 composes it, minus the index: children first, then trash
/// and purge.
pub fn deleteWithoutIndex(env: *const session_mod.Env, id: []const u8) session_mod.DeleteError!void {
    const children = try session_mod.childrenOf(env, id);
    defer {
        for (children) |c| env.gpa.free(c);
        env.gpa.free(children);
    }
    for (children) |child| deleteWithoutIndex(env, child) catch |err| switch (err) {
        error.NotFound => {},
        else => return err,
    };
    try session_mod.trashSession(env, id);
    try session_mod.purgeTrashed(env, id);
}

const item: fold.Event = .{ .item = .{ .type = "assistant", .data = "{\"text\":\"piece\"}" } };

test "a session that never starts a turn leaves nothing on disk" {
    var t: TestEnv = undefined;
    t.init(.{});
    defer t.deinit();
    const s = try newRoot(&t);
    _ = try s.append(&.{.{ .set = .{ .key = .title, .value = "\"draft\"" } }});
    try testing.expectError(error.InvalidTransition, s.append(&.{item}));
    try closeAndDestroy(s);
    var listing = t.env.s.list(t.env.root);
    try testing.expectEqual(@as(?storage.Entry, null), try listing.next());
    try testing.expectEqual(@as(usize, 1), t.recorder.count(.held_lines_dropped));
}

test "the first turn publishes line 1, the held settings and the batch" {
    var t: TestEnv = undefined;
    t.init(.{});
    defer t.deinit();
    const s = try newRoot(&t);
    _ = try s.append(&.{.{ .set = .{ .key = .title, .value = "\"hello\"" } }});
    try testing.expectEqual(@as(u64, 4), try s.append(&.{ .turn_started, item }));
    const dir = try t.env.s.openDir(t.env.root, s.id());
    defer t.env.s.closeDir(dir);
    try testing.expectEqual(storage.Kind.file, (try t.env.s.stat(dir, "lock")).kind);
    try testing.expectEqual(storage.Kind.directory, (try t.env.s.stat(dir, "blobs")).kind);
    // The staging folder is empty again.
    const tmp = try t.env.s.openDir(t.env.root, ".tmp");
    defer t.env.s.closeDir(tmp);
    var listing = t.env.s.list(tmp);
    try testing.expectEqual(@as(?storage.Entry, null), try listing.next());

    try expectKinds(&t, s.id(), &.{ .session_created, .set, .turn_started, .item });
    _ = try s.append(&.{.turn_committed});
    var state = try s.stateCopy(gpa);
    defer state.deinit(gpa);
    try testing.expectEqualStrings("\"hello\"", state.title.?);
    try testing.expectEqual(@as(u64, 1), state.committed);
    const id = try gpa.dupe(u8, s.id());
    defer gpa.free(id);
    try closeAndDestroy(s);
    try expectKinds(&t, id, &.{ .session_created, .set, .turn_started, .item, .turn_committed, .closed });
}

test "a refused batch writes nothing" {
    var t: TestEnv = undefined;
    t.init(.{});
    defer t.deinit();
    const s = try newRoot(&t);
    defer closeAndDestroy(s) catch {};
    _ = try s.append(&.{.turn_started});
    try testing.expectError(error.InvalidTransition, s.append(&.{ item, .turn_started }));
    try testing.expectError(error.InvalidTransition, s.append(&.{ .turn_committed, .turn_committed }));
    try expectKinds(&t, s.id(), &.{ .session_created, .turn_started });
}

test "a crash leaves an open turn, and resume interrupts it" {
    var t: TestEnv = undefined;
    t.init(.{});
    defer t.deinit();
    const s = try newRoot(&t);
    _ = try s.append(&.{ .turn_started, item });
    const id = try gpa.dupe(u8, s.id());
    defer gpa.free(id);
    crash(&t, s);

    const r = try resumeRoot(&t, id);
    var state = try r.stateCopy(gpa);
    defer state.deinit(gpa);
    try testing.expectEqual(@as(?u64, null), state.open_turn);
    try testing.expectEqual(schema.Reason.crash, state.last_interrupted.?.reason);
    try testing.expect(!state.clean_exit);
    // The next turn is turn 2.
    _ = try r.append(&.{ .turn_started, item, .turn_committed });
    var after = try r.stateCopy(gpa);
    defer after.deinit(gpa);
    try testing.expectEqual(@as(u64, 2), after.last_turn);
    try closeAndDestroy(r);
    try expectKinds(&t, id, &.{ .session_created, .turn_started, .item, .turn_interrupted, .turn_started, .item, .turn_committed, .closed });
}

test "a second writer gets Busy while the first holds the session" {
    var t: TestEnv = undefined;
    t.init(.{ .lock_wait_ms = 30 });
    defer t.deinit();
    const s = try newRoot(&t);
    defer closeAndDestroy(s) catch {};
    _ = try s.append(&.{ .turn_started, .turn_committed });
    try testing.expectError(error.Busy, resumeRoot(&t, s.id()));
    try testing.expectError(error.NotFound, resumeRoot(&t, "missing"));
}

test "close interrupts an open turn, and later calls get SessionClosed" {
    var t: TestEnv = undefined;
    t.init(.{});
    defer t.deinit();
    const s = try newRoot(&t);
    _ = try s.append(&.{ .turn_started, item });
    try s.close();
    try testing.expectError(error.SessionClosed, s.append(&.{item}));
    try s.close();
    try expectKinds(&t, s.id(), &.{ .session_created, .turn_started, .item, .turn_interrupted, .closed });
    const r = try resumeRoot(&t, s.id());
    var state = try r.stateCopy(gpa);
    defer state.deinit(gpa);
    try testing.expectEqual(schema.Reason.closed, state.last_interrupted.?.reason);
    try testing.expect(state.clean_exit);
    try closeAndDestroy(r);
    s.destroy();
}

test "resuming from another workspace records it once" {
    var t: TestEnv = undefined;
    t.init(.{});
    defer t.deinit();
    const s = try newRoot(&t);
    _ = try s.append(&.{ .turn_started, .turn_committed });
    const id = try gpa.dupe(u8, s.id());
    defer gpa.free(id);
    try closeAndDestroy(s);
    const moved = try session_mod.openResume(&t.env, .{ .id = id, .workspace = "/other \"place\"", .host = .app });
    try closeAndDestroy(moved);
    const again = try session_mod.openResume(&t.env, .{ .id = id, .workspace = "/other \"place\"", .host = .ask });
    var state = try again.stateCopy(gpa);
    defer state.deinit(gpa);
    try testing.expectEqualStrings("\"/other \\\"place\\\"\"", state.workspace.?);
    try closeAndDestroy(again);
    try expectKinds(&t, id, &.{ .session_created, .turn_started, .turn_committed, .closed, .set, .closed, .closed });
}

test "a child session resumes only with its parent" {
    var t: TestEnv = undefined;
    t.init(.{});
    defer t.deinit();
    const c = try session_mod.openNew(&t.env, .{ .workspace = "/w", .host = .child, .role = .child, .parent = "p1" });
    _ = try c.append(&.{ .turn_started, .turn_committed });
    const id = try gpa.dupe(u8, c.id());
    defer gpa.free(id);
    try closeAndDestroy(c);
    try testing.expectError(error.ChildSession, resumeRoot(&t, id));
    try testing.expectError(error.ChildSession, session_mod.openResume(&t.env, .{ .id = id, .workspace = "/w", .host = .child, .parent = "p2" }));
    const ok = try session_mod.openResume(&t.env, .{ .id = id, .workspace = "/w", .host = .child, .parent = "p1" });
    try closeAndDestroy(ok);
}

/// A full fold of every line, ignoring snapshots: the reference state.
fn fullFold(t: *TestEnv, id: []const u8) !fold.State {
    var page = try session_mod.readPage(&t.env, gpa, id, .start, .forward, 1 << 20);
    defer page.deinit();
    var state: fold.State = .{};
    errdefer state.deinit(gpa);
    for (page.entries) |entry| {
        const body = entry.body.?;
        if (body == .snapshot) {
            state.last_seq = entry.seq;
            state.clean_exit = false;
            continue;
        }
        try fold.apply(gpa, &state, 0, .{ .seq = entry.seq, .offset = entry.offset, .body = body });
    }
    return state;
}

test "resume from snapshots equals a full fold" {
    var t: TestEnv = undefined;
    t.init(.{ .snapshot_every_bytes = 1024 });
    defer t.deinit();
    const s = try newRoot(&t);
    var prng = std.Random.DefaultPrng.init(3);
    const random = prng.random();
    var value_buffer: [32]u8 = undefined;
    for (0..60) |turn| {
        _ = try s.append(&.{ .turn_started, item, item });
        const value = try std.fmt.bufPrint(&value_buffer, "{{\"model\":\"m{d}\"}}", .{turn});
        _ = try s.append(&.{.{ .set = .{ .key = .prefs, .value = value } }});
        if (random.boolean()) {
            _ = try s.append(&.{.turn_committed});
        } else {
            _ = try s.append(&.{.{ .turn_interrupted = .cancel }});
        }
        if (turn % 17 == 0) _ = try s.append(&.{.{ .compacted = "{\"summary\":\"...\"}" }});
    }
    _ = try s.append(&.{ .turn_started, item });
    const id = try gpa.dupe(u8, s.id());
    defer gpa.free(id);
    crash(&t, s);

    const r = try resumeRoot(&t, id);
    var resumed = try r.stateCopy(gpa);
    defer resumed.deinit(gpa);
    try closeAndDestroy(r);
    // Compare against a full fold of the log as it stood before the close.
    var reference = try fullFold(&t, id);
    defer reference.deinit(gpa);
    // The close added turn_interrupted? No: resume already interrupted the
    // open turn, then close appended `closed`. Undo only the close line.
    try testing.expect(reference.clean_exit);
    reference.clean_exit = false;
    reference.last_seq -= 1;
    try testing.expect(resumed.eql(&reference));
    try testing.expectEqualStrings("{\"model\":\"m59\"}", resumed.prefs.?);

    const all = try t.kinds(id);
    defer gpa.free(all);
    var snapshots: usize = 0;
    for (all) |k| {
        if (k == .snapshot) snapshots += 1;
    }
    try testing.expect(snapshots >= 5);
}

test "a compaction is followed by a snapshot" {
    var t: TestEnv = undefined;
    t.init(.{});
    defer t.deinit();
    const s = try newRoot(&t);
    defer closeAndDestroy(s) catch {};
    _ = try s.append(&.{ .turn_started, .{ .compacted = "{}" } });
    try expectKinds(&t, s.id(), &.{ .session_created, .turn_started, .compacted, .snapshot });
    var state = try s.stateCopy(gpa);
    defer state.deinit(gpa);
    try testing.expectEqual(@as(?u64, 3), state.last_compaction_seq);
}

const Worker = struct {
    session: *Session,
    count: usize,
    durable: bool,
    failed: bool = false,

    fn run(w: *Worker) void {
        for (0..w.count) |_| {
            if (w.durable) {
                const seq = w.session.append(&.{.{ .set = .{ .key = .permissions, .value = "{\"allow\":[]}" } }}) catch {
                    w.failed = true;
                    return;
                };
                // A durable-class call returns only after its own sync.
                w.session.sync_mutex.lockUncancelable(io);
                const synced = w.session.synced_seq;
                w.session.sync_mutex.unlock(io);
                if (synced < seq) w.failed = true;
            } else {
                _ = w.session.append(&.{item}) catch {
                    w.failed = true;
                    return;
                };
            }
        }
    }
};

test "set usage returns after its own sync; set prefs does not sync" {
    var t: TestEnv = undefined;
    t.init(.{});
    defer t.deinit();
    const s = try newRoot(&t);
    _ = try s.append(&.{.turn_started});
    const usage_seq = try s.append(&.{.{ .set = .{ .key = .usage, .value = "{\"n\":1}" } }});
    const prefs_seq = try s.append(&.{.{ .set = .{ .key = .prefs, .value = "{}" } }});
    s.sync_mutex.lockUncancelable(io);
    const synced = s.synced_seq;
    s.sync_mutex.unlock(io);
    try testing.expect(synced >= usage_seq);
    try testing.expect(synced < prefs_seq);
    try closeAndDestroy(s);
}

test "threads appending at once get contiguous seqs in file order" {
    var t: TestEnv = undefined;
    t.init(.{});
    defer t.deinit();
    const s = try newRoot(&t);
    _ = try s.append(&.{.turn_started});
    var workers: [8]Worker = undefined;
    var threads: [8]std.Thread = undefined;
    for (&workers, &threads, 0..) |*w, *thread, i| {
        w.* = .{ .session = s, .count = 50, .durable = i % 2 == 0 };
        thread.* = try std.Thread.spawn(.{}, Worker.run, .{w});
    }
    for (threads) |thread| thread.join();
    for (workers) |w| try testing.expect(!w.failed);
    _ = try s.append(&.{.turn_committed});
    const id = try gpa.dupe(u8, s.id());
    defer gpa.free(id);
    try closeAndDestroy(s);
    // readPage checks crc and that seq is contiguous from 1.
    const all = try t.kinds(id);
    defer gpa.free(all);
    var items: usize = 0;
    var sets: usize = 0;
    for (all) |k| switch (k) {
        .item => items += 1,
        .set => sets += 1,
        else => {},
    };
    try testing.expectEqual(@as(usize, 200), items);
    try testing.expectEqual(@as(usize, 200), sets);
}

test "paging backward and forward returns every line once, stable across appends" {
    var t: TestEnv = undefined;
    t.init(.{ .snapshot_every_bytes = 1 << 30 });
    defer t.deinit();
    const s = try newRoot(&t);
    defer closeAndDestroy(s) catch {};
    _ = try s.append(&.{.turn_started});
    for (0..20) |_| _ = try s.append(&.{item});
    // 22 lines. Page backward by 5 from the end.
    var seen: std.ArrayList(u64) = .empty;
    defer seen.deinit(gpa);
    var from: session_mod.From = .end;
    var first = true;
    while (true) {
        var page = try session_mod.readPage(&t.env, gpa, s.id(), from, .backward, 5);
        defer page.deinit();
        for (page.entries) |e| try seen.append(gpa, e.seq);
        if (first) {
            // Appends after the first page do not disturb older pages.
            _ = try s.append(&.{ item, item });
            first = false;
        }
        from = .{ .at = page.next orelse break };
    }
    try testing.expectEqual(@as(usize, 22), seen.items.len);
    for (seen.items, 0..) |seq, i| try testing.expectEqual(@as(u64, 22 - i), seq);

    var forward: std.ArrayList(u64) = .empty;
    defer forward.deinit(gpa);
    var at: session_mod.From = .start;
    while (true) {
        var page = try session_mod.readPage(&t.env, gpa, s.id(), at, .forward, 7);
        defer page.deinit();
        for (page.entries) |e| try forward.append(gpa, e.seq);
        at = .{ .at = page.next orelse break };
    }
    try testing.expectEqual(@as(usize, 24), forward.items.len);
    for (forward.items, 0..) |seq, i| try testing.expectEqual(@as(u64, i + 1), seq);
}

test "a torn tail is cut once and a damaged middle refuses to resume, each reported once" {
    if (!hooks) return error.SkipZigTest;
    var t: TestEnv = undefined;
    t.init(.{});
    defer t.deinit();
    const s = try newRoot(&t);
    _ = try s.append(&.{ .turn_started, item, .turn_committed });
    const id = try gpa.dupe(u8, s.id());
    defer gpa.free(id);
    // Die in the middle of the next line.
    _ = try s.append(&.{.turn_started});
    t.fault.next_write = .{ .keep = 20, .then = .die };
    try testing.expectError(error.Io, s.append(&.{item}));
    crash(&t, s);
    t.fault.restart();
    const r = try resumeRoot(&t, id);
    try closeAndDestroy(r);
    try testing.expectEqual(@as(usize, 1), t.recorder.count(.torn_tail_cut));

    // Now damage line 2 of the log.
    const dir = try t.env.s.openDir(t.env.root, id);
    defer t.env.s.closeDir(dir);
    var path_buffer: [300]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buffer, "{s}/log.jsonl", .{id});
    const bytes = try t.tmp.dir.readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(bytes);
    const line2 = std.mem.findScalar(u8, bytes, '\n').? + 1;
    try @import("storage_fault.zig").flipBit(io, dir, "log.jsonl", line2 + 4, 1);
    try testing.expectError(error.Corrupt, resumeRoot(&t, id));
    try testing.expectEqual(@as(usize, 1), t.recorder.count(.opened_read_only));
    // Reading still works up to the damage.
    var page = try session_mod.readPage(&t.env, gpa, id, .start, .forward, 100);
    defer page.deinit();
    try testing.expect(page.damaged);
    try testing.expectEqual(@as(usize, 1), page.entries.len);
}

test "a failed write or sync reports its OS cause, and every later call reports it too (D40)" {
    if (!hooks) return error.SkipZigTest;
    var t: TestEnv = undefined;
    t.init(.{});
    defer t.deinit();

    // A full disk on a log write (D29).
    const s = try newRoot(&t);
    _ = try s.append(&.{ .turn_started, item, .turn_committed });
    const id = try gpa.dupe(u8, s.id());
    defer gpa.free(id);
    t.fault.fail_error = error.NoSpace;
    t.fault.next_write = .{ .keep = 0, .then = .fail };
    try testing.expectError(error.NoSpaceLeft, s.append(&.{.turn_started}));
    // Durability is unknown after a failed write, so nothing more is
    // written; later calls still name the cause (D40).
    try testing.expectError(error.NoSpaceLeft, s.append(&.{.turn_started}));
    try testing.expectError(error.NoSpaceLeft, s.putBlob("after the full disk"));
    try closeAndDestroy(s);
    try expectKinds(&t, id, &.{ .session_created, .turn_started, .item, .turn_committed });

    // A read-only file system on the sync that ends a turn.
    const r = try resumeRoot(&t, id);
    _ = try r.append(&.{ .turn_started, item });
    t.fault.fail_error = error.ReadOnly;
    t.fault.fail_next_sync = true;
    try testing.expectError(error.ReadOnlyFileSystem, r.append(&.{.turn_committed}));
    try testing.expectError(error.ReadOnlyFileSystem, r.append(&.{.turn_started}));
    try closeAndDestroy(r);
    // The close does not sync again: the failed sync's bytes stay unknown.
    try testing.expectEqual(@as(usize, 1), t.fault.closed_unsynced);
    t.fault.closed_unsynced = 0;

    // A permission denial on the first turn, before the session exists.
    const n = try newRoot(&t);
    t.fault.fail_error = error.Refused;
    t.fault.next_write = .{ .keep = 0, .then = .fail };
    try testing.expectError(error.AccessDenied, n.append(&.{ .turn_started, item }));
    try testing.expectError(error.AccessDenied, n.append(&.{.turn_started}));
    try closeAndDestroy(n);
}

// ---------------------------------------------------------------------------
// Blobs, fork, children, delete (plan checkpoint 4)

/// `refs` must outlive the event: the caller owns the array.
fn itemWith(refs: []const []const u8) fold.Event {
    return .{ .item = .{ .type = "tool_result", .data = "{\"tool\":\"read\"}", .blobs = refs } };
}

test "blobs: stored once, checked on read, required before a line refers to them" {
    var t: TestEnv = undefined;
    t.init(.{});
    defer t.deinit();
    const s = try newRoot(&t);
    defer closeAndDestroy(s) catch {};
    try testing.expectError(error.InvalidTransition, s.putBlob("too early"));
    _ = try s.append(&.{.turn_started});
    const hash = try s.putBlob("tool output");
    try testing.expectEqualSlices(u8, &hash, &(try s.putBlob("tool output")));
    const refs = [_][]const u8{&hash};
    _ = try s.append(&.{itemWith(&refs)});
    const missing = schema.blobHash("never stored");
    const missing_refs = [_][]const u8{&missing};
    try testing.expectError(error.InvalidTransition, s.append(&.{itemWith(&missing_refs)}));
    const bad_refs = [_][]const u8{"../../etc/passwd"};
    try testing.expectError(error.InvalidTransition, s.append(&.{itemWith(&bad_refs)}));

    const bytes = try session_mod.readBlob(&t.env, gpa, s.id(), &hash);
    defer gpa.free(bytes);
    try testing.expectEqualStrings("tool output", bytes);
    try testing.expectError(error.NotFound, session_mod.readBlob(&t.env, gpa, s.id(), &missing));

    if (hooks) {
        const dir = try t.env.s.openDir(t.env.root, s.id());
        defer t.env.s.closeDir(dir);
        const blobs = try t.env.s.openDir(dir, "blobs");
        defer t.env.s.closeDir(blobs);
        try @import("storage_fault.zig").flipBit(io, blobs, &hash, 3, 0);
        try testing.expectError(error.Corrupt, session_mod.readBlob(&t.env, gpa, s.id(), &hash));
    }
}

/// A source session with three committed turns, a blob in turn 2, and an
/// open fourth turn.
fn forkSource(t: *TestEnv) !struct { id: []u8, hash: [64]u8 } {
    const s = try newRoot(t);
    _ = try s.append(&.{ .turn_started, item, .turn_committed });
    _ = try s.append(&.{.turn_started});
    const hash = try s.putBlob("shared body");
    const refs = [_][]const u8{&hash};
    _ = try s.append(&.{ itemWith(&refs), .turn_committed });
    _ = try s.append(&.{ .turn_started, item, .turn_committed });
    _ = try s.append(&.{ .turn_started, item });
    const id = try gpa.dupe(u8, s.id());
    try closeAndDestroy(s);
    return .{ .id = id, .hash = hash };
}

test "fork at a turn boundary copies the prefix and survives its source" {
    var t: TestEnv = undefined;
    t.init(.{});
    defer t.deinit();
    const source = try forkSource(&t);
    defer gpa.free(source.id);
    const before = try t.kinds(source.id);
    defer gpa.free(before);

    const f = try session_mod.openFork(&t.env, .{ .source = source.id, .at = .{ .turn = 2 }, .workspace = "/w", .host = .app });
    var state = try f.stateCopy(gpa);
    defer state.deinit(gpa);
    try testing.expectEqual(@as(u64, 2), state.committed);
    try testing.expectEqual(@as(?u64, null), state.open_turn);
    try testing.expectEqualStrings(source.id, f.identity.forked_from.?.id);
    // The fork continues with turn 3 of its own.
    _ = try f.append(&.{ .turn_started, item, .turn_committed });
    const fork_id = try gpa.dupe(u8, f.id());
    defer gpa.free(fork_id);
    try closeAndDestroy(f);

    // Lines 2..S match the source's, and the source is unchanged.
    const fork_kinds = try t.kinds(fork_id);
    defer gpa.free(fork_kinds);
    try testing.expectEqualSlices(schema.Kind, before[1..7], fork_kinds[1..7]);
    const after = try t.kinds(source.id);
    defer gpa.free(after);
    try testing.expectEqualSlices(schema.Kind, before, after);

    // Delete the source: the fork still reads fully, blob included.
    try deleteWithoutIndex(&t.env, source.id);
    try testing.expectError(error.NotFound, session_mod.readPage(&t.env, gpa, source.id, .start, .forward, 10));
    const bytes = try session_mod.readBlob(&t.env, gpa, fork_id, &source.hash);
    defer gpa.free(bytes);
    try testing.expectEqualStrings("shared body", bytes);
    const r = try resumeRoot(&t, fork_id);
    try closeAndDestroy(r);
}

test "a fork after a compaction and snapshots verifies clean and resumes at its own offsets" {
    var t: TestEnv = undefined;
    t.init(.{ .snapshot_every_bytes = 256 });
    defer t.deinit();
    const s = try newRoot(&t);
    const big: fold.Event = .{ .item = .{ .type = "assistant", .data = "{\"text\":\"" ++ "x" ** 200 ++ "\"}" } };
    _ = try s.append(&.{ .turn_started, big, .turn_committed });
    _ = try s.append(&.{ .turn_started, .{ .compacted = "{}" }, big, .turn_committed });
    _ = try s.append(&.{ .turn_started, big, .turn_committed });
    const id = try gpa.dupe(u8, s.id());
    defer gpa.free(id);
    try closeAndDestroy(s);

    // The fork's line 1 is longer than the source's, so every copied line
    // moves; the copied snapshots must be encoded again with its offsets.
    const f = try session_mod.openFork(&t.env, .{ .source = id, .at = .{ .turn = 3 }, .workspace = "/w", .host = .app });
    const fork_id = try gpa.dupe(u8, f.id());
    defer gpa.free(fork_id);
    try closeAndDestroy(f);
    const verified = try session_mod.verifySession(&t.env, fork_id);
    try testing.expectEqual(@as(?u64, null), verified.damaged_at);
    try testing.expectEqual(@as(u64, 0), verified.bad_snapshots);

    var page = try session_mod.readPage(&t.env, gpa, fork_id, .start, .forward, 100);
    defer page.deinit();
    var compacted_at: ?u64 = null;
    var snapshots: usize = 0;
    for (page.entries) |entry| {
        if (entry.kind == .compacted) compacted_at = entry.offset;
        if (entry.kind == .snapshot) snapshots += 1;
    }
    try testing.expect(snapshots >= 2);
    // Resume starts from the newest snapshot; the model's context must
    // start at the fork's own compacted line.
    const r = try resumeRoot(&t, fork_id);
    var state = try r.stateCopy(gpa);
    defer state.deinit(gpa);
    try testing.expectEqual(compacted_at, state.compaction_offset);
    try closeAndDestroy(r);
}

test "fork points: turn 0, and never inside a turn" {
    var t: TestEnv = undefined;
    t.init(.{});
    defer t.deinit();
    const source = try forkSource(&t);
    defer gpa.free(source.id);
    try testing.expectError(error.InvalidForkPoint, session_mod.openFork(&t.env, .{ .source = source.id, .at = .{ .turn = 5 }, .workspace = "/w", .host = .app }));
    // Turn 4 was left open by the close: its end line is an interrupt, a boundary.
    const four = try session_mod.openFork(&t.env, .{ .source = source.id, .at = .{ .turn = 4 }, .workspace = "/w", .host = .app });
    try closeAndDestroy(four);
    const zero = try session_mod.openFork(&t.env, .{ .source = source.id, .at = .{ .turn = 0 }, .workspace = "/w", .host = .app });
    const zero_id = try gpa.dupe(u8, zero.id());
    defer gpa.free(zero_id);
    try closeAndDestroy(zero);
    try expectKinds(&t, zero_id, &.{ .session_created, .closed });
    try testing.expectError(error.NotFound, session_mod.openFork(&t.env, .{ .source = "missing", .at = .last_good, .workspace = "/w", .host = .app }));
}

test "recover: a damaged source forks up to its last good turn, untouched" {
    if (!hooks) return error.SkipZigTest;
    var t: TestEnv = undefined;
    t.init(.{});
    defer t.deinit();
    const source = try forkSource(&t);
    defer gpa.free(source.id);
    // Damage the item of turn 3 (line 9).
    var path_buffer: [300]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buffer, "{s}/log.jsonl", .{source.id});
    const bytes = try t.tmp.dir.readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(bytes);
    var line_start: usize = 0;
    for (0..8) |_| line_start = std.mem.findScalarPos(u8, bytes, line_start, '\n').? + 1;
    const dir = try t.env.s.openDir(t.env.root, source.id);
    defer t.env.s.closeDir(dir);
    try @import("storage_fault.zig").flipBit(io, dir, "log.jsonl", line_start + 10, 2);
    const damaged = try t.tmp.dir.readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(damaged);

    try testing.expectError(error.Corrupt, resumeRoot(&t, source.id));
    const f = try session_mod.openFork(&t.env, .{ .source = source.id, .at = .last_good, .workspace = "/w", .host = .app });
    var state = try f.stateCopy(gpa);
    defer state.deinit(gpa);
    try testing.expectEqual(@as(u64, 2), state.committed);
    try closeAndDestroy(f);
    const unchanged = try t.tmp.dir.readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(unchanged);
    try testing.expectEqualSlices(u8, damaged, unchanged);
}

fn childOf(t: *TestEnv, parent: []const u8) !*Session {
    return session_mod.openNew(&t.env, .{ .workspace = "/w", .host = .child, .role = .child, .parent = parent });
}

test "children: finished through the parent, and repaired as lost or interrupted" {
    var t: TestEnv = undefined;
    t.init(.{});
    defer t.deinit();
    const p = try newRoot(&t);
    _ = try p.append(&.{.turn_started});
    const parent_id = try gpa.dupe(u8, p.id());
    defer gpa.free(parent_id);

    // c1 runs to completion.
    const c1 = try childOf(&t, parent_id);
    _ = try p.append(&.{.{ .child_spawned = .{ .child = c1.id(), .work_id = "w1" } }});
    _ = try c1.append(&.{ .turn_started, item, .turn_committed });
    _ = try p.append(&.{.{ .child_finished = .{ .child = c1.id(), .work_id = "w1", .outcome = .ok } }});
    try closeAndDestroy(c1);
    // c2 publishes a log, c3 dies before its first turn; then the parent crashes.
    const c2 = try childOf(&t, parent_id);
    const c3 = try childOf(&t, parent_id);
    _ = try p.append(&.{ .{ .child_spawned = .{ .child = c2.id(), .work_id = "w1" } }, .{ .child_spawned = .{ .child = c3.id(), .work_id = "w1" } } });
    _ = try c2.append(&.{.turn_started});
    const c2_id = try gpa.dupe(u8, c2.id());
    defer gpa.free(c2_id);
    crash(&t, c2);
    try c3.close();
    c3.destroy();
    crash(&t, p);

    const r = try resumeRoot(&t, parent_id);
    var state = try r.stateCopy(gpa);
    defer state.deinit(gpa);
    // Every child stays listed with how its work ended; none is open.
    try testing.expectEqual(@as(usize, 3), state.children.items.len);
    const want = [_]schema.Outcome{ .ok, .interrupted, .lost };
    for (state.children.items, want) |child, outcome| {
        try testing.expect(!child.open);
        try testing.expectEqual(@as(?schema.Outcome, outcome), child.outcome);
    }
    try closeAndDestroy(r);
    var page = try session_mod.readPage(&t.env, gpa, parent_id, .start, .forward, 100);
    defer page.deinit();
    var outcomes: std.ArrayList(schema.Outcome) = .empty;
    defer outcomes.deinit(gpa);
    for (page.entries) |e| if (e.body) |body| switch (body) {
        .child_finished => |f| try outcomes.append(gpa, f.outcome),
        else => {},
    };
    try testing.expectEqualSlices(schema.Outcome, &.{ .ok, .interrupted, .lost }, outcomes.items);

    // A child resumes only through its parent; deleting the parent removes it.
    const again = try session_mod.openResume(&t.env, .{ .id = c2_id, .workspace = "/w", .host = .child, .parent = parent_id });
    try testing.expectError(error.Busy, deleteWithoutIndex(&t.env, parent_id));
    try closeAndDestroy(again);
    try deleteWithoutIndex(&t.env, parent_id);
    try testing.expectError(error.NotFound, session_mod.readPage(&t.env, gpa, c2_id, .start, .forward, 1));
    try testing.expectError(error.NotFound, deleteWithoutIndex(&t.env, parent_id));
    const trash = try t.env.s.openDir(t.env.root, ".trash");
    defer t.env.s.closeDir(trash);
    var listing = t.env.s.list(trash);
    try testing.expectEqual(@as(?storage.Entry, null), try listing.next());
}

test "a fork owns none of its source's children" {
    var t: TestEnv = undefined;
    t.init(.{});
    defer t.deinit();
    const p = try newRoot(&t);
    _ = try p.append(&.{ .turn_started, .{ .child_spawned = .{ .child = "kid", .work_id = "w" } }, .turn_committed });
    const id = try gpa.dupe(u8, p.id());
    defer gpa.free(id);
    try closeAndDestroy(p);
    const f = try session_mod.openFork(&t.env, .{ .source = id, .at = .{ .turn = 1 }, .workspace = "/w", .host = .app });
    const fork_id = try gpa.dupe(u8, f.id());
    defer gpa.free(fork_id);
    crash(&t, f);
    // Its reopen repairs nothing that belongs to the source.
    const r = try resumeRoot(&t, fork_id);
    var state = try r.stateCopy(gpa);
    defer state.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), state.children.items.len);
    try closeAndDestroy(r);
    const kinds = try t.kinds(fork_id);
    defer gpa.free(kinds);
    for (kinds) |k| try testing.expect(k != .child_finished);
}

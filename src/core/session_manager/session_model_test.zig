//! L2 against its four specs: every scenario here writes a trace that
//! `zig build traces` checks with TLC, and every spec also gets one planted
//! bug whose trace must be rejected.
//!
//! Each tracer is an Observer: the Session notifies what it just did, and
//! the tracer names the spec action, reads the disk fields straight from the
//! files (never through the session), and writes one trace line.

const std = @import("std");
const schema = @import("schema.zig");
const storage = @import("storage.zig");
const log_mod = @import("log.zig");
const fold = @import("fold.zig");
const trace = @import("trace.zig");
const session_mod = @import("session.zig");
const Fault = @import("storage_fault.zig").Fault;

const testing = std.testing;
const gpa = testing.allocator;
const io = testing.io;
const Session = session_mod.Session;
const Io = std.Io;

// ---------------------------------------------------------------------------
// Disk helpers (direct reads, no fault layer, no session)

fn readLog(root: Io.Dir, id: []const u8) !?[]u8 {
    var path: [300]u8 = undefined;
    const visible = try std.fmt.bufPrint(&path, "{s}/log.jsonl", .{id});
    if (root.readFileAlloc(io, visible, gpa, .limited(8 << 20))) |bytes| return bytes else |_| {}
    const staged = try std.fmt.bufPrint(&path, ".tmp/{s}/log.jsonl", .{id});
    if (root.readFileAlloc(io, staged, gpa, .limited(8 << 20))) |bytes| return bytes else |_| {}
    return null;
}

/// Every complete line's header and body, parsed from the file bytes.
const DiskLine = struct { seq: u64, kind: ?schema.Kind, body: ?schema.Body };

fn diskLines(arena: std.mem.Allocator, bytes: []const u8) ![]DiskLine {
    var out: std.ArrayList(DiskLine) = .empty;
    var at: usize = 0;
    while (std.mem.findScalarPos(u8, bytes, at, '\n')) |nl| : (at = nl + 1) {
        const line = bytes[at .. nl + 1];
        const header = log_mod.checkLine(line) catch {
            try out.append(arena, .{ .seq = 0, .kind = null, .body = null });
            continue;
        };
        const body: ?schema.Body = if (header.kind) |k|
            schema.parseBody(arena, k, log_mod.lineBody(line, header)) catch null
        else
            null;
        try out.append(arena, .{ .seq = header.seq, .kind = header.kind, .body = body });
    }
    return out.items;
}

fn exists(root: Io.Dir, path: []const u8) bool {
    _ = root.statFile(io, path, .{ .follow_symlinks = false }) catch return false;
    return true;
}

fn testEnv(tmp: *testing.TmpDir, fault: *Fault, options: session_mod.Options) session_mod.Env {
    return .{
        .gpa = gpa,
        .s = .{ .io = io, .fault = fault },
        .root = .{ .handle = tmp.dir },
        .options = options,
    };
}

// ---------------------------------------------------------------------------
// TurnLifecycle

fn runTurnTrace(case: []const u8, planted: trace.Planted) !void {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var fault = Fault.init(gpa, io, 1);
    defer fault.deinit();
    var tracer: trace.TurnTraces = .{ .gpa = gpa, .io = io, .root = tmp.dir, .dir = trace.default_dir, .case = case };
    defer tracer.deinit();
    var env = testEnv(&tmp, &fault, .{});
    env.observer = tracer.observer();
    env.planted = planted;
    const item: fold.Event = .{ .item = .{ .type = "assistant", .data = "{}" } };

    const s = try session_mod.openNew(&env, .{ .workspace = "/w", .host = .app });
    _ = try s.append(&.{.{ .set = .{ .key = .title, .value = "\"t\"" } }});
    _ = try s.append(&.{ .turn_started, item, item });
    try testing.expectError(error.InvalidTransition, s.append(&.{.turn_started}));
    _ = try s.append(&.{ .{ .compacted = "{}" }, .turn_committed });
    if (planted == .accept_item_outside_turn) _ = try s.append(&.{item});
    _ = try s.append(&.{ .turn_started, item });
    const id = try gpa.dupe(u8, s.id());
    defer gpa.free(id);
    s.abandon();
    tracer.crash(id);

    const r = try session_mod.openResume(&env, .{ .id = id, .workspace = "/moved", .host = .app });
    _ = try r.append(&.{ .turn_started, item });
    try r.close();
    r.destroy();

    const again = try session_mod.openResume(&env, .{ .id = id, .workspace = "/moved", .host = .ask });
    _ = try again.append(&.{ .{ .set = .{ .key = .prefs, .value = "{}" } }, .turn_started, .{ .turn_interrupted = .cancel } });
    try again.close();
    again.destroy();
    try testing.expectEqual(@as(usize, 1), tracer.traced().len);
    try tracer.finish();
}

test "TurnLifecycle trace: turns, a refusal, a crash, reopen, close" {
    try runTurnTrace("turns-crash-reopen-close", .none);
}

test "TurnLifecycle trace: planted bug, an item outside a turn" {
    try runTurnTrace("planted-accept_item_outside_turn", .accept_item_outside_turn);
}

// ---------------------------------------------------------------------------
// Lifecycle

const LifecycleTracer = struct {
    trace: trace.Trace,
    root: Io.Dir,
    fault: *Fault,
    ids: [2]?[]u8 = .{ null, null },
    step: [2][]const u8 = .{ "idle", "idle" },
    acked: [2]bool = .{ false, false },

    const labels = [2][]const u8{ "s1", "s2" };

    fn observer(t: *LifecycleTracer) session_mod.Observer {
        return .{ .context = t, .notify = notify };
    }

    fn deinit(t: *LifecycleTracer) void {
        for (t.ids) |maybe| if (maybe) |id| gpa.free(id);
    }

    fn slot(t: *LifecycleTracer, id: []const u8) usize {
        for (t.ids, 0..) |maybe, i| {
            if (maybe) |known| if (std.mem.eql(u8, known, id)) return i;
        }
        for (&t.ids, 0..) |*maybe, i| if (maybe.* == null) {
            maybe.* = gpa.dupe(u8, id) catch @panic("oom");
            return i;
        };
        @panic("more than two traced sessions");
    }

    fn notify(context: *anyopaque, session: *Session, what: session_mod.Observed) void {
        const t: *LifecycleTracer = @ptrCast(@alignCast(context));
        const i = t.slot(session.id());
        const move: ?struct { []const u8, []const u8 } = switch (what) {
            .opened_new => .{ "mem", "OpenNew" },
            .made_tmp => .{ "tmpdir", "MkTmp" },
            .wrote_line => |w| if (w.seq == 1) .{ "written", "WriteLine1" } else null,
            .publish_synced => .{ "synced", "FsyncLog" },
            .renamed => .{ "renamed", "Rename" },
            .published => .{ "done", "FsyncDirAndAck" },
            else => null,
        };
        const m = move orelse return;
        t.step[i] = m[0];
        if (std.mem.eql(u8, m[0], "done")) t.acked[i] = true;
        t.emit(m[1], labels[i]);
    }

    /// fx dies: every unfinished publish stops for good.
    fn crashed(t: *LifecycleTracer, event: []const u8) void {
        for (&t.step) |*step| {
            if (!std.mem.eql(u8, step.*, "idle") and !std.mem.eql(u8, step.*, "done")) step.* = "dead";
        }
        t.emit(event, null);
    }

    fn sweep(t: *LifecycleTracer, s: storage.Storage, i: usize) !void {
        const tmp = try s.openDir(.{ .handle = t.root }, ".tmp");
        defer s.closeDir(tmp);
        try s.deleteTree(tmp, t.ids[i].?);
        t.emit("Sweep", labels[i]);
    }

    fn where(t: *LifecycleTracer, i: usize) []const u8 {
        const id = t.ids[i] orelse return "none";
        var path: [300]u8 = undefined;
        if (exists(t.root, id)) return "visible";
        if (exists(t.root, std.fmt.bufPrint(&path, ".tmp/{s}", .{id}) catch return "none")) return "tmp";
        return "none";
    }

    fn line1(t: *LifecycleTracer, i: usize) []const u8 {
        const id = t.ids[i] orelse return "none";
        const folder = t.where(i);
        if (std.mem.eql(u8, folder, "none")) return "none";
        var path: [300]u8 = undefined;
        const dir_path = if (std.mem.eql(u8, folder, "visible")) id else std.fmt.bufPrint(&path, ".tmp/{s}", .{id}) catch return "none";
        var dir = t.root.openDir(io, dir_path, .{}) catch return "none";
        defer dir.close(io);
        const bytes = dir.readFileAlloc(io, "log.jsonl", gpa, .limited(1 << 20)) catch return "none";
        defer gpa.free(bytes);
        const nl = std.mem.findScalar(u8, bytes, '\n') orelse return "none";
        return if (t.fault.isDurable(.{ .handle = dir }, "log.jsonl", nl + 1)) "durable" else "cached";
    }

    fn emit(t: *LifecycleTracer, event: []const u8, s: ?[]const u8) void {
        const Per = struct { s1: []const u8, s2: []const u8 };
        const PerBool = struct { s1: bool, s2: bool };
        const step: Per = .{ .s1 = t.step[0], .s2 = t.step[1] };
        const where_: Per = .{ .s1 = t.where(0), .s2 = t.where(1) };
        const line1_: Per = .{ .s1 = t.line1(0), .s2 = t.line1(1) };
        const acked: PerBool = .{ .s1 = t.acked[0], .s2 = t.acked[1] };
        if (s) |label| {
            t.trace.write(.{ .event = event, .s = label, .step = step, .where = where_, .line1 = line1_, .acked = acked });
        } else {
            t.trace.write(.{ .event = event, .step = step, .where = where_, .line1 = line1_, .acked = acked });
        }
    }
};

/// Kills the process when the Session reaches `at`.
const KillAt = struct {
    inner: session_mod.Observer,
    fault: *Fault,
    at: std.meta.Tag(session_mod.Observed),
    fired: bool = false,

    fn observer(k: *KillAt) session_mod.Observer {
        return .{ .context = k, .notify = notify };
    }

    fn notify(context: *anyopaque, session: *Session, what: session_mod.Observed) void {
        const k: *KillAt = @ptrCast(@alignCast(context));
        k.inner.notify(k.inner.context, session, what);
        if (!k.fired and what == k.at) {
            k.fired = true;
            k.fault.kill();
        }
    }
};

fn runLifecycleTrace(case: []const u8, planted: trace.Planted, power_loss: bool) !void {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var fault = Fault.init(gpa, io, 5);
    defer fault.deinit();
    var tracer: LifecycleTracer = .{ .trace = try trace.Trace.create(gpa, io, "Lifecycle", case), .root = tmp.dir, .fault = &fault };
    defer tracer.deinit();
    var killer: KillAt = .{ .inner = tracer.observer(), .fault = &fault, .at = .renamed };
    var env = testEnv(&tmp, &fault, .{});
    env.observer = killer.observer();
    env.planted = planted;

    // s1: dies right after its rename, before the root folder is synced.
    const s1 = try session_mod.openNew(&env, .{ .workspace = "/w", .host = .app });
    _ = try s1.append(&.{.{ .set = .{ .key = .title, .value = "\"one\"" } }});
    try testing.expectError(error.Io, s1.append(&.{.turn_started}));
    s1.abandon();
    if (power_loss) {
        _ = fault.powerLoss();
        tracer.crashed("PowerLoss");
        fault.reboot();
    } else {
        tracer.crashed("ProcessCrash");
        fault.restart();
    }
    if (std.mem.eql(u8, tracer.where(0), "tmp")) try tracer.sweep(env.s, 0);

    // s2: a complete first turn after the restart.
    killer.fired = true;
    const s2 = try session_mod.openNew(&env, .{ .workspace = "/w", .host = .app });
    _ = try s2.append(&.{ .turn_started, .turn_committed });
    try s2.close();
    s2.destroy();
    try tracer.trace.finish();
}

test "Lifecycle trace: a crash mid-publish, a sweep, then a full publish" {
    try runLifecycleTrace("crash-after-rename", .none, false);
}

test "Lifecycle trace: a power loss mid-publish" {
    try runLifecycleTrace("power-loss-after-rename", .none, true);
}

test "Lifecycle trace: planted bug, rename before the log sync" {
    try runLifecycleTrace("planted-rename_before_fsync", .rename_before_fsync, false);
}

// ---------------------------------------------------------------------------
// ResumeSnapshot

const SnapTracer = struct {
    trace: trace.Trace,
    root: Io.Dir,

    fn observer(t: *SnapTracer) session_mod.Observer {
        return .{ .context = t, .notify = notify };
    }

    fn notify(context: *anyopaque, session: *Session, what: session_mod.Observed) void {
        const t: *SnapTracer = @ptrCast(@alignCast(context));
        const w = switch (what) {
            .wrote_line => |w| w,
            else => return,
        };
        const event: []const u8 = switch (w.kind) {
            .set => "Set",
            .turn_started => "Start",
            .turn_committed => "Commit",
            .snapshot => "Snapshot",
            else => return,
        };
        t.emit(session.id(), event, w.seq);
    }

    const St = struct { pref: []const u8, turns: u64, open: bool };
    const Entry = struct { k: []const u8, v: []const u8, st: St };
    const empty: St = .{ .pref = "none", .turns = 0, .open = false };

    /// The disk view holds the lines up to `upto_seq` (one batch, one write).
    fn emit(t: *SnapTracer, id: []const u8, event: []const u8, upto_seq: u64) void {
        var arena_state = std.heap.ArenaAllocator.init(gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        const bytes = (readLog(t.root, id) catch null) orelse return;
        defer gpa.free(bytes);
        const lines = diskLines(arena, bytes) catch return;
        var entries: std.ArrayList(Entry) = .empty;
        for (lines) |line| {
            if (line.seq > upto_seq) break;
            const body = line.body orelse continue;
            const entry: Entry = switch (body) {
                .set => |s| .{ .k = "set", .v = unquote(s.value), .st = empty },
                .turn_started => .{ .k = "start", .v = "none", .st = empty },
                .turn_committed => .{ .k = "commit", .v = "none", .st = empty },
                .snapshot => |snap| blk: {
                    var state = fold.decodeState(gpa, arena, snap.state) catch return;
                    defer state.deinit(gpa);
                    const pref = if (state.prefs) |p| arena.dupe(u8, unquote(p)) catch return else "none";
                    break :blk .{ .k = "snap", .v = "none", .st = .{ .pref = pref, .turns = state.committed, .open = state.open_turn != null } };
                },
                else => continue,
            };
            entries.append(arena, entry) catch return;
        }
        const last = entries.items[entries.items.len - 1];
        if (std.mem.eql(u8, event, "Set")) {
            t.trace.write(.{ .event = event, .v = last.v, .log = entries.items });
        } else {
            t.trace.write(.{ .event = event, .log = entries.items });
        }
    }

    fn unquote(raw: []const u8) []const u8 {
        return if (raw.len >= 2 and raw[0] == '"') raw[1 .. raw.len - 1] else raw;
    }
};

fn runSnapshotTrace(case: []const u8, planted: trace.Planted) !void {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var fault = Fault.init(gpa, io, 1);
    defer fault.deinit();
    var tracer: SnapTracer = .{ .trace = try trace.Trace.create(gpa, io, "ResumeSnapshot", case), .root = tmp.dir };
    // A snapshot after every batch.
    var env = testEnv(&tmp, &fault, .{ .snapshot_every_bytes = 1 });
    env.observer = tracer.observer();
    env.planted = planted;
    const v1: fold.Event = .{ .set = .{ .key = .prefs, .value = "\"v1\"" } };
    const v2: fold.Event = .{ .set = .{ .key = .prefs, .value = "\"v2\"" } };

    const s = try session_mod.openNew(&env, .{ .workspace = "/w", .host = .app });
    _ = try s.append(&.{ v1, .turn_started });
    _ = try s.append(&.{v2});
    _ = try s.append(&.{.turn_committed});
    _ = try s.append(&.{ .turn_started, v1 });
    _ = try s.append(&.{ .turn_committed, .turn_started });
    const id = try gpa.dupe(u8, s.id());
    defer gpa.free(id);
    s.abandon();
    // Resume must see exactly the state the log implies.
    env.observer = null;
    const r = try session_mod.openResume(&env, .{ .id = id, .workspace = "/w", .host = .app });
    var state = try r.stateCopy(gpa);
    defer state.deinit(gpa);
    try r.close();
    r.destroy();
    if (planted == .none) {
        try testing.expectEqualStrings("\"v1\"", state.prefs.?);
        try testing.expectEqual(@as(u64, 2), state.committed);
    }
    try tracer.trace.finish();
}

test "ResumeSnapshot trace: a snapshot after every batch" {
    try runSnapshotTrace("snapshot-every-batch", .none);
}

test "ResumeSnapshot trace: planted bug, snapshots drop prefs" {
    try runSnapshotTrace("planted-snapshot_drops_field", .snapshot_drops_field);
}

// ---------------------------------------------------------------------------
// WriterLock

threadlocal var thread_label: []const u8 = "t1";

const LockTracer = struct {
    trace: trace.Trace,
    root: Io.Dir,
    id: []const u8,
    /// Lines on disk before the trace began.
    base: u64,
    mutex: Io.Mutex = .init,
    holder: []const u8 = "none",
    alive: [2]bool = .{ true, true },
    opened: [2]bool = .{ false, false },
    next_seq: [2]u64 = .{ 1, 1 },
    inside: [2][]const u8 = .{ "none", "none" },

    const labels = [2][]const u8{ "p1", "p2" };

    const Proc = struct {
        tracer: *LockTracer,
        p: usize,

        fn observer(proc: *Proc) session_mod.Observer {
            return .{ .context = proc, .notify = notify };
        }

        fn notify(context: *anyopaque, session: *Session, what: session_mod.Observed) void {
            const proc: *Proc = @ptrCast(@alignCast(context));
            const t = proc.tracer;
            t.mutex.lockUncancelable(io);
            defer t.mutex.unlock(io);
            switch (what) {
                .mutex_acquired => {
                    t.inside[proc.p] = thread_label;
                    t.emitLocked("Acquire", proc.p, thread_label);
                },
                .mutex_releasing => {
                    t.inside[proc.p] = "none";
                    t.next_seq[proc.p] = t.relative(session.state.last_seq + 1);
                    t.emitLocked("WriteAndRelease", proc.p, thread_label);
                },
                else => {},
            }
        }
    };

    /// Seq relative to the trace start, with `closed` lines left out.
    fn relative(t: *LockTracer, seq: u64) u64 {
        const bytes = (readLog(t.root, t.id) catch null) orelse return 0;
        defer gpa.free(bytes);
        var arena_state = std.heap.ArenaAllocator.init(gpa);
        defer arena_state.deinit();
        const lines = diskLines(arena_state.allocator(), bytes) catch return 0;
        var skipped: u64 = 0;
        for (lines) |line| {
            if (line.seq > t.base and line.seq < seq and line.kind == .closed) skipped += 1;
        }
        return seq - t.base - skipped;
    }

    fn event(t: *LockTracer, name: []const u8, p: usize, session: ?*Session) void {
        t.mutex.lockUncancelable(io);
        defer t.mutex.unlock(io);
        if (std.mem.eql(u8, name, "Open")) {
            t.holder = labels[p];
            t.opened[p] = true;
            t.next_seq[p] = t.relative(session.?.state.last_seq + 1);
        } else if (std.mem.eql(u8, name, "Close")) {
            t.holder = "none";
            t.opened[p] = false;
        } else if (std.mem.eql(u8, name, "Crash")) {
            t.alive[p] = false;
            t.opened[p] = false;
            t.inside[p] = "none";
            if (std.mem.eql(u8, t.holder, labels[p])) t.holder = "none";
        } else if (std.mem.eql(u8, name, "Restart")) {
            t.alive[p] = true;
        }
        t.emitLocked(name, p, null);
    }

    fn emitLocked(t: *LockTracer, name: []const u8, p: usize, thread: ?[]const u8) void {
        var arena_state = std.heap.ArenaAllocator.init(gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        const Line = struct { seq: u64, by: []const u8 };
        var file: std.ArrayList(Line) = .empty;
        if (readLog(t.root, t.id) catch null) |bytes| {
            defer gpa.free(bytes);
            const lines = diskLines(arena, bytes) catch return;
            var skipped: u64 = 0;
            for (lines) |line| {
                if (line.seq <= t.base) continue;
                if (line.kind == .closed) {
                    skipped += 1;
                    continue;
                }
                const by: []const u8 = if (line.body) |body| switch (body) {
                    .set => |s| byOf(arena, s.value),
                    else => "?",
                } else "?";
                file.append(arena, .{ .seq = line.seq - t.base - skipped, .by = by }) catch return;
            }
        }
        const Bools = struct { p1: bool, p2: bool };
        const Nums = struct { p1: u64, p2: u64 };
        const Names = struct { p1: []const u8, p2: []const u8 };
        const state = .{
            .holder = t.holder,
            .alive = Bools{ .p1 = t.alive[0], .p2 = t.alive[1] },
            .opened = Bools{ .p1 = t.opened[0], .p2 = t.opened[1] },
            .nextSeq = Nums{ .p1 = t.next_seq[0], .p2 = t.next_seq[1] },
            .mutex = Names{ .p1 = t.inside[0], .p2 = t.inside[1] },
        };
        if (thread) |label| {
            t.trace.write(.{ .event = name, .p = labels[p], .t = label, .holder = state.holder, .alive = state.alive, .opened = state.opened, .nextSeq = state.nextSeq, .mutex = state.mutex, .file = file.items });
        } else {
            t.trace.write(.{ .event = name, .p = labels[p], .holder = state.holder, .alive = state.alive, .opened = state.opened, .nextSeq = state.nextSeq, .mutex = state.mutex, .file = file.items });
        }
    }

    fn byOf(arena: std.mem.Allocator, value: []const u8) []const u8 {
        const Parsed = struct { by: []const u8 };
        const parsed = std.json.parseFromSliceLeaky(Parsed, arena, value, .{}) catch return "?";
        return parsed.by;
    }
};

const LockWorker = struct {
    session: *Session,
    label: []const u8,
    value: []const u8,
    count: usize,
    failed: bool = false,

    fn run(w: *LockWorker) void {
        thread_label = w.label;
        for (0..w.count) |_| {
            _ = w.session.append(&.{.{ .set = .{ .key = .prefs, .value = w.value } }}) catch {
                w.failed = true;
                return;
            };
        }
    }
};

fn writeFromTwoThreads(s: *Session, value: []const u8) !void {
    var workers = [_]LockWorker{
        .{ .session = s, .label = "t1", .value = value, .count = 3 },
        .{ .session = s, .label = "t2", .value = value, .count = 3 },
    };
    var threads: [2]std.Thread = undefined;
    for (&workers, &threads) |*w, *thread| thread.* = try std.Thread.spawn(.{}, LockWorker.run, .{w});
    for (threads) |thread| thread.join();
    for (workers) |w| try testing.expect(!w.failed);
}

fn runLockTrace(case: []const u8, planted: trace.Planted) !void {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var fault1 = Fault.init(gpa, io, 1);
    defer fault1.deinit();
    var fault2 = Fault.init(gpa, io, 2);
    defer fault2.deinit();
    // Two processes: two managers over the same root, each with its own flock.
    var env1 = testEnv(&tmp, &fault1, .{ .lock_wait_ms = 20 });
    var env2 = testEnv(&tmp, &fault2, .{ .lock_wait_ms = 20 });
    env2.planted = planted;

    // The session exists before the trace begins.
    const setup = try session_mod.openNew(&env1, .{ .workspace = "/w", .host = .app });
    _ = try setup.append(&.{ .turn_started, .turn_committed });
    const id = try gpa.dupe(u8, setup.id());
    defer gpa.free(id);
    try setup.close();
    const base = setup.state.last_seq;
    setup.destroy();

    var tracer: LockTracer = .{ .trace = try trace.Trace.create(gpa, io, "WriterLock", case), .root = tmp.dir, .id = id, .base = base };
    var proc1: LockTracer.Proc = .{ .tracer = &tracer, .p = 0 };
    var proc2: LockTracer.Proc = .{ .tracer = &tracer, .p = 1 };
    env1.observer = proc1.observer();
    env2.observer = proc2.observer();
    const options: session_mod.ResumeOptions = .{ .id = id, .workspace = "/w", .host = .app };

    // p1 opens and writes from two threads; p2 is refused meanwhile.
    const a = try session_mod.openResume(&env1, options);
    tracer.event("Open", 0, a);
    if (planted == .skip_flock) {
        const b = try session_mod.openResume(&env2, options);
        tracer.event("Open", 1, b);
        b.abandon();
    } else {
        try testing.expectError(error.Busy, session_mod.openResume(&env2, options));
    }
    try writeFromTwoThreads(a, "{\"by\":\"p1\"}");
    // p1 crashes; the kernel frees the flock.
    a.abandon();
    tracer.event("Crash", 0, null);

    // p2 takes over, writes, and closes cleanly.
    const b = try session_mod.openResume(&env2, options);
    tracer.event("Open", 1, b);
    try writeFromTwoThreads(b, "{\"by\":\"p2\"}");
    try b.close();
    tracer.event("Close", 1, null);
    b.destroy();

    // p1 restarts and continues where the log ends.
    tracer.event("Restart", 0, null);
    const c = try session_mod.openResume(&env1, options);
    tracer.event("Open", 0, c);
    try writeFromTwoThreads(c, "{\"by\":\"p1\"}");
    try c.close();
    tracer.event("Close", 0, null);
    c.destroy();
    try tracer.trace.finish();
}

test "WriterLock trace: two processes, two threads each, a crash and a takeover" {
    try runLockTrace("two-processes-two-threads", .none);
}

test "WriterLock trace: planted bug, resume skips the flock" {
    try runLockTrace("planted-skip_flock", .skip_flock);
}

// ---------------------------------------------------------------------------
// Fork

const ForkTracer = struct {
    trace: trace.Trace,
    root: Io.Dir,
    ids: [3]?[]u8 = .{ null, null, null },
    st: [3][]const u8 = .{ "none", "none", "none" },
    /// A fork's staging writes are part of its Fork action; its writes
    /// after the publish are ordinary host writes.
    published: [3]bool = .{ false, false, false },
    /// Ghost: the prefix each fork copied, as the code copied it.
    base: [3][]const Lg = .{ &.{}, &.{}, &.{} },
    base_arena: std.heap.ArenaAllocator,
    blob_hashes: [2][64]u8,
    /// During a line's notify: that session's disk view ends at the line
    /// (a batch is one write, observed line by line after it).
    horizon: ?struct { id: []const u8, seq: u64 } = null,

    const labels = [3][]const u8{ "s1", "s2", "s3" };
    const blob_labels = [2][]const u8{ "b1", "b2" };
    const Lg = struct { k: []const u8, b: []const u8 };

    fn init(t: *ForkTracer, case: []const u8, root: Io.Dir) !void {
        t.* = .{
            .trace = try trace.Trace.create(gpa, io, "Fork", case),
            .root = root,
            .base_arena = .init(gpa),
            .blob_hashes = .{ schema.blobHash("b1"), schema.blobHash("b2") },
        };
    }

    fn deinit(t: *ForkTracer) void {
        for (t.ids) |maybe| if (maybe) |id| gpa.free(id);
        t.base_arena.deinit();
    }

    fn observer(t: *ForkTracer) session_mod.Observer {
        return .{ .context = t, .notify = notify };
    }

    fn slot(t: *ForkTracer, id: []const u8) usize {
        for (t.ids, 0..) |maybe, i| {
            if (maybe) |known| if (std.mem.eql(u8, known, id)) return i;
        }
        for (&t.ids, 0..) |*maybe, i| if (maybe.* == null) {
            maybe.* = gpa.dupe(u8, id) catch @panic("oom");
            return i;
        };
        @panic("more than three traced sessions");
    }

    fn notify(context: *anyopaque, session: *Session, what: session_mod.Observed) void {
        const t: *ForkTracer = @ptrCast(@alignCast(context));
        const i = t.slot(session.id());
        if (session.identity.forked_from != null and !t.published[i]) {
            const origin = session.identity.forked_from.?;
            // A fork is one action: its staging writes are not host writes.
            if (what != .published) return;
            t.published[i] = true;
            const src = t.slot(origin.id);
            var arena_state = std.heap.ArenaAllocator.init(gpa);
            defer arena_state.deinit();
            const source_lg = t.lgOf(arena_state.allocator(), src) catch return;
            const copied = source_lg[1..@intCast(origin.seq)];
            const kept = t.base_arena.allocator().alloc(Lg, copied.len) catch return;
            for (copied, kept) |from, *to| to.* = .{
                .k = t.base_arena.allocator().dupe(u8, from.k) catch return,
                .b = t.base_arena.allocator().dupe(u8, from.b) catch return,
            };
            t.base[i] = kept;
            t.st[i] = "live";
            t.emit("Fork", .{ .src = labels[src], .f = labels[i], .at = origin.seq });
            return;
        }
        const w = switch (what) {
            .wrote_line => |w| w,
            else => return,
        };
        t.horizon = .{ .id = session.id(), .seq = w.seq };
        defer t.horizon = null;
        switch (w.kind) {
            .session_created => {
                t.st[i] = "live";
                t.emit("Create", .{ .s = labels[i] });
            },
            .turn_started => t.emit("Start", .{ .s = labels[i] }),
            .turn_committed => t.emit("Commit", .{ .s = labels[i] }),
            .item => {
                var arena_state = std.heap.ArenaAllocator.init(gpa);
                defer arena_state.deinit();
                const lg = t.lgOf(arena_state.allocator(), i) catch return;
                t.emit("Item", .{ .s = labels[i], .b = lg[lg.len - 1].b });
            },
            else => {},
        }
    }

    fn putBlob(t: *ForkTracer, s: *Session, content: []const u8) ![64]u8 {
        const hash = try s.putBlob(content);
        t.emit("PutBlob", .{ .s = labels[t.slot(s.id())], .b = t.blobLabel(&hash) });
        return hash;
    }

    fn deleted(t: *ForkTracer, id: []const u8) void {
        const i = t.slot(id);
        t.st[i] = "deleted";
        t.emit("Delete", .{ .s = labels[i] });
    }

    fn blobLabel(t: *ForkTracer, hash: []const u8) []const u8 {
        for (t.blob_hashes, blob_labels) |known, label| {
            if (std.mem.eql(u8, &known, hash)) return label;
        }
        return "?";
    }

    fn folder(t: *ForkTracer, buffer: []u8, i: usize) ?[]const u8 {
        const id = t.ids[i] orelse return null;
        if (exists(t.root, id)) return id;
        const staged = std.fmt.bufPrint(buffer, ".tmp/{s}", .{id}) catch return null;
        return if (exists(t.root, staged)) staged else null;
    }

    fn lgOf(t: *ForkTracer, arena: std.mem.Allocator, i: usize) ![]Lg {
        var out: std.ArrayList(Lg) = .empty;
        const id = t.ids[i] orelse return out.items;
        const bytes = (try readLog(t.root, id)) orelse return out.items;
        defer gpa.free(bytes);
        for (try diskLines(arena, bytes)) |line| {
            if (t.horizon) |h| if (std.mem.eql(u8, h.id, id) and line.seq > h.seq) break;
            const body = line.body orelse continue;
            const entry: Lg = switch (body) {
                .session_created => .{ .k = "created", .b = "none" },
                .turn_started => .{ .k = "start", .b = "none" },
                .turn_committed => .{ .k = "commit", .b = "none" },
                .item => |piece| .{ .k = "item", .b = if (piece.blobs.len > 0) t.blobLabel(piece.blobs[0]) else "none" },
                else => continue,
            };
            try out.append(arena, entry);
        }
        return out.items;
    }

    fn linksOf(t: *ForkTracer, arena: std.mem.Allocator, i: usize) ![]const []const u8 {
        var out: std.ArrayList([]const u8) = .empty;
        var buffer: [300]u8 = undefined;
        const path = t.folder(&buffer, i) orelse return out.items;
        var dir = t.root.openDir(io, path, .{}) catch return out.items;
        defer dir.close(io);
        var blobs = dir.openDir(io, "blobs", .{ .iterate = true }) catch return out.items;
        defer blobs.close(io);
        var it = blobs.iterate();
        while (try it.next(io)) |entry| {
            if (!schema.validBlobHash(entry.name)) continue;
            try out.append(arena, t.blobLabel(entry.name));
        }
        std.mem.sort([]const u8, out.items, {}, lessString);
        return out.items;
    }

    fn lessString(_: void, a: []const u8, b: []const u8) bool {
        return std.mem.lessThan(u8, a, b);
    }

    fn emit(t: *ForkTracer, event: []const u8, args: anytype) void {
        var arena_state = std.heap.ArenaAllocator.init(gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        const Set = struct { @"$set": []const []const u8 };
        const PerLg = struct { s1: []const Lg, s2: []const Lg, s3: []const Lg };
        const PerSet = struct { s1: Set, s2: Set, s3: Set };
        const PerSt = struct { s1: []const u8, s2: []const u8, s3: []const u8 };
        var lg: [3][]const Lg = undefined;
        var links: [3]Set = undefined;
        for (0..3) |i| {
            lg[i] = t.lgOf(arena, i) catch return;
            links[i] = .{ .@"$set" = t.linksOf(arena, i) catch return };
        }
        const state = .{
            .st = PerSt{ .s1 = t.st[0], .s2 = t.st[1], .s3 = t.st[2] },
            .lg = PerLg{ .s1 = lg[0], .s2 = lg[1], .s3 = lg[2] },
            .links = PerSet{ .s1 = links[0], .s2 = links[1], .s3 = links[2] },
            .base = PerLg{ .s1 = t.base[0], .s2 = t.base[1], .s3 = t.base[2] },
        };
        const Args = @TypeOf(args);
        if (@hasField(Args, "src")) {
            t.trace.write(.{ .event = event, .src = args.src, .f = args.f, .at = args.at, .st = state.st, .lg = state.lg, .links = state.links, .base = state.base });
        } else if (@hasField(Args, "b")) {
            t.trace.write(.{ .event = event, .s = args.s, .b = args.b, .st = state.st, .lg = state.lg, .links = state.links, .base = state.base });
        } else {
            t.trace.write(.{ .event = event, .s = args.s, .st = state.st, .lg = state.lg, .links = state.links, .base = state.base });
        }
    }
};

fn runForkTrace(case: []const u8, planted: trace.Planted) !void {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var fault = Fault.init(gpa, io, 1);
    defer fault.deinit();
    var tracer: ForkTracer = undefined;
    try tracer.init(case, tmp.dir);
    defer tracer.deinit();
    var env = testEnv(&tmp, &fault, .{});
    env.observer = tracer.observer();
    env.planted = planted;

    const s1 = try session_mod.openNew(&env, .{ .workspace = "/w", .host = .app });
    _ = try s1.append(&.{.turn_started});
    const b1 = try tracer.putBlob(s1, "b1");
    const r1 = [_][]const u8{&b1};
    _ = try s1.append(&.{ .{ .item = .{ .type = "tool_result", .data = "{}", .blobs = &r1 } }, .turn_committed });
    _ = try s1.append(&.{.turn_started});
    const b2 = try tracer.putBlob(s1, "b2");
    const r2 = [_][]const u8{&b2};
    _ = try s1.append(&.{ .{ .item = .{ .type = "tool_result", .data = "{}", .blobs = &r2 } }, .turn_committed });

    // Fork s1 at the end of turn 1, while s1 is still open elsewhere.
    const s2 = try session_mod.openFork(&env, .{ .source = s1.id(), .at = .{ .turn = 1 }, .workspace = "/w", .host = .app });
    if (planted == .fork_skips_blob_link) {
        // The code notices too: one source blob is missing from the fork.
        var missing: usize = 0;
        for ([_][]const u8{ &b1, &b2 }) |hash| {
            if (session_mod.readBlob(&env, gpa, s2.id(), hash)) |bytes| gpa.free(bytes) else |err| {
                try testing.expectEqual(error.NotFound, err);
                missing += 1;
            }
        }
        try testing.expectEqual(@as(usize, 1), missing);
        s1.abandon();
        s2.abandon();
        return tracer.trace.finish();
    }
    _ = try s2.append(&.{.turn_started});
    _ = try s2.append(&.{ .{ .item = .{ .type = "tool_result", .data = "{}", .blobs = &r2 } }, .turn_committed });

    // The source goes away; the fork keeps its prefix and its blobs.
    const s1_id = try gpa.dupe(u8, s1.id());
    defer gpa.free(s1_id);
    s1.abandon();
    try @import("session_test.zig").deleteWithoutIndex(&env, s1_id);
    tracer.deleted(s1_id);
    _ = try s2.append(&.{ .turn_started, .{ .item = .{ .type = "tool_result", .data = "{}", .blobs = &r1 } }, .turn_committed });

    // A fork of the fork.
    const s3 = try session_mod.openFork(&env, .{ .source = s2.id(), .at = .{ .turn = 2 }, .workspace = "/w", .host = .app });
    _ = try s3.append(&.{ .turn_started, .turn_committed });
    s2.abandon();
    s3.abandon();
    try tracer.trace.finish();
}

test "Fork trace: fork, delete the source, fork the fork" {
    try runForkTrace("fork-delete-source-fork-again", .none);
}

test "Fork trace: planted bug, a fork skips a blob link" {
    try runForkTrace("planted-fork_skips_blob_link", .fork_skips_blob_link);
}

// ---------------------------------------------------------------------------
// Subagents

const SubTracer = struct {
    trace: trace.Trace,
    root: Io.Dir,
    parent: ?[]u8 = null,
    children: [2]?[]u8 = .{ null, null },
    /// The child's log has been published.
    published: [2]bool = .{ false, false },
    thread: [2][]const u8 = .{ "none", "none" },
    up: bool = true,
    repairing: bool = false,
    /// During a parent line's notify: the parent's disk view ends at it.
    horizon: ?u64 = null,

    const labels = [2][]const u8{ "c1", "c2" };
    const Plog = struct { k: []const u8, c: []const u8, w: u64, o: []const u8 };

    fn deinit(t: *SubTracer) void {
        if (t.parent) |p| gpa.free(p);
        for (t.children) |maybe| if (maybe) |c| gpa.free(c);
    }

    fn observer(t: *SubTracer) session_mod.Observer {
        return .{ .context = t, .notify = notify };
    }

    fn slot(t: *SubTracer, child: []const u8) usize {
        for (t.children, 0..) |maybe, i| {
            if (maybe) |known| if (std.mem.eql(u8, known, child)) return i;
        }
        for (&t.children, 0..) |*maybe, i| if (maybe.* == null) {
            maybe.* = gpa.dupe(u8, child) catch @panic("oom");
            return i;
        };
        @panic("more than two traced children");
    }

    fn notify(context: *anyopaque, session: *Session, what: session_mod.Observed) void {
        const t: *SubTracer = @ptrCast(@alignCast(context));
        if (session.identity.role == .child) {
            const i = t.slot(session.id());
            switch (what) {
                // The first work's turn publishes the child's log.
                .published => {
                    t.published[i] = true;
                    t.thread[i] = "running";
                    t.emit("ChildStart", labels[i]);
                },
                // Later work runs in the published log.
                .wrote_line => |w| if (w.kind == .turn_started and t.published[i] and
                    std.mem.eql(u8, t.thread[i], "starting"))
                {
                    t.thread[i] = "running";
                    t.emit("ChildStart", labels[i]);
                },
                else => {},
            }
            return;
        }
        if (t.parent == null) t.parent = gpa.dupe(u8, session.id()) catch @panic("oom");
        switch (what) {
            .reopened => {
                t.up = true;
                t.repairing = true;
                t.emit("Reopen", null);
            },
            .wrote_line => |w| {
                if (w.kind != .child_spawned and w.kind != .child_finished) return;
                t.horizon = w.seq;
                defer t.horizon = null;
                var arena_state = std.heap.ArenaAllocator.init(gpa);
                defer arena_state.deinit();
                const plog = t.plogOf(arena_state.allocator()) catch return;
                const last = plog[plog.len - 1];
                const i = t.slot(t.children[labelIndex(last.c)].?);
                switch (w.kind) {
                    .child_spawned => {
                        t.thread[i] = "starting";
                        t.emit("Spawn", labels[i]);
                    },
                    else => if (w.cause == .child_repair) {
                        t.emit("RepairChild", labels[i]);
                    } else {
                        t.thread[i] = "none";
                        t.emit(if (std.mem.eql(u8, last.o, "cancelled")) "Cancel" else "ChildFinish", labels[i]);
                    },
                }
            },
            else => {},
        }
    }

    fn labelIndex(label: []const u8) usize {
        return if (std.mem.eql(u8, label, "c1")) 0 else 1;
    }

    fn spawnable(t: *SubTracer, child: []const u8) void {
        _ = t.slot(child);
    }

    fn crash(t: *SubTracer) void {
        t.up = false;
        t.repairing = false;
        t.thread = .{ "none", "none" };
        t.emit("Crash", null);
    }

    fn repairDone(t: *SubTracer) void {
        t.repairing = false;
        t.emit("RepairDone", null);
    }

    fn plogOf(t: *SubTracer, arena: std.mem.Allocator) ![]Plog {
        var out: std.ArrayList(Plog) = .empty;
        const parent = t.parent orelse return out.items;
        const bytes = (try readLog(t.root, parent)) orelse return out.items;
        defer gpa.free(bytes);
        // A child's work number counts its spawn lines (`tla/Subagents.tla`).
        var works: [2]u64 = .{ 0, 0 };
        for (try diskLines(arena, bytes)) |line| {
            if (t.horizon) |h| if (line.seq > h) break;
            const body = line.body orelse continue;
            switch (body) {
                .child_spawned => |c| {
                    const i = t.slot(c.child);
                    works[i] += 1;
                    try out.append(arena, .{ .k = "spawned", .c = labels[i], .w = works[i], .o = "none" });
                },
                .child_finished => |c| {
                    const i = t.slot(c.child);
                    try out.append(arena, .{ .k = "finished", .c = labels[i], .w = works[i], .o = @tagName(c.outcome) });
                },
                else => {},
            }
        }
        return out.items;
    }

    fn emit(t: *SubTracer, event: []const u8, c: ?[]const u8) void {
        var arena_state = std.heap.ArenaAllocator.init(gpa);
        defer arena_state.deinit();
        const plog = t.plogOf(arena_state.allocator()) catch return;
        var path: [300]u8 = undefined;
        var has_log: [2]bool = .{ false, false };
        for (t.children, 0..) |maybe, i| if (maybe) |child| {
            has_log[i] = exists(t.root, std.fmt.bufPrint(&path, "{s}/log.jsonl", .{child}) catch continue);
        };
        const Bools = struct { c1: bool, c2: bool };
        const Names = struct { c1: []const u8, c2: []const u8 };
        const child_log: Bools = .{ .c1 = has_log[0], .c2 = has_log[1] };
        const thread: Names = .{ .c1 = t.thread[0], .c2 = t.thread[1] };
        if (c) |label| {
            t.trace.write(.{ .event = event, .c = label, .plog = plog, .childLog = child_log, .thread = thread, .up = t.up, .repairing = t.repairing });
        } else {
            t.trace.write(.{ .event = event, .plog = plog, .childLog = child_log, .thread = thread, .up = t.up, .repairing = t.repairing });
        }
    }
};

fn runSubagentsTrace(case: []const u8, planted: trace.Planted, finish_first: bool) !void {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var fault = Fault.init(gpa, io, 1);
    defer fault.deinit();
    var tracer: SubTracer = .{ .trace = try trace.Trace.create(gpa, io, "Subagents", case), .root = tmp.dir };
    defer tracer.deinit();
    var env = testEnv(&tmp, &fault, .{});
    env.observer = tracer.observer();
    env.planted = planted;

    const p = try session_mod.openNew(&env, .{ .workspace = "/w", .host = .app });
    _ = try p.append(&.{.turn_started});
    const parent_id = try gpa.dupe(u8, p.id());
    defer gpa.free(parent_id);

    const c1 = try session_mod.openNew(&env, .{ .workspace = "/w", .host = .child, .role = .child, .parent = parent_id });
    tracer.spawnable(c1.id());
    _ = try p.append(&.{.{ .child_spawned = .{ .child = c1.id(), .work_id = "w1" } }});
    _ = try c1.append(&.{ .turn_started, .turn_committed });
    if (finish_first) {
        _ = try p.append(&.{.{ .child_finished = .{ .child = c1.id(), .work_id = "w1", .outcome = .ok } }});
    }
    const c2 = try session_mod.openNew(&env, .{ .workspace = "/w", .host = .child, .role = .child, .parent = parent_id });
    tracer.spawnable(c2.id());
    _ = try p.append(&.{.{ .child_spawned = .{ .child = c2.id(), .work_id = "w1" } }});

    // The process dies with c2 never started (and c1 still running unless finished).
    c1.abandon();
    c2.abandon();
    p.abandon();
    tracer.crash();

    const r = try session_mod.openResume(&env, .{ .id = parent_id, .workspace = "/w", .host = .app });
    tracer.repairDone();
    r.abandon();
    try tracer.trace.finish();
}

test "Subagents trace: one child finishes, one is lost in a crash" {
    try runSubagentsTrace("finish-then-lost", .none, true);
}

test "Subagents trace: a crash leaves one child interrupted and one lost" {
    try runSubagentsTrace("interrupted-and-lost", .none, false);
}

test "Subagents trace: planted bug, repair finishes a child twice" {
    try runSubagentsTrace("planted-repair_marks_child_twice", .repair_marks_child_twice, false);
}

test "Subagents trace: a named child fails, works again, is cancelled; a crash interrupts another" {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var fault = Fault.init(gpa, io, 1);
    defer fault.deinit();
    var tracer: SubTracer = .{ .trace = try trace.Trace.create(gpa, io, "Subagents", "second-work-cancel-interrupt"), .root = tmp.dir };
    defer tracer.deinit();
    var env = testEnv(&tmp, &fault, .{});
    env.observer = tracer.observer();

    const p = try session_mod.openNew(&env, .{ .workspace = "/w", .host = .app });
    _ = try p.append(&.{.turn_started});
    const parent_id = try gpa.dupe(u8, p.id());
    defer gpa.free(parent_id);

    // c1's first work fails; its second runs in the same log and is cancelled.
    const c1 = try session_mod.openNew(&env, .{ .workspace = "/w", .host = .child, .role = .child, .parent = parent_id });
    tracer.spawnable(c1.id());
    _ = try p.append(&.{.{ .child_spawned = .{ .child = c1.id(), .work_id = "w1", .data = "{\"name\":\"reviewer\"}" } }});
    _ = try c1.append(&.{ .turn_started, .turn_committed });
    _ = try p.append(&.{.{ .child_finished = .{ .child = c1.id(), .work_id = "w1", .outcome = .failed, .data = "{\"error\":\"e\"}" } }});
    _ = try p.append(&.{.{ .child_spawned = .{ .child = c1.id(), .work_id = "w2" } }});
    _ = try c1.append(&.{.turn_started});
    _ = try p.append(&.{.{ .child_finished = .{ .child = c1.id(), .work_id = "w2", .outcome = .cancelled } }});
    _ = try c1.append(&.{.{ .turn_interrupted = .cancel }});

    // c2 is running when the process dies.
    const c2 = try session_mod.openNew(&env, .{ .workspace = "/w", .host = .child, .role = .child, .parent = parent_id });
    tracer.spawnable(c2.id());
    _ = try p.append(&.{.{ .child_spawned = .{ .child = c2.id(), .work_id = "w1" } }});
    _ = try c2.append(&.{.turn_started});
    c1.abandon();
    c2.abandon();
    p.abandon();
    tracer.crash();

    const r = try session_mod.openResume(&env, .{ .id = parent_id, .workspace = "/w", .host = .app });
    tracer.repairDone();
    var state = try r.stateCopy(gpa);
    defer state.deinit(gpa);
    r.abandon();
    try tracer.trace.finish();

    try testing.expectEqual(@as(usize, 2), state.children.items.len);
    const named = state.children.items[0];
    try testing.expectEqualStrings("w2", named.work_id);
    try testing.expectEqual(@as(?schema.Outcome, .cancelled), named.outcome);
    try testing.expectEqual(@as(?[]u8, null), named.spawn_data);
    try testing.expectEqual(@as(?[]u8, null), named.finish_data);
    try testing.expectEqual(@as(?schema.Outcome, .interrupted), state.children.items[1].outcome);
    try testing.expect(!state.children.items[1].open);
}

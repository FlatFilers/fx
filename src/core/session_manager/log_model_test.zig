//! L1 against `tla/SessionLog.tla`: random schedules of appends, syncs,
//! process crashes (including in the middle of a write), machine crashes and
//! recoveries. After every step the four spec invariants are checked on the
//! real file, and selected runs write a trace for TLC (`zig build traces`).

const std = @import("std");
const schema = @import("schema.zig");
const storage = @import("storage.zig");
const log_mod = @import("log.zig");
const trace = @import("trace.zig");
const Fault = @import("storage_fault.zig").Fault;

const testing = std.testing;
const gpa = testing.allocator;
const io = testing.io;
const Log = log_mod.Log;

const name = "log.jsonl";
/// `SessionLogTrace.cfg` sets MaxLines to 64; stay under it.
const max_lines = 48;

const Harness = struct {
    tmp: testing.TmpDir,
    fault: Fault,
    log: ?Log = null,
    /// Every byte this harness wrote, in file order. After recovery the file
    /// must equal a prefix of it.
    history: std.ArrayList(u8) = .empty,
    /// Byte length and line count acknowledged durable (ghost state).
    acked_bytes: usize = 0,
    acked_lines: u64 = 0,
    /// Durable lines of the last open log: survives a crash, like the spec's
    /// `durable`, until the next recovery sets it.
    durable: u64 = 0,
    tracer: ?*trace.SessionLogTracer = null,
    planted: trace.Planted = .none,

    fn init(seed: u64) Harness {
        return .{ .tmp = testing.tmpDir(.{ .iterate = true }), .fault = .init(gpa, io, seed) };
    }

    fn deinit(h: *Harness) void {
        if (h.log) |*log| log.close();
        h.history.deinit(gpa);
        h.fault.deinit();
        h.tmp.cleanup();
    }

    fn storageFor(h: *Harness) storage.Storage {
        return .{ .io = io, .fault = &h.fault };
    }

    fn dir(h: *Harness) storage.Dir {
        return .{ .handle = h.tmp.dir };
    }

    fn hook(h: *Harness) log_mod.Hook {
        return .{ .tracer = h.tracer, .planted = h.planted };
    }

    fn create(h: *Harness) !void {
        h.log = try Log.create(h.storageFor(), h.dir(), name, h.hook());
        // The name is made durable, as the first-turn publish does.
        try h.storageFor().syncDir(h.dir());
    }

    fn lines(h: *Harness) u64 {
        return std.mem.count(u8, h.history.items, "\n");
    }

    fn append(h: *Harness, count: u64) !void {
        const log = &h.log.?;
        var batch: std.ArrayList(u8) = .empty;
        defer batch.deinit(gpa);
        var i: u64 = 0;
        while (i < count) : (i += 1) {
            try log_mod.appendLine(gpa, &batch, log.next_seq + i, 1, .item, ",\"turn\":1,\"data\":{}");
        }
        // Recorded first: after a death part of the batch may be on disk,
        // and recovery checks the file against a prefix of the history.
        try h.history.appendSlice(gpa, batch.items);
        log.append(batch.items, count) catch |err| switch (err) {
            // An injected death: the harness records the crash.
            error.Io => return h.crashed(),
            else => return err,
        };
    }

    fn sync(h: *Harness) !void {
        const log = &h.log.?;
        log.sync() catch |err| switch (err) {
            error.Io => return h.crashed(),
            else => return err,
        };
        h.durable = log.synced_lines;
        h.acked_lines = log.synced_lines;
        h.acked_bytes = h.history.items.len;
    }

    /// The process died (a kill, or an injected failure that stops the writer).
    fn crashed(h: *Harness) !void {
        if (h.log) |*log| {
            h.durable = log.synced_lines;
            log.close();
        }
        h.log = null;
        if (h.tracer) |t| t.step("ProcessCrash", h.durable, .down);
        try h.checkDown();
    }

    fn kill(h: *Harness) !void {
        h.fault.kill();
        try h.crashed();
        h.fault.restart();
    }

    /// Dies between the two halves of the next line (trace mode), or after
    /// part of the next batch (plain mode).
    fn planMidWriteDeath(h: *Harness, random: std.Random) void {
        h.fault.next_write = if (h.tracer != null)
            .{ .after_calls = 1, .keep = 0, .then = .die }
        else
            .{ .keep = random.uintLessThan(usize, 200), .then = .die };
    }

    fn powerLoss(h: *Harness) !void {
        if (h.log) |*log| {
            h.durable = log.synced_lines;
            log.close();
        }
        h.log = null;
        _ = h.fault.powerLoss();
        if (h.tracer) |t| t.step("PowerLoss", h.durable, .down);
        h.fault.reboot();
        try h.checkDown();
    }

    fn recover(h: *Harness) !void {
        std.debug.assert(h.log == null);
        h.fault.restart();
        const opened = try Log.open(gpa, h.storageFor(), h.dir(), name, .read_write, h.hook());
        h.log = opened.log;
        h.durable = opened.log.synced_lines;
        // The file must now be exactly a prefix of what was written.
        const bytes = try h.readFile();
        defer gpa.free(bytes);
        try testing.expect(std.mem.startsWith(u8, h.history.items, bytes));
        h.history.shrinkRetainingCapacity(bytes.len);
        if (h.planted == .none) try h.checkOpen();
    }

    fn readFile(h: *Harness) ![]u8 {
        return h.tmp.dir.readFileAlloc(io, name, gpa, .limited(1 << 20));
    }

    /// `SeqContiguous`, `OnlyTailTorn` and `AckedSurvive` hold at every step.
    fn checkDown(h: *Harness) !void {
        const bytes = try h.readFile();
        defer gpa.free(bytes);
        const scan = log_mod.scanBytes(bytes, 0, 1);
        switch (scan.verdict) {
            .clean, .torn => {},
            .corrupt, .newer_version => return error.InvariantViolated,
        }
        // Every acknowledged line is present and unchanged.
        try testing.expect(bytes.len >= h.acked_bytes);
        try testing.expectEqualSlices(u8, h.history.items[0..h.acked_bytes], bytes[0..h.acked_bytes]);
        try testing.expect(scan.scanner.last_seq >= h.acked_lines);
        try testing.expect(h.acked_lines <= h.durable);
    }

    /// Plus `OpenIsClean` while the log is open.
    fn checkOpen(h: *Harness) !void {
        try h.checkDown();
        const bytes = try h.readFile();
        defer gpa.free(bytes);
        try testing.expectEqual(log_mod.Verdict.clean, log_mod.scanBytes(bytes, 0, 1).verdict);
        try testing.expectEqual(h.log.?.lineCount(), std.mem.count(u8, bytes, "\n"));
    }
};

/// One random schedule. Returns the number of recoveries it exercised.
fn runSchedule(seed: u64, tracer: ?*trace.SessionLogTracer) !usize {
    var h = Harness.init(seed);
    defer h.deinit();
    h.tracer = tracer;
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    try h.create();
    var recoveries: usize = 0;
    var steps: usize = 0;
    while (steps < 40) : (steps += 1) {
        if (h.log == null) {
            try h.recover();
            recoveries += 1;
            continue;
        }
        const room = max_lines -| h.lines();
        switch (random.uintLessThan(u8, 10)) {
            0...3 => if (room >= 3) try h.append(random.intRangeAtMost(u64, 1, 3)),
            4, 5 => try h.sync(),
            6 => try h.kill(),
            7 => if (room >= 3) {
                h.planMidWriteDeath(random);
                try h.append(random.intRangeAtMost(u64, 1, 3));
            },
            8 => try h.powerLoss(),
            else => {
                h.fault.fail_next_sync = true;
                try h.sync();
            },
        }
        if (h.log != null) try h.checkOpen();
    }
    return recoveries;
}

test "SessionLog invariants hold under random fault schedules" {
    var recoveries: usize = 0;
    var seed: u64 = 0;
    while (seed < 300) : (seed += 1) recoveries += try runSchedule(seed, null);
    // The schedules really exercised recovery.
    try testing.expect(recoveries > 300);
}

fn tracedSchedule(seed: u64) !void {
    var case_buffer: [64]u8 = undefined;
    const case = try std.fmt.bufPrint(&case_buffer, "fault-schedule-seed-{d}", .{seed});
    try runTraced(seed, case, .none);
}

fn runTraced(seed: u64, case: []const u8, planted: trace.Planted) !void {
    var h = Harness.init(seed);
    defer h.deinit();
    var tracer: trace.SessionLogTracer = .{
        .trace = try trace.Trace.create(gpa, io, "SessionLog", case),
        .dir = h.tmp.dir,
        .name = name,
    };
    h.tracer = &tracer;
    h.planted = planted;
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    try h.create();
    if (planted == .keep_torn_tail) {
        // A torn tail must exist at the next recovery.
        try h.append(2);
        try h.sync();
        h.planMidWriteDeath(random);
        try h.append(1);
        try h.recover();
        // The in-process check sees the bug too: the open log ends torn,
        // which `OpenIsClean` forbids.
        const bytes = try h.readFile();
        defer gpa.free(bytes);
        try testing.expect(log_mod.scanBytes(bytes, 0, 1).verdict == .torn);
    } else {
        var steps: usize = 0;
        while (steps < 30) : (steps += 1) {
            if (h.log == null) {
                try h.recover();
                continue;
            }
            const room = max_lines -| h.lines();
            switch (random.uintLessThan(u8, 9)) {
                0...3 => if (room >= 3) try h.append(random.intRangeAtMost(u64, 1, 3)),
                4, 5 => try h.sync(),
                6 => try h.kill(),
                7 => if (room >= 3) {
                    h.planMidWriteDeath(random);
                    try h.append(1);
                },
                else => try h.powerLoss(),
            }
        }
    }
    try tracer.finish();
}

test "SessionLog traces: fault schedules" {
    for ([_]u64{ 1, 2, 3, 4, 5, 6 }) |seed| try tracedSchedule(seed);
}

test "SessionLog traces: crash in the middle of a write, then a torn machine crash" {
    var h = Harness.init(99);
    defer h.deinit();
    var tracer: trace.SessionLogTracer = .{
        .trace = try trace.Trace.create(gpa, io, "SessionLog", "crash-mid-write"),
        .dir = h.tmp.dir,
        .name = name,
    };
    h.tracer = &tracer;
    var prng = std.Random.DefaultPrng.init(99);
    try h.create();
    try h.append(2);
    try h.sync();
    h.planMidWriteDeath(prng.random());
    try h.append(1);
    try h.recover();
    try h.append(1);
    try h.powerLoss();
    try h.recover();
    try h.append(1);
    try h.sync();
    try tracer.finish();
}

test "SessionLog traces: planted bug, recovery keeps the torn tail" {
    try runTraced(7, "planted-keep_torn_tail", .keep_torn_tail);
}

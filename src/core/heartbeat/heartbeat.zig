//! `/every` heartbeat: re-submits one prompt into the interactive session on a
//! fixed interval while the session is idle. State is session-scoped and in
//! memory only; it does not survive restarting fx.

const std = @import("std");

const Allocator = std.mem.Allocator;

pub const min_interval_ms: i64 = 10 * std.time.ms_per_s;
pub const max_interval_ms: i64 = 24 * std.time.ms_per_hour;
pub const max_prompt_bytes: usize = 4096;

pub const usage = "/every <interval> <prompt> | /every off | /every";

/// File inside the session directory that holds the saved schedule.
pub const sidecar_name = "every.txt";

pub const Command = union(enum) {
    show,
    off,
    set: Set,
    invalid: []const u8,

    pub const Set = struct {
        interval_ms: i64,
        interval_label: []const u8,
        prompt: []const u8,
    };
};

/// Parses the text after `/every`. Returned slices borrow from `rest`.
pub fn parse(rest: []const u8) Command {
    const trimmed = std.mem.trim(u8, rest, " \t\r\n");
    if (trimmed.len == 0) return .show;
    if (std.ascii.eqlIgnoreCase(trimmed, "off") or std.ascii.eqlIgnoreCase(trimmed, "stop")) return .off;

    const split = std.mem.findAny(u8, trimmed, " \t") orelse
        return .{ .invalid = "add a prompt after the interval, e.g. /every 10m brief me" };
    const interval_label = trimmed[0..split];
    const prompt = std.mem.trim(u8, trimmed[split..], " \t\r\n");
    const interval_ms = parseInterval(interval_label) orelse
        return .{ .invalid = "interval must be a number with s, m, or h, e.g. 30s, 10m, 1h" };
    if (interval_ms < min_interval_ms) return .{ .invalid = "interval must be at least 10s" };
    if (interval_ms > max_interval_ms) return .{ .invalid = "interval must be at most 24h" };
    if (prompt.len == 0) return .{ .invalid = "add a prompt after the interval, e.g. /every 10m brief me" };
    if (prompt.len > max_prompt_bytes) return .{ .invalid = "prompt is too long for /every (4096 bytes max)" };
    return .{ .set = .{ .interval_ms = interval_ms, .interval_label = interval_label, .prompt = prompt } };
}

/// Parses `<digits><s|m|h>` into milliseconds. Returns null when malformed.
pub fn parseInterval(label: []const u8) ?i64 {
    if (label.len < 2) return null;
    const unit_ms: i64 = switch (std.ascii.toLower(label[label.len - 1])) {
        's' => std.time.ms_per_s,
        'm' => std.time.ms_per_min,
        'h' => std.time.ms_per_hour,
        else => return null,
    };
    const count = std.fmt.parseInt(i64, label[0 .. label.len - 1], 10) catch return null;
    if (count <= 0) return null;
    return std.math.mul(i64, count, unit_ms) catch null;
}

pub const Beat = enum {
    /// Nothing scheduled, or not due yet.
    none,
    /// Due and the session is idle: submit the prompt now.
    fire,
    /// Due while the session was busy: this beat was skipped and rearmed.
    skipped,
};

/// One active heartbeat. Owns `prompt` and `interval_label`.
pub const State = struct {
    prompt: ?[]u8 = null,
    interval_label: []u8 = &.{},
    interval_ms: i64 = 0,
    next_due_ms: i64 = 0,
    fired_count: u64 = 0,
    skipped_count: u64 = 0,

    pub fn deinit(self: *State, alloc: Allocator) void {
        self.clear(alloc);
    }

    pub fn active(self: *const State) bool {
        return self.prompt != null;
    }

    /// Replaces any active heartbeat. The first beat is one interval from `now_ms`.
    pub fn start(self: *State, alloc: Allocator, set: Command.Set, now_ms: i64) Allocator.Error!void {
        const prompt = try alloc.dupe(u8, set.prompt);
        errdefer alloc.free(prompt);
        const label = try alloc.dupe(u8, set.interval_label);
        self.clear(alloc);
        self.* = .{
            .prompt = prompt,
            .interval_label = label,
            .interval_ms = set.interval_ms,
            .next_due_ms = now_ms + set.interval_ms,
        };
    }

    pub fn clear(self: *State, alloc: Allocator) void {
        if (self.prompt) |prompt| alloc.free(prompt);
        if (self.interval_label.len > 0) alloc.free(self.interval_label);
        self.* = .{};
    }

    /// Decides what the loop should do this tick. `idle` means no turn is
    /// running or queued and the composer is empty, so a submitted prompt
    /// cannot interrupt the user or an in-flight turn.
    pub fn tick(self: *State, now_ms: i64, idle: bool) Beat {
        if (!self.active() or now_ms < self.next_due_ms) return .none;
        self.next_due_ms = now_ms + self.interval_ms;
        if (!idle) {
            self.skipped_count += 1;
            return .skipped;
        }
        self.fired_count += 1;
        return .fire;
    }

    pub fn secondsUntilNext(self: *const State, now_ms: i64) i64 {
        return @divFloor(@max(self.next_due_ms - now_ms, 0) + 999, std.time.ms_per_s);
    }
};

/// The saved form of an active schedule: the same `<interval> <prompt>` text
/// `/every` accepts, so restoring is `parse`. Caller owns the result.
pub fn serialize(alloc: Allocator, state: *const State) Allocator.Error![]u8 {
    return std.fmt.allocPrint(alloc, "{s} {s}\n", .{ state.interval_label, state.prompt.? });
}

/// Text submitted as the user turn for a beat. Caller owns the result.
pub fn beatPrompt(alloc: Allocator, state: *const State) Allocator.Error![]u8 {
    return std.fmt.allocPrint(alloc, "[every {s}] {s}", .{ state.interval_label, state.prompt.? });
}

test "parse recognizes show, off, and set" {
    try std.testing.expectEqual(Command.show, parse("  "));
    try std.testing.expectEqual(Command.off, parse("off"));
    try std.testing.expectEqual(Command.off, parse("STOP"));
    const cmd = parse(" 10m  brief me on Routines ");
    try std.testing.expectEqual(@as(i64, 10 * std.time.ms_per_min), cmd.set.interval_ms);
    try std.testing.expectEqualStrings("10m", cmd.set.interval_label);
    try std.testing.expectEqualStrings("brief me on Routines", cmd.set.prompt);
}

test "parse rejects malformed intervals and missing prompts" {
    try std.testing.expect(parse("10m") == .invalid);
    try std.testing.expect(parse("10 brief") == .invalid);
    try std.testing.expect(parse("5s brief") == .invalid);
    try std.testing.expect(parse("25h brief") == .invalid);
    try std.testing.expect(parse("0m brief") == .invalid);
    try std.testing.expect(parse("-1m brief") == .invalid);
}

test "parseInterval handles units and overflow" {
    try std.testing.expectEqual(@as(?i64, 30_000), parseInterval("30s"));
    try std.testing.expectEqual(@as(?i64, 3_600_000), parseInterval("1H"));
    try std.testing.expectEqual(@as(?i64, null), parseInterval("m"));
    try std.testing.expectEqual(@as(?i64, null), parseInterval("99999999999999999999h"));
}

test "tick fires when idle, skips when busy, and rearms each beat" {
    const alloc = std.testing.allocator;
    var state: State = .{};
    defer state.deinit(alloc);
    try std.testing.expectEqual(Beat.none, state.tick(0, true));

    try state.start(alloc, parse("1m check holding").set, 1_000);
    try std.testing.expectEqual(Beat.none, state.tick(60_999, true));
    try std.testing.expectEqual(Beat.fire, state.tick(61_000, true));
    try std.testing.expectEqual(Beat.none, state.tick(61_001, true));
    try std.testing.expectEqual(Beat.skipped, state.tick(121_000, false));
    try std.testing.expectEqual(Beat.fire, state.tick(181_000, true));
    try std.testing.expectEqual(@as(u64, 2), state.fired_count);
    try std.testing.expectEqual(@as(u64, 1), state.skipped_count);
    try std.testing.expectEqual(@as(i64, 60), state.secondsUntilNext(181_000));

    const text = try beatPrompt(alloc, &state);
    defer alloc.free(text);
    try std.testing.expectEqualStrings("[every 1m] check holding", text);
}

test "serialize round-trips through parse" {
    const alloc = std.testing.allocator;
    var state: State = .{};
    defer state.deinit(alloc);
    try state.start(alloc, parse("15m check status and manage work").set, 0);
    const text = try serialize(alloc, &state);
    defer alloc.free(text);
    const restored = parse(text).set;
    try std.testing.expectEqualStrings("15m", restored.interval_label);
    try std.testing.expectEqualStrings("check status and manage work", restored.prompt);
    try std.testing.expectEqual(@as(i64, 15 * std.time.ms_per_min), restored.interval_ms);
}

test "start replaces and clear stops an active heartbeat" {
    const alloc = std.testing.allocator;
    var state: State = .{};
    defer state.deinit(alloc);
    try state.start(alloc, parse("10m first").set, 0);
    try state.start(alloc, parse("1h second").set, 0);
    try std.testing.expectEqualStrings("second", state.prompt.?);
    try std.testing.expectEqual(@as(i64, std.time.ms_per_hour), state.next_due_ms);
    state.clear(alloc);
    try std.testing.expect(!state.active());
    try std.testing.expectEqual(Beat.none, state.tick(std.math.maxInt(i64), true));
}

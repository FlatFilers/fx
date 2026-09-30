//! Best-effort lifecycle reporter for the Inby terminal sidebar.
//!
//! When fx runs inside an Inby tab (`INBY_SESSION` is set), the tab's icon,
//! colour and sidebar pin follow fx's semantic state: working while a turn
//! runs, "needs you" while blocked on the human, and back to the tab's resting
//! look when the turn ends. The title is never touched; the agent or the human
//! owns it. Fields the human set (origin `user`) are left alone, and a look the
//! agent chose mid-turn is kept rather than restored over.
//!
//! Reports go through the `inby` CLI on a worker thread so a slow or missing
//! CLI cannot block the UI loop. Failures are traced and otherwise ignored.

const std = @import("std");
const io_mod = @import("../../core/shared/io.zig");
const debug_trace = @import("../../core/shared/debug_trace.zig");
const host_target = @import("../../core/hosts/target.zig");

const Allocator = std.mem.Allocator;

pub const State = enum { idle, working, needs_you };

pub const Look = struct {
    icon: []const u8,
    color: []const u8,
};

pub const working_look: Look = .{ .icon = "hammer.fill", .color = "aqua" };
pub const needs_you_look: Look = .{ .icon = "hand.raised.fill", .color = "amber" };

/// The fields of `inby tab --get` this reporter reads.
pub const TabSnapshot = struct {
    icon: []const u8 = "",
    color: []const u8 = "",
    iconOrigin: []const u8 = "default",
    colorOrigin: []const u8 = "default",

    fn iconLocked(self: TabSnapshot) bool {
        return std.mem.eql(u8, self.iconOrigin, "user");
    }

    fn colorLocked(self: TabSnapshot) bool {
        return std.mem.eql(u8, self.colorOrigin, "user");
    }
};

pub const Pin = enum { active, auto };

/// One tab update. A null field is left unchanged.
pub const Action = struct {
    icon: ?[]const u8 = null,
    color: ?[]const u8 = null,
    pin: ?Pin = null,
};

/// Decides the update for a transition. `resting` is the look captured when
/// the tab last left idle; `last_set` is the look fx applied most recently.
pub fn plan(target: State, current: TabSnapshot, resting: ?Look, last_set: ?Look) Action {
    switch (target) {
        .working, .needs_you => {
            const look = if (target == .working) working_look else needs_you_look;
            return .{
                .icon = if (current.iconLocked()) null else look.icon,
                .color = if (current.colorLocked()) null else look.color,
                .pin = .active,
            };
        },
        .idle => {
            const rest = resting orelse return .{ .pin = .auto };
            const set = last_set orelse return .{ .pin = .auto };
            // Restore only fields that still show what fx set; anything else
            // was changed by the agent or the human during the turn.
            return .{
                .icon = if (!current.iconLocked() and std.mem.eql(u8, current.icon, set.icon)) rest.icon else null,
                .color = if (!current.colorLocked() and std.mem.eql(u8, current.color, set.color)) rest.color else null,
                .pin = .auto,
            };
        },
    }
}

pub fn shouldEnable(fx_inby: ?[]const u8, inby_session: ?[]const u8) bool {
    if (fx_inby) |val| {
        if (std.mem.eql(u8, val, "0") or std.ascii.eqlIgnoreCase(val, "false")) return false;
    }
    const session = inby_session orelse return false;
    return session.len > 0;
}

pub const Client = struct {
    enabled: bool = false,
    alloc: Allocator = undefined,
    mutex: std.Io.Mutex = .init,
    desired: State = .idle,
    desired_generation: u64 = 0,
    stop_requested: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,

    // Owned by the worker thread.
    applied: State = .idle,
    applied_generation: u64 = 0,
    resting_icon: []u8 = &.{},
    resting_color: []u8 = &.{},
    has_resting: bool = false,
    last_set: ?Look = null,

    pub fn initFromEnv(self: *Client, alloc: Allocator) void {
        if (comptime host_target.is_wasm or @import("builtin").single_threaded or @import("builtin").is_test) return;
        if (!shouldEnable(io_mod.getenv("FX_INBY"), io_mod.getenv("INBY_SESSION"))) {
            debug_trace.logf("inby", "disabled", .{});
            return;
        }
        self.alloc = alloc;
        self.thread = std.Thread.spawn(.{}, workerMain, .{self}) catch |err| {
            debug_trace.logf("inby", "worker_spawn_failed err={s}", .{@errorName(err)});
            return;
        };
        self.enabled = true;
        debug_trace.logf("inby", "enabled", .{});
    }

    pub fn report(self: *Client, state: State) void {
        if (!self.enabled) return;
        const io = io_mod.getIo();
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.desired = state;
        self.desired_generation += 1;
    }

    /// Returns the tab to its resting look, then stops the worker.
    pub fn deinit(self: *Client) void {
        if (!self.enabled) return;
        self.report(.idle);
        self.stop_requested.store(true, .release);
        if (self.thread) |thread| thread.join();
        self.thread = null;
        self.freeResting();
        self.enabled = false;
    }

    fn workerMain(self: *Client) void {
        while (true) {
            const stopping = self.stop_requested.load(.acquire);
            self.applyPending();
            if (stopping) return;
            io_mod.sleep(40 * std.time.ns_per_ms);
        }
    }

    fn applyPending(self: *Client) void {
        const io = io_mod.getIo();
        self.mutex.lockUncancelable(io);
        const target = self.desired;
        const generation = self.desired_generation;
        self.mutex.unlock(io);
        if (generation == self.applied_generation) return;
        self.applied_generation = generation;
        if (target == self.applied) return;
        self.transition(target) catch |err| {
            debug_trace.logf("inby", "report_failed state={s} err={s}", .{ @tagName(target), @errorName(err) });
        };
        self.applied = target;
    }

    fn transition(self: *Client, target: State) !void {
        const alloc = self.alloc;
        const raw = try runInby(alloc, &.{ "inby", "tab", "--get" });
        defer alloc.free(raw);
        const parsed = try std.json.parseFromSlice(TabSnapshot, alloc, raw, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        const current = parsed.value;

        if (self.applied == .idle and target != .idle) {
            self.freeResting();
            self.resting_icon = try alloc.dupe(u8, current.icon);
            self.resting_color = try alloc.dupe(u8, current.color);
            self.has_resting = true;
        }
        const resting: ?Look = if (self.has_resting) .{ .icon = self.resting_icon, .color = self.resting_color } else null;
        const action = plan(target, current, resting, self.last_set);
        try self.apply(action);

        self.last_set = switch (target) {
            .working => working_look,
            .needs_you => needs_you_look,
            .idle => null,
        };
        if (target == .idle) self.freeResting();
        debug_trace.logf("inby", "reported state={s}", .{@tagName(target)});
    }

    fn apply(self: *Client, action: Action) !void {
        const alloc = self.alloc;
        if (action.icon != null or action.color != null) {
            var argv: std.ArrayList([]const u8) = .empty;
            defer argv.deinit(alloc);
            var owned: std.ArrayList([]u8) = .empty;
            defer {
                for (owned.items) |item| alloc.free(item);
                owned.deinit(alloc);
            }
            try argv.appendSlice(alloc, &.{ "inby", "tab" });
            if (action.icon) |icon| {
                const flag = try std.fmt.allocPrint(alloc, "--icon={s}", .{icon});
                try owned.append(alloc, flag);
                try argv.append(alloc, flag);
            }
            if (action.color) |color| {
                const flag = try std.fmt.allocPrint(alloc, "--color={s}", .{color});
                try owned.append(alloc, flag);
                try argv.append(alloc, flag);
            }
            alloc.free(try runInby(alloc, argv.items));
        }
        if (action.pin) |pin| {
            alloc.free(try runInby(alloc, &.{ "inby", "pin", @tagName(pin) }));
        }
    }

    fn freeResting(self: *Client) void {
        if (!self.has_resting) return;
        self.alloc.free(self.resting_icon);
        self.alloc.free(self.resting_color);
        self.resting_icon = &.{};
        self.resting_color = &.{};
        self.has_resting = false;
    }
};

/// Runs an `inby` command and returns its stdout (caller frees).
fn runInby(alloc: Allocator, argv: []const []const u8) ![]u8 {
    const result = try std.process.run(alloc, io_mod.getIo(), .{ .argv = argv });
    defer alloc.free(result.stderr);
    errdefer alloc.free(result.stdout);
    if (result.term != .exited or result.term.exited != 0) return error.InbyCommandFailed;
    return result.stdout;
}

test "working and needs-you set the state look and pin active" {
    const idle_tab: TabSnapshot = .{ .icon = "person.2.fill", .color = "mint", .iconOrigin = "agent", .colorOrigin = "agent" };
    const working = plan(.working, idle_tab, null, null);
    try std.testing.expectEqualStrings("hammer.fill", working.icon.?);
    try std.testing.expectEqualStrings("aqua", working.color.?);
    try std.testing.expectEqual(Pin.active, working.pin.?);
    const blocked = plan(.needs_you, idle_tab, null, working_look);
    try std.testing.expectEqualStrings("hand.raised.fill", blocked.icon.?);
    try std.testing.expectEqualStrings("amber", blocked.color.?);
}

test "fields the human set are never changed" {
    const tab: TabSnapshot = .{ .icon = "star", .color = "rose", .iconOrigin = "user", .colorOrigin = "agent" };
    const action = plan(.working, tab, null, null);
    try std.testing.expectEqual(@as(?[]const u8, null), action.icon);
    try std.testing.expectEqualStrings("aqua", action.color.?);
}

test "idle restores the resting look only where fx's look is still showing" {
    const resting: Look = .{ .icon = "person.2.fill", .color = "mint" };
    const untouched: TabSnapshot = .{ .icon = "hammer.fill", .color = "aqua", .iconOrigin = "agent", .colorOrigin = "agent" };
    const restored = plan(.idle, untouched, resting, working_look);
    try std.testing.expectEqualStrings("person.2.fill", restored.icon.?);
    try std.testing.expectEqualStrings("mint", restored.color.?);
    try std.testing.expectEqual(Pin.auto, restored.pin.?);

    // The agent switched to an at-risk look mid-turn: keep it.
    const agent_changed: TabSnapshot = .{ .icon = "exclamationmark.triangle.fill", .color = "coral", .iconOrigin = "agent", .colorOrigin = "agent" };
    const kept = plan(.idle, agent_changed, resting, working_look);
    try std.testing.expectEqual(@as(?[]const u8, null), kept.icon);
    try std.testing.expectEqual(@as(?[]const u8, null), kept.color);
}

test "idle without a captured resting look only unpins" {
    const action = plan(.idle, .{}, null, null);
    try std.testing.expectEqual(@as(?[]const u8, null), action.icon);
    try std.testing.expectEqual(Pin.auto, action.pin.?);
}

test "enable requires an Inby session and respects FX_INBY" {
    try std.testing.expect(shouldEnable(null, "5D8F"));
    try std.testing.expect(!shouldEnable(null, null));
    try std.testing.expect(!shouldEnable(null, ""));
    try std.testing.expect(!shouldEnable("0", "5D8F"));
    try std.testing.expect(!shouldEnable("false", "5D8F"));
}

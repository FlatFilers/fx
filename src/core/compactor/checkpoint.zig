//! The saved form of a context compaction checkpoint.
//!
//! A checkpoint's `summary` string holds `marker` followed by the JSON of a
//! `Payload`: one summary of the earlier conversation, the newest compacted
//! turns with their user messages and final replies exact, and how many turns
//! and tool calls are saved word for word (M1 through M<turn_count>, T1
//! through T<tool_count>). Every session codec keeps treating the string as
//! opaque text. Checkpoints written before this format hold model-visible text
//! directly and keep working.
//!
//! This file is the one place that tells the formats apart and renders what
//! the model reads.

const std = @import("std");
const types = @import("../shared/types.zig");
const debug_trace = @import("../shared/debug_trace.zig");

const Allocator = std.mem.Allocator;

pub const marker = "fx-compactor-v1\n";

/// One compacted turn, shown word for word except for the summary of its
/// work.
pub const Turn = struct {
    /// Saved word for word as M<number>. Zero only for user messages carried
    /// over from an older checkpoint format, which have no saved turn.
    number: usize = 0,
    /// The message that started the turn, then any the user added while it
    /// ran. Exact.
    users: []const []const u8 = &.{},
    /// What the assistant did before its final reply, summarized.
    work: []const u8 = "",
    /// The assistant's final reply, exact. Empty when the turn ended without
    /// one.
    final: []const u8 = "",
    /// The turn's tool calls are T<first_tool> through T<last_tool>; zero when
    /// it made none.
    first_tool: usize = 0,
    last_tool: usize = 0,
};

/// The turn that was still running when it was compacted. Its first user
/// message stays in the conversation right after the checkpoint.
pub const OpenTurn = struct {
    /// Messages the user added while it ran, exact.
    users: []const []const u8 = &.{},
    /// What the assistant has done so far, summarized.
    work: []const u8 = "",
    /// Its exact text so far, kept so the saved turn is complete once the
    /// turn ends.
    text: []const u8 = "",
    first_tool: usize = 0,
    last_tool: usize = 0,
};

pub const Payload = struct {
    /// One summary of everything before `turns`.
    earlier: []const u8 = "",
    /// The newest compacted turns, oldest first.
    turns: []const Turn = &.{},
    open: ?OpenTurn = null,
    /// Turns and tool calls are numbered through these counts.
    turn_count: usize = 0,
    tool_count: usize = 0,
    /// False when the session is not saved, so no turn or tool call can be
    /// opened later.
    saved: bool = true,
};

/// Returns the checkpoint string for `payload`. Caller owns it.
pub fn encode(alloc: Allocator, payload: Payload) Allocator.Error![]u8 {
    var out: std.Io.Writer.Allocating = .init(alloc);
    errdefer out.deinit();
    out.writer.writeAll(marker) catch return error.OutOfMemory;
    std.json.Stringify.value(payload, .{}, &out.writer) catch return error.OutOfMemory;
    return out.toOwnedSlice() catch error.OutOfMemory;
}

pub fn isPayload(summary: []const u8) bool {
    return std.mem.startsWith(u8, summary, marker);
}

/// True when the checkpoint stands in for everything before it, so earlier
/// raw turns and older checkpoints are no longer part of the context.
pub fn replacesPriorContext(summary: []const u8) bool {
    return isPayload(summary) or std.mem.startsWith(u8, summary, types.context_handoff_open);
}

/// Checkpoints written by the previous compactor name a state file in the
/// session's tool-results store that holds the user messages word for word
/// and the summary, as `fx-compaction-state-v1 <handle> <bytes> <sha256>`.
pub const LegacyStateRef = struct {
    handle: []const u8,
    bytes: usize,
    sha256: [32]u8,
};

const legacy_state_tag = "fx-compaction-state-v1 ";

/// The state file named by an older checkpoint, if it names one.
pub fn legacyStateRef(summary: []const u8) ?LegacyStateRef {
    if (isPayload(summary)) return null;
    const start = (std.mem.find(u8, summary, legacy_state_tag) orelse return null) + legacy_state_tag.len;
    const line_end = std.mem.findScalarPos(u8, summary, start, '\n') orelse summary.len;
    var fields = std.mem.tokenizeScalar(u8, summary[start..line_end], ' ');
    const handle = fields.next() orelse return null;
    const bytes = std.fmt.parseUnsigned(usize, fields.next() orelse return null, 10) catch return null;
    const digest_hex = fields.next() orelse return null;
    if (digest_hex.len != 64) return null;
    var ref: LegacyStateRef = .{ .handle = handle, .bytes = bytes, .sha256 = undefined };
    _ = std.fmt.hexToBytes(&ref.sha256, digest_hex) catch return null;
    return ref;
}

/// Reads the previous compactor's state file. Its user messages become
/// unnumbered turns, since that compactor saved no turns or tool calls.
/// Returns null when the bytes do not match the checkpoint or do not parse.
/// `arena` owns the result.
pub fn parseLegacyState(arena: Allocator, ref: LegacyStateRef, bytes: []const u8) Allocator.Error!?Payload {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    if (bytes.len != ref.bytes or !std.mem.eql(u8, &digest, &ref.sha256)) {
        debug_trace.logf("context_compaction", "earlier state file does not match its checkpoint handle={s} bytes={d}", .{ ref.handle, bytes.len });
        return null;
    }
    const State = struct { summary: []const u8, users: []const []const u8 };
    const state = std.json.parseFromSliceLeaky(State, arena, bytes, .{
        .allocate = .alloc_always,
        .ignore_unknown_fields = true,
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            debug_trace.logf("context_compaction", "earlier state file unreadable handle={s} err={s}", .{ ref.handle, @errorName(err) });
            return null;
        },
    };
    const turns = try arena.alloc(Turn, state.users.len);
    for (turns, 0..) |*turn, index| turn.* = .{ .users = state.users[index .. index + 1] };
    return .{ .earlier = state.summary, .turns = turns };
}

/// Parses a payload checkpoint. Returns null for older formats and for a
/// damaged payload, which is traced. `arena` owns everything returned.
pub fn parse(arena: Allocator, summary: []const u8) Allocator.Error!?Payload {
    if (!isPayload(summary)) return null;
    return std.json.parseFromSliceLeaky(Payload, arena, summary[marker.len..], .{
        .allocate = .alloc_always,
        .ignore_unknown_fields = true,
    }) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => {
            debug_trace.logf("context_compaction", "checkpoint payload unreadable bytes={d} err={s}; using its raw text", .{ summary.len, @errorName(err) });
            return null;
        },
    };
}

/// What the model reads for a payload checkpoint. Caller owns it.
pub fn render(alloc: Allocator, payload: Payload) Allocator.Error![]u8 {
    var text: std.ArrayList(u8) = .empty;
    errdefer text.deinit(alloc);
    try text.appendSlice(alloc,
        \\<compacted_conversation>
        \\This is the earlier part of this conversation, compacted. The user's messages and the assistant's final replies shown here are exact. The rest of the assistant's work is summarized, and tool calls were removed.
        \\
        \\
    );
    if (payload.earlier.len > 0) try text.print(alloc, "Earlier summary:\n{s}\n\n", .{payload.earlier});
    for (payload.turns) |turn| try renderTurn(alloc, &text, turn);
    if (payload.open) |open| {
        if (open.work.len > 0) try text.print(alloc, "The turn still in progress, whose user message follows this, summary of its work so far:\n{s}\n\n", .{open.work});
        for (open.users) |user| try text.print(alloc, "User, added while the assistant worked on the turn in progress:\n{s}\n\n", .{user});
        if (open.first_tool > 0) {
            try text.appendSlice(alloc, "Its tools so far: ");
            try appendRange(alloc, &text, 'T', open.first_tool, open.last_tool);
            try text.appendSlice(alloc, "\n\n");
        }
    }
    try appendSavedLine(alloc, &text, payload);
    try text.appendSlice(alloc, "</compacted_conversation>\n");
    return text.toOwnedSlice(alloc);
}

fn renderTurn(alloc: Allocator, text: *std.ArrayList(u8), turn: Turn) Allocator.Error!void {
    for (turn.users, 0..) |user, index| {
        try appendLabel(alloc, text, "User", turn.number);
        if (index > 0) try text.appendSlice(alloc, ", added while the assistant worked");
        try text.print(alloc, ":\n{s}\n\n", .{user});
    }
    if (turn.work.len > 0) {
        try appendLabel(alloc, text, "Assistant", turn.number);
        try text.print(alloc, ", summary of its work:\n{s}\n\n", .{turn.work});
    }
    if (turn.final.len > 0) {
        try appendLabel(alloc, text, "Assistant", turn.number);
        try text.print(alloc, ", final reply:\n{s}\n\n", .{turn.final});
    }
    if (turn.first_tool > 0) {
        try text.appendSlice(alloc, "Tools: ");
        try appendRange(alloc, text, 'T', turn.first_tool, turn.last_tool);
        try text.appendSlice(alloc, "\n\n");
    }
}

fn appendLabel(alloc: Allocator, text: *std.ArrayList(u8), who: []const u8, number: usize) Allocator.Error!void {
    try text.appendSlice(alloc, who);
    if (number > 0) try text.print(alloc, " {d}", .{number});
}

/// `T3` or `T3–T9`.
fn appendRange(alloc: Allocator, text: *std.ArrayList(u8), prefix: u8, first: usize, last: usize) Allocator.Error!void {
    try text.print(alloc, "{c}{d}", .{ prefix, first });
    if (last > first) try text.print(alloc, "–{c}{d}", .{ prefix, last });
}

fn appendSavedLine(alloc: Allocator, text: *std.ArrayList(u8), payload: Payload) Allocator.Error!void {
    if (!payload.saved or (payload.turn_count == 0 and payload.tool_count == 0)) return;
    try text.appendSlice(alloc, "Saved word for word: ");
    if (payload.turn_count > 0) {
        try text.appendSlice(alloc, if (payload.turn_count == 1) "turn " else "turns ");
        try appendRange(alloc, text, 'M', 1, payload.turn_count);
        if (payload.tool_count > 0) try text.appendSlice(alloc, " and ");
    }
    if (payload.tool_count > 0) {
        try text.appendSlice(alloc, if (payload.tool_count == 1) "tool call " else "tool calls ");
        try appendRange(alloc, text, 'T', 1, payload.tool_count);
    }
    try text.appendSlice(alloc, ". Search them by text, or open one by its ID (like ");
    if (payload.turn_count > 0) try text.print(alloc, "M{d}", .{payload.turn_count});
    if (payload.turn_count > 0 and payload.tool_count > 0) try text.appendSlice(alloc, " or ");
    if (payload.tool_count > 0) try text.print(alloc, "T{d}", .{payload.tool_count});
    try text.appendSlice(alloc, "), with read_tool_result.\n");
}

/// Model-visible text for a payload checkpoint, or null for older formats.
/// A damaged payload yields its raw text so the session stays usable.
pub fn modelText(alloc: Allocator, summary: []const u8) Allocator.Error!?[]u8 {
    if (!isPayload(summary)) return null;
    var arena_state = std.heap.ArenaAllocator.init(alloc);
    defer arena_state.deinit();
    const payload = try parse(arena_state.allocator(), summary) orelse
        return try alloc.dupe(u8, summary[marker.len..]);
    return try render(alloc, payload);
}

const testing = std.testing;

const sample_payload: Payload = .{
    .earlier = "The assistant set up the repo and verified the build [M1].",
    .turns = &.{
        .{
            .number = 2,
            .users = &.{ "Fix the build.\nIt fails on main.", "also check the tests" },
            .work = "Found a missing semicolon (T4), fixed it, and ran the tests (T5).",
            .final = "Fixed. The build and all 12 tests pass.",
            .first_tool = 4,
            .last_tool = 5,
        },
        .{ .number = 3, .users = &.{"thanks"}, .final = "You're welcome." },
    },
    .turn_count = 3,
    .tool_count = 5,
};

test "payload checkpoints round trip and render exact users and final replies" {
    const alloc = testing.allocator;
    const saved = try encode(alloc, sample_payload);
    defer alloc.free(saved);
    try testing.expect(isPayload(saved));
    try testing.expect(replacesPriorContext(saved));

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const back = (try parse(arena.allocator(), saved)).?;
    try testing.expectEqual(@as(usize, 2), back.turns.len);
    try testing.expectEqualStrings("also check the tests", back.turns[0].users[1]);
    try testing.expectEqual(@as(usize, 5), back.tool_count);
    try testing.expect(back.open == null);

    const text = (try modelText(alloc, saved)).?;
    defer alloc.free(text);
    const expected =
        \\Earlier summary:
        \\The assistant set up the repo and verified the build [M1].
        \\
        \\User 2:
        \\Fix the build.
        \\It fails on main.
        \\
        \\User 2, added while the assistant worked:
        \\also check the tests
        \\
        \\Assistant 2, summary of its work:
        \\Found a missing semicolon (T4), fixed it, and ran the tests (T5).
        \\
        \\Assistant 2, final reply:
        \\Fixed. The build and all 12 tests pass.
        \\
        \\Tools: T4–T5
        \\
        \\User 3:
        \\thanks
        \\
        \\Assistant 3, final reply:
        \\You're welcome.
        \\
        \\Saved word for word: turns M1–M3 and tool calls T1–T5. Search them by text, or open one by its ID (like M3 or T5), with read_tool_result.
        \\</compacted_conversation>
        \\
    ;
    try testing.expect(std.mem.endsWith(u8, text, expected));
    try testing.expect(std.mem.startsWith(u8, text, "<compacted_conversation>\n"));
    try testing.expect(std.mem.find(u8, text, marker) == null);
    try testing.expect(std.mem.find(u8, text, "\"turn_count\"") == null);
}

test "the turn in progress shows its summary, added messages and tools" {
    const alloc = testing.allocator;
    const text = try render(alloc, .{
        .turns = &.{},
        .open = .{ .users = &.{"use the staging db"}, .work = "Ran the migration dry run (T1).", .text = "exact text", .first_tool = 1, .last_tool = 1 },
        .tool_count = 1,
    });
    defer alloc.free(text);
    try testing.expect(std.mem.find(u8, text, "summary of its work so far:\nRan the migration dry run (T1).\n") != null);
    try testing.expect(std.mem.find(u8, text, "turn in progress:\nuse the staging db\n") != null);
    try testing.expect(std.mem.find(u8, text, "Its tools so far: T1\n") != null);
    try testing.expect(std.mem.find(u8, text, "exact text") == null);
    try testing.expect(std.mem.find(u8, text, "Saved word for word: tool call T1. Search them by text, or open one by its ID (like T1)") != null);
}

test "the saved line names only what can be opened" {
    const alloc = testing.allocator;
    const none = try render(alloc, .{ .turns = &.{.{ .number = 1, .users = &.{"hi"} }} });
    defer alloc.free(none);
    try testing.expect(std.mem.find(u8, none, "Saved") == null);
    const unsaved = try render(alloc, .{ .turn_count = 4, .tool_count = 9, .saved = false });
    defer alloc.free(unsaved);
    try testing.expect(std.mem.find(u8, unsaved, "read_tool_result") == null);
    const one = try render(alloc, .{ .turn_count = 1 });
    defer alloc.free(one);
    try testing.expect(std.mem.find(u8, one, "Saved word for word: turn M1. Search them by text, or open one by its ID (like M1)") != null);
}

test "older checkpoints are recognized but not parsed" {
    const alloc = testing.allocator;
    const handoff = types.context_handoff_open ++ "\n## Conversation summary\n> earlier\n" ++ types.context_handoff_close;
    try testing.expect(replacesPriorContext(handoff));
    try testing.expect(!isPayload(handoff));
    try testing.expect(try modelText(alloc, handoff) == null);
    try testing.expect(!replacesPriorContext("Budget trimmed summary"));
}

test "a damaged payload falls back to its raw text" {
    const alloc = testing.allocator;
    const damaged = marker ++ "{\"turns\":[\"cut off";
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    try testing.expect(try parse(arena.allocator(), damaged) == null);
    const text = (try modelText(alloc, damaged)).?;
    defer alloc.free(text);
    try testing.expectEqualStrings("{\"turns\":[\"cut off", text);
}

test "older checkpoints yield their exact users from the named state file" {
    const alloc = testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(alloc);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const state = "{\"version\":1,\"summary\":\"Earlier work.\",\"users\":[\"okay so you saying that if 2 GB exeeds then what happens ? \",\"yes\"],\"archives\":[]}";
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(state, &digest, .{});
    const summary = try std.fmt.allocPrint(arena, "{s}\n## Conversation summary\n> fx-compaction-state-v1 result-state-1-2.txt {d} {x}\n> Task state:\n", .{ types.context_handoff_open, state.len, digest });
    const ref = legacyStateRef(summary).?;
    try testing.expectEqualStrings("result-state-1-2.txt", ref.handle);
    const payload = (try parseLegacyState(arena, ref, state)).?;
    try testing.expectEqual(@as(usize, 2), payload.turns.len);
    try testing.expectEqualStrings("okay so you saying that if 2 GB exeeds then what happens ? ", payload.turns[0].users[0]);
    try testing.expectEqual(@as(usize, 0), payload.turns[0].number);
    try testing.expectEqualStrings("yes", payload.turns[1].users[0]);
    try testing.expectEqualStrings("Earlier work.", payload.earlier);
    try testing.expectEqual(@as(usize, 0), payload.tool_count);
    try testing.expectEqual(@as(usize, 0), payload.turn_count);
    // Bytes that do not match the checkpoint are not trusted.
    try testing.expect(try parseLegacyState(arena, ref, state[0 .. state.len - 1]) == null);
    try testing.expect(legacyStateRef(types.context_handoff_open ++ "no state here") == null);
    try testing.expect(legacyStateRef(marker ++ "{}") == null);
}

//! The summary step of fx-compactor: turns to compact in, compacted
//! conversation out.
//!
//! In: the raw turns to compact (user messages, assistant messages, tool calls
//! and tool results, all unchanged), the previous compaction if any, and the
//! model the conversation uses.
//! Out: the compacted conversation.
//!
//! - The newest compacted turns keep their user messages and final replies
//!   exactly. The conversation's own model summarizes the rest of each turn.
//! - Older turns, including the ones the previous compaction showed exactly,
//!   fold into one earlier summary.
//! - Every turn is saved word for word as M1, M2, ... and every tool call with
//!   its result as T1, T2, ..., so the agent can search or open them.
//!
//! This file does no I/O of its own. `compactor.zig` passes in the model and
//! the store, so every compaction and every test runs this same code.

const std = @import("std");
const checkpoint = @import("checkpoint.zig");
const compacted_records = @import("records.zig");
const token_estimate = @import("../shared/token_estimate.zig");

const Allocator = std.mem.Allocator;

pub const ToolCall = struct {
    id: []const u8,
    name: []const u8,
    arguments: []const u8,
};

pub const ToolResult = struct {
    call_id: []const u8,
    name: []const u8,
    output: []const u8,
};

/// A compacted conversation. Pass it back as `Request.earlier` of the next
/// compaction so numbering continues and the shown turns fold correctly.
pub const Compacted = checkpoint.Payload;

/// One item of a turn after its first user message, oldest first.
pub const Item = union(enum) {
    /// A message the user added while the turn ran.
    user: []const u8,
    assistant: []const u8,
    /// Text fx itself added, such as permission feedback. Never a user
    /// message.
    note: []const u8,
    tool_call: ToolCall,
    tool_result: ToolResult,
};

pub const Turn = struct {
    /// The message that started the turn.
    user: []const u8,
    items: []const Item = &.{},
};

pub const Request = struct {
    /// The model the conversation uses. It writes the summary.
    model: []const u8,
    /// The previous compaction. When it left a turn in progress, `turns[0]`
    /// continues that turn.
    earlier: ?Compacted = null,
    /// Raw turns to compact, oldest first.
    turns: []const Turn,
    /// The last turn is still running: its first user message stays in the
    /// conversation after the checkpoint.
    last_turn_open: bool = false,
    /// Estimated tokens the newest turns may use for their exact user messages
    /// and final replies. Older turns fold into the earlier summary.
    exact_tokens: usize = std.math.maxInt(usize),
};

/// What the model is asked. A `Model` sends it as one system message and one
/// user message.
pub const Prompt = struct {
    model: []const u8,
    system: []const u8,
    user: []const u8,
};

pub const ModelError = error{
    /// The provider call failed. The model implementation keeps the details.
    ModelFailed,
    /// The model stopped before finishing, for example at its output limit.
    SummaryIncomplete,
    Cancelled,
    OutOfMemory,
};

pub const Model = struct {
    context: *anyopaque,
    /// Returns the summary text, allocated with `alloc`.
    summarize_fn: *const fn (context: *anyopaque, alloc: Allocator, prompt: Prompt) ModelError![]u8,
};

pub const StoreError = error{ StoreFailed, OutOfMemory };

pub const Store = struct {
    context: *anyopaque,
    /// Saves one record unchanged: a tool call with its result as T<number>,
    /// or a whole turn as M<number>. A number is never saved twice with
    /// different content.
    save_fn: *const fn (context: *anyopaque, kind: compacted_records.Kind, number: usize, content: []const u8) StoreError!void,
};

pub const Error = ModelError || StoreError || error{ NothingToCompact, EmptySummary };

/// Owns everything it points to. Call `deinit` when done.
pub const Result = struct {
    arena: *std.heap.ArenaAllocator,
    /// Keep this and pass it into the next compaction.
    compacted: Compacted,
    /// Give this to the agent in place of the compacted conversation.
    text: []const u8,

    pub fn deinit(self: *Result) void {
        const child = self.arena.child_allocator;
        self.arena.deinit();
        child.destroy(self.arena);
        self.* = undefined;
    }
};

pub const system_prompt =
    "You summarize the assistant's side of a conversation between a user and an AI coding assistant. " ++
    "Another assistant will use your summary to continue the work.";

const earlier_heading = "Earlier";
const open_heading = "Turn in progress";

/// The summary step. `compactor.compact` calls it once per compaction.
/// Blocks while the model writes the summary. `request` is borrowed and not
/// modified; the returned `Result` owns copies of everything it keeps.
/// Without a store nothing is saved, so the summary carries what tool calls
/// showed.
pub fn compact(alloc: Allocator, request: Request, model: Model, store: ?Store) Error!Result {
    if (request.turns.len == 0) return error.NothingToCompact;
    const earlier: Compacted = request.earlier orelse .{};

    const arena = try alloc.create(std.heap.ArenaAllocator);
    errdefer alloc.destroy(arena);
    arena.* = .init(alloc);
    errdefer arena.deinit();
    const out = arena.allocator();

    var scratch_state: std.heap.ArenaAllocator = .init(alloc);
    defer scratch_state.deinit();
    const scratch = scratch_state.allocator();

    const open_index: ?usize = if (request.last_turn_open) request.turns.len - 1 else null;
    const complete_end = open_index orelse request.turns.len;
    var next_tool = earlier.tool_count + 1;
    var next_turn = earlier.turn_count + 1;
    const turns = try scratch.alloc(Prepared, request.turns.len);
    for (request.turns, turns, 0..) |turn, *slot, index| {
        const continued = if (index == 0) earlier.open else null;
        const is_open = open_index == index;
        slot.* = try prepare(scratch, turn, continued, is_open, &next_tool);
        if (!is_open) {
            slot.number = next_turn;
            next_turn += 1;
        }
    }

    // Within the budget, the newest complete turns keep their exact user
    // messages first, then as many final replies as still fit, newest first.
    // Older turns fold into the earlier summary.
    var chunk_start = complete_end;
    var used: usize = 0;
    while (chunk_start > 0) {
        const cost = tokens(turns[chunk_start - 1].users);
        if (used +| cost > request.exact_tokens) break;
        used += cost;
        chunk_start -= 1;
    }
    var newest = complete_end;
    while (newest > chunk_start) {
        newest -= 1;
        const cost = tokens(&.{turns[newest].final});
        if (used +| cost > request.exact_tokens) continue;
        used += cost;
        turns[newest].show_final = true;
    }
    const plan: Plan = .{
        .earlier = earlier,
        .turns = turns,
        .chunk_start = chunk_start,
        .complete_end = complete_end,
        .saved = store != null and earlier.saved,
    };

    if (store) |saved| for (turns) |turn| {
        for (turn.tools) |tool| try saved.save_fn(saved.context, .tool, tool.number, try toolFile(scratch, tool));
        if (turn.number > 0) try saved.save_fn(saved.context, .turn, turn.number, try turnFile(scratch, turn));
    };

    var sections: Sections = .{};
    if (plan.needsModel()) {
        const written = try model.summarize_fn(model.context, out, .{
            .model = request.model,
            .system = system_prompt,
            .user = try renderTranscript(scratch, plan),
        });
        const summary = std.mem.trim(u8, written, " \t\r\n");
        if (summary.len == 0) return error.EmptySummary;
        sections = try parseSections(out, summary, plan);
    }

    const shown = try out.alloc(checkpoint.Turn, complete_end - chunk_start);
    for (shown, turns[chunk_start..complete_end], 0..) |*slot, turn, index| slot.* = .{
        .number = turn.number,
        .users = try dupeAll(out, turn.users),
        .work = if (index < sections.turn_work.len) sections.turn_work[index] else "",
        .final = if (turn.show_final) try out.dupe(u8, turn.final) else "",
        .first_tool = turn.first_tool,
        .last_tool = turn.last_tool,
    };
    const compacted: Compacted = .{
        .earlier = if (plan.foldsEarlier())
            sections.earlier
        else if (sections.earlier.len == 0)
            try out.dupe(u8, earlier.earlier)
        else
            // Text the model wrote outside any section asked for.
            std.mem.trim(u8, try std.mem.concat(out, u8, &.{ earlier.earlier, "\n\n", sections.earlier }), " \t\r\n"),
        .turns = shown,
        .open = if (open_index) |index| .{
            .users = try dupeAll(out, turns[index].users),
            .work = sections.open_work,
            .text = try out.dupe(u8, turns[index].text),
            .first_tool = turns[index].first_tool,
            .last_tool = turns[index].last_tool,
        } else null,
        .turn_count = next_turn - 1,
        .tool_count = next_tool - 1,
        .saved = plan.saved,
    };
    return .{
        .arena = arena,
        .compacted = compacted,
        .text = try checkpoint.render(out, compacted),
    };
}

/// One turn ready to compact.
const Prepared = struct {
    source: Turn,
    /// M<number>; zero for the turn still in progress.
    number: usize = 0,
    /// The turn's own item tool numbers, zero for items that are not tools.
    tool_numbers: []const usize,
    tools: []const PendingTool,
    first_tool: usize,
    last_tool: usize,
    /// Exact user messages: the first, then any added while it ran.
    users: []const []const u8,
    /// The exact final reply, or empty.
    final: []const u8,
    final_index: ?usize,
    /// There is something to summarize besides the exact messages and reply.
    has_work: bool,
    /// The final reply fits the budget and stays word for word.
    show_final: bool = false,
    /// The earlier part of this turn, from the previous compaction.
    continued: ?checkpoint.OpenTurn,
    /// Exact text of the turn after its first user message, tool calls
    /// shown by ID.
    text: []const u8,

    /// Something in the turn is not shown word for word, so the model
    /// summarizes it.
    fn summarized(self: Prepared) bool {
        return self.has_work or (self.final.len > 0 and !self.show_final);
    }
};

const PendingTool = struct {
    number: usize,
    name: []const u8,
    call: ?ToolCall = null,
    result: ?ToolResult = null,
};

fn prepare(arena: Allocator, turn: Turn, continued: ?checkpoint.OpenTurn, is_open: bool, next_tool: *usize) Allocator.Error!Prepared {
    // Pair every call with its result so both are saved as one numbered tool.
    const numbers = try arena.alloc(usize, turn.items.len);
    @memset(numbers, 0);
    var tools: std.ArrayList(PendingTool) = .empty;
    var open_calls: std.StringHashMapUnmanaged(usize) = .empty;
    for (turn.items, 0..) |item, index| switch (item) {
        .tool_call => |call| {
            numbers[index] = next_tool.*;
            try open_calls.put(arena, call.id, tools.items.len);
            try tools.append(arena, .{ .number = next_tool.*, .name = call.name, .call = call });
            next_tool.* += 1;
        },
        .tool_result => |result| {
            if (open_calls.fetchRemove(result.call_id)) |open| {
                tools.items[open.value].result = result;
                numbers[index] = tools.items[open.value].number;
            } else {
                numbers[index] = next_tool.*;
                try tools.append(arena, .{ .number = next_tool.*, .name = result.name, .result = result });
                next_tool.* += 1;
            }
        },
        else => {},
    };

    // The final reply is the last assistant message with no tool call after
    // it. A turn still in progress has none yet.
    var final_index: ?usize = null;
    if (!is_open) for (turn.items, 0..) |item, index| switch (item) {
        .assistant => |text| {
            if (text.len > 0) final_index = index;
        },
        .tool_call => final_index = null,
        else => {},
    };

    var users: std.ArrayList([]const u8) = .empty;
    if (!is_open) try users.append(arena, turn.user);
    if (continued) |earlier| try users.appendSlice(arena, earlier.users);
    var has_work = if (continued) |earlier| earlier.work.len > 0 or earlier.text.len > 0 else false;
    for (turn.items, 0..) |item, index| switch (item) {
        .user => |text| try users.append(arena, text),
        .assistant => |text| if (text.len > 0 and index != final_index) {
            has_work = true;
        },
        else => has_work = true,
    };

    var text: std.ArrayList(u8) = .empty;
    if (continued) |earlier| try text.appendSlice(arena, earlier.text);
    for (turn.items, numbers, 0..) |item, number, index| switch (item) {
        .user => |message| try text.print(arena, "User, added while the assistant worked:\n{s}\n\n", .{message}),
        .assistant => |message| if (message.len > 0) {
            try text.print(arena, "{s}:\n{s}\n\n", .{ if (index == final_index) "Assistant, final reply" else "Assistant", message });
        },
        .note => |message| try text.print(arena, "From fx, not the user:\n{s}\n\n", .{message}),
        .tool_call => |call| {
            const index_line = try indexLine(arena, call.arguments);
            try text.print(arena, "[T{d} {s}{s}{s}]\n\n", .{ number, call.name, if (index_line.len > 0) ": " else "", index_line });
        },
        .tool_result => |result| if (!hasCall(tools.items, number)) {
            try text.print(arena, "[T{d} {s}, result only]\n\n", .{ number, result.name });
        },
    };

    const first_own = if (tools.items.len > 0) tools.items[0].number else 0;
    const last_own = if (tools.items.len > 0) tools.items[tools.items.len - 1].number else 0;
    const first_earlier = if (continued) |earlier| earlier.first_tool else 0;
    return .{
        .source = turn,
        .tool_numbers = numbers,
        .tools = tools.items,
        .first_tool = if (first_earlier > 0) first_earlier else first_own,
        .last_tool = if (last_own > 0) last_own else if (continued) |earlier| earlier.last_tool else 0,
        .users = users.items,
        .final = if (final_index) |index| turn.items[index].assistant else "",
        .final_index = final_index,
        .has_work = has_work,
        .continued = continued,
        .text = text.items,
    };
}

fn hasCall(tools: []const PendingTool, number: usize) bool {
    for (tools) |tool| if (tool.number == number) return tool.call != null;
    return false;
}

fn tokens(texts: []const []const u8) usize {
    var estimator: token_estimate.StreamingEstimator = .{};
    for (texts) |text| {
        estimator.consume(text);
        estimator.consume(" ");
    }
    return std.math.cast(usize, estimator.estimate()) orelse std.math.maxInt(usize);
}

fn dupeAll(alloc: Allocator, texts: []const []const u8) Allocator.Error![]const []const u8 {
    const copies = try alloc.alloc([]const u8, texts.len);
    for (copies, texts) |*copy, text| copy.* = try alloc.dupe(u8, text);
    return copies;
}

/// Which turns are shown exactly and which fold into the earlier summary.
const Plan = struct {
    earlier: Compacted,
    turns: []const Prepared,
    /// Complete turns before this index fold into the earlier summary.
    chunk_start: usize,
    /// Turns from `chunk_start` to here are shown exactly; a turn after it is
    /// still in progress.
    complete_end: usize,
    saved: bool,

    /// Something new joins the earlier summary, so the model rewrites it.
    /// Otherwise the earlier summary carries over unchanged.
    fn foldsEarlier(self: Plan) bool {
        return self.earlier.turns.len > 0 or self.chunk_start > 0;
    }

    fn hasOpenTurn(self: Plan) bool {
        return self.complete_end < self.turns.len;
    }

    fn needsModel(self: Plan) bool {
        return self.sectionCount() > 0;
    }

    /// How many sections the model is asked to write.
    fn sectionCount(self: Plan) usize {
        var count: usize = 0;
        if (self.foldsEarlier()) count += 1;
        if (self.hasOpenTurn()) count += 1;
        for (self.turns[self.chunk_start..self.complete_end]) |turn| {
            if (turn.summarized()) count += 1;
        }
        return count;
    }
};

/// The saved file for one tool call: an index line, then its arguments and
/// result, unchanged. `alloc` should be an arena; parsing scratch is not freed.
fn toolFile(alloc: Allocator, tool: PendingTool) Allocator.Error![]u8 {
    var text: std.ArrayList(u8) = .empty;
    try text.print(alloc, "T{d} {s}", .{ tool.number, tool.name });
    if (tool.call) |call| {
        const index = try indexLine(alloc, call.arguments);
        if (index.len > 0) try text.print(alloc, ": {s}", .{index});
    }
    try text.append(alloc, '\n');
    if (tool.call) |call| {
        try text.print(alloc, "Call ID: {s}\n\nArguments:\n{s}\n", .{ call.id, call.arguments });
    } else {
        try text.print(alloc, "Call ID: {s}\n\nArguments: (not recorded)\n", .{tool.result.?.call_id});
    }
    if (tool.result) |result| {
        try text.print(alloc, "\nResult:\n{s}\n", .{result.output});
    } else {
        try text.appendSlice(alloc, "\nResult: (not recorded)\n");
    }
    return text.toOwnedSlice(alloc);
}

/// The saved file for one turn: an index line with the start of its first
/// user message, then the whole turn word for word, tool calls by ID.
fn turnFile(alloc: Allocator, turn: Prepared) Allocator.Error![]u8 {
    var text: std.ArrayList(u8) = .empty;
    try text.print(alloc, "M{d} turn", .{turn.number});
    var index: std.ArrayList(u8) = .empty;
    try appendFlat(alloc, &index, turn.source.user);
    const line = std.mem.trim(u8, index.items[0..utf8Floor(index.items, max_index_bytes)], " ");
    if (line.len > 0) try text.print(alloc, ": {s}", .{line});
    try text.print(alloc, "\nUser {d}:\n{s}\n\n{s}", .{ turn.number, turn.source.user, turn.text });
    return text.toOwnedSlice(alloc);
}

const max_index_bytes = 240;

/// One line that says what a tool call was for: the text values in its
/// arguments (command, path, pattern, ...), in order, on one line. It works
/// the same for every tool, so search can rank a match here above a match
/// deep in the output. Arguments that are not JSON are used as plain text.
fn indexLine(alloc: Allocator, arguments: []const u8) Allocator.Error![]const u8 {
    var line: std.ArrayList(u8) = .empty;
    if (std.json.parseFromSliceLeaky(std.json.Value, alloc, arguments, .{})) |value| {
        try appendValues(alloc, &line, value, 0);
    } else |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => try appendFlat(alloc, &line, arguments),
    }
    return std.mem.trim(u8, line.items[0..utf8Floor(line.items, max_index_bytes)], " ");
}

/// The longest prefix of `text` within `limit` bytes that ends on a UTF-8
/// character boundary.
fn utf8Floor(text: []const u8, limit: usize) usize {
    var end = @min(text.len, limit);
    while (end > 0 and end < text.len and text[end] & 0xc0 == 0x80) end -= 1;
    return end;
}

fn appendValues(alloc: Allocator, line: *std.ArrayList(u8), value: std.json.Value, depth: usize) Allocator.Error!void {
    if (line.items.len >= max_index_bytes or depth > 4) return;
    switch (value) {
        .string => |text| try appendFlat(alloc, line, text),
        .object => |map| for (map.values()) |child| try appendValues(alloc, line, child, depth + 1),
        .array => |items| for (items.items) |child| try appendValues(alloc, line, child, depth + 1),
        // Numbers, booleans and null rarely say what a call was for.
        else => {},
    }
}

/// Appends `text` with every run of whitespace collapsed to one space.
fn appendFlat(alloc: Allocator, line: *std.ArrayList(u8), text: []const u8) Allocator.Error!void {
    if (line.items.len > 0 and line.items[line.items.len - 1] != ' ') try line.append(alloc, ' ');
    for (text) |byte| {
        if (line.items.len > max_index_bytes) return;
        const space = std.ascii.isWhitespace(byte);
        if (space and (line.items.len == 0 or line.items[line.items.len - 1] == ' ')) continue;
        try line.append(alloc, if (space) ' ' else byte);
    }
}

/// Everything the model reads, unchanged, followed by the one request.
fn renderTranscript(alloc: Allocator, plan: Plan) Allocator.Error![]u8 {
    var text: std.ArrayList(u8) = .empty;
    const earlier = plan.earlier;
    if (earlier.earlier.len > 0) {
        const label = if (plan.foldsEarlier()) "[Earlier summary]" else "[Earlier summary, kept as it is]";
        try text.print(alloc, "{s}\n{s}\n\n", .{ label, earlier.earlier });
    }
    for (earlier.turns) |turn| {
        if (turn.number == 0) {
            for (turn.users) |user| try text.print(alloc, "[Earlier user message]\n{s}\n\n", .{user});
            continue;
        }
        try text.print(alloc, "[Turn {d}, from the previous compaction]\n", .{turn.number});
        for (turn.users, 0..) |user, index| {
            try text.print(alloc, "[{s}]\n{s}\n\n", .{ if (index == 0) "User" else "User, added while the assistant worked", user });
        }
        if (turn.work.len > 0) try text.print(alloc, "[Assistant's work, summarized]\n{s}\n\n", .{turn.work});
        if (turn.final.len > 0) try text.print(alloc, "[Assistant's final reply]\n{s}\n\n", .{turn.final});
        if (turn.first_tool > 0) try text.print(alloc, "[Tools T{d} to T{d}]\n\n", .{ turn.first_tool, turn.last_tool });
    }
    for (plan.turns) |turn| {
        if (turn.number > 0) {
            try text.print(alloc, "[Turn {d}]\n[User]\n{s}\n\n", .{ turn.number, turn.source.user });
        } else {
            try text.print(alloc, "[Turn in progress]\n[User, this message stays in the conversation after the summary]\n{s}\n\n", .{turn.source.user});
        }
        if (turn.continued) |part| {
            if (part.work.len > 0) try text.print(alloc, "[Earlier part of this turn, summarized]\n{s}\n\n", .{part.work});
            for (part.users) |user| try text.print(alloc, "[User, added while the assistant worked]\n{s}\n\n", .{user});
            if (part.first_tool > 0) try text.print(alloc, "[Its tools so far: T{d} to T{d}]\n\n", .{ part.first_tool, part.last_tool });
        }
        for (turn.source.items, turn.tool_numbers) |item, number| switch (item) {
            .user => |user| try text.print(alloc, "[User, added while the assistant worked]\n{s}\n\n", .{user}),
            .assistant => |assistant| if (assistant.len > 0) try text.print(alloc, "[Assistant]\n{s}\n\n", .{assistant}),
            .note => |note| try text.print(alloc, "[From fx, not the user]\n{s}\n\n", .{note}),
            .tool_call => |call| try text.print(alloc, "[Tool call T{d}: {s}]\n{s}\n\n", .{ number, call.name, call.arguments }),
            .tool_result => |result| try text.print(alloc, "[Tool result T{d}: {s}]\n{s}\n\n", .{ number, result.name, result.output }),
        };
    }
    try writeRequest(alloc, &text, plan);
    return text.toOwnedSlice(alloc);
}

/// The one request: which sections to write and how.
fn writeRequest(alloc: Allocator, text: *std.ArrayList(u8), plan: Plan) Allocator.Error!void {
    try text.appendSlice(alloc, "Summarize the conversation above in the sections below. Start each section with its heading alone on its own line, exactly as shown.\n\n");
    if (plan.foldsEarlier()) {
        try text.print(alloc, "{s}:\nOne summary of ", .{earlier_heading});
        var parts: std.ArrayList([]const u8) = .empty;
        if (plan.earlier.earlier.len > 0) try parts.append(alloc, "the earlier summary");
        var first: usize = 0;
        var last: usize = 0;
        for (plan.earlier.turns) |turn| {
            if (turn.number == 0) continue;
            if (first == 0) first = turn.number;
            last = turn.number;
        }
        if (plan.earlier.turns.len > 0 and plan.earlier.turns[0].number == 0) try parts.append(alloc, "the earlier user messages");
        for (plan.turns[0..plan.chunk_start]) |turn| {
            if (first == 0) first = turn.number;
            last = turn.number;
        }
        if (first > 0) try parts.append(alloc, if (first == last)
            try std.fmt.allocPrint(alloc, "turn {d}", .{first})
        else
            try std.fmt.allocPrint(alloc, "turns {d} to {d}", .{ first, last }));
        for (parts.items, 0..) |part, index| {
            if (index > 0) try text.appendSlice(alloc, if (index + 1 == parts.items.len) " and " else ", ");
            try text.appendSlice(alloc, part);
        }
        try text.appendSlice(alloc, ". Only this summary of them stays in the conversation, so keep every detail from them that the work may still need.\n\n");
    }
    var worked: std.ArrayList(usize) = .empty;
    var long_final: std.ArrayList(usize) = .empty;
    for (plan.turns[plan.chunk_start..plan.complete_end]) |turn| {
        if (turn.final.len > 0 and !turn.show_final) {
            try long_final.append(alloc, turn.number);
        } else if (turn.has_work) {
            try worked.append(alloc, turn.number);
        }
    }
    if (worked.items.len > 0) {
        for (worked.items) |number| try text.print(alloc, "Turn {d}:\n", .{number});
        try text.appendSlice(alloc, "One section for each of these turns: what the assistant did in that turn before its final reply. The turn's user messages and final reply stay in the conversation word for word, so don't repeat them.\n\n");
    }
    if (long_final.items.len > 0) {
        for (long_final.items) |number| try text.print(alloc, "Turn {d}:\n", .{number});
        try text.appendSlice(alloc, "One section for each of these turns: what the assistant did in that turn and what its final reply said. Their final replies are too long to stay in the conversation word for word; their user messages stay, so don't repeat those.\n\n");
    }
    if (plan.hasOpenTurn()) {
        try text.print(alloc, "{s}:\nWhat the assistant has done so far in the turn in progress, and where it stands. Its user message stays in the conversation word for word.\n\n", .{open_heading});
    }
    try text.appendSlice(alloc, "Cover what the assistant did, found, decided and changed, and what is left to do. Be exact: say what was verified, and mark anything that was only planned, assumed, or not checked. ");
    if (plan.saved) {
        try text.appendSlice(alloc, "When a turn or tool call matters, refer to it by its ID, like M3 or T12. Everything under an ID stays saved word for word.");
    } else {
        try text.appendSlice(alloc, "The tool calls will not be available later, so keep the details from them that the work still needs.");
    }
}

/// The summary split into the sections the model was asked for.
const Sections = struct {
    earlier: []const u8 = "",
    /// One per shown turn, in order; empty when the model wrote none.
    turn_work: []const []const u8 = &.{},
    open_work: []const u8 = "",
};

const Target = union(enum) { preamble, earlier, turn: usize, open };

fn parseSections(alloc: Allocator, summary: []const u8, plan: Plan) Allocator.Error!Sections {
    const shown = plan.turns[plan.chunk_start..plan.complete_end];
    const turn_work = try alloc.alloc(std.ArrayList(u8), shown.len);
    for (turn_work) |*work| work.* = .empty;
    var earlier: std.ArrayList(u8) = .empty;
    var extra: std.ArrayList(u8) = .empty;
    var open: std.ArrayList(u8) = .empty;
    var found_heading = false;
    var target: Target = .preamble;
    var lines = std.mem.splitScalar(u8, summary, '\n');
    while (lines.next()) |line| {
        var content = line;
        if (parseHeading(line)) |heading| {
            found_heading = true;
            target = heading.target;
            content = heading.rest;
            if (content.len == 0) continue;
        }
        const destination = switch (target) {
            .preamble => &extra,
            .earlier => &earlier,
            .open => &open,
            .turn => |number| for (shown, turn_work) |turn, *work| {
                if (turn.number == number) break work;
            } else &extra,
        };
        try destination.appendSlice(alloc, content);
        try destination.append(alloc, '\n');
    }

    var sections: Sections = .{};
    const work = try alloc.alloc([]const u8, shown.len);
    for (work, turn_work) |*slot, collected| slot.* = std.mem.trim(u8, collected.items, " \t\r\n");
    sections.turn_work = work;
    sections.open_work = std.mem.trim(u8, open.items, " \t\r\n");
    if (!found_heading) {
        // The model ignored the headings. Keep all of its text in the one
        // section that was asked for, or else with the earlier summary.
        const whole = try alloc.dupe(u8, summary);
        if (plan.sectionCount() == 1 and !plan.foldsEarlier()) {
            if (plan.hasOpenTurn()) {
                sections.open_work = whole;
            } else for (shown, work) |turn, *slot| {
                if (turn.summarized()) slot.* = whole;
            }
        } else {
            sections.earlier = whole;
        }
        return sections;
    }
    if (plan.foldsEarlier()) {
        if (extra.items.len > 0) {
            try earlier.append(alloc, '\n');
            try earlier.appendSlice(alloc, extra.items);
        }
        sections.earlier = std.mem.trim(u8, earlier.items, " \t\r\n");
        // Without its own section the earlier part would vanish from the
        // conversation; keep everything the model wrote instead.
        if (sections.earlier.len == 0) sections.earlier = try alloc.dupe(u8, summary);
    }
    return sections;
}

const Heading = struct { target: Target, rest: []const u8 };

/// A section heading like `Earlier:`, `**Turn 12:**`, `Turn 12 (M12):` or
/// `## Turn in progress`, with any text after its colon.
fn parseHeading(line: []const u8) ?Heading {
    const decoration = " \t#*_";
    var rest = std.mem.trimStart(u8, line, decoration);
    var target: Target = undefined;
    if (startsWithIgnoreCase(rest, open_heading)) {
        target = .open;
        rest = rest[open_heading.len..];
    } else if (startsWithIgnoreCase(rest, earlier_heading)) {
        target = .earlier;
        rest = std.mem.trimStart(u8, rest[earlier_heading.len..], " ");
        if (startsWithIgnoreCase(rest, "summary")) rest = rest["summary".len..];
    } else if (startsWithIgnoreCase(rest, "Turn ")) {
        rest = rest["Turn ".len..];
        var digits: usize = 0;
        while (digits < rest.len and std.ascii.isDigit(rest[digits])) digits += 1;
        const number = std.fmt.parseUnsigned(usize, rest[0..digits], 10) catch return null;
        target = .{ .turn = number };
        rest = std.mem.trimStart(u8, rest[digits..], " ");
        // A short note in parentheses, like "(M12)".
        if (rest.len > 0 and rest[0] == '(') {
            const close = std.mem.findScalar(u8, rest, ')') orelse return null;
            if (close > 24) return null;
            rest = rest[close + 1 ..];
        }
    } else return null;
    rest = std.mem.trimStart(u8, rest, decoration);
    if (rest.len == 0) return .{ .target = target, .rest = "" };
    if (rest[0] != ':') return null;
    return .{ .target = target, .rest = std.mem.trim(u8, rest[1..], decoration) };
}

fn startsWithIgnoreCase(text: []const u8, prefix: []const u8) bool {
    return text.len >= prefix.len and std.ascii.eqlIgnoreCase(text[0..prefix.len], prefix);
}

// Tests

const testing = std.testing;

const FakeModel = struct {
    reply: []const u8 = "Turn 1:\nThe assistant ran the build (T1) and found the error.",
    fail: ?ModelError = null,
    calls: usize = 0,
    seen_model: []const u8 = "",
    seen_system: []const u8 = "",
    seen_user: std.ArrayList(u8) = .empty,

    fn deinit(self: *FakeModel) void {
        self.seen_user.deinit(testing.allocator);
    }

    fn model(self: *FakeModel) Model {
        return .{ .context = self, .summarize_fn = summarize };
    }

    fn summarize(context: *anyopaque, alloc: Allocator, prompt: Prompt) ModelError![]u8 {
        const self: *FakeModel = @ptrCast(@alignCast(context));
        self.calls += 1;
        self.seen_model = prompt.model;
        self.seen_system = prompt.system;
        self.seen_user.clearRetainingCapacity();
        try self.seen_user.appendSlice(testing.allocator, prompt.user);
        if (self.fail) |err| return err;
        return alloc.dupe(u8, self.reply);
    }
};

const MemoryStore = struct {
    saved: std.ArrayList(Saved) = .empty,
    fail: bool = false,

    const Saved = struct { kind: compacted_records.Kind, number: usize, content: []u8 };

    fn deinit(self: *MemoryStore) void {
        for (self.saved.items) |item| testing.allocator.free(item.content);
        self.saved.deinit(testing.allocator);
    }

    fn store(self: *MemoryStore) Store {
        return .{ .context = self, .save_fn = save };
    }

    fn save(context: *anyopaque, kind: compacted_records.Kind, number: usize, content: []const u8) StoreError!void {
        const self: *MemoryStore = @ptrCast(@alignCast(context));
        if (self.fail) return error.StoreFailed;
        const copy = try testing.allocator.dupe(u8, content);
        errdefer testing.allocator.free(copy);
        try self.saved.append(testing.allocator, .{ .kind = kind, .number = number, .content = copy });
    }

    fn find(self: *const MemoryStore, kind: compacted_records.Kind, number: usize) ?[]const u8 {
        for (self.saved.items) |item| if (item.kind == kind and item.number == number) return item.content;
        return null;
    }
};

const sample = [_]Turn{
    .{ .user = "Fix the build.\nIt fails on main.", .items = &.{
        .{ .assistant = "I'll run the build first." },
        .{ .tool_call = .{ .id = "call-1", .name = "shell", .arguments = "{\"command\":\"zig build\"}" } },
        .{ .tool_result = .{ .call_id = "call-1", .name = "shell", .output = "error: missing semicolon at src/a.zig:4" } },
        .{ .assistant = "Found it: a missing semicolon at src/a.zig:4." },
    } },
    .{ .user = "thanks", .items = &.{.{ .assistant = "You're welcome." }} },
};

test "users and final replies stay exact, the rest of each turn is summarized" {
    var model = FakeModel{};
    defer model.deinit();
    var store = MemoryStore{};
    defer store.deinit();

    var result = try compact(testing.allocator, .{ .model = "fixture/model", .turns = &sample }, model.model(), store.store());
    defer result.deinit();

    const turns = result.compacted.turns;
    try testing.expectEqual(@as(usize, 2), turns.len);
    try testing.expectEqualStrings("Fix the build.\nIt fails on main.", turns[0].users[0]);
    try testing.expectEqualStrings("Found it: a missing semicolon at src/a.zig:4.", turns[0].final);
    try testing.expectEqualStrings("The assistant ran the build (T1) and found the error.", turns[0].work);
    try testing.expectEqual(@as(usize, 1), turns[0].first_tool);
    try testing.expectEqualStrings("You're welcome.", turns[1].final);
    try testing.expectEqualStrings("", turns[1].work);
    try testing.expectEqual(@as(usize, 2), result.compacted.turn_count);
    try testing.expectEqual(@as(usize, 1), result.compacted.tool_count);
    try testing.expectEqualStrings("", result.compacted.earlier);

    // The agent sees exact users and replies, but no tool payloads or the
    // assistant's in-between text.
    try testing.expect(std.mem.find(u8, result.text, "User 1:\nFix the build.\nIt fails on main.\n") != null);
    try testing.expect(std.mem.find(u8, result.text, "Assistant 1, final reply:\nFound it: a missing semicolon at src/a.zig:4.\n") != null);
    try testing.expect(std.mem.find(u8, result.text, "Assistant 1, summary of its work:\nThe assistant ran the build (T1)") != null);
    try testing.expect(std.mem.find(u8, result.text, "Tools: T1\n") != null);
    try testing.expect(std.mem.find(u8, result.text, "User 2:\nthanks\n") != null);
    try testing.expect(std.mem.find(u8, result.text, "Saved word for word: turns M1–M2 and tool call T1.") != null);
    try testing.expect(std.mem.find(u8, result.text, "I'll run the build first.") == null);
    try testing.expect(std.mem.find(u8, result.text, "error: missing semicolon") == null);
}

test "the model reads the whole conversation as is and is asked only for what it must write" {
    var model = FakeModel{};
    defer model.deinit();
    var store = MemoryStore{};
    defer store.deinit();

    var result = try compact(testing.allocator, .{ .model = "provider/exact-model", .turns = &sample }, model.model(), store.store());
    defer result.deinit();

    try testing.expectEqual(@as(usize, 1), model.calls);
    try testing.expectEqualStrings("provider/exact-model", model.seen_model);
    try testing.expectEqualStrings(system_prompt, model.seen_system);
    const seen = model.seen_user.items;
    try testing.expect(std.mem.find(u8, seen, "[Turn 1]\n[User]\nFix the build.\nIt fails on main.\n") != null);
    try testing.expect(std.mem.find(u8, seen, "[Assistant]\nI'll run the build first.\n") != null);
    try testing.expect(std.mem.find(u8, seen, "[Tool call T1: shell]\n{\"command\":\"zig build\"}\n") != null);
    try testing.expect(std.mem.find(u8, seen, "[Tool result T1: shell]\nerror: missing semicolon at src/a.zig:4\n") != null);
    try testing.expect(std.mem.find(u8, seen, "[Turn 2]\n[User]\nthanks\n") != null);
    // Only turn 1 has work to summarize; there is nothing earlier to fold.
    try testing.expect(std.mem.find(u8, seen, "\nTurn 1:\nOne section for each of these turns") != null);
    try testing.expect(std.mem.find(u8, seen, "Turn 2:") == null);
    try testing.expect(std.mem.find(u8, seen, "Earlier:") == null);
    try testing.expect(std.mem.find(u8, seen, "Turn in progress:") == null);
    try testing.expect(std.mem.endsWith(u8, seen, "Everything under an ID stays saved word for word."));
}

test "each turn and each tool call is saved word for word under its ID" {
    var model = FakeModel{};
    defer model.deinit();
    var store = MemoryStore{};
    defer store.deinit();

    var result = try compact(testing.allocator, .{ .model = "m", .turns = &sample }, model.model(), store.store());
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), store.saved.items.len);
    try testing.expectEqualStrings(
        "T1 shell: zig build\nCall ID: call-1\n\nArguments:\n{\"command\":\"zig build\"}\n\nResult:\nerror: missing semicolon at src/a.zig:4\n",
        store.find(.tool, 1).?,
    );
    try testing.expectEqualStrings(
        "M1 turn: Fix the build. It fails on main.\nUser 1:\nFix the build.\nIt fails on main.\n\n" ++
            "Assistant:\nI'll run the build first.\n\n[T1 shell: zig build]\n\n" ++
            "Assistant, final reply:\nFound it: a missing semicolon at src/a.zig:4.\n\n",
        store.find(.turn, 1).?,
    );
    try testing.expectEqualStrings("M2 turn: thanks\nUser 2:\nthanks\n\nAssistant, final reply:\nYou're welcome.\n\n", store.find(.turn, 2).?);
}

test "turns with nothing to summarize need no model call" {
    var model = FakeModel{};
    defer model.deinit();
    var store = MemoryStore{};
    defer store.deinit();
    const chat = [_]Turn{
        .{ .user = "what is 2+2?", .items = &.{.{ .assistant = "4" }} },
        .{ .user = "yes", .items = &.{.{ .assistant = "ok" }} },
        .{ .user = "yes", .items = &.{.{ .assistant = "ok" }} },
    };
    var result = try compact(testing.allocator, .{ .model = "m", .turns = &chat }, model.model(), store.store());
    defer result.deinit();
    try testing.expectEqual(@as(usize, 0), model.calls);
    try testing.expectEqual(@as(usize, 3), result.compacted.turns.len);
    // Repeated identical messages are all kept.
    try testing.expectEqualStrings("yes", result.compacted.turns[2].users[0]);
    try testing.expectEqualStrings("ok", result.compacted.turns[2].final);
}

test "compacting again folds the shown turns into the earlier summary and keeps numbering" {
    var store = MemoryStore{};
    defer store.deinit();

    var first_model = FakeModel{};
    defer first_model.deinit();
    var first = try compact(testing.allocator, .{ .model = "m", .turns = &sample }, first_model.model(), store.store());
    defer first.deinit();

    const next = [_]Turn{.{ .user = "now run the tests", .items = &.{
        .{ .tool_call = .{ .id = "call-2", .name = "shell", .arguments = "{\"command\":\"zig build test\"}" } },
        .{ .tool_result = .{ .call_id = "call-2", .name = "shell", .output = "All 12 tests passed." } },
        .{ .assistant = "All 12 tests pass." },
    } }};
    var second_model = FakeModel{ .reply = "**Earlier:**\nThe build failed on a missing semicolon [M1, T1]; the user said thanks [M2].\n\nTurn 3: Ran the tests (T2)." };
    defer second_model.deinit();
    var second = try compact(testing.allocator, .{ .model = "m", .earlier = first.compacted, .turns = &next }, second_model.model(), store.store());
    defer second.deinit();

    const seen = second_model.seen_user.items;
    try testing.expect(std.mem.find(u8, seen, "[Turn 1, from the previous compaction]\n[User]\nFix the build.\nIt fails on main.\n") != null);
    try testing.expect(std.mem.find(u8, seen, "[Assistant's final reply]\nYou're welcome.\n") != null);
    try testing.expect(std.mem.find(u8, seen, "[Turn 3]\n[User]\nnow run the tests\n") != null);
    try testing.expect(std.mem.find(u8, seen, "Earlier:\nOne summary of turns 1 to 2.") != null);

    try testing.expectEqualStrings("The build failed on a missing semicolon [M1, T1]; the user said thanks [M2].", second.compacted.earlier);
    try testing.expectEqual(@as(usize, 1), second.compacted.turns.len);
    try testing.expectEqual(@as(usize, 3), second.compacted.turns[0].number);
    try testing.expectEqualStrings("Ran the tests (T2).", second.compacted.turns[0].work);
    try testing.expectEqual(@as(usize, 2), second.compacted.turns[0].first_tool);
    try testing.expectEqual(@as(usize, 3), second.compacted.turn_count);
    try testing.expectEqual(@as(usize, 2), second.compacted.tool_count);
    try testing.expect(std.mem.find(u8, second.text, "User 1:") == null);
    try testing.expect(std.mem.find(u8, second.text, "User 3:\nnow run the tests\n") != null);
    try testing.expect(std.mem.find(u8, second.text, "Saved word for word: turns M1–M3 and tool calls T1–T2.") != null);
    // Only the new turn and tool are saved; nothing is saved twice.
    try testing.expectEqual(@as(usize, 5), store.saved.items.len);
    try testing.expect(store.find(.turn, 3) != null);
    try testing.expect(store.find(.tool, 2) != null);
}

test "older new turns fold into the earlier summary when the exact budget is spent" {
    var model = FakeModel{ .reply = "Earlier:\nThe build was fixed [M1]." };
    defer model.deinit();
    var store = MemoryStore{};
    defer store.deinit();
    var result = try compact(testing.allocator, .{ .model = "m", .turns = &sample, .exact_tokens = 6 }, model.model(), store.store());
    defer result.deinit();
    try testing.expectEqual(@as(usize, 1), result.compacted.turns.len);
    try testing.expectEqual(@as(usize, 2), result.compacted.turns[0].number);
    try testing.expectEqualStrings("The build was fixed [M1].", result.compacted.earlier);
    try testing.expect(std.mem.find(u8, model.seen_user.items, "Earlier:\nOne summary of turn 1.") != null);
    // Folded turns are still saved word for word.
    try testing.expect(store.find(.turn, 1) != null);

    var nothing_exact = FakeModel{ .reply = "Earlier:\nEverything." };
    defer nothing_exact.deinit();
    var folded = try compact(testing.allocator, .{ .model = "m", .turns = &sample, .exact_tokens = 0 }, nothing_exact.model(), null);
    defer folded.deinit();
    try testing.expectEqual(@as(usize, 0), folded.compacted.turns.len);
    try testing.expectEqualStrings("Everything.", folded.compacted.earlier);
}

test "a final reply too long to show is summarized while its user message stays" {
    var model = FakeModel{ .reply = "Turn 1:\nWrote the full plan: three phases, tests first." };
    defer model.deinit();
    var store = MemoryStore{};
    defer store.deinit();
    const long = [_]Turn{
        .{ .user = "write the plan", .items = &.{.{ .assistant = "PLAN " ** 400 }} },
        .{ .user = "thanks", .items = &.{.{ .assistant = "Sure." }} },
    };
    var result = try compact(testing.allocator, .{ .model = "m", .turns = &long, .exact_tokens = 50 }, model.model(), store.store());
    defer result.deinit();
    const turns = result.compacted.turns;
    try testing.expectEqual(@as(usize, 2), turns.len);
    try testing.expectEqualStrings("write the plan", turns[0].users[0]);
    try testing.expectEqualStrings("", turns[0].final);
    try testing.expectEqualStrings("Wrote the full plan: three phases, tests first.", turns[0].work);
    // A smaller older or newer reply still fits.
    try testing.expectEqualStrings("Sure.", turns[1].final);
    try testing.expect(std.mem.find(u8, model.seen_user.items, "Turn 1:\nOne section for each of these turns: what the assistant did in that turn and what its final reply said.") != null);
    try testing.expect(std.mem.find(u8, store.find(.turn, 1).?, "PLAN PLAN PLAN") != null);
    try testing.expect(std.mem.find(u8, result.text, "PLAN PLAN") == null);
}

test "a turn in progress carries over and its saved turn is complete once it ends" {
    var store = MemoryStore{};
    defer store.deinit();
    const running = [_]Turn{.{ .user = "migrate the db", .items = &.{
        .{ .assistant = "Starting with a dry run." },
        .{ .tool_call = .{ .id = "c1", .name = "shell", .arguments = "{\"command\":\"migrate --dry-run\"}" } },
        .{ .tool_result = .{ .call_id = "c1", .name = "shell", .output = "3 tables to change" } },
        .{ .user = "use the staging db" },
    } }};
    var first_model = FakeModel{ .reply = "Turn in progress:\nDry run found 3 tables to change (T1)." };
    defer first_model.deinit();
    var first = try compact(testing.allocator, .{ .model = "m", .turns = &running, .last_turn_open = true }, first_model.model(), store.store());
    defer first.deinit();

    const open = first.compacted.open.?;
    try testing.expectEqualStrings("Dry run found 3 tables to change (T1).", open.work);
    try testing.expectEqualStrings("use the staging db", open.users[0]);
    try testing.expectEqual(@as(usize, 1), open.first_tool);
    try testing.expectEqual(@as(usize, 0), first.compacted.turns.len);
    try testing.expectEqual(@as(usize, 0), first.compacted.turn_count);
    try testing.expect(store.find(.turn, 1) == null);
    try testing.expect(std.mem.find(u8, first_model.seen_user.items, "[Turn in progress]\n[User, this message stays in the conversation after the summary]\nmigrate the db\n") != null);
    // The first user message stays in the conversation, so it is not shown.
    try testing.expect(std.mem.find(u8, first.text, "migrate the db") == null);

    // The same turn later ends; its first user message and the rest arrive
    // as the next compaction's first turn.
    const finished = [_]Turn{.{ .user = "migrate the db", .items = &.{
        .{ .tool_call = .{ .id = "c2", .name = "shell", .arguments = "{\"command\":\"migrate --target staging\"}" } },
        .{ .tool_result = .{ .call_id = "c2", .name = "shell", .output = "migrated" } },
        .{ .assistant = "Migrated staging." },
    } }};
    var second_model = FakeModel{ .reply = "Turn 1:\nMigrated staging (T2) after a dry run (T1)." };
    defer second_model.deinit();
    var second = try compact(testing.allocator, .{ .model = "m", .earlier = first.compacted, .turns = &finished }, second_model.model(), store.store());
    defer second.deinit();

    try testing.expect(std.mem.find(u8, second_model.seen_user.items, "[Earlier part of this turn, summarized]\nDry run found 3 tables to change (T1).\n") != null);
    const turn = second.compacted.turns[0];
    try testing.expectEqual(@as(usize, 1), turn.number);
    try testing.expectEqual(@as(usize, 2), turn.users.len);
    try testing.expectEqualStrings("use the staging db", turn.users[1]);
    try testing.expectEqualStrings("Migrated staging.", turn.final);
    try testing.expectEqual(@as(usize, 1), turn.first_tool);
    try testing.expectEqual(@as(usize, 2), turn.last_tool);
    try testing.expect(second.compacted.open == null);
    try testing.expectEqualStrings(
        "M1 turn: migrate the db\nUser 1:\nmigrate the db\n\n" ++
            "Assistant:\nStarting with a dry run.\n\n[T1 shell: migrate --dry-run]\n\n" ++
            "User, added while the assistant worked:\nuse the staging db\n\n" ++
            "[T2 shell: migrate --target staging]\n\nAssistant, final reply:\nMigrated staging.\n\n",
        store.find(.turn, 1).?,
    );
}

test "a turn in progress compacted twice keeps growing" {
    var store = MemoryStore{};
    defer store.deinit();
    const part = [_]Turn{.{ .user = "long task", .items = &.{
        .{ .tool_call = .{ .id = "a", .name = "shell", .arguments = "{\"command\":\"step one\"}" } },
        .{ .tool_result = .{ .call_id = "a", .name = "shell", .output = "ok" } },
    } }};
    var model = FakeModel{ .reply = "Turn in progress:\nDid step one (T1)." };
    defer model.deinit();
    var first = try compact(testing.allocator, .{ .model = "m", .turns = &part, .last_turn_open = true }, model.model(), store.store());
    defer first.deinit();
    const more = [_]Turn{.{ .user = "long task", .items = &.{
        .{ .tool_call = .{ .id = "b", .name = "shell", .arguments = "{\"command\":\"step two\"}" } },
        .{ .tool_result = .{ .call_id = "b", .name = "shell", .output = "ok" } },
    } }};
    var again = FakeModel{ .reply = "Turn in progress:\nDid steps one and two (T1, T2)." };
    defer again.deinit();
    var second = try compact(testing.allocator, .{ .model = "m", .earlier = first.compacted, .turns = &more, .last_turn_open = true }, again.model(), store.store());
    defer second.deinit();
    const open = second.compacted.open.?;
    try testing.expectEqualStrings("Did steps one and two (T1, T2).", open.work);
    try testing.expectEqual(@as(usize, 1), open.first_tool);
    try testing.expectEqual(@as(usize, 2), open.last_tool);
    try testing.expectEqualStrings("[T1 shell: step one]\n\n[T2 shell: step two]\n\n", open.text);
    try testing.expectEqual(@as(usize, 0), second.compacted.turn_count);
}

test "user messages from an older checkpoint fold into the earlier summary" {
    var model = FakeModel{ .reply = "Earlier:\nThe user asked about a 2 GB limit; there is none." };
    defer model.deinit();
    const legacy: Compacted = .{
        .earlier = "Old summary.",
        .turns = &.{ .{ .users = &.{"what if 2 GB exceeds?"} }, .{ .users = &.{"yes"} } },
    };
    var result = try compact(testing.allocator, .{ .model = "m", .earlier = legacy, .turns = sample[1..] }, model.model(), null);
    defer result.deinit();
    const seen = model.seen_user.items;
    try testing.expect(std.mem.find(u8, seen, "[Earlier summary]\nOld summary.\n") != null);
    try testing.expect(std.mem.find(u8, seen, "[Earlier user message]\nwhat if 2 GB exceeds?\n") != null);
    try testing.expect(std.mem.find(u8, seen, "Earlier:\nOne summary of the earlier summary and the earlier user messages.") != null);
    try testing.expectEqualStrings("The user asked about a 2 GB limit; there is none.", result.compacted.earlier);
    try testing.expectEqual(@as(usize, 1), result.compacted.turns[0].number);
}

test "sections are found with markdown decoration and text on the heading line" {
    const turns = [_]Prepared{
        .{ .source = .{ .user = "a" }, .number = 4, .tool_numbers = &.{}, .tools = &.{}, .first_tool = 0, .last_tool = 0, .users = &.{}, .final = "", .final_index = null, .has_work = true, .continued = null, .text = "" },
        .{ .source = .{ .user = "b" }, .number = 5, .tool_numbers = &.{}, .tools = &.{}, .first_tool = 0, .last_tool = 0, .users = &.{}, .final = "", .final_index = null, .has_work = true, .continued = null, .text = "" },
        .{ .source = .{ .user = "c" }, .tool_numbers = &.{}, .tools = &.{}, .first_tool = 0, .last_tool = 0, .users = &.{}, .final = "", .final_index = null, .has_work = true, .continued = null, .text = "" },
    };
    const plan: Plan = .{ .earlier = .{ .turns = &.{.{ .number = 3 }} }, .turns = &turns, .chunk_start = 0, .complete_end = 2, .saved = true };
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const parsed = try parseSections(arena, "Here is the summary.\n## Earlier\nDid the setup.\n**Turn 4:** Ran tests.\nTurn 5\nFixed it.\nEarlier, it failed.\nTurn 9: unknown turn.\n__Turn in progress:__\nStill going.", plan);
    try testing.expectEqualStrings("Did the setup.\n\nHere is the summary.\nunknown turn.", parsed.earlier);
    try testing.expectEqualStrings("Ran tests.", parsed.turn_work[0]);
    try testing.expectEqualStrings("Fixed it.\nEarlier, it failed.", parsed.turn_work[1]);
    try testing.expectEqualStrings("Still going.", parsed.open_work);

    const variants = try parseSections(arena, "Earlier summary:\nSetup done.\nTurn 4 (M4): Ran tests.", plan);
    try testing.expectEqualStrings("Setup done.", variants.earlier);
    try testing.expectEqualStrings("Ran tests.", variants.turn_work[0]);

    const bare = try parseSections(arena, "No headings at all.", plan);
    try testing.expectEqualStrings("No headings at all.", bare.earlier);
    const no_earlier = try parseSections(arena, "Turn 4:\nRan tests.", plan);
    try testing.expectEqualStrings("Turn 4:\nRan tests.", no_earlier.earlier);
    try testing.expectEqualStrings("Ran tests.", no_earlier.turn_work[0]);
}

test "the index line is built the same way for any tool" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // Nested requests, multi-line text, and non-text values.
    try testing.expectEqualStrings(
        "run cd /repo && wc -l src/app.zig",
        try indexLine(arena, "{\"request\":{\"action\":\"run\",\"command\":\"cd /repo &&\\n  wc -l src/app.zig\",\"yield_time_ms\":300000}}"),
    );
    try testing.expectEqualStrings("resumeForWrite src/core *.zig", try indexLine(arena, "{\"pattern\":\"resumeForWrite\",\"path\":\"src/core\",\"include\":[\"*.zig\"],\"case_insensitive\":true}"));
    try testing.expectEqualStrings("", try indexLine(arena, "{}"));
    try testing.expectEqualStrings("not json at all", try indexLine(arena, "not   json\nat all"));

    // Long arguments are cut to one short line without splitting a character.
    const long = try indexLine(arena, "{\"content\":\"" ++ ("é" ** 300) ++ "\"}");
    try testing.expect(long.len <= max_index_bytes);
    try testing.expect(std.unicode.utf8ValidateSlice(long));
}

test "a call without a result and a result without a call are both saved" {
    var model = FakeModel{};
    defer model.deinit();
    var store = MemoryStore{};
    defer store.deinit();
    const turns = [_]Turn{.{ .user = "go", .items = &.{
        .{ .tool_result = .{ .call_id = "orphan", .name = "read_file", .output = "file text" } },
        .{ .tool_call = .{ .id = "cut-off", .name = "shell", .arguments = "{}" } },
    } }};

    var result = try compact(testing.allocator, .{ .model = "m", .turns = &turns }, model.model(), store.store());
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.compacted.tool_count);
    try testing.expect(std.mem.find(u8, store.find(.tool, 1).?, "T1 read_file\nCall ID: orphan\n\nArguments: (not recorded)") != null);
    try testing.expect(std.mem.find(u8, store.find(.tool, 2).?, "Result: (not recorded)") != null);
    try testing.expect(std.mem.find(u8, store.find(.turn, 1).?, "[T1 read_file, result only]\n\n[T2 shell]\n") != null);
    try testing.expectEqualStrings("", result.compacted.turns[0].final);
}

test "input errors are reported before any work" {
    var model = FakeModel{};
    defer model.deinit();
    var store = MemoryStore{};
    defer store.deinit();
    try testing.expectError(error.NothingToCompact, compact(testing.allocator, .{ .model = "m", .turns = &.{} }, model.model(), store.store()));
    try testing.expectEqual(@as(usize, 0), model.calls);
    try testing.expectEqual(@as(usize, 0), store.saved.items.len);
}

test "model and store failures return errors without leaking" {
    var store = MemoryStore{};
    defer store.deinit();

    var empty = FakeModel{ .reply = " \n\t " };
    defer empty.deinit();
    try testing.expectError(error.EmptySummary, compact(testing.allocator, .{ .model = "m", .turns = &sample }, empty.model(), store.store()));

    var failing = FakeModel{ .fail = error.SummaryIncomplete };
    defer failing.deinit();
    try testing.expectError(error.SummaryIncomplete, compact(testing.allocator, .{ .model = "m", .turns = &sample }, failing.model(), store.store()));

    var broken_store = MemoryStore{ .fail = true };
    defer broken_store.deinit();
    var unused = FakeModel{};
    defer unused.deinit();
    try testing.expectError(error.StoreFailed, compact(testing.allocator, .{ .model = "m", .turns = &sample }, unused.model(), broken_store.store()));
    try testing.expectEqual(@as(usize, 0), unused.calls);
}

test "compaction survives every allocation failure" {
    const Run = struct {
        fn run(alloc: Allocator) !void {
            var model = FakeModel{ .reply = "Earlier:\nx\nTurn 2:\ny" };
            defer model.deinit();
            var store = MemoryStore{};
            defer store.deinit();
            const earlier: Compacted = .{ .earlier = "e", .turns = &.{.{ .number = 1, .users = &.{"u"}, .final = "f" }}, .open = .{ .text = "t", .work = "w" }, .turn_count = 1 };
            var result = try compact(alloc, .{ .model = "m", .earlier = earlier, .turns = &sample }, model.model(), store.store());
            result.deinit();
        }
    };
    try testing.checkAllAllocationFailures(testing.allocator, Run.run, .{});
}

test "without a store nothing is saved and the summary keeps tool details" {
    var model = FakeModel{};
    defer model.deinit();
    var result = try compact(testing.allocator, .{ .model = "m", .turns = &sample }, model.model(), null);
    defer result.deinit();
    try testing.expect(!result.compacted.saved);
    try testing.expect(std.mem.endsWith(u8, model.seen_user.items, "The tool calls will not be available later, so keep the details from them that the work still needs."));
    try testing.expect(std.mem.find(u8, result.text, "read_tool_result") == null);
    try testing.expect(std.mem.find(u8, result.text, "User 1:\nFix the build.") != null);
}

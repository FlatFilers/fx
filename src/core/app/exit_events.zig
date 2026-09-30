//! Formats background shell exit notices as the prompt for an `[event]` turn.

const std = @import("std");
const managed_execution = @import("../execution/managed_execution.zig");

const Allocator = std.mem.Allocator;

/// One `[event]` line per notice, each followed by its fenced output tail.
/// Caller owns the result.
pub fn formatPrompt(alloc: Allocator, notices: []const managed_execution.ExitNotice) Allocator.Error![]u8 {
    var out: std.Io.Writer.Allocating = .init(alloc);
    errdefer out.deinit();
    const writer = &out.writer;
    for (notices, 0..) |notice, index| {
        if (index > 0) writer.writeAll("\n\n") catch return error.OutOfMemory;
        writer.print("[event] {s} ", .{notice.execution_id}) catch return error.OutOfMemory;
        writeState(writer, notice.state) catch return error.OutOfMemory;
        writer.print(": {s}", .{notice.command}) catch return error.OutOfMemory;
        const tail = std.mem.trim(u8, notice.output_tail, " \t\r\n");
        if (tail.len > 0) {
            writer.print("\n```\n{s}\n```", .{tail}) catch return error.OutOfMemory;
        }
    }
    return out.toOwnedSlice() catch error.OutOfMemory;
}

fn writeState(writer: *std.Io.Writer, state: managed_execution.SnapshotState) !void {
    switch (state) {
        .running => try writer.writeAll("is still running"),
        .completed => |status| try writeStatus(writer, "exited", status),
        .stopped => |maybe| if (maybe) |status| try writeStatus(writer, "stopped", status) else try writer.writeAll("stopped"),
        .lost => try writer.writeAll("was lost"),
    }
}

fn writeStatus(writer: *std.Io.Writer, verb: []const u8, status: anytype) !void {
    switch (status) {
        .exit_code => |code| try writer.print("{s} {d}", .{ verb, code }),
        .signal => |signal| try writer.print("{s} by signal {d}", .{ verb, signal }),
        .finished, .indeterminate => try writer.writeAll(verb),
    }
}

test "formats exit status, command, and fenced tail" {
    const alloc = std.testing.allocator;
    var id = "shell-3".*;
    var cmd = "bin/gh-watch pr 36320 --on approved".*;
    var tail = "EVENT approved · lion approved PR #36320\n".*;
    var none = "".*;
    const notices = [_]managed_execution.ExitNotice{
        .{ .execution_id = &id, .command = &cmd, .state = .{ .completed = .{ .exit_code = 0 } }, .output_tail = &tail },
        .{ .execution_id = &id, .command = &cmd, .state = .lost, .output_tail = &none },
    };
    const text = try formatPrompt(alloc, &notices);
    defer alloc.free(text);
    try std.testing.expectEqualStrings(
        "[event] shell-3 exited 0: bin/gh-watch pr 36320 --on approved\n```\nEVENT approved · lion approved PR #36320\n```\n\n" ++
            "[event] shell-3 was lost: bin/gh-watch pr 36320 --on approved",
        text,
    );
}

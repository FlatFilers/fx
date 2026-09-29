const std = @import("std");
const io_mod = @import("../core/shared/io.zig");
const secret = @import("../core/auth/secret.zig");
const gateway_client = @import("client.zig");

pub const Response = struct {
    status: std.http.Status,
    /// Owned by the caller; release with deinit.
    body: []u8,

    pub fn deinit(self: *Response, alloc: std.mem.Allocator) void {
        secret.zeroAndFree(alloc, self.body);
        self.* = undefined;
    }
};

pub const Operation = struct {
    alloc: std.mem.Allocator,
    url: []const u8,
    credential: ?[]const u8,
    extra_headers: []const std.http.Header,
    max_body_bytes: usize,
    too_large_error: error{ GrokModelCatalogTooLarge, CodexModelCatalogTooLarge },

    pub fn run(self: *@This()) !Response {
        var client: std.http.Client = .{ .allocator = self.alloc, .io = io_mod.getIo() };
        defer client.deinit();
        var auth_header: ?[]u8 = null;
        defer if (auth_header) |value| secret.zeroAndFree(self.alloc, value);
        var headers: std.http.Client.Request.Headers = .{
            .user_agent = .{ .override = gateway_client.user_agent },
            .accept_encoding = .omit,
        };
        if (self.credential) |credential| {
            auth_header = try std.fmt.allocPrint(self.alloc, "Bearer {s}", .{credential});
            headers.authorization = .{ .override = auth_header.? };
        }
        const body_buffer = try self.alloc.alloc(u8, self.max_body_bytes + 1);
        defer secret.zeroAndFree(self.alloc, body_buffer);
        var response_writer = std.Io.Writer.fixed(body_buffer);
        const result = client.fetch(.{
            .location = .{ .url = self.url },
            .method = .GET,
            .headers = headers,
            .extra_headers = self.extra_headers,
            .response_writer = &response_writer,
            .redirect_behavior = .unhandled,
        }) catch |err| switch (err) {
            error.WriteFailed => return self.too_large_error,
            else => return err,
        };
        const body = response_writer.buffered();
        if (body.len > self.max_body_bytes) return self.too_large_error;
        return .{
            .status = result.status,
            .body = try self.alloc.dupe(u8, body),
        };
    }
};

pub fn fetch(
    operation: *Operation,
    cancel_flag: *std.atomic.Value(bool),
    deadline: std.Io.Clock.Timestamp,
) !Response {
    // The bounded runner drains its concurrent request before returning, so
    // the operation and its borrowed header values remain live until then.
    return gateway_client.runBoundedHttpOperation(
        Response,
        operation.alloc,
        cancel_flag,
        deadline,
        operation,
    );
}

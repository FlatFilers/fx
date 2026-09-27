//! The module boundary, checked on every test run: files under `src/` import
//! only the standard library, the build options, or a sibling in `src/`.
//! Nothing from fx reaches in, and nothing here reaches out.

const std = @import("std");
const build_options = @import("build_options");

const Violation = struct {
    import: []const u8,
};

/// Pure: returns the first import in `source` that crosses the boundary.
fn firstViolation(source: []const u8) ?Violation {
    const marker = "@import(\"";
    var at: usize = 0;
    while (std.mem.findPos(u8, source, at, marker)) |start| {
        const name_start = start + marker.len;
        const name_end = std.mem.findScalarPos(u8, source, name_start, '"') orelse
            return .{ .import = source[name_start..] };
        const name = source[name_start..name_end];
        if (!allowed(name)) return .{ .import = name };
        at = name_end;
    }
    return null;
}

fn allowed(name: []const u8) bool {
    const packages = [_][]const u8{ "std", "builtin", "build_options" };
    for (packages) |package| {
        if (std.mem.eql(u8, name, package)) return true;
    }
    // A sibling file: `x.zig` with no path separator and no parent reference.
    return std.mem.endsWith(u8, name, ".zig") and
        std.mem.findScalar(u8, name, '/') == null and
        std.mem.findScalar(u8, name, '\\') == null;
}

test "boundary scanner accepts std and siblings" {
    try std.testing.expectEqual(@as(?Violation, null), firstViolation(
        \\const std = @import("std");
        \\const log = @import("log.zig");
        \\const options = @import("build_options");
    ));
}

test "boundary scanner rejects parent paths and foreign packages" {
    const up = firstViolation("const x = @import(\"../fx/src/main.zig\");").?;
    try std.testing.expectEqualStrings("../fx/src/main.zig", up.import);
    const foreign = firstViolation("const z = @import(\"zero\");").?;
    try std.testing.expectEqualStrings("zero", foreign.import);
    const unterminated = firstViolation("@import(\"std").?;
    try std.testing.expectEqualStrings("std", unterminated.import);
}

test "every file under src stays inside the boundary" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var dir = try std.Io.Dir.cwd().openDir(io, build_options.src_dir, .{ .iterate = true });
    defer dir.close(io);
    var it = dir.iterate();
    var files: usize = 0;
    while (try it.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".zig")) continue;
        const source = try dir.readFileAlloc(io, entry.name, gpa, .limited(4 << 20));
        defer gpa.free(source);
        if (firstViolation(source)) |violation| {
            std.debug.print("{s} imports \"{s}\"\n", .{ entry.name, violation.import });
            return error.BoundaryViolation;
        }
        files += 1;
    }
    try std.testing.expect(files >= 2);
}

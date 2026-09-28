//! Builds git invocations for fx's own read-only work: workspace discovery,
//! search, prompt snapshots, and shell commands fx runs without review.
//!
//! Repository config is untrusted when fx starts in a directory that arrived
//! with its `.git` directory, such as an extracted archive. Several config
//! keys name programs that ordinary read commands run. Configuration passed
//! with `-c` takes precedence over repository config, so every invocation
//! starts with `global_options`, and commands that compare file contents also
//! add `repositoryFilterOverrides`.

const std = @import("std");
const builtin = @import("builtin");
const io_mod = @import("../shared/io.zig");

const Allocator = std.mem.Allocator;

/// Options placed before the subcommand of every fx-owned git invocation.
/// `core.fsmonitor` and `core.hooksPath` otherwise run configured programs
/// while git reads or refreshes the index. `safe.bareRepository=explicit`
/// keeps git from adopting a bare repository nested inside a checkout.
pub const global_options = [_][]const u8{
    "--no-pager",
    "--no-optional-locks",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "safe.bareRepository=explicit",
};

/// Fixed-length argv holding the executable, `global_options`, and a tail.
pub fn Argv(comptime tail_len: usize) type {
    return [1 + global_options.len + tail_len][]const u8;
}

/// Returns `executable`, `global_options`, then `tail`. The result borrows
/// `executable`.
pub fn argv(executable: []const u8, comptime tail: []const []const u8) Argv(tail.len) {
    return [_][]const u8{executable} ++ global_options ++ tail[0..tail.len].*;
}

/// Appends `executable` and `global_options` to `list`. The appended entries
/// borrow `executable` and static strings.
pub fn appendPrefix(
    alloc: Allocator,
    list: *std.ArrayList([]const u8),
    executable: []const u8,
) Allocator.Error!void {
    try list.append(alloc, executable);
    try list.appendSlice(alloc, &global_options);
}

/// Returns the first git executable at a fixed system location, so a
/// directory on PATH cannot substitute its own `git`.
pub fn trustedExecutable() ?[]const u8 {
    const candidates = switch (builtin.os.tag) {
        .windows => &[_][]const u8{
            "C:\\Program Files\\Git\\cmd\\git.exe",
            "C:\\Program Files\\Git\\bin\\git.exe",
        },
        else => &[_][]const u8{
            "/usr/bin/git",
            "/bin/git",
            "/usr/local/bin/git",
            "/opt/homebrew/bin/git",
            "/opt/local/bin/git",
            "/run/current-system/sw/bin/git",
        },
    };
    for (candidates) |candidate| {
        const stat = std.Io.Dir.cwd().statFile(io_mod.getIo(), candidate, .{ .follow_symlinks = true }) catch continue;
        if (stat.kind == .file) return candidate;
    }
    return null;
}

pub const FilterOverrideError = Allocator.Error || error{RepositoryFiltersUnverified};

const filter_query_stdout_limit: usize = 64 * 1024;
const filter_driver_settings = [_][]const u8{ "clean=", "process=", "required=false" };

/// Returns `-c` options that disable each filter driver defined in repository
/// config: `.git/config`, worktree config, and files either one includes.
/// `git status` and `git diff` otherwise run a driver's clean or process
/// command while comparing file contents. Drivers from user config, such as
/// Git LFS, stay active. Returns `error.RepositoryFiltersUnverified` when the
/// config cannot be read or names a driver that `-c` cannot address.
///
/// Runs `executable` in `cwd` with `environ_map`, or with the inherited
/// environment when it is null, so the query sees the same config as the
/// caller's command. The returned slice and its strings are owned by `arena`.
pub fn repositoryFilterOverrides(
    arena: Allocator,
    executable: []const u8,
    cwd: []const u8,
    environ_map: ?*const std.process.Environ.Map,
) FilterOverrideError![]const []const u8 {
    const query = argv(executable, &.{
        "config",
        "-z",
        "--show-scope",
        "--name-only",
        "--get-regexp",
        "^filter\\..*\\.(clean|process)$",
    });
    const result = std.process.run(arena, io_mod.getIo(), .{
        .argv = &query,
        .cwd = .{ .path = cwd },
        .environ_map = environ_map,
        .stdout_limit = .limited(filter_query_stdout_limit),
        .stderr_limit = .limited(1024),
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.RepositoryFiltersUnverified,
    };
    switch (result.term) {
        .exited => |code| switch (code) {
            0 => return parseFilterOverrides(arena, result.stdout),
            // No filter key in any scope, which includes running outside a repository.
            1 => return &.{},
            else => return error.RepositoryFiltersUnverified,
        },
        .signal, .stopped, .unknown => return error.RepositoryFiltersUnverified,
    }
}

/// Parses `git config -z --show-scope --name-only` output, a sequence of
/// NUL-terminated scope and key pairs.
fn parseFilterOverrides(arena: Allocator, output: []const u8) FilterOverrideError![]const []const u8 {
    var overrides: std.ArrayList([]const u8) = .empty;
    var drivers: std.ArrayList([]const u8) = .empty;
    var fields = std.mem.splitScalar(u8, output, 0);
    while (fields.next()) |scope| {
        if (scope.len == 0 and fields.peek() == null) break;
        const key = fields.next() orelse return error.RepositoryFiltersUnverified;
        if (!isRepositoryScope(scope)) continue;

        const driver = filterDriver(key) orelse return error.RepositoryFiltersUnverified;
        // `-c` splits the key from the value at the first '='.
        if (std.mem.findScalar(u8, driver, '=') != null) return error.RepositoryFiltersUnverified;
        if (containsString(drivers.items, driver)) continue;
        try drivers.append(arena, driver);

        for (filter_driver_settings) |setting| {
            try overrides.append(arena, "-c");
            try overrides.append(arena, try std.fmt.allocPrint(arena, "filter.{s}.{s}", .{ driver, setting }));
        }
    }
    return overrides.toOwnedSlice(arena);
}

fn isRepositoryScope(scope: []const u8) bool {
    return std.mem.eql(u8, scope, "local") or std.mem.eql(u8, scope, "worktree");
}

fn filterDriver(key: []const u8) ?[]const u8 {
    const prefix = "filter.";
    if (!std.mem.startsWith(u8, key, prefix)) return null;
    const last_dot = std.mem.findScalarLast(u8, key, '.') orelse return null;
    if (last_dot <= prefix.len) return null;
    return key[prefix.len..last_dot];
}

fn containsString(items: []const []const u8, needle: []const u8) bool {
    for (items) |item| {
        if (std.mem.eql(u8, item, needle)) return true;
    }
    return false;
}

/// A repository whose config makes each known git read path run a program
/// that appends to `marker`. Tests assert that the marker never appears.
pub const TrapRepositoryForTest = struct {
    root: []u8,
    marker: []u8,

    pub fn deinit(self: TrapRepositoryForTest, alloc: Allocator) void {
        alloc.free(self.marker);
        alloc.free(self.root);
    }

    /// Rewrites the tracked file so git must compare its contents again.
    pub fn touchTrackedFile(tmp: *std.testing.TmpDir) !void {
        try writeFileForTest(tmp.dir, "repo/tracked.txt", "needle\n", .default_file);
    }

    pub fn markerExists(self: TrapRepositoryForTest) bool {
        _ = std.Io.Dir.cwd().statFile(io_mod.getIo(), self.marker, .{}) catch return false;
        return true;
    }
};

/// Creates `repo/` in `tmp` with a committed `tracked.txt`, then configures
/// an fsmonitor, a filter driver, a textconv driver, an external diff, and a
/// post-index-change hook that each append to `<tmp>/marker`. Returns
/// `error.SkipZigTest` when git is unavailable.
pub fn createTrapRepositoryForTest(alloc: Allocator, tmp: *std.testing.TmpDir) !TrapRepositoryForTest {
    if (comptime builtin.os.tag == .windows or builtin.os.tag == .wasi) return error.SkipZigTest;
    const executable = trustedExecutable() orelse return error.SkipZigTest;

    try tmp.dir.createDirPath(std.testing.io, "repo/hooks");
    try writeFileForTest(tmp.dir, "repo/.gitattributes", "*.txt filter=trap diff=trap\n", .default_file);
    try writeFileForTest(tmp.dir, "repo/tracked.txt", "needle\n", .default_file);

    const base = try io_mod.dirRealpathAlloc(alloc, tmp.dir, ".");
    defer alloc.free(base);
    const root = try std.fs.path.join(alloc, &.{ base, "repo" });
    errdefer alloc.free(root);
    const marker = try std.fs.path.join(alloc, &.{ base, "marker" });
    errdefer alloc.free(marker);

    const hook = try std.fmt.allocPrint(alloc, "#!/bin/sh\necho hook >> '{s}'\n", .{marker});
    defer alloc.free(hook);
    try writeFileForTest(tmp.dir, "repo/hooks/post-index-change", hook, .executable_file);

    try runGitForTest(alloc, executable, root, &.{ "init", "--quiet" });
    try runGitForTest(alloc, executable, root, &.{ "add", "." });
    try runGitForTest(alloc, executable, root, &.{
        "-c", "user.name=fx", "-c", "user.email=fx@example.invalid", "commit", "--quiet", "-m", "trap",
    });

    const fsmonitor = try std.fmt.allocPrint(alloc, "echo fsmonitor >> '{s}'; false", .{marker});
    defer alloc.free(fsmonitor);
    const clean = try std.fmt.allocPrint(alloc, "sh -c 'echo filter >> \"{s}\"; cat'", .{marker});
    defer alloc.free(clean);
    const textconv = try std.fmt.allocPrint(alloc, "sh -c 'echo textconv >> \"{s}\"; cat'", .{marker});
    defer alloc.free(textconv);
    const external = try std.fmt.allocPrint(alloc, "sh -c 'echo external-diff >> \"{s}\"'", .{marker});
    defer alloc.free(external);
    const hooks_path = try std.fs.path.join(alloc, &.{ root, "hooks" });
    defer alloc.free(hooks_path);

    const settings = [_][2][]const u8{
        .{ "core.fsmonitor", fsmonitor },
        .{ "filter.trap.clean", clean },
        .{ "filter.trap.required", "true" },
        .{ "diff.trap.textconv", textconv },
        .{ "diff.external", external },
        .{ "core.hooksPath", hooks_path },
    };
    for (settings) |setting| {
        try runGitForTest(alloc, executable, root, &.{ "config", setting[0], setting[1] });
    }

    return .{ .root = root, .marker = marker };
}

fn writeFileForTest(
    dir: std.Io.Dir,
    sub_path: []const u8,
    content: []const u8,
    permissions: std.Io.File.Permissions,
) !void {
    var file = try dir.createFile(std.testing.io, sub_path, .{ .permissions = permissions });
    defer file.close(std.testing.io);
    try file.writeStreamingAll(std.testing.io, content);
}

fn runGitForTest(alloc: Allocator, executable: []const u8, cwd: []const u8, args: []const []const u8) !void {
    var command: std.ArrayList([]const u8) = .empty;
    defer command.deinit(alloc);
    try command.append(alloc, executable);
    try command.appendSlice(alloc, args);

    const result = std.process.run(alloc, std.testing.io, .{
        .argv = command.items,
        .cwd = .{ .path = cwd },
        .stdout_limit = .limited(4096),
        .stderr_limit = .limited(4096),
    }) catch return error.SkipZigTest;
    defer alloc.free(result.stdout);
    defer alloc.free(result.stderr);

    switch (result.term) {
        .exited => |code| if (code != 0) return error.SkipZigTest,
        .signal, .stopped, .unknown => return error.SkipZigTest,
    }
}

fn runHardenedForTest(
    alloc: Allocator,
    cwd: []const u8,
    overrides: []const []const u8,
    args: []const []const u8,
) !u8 {
    const executable = trustedExecutable() orelse return error.SkipZigTest;
    var command: std.ArrayList([]const u8) = .empty;
    defer command.deinit(alloc);
    try appendPrefix(alloc, &command, executable);
    try command.appendSlice(alloc, overrides);
    try command.appendSlice(alloc, args);

    const result = try std.process.run(alloc, std.testing.io, .{
        .argv = command.items,
        .cwd = .{ .path = cwd },
        .stdout_limit = .limited(64 * 1024),
        .stderr_limit = .limited(4096),
    });
    defer alloc.free(result.stdout);
    defer alloc.free(result.stderr);
    return switch (result.term) {
        .exited => |code| code,
        .signal, .stopped, .unknown => error.TestUnexpectedResult,
    };
}

test "git argv places hardening options before the subcommand" {
    const command = argv("/usr/bin/git", &.{ "ls-files", "-z" });
    try std.testing.expectEqualStrings("/usr/bin/git", command[0]);
    try std.testing.expectEqualSlices([]const u8, &global_options, command[1 .. 1 + global_options.len]);
    try std.testing.expectEqualStrings("ls-files", command[1 + global_options.len]);
    try std.testing.expectEqualStrings("-z", command[command.len - 1]);
}

test "filter overrides disable repository drivers once and keep user drivers" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const overrides = try parseFilterOverrides(
        arena_state.allocator(),
        "global\x00filter.lfs.clean\x00local\x00filter.Trap.clean\x00local\x00filter.Trap.process\x00" ++
            "worktree\x00filter.dotted.name.process\x00command\x00filter.cli.clean\x00",
    );
    const expected = [_][]const u8{
        "-c", "filter.Trap.clean=",
        "-c", "filter.Trap.process=",
        "-c", "filter.Trap.required=false",
        "-c", "filter.dotted.name.clean=",
        "-c", "filter.dotted.name.process=",
        "-c", "filter.dotted.name.required=false",
    };
    try std.testing.expectEqual(expected.len, overrides.len);
    for (expected, overrides) |want, got| try std.testing.expectEqualStrings(want, got);
}

test "filter overrides reject output they cannot address" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    try std.testing.expectEqual(@as(usize, 0), (try parseFilterOverrides(arena, "")).len);
    const rejected = [_][]const u8{
        "local\x00filter.a=b.clean\x00",
        "local\x00filter.clean\x00",
        "local\x00core.fsmonitor\x00",
        "local\x00",
    };
    for (rejected) |output| {
        try std.testing.expectError(error.RepositoryFiltersUnverified, parseFilterOverrides(arena, output));
    }
}

test "trap repository runs programs through plain git but not hardened git" {
    const alloc = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const trap = try createTrapRepositoryForTest(alloc, &tmp);
    defer trap.deinit(alloc);

    // Plain git runs the configured monitor, which proves the trap is armed.
    try runGitForTest(alloc, trustedExecutable().?, trap.root, &.{ "ls-files", "-z" });
    try std.testing.expect(trap.markerExists());
    try tmp.dir.deleteFile(std.testing.io, "marker");

    var arena_state = std.heap.ArenaAllocator.init(alloc);
    defer arena_state.deinit();
    const overrides = try repositoryFilterOverrides(arena_state.allocator(), trustedExecutable().?, trap.root, null);
    try std.testing.expect(overrides.len > 0);

    const commands = [_][]const []const u8{
        &.{ "ls-files", "-z", "--cached", "--others", "--exclude-standard" },
        &.{ "status", "--short", "--ignore-submodules=dirty" },
        &.{ "diff", "--no-ext-diff", "--no-textconv", "--ignore-submodules=dirty" },
        &.{ "diff", "--no-ext-diff", "--no-textconv", "--stat", "--ignore-submodules=dirty" },
    };
    for (commands) |command| {
        try TrapRepositoryForTest.touchTrackedFile(&tmp);
        try std.testing.expectEqual(@as(u8, 0), try runHardenedForTest(alloc, trap.root, overrides, command));
        try std.testing.expect(!trap.markerExists());
    }
}

//! The one door between fx and the session manager (sessions v2).
//!
//! Behind `--sessions-v2` or FX_SESSIONS_V2=1, with one backend per process.
//! Hosts call this file; it is the only fx file that imports
//! `session_manager`, and the boundary test at the bottom keeps it so. v1
//! code is never called from here except for its encoders and its turn
//! builder, so both backends store the same bytes for the same history.
//!
//! What goes where:
//! - each completed piece of a turn is one `item` whose `type` is the v1
//!   piece kind and whose `data` is the v1 payload;
//! - a turn ends with a `turn_end` item and `turn_committed`, or an
//!   `interruption` item and `turn_interrupted` (`cancel` or `failed`);
//! - preferences, permissions, conversation language, title and usage are
//!   `set` values in v1's encodings;
//! - side files (tool results, images, command logs, artifacts) live in
//!   `~/.fx/session-files/{id}/` until the default flips.

const std = @import("std");
const sm = @import("session_manager");
const debug_trace = @import("../shared/debug_trace.zig");
const io_mod = @import("../shared/io.zig");
const types = @import("../shared/types.zig");
const profile_paths = @import("../shared/profile_paths.zig");
const session_event = @import("session_event.zig");
const result_store = @import("result_store.zig");
const session_log = @import("session_log.zig");
const session_codec = @import("session_codec.zig");
const session_usage = @import("session_usage.zig");
const session_layout = @import("session_layout.zig");
const session_child_store = @import("session_child_store.zig");
const session_display_metadata = @import("session_display_metadata.zig");
const session_store = @import("session_store.zig");
const session_summary_codec = @import("session_summary_codec.zig");
const session_permission_state = @import("../permissions/session_permission_state.zig");
const model_provider = @import("../config/model_provider.zig");

const Allocator = std.mem.Allocator;
const Event = session_event.ConversationEvent;
const PieceKind = std.meta.Tag(Event);

/// Folder of the side files of v2 sessions, under `~/.fx`.
pub const files_dir_name = profile_paths.session_files_dir_name;
/// Folder of the usage-recovery markers of v2 sessions, under `~/.fx`;
/// v1's readers load only v1 sessions, so v2 markers live apart.
pub const usage_markers_dir_name = "usage-recovery-v2";
/// The profile's home folder, opened for listing: creating `~/.fx` in it
/// syncs it, and Linux cannot sync a folder opened any other way (`O_PATH`).
fn openHome(home: []const u8) !io_mod.VerifiedDir {
    return .{ .dir = try std.Io.Dir.openDirAbsolute(io_mod.getIo(), home, .{ .iterate = true, .follow_symlinks = false }) };
}

/// Pieces larger than this go to a blob; the line keeps a reference.
const max_inline_piece_bytes: usize = 256 * 1024;
/// Lines per page when replaying history.
const replay_page_lines: usize = 256;

/// Whether this process keeps its sessions in v2: the flag, or
/// FX_SESSIONS_V2 set to `1` or `true`.
pub fn enabled(flag: bool) bool {
    if (flag) return true;
    const value = io_mod.getenv("FX_SESSIONS_V2") orelse return false;
    return std.mem.eql(u8, value, "1") or std.ascii.eqlIgnoreCase(value, "true");
}

pub const Host = sm.Host;
pub const ChildOutcome = sm.Outcome;

/// One child as its parent's log folds it (D22). Owns its strings.
pub const Child = struct {
    id: []u8,
    /// The child's newest work item.
    work_id: []u8,
    /// That work item is spawned and not finished.
    open: bool,
    /// How the newest finished work item ended; null before the first.
    outcome: ?ChildOutcome,
    /// fx's data on the newest `child_spawned` and `child_finished`.
    spawn_data: ?[]u8,
    finish_data: ?[]u8,
    /// Seq of the newest line about this child.
    seq: u64,
};

pub fn freeChildren(alloc: Allocator, children: []Child) void {
    for (children) |child| {
        alloc.free(child.id);
        alloc.free(child.work_id);
        if (child.spawn_data) |data| alloc.free(data);
        if (child.finish_data) |data| alloc.free(data);
    }
    alloc.free(children);
}

/// A line about a child, appended through its parent.
pub const ChildLine = union(enum) {
    spawned: struct { child: []const u8, work_id: []const u8, data: ?[]const u8 = null },
    finished: struct { child: []const u8, work_id: []const u8, outcome: ChildOutcome, data: ?[]const u8 = null },
};

// ---------------------------------------------------------------------------
// Store: one per process

pub const Store = struct {
    manager: *sm.Manager,
    /// `$HOME`, owned: the base of `~/.fx/sessions/v2` and of the side folders.
    home: []u8,

    /// Touches no disk: a session's folder appears with its first turn.
    pub fn open(alloc: Allocator, home: []const u8) !Store {
        // Resolved as v1 resolves its sessions root, so a home reached
        // through a symlink (macOS `/var`) passes the no-follow opens below.
        const owned_home = io_mod.realpathAlloc(alloc, home) catch |err| blk: {
            debug_trace.logf("session", "event=sessions_v2_home_unresolved err={s} using=given", .{@errorName(err)});
            break :blk try alloc.dupe(u8, home);
        };
        errdefer alloc.free(owned_home);
        const root = try std.fs.path.join(alloc, &.{
            home,
            profile_paths.root_dir_name,
            profile_paths.sessions_dir_name,
            session_layout.sessions_v2_dir,
        });
        defer alloc.free(root);
        const manager = try sm.Manager.init(alloc, io_mod.getIo(), .{
            .root = root,
            .diagnostics = .{ .context = null, .emit = traceDiagnostic },
        });
        return .{ .manager = manager, .home = owned_home };
    }

    /// `$HOME` from the environment.
    pub fn openFromEnv(alloc: Allocator) !Store {
        return open(alloc, io_mod.getenv("HOME") orelse return error.HomeNotSet);
    }

    /// Every Session must be closed first.
    pub fn deinit(store: *Store, alloc: Allocator) void {
        store.manager.deinit();
        alloc.free(store.home);
        store.* = undefined;
    }

    /// A child's whole history, read without its lock: every turn, oldest
    /// first. Free with `types.freeHistoryTurnSlice`.
    /// `~/.fx/session-files`, or null when nothing has used it yet.
    fn openFilesRoot(store: *Store) !?io_mod.VerifiedDir {
        var home = io_mod.VerifiedDir{ .dir = try std.Io.Dir.openDirAbsolute(io_mod.getIo(), store.home, .{ .follow_symlinks = false }) };
        defer home.close();
        var fx = try io_mod.openVerifiedPrivateDirIfPresent(&home, profile_paths.root_dir_name) orelse return null;
        defer fx.close();
        return io_mod.openVerifiedPrivateDirIfPresent(&fx, files_dir_name);
    }

    /// Removes `~/.fx/session-files/{id}`; `why` names the caller in the trace.
    fn removeFiles(store: *Store, id: []const u8, why: []const u8) void {
        var files = (store.openFilesRoot() catch |err| {
            debug_trace.logf("session", "event=sessions_v2_files_kept session={s} why={s} err={s}", .{ id, why, @errorName(err) });
            return;
        }) orelse return;
        defer files.close();
        files.dir.deleteTree(io_mod.getIo(), id) catch |err| {
            debug_trace.logf("session", "event=sessions_v2_files_kept session={s} why={s} err={s}", .{ id, why, @errorName(err) });
            return;
        };
        debug_trace.logf("session", "event=sessions_v2_files_removed session={s} why={s}", .{ id, why });
    }

    /// Copies `session-files/{from}` to `session-files/{to}`: plain files and
    /// folders only, never through a link. False when anything was left out;
    /// a source with no side files has nothing to copy.
    fn copyFiles(store: *Store, from: []const u8, to: []const u8) bool {
        var files = (store.openFilesRoot() catch |err| return copyFailed(from, err)) orelse return true;
        defer files.close();
        var source = (io_mod.openVerifiedPrivateDirIfPresent(&files, from) catch |err| return copyFailed(from, err)) orelse return true;
        defer source.close();
        var target = io_mod.openOrCreateVerifiedPrivateDir(&files, to) catch |err| return copyFailed(from, err);
        defer target.close();
        return copyTree(&source, &target, 0);
    }

    /// Removes side folders whose session is gone, except young ones (D36).
    fn sweepFiles(store: *Store, alloc: Allocator, report: *Doctor, now_ms: i64) !void {
        var files = try store.openFilesRoot() orelse return;
        defer files.close();
        var it = files.dir.iterate();
        while (try it.next(io_mod.getIo())) |entry| {
            if (entry.kind != .directory) continue;
            session_layout.validateSessionId(entry.name) catch continue;
            var probe = store.manager.read(alloc, entry.name, .start, .forward, 1) catch |err| switch (err) {
                error.NotFound => null,
                else => continue,
            };
            if (probe) |*page| {
                page.deinit();
                continue;
            }
            const stat = files.dir.statFile(io_mod.getIo(), entry.name, .{ .follow_symlinks = false }) catch {
                report.kept += 1;
                continue;
            };
            const changed_ms: i64 = @intCast(@divFloor(stat.mtime.toNanoseconds(), std.time.ns_per_ms));
            if (now_ms - changed_ms < orphan_min_age_ms) {
                debug_trace.logf("session", "event=sessions_v2_orphan_files_young session={s}", .{entry.name});
                continue;
            }
            files.dir.deleteTree(io_mod.getIo(), entry.name) catch |err| {
                debug_trace.logf("session", "event=sessions_v2_files_kept session={s} why=orphan err={s}", .{ entry.name, @errorName(err) });
                report.kept += 1;
                continue;
            };
            debug_trace.logf("session", "event=sessions_v2_files_removed session={s} why=orphan", .{entry.name});
            report.removed += 1;
        }
    }

    pub fn childHistory(store: *Store, alloc: Allocator, child_id: []const u8) ![]types.HistoryTurn {
        var history: std.ArrayList(types.HistoryTurn) = .empty;
        errdefer {
            for (history.items) |turn| types.freeHistoryTurn(alloc, turn);
            history.deinit(alloc);
        }
        const Sink = struct {
            alloc: Allocator,
            history: *std.ArrayList(types.HistoryTurn),

            fn turn(sink: *@This(), value: types.HistoryTurn, _: ?u64) !void {
                errdefer types.freeHistoryTurn(sink.alloc, value);
                try sink.history.append(sink.alloc, value);
            }
        };
        var sink: Sink = .{ .alloc = alloc, .history = &history };
        try replay(.{ .store = store, .id = child_id }, alloc, alloc, .start, null, &sink);
        return history.toOwnedSlice(alloc);
    }

    /// A child's newest preferences, read without its lock. Caller owns.
    pub fn childPreferences(store: *Store, alloc: Allocator, child_id: []const u8) !session_codec.DurableSessionPreferences {
        var from: sm.From = .end;
        while (true) {
            var page = try store.manager.read(alloc, child_id, from, .backward, replay_page_lines);
            defer page.deinit();
            for (page.entries) |entry| {
                const body = entry.body orelse continue;
                if (body == .set and body.set.key == .prefs) return decodePreferences(alloc, body.set.value);
            }
            from = .{ .at = page.next orelse return error.InvalidSessionFormat };
        }
    }
};

/// Every repair or drop the manager makes reaches the trace log.
fn traceDiagnostic(_: ?*anyopaque, event: sm.Diagnostic) void {
    debug_trace.logf("session", "event=sessions_v2_diagnostic kind={s} session={s} count={d} offset={d}", .{
        @tagName(event.kind), event.session_id, event.count, event.offset,
    });
}

// ---------------------------------------------------------------------------
// Session: one per open session

/// Settings a new session starts with; held in memory until its first turn.
// ---------------------------------------------------------------------------
// Commands: `fx session {id}`, `fx session recover`, doctor

/// What the session commands report, in the names fx's CLI knows.
pub const CommandError = error{
    SessionNotFound,
    InvalidSessionId,
    SessionBusy,
    InvalidSessionFormat,
    UnsupportedSessionSchema,
    SessionRecoveryBoundaryInvalid,
    SessionStoreUnavailable,
    DurablePathUnsafe,
    HomeNotSet,
    OutOfMemory,
};

/// Maps an error from `Store.openFromEnv`, `listPage`, `readSession`,
/// `recover` or `doctor` onto `CommandError`: a storage fault leaves the
/// store unavailable, and anything else a record fails with is damage.
pub fn commandError(err: anyerror) CommandError {
    return switch (err) {
        error.SessionNotFound, error.NotFound, error.ChildSession => error.SessionNotFound,
        error.InvalidArgument, error.InvalidSessionId => error.InvalidSessionId,
        error.Busy, error.SessionBusy => error.SessionBusy,
        error.UnsupportedVersion => error.UnsupportedSessionSchema,
        error.Corrupt => error.InvalidSessionFormat,
        // `recover` of a session whose first turn never ended whole (D15).
        error.InvalidForkPoint => error.SessionRecoveryBoundaryInvalid,
        error.DurablePathUnsafe, error.SessionPathUnsafe => error.DurablePathUnsafe,
        error.HomeNotSet => error.HomeNotSet,
        error.OutOfMemory => error.OutOfMemory,
        error.Io, error.NoSpaceLeft, error.AccessDenied, error.ReadOnlyFileSystem, error.FileTooBig => blk: {
            debug_trace.logf("session", "event=sessions_v2_command_failed kind=store err={s}", .{@errorName(err)});
            break :blk error.SessionStoreUnavailable;
        },
        else => blk: {
            debug_trace.logf("session", "event=sessions_v2_command_failed kind=record err={s}", .{@errorName(err)});
            break :blk error.InvalidSessionFormat;
        },
    };
}

/// A saved root session as v1's state, read without its lock while another
/// process may hold it (D37), for `fx session {id}`. A missing session, a
/// child, or an id that cannot name one is `error.SessionNotFound`. Caller
/// owns the result.
pub fn readSession(store: *Store, alloc: Allocator, id: []const u8) !Resumed {
    var peeked = store.manager.peek(alloc, id) catch |err| return switch (err) {
        error.NotFound, error.InvalidArgument => error.SessionNotFound,
        else => err,
    };
    defer peeked.deinit(alloc);
    if (peeked.role != .root) return error.SessionNotFound;
    var scratch = std.heap.ArenaAllocator.init(alloc);
    defer scratch.deinit();
    var restored = try restoreFrom(.{ .store = store, .id = id }, alloc, scratch.allocator(), peeked.state, null);
    defer restored.deinit(alloc);
    return resumedOf(alloc, &restored, id, peeked.workspace);
}

/// A page of saved root sessions as v1 pages them: newest first, one
/// workspace when `workspace` is set, and only what follows `continuation`.
/// Caller owns the page.
pub fn listPage(
    store: *Store,
    alloc: Allocator,
    workspace: ?[]const u8,
    continuation: ?session_store.ResumableSessionContinuation,
    limit: usize,
) !session_store.SessionListPage {
    if (limit == 0 or limit > session_store.session_list_max_limit) return error.InvalidSessionListLimit;
    var cancel = std.atomic.Value(bool).init(false);
    var summaries = try listSummaries(store, alloc, null, &cancel);
    defer {
        for (summaries.items) |*summary| summary.deinit(alloc);
        summaries.deinit(alloc);
    }
    session_summary_codec.sortSummariesNewestFirst(summaries.items);
    return session_summary_codec.sessionListPageFromSummaries(alloc, summaries.items, workspace, continuation, limit);
}

pub const Recovered = struct {
    id: []u8,
    /// Entries in the copy's history; a compaction summary counts as one.
    history_len: usize,
    /// False when a side file could not be copied.
    files_complete: bool,

    pub fn deinit(recovered: *Recovered, alloc: Allocator) void {
        alloc.free(recovered.id);
        recovered.* = undefined;
    }
};

/// `fx session recover` (D15): a new root session copied from `id` up to
/// its last turn that ended before any damage, with a copy of its side
/// files. The source is never changed, and may be held by another process.
/// Caller owns the result.
pub fn recover(store: *Store, alloc: Allocator, id: []const u8) !Recovered {
    var scratch = std.heap.ArenaAllocator.init(alloc);
    defer scratch.deinit();
    const sa = scratch.allocator();
    // Line 1 names the session's kind and workspace, however damaged the rest is.
    var first = store.manager.read(sa, id, .start, .forward, 1) catch |err| return switch (err) {
        error.NotFound, error.InvalidArgument => error.SessionNotFound,
        else => err,
    };
    defer first.deinit();
    if (first.entries.len == 0) return error.SessionNotFound;
    const created = switch (first.entries[0].body orelse return error.InvalidSessionFormat) {
        .session_created => |value| value,
        else => return error.InvalidSessionFormat,
    };
    if (created.role != .root) return error.SessionNotFound;
    const copy = store.manager.openFork(.{ .source = id, .at = .last_good, .workspace = created.workspace, .host = .ask }) catch |err| return switch (err) {
        error.NotFound, error.ChildSession => error.SessionNotFound,
        else => err,
    };
    defer copy.release();
    var restored = try restoreFrom(.{ .store = store, .id = copy.id(), .handle = copy }, alloc, sa, try copy.state(sa), null);
    defer restored.deinit(alloc);
    const copy_id = try alloc.dupe(u8, copy.id());
    errdefer alloc.free(copy_id);
    copy.close() catch |err| debug_trace.logf("session", "event=sessions_v2_recovered_close_failed session={s} err={s}", .{ copy_id, @errorName(err) });
    return .{ .id = copy_id, .history_len = restored.history.len, .files_complete = store.copyFiles(id, copy_id) };
}

pub const Doctor = struct {
    /// Saved root sessions.
    sessions: usize = 0,
    /// The most recently updated one.
    latest: ?[]u8 = null,
    /// Sessions whose log was checked, up to the limit.
    checked: usize = 0,
    /// Checked sessions with a damaged line or a stale snapshot.
    damaged: std.ArrayList([]u8) = .empty,
    /// Side folders removed because their session is gone.
    removed: usize = 0,
    /// Side folders with no session that could not be removed.
    kept: usize = 0,

    pub fn deinit(report: *Doctor, alloc: Allocator) void {
        if (report.latest) |id| alloc.free(id);
        for (report.damaged.items) |id| alloc.free(id);
        report.damaged.deinit(alloc);
        report.* = undefined;
    }
};

/// A side folder younger than this may belong to a new session whose first
/// turn has not created its log yet (D36).
const orphan_min_age_ms: i64 = 24 * std.time.ms_per_hour;

/// `fx doctor` on v2 (D36): verifies up to `limit` sessions and removes side
/// folders whose session is gone. Rebuilds nothing. Caller owns the report.
pub fn doctor(store: *Store, alloc: Allocator, limit: usize, now_ms: i64) !Doctor {
    var report: Doctor = .{};
    errdefer report.deinit(alloc);
    var latest_ms: u64 = 0;
    var cursor: ?sm.ListCursor = null;
    while (true) {
        var page = try store.manager.list(alloc, .all, cursor, list_page_size);
        defer page.deinit();
        for (page.items) |item| {
            if (item.role != .root) continue;
            report.sessions += 1;
            if (report.latest == null or item.updated_ms > latest_ms) {
                const id = try alloc.dupe(u8, item.id);
                if (report.latest) |old| alloc.free(old);
                report.latest = id;
                latest_ms = item.updated_ms;
            }
            if (report.checked == limit) continue;
            report.checked += 1;
            const verified = store.manager.verify(item.id) catch |err| switch (err) {
                error.NotFound => continue,
                error.OutOfMemory => return error.OutOfMemory,
                else => {
                    debug_trace.logf("session", "event=sessions_v2_doctor_unreadable session={s} err={s}", .{ item.id, @errorName(err) });
                    try report.damaged.append(alloc, try alloc.dupe(u8, item.id));
                    continue;
                },
            };
            if (verified.damaged_at != null or verified.bad_snapshots > 0) try report.damaged.append(alloc, try alloc.dupe(u8, item.id));
        }
        cursor = page.next orelse break;
    }
    try store.sweepFiles(alloc, &report, now_ms);
    return report;
}

pub const Seed = struct {
    preferences: session_codec.DurableSessionPreferences,
    language: types.ConversationLanguage,
    permission_state: session_permission_state.State,
    /// A child's instructions, kept in its own `prefs` (D34); roots leave
    /// it empty.
    instructions: []const u8 = "",
};

pub const Target = union(enum) {
    id: []const u8,
    /// The newest updated root session in the workspace.
    last,
    /// The session this host last opened in the workspace (`fx -c`).
    last_opened,
};

/// What resume gives back. Owns everything; free with `deinit`.
pub const Restored = struct {
    history: []types.HistoryTurn,
    language: types.ConversationLanguage,
    preferences: ?session_codec.DurableSessionPreferences = null,
    permission_state: ?session_permission_state.State = null,
    usage: ?session_usage.Snapshot = null,
    /// The stored title, generated or chosen by the user.
    title: ?[]u8 = null,
    created_at_ms: i64,
    updated_at_ms: i64 = 0,

    pub fn deinit(restored: *Restored, alloc: Allocator) void {
        types.freeHistoryTurnSlice(alloc, restored.history);
        if (restored.title) |value| alloc.free(value);
        if (restored.preferences) |*value| value.deinit(alloc);
        if (restored.permission_state) |*value| value.deinit(alloc);
        if (restored.usage) |*value| value.deinit(alloc);
        restored.* = undefined;
    }
};

/// A resumed session as v1's state. Owns everything; free with `deinit`.
pub const Resumed = struct {
    state: session_codec.DurableSessionState,
    title: ?[]u8,

    pub fn deinit(resumed: *Resumed, alloc: Allocator) void {
        resumed.state.deinit(alloc);
        if (resumed.title) |value| alloc.free(value);
        resumed.* = undefined;
    }
};

pub const Session = struct {
    alloc: Allocator,
    store: *Store,
    handle: sm.Session,
    /// `~/.fx/session-files/{id}`, owned.
    files_path: []u8,
    files_dir: ?io_mod.VerifiedDir = null,
    capability: ?session_child_store.SessionChildCapability = null,
    /// Serializes this adapter's own state; the manager's Session is
    /// thread-safe on its own.
    mutex: std.Io.Mutex = .init,
    /// Encodings of the open turn's pieces already appended, in order.
    streamed: std.ArrayList([]u8) = .empty,
    /// Result files the open turn's stream wrote, by call id; both owned.
    stored_results: std.StringHashMapUnmanaged([]u8) = .empty,
    /// Ids of the open turn's calls already saved as running (D28), owned.
    running: std.ArrayList([]u8) = .empty,
    turn_open: bool = false,
    /// Highest turn number started; the manager numbers turns the same way.
    last_turn: u64 = 0,
    /// The v2 turn behind each of fx's history turns, in order; null for a
    /// compacted summary.
    turn_numbers: std.ArrayList(?u64) = .empty,
    /// The language tag last written, owned.
    language: ?[]u8 = null,
    /// Started in this process: its first commit may name it.
    fresh: bool,
    titled: bool = false,
    /// A usage-recovery marker protects a checkpoint still waiting for the
    /// profile ledger; it keeps its first time until nothing is pending.
    usage_marked: bool = false,
    /// Time of the newest usage checkpoint; each new one is later.
    usage_at_ms: i64 = 0,
    /// A child's instructions as its `prefs` hold them (D34), owned; null
    /// for a root and for a child without any.
    instructions: ?[]u8 = null,

    pub fn create(alloc: Allocator, store: *Store, workspace: []const u8, host: Host, seed: Seed) !*Session {
        // Every v2 write needs a writable open, and both start here or in
        // resumeSession: E2E tests use this to prove read-only paths never
        // reach one.
        io_mod.e2eFailIfDurableMutationAttempted();
        const handle = try store.manager.openNew(.{ .workspace = workspace, .host = host });
        errdefer handle.release();
        const self = try init(alloc, store, handle, true);
        errdefer self.destroyInner();
        try self.appendSeed(seed);
        return self;
    }

    /// The settings a new session holds until its first turn.
    fn appendSeed(self: *Session, seed: Seed) !void {
        var arena = std.heap.ArenaAllocator.init(self.alloc);
        defer arena.deinit();
        const a = arena.allocator();
        _ = try self.handle.append(&.{
            .{ .set = .{ .key = .prefs, .value = try encodePreferences(a, seed.preferences, self.instructions) } },
            .{ .set = .{ .key = .permissions, .value = try session_codec.encodePermissionState(a, seed.permission_state) } },
            .{ .set = .{ .key = .language, .value = try jsonString(a, seed.language.view()) } },
        });
        self.language = try self.alloc.dupe(u8, seed.language.view());
    }

    /// Opens a child of `parent_id` for one work item (D22, D34): its log
    /// when it has one, or a new child under `child_id`, the id the parent's
    /// `child_spawned` already names, holding `seed` until its first turn.
    /// Instructions in `seed` replace the stored ones; empty keeps them.
    pub fn openChild(alloc: Allocator, store: *Store, parent_id: []const u8, child_id: []const u8, workspace: []const u8, seed: Seed) !*Session {
        io_mod.e2eFailIfDurableMutationAttempted();
        if (store.manager.openResume(.{ .target = .{ .id = child_id }, .workspace = workspace, .host = .child, .parent = parent_id })) |handle| {
            errdefer handle.release();
            const self = try init(alloc, store, handle, false);
            errdefer self.destroyInner();
            var scratch = std.heap.ArenaAllocator.init(alloc);
            defer scratch.deinit();
            const state = try handle.state(scratch.allocator());
            self.last_turn = state.last_turn;
            // Every child starts with its preferences (`appendSeed`).
            const raw = state.prefs orelse return error.InvalidSessionFormat;
            self.instructions = try decodeInstructions(alloc, raw);
            if (seed.instructions.len > 0 and !std.mem.eql(u8, seed.instructions, self.childInstructions())) {
                var preferences = try decodePreferences(alloc, raw);
                defer preferences.deinit(alloc);
                try self.replaceInstructions(preferences, seed.instructions);
            }
            return self;
        } else |err| switch (err) {
            error.NotFound => {},
            else => return err,
        }
        const handle = try store.manager.openNew(.{ .workspace = workspace, .host = .child, .role = .child, .parent = parent_id, .id = child_id });
        errdefer handle.release();
        const self = try init(alloc, store, handle, false);
        errdefer self.destroyInner();
        if (seed.instructions.len > 0) self.instructions = try alloc.dupe(u8, seed.instructions);
        try self.appendSeed(seed);
        return self;
    }

    /// The session's current preferences. Caller owns.
    pub fn currentPreferences(self: *Session, alloc: Allocator) !session_codec.DurableSessionPreferences {
        var scratch = std.heap.ArenaAllocator.init(alloc);
        defer scratch.deinit();
        const state = try self.handle.state(scratch.allocator());
        return decodePreferences(alloc, state.prefs orelse return error.InvalidSessionFormat);
    }

    /// A child's instructions (D34), or empty. Borrowed until `close`.
    pub fn childInstructions(self: *const Session) []const u8 {
        return self.instructions orelse "";
    }

    fn replaceInstructions(self: *Session, preferences: session_codec.DurableSessionPreferences, text: []const u8) !void {
        const owned = try self.alloc.dupe(u8, text);
        errdefer self.alloc.free(owned);
        var arena = std.heap.ArenaAllocator.init(self.alloc);
        defer arena.deinit();
        _ = try self.handle.append(&.{.{ .set = .{ .key = .prefs, .value = try encodePreferences(arena.allocator(), preferences, owned) } }});
        if (self.instructions) |old| self.alloc.free(old);
        self.instructions = owned;
    }

    pub fn resumeSession(alloc: Allocator, store: *Store, target: Target, workspace: []const u8, host: Host) !*Session {
        return resumeWaiting(alloc, store, target, workspace, host, null);
    }

    /// As `resumeSession`, but SessionBusy at once when another process has
    /// the session open, for the session picker (D38).
    pub fn resumeSessionWithoutWaiting(alloc: Allocator, store: *Store, target: Target, workspace: []const u8, host: Host) !*Session {
        return resumeWaiting(alloc, store, target, workspace, host, 0);
    }

    /// `lock_wait_ms` null waits the manager's 2 s for the writer lock.
    fn resumeWaiting(alloc: Allocator, store: *Store, target: Target, workspace: []const u8, host: Host, lock_wait_ms: ?u64) !*Session {
        io_mod.e2eFailIfDurableMutationAttempted();
        const handle = store.manager.openResume(.{
            .target = switch (target) {
                .id => |session_id| .{ .id = session_id },
                .last => .last,
                .last_opened => .{ .last_opened = host },
            },
            .workspace = workspace,
            .host = host,
            .lock_wait_ms = lock_wait_ms,
        }) catch |err| return resumeError(err, target);
        errdefer handle.release();
        const self = try init(alloc, store, handle, false);
        errdefer self.destroyInner();
        var state = try handle.state(alloc);
        defer state.deinit(alloc);
        self.last_turn = state.last_turn;
        return self;
    }

    fn init(alloc: Allocator, store: *Store, handle: sm.Session, fresh: bool) !*Session {
        const files_path = try std.fs.path.join(alloc, &.{ store.home, profile_paths.root_dir_name, files_dir_name, handle.id() });
        errdefer alloc.free(files_path);
        const self = try alloc.create(Session);
        self.* = .{ .alloc = alloc, .store = store, .handle = handle, .files_path = files_path, .fresh = fresh };
        return self;
    }

    pub fn id(self: *const Session) []const u8 {
        return self.handle.id();
    }

    /// The session is on disk: it has a turn, ended or open.
    pub fn saved(self: *const Session) bool {
        return self.last_turn > 0;
    }

    /// The session's folder under `~/.fx/sessions/v2`, for display. Caller owns it.
    pub fn folderPath(self: *const Session, alloc: Allocator) ![]u8 {
        return std.fs.path.join(alloc, &.{
            self.store.home,
            profile_paths.root_dir_name,
            profile_paths.sessions_dir_name,
            session_layout.sessions_v2_dir,
            self.id(),
        });
    }

    /// `~/.fx/session-files/{id}`: borrowed until `close`.
    pub fn filesPath(self: *const Session) []const u8 {
        return self.files_path;
    }

    /// Closes the session and frees the adapter. Every child thread of this
    /// session must have joined (`tla/Wiring.tla` ParentOutlivesChildren).
    pub fn close(self: *Session) void {
        self.handle.close() catch |err| debug_trace.logf("session", "event=sessions_v2_close_failed session={s} err={s}", .{ self.id(), @errorName(err) });
        if (self.turn_open) debug_trace.logf("session", "event=sessions_v2_turn_closed_open session={s} streamed={d}", .{ self.id(), self.streamed.items.len });
        // Before `release`: the id lives in the handle.
        if (self.files_dir != null and !self.saved()) self.removeUnsavedFiles();
        self.handle.release();
        self.destroyInner();
    }

    /// A session that never reached the disk takes its side folder with it,
    /// as v1 removes a pristine session's folder. Asks the manager first: a
    /// first write that failed may still have landed, and its turn points
    /// into this folder.
    fn removeUnsavedFiles(self: *Session) void {
        var probe = self.store.manager.read(self.alloc, self.id(), .start, .forward, 1) catch |err| switch (err) {
            error.NotFound => null,
            else => {
                debug_trace.logf("session", "event=sessions_v2_unsaved_files_kept session={s} err={s}", .{ self.id(), @errorName(err) });
                return;
            },
        };
        if (probe) |*page| {
            page.deinit();
            return;
        }
        // The capability borrows the folder.
        if (self.capability) |*capability| capability.deinit();
        self.capability = null;
        if (self.files_dir) |*dir| dir.close();
        self.files_dir = null;
        self.store.removeFiles(self.id(), "unsaved");
    }

    fn destroyInner(self: *Session) void {
        const alloc = self.alloc;
        if (self.capability) |*capability| capability.deinit();
        if (self.files_dir) |*dir| dir.close();
        self.clearStreamed();
        self.streamed.deinit(alloc);
        self.stored_results.deinit(alloc);
        self.running.deinit(alloc);
        self.turn_numbers.deinit(alloc);
        if (self.language) |value| alloc.free(value);
        if (self.instructions) |value| alloc.free(value);
        alloc.free(self.files_path);
        alloc.destroy(self);
    }

    /// Forgets what the open turn holds, once it ends or is superseded.
    fn clearStreamed(self: *Session) void {
        for (self.streamed.items) |bytes| self.alloc.free(bytes);
        self.streamed.clearRetainingCapacity();
        var stored = self.stored_results.iterator();
        while (stored.next()) |entry| {
            self.alloc.free(entry.key_ptr.*);
            self.alloc.free(entry.value_ptr.*);
        }
        self.stored_results.clearRetainingCapacity();
        for (self.running.items) |call_id| self.alloc.free(call_id);
        self.running.clearRetainingCapacity();
    }

    // -- side files ----------------------------------------------------------

    /// The side-file capability over `~/.fx/session-files/{id}`, created
    /// `0700` on first use. Borrowed until `close`.
    pub fn childCapability(self: *Session) !*session_child_store.SessionChildCapability {
        self.mutex.lockUncancelable(io_mod.getIo());
        defer self.mutex.unlock(io_mod.getIo());
        return self.capabilityLocked();
    }

    fn capabilityLocked(self: *Session) !*session_child_store.SessionChildCapability {
        if (self.capability) |*capability| return capability;
        const dir = try self.openFilesDir();
        self.capability = try session_child_store.SessionChildCapability.init(self.alloc, dir.dir, self.files_path, .writable);
        return &self.capability.?;
    }

    /// Creates the side folder (`0700`) if it is missing, for writers that
    /// use it before any capability does, such as a prompt's images, and
    /// returns `filesPath()`.
    pub fn ensureFilesPath(self: *Session) ![]const u8 {
        self.mutex.lockUncancelable(io_mod.getIo());
        defer self.mutex.unlock(io_mod.getIo());
        _ = try self.openFilesDir();
        return self.files_path;
    }

    fn openFilesDir(self: *Session) !*io_mod.VerifiedDir {
        if (self.files_dir) |*dir| return dir;
        var home = try openHome(self.store.home);
        defer home.close();
        var fx = try io_mod.openOrCreateVerifiedPrivateDir(&home, profile_paths.root_dir_name);
        defer fx.close();
        var files = try io_mod.openOrCreateVerifiedPrivateDir(&fx, files_dir_name);
        defer files.close();
        self.files_dir = try io_mod.openOrCreateVerifiedPrivateDir(&files, self.id());
        return &self.files_dir.?;
    }

    // -- turns ---------------------------------------------------------------

    /// Stores the turn's tool results and images as side files and gives
    /// them handles, as v1 does at commit. A result whose file the stream
    /// already wrote (`withResultFiles`) keeps it, unwritten a second time.
    pub fn prepareTurn(self: *Session, turn: *types.HistoryTurn) !void {
        try self.reuseStreamedFiles(turn);
        try session_log.externalizeConversationTurnResults(self.alloc, turn, try self.childCapability());
    }

    fn reuseStreamedFiles(self: *Session, turn: *types.HistoryTurn) !void {
        self.mutex.lockUncancelable(io_mod.getIo());
        defer self.mutex.unlock(io_mod.getIo());
        if (self.stored_results.count() == 0) return;
        const execution = switch (turn.*) {
            .assistant => |*entry| &entry.execution,
            .interrupted => |*entry| &entry.execution,
            .compacted_summary => return,
        };
        for (execution.tool_steps) |step| for (step.tool_results) |*result| {
            if (result.output_handle != null) continue;
            const stored = self.stored_results.get(result.tool_call_id) orelse continue;
            // The name holds the content's hash, so a match means the same bytes.
            const handle = try result_store.makeHandle(self.alloc, result.tool_call_id, result.tool_name, result.output);
            if (!std.mem.eql(u8, handle, stored)) {
                self.alloc.free(handle);
                continue;
            }
            result.output_handle = handle;
            result.stored_output_bytes = result.output.len;
        };
    }

    /// Appends a finished turn: the pieces not streamed yet, then its end,
    /// in one durable batch. `turn` must already be prepared.
    pub fn commitTurn(self: *Session, turn: types.HistoryTurn, language: types.ConversationLanguage) !void {
        self.mutex.lockUncancelable(io_mod.getIo());
        defer self.mutex.unlock(io_mod.getIo());
        var arena = std.heap.ArenaAllocator.init(self.alloc);
        defer arena.deinit();
        const a = arena.allocator();

        var events: std.ArrayList(Event) = .empty;
        try session_event.appendHistoryTurnConversationEvents(a, &events, turn);
        if (events.items.len < 2) return error.InvalidConversationFrame;
        const pieces = events.items[0 .. events.items.len - 1];
        const end = events.items[events.items.len - 1];

        // The end piece joins the others; its line ends the turn.
        const all = events.items;
        const encoded = try a.alloc([]u8, all.len);
        for (all, encoded) |piece, *bytes| bytes.* = try encodePiece(a, piece);
        const start = try self.streamedPrefix(encoded[0..pieces.len]);

        var tail: std.ArrayList(sm.Event) = .empty;
        switch (end) {
            .turn_completed => try tail.append(a, .turn_committed),
            .interrupted => |interruption| try tail.append(a, .{ .turn_interrupted = switch (interruption.reason) {
                .cancelled => .cancel,
                .failed => .failed,
            } }),
            else => return error.InvalidConversationFrame,
        }
        const language_tag = language.view();
        const language_changed = self.language == null or !std.mem.eql(u8, self.language.?, language_tag);
        if (language_changed) try tail.append(a, .{ .set = .{ .key = .language, .value = try jsonString(a, language_tag) } });
        const derived_title = if (self.fresh and !self.titled) try deriveTitle(a, turn) else null;
        if (derived_title) |title| try tail.append(a, .{ .set = .{ .key = .title, .value = try jsonString(a, title) } });

        try self.writePieces(a, all[start..], encoded[start..], tail.items);
        try self.turn_numbers.append(self.alloc, self.last_turn);
        self.turn_open = false;
        self.clearStreamed();
        if (language_changed) {
            const owned = try self.alloc.dupe(u8, language_tag);
            if (self.language) |old| self.alloc.free(old);
            self.language = owned;
        }
        if (derived_title != null) self.titled = true;
        self.fresh = false;
    }

    /// Streams the turn so far (`AgentRuntimeDeps.append_turn_piece`): the
    /// first call starts the turn, later calls append only the pieces
    /// completed since. A tool result gets the side file the commit would
    /// write for it (`withResultFiles`); a tool image without its handle
    /// waits for the commit, which is authoritative. `running_calls` are
    /// saved once each as `tool_running` items, outside the streamed pieces,
    /// so a finished step still streams and commits as it always did (D28).
    pub fn appendProgress(
        self: *Session,
        user: types.UserTurn,
        execution: types.ExecutionMemory,
        running_calls: []const types.ToolCall,
    ) !void {
        self.mutex.lockUncancelable(io_mod.getIo());
        defer self.mutex.unlock(io_mod.getIo());
        var arena = std.heap.ArenaAllocator.init(self.alloc);
        defer arena.deinit();
        const a = arena.allocator();
        var events: std.ArrayList(Event) = .empty;
        try events.append(a, .{ .user = .{ .text = user.text, .images = user.images, .work_id = user.work_id } });
        // On a missing handle the list keeps the pieces before it.
        session_event.appendExecutionConversationEvents(a, &events, try self.withResultFiles(a, execution)) catch |err| switch (err) {
            error.ConversationArtifactRequired => {},
            else => return err,
        };
        const encoded = try a.alloc([]u8, events.items.len);
        for (events.items, encoded) |event, *bytes| bytes.* = try encodePiece(a, event);
        const start = try self.streamedPrefix(encoded);
        if (!self.turn_open or start < encoded.len) {
            try self.writePieces(a, events.items[start..], encoded[start..], &.{});
            try self.streamed.ensureUnusedCapacity(self.alloc, encoded.len - start);
            for (encoded[start..]) |bytes| self.streamed.appendAssumeCapacity(try self.alloc.dupe(u8, bytes));
        }
        try self.saveRunning(a, running_calls);
    }

    /// Saves the calls not yet saved as running in the open turn, which the
    /// pieces above have just opened if it was not.
    fn saveRunning(self: *Session, a: Allocator, calls: []const types.ToolCall) !void {
        var batch: std.ArrayList(sm.Event) = .empty;
        var ids: std.ArrayList([]const u8) = .empty;
        for (calls) |call| {
            if (self.isRunning(call.id)) continue;
            const bytes = try encodePiece(a, .{ .tool_call = conversationToolCall(call) });
            try batch.append(a, .{ .item = try self.itemAs(a, running_type, bytes) });
            try ids.append(a, call.id);
        }
        if (batch.items.len == 0) return;
        _ = try self.handle.append(batch.items);
        try self.running.ensureUnusedCapacity(self.alloc, ids.items.len);
        for (ids.items) |call_id| self.running.appendAssumeCapacity(try self.alloc.dupe(u8, call_id));
    }

    fn isRunning(self: *const Session, call_id: []const u8) bool {
        for (self.running.items) |running_id| if (std.mem.eql(u8, running_id, call_id)) return true;
        return false;
    }

    /// A copy of `execution` in which each tool result without a result file
    /// has the one the commit gives it (`session_log.externalizeConversationTurnResults`):
    /// the same handle, size and preview, so its streamed piece is the
    /// committed one. A result backed only by its command replay has no file
    /// before then. Each file is written once a turn.
    fn withResultFiles(self: *Session, a: Allocator, execution: types.ExecutionMemory) !types.ExecutionMemory {
        var copy = execution;
        copy.tool_steps = try a.dupe(types.ToolExecutionStep, execution.tool_steps);
        for (copy.tool_steps) |*step| {
            step.tool_results = try a.dupe(types.PersistedToolResult, step.tool_results);
            for (step.tool_results) |*result| {
                if (result.output_handle != null) continue;
                const handle = self.stored_results.get(result.tool_call_id) orelse blk: {
                    const stored = try result_store.storeLargeResultManaged(self.alloc, try self.capabilityLocked(), result.tool_call_id, result.tool_name, result.output);
                    errdefer self.alloc.free(stored);
                    const key = try self.alloc.dupe(u8, result.tool_call_id);
                    errdefer self.alloc.free(key);
                    try self.stored_results.put(self.alloc, key, stored);
                    break :blk stored;
                };
                // Copied: a superseded turn frees the cache before its pieces are written.
                result.output_handle = try a.dupe(u8, handle);
                result.stored_output_bytes = result.output.len;
                if (result.preview == null) result.preview = try result_store.previewText(a, result.output, result_store.preview_bytes);
            }
        }
        return copy;
    }

    /// Appends `events` (already encoded) as items, then `tail`, starting
    /// the turn first if none is open. One durable batch, except that a
    /// blob needs a published session with an open turn before it.
    fn writePieces(self: *Session, a: Allocator, events: []const Event, encoded: []const []u8, tail: []const sm.Event) !void {
        var batch: std.ArrayList(sm.Event) = .empty;
        if (!self.turn_open) {
            try batch.append(a, .turn_started);
            const needs_blob = for (encoded) |bytes| {
                if (bytes.len > max_inline_piece_bytes) break true;
            } else false;
            if (needs_blob) {
                _ = try self.handle.append(batch.items);
                self.startedTurn();
                batch.clearRetainingCapacity();
            }
        }
        for (events, encoded) |event, bytes| try batch.append(a, .{ .item = try self.item(a, std.meta.activeTag(event), bytes) });
        try batch.appendSlice(a, tail);
        if (batch.items.len == 0) return;
        _ = try self.handle.append(batch.items);
        if (!self.turn_open) self.startedTurn();
    }

    fn startedTurn(self: *Session) void {
        self.last_turn += 1;
        self.turn_open = true;
    }

    /// How many of `encoded` the open turn already holds. A streamed piece
    /// that differs from the final turn is a bug; the stale turn is closed
    /// as superseded and the whole turn is written afresh, never mixed.
    fn streamedPrefix(self: *Session, encoded: []const []u8) !usize {
        if (!self.turn_open) return 0;
        const streamed = self.streamed.items;
        const same = streamed.len <= encoded.len and (for (streamed, encoded[0..streamed.len]) |x, y| {
            if (!samePiece(self.alloc, x, y)) break false;
        } else true);
        if (same) return streamed.len;
        debug_trace.logf("session", "event=sessions_v2_stream_mismatch session={s} streamed={d} final={d} dropped=stale_turn", .{ self.id(), streamed.len, encoded.len });
        _ = try self.handle.append(&.{
            .{ .item = .{ .type = superseded_type, .data = "{}" } },
            .{ .turn_interrupted = .failed },
        });
        self.turn_open = false;
        self.clearStreamed();
        return 0;
    }

    /// One piece as an item; a large one goes to a blob.
    fn item(self: *Session, a: Allocator, kind: PieceKind, bytes: []u8) !sm.Piece {
        return self.itemAs(a, itemType(kind) orelse return error.InvalidConversationFrame, bytes);
    }

    fn itemAs(self: *Session, a: Allocator, item_type: []const u8, bytes: []u8) !sm.Piece {
        if (bytes.len <= max_inline_piece_bytes) return .{ .type = item_type, .data = bytes };
        const hash = try self.handle.putBlob(bytes);
        const hash_copy = try a.dupe(u8, &hash);
        const refs = try a.alloc([]const u8, 1);
        refs[0] = hash_copy;
        return .{
            .type = item_type,
            .data = try std.fmt.allocPrint(a, "{{\"{s}\":\"{s}\"}}", .{ blob_ref_key, hash_copy }),
            .blobs = refs,
        };
    }

    // -- compaction ----------------------------------------------------------

    /// Records a compaction. The history after it starts with the summary,
    /// then the retained turns, which the log already holds.
    pub fn commitCompaction(
        self: *Session,
        summary: types.CompactedSummaryHistoryTurn,
        active_prefix: bool,
        retained_from: ?types.ContextHistoryCut,
    ) !void {
        self.mutex.lockUncancelable(io_mod.getIo());
        defer self.mutex.unlock(io_mod.getIo());
        var arena = std.heap.ArenaAllocator.init(self.alloc);
        defer arena.deinit();
        const cut = retained_from orelse types.ContextHistoryCut{ .turns = self.turn_numbers.items.len };
        const first_kept = @min(cut.turns, self.turn_numbers.items.len);
        var keep_from: ?u64 = null;
        for (self.turn_numbers.items[first_kept..]) |number| {
            if (number) |n| {
                keep_from = n;
                break;
            }
        }
        if (keep_from == null and active_prefix and self.turn_open) keep_from = self.last_turn;
        if (cut.tool_steps != 0 or cut.steering != 0) {
            debug_trace.logf("session", "event=sessions_v2_compaction_cut session={s} kept=whole_turn tool_steps={d} steering={d}", .{ self.id(), cut.tool_steps, cut.steering });
        }
        const data: CompactedData = .{
            .summary = summary.summary,
            .removed_turn_count = summary.removed_turn_count,
            .compaction_count = summary.compaction_count,
            .keep_from_turn = keep_from,
        };
        _ = try self.handle.append(&.{.{ .compacted = try jsonValue(arena.allocator(), data) }});
        // fx's history is now the summary, then the retained turns.
        var kept: std.ArrayList(?u64) = .empty;
        errdefer kept.deinit(self.alloc);
        try kept.append(self.alloc, null);
        try kept.appendSlice(self.alloc, self.turn_numbers.items[first_kept..]);
        self.turn_numbers.deinit(self.alloc);
        self.turn_numbers = kept;
    }

    // -- settings ------------------------------------------------------------

    pub fn setPreferences(self: *Session, preferences: session_codec.DurableSessionPreferences) !void {
        var arena = std.heap.ArenaAllocator.init(self.alloc);
        defer arena.deinit();
        _ = try self.handle.append(&.{.{ .set = .{ .key = .prefs, .value = try encodePreferences(arena.allocator(), preferences, self.instructions) } }});
    }

    // -- children (D22) ------------------------------------------------------

    /// Every child of this session as its log folds them, in first-spawn
    /// order. Free with `freeChildren`.
    pub fn children(self: *Session, alloc: Allocator) ![]Child {
        var scratch = std.heap.ArenaAllocator.init(alloc);
        defer scratch.deinit();
        const state = try self.handle.state(scratch.allocator());
        const out = try alloc.alloc(Child, state.children.items.len);
        var built: usize = 0;
        errdefer freeChildren(alloc, out[0..built]);
        for (state.children.items) |child| {
            const id_copy = try alloc.dupe(u8, child.id);
            errdefer alloc.free(id_copy);
            const work_copy = try alloc.dupe(u8, child.work_id);
            errdefer alloc.free(work_copy);
            const spawn_copy = if (child.spawn_data) |data| try alloc.dupe(u8, data) else null;
            errdefer if (spawn_copy) |data| alloc.free(data);
            out[built] = .{
                .id = id_copy,
                .work_id = work_copy,
                .open = child.open,
                .outcome = child.outcome,
                .spawn_data = spawn_copy,
                .finish_data = if (child.finish_data) |data| try alloc.dupe(u8, data) else null,
                .seq = child.seq,
            };
            built += 1;
        }
        return out;
    }

    /// Appends child lines as one durable batch. The manager refuses a spawn
    /// while that child has unfinished work, and a finish for other work.
    pub fn appendChildLines(self: *Session, lines: []const ChildLine) !void {
        if (lines.len == 0) return;
        const events = try self.alloc.alloc(sm.Event, lines.len);
        defer self.alloc.free(events);
        for (lines, events) |line, *event| event.* = switch (line) {
            .spawned => |spawned| .{ .child_spawned = .{ .child = spawned.child, .work_id = spawned.work_id, .data = spawned.data } },
            .finished => |finished| .{ .child_finished = .{ .child = finished.child, .work_id = finished.work_id, .outcome = finished.outcome, .data = finished.data } },
        };
        _ = try self.handle.append(events);
    }

    pub fn setPermissions(self: *Session, state: session_permission_state.State) !void {
        const value = try session_codec.encodePermissionState(self.alloc, state);
        defer self.alloc.free(value);
        _ = try self.handle.append(&.{.{ .set = .{ .key = .permissions, .value = value } }});
    }

    /// A title the user chose. It differs from the derived one, so a
    /// generated title never replaces it.
    pub fn rename(self: *Session, title: []const u8) !void {
        self.mutex.lockUncancelable(io_mod.getIo());
        defer self.mutex.unlock(io_mod.getIo());
        const value = try jsonString(self.alloc, title);
        defer self.alloc.free(value);
        _ = try self.handle.append(&.{.{ .set = .{ .key = .title, .value = value } }});
        self.titled = true;
    }

    /// v1's rule: a generated title never replaces one the user chose, only
    /// none or the one derived from the first message.
    pub fn installGeneratedTitle(self: *Session, history: []const types.HistoryTurn, title: []const u8) !bool {
        self.mutex.lockUncancelable(io_mod.getIo());
        defer self.mutex.unlock(io_mod.getIo());
        var arena = std.heap.ArenaAllocator.init(self.alloc);
        defer arena.deinit();
        const a = arena.allocator();
        var state = try self.handle.state(a);
        defer state.deinit(a);
        if (state.title) |raw| {
            const current = try std.json.parseFromSliceLeaky([]const u8, a, raw, .{});
            var display = try session_display_metadata.deriveFromHistory(a, history);
            defer display.deinit(a);
            if (!display.present or !std.mem.eql(u8, current, display.title)) {
                debug_trace.logf("session", "event=title_generation_apply result=dropped reason=user_title_present", .{});
                return false;
            }
        }
        _ = try self.handle.append(&.{.{ .set = .{ .key = .title, .value = try jsonString(a, title) } }});
        self.titled = true;
        return true;
    }

    // -- usage ---------------------------------------------------------------

    /// Saves a usage checkpoint with v1's marker rules: a checkpoint that
    /// still owes the profile ledger is covered by a marker written first
    /// (keeping the time of the first such checkpoint), and the marker goes
    /// once a durable checkpoint owes nothing (`tla/Wiring.tla`
    /// UsageNeverSilent).
    pub fn persistUsage(self: *Session, snapshot: session_usage.Snapshot) !void {
        const now_ms = @max(io_mod.milliTimestamp(), 0);
        const at_ms = if (now_ms > self.usage_at_ms) now_ms else try std.math.add(i64, self.usage_at_ms, 1);
        const pending = session_usage.needsProfileRecovery(snapshot);
        if (pending and !self.usage_marked) {
            try self.writeUsageMarker(at_ms);
            self.usage_marked = true;
        }
        var arena = std.heap.ArenaAllocator.init(self.alloc);
        defer arena.deinit();
        const value = try encodeUsage(arena.allocator(), snapshot, at_ms);
        _ = try self.handle.append(&.{.{ .set = .{ .key = .usage, .value = value } }});
        self.usage_at_ms = at_ms;
        if (pending) return;
        // Before the first turn a `set` waits in memory, not on disk.
        if (!self.saved()) return;
        self.clearUsageMarker();
        self.usage_marked = false;
    }

    fn writeUsageMarker(self: *Session, now_ms: i64) !void {
        var dir = try self.openUsageMarkers();
        defer dir.close();
        var buffer: [48]u8 = undefined;
        const content = try std.fmt.bufPrint(&buffer, "v1 {d}\n", .{now_ms});
        // A full disk stops the turn here, before the model is asked, so it
        // must read as one (D29) and not as a failed replace.
        var cause: ?anyerror = null;
        io_mod.durableReplaceVerifiedWithOps(self.alloc, &dir, self.id(), content, .{ .pre_rename_cause = &cause }) catch |err| {
            if (cause) |stopped| if (storageCause(stopped)) |named| return named;
            return err;
        };
    }

    fn clearUsageMarker(self: *Session) void {
        var dir = self.openUsageMarkers() catch |err| {
            debug_trace.logf("session", "event=sessions_v2_usage_marker_kept session={s} err={s}", .{ self.id(), @errorName(err) });
            return;
        };
        defer dir.close();
        dir.dir.deleteFile(io_mod.getIo(), self.id()) catch |err| switch (err) {
            error.FileNotFound => {},
            else => debug_trace.logf("session", "event=sessions_v2_usage_marker_kept session={s} err={s}", .{ self.id(), @errorName(err) }),
        };
    }

    fn openUsageMarkers(self: *Session) !io_mod.VerifiedDir {
        var home = try openHome(self.store.home);
        defer home.close();
        var fx = try io_mod.openOrCreateVerifiedPrivateDir(&home, profile_paths.root_dir_name);
        defer fx.close();
        return io_mod.openOrCreateVerifiedPrivateDir(&fx, usage_markers_dir_name);
    }

    // -- resume --------------------------------------------------------------

    /// Rebuilds fx's history and settings from the open log (`restoreFrom`),
    /// and keeps what later appends compare against: the last turn, the turn
    /// behind each history entry, and the stored usage and language.
    pub fn restore(self: *Session, alloc: Allocator) !Restored {
        self.mutex.lockUncancelable(io_mod.getIo());
        defer self.mutex.unlock(io_mod.getIo());
        var scratch = std.heap.ArenaAllocator.init(alloc);
        defer scratch.deinit();
        const sa = scratch.allocator();
        const state = try self.handle.state(sa);
        self.last_turn = state.last_turn;
        self.turn_numbers.clearRetainingCapacity();
        var restored = try restoreFrom(self.source(), alloc, sa, state, .{ .alloc = self.alloc, .list = &self.turn_numbers });
        errdefer restored.deinit(alloc);
        if (state.usage) |raw| {
            const checkpoint = try decodeUsage(sa, raw);
            self.usage_at_ms = checkpoint.at_ms;
            self.usage_marked = session_usage.needsProfileRecovery(checkpoint.snapshot);
        }
        if (state.language != null) {
            const owned = try self.alloc.dupe(u8, restored.language.view());
            if (self.language) |old| self.alloc.free(old);
            self.language = owned;
        }
        return restored;
    }

    /// Keeps each turn for resume, with the number of the turn that holds it.
    /// Hands `visitor.append` every turn in the log, oldest first, as v1's
    /// conversation reader does: a compaction hides no turn from the
    /// transcript. Each turn is freed after its call, and pages are freed as
    /// they are read, so memory stays bounded by one page and one turn.
    /// Leaves the state resume keeps untouched. It reads only through the
    /// manager's thread-safe handle and changes no adapter state, so it
    /// takes no lock: a visitor may call back into this session, as the app
    /// does to read a command replay's side file while it draws a turn.
    pub fn visitHistory(self: *Session, alloc: Allocator, visitor: anytype) !void {
        const Sink = struct {
            alloc: Allocator,
            visitor: @TypeOf(visitor),

            fn turn(sink: *@This(), value: types.HistoryTurn, _: ?u64) !void {
                defer types.freeHistoryTurn(sink.alloc, value);
                try sink.visitor.append(value);
            }
        };
        var sink: Sink = .{ .alloc = alloc, .visitor = visitor };
        try replay(self.source(), alloc, alloc, .start, null, &sink);
    }

    /// The session as v1's `DurableSessionState`, for hosts that restore
    /// through it, and its stored title. Caller owns both.
    pub fn durableState(self: *Session, alloc: Allocator, workspace: []const u8) !Resumed {
        var restored = try self.restore(alloc);
        defer restored.deinit(alloc);
        return resumedOf(alloc, &restored, self.id(), workspace);
    }

    pub const Info = struct {
        /// The stored title, decoded; owned by the caller.
        title: ?[]u8,
        /// The time of the newest line; 0 before the first turn.
        updated_ms: i64,
    };

    /// What a host shows about the open session.
    pub fn info(self: *Session, alloc: Allocator) !Info {
        var scratch = std.heap.ArenaAllocator.init(alloc);
        defer scratch.deinit();
        const sa = scratch.allocator();
        const st = try self.handle.state(sa);
        const title = if (st.title) |raw| try alloc.dupe(u8, try std.json.parseFromSliceLeaky([]const u8, sa, raw, .{})) else null;
        return .{ .title = title, .updated_ms = std.math.cast(i64, st.updated_ms) orelse 0 };
    }

    fn source(self: *Session) Source {
        return .{ .store = self.store, .id = self.id(), .handle = self.handle };
    }
};

/// Where replay reads a log: an open session's own handle, or the store by
/// id without a lock, as for a finished child's result.
const Source = struct {
    store: *Store,
    id: []const u8,
    handle: ?sm.Session = null,

    fn read(src: Source, sa: Allocator, from: sm.From, limit: usize) !sm.Page {
        return src.readTo(sa, from, .forward, limit);
    }

    fn readTo(src: Source, sa: Allocator, from: sm.From, direction: sm.Direction, limit: usize) !sm.Page {
        if (src.handle) |handle| return handle.read(sa, from, direction, limit);
        return src.store.manager.read(sa, src.id, from, direction, limit);
    }

    /// The cursor of `turn`'s `turn_started`, reading back from `before`.
    fn findTurnStart(src: Source, sa: Allocator, before: sm.Cursor, turn: u64) !sm.Cursor {
        var from: sm.From = .{ .at = before };
        while (true) {
            var page = try src.readTo(sa, from, .backward, replay_page_lines);
            defer page.deinit();
            for (page.entries) |entry| {
                const body = entry.body orelse continue;
                if (body == .turn_started and body.turn_started.turn == turn) return .{ .offset = entry.offset, .seq = entry.seq };
            }
            from = .{ .at = page.next orelse return error.InvalidConversationFrame };
        }
    }

    /// A piece's bytes, from its blob when the line holds a reference.
    fn pieceData(src: Source, pa: Allocator, piece: sm.Body.Piece) ![]const u8 {
        if (piece.blobs.len != 1 or !std.mem.startsWith(u8, piece.data, "{\"" ++ blob_ref_key ++ "\":")) return piece.data;
        return src.store.manager.getBlob(pa, src.id, piece.blobs[0]) catch |err| switch (err) {
            // A blob the log names that is gone or damaged damages the
            // session, as a bad line does (D39).
            error.NotFound, error.Corrupt => {
                debug_trace.logf("session", "event=sessions_v2_blob_unreadable session={s} err={s}", .{ src.id, @errorName(err) });
                return error.InvalidSessionFormat;
            },
            else => |e| return e,
        };
    }
};

fn copyFailed(id: []const u8, err: anyerror) bool {
    debug_trace.logf("session", "event=sessions_v2_files_copy_incomplete session={s} err={s}", .{ id, @errorName(err) });
    return false;
}

/// Side folders are a few levels deep (a kind, then files); deeper trees
/// are not fx's and are left out.
const max_copy_depth = 8;

fn copyTree(source: *io_mod.VerifiedDir, target: *io_mod.VerifiedDir, depth: usize) bool {
    const io = io_mod.getIo();
    var complete = true;
    var it = source.dir.iterate();
    while (it.next(io) catch |err| return copyFailed("tree", err)) |entry| switch (entry.kind) {
        .directory => {
            if (depth + 1 == max_copy_depth) {
                complete = copyFailed(entry.name, error.TooDeep);
                continue;
            }
            var from = (io_mod.openVerifiedPrivateDirIfPresent(source, entry.name) catch |err| {
                complete = copyFailed(entry.name, err);
                continue;
            }) orelse continue;
            defer from.close();
            var to = io_mod.openOrCreateVerifiedPrivateDir(target, entry.name) catch |err| {
                complete = copyFailed(entry.name, err);
                continue;
            };
            defer to.close();
            if (!copyTree(&from, &to, depth + 1)) complete = false;
        },
        .file => copyFile(source.dir, target.dir, entry.name) catch |err| {
            complete = copyFailed(entry.name, err);
        },
        else => complete = copyFailed(entry.name, error.NotAFile),
    };
    return complete;
}

fn copyFile(from: std.Io.Dir, to: std.Io.Dir, name: []const u8) !void {
    const io = io_mod.getIo();
    var source = try from.openFile(io, name, .{ .follow_symlinks = false, .resolve_beneath = true, .allow_directory = false });
    defer source.close(io);
    var target = try to.createFile(io, name, .{ .exclusive = true, .permissions = .fromMode(0o600), .resolve_beneath = true });
    defer target.close(io);
    var buffer: [16 * 1024]u8 = undefined;
    var offset: u64 = 0;
    while (true) {
        const n = try source.readPositional(io, &.{&buffer}, offset);
        if (n == 0) break;
        try target.writePositionalAll(io, buffer[0..n], offset);
        offset += n;
    }
}

/// Where `restoreFrom` puts the number of the turn behind each history entry.
const TurnNumbers = struct {
    alloc: Allocator,
    list: *std.ArrayList(?u64),
};

/// Rebuilds fx's history and settings from `state` and its log: the newest
/// compaction's summary, then every turn after it (or after the turn it
/// kept). A turn a crash or close ended has no `interruption` item and
/// comes back interrupted: `failed` after a crash, `cancelled` after a
/// close. Caller owns the result.
fn restoreFrom(src: Source, alloc: Allocator, sa: Allocator, state: sm.State, numbers: ?TurnNumbers) !Restored {
    const language = if (state.language) |raw| try decodeLanguage(sa, raw) else types.ConversationLanguage.default();
    var restored: Restored = .{
        .history = &.{},
        .language = language,
        .created_at_ms = std.math.cast(i64, state.created_ms) orelse 0,
        .updated_at_ms = std.math.cast(i64, state.updated_ms) orelse 0,
    };
    errdefer restored.deinit(alloc);
    if (state.title) |raw| restored.title = try alloc.dupe(u8, try std.json.parseFromSliceLeaky([]const u8, sa, raw, .{}));
    if (state.prefs) |raw| restored.preferences = try decodePreferences(alloc, raw);
    if (state.permissions) |raw| restored.permission_state = try session_codec.decodePermissionState(alloc, raw);
    if (state.usage) |raw| restored.usage = (try decodeUsage(alloc, raw)).snapshot;

    var history: std.ArrayList(types.HistoryTurn) = .empty;
    errdefer {
        for (history.items) |turn| types.freeHistoryTurn(alloc, turn);
        history.deinit(alloc);
    }
    var from: sm.From = .start;
    var skip_offset: ?u64 = null;
    if (state.compaction_offset) |offset| {
        const cursor: sm.Cursor = .{ .offset = offset, .seq = state.last_compaction_seq.? };
        var page = try src.read(sa, .{ .at = cursor }, 1);
        defer page.deinit();
        const data = page.entries[0].body.?.compacted.data;
        const compacted = try std.json.parseFromSliceLeaky(CompactedData, sa, data, .{});
        try history.ensureUnusedCapacity(alloc, 1);
        if (numbers) |n| try n.list.append(n.alloc, null);
        history.appendAssumeCapacity(.{ .compacted_summary = .{
            .summary = try alloc.dupe(u8, compacted.summary),
            .removed_turn_count = compacted.removed_turn_count,
            .compaction_count = compacted.compaction_count,
            .root_user_messages_complete = false,
            .permission_feedback_complete = false,
        } });
        skip_offset = offset;
        from = .{ .at = cursor };
        if (compacted.keep_from_turn) |turn| from = .{ .at = try src.findTurnStart(sa, cursor, turn) };
    }
    var sink: RestoreSink = .{ .alloc = alloc, .history = &history, .numbers = numbers };
    try replay(src, alloc, sa, from, skip_offset, &sink);
    restored.history = try history.toOwnedSlice(alloc);
    return restored;
}

/// Keeps each turn for resume, with the number of the turn that holds it.
const RestoreSink = struct {
    alloc: Allocator,
    history: *std.ArrayList(types.HistoryTurn),
    numbers: ?TurnNumbers,

    fn turn(sink: *RestoreSink, value: types.HistoryTurn, number: ?u64) !void {
        errdefer types.freeHistoryTurn(sink.alloc, value);
        try sink.history.ensureUnusedCapacity(sink.alloc, 1);
        if (sink.numbers) |n| try n.list.append(n.alloc, number);
        sink.history.appendAssumeCapacity(value);
    }
};

/// Moves `restored` into v1's `DurableSessionState` with its stored title;
/// `restored` keeps only what it still owns. Caller owns the result.
fn resumedOf(alloc: Allocator, restored: *Restored, id: []const u8, workspace: []const u8) !Resumed {
    const id_copy = try alloc.dupe(u8, id);
    errdefer alloc.free(id_copy);
    const origin = try alloc.dupe(u8, workspace);
    errdefer alloc.free(origin);
    const workspace_copy = try alloc.dupe(u8, workspace);
    errdefer alloc.free(workspace_copy);
    // Every session starts with its preferences (`create`).
    const preferences = restored.preferences orelse return error.InvalidSessionFormat;
    restored.preferences = null;
    const resumed: Resumed = .{
        .state = .{
            .id = id_copy,
            .origin_workspace_root = origin,
            .workspace_root = workspace_copy,
            .created_at_ms = restored.created_at_ms,
            .updated_at_ms = restored.updated_at_ms,
            .conversation_language = restored.language,
            .preferences = preferences,
            .history = restored.history,
            // v1 reloads its totals as zero as well.
            .total_input_tokens = 0,
            .total_output_tokens = 0,
            .permission_state = restored.permission_state orelse .{},
            .usage = restored.usage,
        },
        .title = restored.title,
    };
    restored.history = &.{};
    restored.permission_state = null;
    restored.usage = null;
    restored.title = null;
    return resumed;
}

fn lastStarted(entry: sm.Entry) ?u64 {
    return switch (entry.body.?) {
        .item => |piece| piece.turn,
        else => null,
    };
}

fn replay(
    src: Source,
    alloc: Allocator,
    sa: Allocator,
    start: sm.From,
    skip_offset: ?u64,
    /// Takes each finished turn, even when it fails: `turn(value, number)`.
    sink: anytype,
) !void {
    var builder = session_log.ConversationTurnBuilder.init(alloc);
    defer builder.deinit();
    var interrupted_item = false;
    var superseded = false;
    var piece_arena = std.heap.ArenaAllocator.init(alloc);
    defer piece_arena.deinit();
    // The open turn's calls saved as running, and the call ids its pieces
    // already answer or hold (D28).
    var turn_arena = std.heap.ArenaAllocator.init(alloc);
    defer turn_arena.deinit();
    var running: std.ArrayList(session_event.ConversationToolCall) = .empty;
    var represented: std.ArrayList([]const u8) = .empty;
    var from = start;
    while (true) {
        var page = try src.read(sa, from, replay_page_lines);
        defer page.deinit();
        if (page.damaged) debug_trace.logf("session", "event=sessions_v2_replay_damaged session={s} dropped=lines_after_damage", .{src.id});
        for (page.entries) |entry| {
            _ = piece_arena.reset(.retain_capacity);
            const pa = piece_arena.allocator();
            if (skip_offset) |offset| if (entry.offset == offset) continue;
            const body = entry.body orelse continue;
            switch (body) {
                .turn_started => {
                    interrupted_item = false;
                    superseded = false;
                    _ = turn_arena.reset(.retain_capacity);
                    running = .empty;
                    represented = .empty;
                },
                .item => |piece| {
                    if (std.mem.eql(u8, piece.type, superseded_type)) {
                        superseded = true;
                        continue;
                    }
                    const ta = turn_arena.allocator();
                    if (std.mem.eql(u8, piece.type, running_type)) {
                        // Kept for the whole turn: copies, not views of the page.
                        const call = try decodePiece(ta, .tool_call, try src.pieceData(ta, piece), .alloc_always);
                        try running.append(ta, call.tool_call);
                        continue;
                    }
                    const kind = pieceKind(piece.type) orelse {
                        debug_trace.logf("session", "event=sessions_v2_unknown_item session={s} type={s} dropped=item", .{ src.id, piece.type });
                        continue;
                    };
                    const data = try src.pieceData(pa, piece);
                    // The builder copies what it keeps, as it does for v1's
                    // reader, so strings may point into the page.
                    switch (try decodePiece(pa, kind, data, .alloc_if_needed)) {
                        .user => |value| try builder.begin(value),
                        .assistant => |value| try builder.appendAssistant(value),
                        .tool_call => |value| {
                            try builder.appendToolCall(value);
                            try represented.append(ta, try ta.dupe(u8, value.call_id));
                        },
                        .tool_result => |value| {
                            try builder.appendToolResult(value);
                            try represented.append(ta, try ta.dupe(u8, value.call_id));
                        },
                        .steering => |value| try builder.appendSteering(value.text),
                        .turn_completed => |value| try sink.turn(try builder.finishAssistant(value), lastStarted(entry)),
                        .interrupted => |value| {
                            interrupted_item = true;
                            try sink.turn(try builder.finishInterrupted(value), lastStarted(entry));
                        },
                        .context_checkpoint => return error.InvalidConversationFrame,
                    }
                },
                .turn_interrupted => |ended| {
                    if (superseded) {
                        debug_trace.logf("session", "event=sessions_v2_replay_superseded session={s} turn={d} dropped=stale_turn", .{ src.id, ended.turn });
                        builder.deinit();
                        builder = session_log.ConversationTurnBuilder.init(alloc);
                        superseded = false;
                        continue;
                    }
                    if (interrupted_item or builder.isIdle()) continue;
                    var turn = try builder.finishInterrupted(.{ .reason = switch (ended.reason) {
                        .cancel, .closed => .cancelled,
                        .failed, .crash => .failed,
                    } });
                    const answered = answerRunning(alloc, &turn.interrupted, running.items, represented.items) catch |err| {
                        types.freeHistoryTurn(alloc, turn);
                        return err;
                    };
                    if (answered > 0) debug_trace.logf("session", "event=sessions_v2_replay_unfinished_tools session={s} turn={d} calls={d}", .{ src.id, ended.turn, answered });
                    try sink.turn(turn, ended.turn);
                },
                .session_created, .compacted, .turn_committed, .set, .child_spawned, .child_finished, .snapshot, .closed => {},
            }
        }
        from = .{ .at = page.next orelse break };
    }
    if (!builder.isIdle()) debug_trace.logf("session", "event=sessions_v2_replay_open_turn session={s} dropped=unfinished_pieces", .{src.id});
}

/// Whether a streamed piece is the final one. fx stamps a tool result's
/// `created_at_ms` each time it rebuilds a turn, so that field alone may
/// differ; the streamed, earlier stamp stands.
fn samePiece(alloc: Allocator, streamed: []const u8, final: []const u8) bool {
    if (std.mem.eql(u8, streamed, final)) return true;
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const a = arena.allocator();
    const x = withoutCreatedAt(a, streamed) catch return false;
    const y = withoutCreatedAt(a, final) catch return false;
    return std.mem.eql(u8, x, y);
}

fn withoutCreatedAt(a: Allocator, bytes: []const u8) ![]u8 {
    var value = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
    if (value != .object) return error.NotAnObject;
    if (!value.object.swapRemove("created_at_ms")) return error.NoCreatedAt;
    return jsonValue(a, value);
}

/// Gives each call that was saved as running but that `represented` does not
/// name a failed result saying it may have partly run (D28), as one more
/// step of `turn`, so every call keeps its result. Returns how many.
fn answerRunning(
    alloc: Allocator,
    turn: *types.InterruptedHistoryTurn,
    running: []const session_event.ConversationToolCall,
    represented: []const []const u8,
) !usize {
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const a = arena.allocator();
    var calls: std.ArrayList(types.ToolCall) = .empty;
    var results: std.ArrayList(types.PersistedToolResult) = .empty;
    for (running) |call| {
        if (containsId(represented, call.call_id)) continue;
        try calls.append(a, historyToolCall(call));
        try results.append(a, .{
            .tool_call_id = @constCast(call.call_id),
            .tool_name = @constCast(call.tool_name),
            .status = .failure,
            .output = @constCast(unfinished_tool_output),
            .output_bytes = unfinished_tool_output.len,
            .stored_output_bytes = unfinished_tool_output.len,
        });
    }
    if (calls.items.len == 0) return 0;
    const owned_calls = try types.dupeToolCallSlice(alloc, calls.items);
    errdefer types.freeToolCallSlice(alloc, owned_calls);
    const owned_results = try types.dupePersistedToolResults(alloc, results.items);
    errdefer types.freePersistedToolResults(alloc, owned_results);
    const old = turn.execution.tool_steps;
    const steps = try alloc.alloc(types.ToolExecutionStep, old.len + 1);
    @memcpy(steps[0..old.len], old);
    steps[old.len] = .{ .tool_calls = owned_calls, .tool_results = owned_results };
    if (old.len > 0) alloc.free(old);
    turn.execution.tool_steps = steps;
    return calls.items.len;
}

fn containsId(ids: []const []const u8, id: []const u8) bool {
    for (ids) |candidate| if (std.mem.eql(u8, candidate, id)) return true;
    return false;
}

fn conversationToolCall(call: types.ToolCall) session_event.ConversationToolCall {
    return .{
        .call_id = call.id,
        .tool_name = call.name,
        .arguments_json = call.arguments_json,
        .argument_integrity = call.argument_integrity,
        .provisional_id = call.provisional_id,
        .provider_result = call.provider_result,
        .final_identity = call.final_identity,
        .provenance = call.provenance,
    };
}

fn historyToolCall(call: session_event.ConversationToolCall) types.ToolCall {
    return .{
        .id = call.call_id,
        .name = call.tool_name,
        .arguments_json = call.arguments_json,
        .argument_integrity = call.argument_integrity,
        .provisional_id = call.provisional_id,
        .provider_result = call.provider_result,
        .final_identity = call.final_identity,
        .provenance = call.provenance,
    };
}

// ---------------------------------------------------------------------------
// Listing

const list_page_size: usize = 256;

/// Every saved root session, newest first, as v1's picker summaries, leaving
/// out `active_id`. A saved session always has a turn (D2), so each one can
/// be resumed. Stops with `error.Cancelled` once `cancel` is set. Caller
/// owns the list and every summary; safe from any thread.
pub fn listSummaries(
    store: *Store,
    alloc: Allocator,
    active_id: ?[]const u8,
    cancel: *const std.atomic.Value(bool),
) !std.ArrayList(session_store.SessionSummary) {
    var list: std.ArrayList(session_store.SessionSummary) = .empty;
    errdefer {
        for (list.items) |*summary| summary.deinit(alloc);
        list.deinit(alloc);
    }
    var scratch = std.heap.ArenaAllocator.init(alloc);
    defer scratch.deinit();
    var cursor: ?sm.ListCursor = null;
    while (true) {
        if (cancel.load(.acquire)) return error.Cancelled;
        var page = try store.manager.list(alloc, .all, cursor, list_page_size);
        defer page.deinit();
        for (page.items) |item| {
            if (item.role != .root) continue;
            if (active_id) |active| if (std.mem.eql(u8, active, item.id)) continue;
            try list.append(alloc, try summaryOf(alloc, scratch.allocator(), item));
        }
        cursor = page.next orelse break;
    }
    return list;
}

fn summaryOf(alloc: Allocator, scratch: Allocator, item: sm.Summary) !session_store.SessionSummary {
    const id = try alloc.dupe(u8, item.id);
    errdefer alloc.free(id);
    const workspace = try alloc.dupe(u8, item.workspace);
    errdefer alloc.free(workspace);
    const origin = try alloc.dupe(u8, item.workspace);
    errdefer alloc.free(origin);
    const title: ?[]u8 = if (item.title) |raw|
        try alloc.dupe(u8, try std.json.parseFromSliceLeaky([]const u8, scratch, raw, .{}))
    else
        null;
    errdefer if (title) |value| alloc.free(value);
    return .{
        .id = id,
        .workspace_root = workspace,
        .origin_workspace_root = origin,
        .title = title,
        .display_metadata_present = title != null,
        .created_at_ms = std.math.cast(i64, item.created_ms) orelse 0,
        .updated_at_ms = std.math.cast(i64, item.updated_ms) orelse 0,
        .conversation_language = if (item.language) |raw| try decodeLanguage(scratch, raw) else types.ConversationLanguage.default(),
        .history_len = @max(item.turns, 1),
    };
}

// ---------------------------------------------------------------------------
// Usage recovery: what the profile's readers need from v2 sessions

/// A v2 session's usage-recovery marker and its newest usage checkpoint,
/// read without the session's lock. Owns everything.
pub const MarkedUsage = struct {
    id: []u8,
    /// Null when the session or its checkpoint cannot be read.
    snapshot: ?session_usage.Snapshot,
    /// When the checkpoint was written.
    at_ms: i64 = 0,
    protected_updated_at_ms: ?i64,
    marker_modified_at_ns: i128,

    pub fn deinit(marked: *MarkedUsage, alloc: Allocator) void {
        alloc.free(marked.id);
        if (marked.snapshot) |*snapshot| snapshot.deinit(alloc);
        marked.* = undefined;
    }
};

const max_usage_markers: usize = 512;

/// Every v2 usage-recovery marker under `home`, with v1's validation.
pub fn collectMarkedUsage(alloc: Allocator, home: []const u8) !std.ArrayList(MarkedUsage) {
    var list: std.ArrayList(MarkedUsage) = .empty;
    errdefer {
        for (list.items) |*entry| entry.deinit(alloc);
        list.deinit(alloc);
    }
    const path = try std.fs.path.join(alloc, &.{ home, profile_paths.root_dir_name, usage_markers_dir_name });
    defer alloc.free(path);
    var markers = io_mod.VerifiedDir{ .dir = std.Io.Dir.openDirAbsolute(io_mod.getIo(), path, .{ .iterate = true, .follow_symlinks = false }) catch |err| switch (err) {
        error.FileNotFound => return list,
        else => return err,
    } };
    defer markers.close();
    var store = try Store.open(alloc, home);
    defer store.deinit(alloc);
    var it = markers.dir.iterate();
    while (try it.next(io_mod.getIo())) |entry| {
        if (entry.kind != .file or list.items.len == max_usage_markers) return error.InvalidUsageRecoveryIndex;
        const protected = session_store.validateUsageRecoveryMarker(&markers, entry.name) catch return error.InvalidUsageRecoveryIndex;
        const stat = try markers.dir.statFile(io_mod.getIo(), entry.name, .{ .follow_symlinks = false });
        const id = try alloc.dupe(u8, entry.name);
        errdefer alloc.free(id);
        const checkpoint = newestUsage(&store, alloc, id) catch |err| blk: {
            if (err == error.OutOfMemory) return err;
            debug_trace.logf("usage", "event=sessions_v2_usage_unreadable session={s} err={s}", .{ id, @errorName(err) });
            break :blk null;
        };
        try list.append(alloc, .{
            .id = id,
            .snapshot = if (checkpoint) |c| c.snapshot else null,
            .at_ms = if (checkpoint) |c| c.at_ms else 0,
            .protected_updated_at_ms = protected,
            .marker_modified_at_ns = stat.mtime.nanoseconds,
        });
    }
    return list;
}

/// The newest `set usage`, or the usage in the newest snapshot, reading
/// back from the end without the session's lock.
fn newestUsage(store: *Store, alloc: Allocator, id: []const u8) !?UsageCheckpoint {
    var scratch = std.heap.ArenaAllocator.init(alloc);
    defer scratch.deinit();
    const sa = scratch.allocator();
    var from: sm.From = .end;
    while (true) {
        var page = try store.manager.read(sa, id, from, .backward, replay_page_lines);
        defer page.deinit();
        for (page.entries) |entry| {
            const body = entry.body orelse continue;
            switch (body) {
                .set => |setting| if (setting.key == .usage) return try decodeUsage(alloc, setting.value),
                .snapshot => |snapshot| {
                    const state = try std.json.parseFromSliceLeaky(std.json.Value, sa, snapshot.state, .{});
                    if (state != .object) return error.InvalidUsageCheckpoint;
                    const usage = state.object.get("usage") orelse return null;
                    return try decodeUsageValue(alloc, usage);
                },
                else => {},
            }
        }
        from = .{ .at = page.next orelse return null };
    }
}

/// v1's error names for a failed resume, so every host reports the same
/// error whichever backend is on.
/// Whether a failed write may still have reached the log: after any I/O
/// fault, whatever its cause, durability is unknown (D29).
pub fn writeMayHaveLanded(err: anyerror) bool {
    inline for (@typeInfo(sm.IoFault).error_set.?) |fault| {
        if (err == @field(sm.IoFault, fault.name)) return true;
    }
    return false;
}

const ResumeError = error{ SessionNotFound, NoSavedSessions, NoRememberedSession, SessionBusy, InvalidSessionFormat, UnsupportedSessionFormat } || sm.OpenError;

/// The storage causes a host names (D29), from an OS error; null for any
/// other error.
fn storageCause(err: anyerror) ?error{ NoSpaceLeft, AccessDenied, ReadOnlyFileSystem, FileTooBig } {
    return switch (err) {
        error.NoSpaceLeft => error.NoSpaceLeft,
        error.AccessDenied, error.PermissionDenied => error.AccessDenied,
        error.ReadOnlyFileSystem => error.ReadOnlyFileSystem,
        error.FileTooBig => error.FileTooBig,
        else => null,
    };
}

fn resumeError(err: sm.OpenError, target: Target) ResumeError {
    return switch (err) {
        error.NotFound => switch (target) {
            .id => error.SessionNotFound,
            .last => error.NoSavedSessions,
            .last_opened => error.NoRememberedSession,
        },
        // A child is resumed only through its parent, as in v1.
        error.ChildSession => error.SessionNotFound,
        error.Busy => error.SessionBusy,
        error.Corrupt => error.InvalidSessionFormat,
        error.UnsupportedVersion => error.UnsupportedSessionFormat,
        else => err,
    };
}

/// Item type of a turn closed because its streamed pieces did not match
/// the final turn; resume drops that turn.
const superseded_type = "superseded";
/// A tool call saved before it runs (D28); its data is a `tool_call` piece.
const running_type = "tool_running";
/// What the model reads for a call that was running when its turn ended.
const unfinished_tool_output = "fx stopped while this tool was running, so it may have partly run. Check its effects before running it again.";
const blob_ref_key = "$blob";

const CompactedData = struct {
    summary: []const u8,
    removed_turn_count: usize,
    compaction_count: usize,
    /// The first turn the compaction kept, read back to on resume.
    keep_from_turn: ?u64 = null,
};

// ---------------------------------------------------------------------------
// Encodings (pure)

/// The item `type` of each v1 piece kind.
fn itemType(kind: PieceKind) ?[]const u8 {
    return switch (kind) {
        .user => "user",
        .assistant => "assistant",
        .tool_call => "tool_call",
        .tool_result => "tool_result",
        .steering => "steering",
        .turn_completed => "turn_end",
        .interrupted => "interruption",
        .context_checkpoint => null,
    };
}

fn pieceKind(item_type: []const u8) ?PieceKind {
    inline for (@typeInfo(PieceKind).@"enum".fields) |field| {
        const kind: PieceKind = @enumFromInt(field.value);
        if (itemType(kind)) |name| {
            if (std.mem.eql(u8, name, item_type)) return kind;
        }
    }
    return null;
}

/// A piece's payload as v1 writes it inside its frame.
fn encodePiece(alloc: Allocator, event: Event) ![]u8 {
    try session_event.validateConversationEventShape(event, session_event.conversation_schema_version);
    var out: std.Io.Writer.Allocating = .init(alloc);
    errdefer out.deinit();
    switch (event) {
        inline else => |payload| try std.json.Stringify.value(payload, .{}, &out.writer),
    }
    return out.toOwnedSlice();
}

/// Payload slices point into `arena` or `data`.
/// With `.alloc_if_needed`, strings may point into `data`.
fn decodePiece(arena: Allocator, kind: PieceKind, data: []const u8, allocate: std.json.AllocWhen) !Event {
    const event: Event = switch (kind) {
        inline else => |tag| @unionInit(Event, @tagName(tag), try std.json.parseFromSliceLeaky(
            @FieldType(Event, @tagName(tag)),
            arena,
            data,
            .{ .allocate = allocate },
        )),
    };
    try session_event.validateConversationEventShape(event, session_event.conversation_schema_version);
    return event;
}

fn jsonValue(alloc: Allocator, value: anytype) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(alloc);
    errdefer out.deinit();
    try std.json.Stringify.value(value, .{}, &out.writer);
    return out.toOwnedSlice();
}

fn jsonString(alloc: Allocator, value: []const u8) ![]u8 {
    return jsonValue(alloc, value);
}

/// v1's shape, which `session_codec.parse_preferences` reads back: effort
/// as its label, the provider in its saved form.
/// A child's `prefs` also hold its instructions under this key (D34).
const instructions_key = "instructions";

fn encodePreferences(alloc: Allocator, preferences: session_codec.DurableSessionPreferences, instructions: ?[]const u8) ![]u8 {
    const Saved = struct {
        provider: model_provider.ProviderId,
        model: []const u8,
        effort: []const u8,
        fast_mode: bool,
    };
    const saved: Saved = .{
        .provider = preferences.provider,
        .model = preferences.model,
        .effort = preferences.effort.label(),
        .fast_mode = preferences.fast_mode,
    };
    const text = instructions orelse return jsonValue(alloc, saved);
    const ChildSaved = struct {
        provider: model_provider.ProviderId,
        model: []const u8,
        effort: []const u8,
        fast_mode: bool,
        instructions: []const u8,
    };
    return jsonValue(alloc, ChildSaved{
        .provider = saved.provider,
        .model = saved.model,
        .effort = saved.effort,
        .fast_mode = saved.fast_mode,
        .instructions = text,
    });
}

fn decodePreferences(alloc: Allocator, raw: []const u8) !session_codec.DurableSessionPreferences {
    var parsed = try std.json.parseFromSlice(std.json.Value, alloc, raw, .{});
    defer parsed.deinit();
    // v1's codec knows every field but a child's instructions.
    if (parsed.value == .object) _ = parsed.value.object.swapRemove(instructions_key);
    return session_codec.parse_preferences(alloc, parsed.value);
}

/// A child's instructions from its `prefs`, or null. Caller owns.
fn decodeInstructions(alloc: Allocator, raw: []const u8) !?[]u8 {
    var parsed = try std.json.parseFromSlice(std.json.Value, alloc, raw, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidSessionFormat;
    const value = parsed.value.object.get(instructions_key) orelse return null;
    if (value != .string) return error.InvalidSessionFormat;
    return try alloc.dupe(u8, value.string);
}

fn decodeLanguage(arena: Allocator, raw: []const u8) !types.ConversationLanguage {
    const tag = try std.json.parseFromSliceLeaky([]const u8, arena, raw, .{});
    return session_codec.parseConversationLanguage(tag);
}

/// `{"at_ms":N,"snapshot":...}`: the snapshot as v1's usage file holds it.
fn encodeUsage(alloc: Allocator, snapshot: session_usage.Snapshot, at_ms: i64) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(alloc);
    errdefer out.deinit();
    try out.writer.print("{{\"at_ms\":{d},\"snapshot\":", .{at_ms});
    try session_usage.writeRichSnapshot(&out.writer, snapshot);
    try out.writer.writeByte('}');
    return out.toOwnedSlice();
}

const UsageCheckpoint = struct { snapshot: session_usage.Snapshot, at_ms: i64 };

fn decodeUsage(alloc: Allocator, raw: []const u8) !UsageCheckpoint {
    var parsed = try std.json.parseFromSlice(std.json.Value, alloc, raw, .{});
    defer parsed.deinit();
    return decodeUsageValue(alloc, parsed.value);
}

fn decodeUsageValue(alloc: Allocator, value: std.json.Value) !UsageCheckpoint {
    if (value != .object) return error.InvalidUsageCheckpoint;
    const at = value.object.get("at_ms") orelse return error.InvalidUsageCheckpoint;
    if (at != .integer) return error.InvalidUsageCheckpoint;
    const snapshot = value.object.get("snapshot") orelse return error.InvalidUsageCheckpoint;
    return .{ .snapshot = try session_usage.parseSnapshotValue(alloc, snapshot), .at_ms = at.integer };
}

/// The title v1 derives from a fresh session's first turn, if any.
fn deriveTitle(arena: Allocator, turn: types.HistoryTurn) !?[]const u8 {
    const display = session_display_metadata.deriveFromHistory(arena, &.{turn}) catch return null;
    if (!display.present or std.mem.eql(u8, display.title, session_display_metadata.fallback_title)) return null;
    return display.title;
}

// ---------------------------------------------------------------------------
// Tests

const testing = std.testing;

test "every piece kind but checkpoints maps to one item type and back" {
    inline for (@typeInfo(PieceKind).@"enum".fields) |field| {
        const kind: PieceKind = @enumFromInt(field.value);
        if (itemType(kind)) |name| {
            try testing.expectEqual(@as(?PieceKind, kind), pieceKind(name));
        } else {
            try testing.expectEqual(PieceKind.context_checkpoint, kind);
        }
    }
    try testing.expectEqual(@as(?PieceKind, null), pieceKind("compacted"));
}

test "preferences round trip through v1's decoder" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var model = "vendor/model-1".*;
    const efforts = [_]types.ReasoningEffort{ .auto, types.ReasoningEffort.parse("high").? };
    for (efforts) |effort| {
        const raw = try encodePreferences(arena.allocator(), .{ .model = &model, .effort = effort, .fast_mode = true }, null);
        try testing.expect((try decodeInstructions(testing.allocator, raw)) == null);
        var decoded = try decodePreferences(testing.allocator, raw);
        defer decoded.deinit(testing.allocator);
        try testing.expectEqualStrings("vendor/model-1", decoded.model);
        try testing.expectEqualStrings(effort.label(), decoded.effort.label());
        try testing.expect(decoded.fast_mode);
        try testing.expectEqual(model_provider.ProviderId.gateway, decoded.provider);
    }
}

test "the switch is the flag or FX_SESSIONS_V2" {
    try testing.expect(enabled(true));
}

/// A HOME in a temp folder with an adapter store over it.
const TestHome = struct {
    tmp: testing.TmpDir,
    home: []u8,
    store: Store,

    fn init(t: *TestHome) !void {
        t.tmp = testing.tmpDir(.{});
        errdefer t.tmp.cleanup();
        t.home = try io_mod.dirRealpathAlloc(testing.allocator, t.tmp.dir, ".");
        errdefer testing.allocator.free(t.home);
        t.store = try Store.open(testing.allocator, t.home);
    }

    fn deinit(t: *TestHome) void {
        t.store.deinit(testing.allocator);
        testing.allocator.free(t.home);
        t.tmp.cleanup();
    }
};

fn testSeed(model: []u8) Seed {
    return .{
        .preferences = .{ .model = model, .effort = .auto, .fast_mode = false },
        .language = types.ConversationLanguage.default(),
        .permission_state = .{},
    };
}

fn assistantTurn(user: []const u8, reply: []const u8) types.HistoryTurn {
    return .{ .assistant = .{
        .user = .{ .text = @constCast(user) },
        .assistant = @constCast(reply),
    } };
}

test "a new session commits turns, and resume gives them back with its settings" {
    var t: TestHome = undefined;
    try t.init();
    defer t.deinit();
    var model = "test-model".*;
    const s = try Session.create(testing.allocator, &t.store, "/w", .ask, testSeed(&model));
    try s.commitTurn(assistantTurn("first question", "first answer"), types.ConversationLanguage.default());
    try s.commitTurn(.{ .interrupted = .{
        .user = .{ .text = @constCast("second question") },
        .assistant = @constCast("partial"),
        .terminal_reason = .failed,
    } }, types.ConversationLanguage.default());
    const id = try testing.allocator.dupe(u8, s.id());
    defer testing.allocator.free(id);
    s.close();

    const r = try Session.resumeSession(testing.allocator, &t.store, .{ .id = id }, "/w", .ask);
    defer r.close();
    var restored = try r.restore(testing.allocator);
    defer restored.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 2), restored.history.len);
    try testing.expectEqualStrings("first question", restored.history[0].assistant.user.text);
    try testing.expectEqualStrings("first answer", restored.history[0].assistant.assistant);
    try testing.expectEqualStrings("partial", restored.history[1].interrupted.assistant.?);
    try testing.expectEqual(types.InterruptedTerminalReason.failed, restored.history[1].interrupted.terminal_reason);
    try testing.expectEqualStrings("test-model", restored.preferences.?.model);
    try testing.expect(restored.created_at_ms > 0);
    // The first turn named the session.
    var st = try r.handle.state(testing.allocator);
    defer st.deinit(testing.allocator);
    try testing.expectEqualStrings("\"first question\"", st.title.?);
}

test "a child's prefs keep its instructions, and v1's decoder still reads the rest (D34)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var model = "vendor/model-1".*;
    const raw = try encodePreferences(arena.allocator(), .{ .model = &model, .effort = .auto, .fast_mode = false }, "Answer in one line.");
    const instructions = (try decodeInstructions(testing.allocator, raw)).?;
    defer testing.allocator.free(instructions);
    try testing.expectEqualStrings("Answer in one line.", instructions);
    var decoded = try decodePreferences(testing.allocator, raw);
    defer decoded.deinit(testing.allocator);
    try testing.expectEqualStrings("vendor/model-1", decoded.model);
}

fn childSeed(model: []u8, instructions: []const u8) Seed {
    var seed = testSeed(model);
    seed.instructions = instructions;
    return seed;
}

test "a child opens under the id its parent names, and its parent folds its lines (D22, D34)" {
    var t: TestHome = undefined;
    try t.init();
    defer t.deinit();
    const alloc = testing.allocator;
    var model = "test-model".*;
    const parent = try Session.create(alloc, &t.store, "/w", .ask, testSeed(&model));
    defer parent.close();
    try parent.commitTurn(assistantTurn("delegate this", "delegated"), types.ConversationLanguage.default());
    const child_id = "1786460757753-kid";
    try parent.appendChildLines(&.{.{ .spawned = .{ .child = child_id, .work_id = "w1", .data = "{\"kind\":\"persistent\"}" } }});

    {
        const child = try Session.openChild(alloc, &t.store, parent.id(), child_id, "/w", childSeed(&model, "Be brief."));
        defer child.close();
        try testing.expectEqualStrings(child_id, child.id());
        try testing.expectEqualStrings("Be brief.", child.childInstructions());
        try child.commitTurn(assistantTurn("task one", "done one"), types.ConversationLanguage.default());
    }
    try parent.appendChildLines(&.{.{ .finished = .{ .child = child_id, .work_id = "w1", .outcome = .ok, .data = "{}" } }});

    const children = try parent.children(alloc);
    defer freeChildren(alloc, children);
    try testing.expectEqual(@as(usize, 1), children.len);
    try testing.expectEqualStrings(child_id, children[0].id);
    try testing.expectEqualStrings("w1", children[0].work_id);
    try testing.expect(!children[0].open);
    try testing.expectEqual(ChildOutcome.ok, children[0].outcome.?);
    try testing.expectEqualStrings("{\"kind\":\"persistent\"}", children[0].spawn_data.?);
    try testing.expectEqualStrings("{}", children[0].finish_data.?);

    // Read by id, without the child's lock.
    const history = try t.store.childHistory(alloc, child_id);
    defer types.freeHistoryTurnSlice(alloc, history);
    try testing.expectEqual(@as(usize, 1), history.len);
    try testing.expectEqualStrings("done one", history[0].assistant.assistant);
    var preferences = try t.store.childPreferences(alloc, child_id);
    defer preferences.deinit(alloc);
    try testing.expectEqualStrings("test-model", preferences.model);

    // New instructions replace the stored ones; none keeps them.
    {
        const child = try Session.openChild(alloc, &t.store, parent.id(), child_id, "/w", childSeed(&model, "Be thorough."));
        defer child.close();
        try testing.expectEqualStrings("Be thorough.", child.childInstructions());
    }
    {
        const child = try Session.openChild(alloc, &t.store, parent.id(), child_id, "/w", testSeed(&model));
        defer child.close();
        try testing.expectEqualStrings("Be thorough.", child.childInstructions());
        var restored = try child.restore(alloc);
        defer restored.deinit(alloc);
        try testing.expectEqualStrings("test-model", restored.preferences.?.model);
        try testing.expectEqual(@as(usize, 1), restored.history.len);
    }
    // Children stay out of the session list.
    var page = try t.store.manager.list(alloc, .all, null, 10);
    defer page.deinit();
    try testing.expectEqual(@as(usize, 1), page.items.len);
}

test "the manager guards child lines, and a child without a log reopens under its id (D22, D34)" {
    var t: TestHome = undefined;
    try t.init();
    defer t.deinit();
    const alloc = testing.allocator;
    var model = "test-model".*;
    const parent = try Session.create(alloc, &t.store, "/w", .ask, testSeed(&model));
    defer parent.close();
    try parent.commitTurn(assistantTurn("delegate", "ok"), types.ConversationLanguage.default());
    const child_id = "1786460757753-lost";
    try parent.appendChildLines(&.{.{ .spawned = .{ .child = child_id, .work_id = "w1" } }});
    try testing.expectError(error.InvalidTransition, parent.appendChildLines(&.{.{ .spawned = .{ .child = child_id, .work_id = "w2" } }}));
    try testing.expectError(error.InvalidTransition, parent.appendChildLines(&.{.{ .finished = .{ .child = child_id, .work_id = "w9", .outcome = .ok } }}));

    // Closed before its first turn, as a crash would leave it: nothing on disk.
    (try Session.openChild(alloc, &t.store, parent.id(), child_id, "/w", childSeed(&model, "Stay short."))).close();
    try testing.expectError(error.NotFound, t.store.childHistory(alloc, child_id));
    const child = try Session.openChild(alloc, &t.store, parent.id(), child_id, "/w", testSeed(&model));
    defer child.close();
    try testing.expectEqualStrings(child_id, child.id());
    try testing.expectEqualStrings("", child.childInstructions());
}

fn countItems(manager: *sm.Manager, id: []const u8, item_type: []const u8) !usize {
    var page = try manager.read(testing.allocator, id, .start, .forward, 1000);
    defer page.deinit();
    var n: usize = 0;
    for (page.entries) |entry| {
        const body = entry.body orelse continue;
        if (body == .item and std.mem.eql(u8, body.item.type, item_type)) n += 1;
    }
    return n;
}

test "streamed pieces are written once, and the commit adds only the rest" {
    var t: TestHome = undefined;
    try t.init();
    defer t.deinit();
    var model = "m".*;
    const s = try Session.create(testing.allocator, &t.store, "/w", .ask, testSeed(&model));
    const user: types.UserTurn = .{ .text = @constCast("streamed question") };
    try s.appendProgress(user, .{}, &.{});
    // The first piece published the session.
    try testing.expect(s.saved());
    try s.appendProgress(user, .{}, &.{});
    try s.commitTurn(.{ .assistant = .{ .user = user, .assistant = @constCast("streamed answer") } }, types.ConversationLanguage.default());
    const id = try testing.allocator.dupe(u8, s.id());
    defer testing.allocator.free(id);
    s.close();
    try testing.expectEqual(@as(usize, 1), try countItems(t.store.manager, id, "user"));
    try testing.expectEqual(@as(usize, 1), try countItems(t.store.manager, id, "turn_end"));

    const r = try Session.resumeSession(testing.allocator, &t.store, .{ .id = id }, "/w", .ask);
    defer r.close();
    var restored = try r.restore(testing.allocator);
    defer restored.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 1), restored.history.len);
    try testing.expectEqualStrings("streamed answer", restored.history[0].assistant.assistant);
}

test "any I/O fault may have left a write in the log" {
    inline for (.{ error.Io, error.NoSpaceLeft, error.AccessDenied, error.ReadOnlyFileSystem, error.FileTooBig }) |err| {
        try testing.expect(writeMayHaveLanded(err));
    }
    try testing.expect(!writeMayHaveLanded(error.Busy));
    try testing.expect(!writeMayHaveLanded(error.InvalidTransition));
    try testing.expect(!writeMayHaveLanded(error.OutOfMemory));
}

test "a running tool call is saved once, and resume answers it as possibly run" {
    var t: TestHome = undefined;
    try t.init();
    defer t.deinit();
    var model = "m".*;
    const s = try Session.create(testing.allocator, &t.store, "/w", .ask, testSeed(&model));
    const user: types.UserTurn = .{ .text = @constCast("run it") };
    const call: types.ToolCall = .{ .id = "call_1", .name = "bash", .arguments_json = "{\"command\":\"sleep 9\"}" };
    try s.appendProgress(user, .{}, &.{call});
    try testing.expect(s.saved());
    try s.appendProgress(user, .{}, &.{call});
    const id = try testing.allocator.dupe(u8, s.id());
    defer testing.allocator.free(id);
    try testing.expectEqual(@as(usize, 1), try countItems(t.store.manager, id, running_type));
    try testing.expectEqual(@as(usize, 1), try countItems(t.store.manager, id, "user"));
    // Closing in the middle of the turn ends it `closed`.
    s.close();

    const r = try Session.resumeSession(testing.allocator, &t.store, .{ .id = id }, "/w", .ask);
    defer r.close();
    var restored = try r.restore(testing.allocator);
    defer restored.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 1), restored.history.len);
    const turn = restored.history[0].interrupted;
    try testing.expectEqualStrings("run it", turn.user.text);
    try testing.expect(turn.tool_call == null);
    try testing.expectEqual(@as(usize, 1), turn.execution.tool_steps.len);
    const step = turn.execution.tool_steps[0];
    try testing.expectEqual(@as(usize, 1), step.tool_calls.len);
    try testing.expectEqualStrings("call_1", step.tool_calls[0].id);
    try testing.expectEqualStrings("bash", step.tool_calls[0].name);
    try testing.expectEqualStrings("{\"command\":\"sleep 9\"}", step.tool_calls[0].arguments_json);
    try testing.expectEqual(@as(usize, 1), step.tool_results.len);
    try testing.expectEqualStrings("call_1", step.tool_results[0].tool_call_id);
    try testing.expectEqualStrings("bash", step.tool_results[0].tool_name);
    try testing.expectEqual(types.PersistedToolStatus.failure, step.tool_results[0].status);
    try testing.expectEqualStrings(unfinished_tool_output, step.tool_results[0].output);
}

test "a crash answers only the running calls its turn does not already hold" {
    var t: TestHome = undefined;
    try t.init();
    defer t.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const user_piece = try encodePiece(a, .{ .user = .{ .text = "two tools" } });
    const first = try encodePiece(a, .{ .tool_call = .{ .call_id = "call_1", .tool_name = "bash", .arguments_json = "{}" } });
    const second = try encodePiece(a, .{ .tool_call = .{ .call_id = "call_2", .tool_name = "read_file", .arguments_json = "{\"path\":\"a\"}" } });
    const imported = try t.store.manager.openImport(.{ .id = "1786460757753-tools", .workspace = "/w", .host = .ask, .created_ms = 1000 });
    _ = try imported.appendAt(&.{
        .turn_started,
        .{ .item = .{ .type = "user", .data = user_piece } },
        .{ .item = .{ .type = running_type, .data = first } },
        .{ .item = .{ .type = running_type, .data = second } },
        // The turn already holds call_1, as its one pending call.
        .{ .item = .{ .type = "tool_call", .data = first } },
        .{ .turn_interrupted = .crash },
    }, 2000);
    imported.release();

    const r = try Session.resumeSession(testing.allocator, &t.store, .{ .id = "1786460757753-tools" }, "/w", .ask);
    defer r.close();
    var restored = try r.restore(testing.allocator);
    defer restored.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 1), restored.history.len);
    const turn = restored.history[0].interrupted;
    try testing.expectEqual(types.InterruptedTerminalReason.failed, turn.terminal_reason);
    try testing.expectEqualStrings("call_1", turn.tool_call.?.id);
    try testing.expectEqual(@as(usize, 1), turn.execution.tool_steps.len);
    const step = turn.execution.tool_steps[0];
    try testing.expectEqual(@as(usize, 1), step.tool_calls.len);
    try testing.expectEqualStrings("call_2", step.tool_calls[0].id);
    try testing.expectEqualStrings("call_2", step.tool_results[0].tool_call_id);
    try testing.expectEqualStrings(unfinished_tool_output, step.tool_results[0].output);
}

test "a finished turn keeps no trace of its running calls" {
    var t: TestHome = undefined;
    try t.init();
    defer t.deinit();
    var model = "m".*;
    const s = try Session.create(testing.allocator, &t.store, "/w", .ask, testSeed(&model));
    const user: types.UserTurn = .{ .text = @constCast("asked") };
    const call: types.ToolCall = .{ .id = "call_9", .name = "bash", .arguments_json = "{}" };
    try s.appendProgress(user, .{}, &.{call});
    try s.commitTurn(.{ .assistant = .{ .user = user, .assistant = @constCast("answered") } }, types.ConversationLanguage.default());
    // The next turn starts with nothing saved as running.
    try testing.expectEqual(@as(usize, 0), s.running.items.len);
    const id = try testing.allocator.dupe(u8, s.id());
    defer testing.allocator.free(id);
    s.close();

    const r = try Session.resumeSession(testing.allocator, &t.store, .{ .id = id }, "/w", .ask);
    defer r.close();
    var restored = try r.restore(testing.allocator);
    defer restored.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 1), restored.history.len);
    try testing.expectEqualStrings("answered", restored.history[0].assistant.assistant);
    try testing.expectEqual(@as(usize, 0), restored.history[0].assistant.execution.tool_steps.len);
}

test "the history visit shows every turn, even those a compaction summarized" {
    var t: TestHome = undefined;
    try t.init();
    defer t.deinit();
    var model = "m".*;
    const s = try Session.create(testing.allocator, &t.store, "/w", .ask, testSeed(&model));
    defer s.close();
    const language = types.ConversationLanguage.default();
    try s.commitTurn(assistantTurn("one", "a"), language);
    try s.commitTurn(assistantTurn("two", "b"), language);
    try s.commitCompaction(.{ .summary = @constCast("summary"), .removed_turn_count = 2, .compaction_count = 1 }, false, null);
    try s.commitTurn(assistantTurn("three", "c"), language);

    const Visitor = struct {
        prompts: std.ArrayList([]u8) = .empty,
        fail_at: ?usize = null,

        pub fn append(self: *@This(), turn: types.HistoryTurn) !void {
            if (self.fail_at == self.prompts.items.len) return error.ConsumerFailed;
            try self.prompts.append(testing.allocator, try testing.allocator.dupe(u8, turn.assistant.user.text));
        }

        fn deinit(self: *@This()) void {
            for (self.prompts.items) |prompt| testing.allocator.free(prompt);
            self.prompts.deinit(testing.allocator);
        }
    };
    const numbers = try testing.allocator.dupe(?u64, s.turn_numbers.items);
    defer testing.allocator.free(numbers);
    var visitor: Visitor = .{};
    defer visitor.deinit();
    try s.visitHistory(testing.allocator, &visitor);
    try testing.expectEqual(@as(usize, 3), visitor.prompts.items.len);
    for (visitor.prompts.items, [_][]const u8{ "one", "two", "three" }) |got, want| try testing.expectEqualStrings(want, got);
    // The visit leaves what resume and compaction rely on unchanged.
    try testing.expectEqualSlices(?u64, numbers, s.turn_numbers.items);

    // A visitor that fails stops the visit without leaking its turn.
    var failing: Visitor = .{ .fail_at = 1 };
    defer failing.deinit();
    try testing.expectError(error.ConsumerFailed, s.visitHistory(testing.allocator, &failing));
    try testing.expectEqual(@as(usize, 1), failing.prompts.items.len);

    // A visitor may call back into the session, as the app does to read a
    // command replay's side file while it draws a turn.
    const Reentrant = struct {
        session: *Session,
        visits: usize = 0,

        pub fn append(self: *@This(), _: types.HistoryTurn) !void {
            // Fails rather than hangs if the visit holds the adapter's lock.
            if (!self.session.mutex.tryLock()) return error.VisitHeldSessionLock;
            self.session.mutex.unlock(io_mod.getIo());
            _ = try self.session.childCapability();
            self.visits += 1;
        }
    };
    var reentrant: Reentrant = .{ .session = s };
    try s.visitHistory(testing.allocator, &reentrant);
    try testing.expectEqual(@as(usize, 3), reentrant.visits);

    // Resume still starts from the summary.
    var restored = try s.restore(testing.allocator);
    defer restored.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 2), restored.history.len);
    try testing.expectEqualStrings("summary", restored.history[0].compacted_summary.summary);
    try testing.expectEqualStrings("three", restored.history[1].assistant.user.text);
}

test "a tool result backed only by its command replay streams as it commits" {
    var t: TestHome = undefined;
    try t.init();
    defer t.deinit();
    var model = "m".*;
    const s = try Session.create(testing.allocator, &t.store, "/w", .ask, testSeed(&model));
    const user: types.UserTurn = .{ .text = @constCast("run it") };
    var calls = [_]types.ToolCall{.{ .id = "call-1", .name = "shell", .arguments_json = "{}" }};
    // As a shell result arrives with `.required` command replay: no result file yet.
    var results = [_]types.PersistedToolResult{.{
        .tool_call_id = @constCast("call-1"),
        .tool_name = @constCast("shell"),
        .status = .success,
        .output = @constCast("REPLAY_ONLY_OUTPUT"),
        .output_bytes = 18,
        .stored_output_bytes = 18,
        .command_output_replay = .{ .available = .{ .handle = "fx-command-replay-0-0.bin", .framed_bytes = 27 } },
    }};
    var steps = [_]types.ToolExecutionStep{.{ .tool_calls = &calls, .tool_results = &results }};
    const execution: types.ExecutionMemory = .{ .tool_steps = &steps };
    try s.appendProgress(user, execution, &.{});
    try s.appendProgress(user, execution, &.{});
    const streamed = s.stored_results.get("call-1").?;
    const written = try (try s.childCapability()).stat(.tool_results, streamed);
    // The commit gets the agent's own turn, still without a result file;
    // preparing it fills in the handle and preview.
    try testing.expectEqual(@as(?[]u8, null), results[0].output_handle);
    try io_mod.getIo().sleep(.fromMilliseconds(5), .awake);
    var turn: types.HistoryTurn = .{ .assistant = .{ .user = user, .assistant = @constCast("done"), .execution = execution } };
    try s.prepareTurn(&turn);
    defer testing.allocator.free(results[0].output_handle.?);
    defer testing.allocator.free(results[0].preview.?);
    // The same file, not written again.
    try testing.expectEqualStrings(streamed, results[0].output_handle.?);
    const after = try (try s.childCapability()).stat(.tool_results, streamed);
    try testing.expectEqual(written.modified_at_ns, after.modified_at_ns);
    try s.commitTurn(turn, types.ConversationLanguage.default());
    const id = try testing.allocator.dupe(u8, s.id());
    defer testing.allocator.free(id);
    s.close();
    try testing.expectEqual(@as(usize, 0), try countItems(t.store.manager, id, superseded_type));
    try testing.expectEqual(@as(usize, 1), try countItems(t.store.manager, id, "tool_result"));
}

test "a tool result's rebuild time alone does not supersede its stream" {
    const a = "{\"call_id\":\"c\",\"created_at_ms\":100,\"preview\":\"x\"}";
    const b = "{\"call_id\":\"c\",\"created_at_ms\":142,\"preview\":\"x\"}";
    const c = "{\"call_id\":\"c\",\"created_at_ms\":142,\"preview\":\"y\"}";
    try testing.expect(samePiece(testing.allocator, a, a));
    try testing.expect(samePiece(testing.allocator, a, b));
    try testing.expect(!samePiece(testing.allocator, a, c));
    try testing.expect(!samePiece(testing.allocator, "{\"text\":\"a\"}", "{\"text\":\"b\"}"));
}

test "a streamed turn that differs from its commit is superseded, never mixed" {
    var t: TestHome = undefined;
    try t.init();
    defer t.deinit();
    var model = "m".*;
    const s = try Session.create(testing.allocator, &t.store, "/w", .ask, testSeed(&model));
    try s.appendProgress(.{ .text = @constCast("draft") }, .{}, &.{});
    try s.commitTurn(assistantTurn("final", "answer"), types.ConversationLanguage.default());
    const id = try testing.allocator.dupe(u8, s.id());
    defer testing.allocator.free(id);
    s.close();
    try testing.expectEqual(@as(usize, 1), try countItems(t.store.manager, id, superseded_type));

    const r = try Session.resumeSession(testing.allocator, &t.store, .{ .id = id }, "/w", .ask);
    defer r.close();
    var restored = try r.restore(testing.allocator);
    defer restored.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 1), restored.history.len);
    try testing.expectEqualStrings("final", restored.history[0].assistant.user.text);
}

test "a turn ended by a close or a crash comes back interrupted" {
    var t: TestHome = undefined;
    try t.init();
    defer t.deinit();
    const language = types.ConversationLanguage.default();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const user_piece = try encodePiece(arena.allocator(), .{ .user = .{ .text = "unfinished" } });

    // A close in the middle of a turn: the manager ends it `closed`.
    var model = "m".*;
    const s = try Session.create(testing.allocator, &t.store, "/w", .ask, testSeed(&model));
    try s.commitTurn(assistantTurn("done", "yes"), language);
    _ = try s.handle.append(&.{ .turn_started, .{ .item = .{ .type = "user", .data = user_piece } } });
    const id = try testing.allocator.dupe(u8, s.id());
    defer testing.allocator.free(id);
    s.close();
    {
        const r = try Session.resumeSession(testing.allocator, &t.store, .{ .id = id }, "/w", .ask);
        defer r.close();
        var restored = try r.restore(testing.allocator);
        defer restored.deinit(testing.allocator);
        try testing.expectEqual(@as(usize, 2), restored.history.len);
        try testing.expectEqualStrings("unfinished", restored.history[1].interrupted.user.text);
        try testing.expectEqual(types.InterruptedTerminalReason.cancelled, restored.history[1].interrupted.terminal_reason);
    }

    // A crash, written through the API as the v1 converter writes one.
    const crashed_id = "1786460757753-crash";
    const imported = try t.store.manager.openImport(.{ .id = crashed_id, .workspace = "/w", .host = .ask, .created_ms = 1000 });
    _ = try imported.appendAt(&.{ .turn_started, .{ .item = .{ .type = "user", .data = user_piece } }, .{ .turn_interrupted = .crash } }, 2000);
    imported.release();
    const r = try Session.resumeSession(testing.allocator, &t.store, .{ .id = crashed_id }, "/w", .ask);
    defer r.close();
    var restored = try r.restore(testing.allocator);
    defer restored.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 1), restored.history.len);
    try testing.expectEqual(types.InterruptedTerminalReason.failed, restored.history[0].interrupted.terminal_reason);
    try testing.expectEqual(@as(i64, 1000), restored.created_at_ms);
}

test "resume after a compaction starts with its summary and keeps the retained turn" {
    var t: TestHome = undefined;
    try t.init();
    defer t.deinit();
    var model = "m".*;
    const s = try Session.create(testing.allocator, &t.store, "/w", .ask, testSeed(&model));
    try s.commitTurn(assistantTurn("one", "1"), types.ConversationLanguage.default());
    try s.commitTurn(assistantTurn("two", "2"), types.ConversationLanguage.default());
    try s.commitTurn(assistantTurn("three", "3"), types.ConversationLanguage.default());
    var summary = "turns one and two".*;
    try s.commitCompaction(.{ .summary = &summary, .removed_turn_count = 2, .compaction_count = 1 }, false, .{ .turns = 2 });
    try s.commitTurn(assistantTurn("four", "4"), types.ConversationLanguage.default());
    const id = try testing.allocator.dupe(u8, s.id());
    defer testing.allocator.free(id);
    s.close();

    const r = try Session.resumeSession(testing.allocator, &t.store, .{ .id = id }, "/w", .ask);
    defer r.close();
    var restored = try r.restore(testing.allocator);
    defer restored.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 3), restored.history.len);
    try testing.expectEqualStrings("turns one and two", restored.history[0].compacted_summary.summary);
    try testing.expectEqualStrings("three", restored.history[1].assistant.user.text);
    try testing.expectEqualStrings("four", restored.history[2].assistant.user.text);
}

test "a piece above the inline limit goes to a blob and comes back whole" {
    var t: TestHome = undefined;
    try t.init();
    defer t.deinit();
    var model = "m".*;
    const s = try Session.create(testing.allocator, &t.store, "/w", .ask, testSeed(&model));
    const big = try testing.allocator.alloc(u8, max_inline_piece_bytes + 10);
    defer testing.allocator.free(big);
    @memset(big, 'x');
    try s.commitTurn(assistantTurn("big", big), types.ConversationLanguage.default());
    const id = try testing.allocator.dupe(u8, s.id());
    defer testing.allocator.free(id);
    s.close();

    const r = try Session.resumeSession(testing.allocator, &t.store, .{ .id = id }, "/w", .ask);
    defer r.close();
    var restored = try r.restore(testing.allocator);
    defer restored.deinit(testing.allocator);
    try testing.expectEqualStrings(big, restored.history[0].assistant.assistant);
}

test "usage is durable before its marker goes, and resume restores it" {
    var t: TestHome = undefined;
    try t.init();
    defer t.deinit();
    var model = "m".*;
    const s = try Session.create(testing.allocator, &t.store, "/w", .ask, testSeed(&model));
    try s.commitTurn(assistantTurn("q", "a"), types.ConversationLanguage.default());
    var usage = session_usage.Usage.initFresh();
    defer usage.deinit(testing.allocator);
    var snapshot = try usage.snapshot(testing.allocator);
    defer snapshot.deinit(testing.allocator);
    try s.persistUsage(snapshot);
    const id = try testing.allocator.dupe(u8, s.id());
    defer testing.allocator.free(id);
    // Nothing left to publish: the marker is gone.
    var markers = try t.tmp.dir.openDir(io_mod.getIo(), ".fx/" ++ usage_markers_dir_name, .{});
    defer markers.close(io_mod.getIo());
    try testing.expectError(error.FileNotFound, markers.statFile(io_mod.getIo(), id, .{}));
    s.close();

    const r = try Session.resumeSession(testing.allocator, &t.store, .{ .id = id }, "/w", .ask);
    defer r.close();
    var restored = try r.restore(testing.allocator);
    defer restored.deinit(testing.allocator);
    try testing.expect(restored.usage != null);
}

test "a usage marker that cannot be written names a storage cause (D29)" {
    // The real write is proven on a full disk end to end; opening the
    // markers folder makes it private again, so a unit test cannot block it.
    const Cause = ?error{ NoSpaceLeft, AccessDenied, ReadOnlyFileSystem, FileTooBig };
    try testing.expectEqual(@as(Cause, error.NoSpaceLeft), storageCause(error.NoSpaceLeft));
    try testing.expectEqual(@as(Cause, error.AccessDenied), storageCause(error.AccessDenied));
    try testing.expectEqual(@as(Cause, error.AccessDenied), storageCause(error.PermissionDenied));
    try testing.expectEqual(@as(Cause, error.ReadOnlyFileSystem), storageCause(error.ReadOnlyFileSystem));
    try testing.expectEqual(@as(Cause, error.FileTooBig), storageCause(error.FileTooBig));
    try testing.expectEqual(@as(Cause, null), storageCause(error.InputOutput));
}

test "a failed resume reports v1's error names" {
    var t: TestHome = undefined;
    try t.init();
    defer t.deinit();
    try testing.expectError(error.NoSavedSessions, Session.resumeSession(testing.allocator, &t.store, .last, "/w", .ask));
    try testing.expectError(error.SessionNotFound, Session.resumeSession(testing.allocator, &t.store, .{ .id = "AAAAAAAAAAAA" }, "/w", .ask));
    try testing.expectEqual(error.SessionBusy, resumeError(error.Busy, .last));
    try testing.expectEqual(error.SessionNotFound, resumeError(error.ChildSession, .{ .id = "AAAAAAAAAAAA" }));
    try testing.expectEqual(error.InvalidSessionFormat, resumeError(error.Corrupt, .last));
    try testing.expectEqual(error.UnsupportedSessionFormat, resumeError(error.UnsupportedVersion, .last));
    try testing.expectEqual(error.Io, resumeError(error.Io, .last));
    // An OS cause reaches the user as itself (D29).
    try testing.expectEqual(error.NoSpaceLeft, resumeError(error.NoSpaceLeft, .last));
    try testing.expectEqual(error.ReadOnlyFileSystem, resumeError(error.ReadOnlyFileSystem, .{ .id = "AAAAAAAAAAAA" }));
}

test "a resumed session gives v1's state and the title the user chose" {
    var t: TestHome = undefined;
    try t.init();
    defer t.deinit();
    var model = "m".*;
    const s = try Session.create(testing.allocator, &t.store, "/w", .app, testSeed(&model));
    try s.commitTurn(assistantTurn("q", "a"), types.ConversationLanguage.default());
    try s.rename("Chosen title");
    const id = try testing.allocator.dupe(u8, s.id());
    defer testing.allocator.free(id);
    s.close();

    const r = try Session.resumeSession(testing.allocator, &t.store, .{ .id = id }, "/w", .app);
    defer r.close();
    var resumed = try r.durableState(testing.allocator, "/w");
    defer resumed.deinit(testing.allocator);
    try testing.expectEqualStrings(id, resumed.state.id);
    try testing.expectEqualStrings("/w", resumed.state.workspace_root);
    try testing.expectEqual(@as(usize, 1), resumed.state.history.len);
    try testing.expectEqualStrings("m", resumed.state.preferences.model);
    try testing.expectEqualStrings("Chosen title", resumed.title.?);
    try testing.expect(resumed.state.created_at_ms > 0);
    try testing.expect(resumed.state.updated_at_ms >= resumed.state.created_at_ms);
    // A generated title does not replace the one the user chose.
    try testing.expect(!try r.installGeneratedTitle(resumed.state.history, "Generated"));
}

test "the picker lists saved root sessions newest first, without the open one" {
    var t: TestHome = undefined;
    try t.init();
    defer t.deinit();
    var model = "m".*;
    const first = try Session.create(testing.allocator, &t.store, "/w", .app, testSeed(&model));
    try first.commitTurn(assistantTurn("first", "a"), types.ConversationLanguage.default());
    const first_id = try testing.allocator.dupe(u8, first.id());
    defer testing.allocator.free(first_id);
    first.close();
    // A session with no turn is not saved, so it is never listed (D2).
    const empty = try Session.create(testing.allocator, &t.store, "/w", .app, testSeed(&model));
    empty.close();
    io_mod.sleep(2 * std.time.ns_per_ms);
    const second = try Session.create(testing.allocator, &t.store, "/w", .app, testSeed(&model));
    defer second.close();
    try second.commitTurn(assistantTurn("second", "b"), types.ConversationLanguage.default());

    var cancel = std.atomic.Value(bool).init(false);
    var all = try listSummaries(&t.store, testing.allocator, null, &cancel);
    defer {
        for (all.items) |*summary| summary.deinit(testing.allocator);
        all.deinit(testing.allocator);
    }
    try testing.expectEqual(@as(usize, 2), all.items.len);
    try testing.expectEqualStrings(second.id(), all.items[0].id);
    try testing.expectEqualStrings("/w", all.items[0].workspace_root.?);
    try testing.expect(all.items[0].hasResumableContent());

    var others = try listSummaries(&t.store, testing.allocator, second.id(), &cancel);
    defer {
        for (others.items) |*summary| summary.deinit(testing.allocator);
        others.deinit(testing.allocator);
    }
    try testing.expectEqual(@as(usize, 1), others.items.len);
    try testing.expectEqualStrings(first_id, others.items[0].id);

    cancel.store(true, .release);
    try testing.expectError(error.Cancelled, listSummaries(&t.store, testing.allocator, null, &cancel));
}

test "-c resumes the session this host last opened" {
    var t: TestHome = undefined;
    try t.init();
    defer t.deinit();
    var model = "m".*;
    const s = try Session.create(testing.allocator, &t.store, "/w", .app, testSeed(&model));
    try s.commitTurn(assistantTurn("q", "a"), types.ConversationLanguage.default());
    const id = try testing.allocator.dupe(u8, s.id());
    defer testing.allocator.free(id);
    s.close();

    const again = try Session.resumeSession(testing.allocator, &t.store, .last_opened, "/w", .app);
    defer again.close();
    try testing.expectEqualStrings(id, again.id());
    try testing.expectError(error.NoRememberedSession, Session.resumeSession(testing.allocator, &t.store, .last_opened, "/w", .acp));
}

test "side files live in a private session-files folder" {
    var t: TestHome = undefined;
    try t.init();
    defer t.deinit();
    var model = "m".*;
    const s = try Session.create(testing.allocator, &t.store, "/w", .ask, testSeed(&model));
    defer s.close();
    _ = try s.childCapability();
    try testing.expect(std.mem.endsWith(u8, s.filesPath(), s.id()));
    const stat = try t.tmp.dir.statFile(io_mod.getIo(), ".fx/" ++ files_dir_name, .{});
    try testing.expectEqual(@as(u32, 0o700), @as(u32, @intCast(stat.permissions.toMode() & 0o777)));
}

fn filesExist(t: *TestHome, id: []const u8) bool {
    const path = std.fs.path.join(testing.allocator, &.{ ".fx", files_dir_name, id }) catch return false;
    defer testing.allocator.free(path);
    t.tmp.dir.access(io_mod.getIo(), path, .{}) catch return false;
    return true;
}

test "a session that never reached the disk takes its side folder, and a saved one keeps it" {
    var t: TestHome = undefined;
    try t.init();
    defer t.deinit();
    var model = "m".*;
    const unsaved = try Session.create(testing.allocator, &t.store, "/w", .ask, testSeed(&model));
    _ = try unsaved.ensureFilesPath();
    const unsaved_id = try testing.allocator.dupe(u8, unsaved.id());
    defer testing.allocator.free(unsaved_id);
    try testing.expect(filesExist(&t, unsaved_id));
    unsaved.close();
    try testing.expect(!filesExist(&t, unsaved_id));

    const saved = try Session.create(testing.allocator, &t.store, "/w", .ask, testSeed(&model));
    _ = try saved.ensureFilesPath();
    try saved.commitTurn(assistantTurn("q", "a"), types.ConversationLanguage.default());
    const saved_id = try testing.allocator.dupe(u8, saved.id());
    defer testing.allocator.free(saved_id);
    saved.close();
    try testing.expect(filesExist(&t, saved_id));

    // A first write that landed although the adapter never saw it succeed:
    // its turn may point into the folder, so the folder stays.
    const landed = try Session.create(testing.allocator, &t.store, "/w", .ask, testSeed(&model));
    _ = try landed.ensureFilesPath();
    _ = try landed.handle.append(&.{ .turn_started, .turn_committed });
    const landed_id = try testing.allocator.dupe(u8, landed.id());
    defer testing.allocator.free(landed_id);
    landed.close();
    try testing.expect(filesExist(&t, landed_id));
}

test "recover copies side files and folders, never a link, and says when it left one out" {
    var t: TestHome = undefined;
    try t.init();
    defer t.deinit();
    const io = io_mod.getIo();
    var model = "m".*;
    const s = try Session.create(testing.allocator, &t.store, "/w", .ask, testSeed(&model));
    {
        var files = try std.Io.Dir.openDirAbsolute(io, try s.ensureFilesPath(), .{});
        defer files.close(io);
        try files.writeFile(io, .{ .sub_path = "top.txt", .data = "top", .flags = .{ .permissions = .fromMode(0o600) } });
        var nested = try io_mod.openOrCreateVerifiedPrivateDirFromDir(files, "nested");
        defer nested.close();
        try nested.dir.writeFile(io, .{ .sub_path = "inner.txt", .data = "inner", .flags = .{ .permissions = .fromMode(0o600) } });
        try files.symLink(io, "/etc/hosts", "link", .{});
    }
    try s.commitTurn(assistantTurn("q", "a"), types.ConversationLanguage.default());
    const id = try testing.allocator.dupe(u8, s.id());
    defer testing.allocator.free(id);
    s.close();

    var recovered = try recover(&t.store, testing.allocator, id);
    defer recovered.deinit(testing.allocator);
    try testing.expect(!recovered.files_complete);
    try testing.expectEqual(@as(usize, 1), recovered.history_len);
    const copy = try std.fs.path.join(testing.allocator, &.{ ".fx", files_dir_name, recovered.id });
    defer testing.allocator.free(copy);
    var dir = try t.tmp.dir.openDir(io, copy, .{});
    defer dir.close(io);
    const top = try dir.readFileAlloc(io, "top.txt", testing.allocator, .limited(64));
    defer testing.allocator.free(top);
    try testing.expectEqualStrings("top", top);
    const inner = try dir.readFileAlloc(io, "nested/inner.txt", testing.allocator, .limited(64));
    defer testing.allocator.free(inner);
    try testing.expectEqualStrings("inner", inner);
    const stat = try dir.statFile(io, "nested/inner.txt", .{});
    try testing.expectEqual(@as(u32, 0o600), @as(u32, @intCast(stat.permissions.toMode() & 0o777)));
    try testing.expectError(error.FileNotFound, dir.access(io, "link", .{ .follow_symlinks = false }));

    // The copy is a root session of its own; the source is unchanged.
    var copied = try readSession(&t.store, testing.allocator, recovered.id);
    defer copied.deinit(testing.allocator);
    try testing.expectEqualStrings("q", copied.state.history[0].assistant.user.text);
    try testing.expectError(error.SessionNotFound, recover(&t.store, testing.allocator, "NoSuchSession1"));
}

test "only the adapter imports the session manager" {
    // Set by `zig build test`; tests read the process environment directly.
    const root = std.mem.span(std.c.getenv("FX_TEST_SOURCE_ROOT") orelse return error.SkipZigTest);
    const io = io_mod.getIo();
    var dir = try std.Io.Dir.cwd().openDir(io, root, .{ .iterate = true });
    defer dir.close(io);
    var walker = try dir.walk(testing.allocator);
    defer walker.deinit();
    var checked: usize = 0;
    while (try walker.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.basename, ".zig")) continue;
        if (std.mem.startsWith(u8, entry.path, "core/session_manager/")) continue;
        if (std.mem.eql(u8, entry.path, "core/session/session_adapter.zig")) continue;
        const source = try dir.readFileAlloc(io, entry.path, testing.allocator, .limited(16 << 20));
        defer testing.allocator.free(source);
        if (std.mem.find(u8, source, "@import(\"session_manager\")") != null) {
            std.debug.print("{s} imports the session manager; only session_adapter.zig may\n", .{entry.path});
            return error.BoundaryViolation;
        }
        checked += 1;
    }
    try testing.expect(checked > 100);
}

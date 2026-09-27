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
const session_log = @import("session_log.zig");
const session_codec = @import("session_codec.zig");
const session_usage = @import("session_usage.zig");
const session_layout = @import("session_layout.zig");
const session_child_store = @import("session_child_store.zig");
const session_display_metadata = @import("session_display_metadata.zig");
const session_store = @import("session_store.zig");
const session_permission_state = @import("../permissions/session_permission_state.zig");
const model_provider = @import("../config/model_provider.zig");

const Allocator = std.mem.Allocator;
const Event = session_event.ConversationEvent;
const PieceKind = std.meta.Tag(Event);

/// Folder of the side files of v2 sessions, under `~/.fx`.
pub const files_dir_name = "session-files";
/// Folder of the usage-recovery markers of v2 sessions, under `~/.fx`;
/// v1's readers load only v1 sessions, so v2 markers live apart.
pub const usage_markers_dir_name = "usage-recovery-v2";
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

// ---------------------------------------------------------------------------
// Store: one per process

pub const Store = struct {
    manager: *sm.Manager,
    /// `$HOME`, owned: the base of `~/.fx/sessions/v2` and of the side folders.
    home: []u8,

    /// Touches no disk: a session's folder appears with its first turn.
    pub fn open(alloc: Allocator, home: []const u8) !Store {
        const owned_home = try alloc.dupe(u8, home);
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
pub const Seed = struct {
    preferences: session_codec.DurableSessionPreferences,
    language: types.ConversationLanguage,
    permission_state: session_permission_state.State,
};

pub const Target = union(enum) {
    id: []const u8,
    /// The newest updated root session in the workspace.
    last,
};

/// What resume gives back. Owns everything; free with `deinit`.
pub const Restored = struct {
    history: []types.HistoryTurn,
    language: types.ConversationLanguage,
    preferences: ?session_codec.DurableSessionPreferences = null,
    permission_state: ?session_permission_state.State = null,
    usage: ?session_usage.Snapshot = null,
    created_at_ms: i64,

    pub fn deinit(restored: *Restored, alloc: Allocator) void {
        types.freeHistoryTurnSlice(alloc, restored.history);
        if (restored.preferences) |*value| value.deinit(alloc);
        if (restored.permission_state) |*value| value.deinit(alloc);
        if (restored.usage) |*value| value.deinit(alloc);
        restored.* = undefined;
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

    pub fn create(alloc: Allocator, store: *Store, workspace: []const u8, host: Host, seed: Seed) !*Session {
        const handle = try store.manager.openNew(.{ .workspace = workspace, .host = host });
        errdefer handle.release();
        const self = try init(alloc, store, handle, true);
        errdefer self.destroyInner();
        var arena = std.heap.ArenaAllocator.init(alloc);
        defer arena.deinit();
        const a = arena.allocator();
        _ = try self.handle.append(&.{
            .{ .set = .{ .key = .prefs, .value = try encodePreferences(a, seed.preferences) } },
            .{ .set = .{ .key = .permissions, .value = try session_codec.encodePermissionState(a, seed.permission_state) } },
            .{ .set = .{ .key = .language, .value = try jsonString(a, seed.language.view()) } },
        });
        self.language = try alloc.dupe(u8, seed.language.view());
        return self;
    }

    pub fn resumeSession(alloc: Allocator, store: *Store, target: Target, workspace: []const u8, host: Host) !*Session {
        const handle = store.manager.openResume(.{
            .target = switch (target) {
                .id => |session_id| .{ .id = session_id },
                .last => .last,
            },
            .workspace = workspace,
            .host = host,
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

    /// `~/.fx/session-files/{id}`: borrowed until `close`.
    pub fn filesPath(self: *const Session) []const u8 {
        return self.files_path;
    }

    /// Closes the session and frees the adapter. Every child thread of this
    /// session must have joined (`tla/Wiring.tla` ParentOutlivesChildren).
    pub fn close(self: *Session) void {
        self.handle.close() catch |err| debug_trace.logf("session", "event=sessions_v2_close_failed session={s} err={s}", .{ self.id(), @errorName(err) });
        if (self.turn_open) debug_trace.logf("session", "event=sessions_v2_turn_closed_open session={s} streamed={d}", .{ self.id(), self.streamed.items.len });
        self.handle.release();
        self.destroyInner();
    }

    fn destroyInner(self: *Session) void {
        const alloc = self.alloc;
        if (self.capability) |*capability| capability.deinit();
        if (self.files_dir) |*dir| dir.close();
        self.clearStreamed();
        self.streamed.deinit(alloc);
        self.turn_numbers.deinit(alloc);
        if (self.language) |value| alloc.free(value);
        alloc.free(self.files_path);
        alloc.destroy(self);
    }

    fn clearStreamed(self: *Session) void {
        for (self.streamed.items) |bytes| self.alloc.free(bytes);
        self.streamed.clearRetainingCapacity();
    }

    // -- side files ----------------------------------------------------------

    /// The side-file capability over `~/.fx/session-files/{id}`, created
    /// `0700` on first use. Borrowed until `close`.
    pub fn childCapability(self: *Session) !*session_child_store.SessionChildCapability {
        self.mutex.lockUncancelable(io_mod.getIo());
        defer self.mutex.unlock(io_mod.getIo());
        if (self.capability) |*capability| return capability;
        const dir = try self.openFilesDir();
        self.capability = try session_child_store.SessionChildCapability.init(self.alloc, dir.dir, self.files_path, .writable);
        return &self.capability.?;
    }

    fn openFilesDir(self: *Session) !*io_mod.VerifiedDir {
        if (self.files_dir) |*dir| return dir;
        var home = io_mod.VerifiedDir{ .dir = try std.Io.Dir.openDirAbsolute(io_mod.getIo(), self.store.home, .{ .follow_symlinks = false }) };
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
    /// them handles, as v1 does at commit.
    pub fn prepareTurn(self: *Session, turn: *types.HistoryTurn) !void {
        try session_log.externalizeConversationTurnResults(self.alloc, turn, try self.childCapability());
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
    /// completed since. A tool result whose side file does not exist yet
    /// waits for the commit, which is authoritative.
    pub fn appendProgress(self: *Session, user: types.UserTurn, execution: types.ExecutionMemory) !void {
        self.mutex.lockUncancelable(io_mod.getIo());
        defer self.mutex.unlock(io_mod.getIo());
        var arena = std.heap.ArenaAllocator.init(self.alloc);
        defer arena.deinit();
        const a = arena.allocator();
        var events: std.ArrayList(Event) = .empty;
        try events.append(a, .{ .user = .{ .text = user.text, .images = user.images, .work_id = user.work_id } });
        // On a missing handle the list keeps the pieces before it.
        session_event.appendExecutionConversationEvents(a, &events, execution) catch |err| switch (err) {
            error.ConversationArtifactRequired => {},
            else => return err,
        };
        const encoded = try a.alloc([]u8, events.items.len);
        for (events.items, encoded) |event, *bytes| bytes.* = try encodePiece(a, event);
        const start = try self.streamedPrefix(encoded);
        if (self.turn_open and start == encoded.len) return;
        try self.writePieces(a, events.items[start..], encoded[start..], &.{});
        try self.streamed.ensureUnusedCapacity(self.alloc, encoded.len - start);
        for (encoded[start..]) |bytes| self.streamed.appendAssumeCapacity(try self.alloc.dupe(u8, bytes));
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
        const item_type = itemType(kind) orelse return error.InvalidConversationFrame;
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
        _ = try self.handle.append(&.{.{ .set = .{ .key = .prefs, .value = try encodePreferences(arena.allocator(), preferences) } }});
    }

    pub fn setPermissions(self: *Session, state: session_permission_state.State) !void {
        const value = try session_codec.encodePermissionState(self.alloc, state);
        defer self.alloc.free(value);
        _ = try self.handle.append(&.{.{ .set = .{ .key = .permissions, .value = value } }});
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
        try io_mod.durableReplaceVerified(self.alloc, &dir, self.id(), content);
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
        var home = io_mod.VerifiedDir{ .dir = try std.Io.Dir.openDirAbsolute(io_mod.getIo(), self.store.home, .{ .follow_symlinks = false }) };
        defer home.close();
        var fx = try io_mod.openOrCreateVerifiedPrivateDir(&home, profile_paths.root_dir_name);
        defer fx.close();
        return io_mod.openOrCreateVerifiedPrivateDir(&fx, usage_markers_dir_name);
    }

    // -- resume --------------------------------------------------------------

    /// Rebuilds fx's history and settings from the log: the newest
    /// compaction's summary, then every turn after it (or after the turn it
    /// kept). A turn a crash or close ended has no `interruption` item and
    /// comes back interrupted: `failed` after a crash, `cancelled` after a
    /// close.
    pub fn restore(self: *Session, alloc: Allocator) !Restored {
        self.mutex.lockUncancelable(io_mod.getIo());
        defer self.mutex.unlock(io_mod.getIo());
        var scratch = std.heap.ArenaAllocator.init(alloc);
        defer scratch.deinit();
        const sa = scratch.allocator();
        const state = try self.handle.state(sa);
        self.last_turn = state.last_turn;

        const language = if (state.language) |raw| try decodeLanguage(sa, raw) else types.ConversationLanguage.default();
        var restored: Restored = .{
            .history = &.{},
            .language = language,
            .created_at_ms = std.math.cast(i64, state.created_ms) orelse 0,
        };
        errdefer restored.deinit(alloc);
        if (state.prefs) |raw| restored.preferences = try decodePreferences(alloc, raw);
        if (state.permissions) |raw| restored.permission_state = try session_codec.decodePermissionState(alloc, raw);
        if (state.usage) |raw| {
            const checkpoint = try decodeUsage(alloc, raw);
            restored.usage = checkpoint.snapshot;
            self.usage_at_ms = checkpoint.at_ms;
            self.usage_marked = session_usage.needsProfileRecovery(checkpoint.snapshot);
        }
        if (state.language != null) {
            const owned = try self.alloc.dupe(u8, language.view());
            if (self.language) |old| self.alloc.free(old);
            self.language = owned;
        }

        var history: std.ArrayList(types.HistoryTurn) = .empty;
        errdefer {
            for (history.items) |turn| types.freeHistoryTurn(alloc, turn);
            history.deinit(alloc);
        }
        self.turn_numbers.clearRetainingCapacity();
        var from: sm.From = .start;
        var skip_offset: ?u64 = null;
        if (state.compaction_offset) |offset| {
            const cursor: sm.Cursor = .{ .offset = offset, .seq = state.last_compaction_seq.? };
            var page = try self.handle.read(sa, .{ .at = cursor }, .forward, 1);
            defer page.deinit();
            const data = page.entries[0].body.?.compacted.data;
            const compacted = try std.json.parseFromSliceLeaky(CompactedData, sa, data, .{});
            try history.append(alloc, .{ .compacted_summary = .{
                .summary = try alloc.dupe(u8, compacted.summary),
                .removed_turn_count = compacted.removed_turn_count,
                .compaction_count = compacted.compaction_count,
                .root_user_messages_complete = false,
                .permission_feedback_complete = false,
            } });
            try self.turn_numbers.append(self.alloc, null);
            skip_offset = offset;
            from = .{ .at = cursor };
            if (compacted.keep_from_turn) |turn| from = .{ .at = try self.findTurnStart(sa, cursor, turn) };
        }
        try self.replay(alloc, sa, from, skip_offset, &history);
        restored.history = try history.toOwnedSlice(alloc);
        return restored;
    }

    /// The cursor of `turn`'s `turn_started`, reading back from `before`.
    fn findTurnStart(self: *Session, sa: Allocator, before: sm.Cursor, turn: u64) !sm.Cursor {
        var from: sm.From = .{ .at = before };
        while (true) {
            var page = try self.handle.read(sa, from, .backward, replay_page_lines);
            defer page.deinit();
            for (page.entries) |entry| {
                const body = entry.body orelse continue;
                if (body == .turn_started and body.turn_started.turn == turn) return .{ .offset = entry.offset, .seq = entry.seq };
            }
            from = .{ .at = page.next orelse return error.InvalidConversationFrame };
        }
    }

    fn replay(
        self: *Session,
        alloc: Allocator,
        sa: Allocator,
        start: sm.From,
        skip_offset: ?u64,
        history: *std.ArrayList(types.HistoryTurn),
    ) !void {
        var builder = session_log.ConversationTurnBuilder.init(alloc);
        defer builder.deinit();
        var interrupted_item = false;
        var superseded = false;
        var piece_arena = std.heap.ArenaAllocator.init(alloc);
        defer piece_arena.deinit();
        var from = start;
        while (true) {
            var page = try self.handle.read(sa, from, .forward, replay_page_lines);
            defer page.deinit();
            if (page.damaged) debug_trace.logf("session", "event=sessions_v2_replay_damaged session={s} dropped=lines_after_damage", .{self.id()});
            for (page.entries) |entry| {
                _ = piece_arena.reset(.retain_capacity);
                const pa = piece_arena.allocator();
                if (skip_offset) |offset| if (entry.offset == offset) continue;
                const body = entry.body orelse continue;
                switch (body) {
                    .turn_started => {
                        interrupted_item = false;
                        superseded = false;
                    },
                    .item => |piece| {
                        if (std.mem.eql(u8, piece.type, superseded_type)) {
                            superseded = true;
                            continue;
                        }
                        const kind = pieceKind(piece.type) orelse {
                            debug_trace.logf("session", "event=sessions_v2_unknown_item session={s} type={s} dropped=item", .{ self.id(), piece.type });
                            continue;
                        };
                        const data = try self.pieceData(pa, piece);
                        switch (try decodePiece(pa, kind, data)) {
                            .user => |value| try builder.begin(value),
                            .assistant => |value| try builder.appendAssistant(value),
                            .tool_call => |value| try builder.appendToolCall(value),
                            .tool_result => |value| try builder.appendToolResult(value),
                            .steering => |value| try builder.appendSteering(value.text),
                            .turn_completed => |value| {
                                const turn = try builder.finishAssistant(value);
                                errdefer types.freeHistoryTurn(alloc, turn);
                                try history.append(alloc, turn);
                                try self.turn_numbers.append(self.alloc, self.lastStarted(entry));
                            },
                            .interrupted => |value| {
                                interrupted_item = true;
                                const turn = try builder.finishInterrupted(value);
                                errdefer types.freeHistoryTurn(alloc, turn);
                                try history.append(alloc, turn);
                                try self.turn_numbers.append(self.alloc, self.lastStarted(entry));
                            },
                            .context_checkpoint => return error.InvalidConversationFrame,
                        }
                    },
                    .turn_interrupted => |ended| {
                        if (superseded) {
                            debug_trace.logf("session", "event=sessions_v2_replay_superseded session={s} turn={d} dropped=stale_turn", .{ self.id(), ended.turn });
                            builder.deinit();
                            builder = session_log.ConversationTurnBuilder.init(alloc);
                            superseded = false;
                            continue;
                        }
                        if (interrupted_item or builder.isIdle()) continue;
                        const turn = try builder.finishInterrupted(.{ .reason = switch (ended.reason) {
                            .cancel, .closed => .cancelled,
                            .failed, .crash => .failed,
                        } });
                        errdefer types.freeHistoryTurn(alloc, turn);
                        try history.append(alloc, turn);
                        try self.turn_numbers.append(self.alloc, ended.turn);
                    },
                    .session_created, .compacted, .turn_committed, .set, .child_spawned, .child_finished, .snapshot, .closed => {},
                }
            }
            from = .{ .at = page.next orelse break };
        }
        if (!builder.isIdle()) debug_trace.logf("session", "event=sessions_v2_replay_open_turn session={s} dropped=unfinished_pieces", .{self.id()});
    }

    fn lastStarted(self: *const Session, entry: sm.Entry) ?u64 {
        _ = self;
        return switch (entry.body.?) {
            .item => |piece| piece.turn,
            else => null,
        };
    }

    /// A piece's bytes, from its blob when the line holds a reference.
    fn pieceData(self: *Session, pa: Allocator, piece: sm.Body.Piece) ![]const u8 {
        if (piece.blobs.len != 1 or !std.mem.startsWith(u8, piece.data, "{\"" ++ blob_ref_key ++ "\":")) return piece.data;
        return self.store.manager.getBlob(pa, self.id(), piece.blobs[0]);
    }
};

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
const ResumeError = error{ SessionNotFound, NoSavedSessions, SessionBusy, InvalidSessionFormat, UnsupportedSessionFormat } || sm.OpenError;

fn resumeError(err: sm.OpenError, target: Target) ResumeError {
    return switch (err) {
        error.NotFound => switch (target) {
            .id => error.SessionNotFound,
            .last => error.NoSavedSessions,
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
fn decodePiece(arena: Allocator, kind: PieceKind, data: []const u8) !Event {
    const event: Event = switch (kind) {
        inline else => |tag| @unionInit(Event, @tagName(tag), try std.json.parseFromSliceLeaky(
            @FieldType(Event, @tagName(tag)),
            arena,
            data,
            .{ .allocate = .alloc_always },
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
fn encodePreferences(alloc: Allocator, preferences: session_codec.DurableSessionPreferences) ![]u8 {
    const Saved = struct {
        provider: model_provider.ProviderId,
        model: []const u8,
        effort: []const u8,
        fast_mode: bool,
    };
    return jsonValue(alloc, Saved{
        .provider = preferences.provider,
        .model = preferences.model,
        .effort = preferences.effort.label(),
        .fast_mode = preferences.fast_mode,
    });
}

fn decodePreferences(alloc: Allocator, raw: []const u8) !session_codec.DurableSessionPreferences {
    var parsed = try std.json.parseFromSlice(std.json.Value, alloc, raw, .{});
    defer parsed.deinit();
    return session_codec.parse_preferences(alloc, parsed.value);
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
        const raw = try encodePreferences(arena.allocator(), .{ .model = &model, .effort = effort, .fast_mode = true });
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
    try s.appendProgress(user, .{});
    // The first piece published the session.
    try testing.expect(s.saved());
    try s.appendProgress(user, .{});
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
    try s.appendProgress(.{ .text = @constCast("draft") }, .{});
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

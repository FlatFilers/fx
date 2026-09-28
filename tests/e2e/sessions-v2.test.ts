import { expect, test } from "bun:test";
import { spawn } from "node:child_process";
import {
  appendFileSync,
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  realpathSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { FX_BIN, runFx } from "../evals/eval-helpers";
import {
  FAKE_GATEWAY_MODEL,
  type FakeGatewayOptions,
  fakeGatewayFinalText,
  fakeShellRun,
  startDynamicFakeGateway,
  startFakeGateway,
  TmuxSession,
  tmuxAvailable,
} from "./tmux-helpers";

// Sessions v2 behind FX_SESSIONS_V2 and --sessions-v2: every `fx ask` entry
// and exit, the files it writes, and the faults a real disk and a real
// crash produce: kills mid-stream and mid-tool, a torn tail, a flipped
// byte, a second process, a read-only folder and a full disk.

const TIMEOUT = 30_000;

type Fixture = { root: string; home: string; workspace: string };

function createFixture(prefix: string): Fixture {
  const root = mkdtempSync(join(tmpdir(), prefix));
  const home = join(root, "home");
  const workspace = join(root, "workspace");
  mkdirSync(home);
  mkdirSync(workspace);
  return { root, home: realpathSync(home), workspace: realpathSync(workspace) };
}

function env(fixture: Fixture, gateway: { baseUrl: string; chatUrl: string }, v2 = true) {
  return {
    HOME: fixture.home,
    AI_GATEWAY_API_KEY: "sessions-v2-test-key",
    VERCEL_OIDC_TOKEN: undefined,
    FX_GATEWAY_BASE_URL: gateway.baseUrl,
    FX_GATEWAY_CHAT_URL: gateway.chatUrl,
    FX_E2E_GATEWAY_CHAT_URL: gateway.chatUrl,
    FX_MODEL: FAKE_GATEWAY_MODEL,
    FX_AUTO_UPGRADE: "0",
    FX_SESSIONS_V2: v2 ? "1" : undefined,
  };
}

function v2Root(fixture: Fixture) {
  return join(fixture.home, ".fx", "sessions", "v2");
}

type Line = { seq: number; kind: string; type?: string; reason?: string; key?: string };

/// Every line of a session's log. A streamed turn always matches its
/// commit, so no turn is ever superseded.
function logLines(fixture: Fixture, id: string): Line[] {
  const text = readFileSync(join(v2Root(fixture), id, "log.jsonl"), "utf8");
  const lines: Line[] = text.trimEnd().split("\n").map((line) => JSON.parse(line));
  expect(lines.filter((line) => line.type === "superseded")).toEqual([]);
  return lines;
}

const crcTable = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) c = c & 1 ? 0x82f63b78 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

/// CRC32C, computed here rather than trusted from the code under test.
function crc32c(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (const byte of bytes) crc = crcTable[(crc ^ byte) & 0xff]! ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

/// The log is whole: every line ends in a newline and carries a CRC32C over
/// the bytes before `,"crc":"`, `seq` counts from 1 without gaps, and each
/// `turn_started` ends once before the next one begins.
function expectWholeLog(fixture: Fixture, id: string) {
  const bytes = readFileSync(join(v2Root(fixture), id, "log.jsonl"));
  expect(bytes.at(-1)).toBe(0x0a);
  const marker = Buffer.from(',"crc":"');
  let start = 0;
  let seq = 0;
  let open = false;
  while (start < bytes.length) {
    const end = bytes.indexOf(0x0a, start);
    const line = bytes.subarray(start, end + 1);
    const at = line.lastIndexOf(marker);
    const stored = parseInt(line.subarray(at + marker.length, at + marker.length + 8).toString(), 16);
    expect(crc32c(line.subarray(0, at))).toBe(stored);
    const entry = JSON.parse(line.toString());
    seq += 1;
    expect(entry.seq).toBe(seq);
    if (entry.kind === "turn_started") {
      expect(open).toBe(false);
      open = true;
    } else if (entry.kind === "turn_committed" || entry.kind === "turn_interrupted") {
      expect(open).toBe(true);
      open = false;
    }
    start = end + 1;
  }
}

/// Every tool call the model is sent has its result: an unpaired call is
/// rejected by providers.
/// The tool calls and tool results in a Gateway request's prompt.
function promptToolParts(body: string) {
  const prompt: any[] = JSON.parse(body).prompt ?? [];
  const parts: any[] = prompt.flatMap((message) => (Array.isArray(message.content) ? message.content : []));
  return {
    calls: parts.filter((part) => part.type === "tool-call"),
    results: parts.filter((part) => part.type === "tool-result"),
  };
}

function expectPairedToolCalls(body: string) {
  const { calls, results } = promptToolParts(body);
  expect(calls.length).toBeGreaterThan(0);
  const answered = new Set(results.map((part) => part.toolCallId));
  for (const call of calls) expect(answered.has(call.toolCallId)).toBe(true);
}

/// Waits until the log contains `needle`, or fails after `timeoutMs`.
async function waitForLog(fixture: Fixture, id: string, needle: string, timeoutMs = 10_000) {
  const path = join(v2Root(fixture), id, "log.jsonl");
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (existsSync(path) && readFileSync(path, "utf8").includes(needle)) return;
    await Bun.sleep(50);
  }
  throw new Error(`log never contained ${needle}`);
}

function spawnAsk(fixture: Fixture, gateway: any, args: string[]) {
  const child = spawn(FX_BIN, ["ask", "--json", "--auto", ...args], {
    cwd: fixture.workspace,
    env: { ...process.env, ...env(fixture, gateway) } as Record<string, string>,
    stdio: "ignore",
  });
  const exited = new Promise((resolve) => child.on("exit", resolve));
  return { child, exited };
}

/// `fx ask` under a file-size limit of `blocks` 512-byte blocks with SIGXFSZ
/// ignored, so a write past it fails with EFBIG the way a full disk fails
/// with ENOSPC. The ignored signal survives the `exec`.
function askWithSizeLimit(fixture: Fixture, gateway: any, blocks: number, args: string[]) {
  return new Promise<{ code: number | null; stdout: string; stderr: string }>((resolve) => {
    const child = spawn(
      "/bin/sh",
      ["-c", `trap '' XFSZ; ulimit -f ${blocks}; exec "$0" "$@"`, FX_BIN, "ask", "--json", "--auto", ...args],
      { cwd: fixture.workspace, env: { ...process.env, ...env(fixture, gateway) } as Record<string, string> },
    );
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => (stdout += chunk));
    child.stderr.on("data", (chunk) => (stderr += chunk));
    child.on("exit", (code) => resolve({ code, stdout, stderr }));
  });
}

test("the TypeScript CRC32C matches the standard check value", () => {
  expect(crc32c(Buffer.from("123456789"))).toBe(0xe3069283);
});

/// Kinds and item types, in order: `item:user`, `turn_committed`, ...
function shape(lines: Line[]): string[] {
  return lines
    .filter((line) => line.kind !== "snapshot" && line.kind !== "set")
    .map((line) => (line.kind === "item" ? `item:${line.type}` : line.kind));
}

/// The v1 sessions folder holds nothing but the v2 root: no dual writes.
function expectNoV1Sessions(fixture: Fixture) {
  const sessions = join(fixture.home, ".fx", "sessions");
  expect(readdirSync(sessions).filter((name) => name !== "v2")).toEqual([]);
}

async function ask(fixture: Fixture, gateway: any, args: string[], v2 = true) {
  const result = await runFx(["ask", "--json", "--auto", ...args], {
    cwd: fixture.workspace,
    env: env(fixture, gateway, v2),
    timeoutMs: TIMEOUT,
  });
  return result;
}

test("fx ask saves to v2, resumes by id and by last, and never writes v1", async () => {
  const fixture = createFixture("fx-v2-ask-");
  const gateway = startFakeGateway([
    fakeGatewayFinalText("V2_FIRST_ANSWER"),
    fakeGatewayFinalText("V2_SECOND_ANSWER"),
    fakeGatewayFinalText("V2_THIRD_ANSWER"),
  ]);
  try {
    const created = await ask(fixture, gateway, ["First v2 question."]);
    expect(created.code).toBe(0);
    expect(created.stderr).toBe("");
    const first = JSON.parse(created.stdout);
    expect(first.output).toBe("V2_FIRST_ANSWER");
    const id: string = first.session_id;
    expect(id.length).toBeGreaterThan(0);
    expect(shape(logLines(fixture, id))).toEqual([
      "session_created",
      "turn_started",
      "item:user",
      "item:assistant",
      "item:turn_end",
      "turn_committed",
      "closed",
    ]);
    expectNoV1Sessions(fixture);
    // Owner-only folders and files.
    expect(statSync(v2Root(fixture)).mode & 0o777).toBe(0o700);
    expect(statSync(join(v2Root(fixture), id, "log.jsonl")).mode & 0o777).toBe(0o600);

    const byId = await ask(fixture, gateway, ["--resume-id", id, "Second v2 question."]);
    expect(byId.code).toBe(0);
    expect(byId.stderr).toBe("");
    expect(JSON.parse(byId.stdout).session_id).toBe(id);
    expect(gateway.requests[1]!.body).toContain("V2_FIRST_ANSWER");

    const byLast = await ask(fixture, gateway, ["--resume", "last", "Third v2 question."]);
    expect(byLast.code).toBe(0);
    expect(JSON.parse(byLast.stdout).session_id).toBe(id);
    expect(gateway.requests[2]!.body).toContain("V2_SECOND_ANSWER");
    expect(gateway.requests[2]!.body).toContain("First v2 question.");

    const kinds = logLines(fixture, id).map((line) => line.kind);
    expect(kinds.filter((kind) => kind === "turn_committed").length).toBe(3);
    expectNoV1Sessions(fixture);
  } finally {
    gateway.stop();
    rmSync(fixture.root, { recursive: true, force: true });
  }
}, TIMEOUT * 3);

test("the flag works before and after ask, and --no-save writes nothing", async () => {
  const fixture = createFixture("fx-v2-flag-");
  const gateway = startFakeGateway([
    fakeGatewayFinalText("FLAG_BEFORE"),
    fakeGatewayFinalText("FLAG_AFTER"),
    fakeGatewayFinalText("NOT_SAVED"),
  ]);
  try {
    const before = await runFx(["--sessions-v2", "ask", "--json", "--auto", "Flag before ask."], {
      cwd: fixture.workspace,
      env: env(fixture, gateway, false),
      timeoutMs: TIMEOUT,
    });
    expect(before.code).toBe(0);
    expect(before.stderr).toBe("");
    const before_id = JSON.parse(before.stdout).session_id;
    expect(existsSync(join(v2Root(fixture), before_id, "log.jsonl"))).toBe(true);

    const after = await ask(fixture, gateway, ["--sessions-v2", "Flag after ask."], false);
    expect(after.code).toBe(0);
    expect(after.stderr).toBe("");
    const after_id = JSON.parse(after.stdout).session_id;
    expect(existsSync(join(v2Root(fixture), after_id, "log.jsonl"))).toBe(true);
    expectNoV1Sessions(fixture);

    const unsaved = await ask(fixture, gateway, ["--no-save", "Not saved."]);
    expect(unsaved.code).toBe(0);
    expect(JSON.parse(unsaved.stdout).session_id).toBe("");
    const folders = readdirSync(v2Root(fixture)).filter((name) => !name.startsWith(".") && !name.startsWith("index"));
    expect(folders.sort()).toEqual([before_id, after_id].sort());
  } finally {
    gateway.stop();
    rmSync(fixture.root, { recursive: true, force: true });
  }
}, TIMEOUT * 3);

test("a tool turn keeps its result as a side file and v1 ignores the v2 root", async () => {
  const fixture = createFixture("fx-v2-tool-");
  const gateway = startFakeGateway([
    fakeShellRun("v2-shell-1", "echo V2_TOOL_OUTPUT_7731"),
    fakeGatewayFinalText("V2_TOOL_DONE"),
    fakeGatewayFinalText("V2_TOOL_RESUMED"),
  ]);
  try {
    const created = await ask(fixture, gateway, ["Run the tool."]);
    expect(created.code).toBe(0);
    // Tool progress goes to stderr; the answer and session id to stdout.
    expect(created.stderr).toContain("V2_TOOL_OUTPUT_7731");
    const id = JSON.parse(created.stdout).session_id;
    const lines = logLines(fixture, id);
    expect(shape(lines)).toContain("item:tool_call");
    expect(shape(lines)).toContain("item:tool_result");
    // The body is a side file in ~/.fx/session-files/{id}, not in the log.
    const files = join(fixture.home, ".fx", "session-files", id);
    expect(statSync(files).mode & 0o777).toBe(0o700);
    expect(readdirSync(files).length).toBeGreaterThan(0);

    const resumed = await ask(fixture, gateway, ["--resume", "last", "What did the tool print?"]);
    expect(resumed.code).toBe(0);
    expect(gateway.requests.at(-1)!.body).toContain("V2_TOOL_OUTPUT_7731");

    // A v1 process lists no session named v2.
    const listed = await runFx(["sessions", "--json"], {
      cwd: fixture.workspace,
      env: env(fixture, gateway, false),
      timeoutMs: TIMEOUT,
    });
    expect(listed.code).toBe(0);
    expect(listed.stdout).not.toContain("\"v2\"");
    expectNoV1Sessions(fixture);
  } finally {
    gateway.stop();
    rmSync(fixture.root, { recursive: true, force: true });
  }
}, TIMEOUT * 3);

test("fx ask killed in the middle of a turn resumes with that turn interrupted", async () => {
  const fixture = createFixture("fx-v2-kill-");
  let stalled: () => void = () => {};
  const reachedStall = new Promise<void>((resolve) => (stalled = resolve));
  // Replies follow the request, not the call count: a fresh session also
  // asks for a title in the background.
  const gateway = startDynamicFakeGateway(async (body) => {
    if (body.includes("After the kill.")) return fakeGatewayFinalText("AFTER_KILL_ANSWER");
    if (body.includes("This turn is killed.")) {
      stalled();
      return new Promise<Response>(() => {});
    }
    return fakeGatewayFinalText("BEFORE_KILL_ANSWER");
  });
  try {
    const created = await ask(fixture, gateway, ["Before the kill."]);
    expect(created.code).toBe(0);
    const id = JSON.parse(created.stdout).session_id;

    const child = spawn(FX_BIN, ["ask", "--json", "--auto", "--resume-id", id, "This turn is killed."], {
      cwd: fixture.workspace,
      env: { ...process.env, ...env(fixture, gateway) } as Record<string, string>,
      stdio: "ignore",
    });
    await reachedStall;
    const exited = new Promise((resolve) => child.on("exit", resolve));
    child.kill("SIGKILL");
    await exited;

    const resumed = await ask(fixture, gateway, ["--resume-id", id, "After the kill."]);
    expect(resumed.code).toBe(0);
    expect(resumed.stderr).toBe("");
    expect(JSON.parse(resumed.stdout).output).toBe("AFTER_KILL_ANSWER");
    // The history the model sees keeps the first turn.
    expect(gateway.requests.at(-1)!.body).toContain("BEFORE_KILL_ANSWER");
    // The killed turn's user piece was saved before the model call; the
    // reopen ended that turn as a crash, and the model saw it.
    expect(gateway.requests.at(-1)!.body).toContain("This turn is killed.");
    const lines = logLines(fixture, id);
    for (const [index, line] of lines.entries()) expect(line.seq).toBe(index + 1);
    const crashed = lines.filter((line) => line.kind === "turn_interrupted");
    expect(crashed.map((line) => line.reason)).toEqual(["crash"]);
    expect(shape(lines).filter((kind) => kind === "turn_committed").length).toBe(2);
    expect(shape(lines).filter((kind) => kind === "item:user").length).toBe(3);
    expectWholeLog(fixture, id);
    expectNoV1Sessions(fixture);
  } finally {
    gateway.stop();
    rmSync(fixture.root, { recursive: true, force: true });
  }
}, TIMEOUT * 3);

test("a kill while a tool runs keeps the finished tool and answers the running one", async () => {
  const fixture = createFixture("fx-v2-kill-tool-");
  let slowServed: () => void = () => {};
  const slowStarted = new Promise<void>((resolve) => (slowServed = resolve));
  const gateway = startDynamicFakeGateway(async (body) => {
    if (body.includes("After the tool kill.")) return fakeGatewayFinalText("AFTER_TOOL_KILL");
    if (body.includes("Run two tools.") && body.includes("FIRST_TOOL_OUTPUT_5521")) {
      slowServed();
      return fakeShellRun("v2-slow-2", "sleep 5");
    }
    if (body.includes("Run two tools.")) return fakeShellRun("v2-fast-1", "echo FIRST_TOOL_OUTPUT_5521");
    return fakeGatewayFinalText("BEFORE_TOOL_KILL");
  });
  try {
    const created = await ask(fixture, gateway, ["Before the tool kill."]);
    expect(created.code).toBe(0);
    const id = JSON.parse(created.stdout).session_id;

    const run = spawnAsk(fixture, gateway, ["--resume-id", id, "Run two tools."]);
    await slowStarted;
    // The call is saved before it runs (D28).
    await waitForLog(fixture, id, "v2-slow-2");
    run.child.kill("SIGKILL");
    await run.exited;

    const resumed = await ask(fixture, gateway, ["--resume-id", id, "After the tool kill."]);
    expect(resumed.code).toBe(0);
    expect(resumed.stderr).toBe("");
    expect(JSON.parse(resumed.stdout).output).toBe("AFTER_TOOL_KILL");
    const body = gateway.requests.at(-1)!.body;
    expect(body).toContain("BEFORE_TOOL_KILL");
    // The finished tool survives the crash, and the running one comes back
    // answered as possibly run, so every call keeps a result.
    expect(body).toContain("FIRST_TOOL_OUTPUT_5521");
    expectPairedToolCalls(body);
    const { calls, results } = promptToolParts(body);
    expect(calls.map((part) => part.toolCallId)).toEqual(["v2-fast-1", "v2-slow-2"]);
    const slow = results.find((part) => part.toolCallId === "v2-slow-2");
    expect(slow?.output?.type).toBe("error-text");
    expect(slow?.output?.value).toContain("may have partly run");
    const lines = logLines(fixture, id);
    expect(lines.filter((line) => line.kind === "item" && line.type === "tool_running").length).toBe(2);
    expect(lines.filter((line) => line.kind === "turn_interrupted").map((line) => line.reason)).toEqual(["crash"]);
    expectWholeLog(fixture, id);
    expectNoV1Sessions(fixture);
  } finally {
    gateway.stop();
    rmSync(fixture.root, { recursive: true, force: true });
  }
}, TIMEOUT * 3);

test("a torn tail is cut on resume and the session goes on", async () => {
  const fixture = createFixture("fx-v2-torn-");
  const gateway = startFakeGateway([
    fakeGatewayFinalText("BEFORE_TORN_TAIL"),
    fakeGatewayFinalText("AFTER_TORN_TAIL"),
  ]);
  try {
    const created = await ask(fixture, gateway, ["Before the torn tail."]);
    expect(created.code).toBe(0);
    const id = JSON.parse(created.stdout).session_id;
    // A write cut short by a power loss: part of a line, no newline.
    appendFileSync(join(v2Root(fixture), id, "log.jsonl"), '{"v":1,"seq":99,"ts":1,"kind":"item","ty');

    const resumed = await ask(fixture, gateway, ["--resume-id", id, "After the torn tail."]);
    expect(resumed.code).toBe(0);
    expect(resumed.stderr).toBe("");
    expect(JSON.parse(resumed.stdout).output).toBe("AFTER_TORN_TAIL");
    expect(gateway.requests.at(-1)!.body).toContain("BEFORE_TORN_TAIL");
    expect(readFileSync(join(v2Root(fixture), id, "log.jsonl"), "utf8")).not.toContain('"seq":99');
    expectWholeLog(fixture, id);
  } finally {
    gateway.stop();
    rmSync(fixture.root, { recursive: true, force: true });
  }
}, TIMEOUT * 3);

test("a flipped byte inside the log stops resume and leaves the file as it was", async () => {
  const fixture = createFixture("fx-v2-flip-");
  const gateway = startFakeGateway([
    fakeGatewayFinalText("FLIP_FIRST"),
    fakeGatewayFinalText("FLIP_SECOND"),
    fakeGatewayFinalText("FLIP_NEVER"),
  ]);
  try {
    const created = await ask(fixture, gateway, ["Flip question one."]);
    expect(created.code).toBe(0);
    const id = JSON.parse(created.stdout).session_id;
    expect((await ask(fixture, gateway, ["--resume-id", id, "Flip question two."])).code).toBe(0);
    const path = join(v2Root(fixture), id, "log.jsonl");
    const damaged = readFileSync(path, "utf8").replace("Flip question one.", "Flip question 0ne.");
    writeFileSync(path, damaged);
    const requests = gateway.requests.length;

    const resumed = await ask(fixture, gateway, ["--resume-id", id, "Flip question three."]);
    expect(resumed.code).toBe(1);
    expect(resumed.stdout + resumed.stderr).toContain("InvalidSessionFormat");
    // No request was made, and the damaged file is left for recovery.
    expect(gateway.requests.length).toBe(requests);
    expect(readFileSync(path, "utf8")).toBe(damaged);
  } finally {
    gateway.stop();
    rmSync(fixture.root, { recursive: true, force: true });
  }
}, TIMEOUT * 3);

test("a blob that went missing stops resume as damage, not as a missing session", async () => {
  const fixture = createFixture("fx-v2-lost-blob-");
  const big = "LOST_BLOB_START " + "blob-body ".repeat(30_000) + "LOST_BLOB_END";
  const gateway = startFakeGateway([fakeGatewayFinalText(big)]);
  try {
    const created = await ask(fixture, gateway, ["Answer at great length."]);
    expect(created.code).toBe(0);
    const id = JSON.parse(created.stdout).session_id;
    const referenced = (logLines(fixture, id) as any[]).find((line) => Array.isArray(line.blobs) && line.blobs.length === 1);
    rmSync(join(v2Root(fixture), id, "blobs", referenced.blobs[0]));

    const resumed = await ask(fixture, gateway, ["--resume-id", id, "Continue after the lost blob."]);
    expect(resumed.code).toBe(1);
    expect(resumed.stdout + resumed.stderr).toContain("InvalidSessionFormat");
    expect(resumed.stdout + resumed.stderr).not.toContain("NotFound");
    expect(gateway.requests).toHaveLength(1);
  } finally {
    gateway.stop();
    rmSync(fixture.root, { recursive: true, force: true });
  }
}, TIMEOUT * 2);

test("a second process on an open session is refused and writes nothing", async () => {
  const fixture = createFixture("fx-v2-busy-");
  let stalled: () => void = () => {};
  const reachedStall = new Promise<void>((resolve) => (stalled = resolve));
  const gateway = startDynamicFakeGateway(async (body) => {
    if (body.includes("After the busy session.")) return fakeGatewayFinalText("AFTER_BUSY");
    if (body.includes("Hold the session.")) {
      stalled();
      return new Promise<Response>(() => {});
    }
    return fakeGatewayFinalText("BEFORE_BUSY");
  });
  try {
    const created = await ask(fixture, gateway, ["Before the busy session."]);
    expect(created.code).toBe(0);
    const id = JSON.parse(created.stdout).session_id;

    const holder = spawnAsk(fixture, gateway, ["--resume-id", id, "Hold the session."]);
    await reachedStall;
    const second = await ask(fixture, gateway, ["--resume-id", id, "A second process."]);
    expect(second.code).toBe(1);
    expect(second.stdout + second.stderr).toContain("SessionBusy");
    expect(readFileSync(join(v2Root(fixture), id, "log.jsonl"), "utf8")).not.toContain("A second process.");
    holder.child.kill("SIGKILL");
    await holder.exited;

    const resumed = await ask(fixture, gateway, ["--resume-id", id, "After the busy session."]);
    expect(resumed.code).toBe(0);
    expect(JSON.parse(resumed.stdout).output).toBe("AFTER_BUSY");
    expectWholeLog(fixture, id);
  } finally {
    gateway.stop();
    rmSync(fixture.root, { recursive: true, force: true });
  }
}, TIMEOUT * 3);

test("a read-only session folder fails cleanly and resumes once writable", async () => {
  const fixture = createFixture("fx-v2-readonly-");
  const gateway = startFakeGateway([
    fakeGatewayFinalText("BEFORE_READ_ONLY"),
    fakeGatewayFinalText("AFTER_READ_ONLY"),
  ]);
  const folder = () => join(v2Root(fixture), id);
  let id = "";
  try {
    const created = await ask(fixture, gateway, ["Before the read-only folder."]);
    expect(created.code).toBe(0);
    id = JSON.parse(created.stdout).session_id;
    const before = readFileSync(join(folder(), "log.jsonl"));
    chmodSync(join(folder(), "log.jsonl"), 0o400);
    chmodSync(folder(), 0o500);

    const refused = await ask(fixture, gateway, ["--resume-id", id, "While read-only."]);
    expect(refused.code).toBe(1);
    // The OS cause, not a bare `Io` (D29).
    expect(JSON.parse(refused.stdout).error).toBe("AccessDenied");
    expect(gateway.requests.length).toBe(1);
    expect(readFileSync(join(folder(), "log.jsonl")).equals(before)).toBe(true);

    chmodSync(folder(), 0o700);
    chmodSync(join(folder(), "log.jsonl"), 0o600);
    const resumed = await ask(fixture, gateway, ["--resume-id", id, "After the read-only folder."]);
    expect(resumed.code).toBe(0);
    expect(JSON.parse(resumed.stdout).output).toBe("AFTER_READ_ONLY");
    expectWholeLog(fixture, id);
  } finally {
    if (id) {
      chmodSync(folder(), 0o700);
      chmodSync(join(folder(), "log.jsonl"), 0o600);
    }
    gateway.stop();
    rmSync(fixture.root, { recursive: true, force: true });
  }
}, TIMEOUT * 3);

test("a full disk fails the turn cleanly and the session resumes after", async () => {
  const fixture = createFixture("fx-v2-full-");
  const gateway = startDynamicFakeGateway(async (body) => {
    if (body.includes("After the full disk.")) return fakeGatewayFinalText("AFTER_FULL_DISK");
    if (body.includes("The disk is full.")) return fakeGatewayFinalText("X".repeat(8192));
    return fakeGatewayFinalText("BEFORE_FULL_DISK");
  });
  try {
    const created = await ask(fixture, gateway, ["Before the full disk."]);
    expect(created.code).toBe(0);
    const id = JSON.parse(created.stdout).session_id;
    const path = join(v2Root(fixture), id, "log.jsonl");
    // A file-size limit just above the log, with SIGXFSZ ignored: the next
    // growing write fails with EFBIG, as a full disk fails with ENOSPC.
    const blocks = Math.ceil(statSync(path).size / 512) + 1;
    const full = await askWithSizeLimit(fixture, gateway, blocks, ["--resume-id", id, "The disk is full."]);
    // The answer was shown, but the turn could not be saved.
    expect(full.code).toBe(1);
    expect(JSON.parse(full.stdout).error).toBe("FileTooBig");

    const resumed = await ask(fixture, gateway, ["--resume-id", id, "After the full disk."]);
    expect(resumed.code).toBe(0);
    expect(JSON.parse(resumed.stdout).output).toBe("AFTER_FULL_DISK");
    expect(gateway.requests.at(-1)!.body).toContain("BEFORE_FULL_DISK");
    expectWholeLog(fixture, id);
  } finally {
    gateway.stop();
    rmSync(fixture.root, { recursive: true, force: true });
  }
}, TIMEOUT * 3);

// ---------------------------------------------------------------------------
// The interactive app

/// Answers the newest prompt in the request, in the order given: every
/// request carries the earlier prompts, and a fresh session also asks for a
/// title. A null answer holds that request open and calls `held`.
function replyToLatest(pairs: [string, string | null][], held?: () => void, options: FakeGatewayOptions = {}) {
  return startDynamicFakeGateway(async (body) => {
    for (let index = pairs.length - 1; index >= 0; index -= 1) {
      const [prompt, answer] = pairs[index]!;
      if (!body.includes(prompt)) continue;
      if (answer !== null) return fakeGatewayFinalText(answer);
      held?.();
      return new Promise<Response>(() => {});
    }
    return fakeGatewayFinalText("UNEXPECTED_REQUEST");
  }, options);
}

async function startApp(fixture: Fixture, gateway: any, args: string[], waitForComposer = true) {
  const stderrPath = join(fixture.root, "stderr.log");
  writeFileSync(stderrPath, "");
  const session = await TmuxSession.create({
    cmd: `${FX_BIN} --sessions-v2 ${args.join(" ")}`.trim(),
    cwd: fixture.workspace,
    env: { ...env(fixture, gateway, false), NO_COLOR: "1" },
    stderrPath,
  });
  if (waitForComposer) await session.waitForComposer(TIMEOUT);
  return { session, stderrPath };
}

async function quitApp(app: { session: TmuxSession; stderrPath: string }) {
  await app.session.sendText("/quit");
  expect(await app.session.waitForSessionEnd()).toBe(true);
  await app.session.kill();
  expect(readFileSync(app.stderrPath, "utf8")).toBe("");
}

function savedSessions(fixture: Fixture): string[] {
  return readdirSync(v2Root(fixture)).filter((name) => !name.startsWith(".") && !name.startsWith("index"));
}

function onlySession(fixture: Fixture): string {
  const ids = savedSessions(fixture);
  expect(ids.length).toBe(1);
  return ids[0]!;
}

async function scrollbackContains(session: TmuxSession, marker: string) {
  const deadline = Date.now() + TIMEOUT;
  let latest = "";
  while (Date.now() < deadline) {
    latest = await session.captureFullScrollback();
    if (latest.includes(marker)) return latest;
    await Bun.sleep(100);
  }
  throw new Error(`scrollback never showed ${marker}`);
}

test.skipIf(!tmuxAvailable())("the interactive app saves to v2 and resumes with -c, --resume last and the id", async () => {
  const fixture = createFixture("fx-v2-app-");
  const gateway = replyToLatest([
    ["First interactive question.", "APP_V2_FIRST"],
    ["Continue with -c.", "APP_V2_CONTINUE"],
    ["Continue with --resume last.", "APP_V2_LAST"],
    ["Continue with the id.", "APP_V2_BY_ID"],
  ]);
  try {
    const first = await startApp(fixture, gateway, []);
    await first.session.sendText("First interactive question.");
    await first.session.waitForText("APP_V2_FIRST", TIMEOUT);
    await quitApp(first);
    const id = onlySession(fixture);
    expectNoV1Sessions(fixture);

    const runs: [string[], string, string, string][] = [
      [["-c"], "APP_V2_FIRST", "Continue with -c.", "APP_V2_CONTINUE"],
      [["--resume", "last"], "APP_V2_CONTINUE", "Continue with --resume last.", "APP_V2_LAST"],
      [["--resume", id], "APP_V2_LAST", "Continue with the id.", "APP_V2_BY_ID"],
    ];
    for (const [args, restored, prompt, answer] of runs) {
      const app = await startApp(fixture, gateway, args);
      expect(await scrollbackContains(app.session, restored)).toContain(restored);
      await app.session.sendText(prompt);
      await app.session.waitForText(answer, TIMEOUT);
      await quitApp(app);
      expect(onlySession(fixture)).toBe(id);
    }
    expect(gateway.requests.at(-1)!.body).toContain("First interactive question.");
    const kinds = logLines(fixture, id).map((line) => line.kind);
    expect(kinds.filter((kind) => kind === "turn_committed").length).toBe(4);
    expectWholeLog(fixture, id);
    expectNoV1Sessions(fixture);
  } finally {
    gateway.stop();
    rmSync(fixture.root, { recursive: true, force: true });
  }
}, TIMEOUT * 6);

test.skipIf(!tmuxAvailable())("the interactive app resumes a session with a shell result and takes the next prompt", async () => {
  const fixture = createFixture("fx-v2-app-shell-resume-");
  const gateway = startFakeGateway([
    fakeShellRun("app-shell-1", "printf 'APP_SHELL_OUTPUT_4417\\n'"),
    fakeGatewayFinalText("APP_SHELL_DONE"),
    fakeGatewayFinalText("APP_AFTER_SHELL_RESUME"),
  ]);
  try {
    const first = await startApp(fixture, gateway, []);
    await first.session.sendText("Run the shell command.");
    await scrollbackContains(first.session, "APP_SHELL_DONE");
    await first.session.waitForComposer(TIMEOUT);
    await quitApp(first);
    const id = onlySession(fixture);

    // Drawing the saved shell row reads its command replay from the side
    // folder while the history is being visited.
    const resumed = await startApp(fixture, gateway, ["-c"]);
    const shown = await scrollbackContains(resumed.session, "APP_SHELL_DONE");
    expect(shown).toContain("printf 'APP_SHELL_OUTPUT_4417");
    await resumed.session.sendText("Continue after the shell turn.");
    await resumed.session.waitForText("APP_AFTER_SHELL_RESUME", TIMEOUT);
    await quitApp(resumed);
    expect(onlySession(fixture)).toBe(id);
    expect(gateway.requests.at(-1)!.body).toContain("APP_SHELL_OUTPUT_4417");
    expectWholeLog(fixture, id);
    expectNoV1Sessions(fixture);
  } finally {
    gateway.stop();
    rmSync(fixture.root, { recursive: true, force: true });
  }
}, TIMEOUT * 4);

test.skipIf(!tmuxAvailable())("a killed interactive turn comes back interrupted and -c continues it", async () => {
  const fixture = createFixture("fx-v2-app-kill-");
  let held: () => void = () => {};
  const reachedHold = new Promise<void>((resolve) => (held = resolve));
  const gateway = replyToLatest(
    [
      ["Before the interactive kill.", "APP_BEFORE_KILL"],
      ["This interactive turn is killed.", null],
      ["After the interactive kill.", "APP_AFTER_KILL"],
    ],
    () => held(),
  );
  try {
    const app = await startApp(fixture, gateway, []);
    await app.session.sendText("Before the interactive kill.");
    await app.session.waitForText("APP_BEFORE_KILL", TIMEOUT);
    await app.session.sendText("This interactive turn is killed.");
    await reachedHold;
    Bun.spawnSync(["kill", "-9", String(app.session.processPid())]);
    await app.session.kill();
    const id = onlySession(fixture);

    const resumed = await startApp(fixture, gateway, ["-c"]);
    expect(await scrollbackContains(resumed.session, "APP_BEFORE_KILL")).toContain("APP_BEFORE_KILL");
    await resumed.session.sendText("After the interactive kill.");
    await resumed.session.waitForText("APP_AFTER_KILL", TIMEOUT);
    await quitApp(resumed);
    expect(onlySession(fixture)).toBe(id);
    // The killed prompt was saved before its request, and the model sees it.
    expect(gateway.requests.at(-1)!.body).toContain("This interactive turn is killed.");
    const lines = logLines(fixture, id);
    expect(lines.filter((line) => line.kind === "turn_interrupted").map((line) => line.reason)).toEqual(["crash"]);
    expectWholeLog(fixture, id);
    expectNoV1Sessions(fixture);
  } finally {
    gateway.stop();
    rmSync(fixture.root, { recursive: true, force: true });
  }
}, TIMEOUT * 4);

test.skipIf(!tmuxAvailable())("the picker lists v2 sessions, /rename sticks, and /new starts another", async () => {
  const fixture = createFixture("fx-v2-app-picker-");
  const gateway = replyToLatest([
    ["Picker session one.", "PICK_ONE"],
    ["Picker session two.", "PICK_TWO"],
    ["Back in session one.", "PICK_BACK"],
  ]);
  try {
    const app = await startApp(fixture, gateway, []);
    await app.session.sendText("Picker session one.");
    await app.session.waitForText("PICK_ONE", TIMEOUT);
    await app.session.sendText("/rename Renamed picker session");
    await app.session.waitForText('renamed to "Renamed picker session"', TIMEOUT);
    await app.session.waitForStableComposer();
    await app.session.sendText("/new");
    await app.session.waitForStableComposer();
    await app.session.sendText("Picker session two.");
    await app.session.waitForText("PICK_TWO", TIMEOUT);
    await quitApp(app);
    const ids = savedSessions(fixture);
    expect(ids.length).toBe(2);

    const picker = await startApp(fixture, gateway, ["-r"], false);
    await picker.session.waitForPane((pane) => pane.includes("Renamed picker session") && pane.includes("enter resume"), TIMEOUT);
    await picker.session.sendLiteralText("Renamed");
    await picker.session.waitForPane((pane) => pane.includes("Renamed picker session"), TIMEOUT);
    await picker.session.sendKeys("Enter");
    await picker.session.waitForComposer(TIMEOUT);
    expect(await scrollbackContains(picker.session, "PICK_ONE")).toContain("PICK_ONE");
    await picker.session.sendText("Back in session one.");
    await picker.session.waitForText("PICK_BACK", TIMEOUT);
    await quitApp(picker);

    // The resumed session is the renamed one: its log holds both of its turns.
    const renamed = ids.find((id) => readFileSync(join(v2Root(fixture), id, "log.jsonl"), "utf8").includes("Renamed picker session"))!;
    const log = readFileSync(join(v2Root(fixture), renamed, "log.jsonl"), "utf8");
    expect(log).toContain("Back in session one.");
    expect(log).not.toContain("Picker session two.");
    for (const id of ids) expectWholeLog(fixture, id);
    expectNoV1Sessions(fixture);
  } finally {
    gateway.stop();
    rmSync(fixture.root, { recursive: true, force: true });
  }
}, TIMEOUT * 6);

test.skipIf(!tmuxAvailable())("the picker shows a session open in another fx as busy at once, and opens it once the owner quits", async () => {
  const fixture = createFixture("fx-v2-app-picker-busy-");
  const gateway = replyToLatest([["Hold this session open.", "PICKER_BUSY_SAVED"]]);
  const busy = "This session is open in another fx. Close it there, then press enter to retry.";
  let contender: TmuxSession | null = null;
  try {
    const owner = await startApp(fixture, gateway, []);
    await owner.session.sendText("Hold this session open.");
    await owner.session.waitForText("PICKER_BUSY_SAVED", TIMEOUT);
    const id = onlySession(fixture);

    const contenderStderr = join(fixture.root, "contender-stderr.log");
    writeFileSync(contenderStderr, "");
    contender = await TmuxSession.create({
      cmd: `${FX_BIN} --sessions-v2 -r`,
      cwd: fixture.workspace,
      env: { ...env(fixture, gateway, false), NO_COLOR: "1" },
      stderrPath: contenderStderr,
    });
    await contender.waitForPane((pane) => pane.includes("Hold this session open.") && pane.includes("enter resume"), TIMEOUT);
    // v1's picker takes no lock wait, and neither does v2's (D38).
    const pressed = Date.now();
    await contender.sendKeys("Enter");
    await contender.waitForPane((pane) => pane.includes(busy), 1_000);
    expect(Date.now() - pressed).toBeLessThan(1_000);
    expect(owner.session.isPaneAlive()).toBe(true);

    await quitApp(owner);
    await contender.sendKeys("Enter");
    await contender.waitForComposer(TIMEOUT);
    expect(await scrollbackContains(contender, "PICKER_BUSY_SAVED")).toContain("PICKER_BUSY_SAVED");
    await contender.sendText("/quit");
    expect(await contender.waitForSessionEnd()).toBe(true);
    expect(readFileSync(contenderStderr, "utf8")).toBe("");
    expect(onlySession(fixture)).toBe(id);
    expectWholeLog(fixture, id);
    expectNoV1Sessions(fixture);
  } finally {
    if (contender) await contender.kill();
    gateway.stop();
    rmSync(fixture.root, { recursive: true, force: true });
  }
}, TIMEOUT * 4);

// ---------------------------------------------------------------------------
// ACP

/// A minimal ACP client: requests by id, every `session/update` kept,
/// permission requests allowed once.
class AcpRpc {
  private proc: ReturnType<typeof spawn>;
  private buffer = "";
  private nextId = 1;
  private pending = new Map<number, (msg: any) => void>();
  readonly updates: any[] = [];
  readonly exited: Promise<number | null>;

  constructor(fixture: Fixture, gateway: any, extraEnv: Record<string, string | undefined> = {}) {
    this.proc = spawn(FX_BIN, ["acp"], {
      cwd: fixture.workspace,
      env: { ...process.env, ...env(fixture, gateway), ...extraEnv } as Record<string, string>,
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.exited = new Promise((resolve) => this.proc.on("exit", (code) => resolve(code)));
    this.proc.stdout!.on("data", (chunk: Buffer) => {
      this.buffer += chunk.toString();
      let newline: number;
      while ((newline = this.buffer.indexOf("\n")) >= 0) {
        const line = this.buffer.slice(0, newline);
        this.buffer = this.buffer.slice(newline + 1);
        if (line.trim()) this.onMessage(JSON.parse(line));
      }
    });
  }

  static async start(fixture: Fixture, gateway: any, extraEnv: Record<string, string | undefined> = {}) {
    const client = new AcpRpc(fixture, gateway, extraEnv);
    const initialized = await client.request("initialize", { protocolVersion: 1, clientCapabilities: {} });
    if (initialized.error) throw new Error(JSON.stringify(initialized.error));
    return client;
  }

  private onMessage(msg: any) {
    if (msg.method === "session/update") {
      this.updates.push(msg.params);
    } else if (msg.method !== undefined && msg.id !== undefined) {
      const result = msg.method === "session/request_permission" ? { outcome: { outcome: "selected", optionId: "allow_once" } } : {};
      this.proc.stdin!.write(JSON.stringify({ jsonrpc: "2.0", id: msg.id, result }) + "\n");
    } else if (msg.id !== undefined && this.pending.has(msg.id)) {
      this.pending.get(msg.id)!(msg);
      this.pending.delete(msg.id);
    }
  }

  /// Resolves with the whole response, `error` included.
  request(method: string, params: object, timeoutMs = 20_000): Promise<any> {
    const id = this.nextId++;
    this.proc.stdin!.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n");
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`${method} timed out`)), timeoutMs);
      this.pending.set(id, (msg) => {
        clearTimeout(timer);
        resolve(msg);
      });
    });
  }

  async ok(method: string, params: object) {
    const response = await this.request(method, params);
    if (response.error) throw new Error(`${method}: ${JSON.stringify(response.error)}`);
    return response.result;
  }

  /// Text of the updates of one kind, in order.
  texts(kind: "user_message_chunk" | "agent_message_chunk") {
    return this.updates.filter((u) => u.update?.sessionUpdate === kind).map((u) => u.update.content?.text ?? "");
  }

  async close() {
    this.proc.stdin!.end();
    const exited = await Promise.race([this.exited, Bun.sleep(10_000).then(() => "timeout")]);
    if (exited === "timeout") this.proc.kill("SIGKILL");
    return exited;
  }

  kill() {
    this.proc.kill("SIGKILL");
    return this.exited;
  }
}

function acpPrompt(text: string) {
  return { prompt: [{ type: "text", text }] };
}

test("ACP keeps a session once prompted, lists it, and loads every turn after a restart", async () => {
  const fixture = createFixture("fx-v2-acp-");
  const gateway = replyToLatest([
    ["First ACP question.", "ACP_ONE"],
    ["Second ACP question.", "ACP_TWO"],
    ["Third ACP question.", "ACP_THREE"],
  ]);
  let client: AcpRpc | undefined;
  try {
    client = await AcpRpc.start(fixture, gateway);
    // Never prompted, so never written (D24).
    const unused = (await client.ok("session/new", { cwd: fixture.workspace, mcpServers: [] })).sessionId;
    const id = (await client.ok("session/new", { cwd: fixture.workspace, mcpServers: [] })).sessionId;
    expect(id).not.toBe(unused);
    expect((await client.ok("session/prompt", { sessionId: id, ...acpPrompt("First ACP question.") })).stopReason).toBe("end_turn");
    expect((await client.ok("session/prompt", { sessionId: id, ...acpPrompt("Second ACP question.") })).stopReason).toBe("end_turn");
    const listed = await client.ok("session/list", { cwd: fixture.workspace });
    expect(listed.sessions.map((s: any) => s.sessionId)).toEqual([id]);
    expect(listed.sessions[0].cwd).toBe(fixture.workspace);
    expect(Date.parse(listed.sessions[0].updatedAt)).toBeGreaterThan(0);
    expect((await client.ok("session/list", { cwd: join(fixture.root, "elsewhere") })).sessions).toEqual([]);
    // v1 matches the folder after trailing slashes too.
    expect((await client.ok("session/list", { cwd: `${fixture.workspace}/` })).sessions.map((s: any) => s.sessionId)).toEqual([id]);
    expect(await client.close()).toBe(0);
    expect(existsSync(join(v2Root(fixture), unused))).toBe(false);

    client = await AcpRpc.start(fixture, gateway);
    const missing = await client.request("session/load", { sessionId: unused, cwd: fixture.workspace, mcpServers: [] });
    expect(missing.error?.message).toBe("Session not found");
    await client.ok("session/load", { sessionId: id, cwd: fixture.workspace, mcpServers: [] });
    expect(client.texts("user_message_chunk")).toEqual(["First ACP question.", "Second ACP question."]);
    expect(client.texts("agent_message_chunk")).toEqual(["ACP_ONE", "ACP_TWO"]);
    // Resume attaches without replaying.
    const replayed = client.updates.length;
    await client.ok("session/resume", { sessionId: id, cwd: fixture.workspace, mcpServers: [] });
    expect(client.updates.slice(replayed).filter((u) => u.update?.sessionUpdate === "user_message_chunk")).toEqual([]);
    await client.ok("session/prompt", { sessionId: id, ...acpPrompt("Third ACP question.") });
    const body = gateway.requests.at(-1)!.body;
    expect(body).toContain("ACP_ONE");
    expect(body).toContain("ACP_TWO");
    expect(await client.close()).toBe(0);
    client = undefined;

    const lines = logLines(fixture, id);
    expect(lines.filter((line) => line.kind === "turn_committed").length).toBe(3);
    expectWholeLog(fixture, id);
    expectNoV1Sessions(fixture);
  } finally {
    if (client) await client.kill();
    gateway.stop();
    rmSync(fixture.root, { recursive: true, force: true });
  }
}, TIMEOUT * 4);

test("ACP killed in the middle of a prompt loads with that turn interrupted", async () => {
  const fixture = createFixture("fx-v2-acp-kill-");
  let held: () => void = () => {};
  const holding = new Promise<void>((resolve) => (held = resolve));
  const gateway = replyToLatest(
    [
      ["Before the ACP kill.", "ACP_BEFORE"],
      ["Held ACP question.", null],
      ["After the ACP kill.", "ACP_AFTER"],
    ],
    () => held(),
  );
  let client: AcpRpc | undefined;
  try {
    client = await AcpRpc.start(fixture, gateway);
    const id = (await client.ok("session/new", { cwd: fixture.workspace, mcpServers: [] })).sessionId;
    await client.ok("session/prompt", { sessionId: id, ...acpPrompt("Before the ACP kill.") });
    void client.request("session/prompt", { sessionId: id, ...acpPrompt("Held ACP question.") }).catch(() => {});
    await holding;
    await waitForLog(fixture, id, "Held ACP question.");
    await client.kill();

    client = await AcpRpc.start(fixture, gateway);
    await client.ok("session/load", { sessionId: id, cwd: fixture.workspace, mcpServers: [] });
    expect(client.texts("user_message_chunk")).toEqual(["Before the ACP kill.", "Held ACP question."]);
    await client.ok("session/prompt", { sessionId: id, ...acpPrompt("After the ACP kill.") });
    expect(gateway.requests.at(-1)!.body).toContain("ACP_BEFORE");
    expect(await client.close()).toBe(0);
    client = undefined;
    const lines = logLines(fixture, id);
    expect(lines.filter((line) => line.kind === "turn_interrupted").map((line) => line.reason)).toEqual(["crash"]);
    expectWholeLog(fixture, id);
  } finally {
    if (client) await client.kill();
    gateway.stop();
    rmSync(fixture.root, { recursive: true, force: true });
  }
}, TIMEOUT * 3);

test("ACP saves a model change with the session and loads it back", async () => {
  const fixture = createFixture("fx-v2-acp-prefs-");
  const gateway = replyToLatest([["Pick a model.", "ACP_MODEL"]]);
  // A process-wide model would win over the saved one on load, as on v1.
  const noModel = { FX_MODEL: undefined };
  let client: AcpRpc | undefined;
  try {
    client = await AcpRpc.start(fixture, gateway, noModel);
    const created = await client.ok("session/new", { cwd: fixture.workspace, mcpServers: [] });
    const id = created.sessionId;
    const current = created.configOptions.find((option: any) => option.id === "model").currentValue;
    const other = "openai/gpt-5-mini";
    expect(other).not.toBe(current);
    const changed = await client.ok("session/set_config_option", { sessionId: id, configId: "model", value: other });
    expect(changed.configOptions.find((option: any) => option.id === "model").currentValue).toBe(other);
    await client.ok("session/prompt", { sessionId: id, ...acpPrompt("Pick a model.") });
    expect(gateway.requests.at(-1)!.headers.get("ai-language-model-id")).toBe(other);
    expect(await client.close()).toBe(0);

    client = await AcpRpc.start(fixture, gateway, noModel);
    const loaded = await client.ok("session/load", { sessionId: id, cwd: fixture.workspace, mcpServers: [] });
    expect(loaded.configOptions.find((option: any) => option.id === "model").currentValue).toBe(other);
    expect(await client.close()).toBe(0);
    client = undefined;
    expect(logLines(fixture, id).filter((line) => line.kind === "set" && line.key === "prefs").length).toBeGreaterThan(1);
  } finally {
    if (client) await client.kill();
    gateway.stop();
    rmSync(fixture.root, { recursive: true, force: true });
  }
}, TIMEOUT * 3);

test("ACP keeps a tool result as a side file, and load replays the call with its result", async () => {
  const fixture = createFixture("fx-v2-acp-tool-");
  const gateway = startDynamicFakeGateway(async (body) => {
    if (body.includes("After the ACP tool.")) return fakeGatewayFinalText("ACP_AFTER_TOOL");
    if (body.includes("Run an ACP tool.") && body.includes("ACP_TOOL_OUTPUT_77")) return fakeGatewayFinalText("ACP_TOOL_DONE");
    if (body.includes("Run an ACP tool.")) return fakeShellRun("acp-tool-1", "echo ACP_TOOL_OUTPUT_77");
    return fakeGatewayFinalText("ACP_OTHER");
  });
  let client: AcpRpc | undefined;
  try {
    client = await AcpRpc.start(fixture, gateway);
    const id = (await client.ok("session/new", { cwd: fixture.workspace, mcpServers: [] })).sessionId;
    await client.ok("session/prompt", { sessionId: id, ...acpPrompt("Run an ACP tool.") });
    // session/close ends the session; it stays saved.
    await client.ok("session/close", { sessionId: id });
    expect(await client.close()).toBe(0);
    expect(readdirSync(join(fixture.home, ".fx", "session-files", id)).length).toBeGreaterThan(0);

    client = await AcpRpc.start(fixture, gateway);
    await client.ok("session/load", { sessionId: id, cwd: fixture.workspace, mcpServers: [] });
    const calls = client.updates.filter((u) => u.update?.sessionUpdate === "tool_call" && u.update.toolCallId === "acp-tool-1");
    expect(calls.length).toBe(1);
    const results = client.updates.filter((u) => u.update?.sessionUpdate === "tool_call_update" && u.update.toolCallId === "acp-tool-1");
    // The replay sends the stored preview, as on v1; the model gets the
    // whole output back from the side file (below).
    expect(results.length).toBe(1);
    expect(results[0].update.status).toBe("completed");
    expect(results[0].update.content[0].content.text.length).toBeGreaterThan(0);
    expect(client.texts("agent_message_chunk")).toEqual(["ACP_TOOL_DONE"]);
    await client.ok("session/prompt", { sessionId: id, ...acpPrompt("After the ACP tool.") });
    expect(gateway.requests.at(-1)!.body).toContain("ACP_TOOL_OUTPUT_77");
    expect(await client.close()).toBe(0);
    client = undefined;
    expectWholeLog(fixture, id);
    expectNoV1Sessions(fixture);
  } finally {
    if (client) await client.kill();
    gateway.stop();
    rmSync(fixture.root, { recursive: true, force: true });
  }
}, TIMEOUT * 3);

test("ACP lists more than a page of sessions with a cursor, newest first, each once", async () => {
  const fixture = createFixture("fx-v2-acp-page-");
  const gateway = startDynamicFakeGateway(async () => fakeGatewayFinalText("ACP_PAGE"));
  let client: AcpRpc | undefined;
  try {
    client = await AcpRpc.start(fixture, gateway);
    const created: string[] = [];
    for (let index = 0; index < 101; index += 1) {
      const id = (await client.ok("session/new", { cwd: fixture.workspace, mcpServers: [] })).sessionId;
      await client.ok("session/prompt", { sessionId: id, ...acpPrompt(`Page session ${index}.`) });
      created.push(id);
    }
    const first = await client.ok("session/list", {});
    expect(first.sessions.length).toBe(100);
    expect(typeof first.nextCursor).toBe("string");
    const second = await client.ok("session/list", { cursor: first.nextCursor });
    expect(second.nextCursor).toBeUndefined();
    const listed = [...first.sessions, ...second.sessions];
    expect(listed.map((s: any) => s.sessionId).sort()).toEqual([...created].sort());
    const times = listed.map((s: any) => Date.parse(s.updatedAt));
    for (let index = 1; index < times.length; index += 1) expect(times[index - 1]).toBeGreaterThanOrEqual(times[index]);
    const bad = await client.request("session/list", { cursor: "not-a-cursor" });
    expect(bad.error?.message).toBe("Invalid params");
    expect(await client.close()).toBe(0);
    client = undefined;
  } finally {
    if (client) await client.kill();
    gateway.stop();
    rmSync(fixture.root, { recursive: true, force: true });
  }
}, TIMEOUT * 4);

/// A catalog whose model can see images, so fx sends them to it as they are.
const VISION_MODEL: FakeGatewayOptions = {
  models: [{ id: FAKE_GATEWAY_MODEL, type: "language", tags: ["vision", "file-input", "tool-use"] }],
};
const PNG_1X1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

function imageFiles(fixture: Fixture, id: string) {
  const dir = join(fixture.home, ".fx", "session-files", id, "images");
  return existsSync(dir) ? readdirSync(dir) : [];
}

test("fx ask keeps an image of a new session with its side files, and resume sends it again", async () => {
  const fixture = createFixture("fx-v2-image-");
  const gateway = replyToLatest([
    ["Describe the image.", "IMAGE_SEEN"],
    ["And again.", "IMAGE_AGAIN"],
  ], undefined, VISION_MODEL);
  try {
    const image = join(fixture.workspace, "dot.png");
    writeFileSync(image, Buffer.from(PNG_1X1, "base64"));
    const created = await ask(fixture, gateway, ["--image", image, "Describe the image."]);
    expect(created.code).toBe(0);
    expect(JSON.parse(created.stdout).output).toBe("IMAGE_SEEN");
    const id = JSON.parse(created.stdout).session_id;
    expect(imageFiles(fixture, id).length).toBe(1);
    const resumed = await ask(fixture, gateway, ["--resume-id", id, "And again."]);
    expect(resumed.code).toBe(0);
    // The prompt text says "image" too, so match the part's media type.
    expect(gateway.requests[0]!.body).toContain("image/png");
    expect(gateway.requests.at(-1)!.body).toContain("image/png");
    expectWholeLog(fixture, id);
    expectNoV1Sessions(fixture);
  } finally {
    gateway.stop();
    rmSync(fixture.root, { recursive: true, force: true });
  }
}, TIMEOUT * 2);

test("ACP keeps an image prompt and replays it with the image on load", async () => {
  const fixture = createFixture("fx-v2-acp-image-");
  const gateway = replyToLatest([["Describe this ACP image.", "ACP_IMAGE_SEEN"]], undefined, VISION_MODEL);
  let client: AcpRpc | undefined;
  try {
    client = await AcpRpc.start(fixture, gateway);
    const id = (await client.ok("session/new", { cwd: fixture.workspace, mcpServers: [] })).sessionId;
    const prompted = await client.ok("session/prompt", {
      sessionId: id,
      prompt: [
        { type: "text", text: "Describe this ACP image." },
        { type: "image", data: PNG_1X1, mimeType: "image/png" },
      ],
    });
    expect(prompted.stopReason).toBe("end_turn");
    expect(imageFiles(fixture, id).length).toBe(1);
    expect(await client.close()).toBe(0);

    client = await AcpRpc.start(fixture, gateway);
    await client.ok("session/load", { sessionId: id, cwd: fixture.workspace, mcpServers: [] });
    const images = client.updates.filter((u) => u.update?.sessionUpdate === "user_message_chunk" && u.update.content?.type === "image");
    expect(images.length).toBe(1);
    expect(await client.close()).toBe(0);
    client = undefined;
    expectWholeLog(fixture, id);
  } finally {
    if (client) await client.kill();
    gateway.stop();
    rmSync(fixture.root, { recursive: true, force: true });
  }
}, TIMEOUT * 3);

test("a second ACP process is refused an open v2 session, then loads it once the owner exits", async () => {
  const fixture = createFixture("fx-v2-acp-busy-");
  const gateway = replyToLatest([["Hold this session.", "ACP_OWNER"]]);
  let owner: AcpRpc | undefined;
  let other: AcpRpc | undefined;
  try {
    owner = await AcpRpc.start(fixture, gateway);
    const id = (await owner.ok("session/new", { cwd: fixture.workspace, mcpServers: [] })).sessionId;
    await owner.ok("session/prompt", { sessionId: id, ...acpPrompt("Hold this session.") });
    other = await AcpRpc.start(fixture, gateway);
    const refused = await other.request("session/load", { sessionId: id, cwd: fixture.workspace, mcpServers: [] });
    expect(refused.error?.message).toBe("Session is busy");
    expect(await owner.close()).toBe(0);
    owner = undefined;
    await other.ok("session/load", { sessionId: id, cwd: fixture.workspace, mcpServers: [] });
    expect(other.texts("agent_message_chunk")).toEqual(["ACP_OWNER"]);
    expect(await other.close()).toBe(0);
    other = undefined;
    expectWholeLog(fixture, id);
  } finally {
    if (owner) await owner.kill();
    if (other) await other.kill();
    gateway.stop();
    rmSync(fixture.root, { recursive: true, force: true });
  }
}, TIMEOUT * 3);

test("ACP loads a session whose saved image was deleted, and says the image is unavailable", async () => {
  const fixture = createFixture("fx-v2-acp-image-gone-");
  const gateway = replyToLatest([["Save this image.", "IMAGE_SAVED"]], undefined, VISION_MODEL);
  let client: AcpRpc | undefined;
  try {
    client = await AcpRpc.start(fixture, gateway);
    const id = (await client.ok("session/new", { cwd: fixture.workspace, mcpServers: [] })).sessionId;
    await client.ok("session/prompt", {
      sessionId: id,
      prompt: [
        { type: "text", text: "Save this image." },
        { type: "image", data: PNG_1X1, mimeType: "image/png" },
      ],
    });
    expect(await client.close()).toBe(0);
    const images = imageFiles(fixture, id);
    expect(images.length).toBe(1);
    rmSync(join(fixture.home, ".fx", "session-files", id, "images", images[0]!));

    client = await AcpRpc.start(fixture, gateway);
    await client.ok("session/load", { sessionId: id, cwd: fixture.workspace, mcpServers: [] });
    const userTexts = client.updates
      .filter((u) => u.update?.sessionUpdate === "user_message_chunk" && u.update.content?.type === "text")
      .map((u) => u.update.content.text);
    expect(userTexts).toEqual(["Save this image.\n[Image #1]", "Image #1 unavailable"]);
    expect(client.updates.some((u) => u.update?.content?.type === "image")).toBe(false);
    expect(await client.close()).toBe(0);
    client = undefined;
    expectWholeLog(fixture, id);
  } finally {
    if (client) await client.kill();
    gateway.stop();
    rmSync(fixture.root, { recursive: true, force: true });
  }
}, TIMEOUT * 3);

function infoTitles(updates: any[]) {
  return updates.filter((u) => u.update?.sessionUpdate === "session_info_update").map((u) => u.update.title);
}

test("ACP keeps a generated title, and list and load show it after a restart", async () => {
  const fixture = createFixture("fx-v2-acp-title-");
  const gateway = startDynamicFakeGateway(() => fakeGatewayFinalText("ACP_TITLED_ANSWER"), {
    titleResponses: [fakeGatewayFinalText("ACP Generated Title")],
  });
  let client: AcpRpc | undefined;
  try {
    client = await AcpRpc.start(fixture, gateway);
    const id = (await client.ok("session/new", { cwd: fixture.workspace, mcpServers: [] })).sessionId;
    await client.ok("session/prompt", { sessionId: id, ...acpPrompt("Name this conversation for me.") });
    expect(infoTitles(client.updates)).toContain("ACP Generated Title");
    expect(await client.close()).toBe(0);

    client = await AcpRpc.start(fixture, gateway);
    const listed = await client.ok("session/list", { cwd: fixture.workspace });
    expect(listed.sessions.map((s: any) => [s.sessionId, s.title])).toEqual([[id, "ACP Generated Title"]]);
    await client.ok("session/load", { sessionId: id, cwd: fixture.workspace, mcpServers: [] });
    expect(infoTitles(client.updates)).toEqual(["ACP Generated Title"]);
    expect(await client.close()).toBe(0);
    client = undefined;
    expectWholeLog(fixture, id);
  } finally {
    if (client) await client.kill();
    gateway.stop();
    rmSync(fixture.root, { recursive: true, force: true });
  }
}, TIMEOUT * 3);

test("ACP lists by workspace: each cwd sees its own sessions, and no cwd sees all", async () => {
  const fixture = createFixture("fx-v2-acp-cwd-");
  const other = join(fixture.root, "other-workspace");
  mkdirSync(other);
  const otherRoot = realpathSync(other);
  const gateway = replyToLatest([
    ["First workspace prompt.", "FIRST_WORKSPACE"],
    ["Second workspace prompt.", "SECOND_WORKSPACE"],
  ]);
  let client: AcpRpc | undefined;
  try {
    // A session belongs to the workspace its ACP process runs in, as in v1.
    client = await AcpRpc.start(fixture, gateway);
    const first = (await client.ok("session/new", { cwd: fixture.workspace, mcpServers: [] })).sessionId;
    await client.ok("session/prompt", { sessionId: first, ...acpPrompt("First workspace prompt.") });
    expect(await client.close()).toBe(0);
    client = await AcpRpc.start({ ...fixture, workspace: otherRoot }, gateway);
    const second = (await client.ok("session/new", { cwd: otherRoot, mcpServers: [] })).sessionId;
    await client.ok("session/prompt", { sessionId: second, ...acpPrompt("Second workspace prompt.") });
    expect(await client.close()).toBe(0);

    client = await AcpRpc.start(fixture, gateway);
    const ids = async (params: object) => (await client!.ok("session/list", params)).sessions.map((s: any) => [s.sessionId, s.cwd]);
    expect(await ids({ cwd: fixture.workspace })).toEqual([[first, fixture.workspace]]);
    expect(await ids({ cwd: `${fixture.workspace}/` })).toEqual([[first, fixture.workspace]]);
    expect(await ids({ cwd: otherRoot })).toEqual([[second, otherRoot]]);
    expect(await ids({ cwd: join(fixture.root, "no-sessions-here") })).toEqual([]);
    expect((await ids({})).sort()).toEqual([[first, fixture.workspace], [second, otherRoot]].sort());
    expect(await client.close()).toBe(0);
    client = undefined;
  } finally {
    if (client) await client.kill();
    gateway.stop();
    rmSync(fixture.root, { recursive: true, force: true });
  }
}, TIMEOUT * 3);

test("ACP load takes a special-token id literally and never loads the newest session", async () => {
  const fixture = createFixture("fx-v2-acp-literal-id-");
  const gateway = replyToLatest([["The only saved prompt.", "ONLY_SAVED_ANSWER"]]);
  let client: AcpRpc | undefined;
  try {
    client = await AcpRpc.start(fixture, gateway);
    const id = (await client.ok("session/new", { cwd: fixture.workspace, mcpServers: [] })).sessionId;
    await client.ok("session/prompt", { sessionId: id, ...acpPrompt("The only saved prompt.") });
    expect(await client.close()).toBe(0);
    const saved = readFileSync(join(v2Root(fixture), id, "log.jsonl"));

    client = await AcpRpc.start(fixture, gateway);
    for (const token of ["last", "last_opened", ".", "..", `../${id}`]) {
      const response = await client.request("session/load", { sessionId: token, cwd: fixture.workspace, mcpServers: [] });
      expect(response.error, token).toBeDefined();
    }
    expect(client.updates).toEqual([]);
    expect(await client.close()).toBe(0);
    client = undefined;
    expect(readFileSync(join(v2Root(fixture), id, "log.jsonl"))).toEqual(saved);
  } finally {
    if (client) await client.kill();
    gateway.stop();
    rmSync(fixture.root, { recursive: true, force: true });
  }
}, TIMEOUT * 2);

test.skipIf(!tmuxAvailable())("ACP load after a compaction shows every earlier turn and never the handoff", async () => {
  const fixture = createFixture("fx-v2-acp-compacted-");
  const gateway = startFakeGateway([
    fakeShellRun("saved-history-effect", "printf 'V2_SAVED_TOOL_OUTPUT\\n' >> replay-effects.txt; printf 'V2_SAVED_TOOL_OUTPUT\\n'"),
    fakeGatewayFinalText("V2_EARLIER_VISIBLE"),
    fakeGatewayFinalText("V2_MIDDLE_VISIBLE"),
    fakeGatewayFinalText("V2_LATEST_VISIBLE"),
    fakeGatewayFinalText("V2_INTERNAL_HANDOFF: continue the task."),
  ]);
  let client: AcpRpc | undefined;
  try {
    const app = await startApp(fixture, gateway, []);
    for (const [prompt, answer] of [
      ["Earlier v2 request", "V2_EARLIER_VISIBLE"],
      ["Middle v2 request", "V2_MIDDLE_VISIBLE"],
      ["Latest v2 request", "V2_LATEST_VISIBLE"],
    ]) {
      await app.session.sendText(prompt!);
      await scrollbackContains(app.session, answer!);
      await app.session.waitForComposer(TIMEOUT);
    }
    const id = onlySession(fixture);
    await app.session.sendText("/compact");
    await waitForLog(fixture, id, "V2_INTERNAL_HANDOFF", TIMEOUT);
    await app.session.waitForComposer(TIMEOUT);
    await quitApp(app);
    const saved = readFileSync(join(v2Root(fixture), id, "log.jsonl"));
    expect(gateway.requests).toHaveLength(5);

    client = await AcpRpc.start(fixture, gateway);
    for (let load = 0; load < 2; load += 1) {
      client.updates.length = 0;
      await client.ok("session/load", { sessionId: id, cwd: fixture.workspace, mcpServers: [] });
      const visible = JSON.stringify(client.updates);
      expect(infoTitles(client.updates)).toEqual(["Earlier v2 request"]);
      expect(client.texts("user_message_chunk")).toEqual(["Earlier v2 request", "Middle v2 request", "Latest v2 request"]);
      expect(visible).toContain("V2_SAVED_TOOL_OUTPUT");
      expect(visible).not.toContain("V2_INTERNAL_HANDOFF");
      const order = ["V2_EARLIER_VISIBLE", "V2_MIDDLE_VISIBLE", "V2_LATEST_VISIBLE"].map((text) => visible.indexOf(text));
      expect(order.every((at) => at >= 0)).toBe(true);
      expect(order).toEqual([...order].sort((a, b) => a - b));
    }
    expect(await client.close()).toBe(0);
    client = undefined;
    const after = readFileSync(join(v2Root(fixture), id, "log.jsonl"));
    // Load adds no history: its writable open ends with the clean-exit
    // marker every close appends.
    expect(after.subarray(0, saved.length)).toEqual(saved);
    const appended = after.subarray(saved.length).toString("utf8").split("\n").filter(Boolean);
    expect(appended.map((line) => JSON.parse(line).kind)).toEqual(["closed"]);
    expect(readFileSync(join(fixture.workspace, "replay-effects.txt"), "utf8")).toBe("V2_SAVED_TOOL_OUTPUT\n");
    expect(gateway.requests).toHaveLength(5);
    expectWholeLog(fixture, id);
  } finally {
    if (client) await client.kill();
    gateway.stop();
    rmSync(fixture.root, { recursive: true, force: true });
  }
}, TIMEOUT * 4);

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
  fakeGatewayFinalText,
  fakeShellRun,
  startDynamicFakeGateway,
  startFakeGateway,
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
function expectPairedToolCalls(body: string) {
  const input: any[] = JSON.parse(body).input ?? [];
  const calls = input.filter((item) => item.type === "function_call").map((item) => item.call_id);
  const results = new Set(input.filter((item) => item.type === "function_call_output").map((item) => item.call_id));
  for (const call of calls) expect(results.has(call)).toBe(true);
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

test("a kill while a tool runs keeps the finished tools and pairs every call", async () => {
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
    await Bun.sleep(500);
    run.child.kill("SIGKILL");
    await run.exited;

    const resumed = await ask(fixture, gateway, ["--resume-id", id, "After the tool kill."]);
    expect(resumed.code).toBe(0);
    expect(resumed.stderr).toBe("");
    expect(JSON.parse(resumed.stdout).output).toBe("AFTER_TOOL_KILL");
    const body = gateway.requests.at(-1)!.body;
    expect(body).toContain("BEFORE_TOOL_KILL");
    // The finished tool survives the crash; the running one is not replayed
    // unpaired.
    expect(body).toContain("FIRST_TOOL_OUTPUT_5521");
    expectPairedToolCalls(body);
    const lines = logLines(fixture, id);
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
    expect(JSON.parse(refused.stdout).error).toBe("Io");
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
    expect(JSON.parse(full.stdout).error).toBe("Io");

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

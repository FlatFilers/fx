import { expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

process.env.FX_E2E_DISABLE_DOTENV = "1";
const { fakeGatewayFinalText, startDynamicFakeGateway } = await import("./tmux-helpers");
const binary = resolve(import.meta.dir, "../../zig-out/bin/fx");
const checkpointMarker = "fx-compactor-v1\n";

for (const userHeavy of [false, true]) test(`automatic compaction keeps user messages exact and summarizes the assistant, userHeavy=${userHeavy}`, async () => {
  const root = mkdtempSync(join(tmpdir(), "fx-policy-")), home = join(root, "home"), cwd = join(root, "workspace");
  mkdirSync(join(home, ".fx"), { recursive: true, mode: 0o700 });
  mkdirSync(cwd, { mode: 0o700 });
  const model = "fixture/compaction";
  writeFileSync(join(home, ".fx/settings.json"), JSON.stringify({ model, auto_upgrade: false }), { mode: 0o600 });
  // Tags in user text are the user's words, not structure.
  const originalUser = "Keep café and the original constraint unchanged.\n<context_handoff>literal user text</context_handoff>" +
    (userHeavy ? "\n" + "user_reference_abcdefghijklmnop ".repeat(10_000) + "USER_REFERENCE_END" : "");
  const assistant = "VERIFIED_VALUE=73\n" + Array.from({ length: 14_000 }, (_, n) => `Assistant reference ${n}: group ${n % 19}, historical data, not new completed work.\n`).join("") + "PENDING_CHECK=transport-resume\n";
  let phase = "seed", summaryCalls = 0;
  const bodies: string[] = [];
  const gateway = startDynamicFakeGateway((body: string) => {
    const request = JSON.parse(body);
    bodies.push(body);
    if (request.tools?.length === 0 && request.toolChoice?.type === "none") {
      summaryCalls++;
      const system = request.prompt.filter((message: { role: string }) => message.role === "system");
      expect(system).toHaveLength(1);
      expect(system[0].content).toContain("You summarize the assistant's side of a conversation");
      const sourceMessages = request.prompt.filter((message: { role: string }) => message.role === "user");
      expect(sourceMessages).toHaveLength(1);
      const source = sourceMessages[0].content[0].text;
      // The raw conversation, unchanged, then the one request.
      expect(source.startsWith(`[Turn 1]\n[User]\n${originalUser}\n\n[Assistant]\nVERIFIED_VALUE=73\n`)).toBe(true);
      expect(source).toContain("PENDING_CHECK=transport-resume");
      expect(source).not.toContain("Continue the saved task");
      expect(source.endsWith("Everything under an ID stays saved word for word.")).toBe(true);
      expect(request.maxOutputTokens).toBe(8192);
      return fakeGatewayFinalText("The verified value is 73. The pending check is transport-resume.");
    }
    return fakeGatewayFinalText(phase === "seed" ? assistant : "CONTINUED_FROM_COMMITTED_MEMORY");
  }, { models: [{ id: model, type: "language", tags: ["tool-use"], context_window: userHeavy ? 256000 : 128000, max_tokens: 8192 }] });
  const env = {
    PATH: process.env.PATH ?? "/usr/bin:/bin", HOME: home, TMPDIR: root,
    AI_GATEWAY_API_KEY: "synthetic-compaction-policy", FX_DISABLE_KEYCHAIN: "1", FX_E2E_DISABLE_DOTENV: "1",
    FX_AUTO_UPGRADE: "0", FX_SOUND: "0", FX_MODEL: model,
    FX_GATEWAY_BASE_URL: gateway.baseUrl, FX_GATEWAY_CHAT_URL: gateway.chatUrl,
    FX_E2E_GATEWAY_CHAT_URL: gateway.chatUrl, FX_E2E_GATEWAY_MODELS_URL: `${gateway.baseUrl}/coding-agent/v1/models`,
  };
  async function ask(args: string[], label: string, prompt?: string) {
    const stdout = join(root, `${label}.stdout`), stderr = join(root, `${label}.stderr`);
    const input = join(root, `${label}.input`);
    if (prompt !== undefined) writeFileSync(input, prompt);
    const child = Bun.spawn([binary, "ask", "--json", ...args], { cwd, env, stdin: prompt === undefined ? "ignore" : Bun.file(input), stdout: Bun.file(stdout), stderr: Bun.file(stderr) });
    const timer = setTimeout(() => child.kill("SIGKILL"), 30_000);
    try {
      expect(await child.exited).toBe(0);
      expect(readFileSync(stderr, "utf8")).toBe("");
      return JSON.parse(readFileSync(stdout, "utf8"));
    } finally { clearTimeout(timer); }
  }
  let passed = false;
  try {
    const seed = await ask([], "seed", originalUser);
    expect(summaryCalls).toBe(0);
    const seededRequest = JSON.parse(bodies[0]!);
    const seededUser = seededRequest.prompt.findLast((message: { role: string }) => message.role === "user");
    expect(seededUser.content[0].text).toBe(originalUser);
    const sessionDir = join(home, ".fx/sessions", seed.session_id), log = join(sessionDir, "events.jsonl"), before = readFileSync(log);
    phase = "continue";
    const result = await ask(["--resume-id", seed.session_id, "Continue the saved task without losing its pending check."], "continue");
    expect(result.output).toBe("CONTINUED_FROM_COMMITTED_MEMORY");
    expect(summaryCalls).toBe(1);
    const rows = readFileSync(log, "utf8").trim().split("\n").map(line => JSON.parse(line));
    const checkpoints = rows.filter(row => row.event?.context_checkpoint);
    expect(checkpoints.length).toBe(1);
    const saved: string = checkpoints[0].event.context_checkpoint.summary;
    expect(saved.startsWith(checkpointMarker)).toBe(true);
    const payload = JSON.parse(saved.slice(checkpointMarker.length));
    const summary = "The verified value is 73. The pending check is transport-resume.";
    expect(payload.turn_count).toBe(1);
    expect(payload.tool_count).toBe(0);
    const continued = JSON.parse(bodies.at(-1)!);
    const continuedText = JSON.stringify(continued.prompt);
    const shown = (text: string) => JSON.stringify(text).slice(1, -1);
    expect(continuedText).toContain("<compacted_conversation>");
    if (userHeavy) {
      // Too large to show word for word: the turn folds into the summary.
      expect(payload.turns).toEqual([]);
      expect(payload.earlier).toBe(summary);
      expect(continuedText).toContain(shown(`Earlier summary:\n${summary}\n`));
      expect(continuedText).not.toContain("USER_REFERENCE_END");
    } else {
      // The user message stays exact; the long reply is summarized with the turn.
      expect(payload.turns).toHaveLength(1);
      expect(payload.turns[0].users).toEqual([originalUser]);
      expect(payload.turns[0].final).toBe("");
      expect(payload.turns[0].work).toBe(summary);
      expect(continuedText).toContain(shown(`User 1:\n${originalUser}\n`));
      expect(continuedText).toContain(shown(`Assistant 1, summary of its work:\n${summary}\n`));
    }
    expect(continuedText).toContain("Saved word for word: turn M1.");
    expect(continuedText).not.toContain("Assistant reference 5:");
    // Either way the whole turn is saved word for word as M1.
    const record = readFileSync(join(sessionDir, "tool-results", "compacted-M1.txt"), "utf8");
    expect(record).toContain(`User 1:\n${originalUser}\n`);
    expect(record).toContain("Assistant reference 5:");
    expect(record).toContain("PENDING_CHECK=transport-resume");
    expect(continuedText.split("Continue the saved task without losing its pending check.").length - 1).toBe(1);
    const reopened = await ask(["--resume-id", seed.session_id, "Continue after this fresh process restart."], "reopen");
    expect(reopened.output).toBe("CONTINUED_FROM_COMMITTED_MEMORY");
    expect(summaryCalls).toBe(1);
    expect(bodies.at(-1)).toContain("The verified value is 73.");
    expect(bodies.at(-1)).toContain("<compacted_conversation>");
    expect(bodies.at(-1)).not.toContain(checkpointMarker.trim());
    expect(readFileSync(log).subarray(0, before.length).equals(before)).toBe(true);
    passed = true;
  } finally {
    gateway.stop();
    if (passed) rmSync(root, { recursive: true, force: true });
    else { writeFileSync(join(root, "requests.json"), JSON.stringify(bodies, null, 2)); console.error(`compaction evidence retained: ${root}`); }
  }
}, 90_000);

import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  buildActionsRunsApiPath,
  buildForceWithLeaseFlag,
  buildRemoteTrackingRefspec,
  buildOpenAiApiError,
  clampText,
  extractResponseText,
  normalizeIdeaSelection,
  normalizePatchPlan,
  parseLsRemoteBranchHead,
  parseJsonResponse,
  prepareShellCommand,
  resolveAutoloopBackend,
  sanitizeRelativePath,
  stripMarkdownFences,
  uniqueStrings,
  writeJsonFileAtomic,
} from "../scripts/autoloop-lib.mjs";

test("stripMarkdownFences unwraps fenced JSON", () => {
  assert.equal(stripMarkdownFences("```json\n{\"ok\":true}\n```"), "{\"ok\":true}");
  assert.equal(stripMarkdownFences("{\"ok\":true}"), "{\"ok\":true}");
});

test("extractResponseText concatenates output_text parts", () => {
  const response = {
    output: [
      {
        type: "message",
        content: [
          { type: "output_text", text: "one" },
          { type: "ignored", text: "skip" },
          { type: "output_text", text: "two" },
        ],
      },
    ],
  };
  assert.equal(extractResponseText(response), "one\ntwo");
});

test("parseJsonResponse rejects invalid JSON", () => {
  assert.throws(() => parseJsonResponse("not-json"), /invalid JSON/);
});

test("sanitizeRelativePath rejects absolute and traversal paths", () => {
  assert.equal(sanitizeRelativePath("./haskell/web/src/App.tsx"), "haskell/web/src/App.tsx");
  assert.throws(() => sanitizeRelativePath("./"), /resolves to empty/);
  assert.throws(() => sanitizeRelativePath("/tmp/nope"), /Absolute path/);
  assert.throws(() => sanitizeRelativePath("C:/tmp/nope"), /Absolute path/);
  assert.throws(() => sanitizeRelativePath("../nope"), /Path traversal/);
});

test("normalizeIdeaSelection validates required fields", () => {
  const idea = normalizeIdeaSelection({
    noChange: false,
    title: "Improve docs",
    rationale: "Tighten user guidance",
    uiReviewPath: "haskell/web/src/App.tsx",
    uiReviewFocus: "Check button copy and loading feedback.",
    correctnessPath: "test/autoloop.test.mjs",
    correctnessFocus: "Keep the autoloop contract covered by tests.",
    filesNeeded: ["README.md", "CHANGELOG.md", "haskell/web/src/App.tsx", "test/autoloop.test.mjs"],
    verificationCommands: ["cd haskell && cabal build"],
  });
  assert.equal(idea.uiReviewPath, "haskell/web/src/App.tsx");
  assert.equal(idea.correctnessPath, "test/autoloop.test.mjs");
  assert.deepEqual(idea.filesNeeded, [
    "README.md",
    "CHANGELOG.md",
    "haskell/web/src/App.tsx",
    "test/autoloop.test.mjs",
  ]);
  assert.throws(
    () =>
      normalizeIdeaSelection({
        noChange: false,
        title: "",
        rationale: "missing title",
        uiReviewPath: "haskell/web/src/App.tsx",
        uiReviewFocus: "Review the main UI.",
        correctnessPath: "test/autoloop.test.mjs",
        correctnessFocus: "Keep tests aligned.",
        filesNeeded: ["README.md", "haskell/web/src/App.tsx", "test/autoloop.test.mjs"],
      }),
    /title must not be empty/,
  );
  assert.throws(
    () =>
      normalizeIdeaSelection({
        noChange: false,
        title: "Bad review coverage",
        rationale: "UI review path is missing from filesNeeded",
        uiReviewPath: "haskell/web/src/App.tsx",
        uiReviewFocus: "Review the main UI.",
        correctnessPath: "test/autoloop.test.mjs",
        correctnessFocus: "Keep tests aligned.",
        filesNeeded: ["test/autoloop.test.mjs"],
      }),
    /filesNeeded must include uiReviewPath/,
  );
});

test("normalizePatchPlan validates change entries", () => {
  const plan = normalizePatchPlan({
    noChange: false,
    title: "Patch docs",
    summary: "Explain setup",
    commitMessage: "Explain setup",
    uiReviewSummary: "Reviewed the UI file and found no safe change in scope.",
    correctnessSummary: "The tests keep the autoloop path contract intact.",
    changes: [{ path: "README.md", content: "# hi" }],
    verificationCommands: [],
  });
  assert.equal(plan.changes[0]?.path, "README.md");
  assert.equal(plan.uiReviewSummary, "Reviewed the UI file and found no safe change in scope.");
  assert.throws(
    () =>
      normalizePatchPlan({
        noChange: false,
        title: "Bad patch",
        summary: "Bad patch",
        commitMessage: "Bad patch",
        uiReviewSummary: "Reviewed the UI file.",
        correctnessSummary: "The contract is unchanged.",
        changes: [{ path: "../oops", content: "x" }],
      }),
    /Path traversal/,
  );
  assert.throws(
    () =>
      normalizePatchPlan({
        noChange: false,
        title: "Duplicate patch",
        summary: "Duplicate patch",
        commitMessage: "Duplicate patch",
        uiReviewSummary: "Reviewed the UI file.",
        correctnessSummary: "The contract is unchanged.",
        changes: [
          { path: "README.md", content: "# one" },
          { path: "README.md", content: "# two" },
        ],
      }),
    /duplicate path/,
  );
});

test("clampText preserves short text and truncates long text", () => {
  assert.equal(clampText("short", 20), "short");
  const clamped = clampText("abcdefghijklmnopqrstuvwxyz", 12);
  assert.ok(clamped.length <= 12, `expected clampText to respect maxChars, got ${clamped.length}`);
  assert.notEqual(clamped, "abcdefghijklmnopqrstuvwxyz");
});

test("prepareShellCommand bootstraps ghcup for Haskell verification commands", () => {
  assert.equal(prepareShellCommand("cd haskell && cabal build"), 'source "$HOME/.ghcup/env" 2>/dev/null || true; cd haskell && cabal build');
  assert.equal(
    prepareShellCommand('source "$HOME/.ghcup/env" 2>/dev/null || true; cd haskell && cabal build'),
    'source "$HOME/.ghcup/env" 2>/dev/null || true; cd haskell && cabal build',
  );
  assert.equal(
    prepareShellCommand("cd haskell && bash scripts/ci_smoke.sh"),
    'source "$HOME/.ghcup/env" 2>/dev/null || true; cd haskell && bash scripts/ci_smoke.sh',
  );
  assert.equal(prepareShellCommand("cd haskell/web && npm --workspaces=false run test"), "cd haskell/web && npm --workspaces=false run test");
});

test("uniqueStrings preserves first occurrence order", () => {
  assert.deepEqual(uniqueStrings(["a", "b", "a", "c"]), ["a", "b", "c"]);
});

test("buildOpenAiApiError marks quota and auth failures as skippable", () => {
  const quotaErr = buildOpenAiApiError(429, {
    error: { code: "insufficient_quota", type: "insufficient_quota" },
  });
  const authErr = buildOpenAiApiError(401, {
    error: { code: "invalid_api_key", type: "invalid_request_error" },
  });
  const forbiddenErr = buildOpenAiApiError(403, {
    error: { code: "insufficient_permissions", type: "permission_error" },
  });
  const serverErr = buildOpenAiApiError(500, {
    error: { code: "server_error", type: "server_error" },
  });
  assert.equal(quotaErr.skipAutoloop, true);
  assert.equal(authErr.skipAutoloop, true);
  assert.equal(forbiddenErr.skipAutoloop, true);
  assert.equal(serverErr.skipAutoloop, false);
});

test("resolveAutoloopBackend prefers OpenAI then Codex in auto mode", () => {
  assert.equal(resolveAutoloopBackend("auto", { hasOpenAiKey: true, hasCodex: true }), "openai");
  assert.equal(resolveAutoloopBackend("", { hasOpenAiKey: false, hasCodex: true }), "codex");
  assert.equal(resolveAutoloopBackend("", { hasOpenAiKey: false, hasCodex: false }), "");
});

test("resolveAutoloopBackend respects explicit backend requests", () => {
  assert.equal(resolveAutoloopBackend("openai", { hasOpenAiKey: true, hasCodex: true }), "openai");
  assert.equal(resolveAutoloopBackend("codex", { hasOpenAiKey: true, hasCodex: true }), "codex");
  assert.equal(resolveAutoloopBackend("codex", { hasOpenAiKey: true, hasCodex: false }), "");
  assert.throws(
    () => resolveAutoloopBackend("mystery", { hasOpenAiKey: true, hasCodex: true }),
    /Unknown autoloop backend/,
  );
});

test("parseLsRemoteBranchHead extracts the requested remote branch head", () => {
  const raw = [
    `${"a".repeat(40)}\trefs/heads/main`,
    `${"b".repeat(40)}\trefs/heads/autoloop/main`,
  ].join("\n");
  assert.equal(parseLsRemoteBranchHead(raw, "autoloop/main"), "b".repeat(40));
  assert.equal(parseLsRemoteBranchHead(raw, "missing"), "");
  assert.equal(parseLsRemoteBranchHead("", "main"), "");
});

test("buildForceWithLeaseFlag uses explicit branch heads and validates object ids", () => {
  assert.equal(
    buildForceWithLeaseFlag("autoloop/main", "a".repeat(40)),
    `--force-with-lease=refs/heads/autoloop/main:${"a".repeat(40)}`,
  );
  assert.equal(
    buildForceWithLeaseFlag("refs/heads/main", "b".repeat(40)),
    `--force-with-lease=refs/heads/main:${"b".repeat(40)}`,
  );
  assert.equal(buildForceWithLeaseFlag("autoloop/main", ""), "--force-with-lease=refs/heads/autoloop/main:");
  assert.throws(() => buildForceWithLeaseFlag("autoloop/main", "not-a-sha"), /expectedOid must be a 40-character hex object id/);
});

test("buildRemoteTrackingRefspec targets refs/remotes/origin for branch heads", () => {
  assert.equal(
    buildRemoteTrackingRefspec("autoloop/main"),
    "refs/heads/autoloop/main:refs/remotes/origin/autoloop/main",
  );
  assert.equal(buildRemoteTrackingRefspec("refs/heads/main"), "refs/heads/main:refs/remotes/origin/main");
});

test("buildActionsRunsApiPath scopes workflow run lookup to a head sha and branch", () => {
  assert.equal(
    buildActionsRunsApiPath("a".repeat(40), "autoloop/main", 30),
    `repos/:owner/:repo/actions/runs?head_sha=${"a".repeat(40)}&per_page=30&branch=autoloop%2Fmain`,
  );
  assert.equal(
    buildActionsRunsApiPath("b".repeat(40), "refs/heads/main", 200),
    `repos/:owner/:repo/actions/runs?head_sha=${"b".repeat(40)}&per_page=100&branch=main`,
  );
});

test("repo root package exposes the autoloop verifier script", async () => {
  const pkgRaw = await fs.readFile(new URL("../package.json", import.meta.url), "utf8");
  const pkg = JSON.parse(pkgRaw);
  const testScript = pkg?.scripts?.["test:autoloop"];
  assert.equal(typeof testScript, "string");
  assert.match(testScript, /\bnode --test test\/autoloop\.test\.mjs\b/);
});
test("writeJsonFileAtomic creates parent directories and writes formatted JSON", async () => {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "autoloop-test-"));
  const filePath = path.join(dir, "nested", "status.json");
  await writeJsonFileAtomic(filePath, { phase: "verify", ok: true });
  const out = await fs.readFile(filePath, "utf8");
  assert.deepEqual(JSON.parse(out), { phase: "verify", ok: true });
});

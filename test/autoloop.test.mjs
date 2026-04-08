import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  buildBranchMergeCandidates,
  buildActionsRunsApiPath,
  buildAutoloopRecoveryBranchName,
  buildForceWithLeaseFlag,
  buildRemoteTrackingRefspec,
  buildOpenAiApiError,
  clampText,
  extractCodexExecLastMessage,
  extractResponseText,
  normalizeGitBranchShortName,
  normalizeIdeaSelection,
  normalizePatchPlan,
  parseGitStatusPaths,
  parseLsRemoteBranchHead,
  parseJsonResponse,
  prepareShellCommand,
  resolveAutoloopBackend,
  sanitizeRelativePath,
  stripMarkdownFences,
  uniqueStrings,
  writeJsonFileAtomic,
} from "../scripts/autoloop-lib.mjs";

function extractAutoloopPhases(script) {
  return Array.from(script.matchAll(/phase:\s*"([^"]+)"/g), (match) => match[1]);
}

function assertOrderedSubsequence(values, expected, label) {
  let cursor = -1;
  for (const value of expected) {
    const next = values.indexOf(value, cursor + 1);
    assert.notEqual(next, -1, `${label} is missing ${value}`);
    cursor = next;
  }
}

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

test("extractCodexExecLastMessage reads the final completed agent message from JSONL", () => {
  const jsonl = [
    "OpenAI Codex v0.117.0",
    '{"type":"thread.started","thread_id":"abc"}',
    '{"type":"item.completed","item":{"type":"agent_message","text":"{\\"step\\":1}"}}',
    '{"type":"item.completed","item":{"type":"agent_message","text":"{\\"ok\\":true}"}}',
    '{"type":"turn.completed","usage":{"output_tokens":40}}',
  ].join("\n");
  assert.equal(extractCodexExecLastMessage(jsonl), '{"ok":true}');
});

test("extractCodexExecLastMessage rejects streams without a completed agent message", () => {
  assert.throws(
    () => extractCodexExecLastMessage('{"type":"turn.started"}\n{"type":"turn.completed"}'),
    /no completed agent message/i,
  );
});

test("parseJsonResponse rejects invalid JSON", () => {
  assert.throws(() => parseJsonResponse("not-json"), /invalid JSON/);
});

test("sanitizeRelativePath rejects absolute and traversal paths", () => {
  assert.equal(sanitizeRelativePath("./haskell/web/src/App.tsx"), "haskell/web/src/App.tsx");
  assert.equal(sanitizeRelativePath("haskell/web/src/./App.tsx"), "haskell/web/src/App.tsx");
  assert.equal(sanitizeRelativePath("docs//guide.md"), "docs/guide.md");
  assert.throws(() => sanitizeRelativePath("./"), /resolves to empty/);
  assert.throws(() => sanitizeRelativePath("/tmp/nope"), /Absolute path/);
  assert.throws(() => sanitizeRelativePath("C:/tmp/nope"), /Absolute path/);
  assert.throws(() => sanitizeRelativePath("../nope"), /Path traversal/);
});

test("normalizeIdeaSelection validates required fields", () => {
  const idea = normalizeIdeaSelection({
    noChange: false,
    title: "Clamp pathological thresholds",
    rationale: "Bias toward backend trading invariants",
    algorithmReviewPath: "haskell/app/Trader/Trading.hs",
    algorithmReviewFocus: "Review threshold and signal-decision invariants.",
    formalMethodsPath: "FORMAL_METHODS.md",
    formalMethodsFocus: "Keep the threshold invariant and proof sketch aligned.",
    filesNeeded: ["README.md", "CHANGELOG.md", "haskell/app/Trader/Trading.hs", "FORMAL_METHODS.md"],
    verificationCommands: ["cd haskell && cabal build"],
  });
  assert.equal(idea.algorithmReviewPath, "haskell/app/Trader/Trading.hs");
  assert.equal(idea.formalMethodsPath, "FORMAL_METHODS.md");
  assert.deepEqual(idea.filesNeeded, [
    "README.md",
    "CHANGELOG.md",
    "haskell/app/Trader/Trading.hs",
    "FORMAL_METHODS.md",
  ]);
  assert.throws(
    () =>
      normalizeIdeaSelection({
        noChange: false,
        title: "",
        rationale: "missing title",
        algorithmReviewPath: "haskell/app/Trader/Trading.hs",
        algorithmReviewFocus: "Review the trading logic.",
        formalMethodsPath: "FORMAL_METHODS.md",
        formalMethodsFocus: "Keep tests aligned.",
        filesNeeded: ["README.md", "haskell/app/Trader/Trading.hs", "FORMAL_METHODS.md"],
      }),
    /title must not be empty/,
  );
  assert.throws(
    () =>
      normalizeIdeaSelection({
        noChange: false,
        title: "Bad review coverage",
        rationale: "Algorithm review path is missing from filesNeeded",
        algorithmReviewPath: "haskell/app/Trader/Trading.hs",
        algorithmReviewFocus: "Review the trading logic.",
        formalMethodsPath: "FORMAL_METHODS.md",
        formalMethodsFocus: "Keep tests aligned.",
        filesNeeded: ["FORMAL_METHODS.md"],
      }),
    /filesNeeded must include algorithmReviewPath/,
  );
  assert.throws(
    () =>
      normalizeIdeaSelection({
        noChange: false,
        title: "Bad formal coverage",
        rationale: "Formal methods path is missing from filesNeeded",
        algorithmReviewPath: "haskell/app/Trader/Trading.hs",
        algorithmReviewFocus: "Review the trading logic.",
        formalMethodsPath: "FORMAL_METHODS.md",
        formalMethodsFocus: "Keep tests aligned.",
        filesNeeded: ["haskell/app/Trader/Trading.hs"],
      }),
    /filesNeeded must include formalMethodsPath/,
  );
  assert.throws(
    () =>
      normalizeIdeaSelection({
        noChange: false,
        title: "Bad algorithm path",
        rationale: "UI files must not satisfy the backend review slot",
        algorithmReviewPath: "haskell/web/src/App.tsx",
        algorithmReviewFocus: "Review the main UI.",
        formalMethodsPath: "FORMAL_METHODS.md",
        formalMethodsFocus: "Keep tests aligned.",
        filesNeeded: ["haskell/web/src/App.tsx", "FORMAL_METHODS.md"],
      }),
    /algorithmReviewPath must be within/,
  );
  assert.throws(
    () =>
      normalizeIdeaSelection({
        noChange: false,
        title: "Bad formal sibling path",
        rationale: "An exact file scope must not accept sibling lookalikes.",
        algorithmReviewPath: "haskell/app/Trader/Trading.hs",
        algorithmReviewFocus: "Review the trading logic.",
        formalMethodsPath: "FORMAL_METHODS.md.bak",
        formalMethodsFocus: "Keep tests aligned.",
        filesNeeded: ["haskell/app/Trader/Trading.hs", "FORMAL_METHODS.md.bak"],
      }),
    /formalMethodsPath must be within/,
  );
});

test("normalizePatchPlan validates change entries", () => {
  const plan = normalizePatchPlan({
    noChange: false,
    title: "Patch docs",
    summary: "Explain setup",
    commitMessage: "Explain setup",
    algorithmReviewSummary: "Reviewed the backend trading file and applied the threshold fix there.",
    formalMethodsSummary: "The tests keep the autoloop path contract intact.",
    changes: [{ path: "README.md", content: "# hi" }],
    verificationCommands: [],
  });
  assert.equal(plan.changes[0]?.path, "README.md");
  assert.equal(plan.algorithmReviewSummary, "Reviewed the backend trading file and applied the threshold fix there.");
  assert.throws(
    () =>
      normalizePatchPlan({
        noChange: false,
        title: "Bad patch",
        summary: "Bad patch",
        commitMessage: "Bad patch",
        algorithmReviewSummary: "Reviewed the backend trading file.",
        formalMethodsSummary: "The contract is unchanged.",
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
        algorithmReviewSummary: "Reviewed the backend trading file.",
        formalMethodsSummary: "The contract is unchanged.",
        changes: [
          { path: "README.md", content: "# one" },
          { path: "README.md", content: "# two" },
        ],
      }),
    /duplicate path/,
  );
  assert.throws(
    () =>
      normalizePatchPlan({
        noChange: false,
        title: "Canonical duplicate patch",
        summary: "Canonical duplicate patch",
        commitMessage: "Canonical duplicate patch",
        algorithmReviewSummary: "Reviewed the backend trading file.",
        formalMethodsSummary: "The contract is unchanged.",
        changes: [
          { path: "haskell/app/Trader/Trading.hs", content: "# one" },
          { path: "haskell/app/Trader/./Trading.hs", content: "# two" },
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

test("parseGitStatusPaths extracts tracked, untracked, and renamed paths", () => {
  const raw = [' M haskell/app/Main.hs', '?? README.md', 'R  old.txt -> docs/new.txt'].join("\n");
  assert.deepEqual(parseGitStatusPaths(raw), ["haskell/app/Main.hs", "README.md", "docs/new.txt"]);
  assert.deepEqual(parseGitStatusPaths("M haskell/app/Trader/Formal/CloseTiming.hs"), [
    "haskell/app/Trader/Formal/CloseTiming.hs",
  ]);
});

test("buildAutoloopRecoveryBranchName scopes rescue branches under autoloop/wip", () => {
  const branch = buildAutoloopRecoveryBranchName({
    loopBranch: "main",
    runId: "cycle-7",
    timestamp: "2026-03-31T06:15:04.123Z",
  });
  assert.equal(branch, "autoloop/wip/main/cycle-7-2026-03-31t06-15-04-123z");
});

test("normalizeGitBranchShortName strips origin and refs prefixes", () => {
  assert.equal(normalizeGitBranchShortName("origin/feature/test"), "feature/test");
  assert.equal(normalizeGitBranchShortName("refs/heads/main"), "main");
  assert.equal(normalizeGitBranchShortName("refs/remotes/origin/topic"), "topic");
  assert.equal(normalizeGitBranchShortName("origin/HEAD"), "");
  assert.equal(normalizeGitBranchShortName(""), "");
});

test("buildBranchMergeCandidates prefers local heads while deduping remote matches", () => {
  assert.deepEqual(
    buildBranchMergeCandidates({
      localBranches: ["feature/local", "topic/only-local", "main"],
      remoteBranches: ["origin/feature/local", "origin/topic/only-remote", "origin/main", "origin/HEAD"],
      baseBranch: "main",
    }),
    [
      {
        shortName: "feature/local",
        ref: "feature/local",
        localRef: "feature/local",
        remoteRef: "origin/feature/local",
      },
      {
        shortName: "topic/only-local",
        ref: "topic/only-local",
        localRef: "topic/only-local",
        remoteRef: "",
      },
      {
        shortName: "topic/only-remote",
        ref: "origin/topic/only-remote",
        localRef: "",
        remoteRef: "origin/topic/only-remote",
      },
    ],
  );
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

test("autoloop script targets the base branch directly without PR helpers", async () => {
  const script = await fs.readFile(new URL("../scripts/autoloop.mjs", import.meta.url), "utf8");
  assert.match(script, /const LOOP_BRANCH = BASE_BRANCH;/);
  assert.match(script, /const SKIP_CI_WAIT = readBooleanEnv\(process\.env\.AUTOLOOP_SKIP_CI_WAIT\);/);
  assert.match(script, /function waitForBranchCi\(headSha, branchName\)/);
  assert.doesNotMatch(script, /function ensurePullRequest\(/);
  assert.doesNotMatch(script, /function mergePullRequest\(/);
});

test("autoloop script polls GitHub CI for each pushed sha before completing", async () => {
  const script = await fs.readFile(new URL("../scripts/autoloop.mjs", import.meta.url), "utf8");
  assert.match(script, /const pushedHeadSha = runGit\(\["rev-parse", "HEAD"\]\);/);
  assert.match(script, /phase: "ci-wait",\s*[\s\S]*headSha: pushedHeadSha/);
  assert.match(script, /const ci = waitForBranchCi\(pushedHeadSha, LOOP_BRANCH\);/);
  assert.match(script, /function pollGitHubActionsForHead\(headSha, branchName, \{ requireWorkflowRun \}\)/);
  assert.match(script, /const runs = listWorkflowRunsForHead\(headSha, branchName\)\.filter\(\(run\) => run\.head_sha === headSha\);/);
  assert.match(script, /const failedRuns = runs\.filter\(/);
  assert.match(script, /const pendingRuns = runs\.filter\(/);
  assert.match(script, /return \{\s*ok: true,\s*headSha,\s*branchName,\s*workflowRuns: runs,/);
});

test("autoloop script feeds failed CI logs back into codex repair prompts", async () => {
  const script = await fs.readFile(new URL("../scripts/autoloop.mjs", import.meta.url), "utf8");
  assert.match(script, /failureContext = \{\s*[\s\S]*failedLog: ci\.failedLog,/);
  assert.match(script, /const idea = failureContext\s*\?\s*await requestFixIdea\(repoContext, failureContext\)/);
  assert.match(script, /"Failed log excerpt:",\s*clampText\(failureContext\.failedLog, 20000\)/);
  assert.match(script, /failureContext \? `Failed CI log excerpt:\\n\$\{clampText\(failureContext\.failedLog, 18000\)\}` : ""/);
  assert.match(script, /function readFailedWorkflowRunLog\(runId\)/);
  assert.match(script, /const failedLog = clampText\(readFailedWorkflowRunLog\(runId\), 18000\);/);
});

test("autoloop script repairs the latest remote branch head before proposing new work", async () => {
  const script = await fs.readFile(new URL("../scripts/autoloop.mjs", import.meta.url), "utf8");
  assert.match(script, /let failureContext = await inspectLatestRemoteBranchFailureContext\(\);/);
  assert.match(script, /Latest remote \$\{failureContext\.branchName\} commit \$\{failureContext\.headSha\} has failing GitHub Actions\./);
  assert.match(script, /async function inspectLatestRemoteBranchFailureContext\(\)/);
  assert.match(script, /const latestHeadSha = readRemoteBranchHead\(LOOP_BRANCH\) \|\| readRemoteBranchHead\(BASE_BRANCH\);/);
  assert.match(script, /const ci = pollGitHubActionsForHead\(latestHeadSha, LOOP_BRANCH, \{ requireWorkflowRun: false \}\);/);
  assert.match(script, /changedPaths: listCommitChangedPaths\(latestHeadSha\),/);
});

test("autoloop script prefers the stored gh auth token over a stale GH_TOKEN environment value", async () => {
  const script = await fs.readFile(new URL("../scripts/autoloop.mjs", import.meta.url), "utf8");
  assert.match(script, /function buildSanitizedGhAuthEnv\(extraEnv = \{\}\)/);
  assert.match(script, /function getStoredGhToken\(\)/);
  assert.match(script, /env: buildSanitizedGhAuthEnv\(\),/);
  assert.match(script, /const storedToken = getStoredGhToken\(\);/);
  assert.match(script, /const envToken =[\s\S]*process\.env\.GITHUB_PAT/);
  assert.match(script, /env: storedToken[\s\S]*buildSanitizedGhAuthEnv\(\{[\s\S]*GH_TOKEN: storedToken,/);
});

test("autoloop codex backend uses JSON mode over stdin with a bounded timeout", async () => {
  const script = await fs.readFile(new URL("../scripts/autoloop.mjs", import.meta.url), "utf8");
  assert.match(script, /const CODEX_EXEC_TIMEOUT_MS = clampInt\(process\.env\.AUTOLOOP_CODEX_TIMEOUT_MS, 300000, 10000, 1800000\);/);
  assert.match(script, /const CODEX_PATCH_TIMEOUT_MS = clampInt\(\s*process\.env\.AUTOLOOP_CODEX_PATCH_TIMEOUT_MS,\s*1800000,\s*CODEX_EXEC_TIMEOUT_MS,\s*3600000,\s*\);/);
  assert.match(script, /const CODEX_RETRY_MAX_ATTEMPTS = clampInt\(process\.env\.AUTOLOOP_CODEX_RETRY_MAX_ATTEMPTS, 2, 1, 5\);/);
  assert.match(script, /const CODEX_RETRY_BACKOFF_MS = clampInt\(process\.env\.AUTOLOOP_CODEX_RETRY_BACKOFF_MS, 15000, 1000, 120000\);/);
  assert.match(script, /const CODEX_REASONING_EFFORT = resolveCodexReasoningEffort\(process\.env\.AUTOLOOP_CODEX_REASONING_EFFORT\);/);
  assert.match(script, /if \(value === "low" \|\| value === "medium" \|\| value === "high" \|\| value === "xhigh"\) return value;/);
  assert.match(script, /return "xhigh";/);
  assert.match(script, /"exec",\s*"--json",\s*"--ephemeral",\s*"--sandbox",\s*"read-only"/);
  assert.match(script, /"--model",\s*OPENAI_MODEL,\s*"-c",\s*`model_reasoning_effort="\$\{CODEX_REASONING_EFFORT\}"`,\s*"-"/);
  assert.match(script, /Do not run shell commands, open files, inspect the repository, or use web search\./);
  assert.match(script, /async function callModelJson\(\{ prompt, maxOutputTokens = 4000, timeoutMs = CODEX_EXEC_TIMEOUT_MS \}\)/);
  assert.match(script, /timeoutMs,\s*\n\s*}\s*,\s*\n\s*\);/);
  assert.match(script, /for \(let attempt = 1; attempt <= CODEX_RETRY_MAX_ATTEMPTS; attempt \+= 1\)/);
  assert.match(script, /if \(!isRetryableCodexExecError\(err\) \|\| attempt >= CODEX_RETRY_MAX_ATTEMPTS\) throw err;/);
  assert.match(script, /await sleep\(delayMs\);/);
  assert.match(script, /function isRetryableCodexExecError\(err\)/);
  assert.match(script, /callModelJson\(\{ prompt, maxOutputTokens: 12000, timeoutMs: CODEX_PATCH_TIMEOUT_MS \}\)/);
  assert.match(script, /parseJsonResponse\(extractCodexExecLastMessage\(rawEvents\)\)/);
  assert.doesNotMatch(script, /--output-last-message/);
});

test("bounded autoloop reports the required lifecycle phases in order", async () => {
  const script = await fs.readFile(new URL("../scripts/autoloop.mjs", import.meta.url), "utf8");
  const phases = extractAutoloopPhases(script);

  assertOrderedSubsequence(
    phases,
    ["choose-change", "algorithm-review", "formal-methods-review", "verify", "commit-push", "ci-wait"],
    "required autoloop lifecycle phases",
  );
  assertOrderedSubsequence(
    phases,
    ["formal-methods-review", "plan-patch", "apply-patch", "verify"],
    "autoloop review-to-verification bridge phases",
  );
  assertOrderedSubsequence(
    phases,
    ["commit-push", "ci-wait", "repair-needed"],
    "autoloop push-to-repair bridge phases",
  );
});

test("autoloop forever script auto-snapshots recoverable dirty cycles before blocking", async () => {
  const script = await fs.readFile(new URL("../scripts/autoloop-forever.mjs", import.meta.url), "utf8");
  assert.match(script, /const dirtyRecovery = await tryAutoSnapshotDirtyCycle\(\);/);
  assert.match(script, /const dirtyCheckpoint = await tryAutoCheckpointDirtyWorktree\(\);/);
  assert.match(script, /runCommand\("git", \["status", "--porcelain"\], \{ trimOutput: false \}\)/);
  assert.match(script, /cycle [^`]*recovery=\$\{dirtyRecovery\?\.recovered \? dirtyRecovery\.branch : "none"\}/);
  assert.match(script, /cycleStatus\?\.phase !== "error" \|\| changedPaths\.length === 0/);
  assert.match(script, /dirty worktree does not exactly match the last failed cycle changedPaths/);
  assert.match(script, /buildAutoloopRecoveryBranchName\(/);
  assert.match(script, /auto-snapshotted failed dirty cycle to/);
  assert.match(script, /buildAutoloopDirtyCheckpointBranchName\(/);
  assert.match(script, /auto-checkpointed dirty worktree to/);
});

test("autoloop forever script reconciles every unmerged branch onto main before bounded cycles", async () => {
  const script = await fs.readFile(new URL("../scripts/autoloop-forever.mjs", import.meta.url), "utf8");
  assert.match(script, /const BASE_BRANCH = normalizeGitBranchShortName\(process\.env\.AUTOLOOP_BASE_BRANCH \|\| "main"\) \|\| "main";/);
  assert.match(script, /const branchSweep = await reconcileUnmergedBranchesOntoBaseBranch\(\);/);
  assert.match(script, /runCommand\("git", \["fetch", "origin", "--prune"\], \{ capture: false \}\);/);
  assert.match(script, /buildBranchMergeCandidates\(\{ localBranches, remoteBranches, baseBranch: BASE_BRANCH \}\)/);
  assert.match(script, /runCommand\("git", \["branch", "--format=%\(refname:short\)", "--no-merged", BASE_BRANCH\], \{ trimOutput: false \}\)/);
  assert.match(script, /runCommand\("git", \["branch", "-r", "--format=%\(refname:short\)", "--no-merged", BASE_BRANCH\], \{ trimOutput: false \}\)/);
  assert.match(script, /runCommand\("git", \["merge", "--no-ff", "--no-edit", branchRef\], \{ capture: false \}\)/);
  assert.match(script, /runCommand\("git", \["restore", "--source=HEAD", "--staged", "--worktree", "--", \.\.\.conflicts\], \{ capture: false \}\)/);
  assert.match(script, /runCommand\("git", \["push", "origin", `\$\{BASE_BRANCH\}:refs\/heads\/\$\{BASE_BRANCH\}`\], \{ capture: false \}\)/);
});

test("autoloop workflow uses an optional dedicated push token and no PR permission", async () => {
  const workflow = await fs.readFile(new URL("../.github/workflows/autoloop.yml", import.meta.url), "utf8");
  assert.match(workflow, /contents:\s+write/);
  assert.doesNotMatch(workflow, /pull-requests:\s+write/);
  assert.match(workflow, /token:\s+\$\{\{\s*secrets\.AUTOLOOP_PUSH_TOKEN \|\| github\.token\s*\}\}/);
  assert.match(workflow, /AUTOLOOP_SKIP_CI_WAIT:\s+\$\{\{\s*secrets\.AUTOLOOP_PUSH_TOKEN == '' && '1' \|\| ''\s*\}\}/);
});

test("repo root package exposes the autoloop verifier script", async () => {
  const pkgRaw = await fs.readFile(new URL("../package.json", import.meta.url), "utf8");
  const pkg = JSON.parse(pkgRaw);
  const testScript = pkg?.scripts?.["test:autoloop"];
  assert.equal(typeof testScript, "string");
  assert.match(testScript, /\bnode --test test\/autoloop\.test\.mjs\b/);
});

test("repo root test command includes the autoloop verifier", async () => {
  const pkgRaw = await fs.readFile(new URL("../package.json", import.meta.url), "utf8");
  const pkg = JSON.parse(pkgRaw);
  const testScript = pkg?.scripts?.test;
  assert.equal(typeof testScript, "string");
  assert.match(testScript, /\bnpm run test:autoloop\b/);
});
test("writeJsonFileAtomic creates parent directories and writes formatted JSON", async () => {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "autoloop-test-"));
  const filePath = path.join(dir, "nested", "status.json");
  await writeJsonFileAtomic(filePath, { phase: "verify", ok: true });
  const out = await fs.readFile(filePath, "utf8");
  assert.deepEqual(JSON.parse(out), { phase: "verify", ok: true });
});

test("autoloop forever runner emits a heartbeat so status timestamps cannot go stale while alive", async () => {
  const script = await fs.readFile(new URL("../scripts/autoloop-forever.mjs", import.meta.url), "utf8");
  assert.match(script, /const STATUS_HEARTBEAT_SECONDS = clampInt\(process\.env\.AUTOLOOP_FOREVER_STATUS_HEARTBEAT_SECONDS, 15, 5, 300\);/);
  assert.match(script, /startStatusHeartbeat\(\);/);
  assert.match(script, /heartbeatAt: new Date\(\)\.toISOString\(\)/);
  assert.match(script, /statusHeartbeatTimer\.unref\?\.\(\);/);
  assert.match(script, /stopStatusHeartbeat\(\);/);
});

test("autoloop forever status command marks dead runner pids as dead instead of trusting stale JSON", async () => {
  const script = await fs.readFile(new URL("../scripts/autoloop-forever.sh", import.meta.url), "utf8");
  assert.match(script, /python3 - "\$\{STATUS_FILE\}" "\$\{PID_FILE\}"/);
  assert.match(script, /status\["pidAlive"\] = alive/);
  assert.match(script, /status\["live"\] = alive and status\.get\("state"\) not in \{"stopped", "error", "dead"\}/);
  assert.match(script, /status\["state"\] = "dead"/);
  assert.match(script, /runner pid recorded in status is not alive/);
});

test("launchagent installer keeps the forever runner alive across login sessions", async () => {
  const script = await fs.readFile(new URL("../scripts/install-autoloop-launchagent.sh", import.meta.url), "utf8");
  assert.match(script, /AUTOLOOP_LAUNCHD_LABEL:-ai\.openclaw\.trader\.autoloop\.forever/);
  assert.match(script, /<key>RunAtLoad<\/key>\s*<true\/>/);
  assert.match(script, /<key>KeepAlive<\/key>\s*<true\/>/);
  assert.match(script, /scripts\/autoloop-forever\.sh<\/string>/);
  assert.match(script, /launchctl bootstrap/);
  assert.match(script, /launchctl kickstart -k/);
});

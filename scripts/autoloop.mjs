#!/usr/bin/env node

import fs from "node:fs/promises";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { execFileSync } from "node:child_process";
import {
  buildActionsRunsApiPath,
  buildRemoteTrackingRefspec,
  buildOpenAiApiError,
  clampText,
  extractCodexExecLastMessage,
  extractResponseText,
  normalizeIdeaSelection,
  normalizePatchPlan,
  parseLsRemoteBranchHead,
  prepareShellCommand,
  sanitizeRelativePath,
  parseJsonResponse,
  resolveAutoloopBackend,
  uniqueStrings,
  writeJsonFileAtomic,
} from "./autoloop-lib.mjs";

const ROOT = process.cwd();
const OPENAI_BASE_URL = (process.env.OPENAI_BASE_URL || "https://api.openai.com/v1").replace(/\/$/, "");
const OPENAI_MODEL = process.env.AUTOLOOP_MODEL || process.env.OPENAI_MODEL || "gpt-5.4";
const BASE_BRANCH =
  process.env.AUTOLOOP_BASE_BRANCH || process.env.GITHUB_BASE_REF || process.env.GITHUB_REF_NAME || "main";
const LOOP_BRANCH = BASE_BRANCH;
const MAX_ITERATIONS = clampInt(process.env.AUTOLOOP_MAX_ITERATIONS, 2, 1, 5);
const MAX_EDITABLE_FILE_BYTES = clampInt(process.env.AUTOLOOP_MAX_FILE_BYTES, 1000000, 4000, 5000000);
const MAX_EDITABLE_FILES = clampInt(process.env.AUTOLOOP_MAX_FILES, 120, 20, 300);
const PATCH_PLAN_PROMPT_MAX_CHARS = clampInt(process.env.AUTOLOOP_PATCH_PLAN_MAX_CHARS, 900000, 200000, 1048576);
const DRY_RUN = process.argv.includes("--dry-run");
const STATUS_FILE = resolveOptionalPath(process.env.AUTOLOOP_STATUS_FILE);
const RUN_ID = process.env.AUTOLOOP_RUN_ID || "";
const RUN_MODE = process.env.AUTOLOOP_RUN_MODE || "bounded";
const REQUESTED_BACKEND = process.env.AUTOLOOP_BACKEND || "auto";
const CI_DISCOVERY_POLL_SECONDS = clampInt(process.env.AUTOLOOP_CI_DISCOVERY_POLL_SECONDS, 30, 5, 300);
const CI_DISCOVERY_TIMEOUT_SECONDS = clampInt(process.env.AUTOLOOP_CI_DISCOVERY_TIMEOUT_SECONDS, 3000, 60, 7200);
const CODEX_EXEC_TIMEOUT_MS = clampInt(process.env.AUTOLOOP_CODEX_TIMEOUT_MS, 300000, 10000, 1800000);
const CODEX_PATCH_TIMEOUT_MS = clampInt(
  process.env.AUTOLOOP_CODEX_PATCH_TIMEOUT_MS,
  1800000,
  CODEX_EXEC_TIMEOUT_MS,
  3600000,
);
const CODEX_RETRY_MAX_ATTEMPTS = clampInt(process.env.AUTOLOOP_CODEX_RETRY_MAX_ATTEMPTS, 2, 1, 5);
const CODEX_RETRY_BACKOFF_MS = clampInt(process.env.AUTOLOOP_CODEX_RETRY_BACKOFF_MS, 15000, 1000, 120000);
const CODEX_REASONING_EFFORT = resolveCodexReasoningEffort(process.env.AUTOLOOP_CODEX_REASONING_EFFORT);
const SKIP_CI_WAIT = readBooleanEnv(process.env.AUTOLOOP_SKIP_CI_WAIT);
const HAS_CODEX = commandExists("codex");
const PLANNER_BACKEND = resolveAutoloopBackend(REQUESTED_BACKEND, {
  hasOpenAiKey: Boolean(process.env.OPENAI_API_KEY),
  hasCodex: HAS_CODEX,
});

const ALLOWED_EDIT_PREFIXES = [
  "README.md",
  "CHANGELOG.md",
  "FORMAL_METHODS.md",
  "test/",
  "haskell/app/",
  "haskell/test/",
  "haskell/web/src/",
  "haskell/web/test/",
  "haskell/scripts/",
];

const BLOCKED_EDIT_PREFIXES = [
  ".env",
  ".github/",
  ".git/",
  "deploy/",
  "dist/",
  "haskell/web/dist/",
  "node_modules/",
  "scripts/autoloop",
  "package-lock.json",
  "haskell/web/package-lock.json",
  "package.json",
  "haskell/web/package.json",
  "haskell/trader.cabal",
];

const FOURMOLU_CHECK_COMMAND = "cd haskell && find app test bench -name '*.hs' -print0 | xargs -0 fourmolu --mode check";

const SAFE_VERIFICATION_COMMANDS = new Set([
  FOURMOLU_CHECK_COMMAND,
  "cd haskell && cabal build",
  "cd haskell && cabal test",
  "cd haskell && bash scripts/ci_smoke.sh",
  "cd haskell/web && npm --workspaces=false run typecheck",
  "cd haskell/web && npm --workspaces=false run test",
  "cd haskell/web && npm --workspaces=false run build",
  "node --test test/autoloop.test.mjs",
]);

const ALGORITHM_REVIEW_PREFIXES = ["haskell/app/"];
const FORMAL_METHODS_REVIEW_PREFIXES = ["FORMAL_METHODS.md", "haskell/app/Trader/Formal/", "test/", "haskell/test/"];
const EDITABLE_FILE_PRIORITY_PREFIXES = [
  "haskell/app/Trader/Formal/",
  "FORMAL_METHODS.md",
  "haskell/test/",
  "haskell/app/Main.hs",
  "haskell/app/OptimizeEquityMain.hs",
  "haskell/app/Trader/",
  "README.md",
  "CHANGELOG.md",
  "test/",
  "haskell/web/test/",
  "haskell/web/src/",
  "haskell/scripts/",
];

let statusState = {
  mode: RUN_MODE,
  runId: RUN_ID,
  dryRun: DRY_RUN,
  backend: PLANNER_BACKEND || "",
  requestedBackend: REQUESTED_BACKEND,
  baseBranch: BASE_BRANCH,
  loopBranch: LOOP_BRANCH,
  maxIterations: MAX_ITERATIONS,
  phase: "starting",
  startedAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
};

let cachedStoredGhToken = null;
let cachedTrackedFiles = null;

async function main() {
  await updateStatus({ phase: "preflight" });
  if (!PLANNER_BACKEND) {
    const message =
      REQUESTED_BACKEND && REQUESTED_BACKEND !== "auto"
        ? `Autoloop backend \"${REQUESTED_BACKEND}\" is unavailable in this workspace; autoloop is skipping.`
        : "Neither OPENAI_API_KEY nor Codex CLI is available; autoloop is skipping.";
    await updateStatus({ phase: "skipped", outcome: "skipped_missing_backend", message });
    if (DRY_RUN) throw new Error(message);
    console.log(message);
    return;
  }

  assertCleanWorktree();
  fetchRemoteTrackingBranch(BASE_BRANCH);
  await checkoutLoopBranch();
  await updateStatus({ phase: "ready" });

  let failureContext = await inspectLatestRemoteBranchFailureContext();
  if (failureContext) {
    await updateStatus({
      phase: "repair-needed",
      iteration: 0,
      failureContext: summarizeFailureContext(failureContext),
      message: `Latest remote ${failureContext.branchName} commit ${failureContext.headSha} has failing GitHub Actions.`,
    });
  }
  for (let iteration = 1; iteration <= MAX_ITERATIONS; iteration += 1) {
    await updateStatus({ phase: "reset-branch", iteration, failureContext: summarizeFailureContext(failureContext) });
    hardResetToCurrentHead();
    const failureRepairPaths = deriveFailureRepairPaths(failureContext);
    const automaticRepair = failureContext ? detectAutomaticRepair(failureContext) : null;
    let automaticRepairFailure = "";
    let plannedPaths = [];
    let verificationCommands = [];
    let commitMessage = "";
    let planSummary = null;

    if (automaticRepair) {
      plannedPaths = automaticRepair.changedPaths;
      verificationCommands = automaticRepair.verificationCommands;
      commitMessage = automaticRepair.commitMessage;
      await updateStatus({
        phase: "auto-repair",
        iteration,
        failureContext: summarizeFailureContext(failureContext),
        automaticRepair: summarizeAutomaticRepair(automaticRepair),
      });
      try {
        applyAutomaticRepair(automaticRepair);
      } catch (err) {
        automaticRepairFailure = err instanceof Error ? err.message : String(err);
        // Formatter failures on syntactically-broken files still need a semantic
        // repair path instead of aborting the cycle.
        console.warn(`Automatic ${automaticRepair.type} repair failed; falling back to semantic repair.\n${automaticRepairFailure}`);
        await updateStatus({
          phase: "auto-repair-fallback",
          iteration,
          failureContext: summarizeFailureContext(failureContext),
          automaticRepair: summarizeAutomaticRepair(automaticRepair),
          message: automaticRepairFailure,
        });
      }
    }

    if (!automaticRepair || automaticRepairFailure) {
      const repoContext = await buildRepoContext(failureRepairPaths);
      await updateStatus({ phase: "choose-change", iteration });
      const selectedIdea = failureContext
        ? await requestFixIdea(repoContext, failureContext, failureRepairPaths, automaticRepairFailure)
        : await requestIdeaSelection(repoContext);
      const idea =
        failureContext && selectedIdea.noChange
          ? buildFailureRepairIdea(failureContext, failureRepairPaths, automaticRepairFailure) || selectedIdea
          : selectedIdea;
      await updateStatus({ phase: "algorithm-review", iteration, idea: summarizeIdea(idea) });
      await updateStatus({ phase: "formal-methods-review", iteration, idea: summarizeIdea(idea) });

      if (idea.noChange) {
        await updateStatus({
          phase: "complete",
          iteration,
          outcome: "no_change",
          message: idea.rationale || "No safe change proposed.",
        });
        console.log(`No safe change proposed${idea.rationale ? `: ${idea.rationale}` : "."}`);
        return;
      }

      const editableFiles = await readEditableFiles(idea.filesNeeded);
      await updateStatus({ phase: "plan-patch", iteration, idea: summarizeIdea(idea) });
      const plan = await requestPatchPlan(repoContext, idea, editableFiles, failureContext);
      if (plan.noChange) {
        await updateStatus({
          phase: "complete",
          iteration,
          outcome: "no_patch_plan",
          message: plan.summary || "No patch plan returned.",
        });
        console.log(`No patch plan returned${plan.summary ? `: ${plan.summary}` : "."}`);
        return;
      }

      assertPlanMatchesEditableFiles(plan.changes, editableFiles);
      plannedPaths = uniqueStrings(plan.changes.map((change) => sanitizeRelativePath(change.path)));
      verificationCommands = planVerificationCommands(plannedPaths, [...idea.verificationCommands, ...plan.verificationCommands]);
      commitMessage = plan.commitMessage;
      planSummary = summarizePlan(plan);
      await updateStatus({ phase: "apply-patch", iteration, plan: planSummary, plannedPaths });
      applyFileChanges(plan.changes);
    }
    let changedPaths = collectChangedPlanPaths(plannedPaths);
    if (changedPaths.length === 0) {
      await updateStatus({
        phase: "complete",
        iteration,
        outcome: "no_changes",
        message: "Model produced no file changes after normalization.",
        plan: planSummary,
        automaticRepair: automaticRepair ? summarizeAutomaticRepair(automaticRepair) : undefined,
      });
      console.log("Model produced no file changes after normalization; stopping.");
      return;
    }
    await updateStatus({
      phase: "verify",
      iteration,
      changedPaths,
      verificationCommands,
      plan: planSummary,
      automaticRepair: automaticRepair ? summarizeAutomaticRepair(automaticRepair) : undefined,
    });
    await runVerificationCommands(verificationCommands);
    changedPaths = collectChangedPlanPaths(plannedPaths);
    if (changedPaths.length === 0) {
      await updateStatus({
        phase: "complete",
        iteration,
        outcome: "verification_reverted_changes",
        message: "Verification returned the worktree to baseline.",
        verificationCommands,
      });
      console.log("Verification returned the worktree to baseline; stopping.");
      return;
    }
    const unexpectedChanges = collectUnexpectedWorktreeChanges(plannedPaths);
    if (unexpectedChanges.length > 0) {
      console.warn(`Skipping unexpected worktree paths outside the plan: ${unexpectedChanges.join(", ")}`);
    }

    if (DRY_RUN) {
      await updateStatus({
        phase: "complete",
        iteration,
        outcome: "dry_run",
        commitMessage,
        changedPaths,
        verificationCommands,
        unexpectedChanges,
        plan: planSummary,
        automaticRepair: automaticRepair ? summarizeAutomaticRepair(automaticRepair) : undefined,
      });
      console.log(JSON.stringify({ commitMessage, changedPaths, verificationCommands }, null, 2));
      return;
    }

    await updateStatus({ phase: "commit-push", iteration, commitMessage, changedPaths, unexpectedChanges });
    commitBranch(commitMessage, changedPaths);
    pushBranch();

    const pushedHeadSha = runGit(["rev-parse", "HEAD"]);
    if (SKIP_CI_WAIT) {
      const skipMessage = "Skipped post-push CI wait because AUTOLOOP_SKIP_CI_WAIT is set.";
      await updateStatus({
        phase: "complete",
        iteration,
        outcome: "pushed_skip_ci_wait",
        branch: LOOP_BRANCH,
        headSha: pushedHeadSha,
        changedPaths,
        plan: planSummary,
        automaticRepair: automaticRepair ? summarizeAutomaticRepair(automaticRepair) : undefined,
        message: skipMessage,
      });
      console.log(`${skipMessage} Pushed ${pushedHeadSha} directly to ${LOOP_BRANCH}.`);
      return;
    }

    await updateStatus({
      phase: "ci-wait",
      iteration,
      branch: LOOP_BRANCH,
      headSha: pushedHeadSha,
      changedPaths,
      plan: planSummary,
      automaticRepair: automaticRepair ? summarizeAutomaticRepair(automaticRepair) : undefined,
    });
    const ci = waitForBranchCi(pushedHeadSha, LOOP_BRANCH);
    if (ci.ok) {
      await updateStatus({ phase: "complete", iteration, outcome: "pushed", branch: LOOP_BRANCH, headSha: pushedHeadSha, ci });
      console.log(`Pushed ${pushedHeadSha} directly to ${LOOP_BRANCH}.`);
      return;
    }

    // A red CI run stays inside the same bounded cycle: capture the failed-job
    // logs and feed them back into the next Codex repair prompts.
    failureContext = {
      iteration,
      branchName: LOOP_BRANCH,
      headSha: pushedHeadSha,
      runId: ci.runId,
      runUrl: ci.runUrl,
      failedLog: ci.failedLog,
      changedPaths,
    };
    await updateStatus({
      phase: "repair-needed",
      iteration,
      outcome: "ci_failed",
      branch: LOOP_BRANCH,
      headSha: pushedHeadSha,
      ci,
      failureContext: summarizeFailureContext(failureContext),
    });
  }

  throw new Error(`Autoloop exhausted ${MAX_ITERATIONS} iteration(s) without a green CI result.`);
}

function clampInt(raw, fallback, min, max) {
  const n = Number(raw);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, Math.trunc(n)));
}

function readBooleanEnv(raw) {
  return /^(1|true|yes|on)$/i.test(String(raw ?? "").trim());
}

function resolveCodexReasoningEffort(raw) {
  const value = String(raw ?? "").trim().toLowerCase();
  if (value === "low" || value === "medium" || value === "high" || value === "xhigh") return value;
  return "xhigh";
}

function runCommand(command, args, opts = {}) {
  const capture = opts.capture !== false;
  const cwd = opts.cwd || ROOT;
  const env = { ...process.env, ...(opts.env || {}) };
  const timeoutMs = Number.isFinite(opts.timeoutMs) ? Math.max(0, Math.trunc(opts.timeoutMs)) : 0;
  try {
    const out = execFileSync(command, args, {
      cwd,
      env,
      encoding: "utf8",
      input: opts.input,
      timeout: timeoutMs || undefined,
      stdio: capture ? ["pipe", "pipe", "pipe"] : "inherit",
    });
    return capture ? out.trim() : "";
  } catch (err) {
    const stdout = err?.stdout ? String(err.stdout) : "";
    const stderr = err?.stderr ? String(err.stderr) : "";
    const detail = err?.message ? `\n${err.message}` : "";
    throw new Error(`${command} ${args.join(" ")} failed.${detail}\n${stdout}${stderr}`.trim());
  }
}

function runGit(args, opts) {
  return runCommand("git", args, opts);
}

function buildSanitizedGhAuthEnv(extraEnv = {}) {
  return {
    ...process.env,
    ...extraEnv,
    GH_TOKEN: "",
    GITHUB_TOKEN: "",
    GITHUB_PAT: "",
  };
}

function getStoredGhToken() {
  if (cachedStoredGhToken !== null) {
    return cachedStoredGhToken;
  }

  try {
    cachedStoredGhToken = execFileSync("gh", ["auth", "token"], {
      cwd: ROOT,
      env: buildSanitizedGhAuthEnv(),
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
  } catch {
    cachedStoredGhToken = "";
  }

  return cachedStoredGhToken;
}

function runGh(args, opts = {}) {
  const storedToken = getStoredGhToken();
  const envToken =
    opts.env?.GITHUB_TOKEN ||
    opts.env?.GH_TOKEN ||
    opts.env?.GITHUB_PAT ||
    process.env.GITHUB_TOKEN ||
    process.env.GH_TOKEN ||
    process.env.GITHUB_PAT ||
    "";
  return runCommand("gh", args, {
    ...opts,
    // Prefer the stored gh login when local shells leak a stale GH_TOKEN.
    env: storedToken
      ? buildSanitizedGhAuthEnv({
          ...(opts.env || {}),
          GH_TOKEN: storedToken,
        })
      : {
          ...(opts.env || {}),
          GH_TOKEN: envToken,
        },
  });
}

function runBash(command, opts = {}) {
  return runCommand("/bin/bash", ["-lc", prepareShellCommand(command)], opts);
}

function isRetryableCodexExecError(err) {
  const message = err instanceof Error ? err.message : String(err);
  return /(ETIMEDOUT|ECONNRESET|stream disconnected before completion|idle timeout waiting for websocket|Reconnecting\.\.\.)/i.test(
    message,
  );
}

function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function commandExists(command) {
  try {
    execFileSync("/bin/bash", ["-lc", `command -v ${JSON.stringify(command)} >/dev/null`], {
      cwd: ROOT,
      stdio: ["ignore", "ignore", "ignore"],
    });
    return true;
  } catch {
    return false;
  }
}

function resolveOptionalPath(rawPath) {
  const value = String(rawPath ?? "").trim();
  if (!value) return "";
  return path.isAbsolute(value) ? value : path.join(ROOT, value);
}

async function updateStatus(patch) {
  statusState = {
    ...statusState,
    ...patch,
    updatedAt: new Date().toISOString(),
  };
  if (!STATUS_FILE) return;
  await writeJsonFileAtomic(STATUS_FILE, statusState);
}

function summarizeIdea(idea) {
  if (!idea || idea.noChange) {
    return {
      noChange: true,
      title: idea?.title || "",
      rationale: idea?.rationale || "",
    };
  }
  return {
    title: idea.title,
    rationale: idea.rationale,
    algorithmReviewPath: idea.algorithmReviewPath,
    algorithmReviewFocus: idea.algorithmReviewFocus,
    formalMethodsPath: idea.formalMethodsPath,
    formalMethodsFocus: idea.formalMethodsFocus,
    filesNeeded: idea.filesNeeded,
    verificationCommands: idea.verificationCommands,
  };
}

function summarizePlan(plan) {
  if (!plan || plan.noChange) {
    return {
      noChange: true,
      title: plan?.title || "",
      summary: plan?.summary || "",
    };
  }
  return {
    title: plan.title,
    summary: plan.summary,
    commitMessage: plan.commitMessage,
    algorithmReviewSummary: plan.algorithmReviewSummary,
    formalMethodsSummary: plan.formalMethodsSummary,
    verificationCommands: plan.verificationCommands,
    changes: plan.changes.map((change) => ({
      path: change.path,
      delete: change.delete === true,
      reason: change.reason || "",
    })),
  };
}

function summarizeFailureContext(failureContext) {
  if (!failureContext) return null;
  return {
    iteration: failureContext.iteration,
    branchName: failureContext.branchName,
    headSha: failureContext.headSha,
    runId: failureContext.runId,
    runUrl: failureContext.runUrl,
    changedPaths: failureContext.changedPaths,
  };
}

function summarizeAutomaticRepair(repair) {
  if (!repair) return null;
  return {
    type: repair.type,
    changedPaths: repair.changedPaths,
    commitMessage: repair.commitMessage,
  };
}

function parseFourmoluFailurePaths(failedLog) {
  return uniqueStrings(
    String(failedLog || "")
      .split(/\r?\n/)
      .map((line) => line.match(/\b((?:app|test|bench)\/[A-Za-z0-9_./-]+\.hs)\b/)?.[1] || "")
      .filter(Boolean)
      .map((relPath) => sanitizeRelativePath(`haskell/${relPath}`)),
  );
}

function hasHaskellParserFailure(logText) {
  return /\bThe GHC parser \(in Haddock mode\) failed\b|\bparse error on input\b/.test(String(logText || ""));
}

function deriveParserFailurePaths(...logTexts) {
  return uniqueStrings(
    logTexts
      .map((text) => parseFourmoluFailurePaths(text))
      .flat()
      .filter(allowedEditPath),
  ).sort(compareEditablePaths);
}

function detectAutomaticRepair(failureContext) {
  if (!failureContext?.failedLog) return null;

  const failedLog = String(failureContext.failedLog);
  const changedPaths = uniqueStrings((failureContext.changedPaths || []).map(sanitizeRelativePath));
  const isFourmoluFailure = /\bfourmolu --mode check\b/.test(failedLog);
  if (!isFourmoluFailure) return null;

  const fourmoluPaths = parseFourmoluFailurePaths(failedLog).filter(
    (filePath) => changedPaths.length === 0 || changedPaths.includes(filePath),
  );
  if (fourmoluPaths.length === 0 || !fourmoluPaths.every(allowedEditPath)) return null;

  return {
    type: "fourmolu",
    changedPaths: fourmoluPaths,
    commitMessage: fourmoluPaths.length === 1 ? `Haskell: format ${path.basename(fourmoluPaths[0], ".hs")}` : "Haskell: apply fourmolu fixes",
    verificationCommands: planVerificationCommands(fourmoluPaths, [FOURMOLU_CHECK_COMMAND]),
  };
}

function assertCleanWorktree() {
  const status = runGit(["status", "--porcelain"]);
  if (status) throw new Error(`Worktree must be clean before autoloop starts.\n${status}`);
}

async function checkoutLoopBranch() {
  const loopBranchHead = readRemoteBranchHead(LOOP_BRANCH);
  if (loopBranchHead) {
    fetchRemoteTrackingBranch(LOOP_BRANCH);
    runGit(["checkout", "-B", LOOP_BRANCH, `origin/${LOOP_BRANCH}`], { capture: false });
  } else {
    runGit(["checkout", "-B", LOOP_BRANCH, `origin/${BASE_BRANCH}`], { capture: false });
  }
}

function fetchRemoteTrackingBranch(branchName) {
  runGit(["fetch", "origin", buildRemoteTrackingRefspec(branchName), "--prune"]);
}

function readRemoteBranchHead(branchName) {
  return parseLsRemoteBranchHead(runGit(["ls-remote", "--heads", "origin", branchName]), branchName);
}

function hardResetToCurrentHead() {
  runGit(["reset", "--hard", "HEAD"], { capture: false });
  runGit(["clean", "-fd", "-e", "node_modules/", "-e", "haskell/web/node_modules/"], { capture: false });
}

function allowedEditPath(filePath) {
  const rel = sanitizeRelativePath(filePath);
  if (BLOCKED_EDIT_PREFIXES.some((prefix) => rel === prefix || rel.startsWith(prefix))) return false;
  return ALLOWED_EDIT_PREFIXES.some((prefix) => rel === prefix || rel.startsWith(prefix));
}

function editableFilePriority(filePath) {
  const rel = sanitizeRelativePath(filePath);
  const index = EDITABLE_FILE_PRIORITY_PREFIXES.findIndex((prefix) => rel === prefix || rel.startsWith(prefix));
  return index === -1 ? EDITABLE_FILE_PRIORITY_PREFIXES.length : index;
}

function compareEditablePaths(left, right) {
  const priorityDelta = editableFilePriority(left) - editableFilePriority(right);
  if (priorityDelta !== 0) return priorityDelta;
  return left.localeCompare(right);
}

function getTrackedFiles() {
  if (cachedTrackedFiles !== null) {
    return cachedTrackedFiles;
  }

  cachedTrackedFiles = new Set(runGit(["ls-files"]).split(/\r?\n/).filter(Boolean));
  return cachedTrackedFiles;
}

function resolveFailureReferencedPath(rawPath) {
  const rel = sanitizeRelativePath(String(rawPath || "").trim().replace(/:\d+(?::\d+)?$/, ""));
  if (!rel) return "";

  const candidates = [rel];
  if (/^(app|test|bench|scripts)\//.test(rel) && /\.(hs|lhs|py|sh)$/.test(rel)) {
    candidates.unshift(`haskell/${rel}`);
  }
  if (/^(src|test)\//.test(rel) && /\.(ts|tsx|js|jsx|mjs|cjs)$/.test(rel)) {
    candidates.unshift(`haskell/web/${rel}`);
  }

  const trackedFiles = getTrackedFiles();
  return candidates.map(sanitizeRelativePath).find((candidate) => trackedFiles.has(candidate)) || rel;
}

function parseFailureReferencedPaths(failedLog) {
  const rawMatches = [];
  for (const pattern of [
    /\b((?:[A-Za-z0-9_.-]+\/)+(?:[A-Za-z0-9_.-]+\.[A-Za-z0-9_.-]+))(?::\d+(?::\d+)?)?\b/g,
    /\b(FORMAL_METHODS\.md|README\.md|CHANGELOG\.md|AGENTS\.md|mission\.md)\b/g,
  ]) {
    for (const match of String(failedLog || "").matchAll(pattern)) {
      rawMatches.push(match[1]);
    }
  }

  return uniqueStrings(rawMatches.map(resolveFailureReferencedPath).filter(Boolean).filter(allowedEditPath)).sort(compareEditablePaths);
}

function deriveFailureRepairPaths(failureContext) {
  if (!failureContext) return [];

  const parserFailurePaths = deriveParserFailurePaths(failureContext.failedLog);
  const referencedPaths = parseFailureReferencedPaths(failureContext.failedLog);
  const changedPaths = uniqueStrings((failureContext.changedPaths || []).map(sanitizeRelativePath).filter(allowedEditPath)).sort(
    compareEditablePaths,
  );
  return uniqueStrings([
    ...parserFailurePaths,
    ...referencedPaths,
    ...changedPaths,
  ]);
}

function chooseRepairReviewPath(paths, prefixes, fallbackCandidates = [], preferredCandidates = []) {
  const trackedFiles = getTrackedFiles();
  return uniqueStrings([...preferredCandidates, ...paths, ...fallbackCandidates].map(sanitizeRelativePath))
    .filter((candidate) => candidate && trackedFiles.has(candidate))
    .find(
      (candidate) =>
        allowedEditPath(candidate) &&
        prefixes.some((prefix) => candidate === prefix || candidate.startsWith(prefix)),
    ) || "";
}

function buildFailureRepairIdea(failureContext, failureRepairPaths, automaticRepairFailure = "") {
  if (!failureContext || failureRepairPaths.length === 0) return null;

  const parserFailurePaths = deriveParserFailurePaths(failureContext.failedLog, automaticRepairFailure);
  const syntaxRepairRequired = parserFailurePaths.length > 0 && hasHaskellParserFailure(`${failureContext.failedLog}\n${automaticRepairFailure}`);
  const algorithmReviewPath = chooseRepairReviewPath(
    failureRepairPaths,
    ALGORITHM_REVIEW_PREFIXES,
    ["haskell/app/Main.hs"],
    parserFailurePaths,
  );
  const formalMethodsPath = chooseRepairReviewPath(
    failureRepairPaths,
    FORMAL_METHODS_REVIEW_PREFIXES,
    ["haskell/test/TestMain.hs", "FORMAL_METHODS.md"],
    parserFailurePaths,
  );
  const filesNeeded = uniqueStrings([
    ...(syntaxRepairRequired ? parserFailurePaths : []),
    ...failureRepairPaths,
    algorithmReviewPath,
    formalMethodsPath,
  ].filter(Boolean));
  if (!algorithmReviewPath || !formalMethodsPath || filesNeeded.length === 0) return null;

  return {
    noChange: false,
    title: `Self-heal failing CI on ${failureContext.branchName}`,
    rationale: syntaxRepairRequired
      ? "The failed CI log shows parser-level Haskell errors in editable files, so the loop must restore valid syntax/module structure in those parser-failing artifacts before any formatter-only or unrelated repair."
      : "The failed CI log names editable files, so the loop must attempt a direct repair on those failure-targeted artifacts before declaring noChange or proposing unrelated work.",
    algorithmReviewPath,
    algorithmReviewFocus: syntaxRepairRequired
      ? `Review ${algorithmReviewPath} and restore the smallest valid Haskell syntax/module/import structure needed to clear the parser failure before any behavioral change.`
      : `Review ${algorithmReviewPath} for the smallest safe behavioral or interface change needed to clear the failing CI run.`,
    formalMethodsPath,
    formalMethodsFocus: syntaxRepairRequired
      ? `Align ${formalMethodsPath} with the parser-failing invariant from the CI log, keep module/test structure parseable, and preserve the proof obligation explicitly.`
      : `Align ${formalMethodsPath} with the failing invariant, requirement, or test assertion from the CI log and keep the proof obligation explicit.`,
    filesNeeded,
    verificationCommands: planVerificationCommands(filesNeeded, syntaxRepairRequired ? ["cd haskell && cabal build", FOURMOLU_CHECK_COMMAND] : []),
  };
}

async function listEditableFiles(prioritizedPaths = []) {
  const files = runGit(["ls-files"]).split(/\r?\n/).filter(Boolean);
  const result = [];
  for (const rel of files) {
    if (!allowedEditPath(rel)) continue;
    const abs = path.join(ROOT, rel);
    const stat = await fs.stat(abs);
    if (!stat.isFile()) continue;
    if (stat.size > MAX_EDITABLE_FILE_BYTES) continue;
    result.push({ path: rel, size: stat.size });
  }
  result.sort((left, right) => {
    return compareEditablePaths(left.path, right.path);
  });

  const prioritized = uniqueStrings(prioritizedPaths.map(sanitizeRelativePath));
  const byPath = new Map(result.map((file) => [file.path, file]));
  const ordered = [];
  const seen = new Set();
  for (const rel of prioritized) {
    const file = byPath.get(rel);
    if (!file || seen.has(file.path)) continue;
    seen.add(file.path);
    ordered.push(file);
  }
  for (const file of result) {
    if (seen.has(file.path)) continue;
    seen.add(file.path);
    ordered.push(file);
  }
  return ordered.slice(0, MAX_EDITABLE_FILES);
}

async function readOptionalText(relativePath, maxChars) {
  try {
    const content = await fs.readFile(path.join(ROOT, relativePath), "utf8");
    return typeof maxChars === "number" ? clampText(content, maxChars) : content;
  } catch (err) {
    if (err && typeof err === "object" && "code" in err && err.code === "ENOENT") return "";
    throw err;
  }
}

async function buildRepoContext(failureRepairPaths = []) {
  const editableFiles = await listEditableFiles(failureRepairPaths);
  const agents = await fs.readFile(path.join(ROOT, "AGENTS.md"), "utf8");
  const ciWorkflow = await fs.readFile(path.join(ROOT, ".github/workflows/ci.yml"), "utf8");
  const readme = clampText(await fs.readFile(path.join(ROOT, "README.md"), "utf8"), 14000);
  const changelog = clampText(await fs.readFile(path.join(ROOT, "CHANGELOG.md"), "utf8"), 10000);
  const formal = clampText(await fs.readFile(path.join(ROOT, "FORMAL_METHODS.md"), "utf8"), 8000);
  const objectives = await readOptionalText("objectives/trader.md", 10000);
  const packageJson = await fs.readFile(path.join(ROOT, "package.json"), "utf8");
  const webPackageJson = await fs.readFile(path.join(ROOT, "haskell/web/package.json"), "utf8");
  const editableList = editableFiles.map((file) => `${file.path} (${file.size} bytes)`).join("\n");
  const failureEditableList = editableFiles
    .filter((file) => failureRepairPaths.includes(file.path))
    .map((file) => `${file.path} (${file.size} bytes)`)
    .join("\n");
  const recentCommits = runGit(["log", "--oneline", "-5"]);

  return {
    agents,
    ciWorkflow,
    readme,
    changelog,
    formal,
    objectives,
    packageJson,
    webPackageJson,
    editableList,
    failureEditableList,
    editablePaths: editableFiles.map((file) => file.path),
    recentCommits,
  };
}

function repoContextText(repoContext) {
  return [
    "AGENTS.md:",
    repoContext.agents,
    "",
    "Recent commits:",
    repoContext.recentCommits,
    "",
    "CI workflow:",
    repoContext.ciWorkflow,
    "",
    "Root package.json:",
    repoContext.packageJson,
    "",
    "Web package.json:",
    repoContext.webPackageJson,
    "",
    "README excerpt:",
    repoContext.readme,
    "",
    "CHANGELOG excerpt:",
    repoContext.changelog,
    "",
    "FORMAL_METHODS excerpt:",
    repoContext.formal,
    "",
    repoContext.objectives ? "Trader objectives excerpt:" : "",
    repoContext.objectives || "",
    repoContext.objectives ? "" : "",
    repoContext.failureEditableList ? "Failure-targeted editable files:" : "",
    repoContext.failureEditableList || "",
    repoContext.failureEditableList ? "" : "",
    `Editable files (limited to ${MAX_EDITABLE_FILE_BYTES} bytes each):`,
    repoContext.editableList,
  ].join("\n");
}

async function callModelJson({ prompt, maxOutputTokens = 4000, timeoutMs = CODEX_EXEC_TIMEOUT_MS }) {
  if (PLANNER_BACKEND === "codex") {
    return callModelJsonViaCodex({ prompt, maxOutputTokens, timeoutMs });
  }

  const response = await fetch(`${OPENAI_BASE_URL}/responses`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: OPENAI_MODEL,
      instructions:
        "Return JSON only. The word JSON must appear in the output context, and the final output must be a single valid JSON object with no markdown fences.",
      input: prompt,
      text: { format: { type: "json_object" } },
      temperature: 0.2,
      max_output_tokens: maxOutputTokens,
      store: false,
    }),
  });

  const json = await response.json();
  if (!response.ok) {
    throw buildOpenAiApiError(response.status, json);
  }
  return parseJsonResponse(extractResponseText(json));
}

async function callModelJsonViaCodex({ prompt, maxOutputTokens, timeoutMs }) {
  const input = [
    "Return JSON only. The final response must be a single valid JSON object with no markdown fences.",
    "Use only the prompt contents below. Do not run shell commands, open files, inspect the repository, or use web search.",
    "Do not narrate progress or emit intermediate messages. Reply with the final JSON object immediately.",
    `Treat this max_output_tokens hint as advisory: ${maxOutputTokens}.`,
    prompt,
  ].join("\n\n");

  let lastError = null;
  for (let attempt = 1; attempt <= CODEX_RETRY_MAX_ATTEMPTS; attempt += 1) {
    try {
      const rawEvents = runCommand(
        "codex",
        [
          "exec",
          "--json",
          "--ephemeral",
          "--sandbox",
          "read-only",
          "--color",
          "never",
          "--model",
          OPENAI_MODEL,
          "-c",
          `model_reasoning_effort="${CODEX_REASONING_EFFORT}"`,
          "-",
        ],
        {
          input,
          timeoutMs,
        },
      );
      return parseJsonResponse(extractCodexExecLastMessage(rawEvents));
    } catch (err) {
      lastError = err;
      if (!isRetryableCodexExecError(err) || attempt >= CODEX_RETRY_MAX_ATTEMPTS) throw err;
      const delayMs = CODEX_RETRY_BACKOFF_MS * attempt;
      console.warn(
        `Codex exec transient failure on attempt ${attempt}/${CODEX_RETRY_MAX_ATTEMPTS}; retrying in ${delayMs}ms.`,
      );
      await sleep(delayMs);
    }
  }

  throw lastError ?? new Error("Codex exec failed without an error.");
}

async function requestIdeaSelection(repoContext) {
  const prompt = [
    "You are selecting exactly one safe autonomous improvement for this repository.",
    "Bias strongly toward backend Haskell trading-algorithm improvements and formal-methods-backed changes over UI polish or general maintenance.",
    "Respond in JSON with keys: noChange, title, rationale, algorithmReviewPath, algorithmReviewFocus, formalMethodsPath, formalMethodsFocus, filesNeeded, verificationCommands.",
    "Constraints:",
    "- Choose one small, high-confidence backend Haskell trading-algorithm change with measurable value.",
    "- Every cycle must explicitly cover these phases: choose one valuable backend trading improvement, review one local Haskell algorithm file, review one formal-methods artifact with an explicit invariant/property/proof obligation, then commit/push and wait for GitHub CI.",
    "- Touch only files from the editable file list.",
    "- Do not propose dependency updates, workflow changes, deploy changes, or secrets.",
    "- Prefer changes that can be verified by local Haskell build/test commands.",
    `- algorithmReviewPath must be within ${ALGORITHM_REVIEW_PREFIXES.join(" or ")} and filesNeeded must include it.`,
    `- formalMethodsPath must be within ${FORMAL_METHODS_REVIEW_PREFIXES.join(" or ")} and filesNeeded must include it.`,
    "- Prefer algorithmReviewPath values under haskell/app/Trader/ or the optimize-equity/trader entrypoints when possible.",
    "- Prefer trading logic, signal gates, predictors, optimizer behavior, position/risk management, or market-state inference over web UI changes.",
    "- Prefer ideas where the formal-methods review updates FORMAL_METHODS.md, haskell/app/Trader/Formal/*, or Haskell tests with a concrete invariant/property/proof sketch.",
    "- If the change is user-visible, include README.md and CHANGELOG.md in filesNeeded.",
    `- Verification commands must be chosen from this allowlist: ${Array.from(SAFE_VERIFICATION_COMMANDS).join(" | ")}`,
    "- If no safe backend trading improvement with formal-methods coverage is apparent, return {\"noChange\":true,...}.",
    "",
    repoContextText(repoContext),
  ].join("\n");

  return normalizeIdeaSelection(await callModelJson({ prompt, maxOutputTokens: 2000 }));
}

async function requestFixIdea(repoContext, failureContext, failureRepairPaths = [], automaticRepairFailure = "") {
  const parserFailurePaths = deriveParserFailurePaths(failureContext?.failedLog, automaticRepairFailure);
  const syntaxRepairRequired = parserFailurePaths.length > 0 && hasHaskellParserFailure(`${failureContext?.failedLog || ""}\n${automaticRepairFailure}`);
  const prompt = [
    "You are selecting a repair for a failed autonomous CI run on the repository branch.",
    "Bias strongly toward backend Haskell trading-algorithm fixes with formal-methods-backed coverage unless the failure clearly requires another file to be touched.",
    "Respond in JSON with keys: noChange, title, rationale, algorithmReviewPath, algorithmReviewFocus, formalMethodsPath, formalMethodsFocus, filesNeeded, verificationCommands.",
    "Constraints:",
    "- Focus on fixing the reported failure with the smallest safe change.",
    "- Self-heal is required for any actionable failure or error when the failed log names editable files.",
    "- If the failed log shows parser-level Haskell errors, restore valid syntax, module headers, import/export structure, and declaration shape in the named files before attempting formatter-only cleanup or unrelated semantic edits.",
    "- Still explicitly cover the cycle phases: choose the repair, review one local Haskell algorithm file, review one formal-methods artifact with an explicit invariant/property/proof obligation, then commit/push and wait for GitHub CI.",
    "- Touch only files from the editable file list.",
    "- When the failed log names editable files, filesNeeded must include the smallest relevant subset of those failure-targeted files before unrelated context files.",
    "- When parser-failing files are named, filesNeeded must include those parser-failing files first and may omit unrelated changed files unless they are required to make the named files build and format cleanly.",
    `- algorithmReviewPath must be within ${ALGORITHM_REVIEW_PREFIXES.join(" or ")} and filesNeeded must include it.`,
    `- formalMethodsPath must be within ${FORMAL_METHODS_REVIEW_PREFIXES.join(" or ")} and filesNeeded must include it.`,
    "- Prefer trading logic, signal gates, predictors, optimizer behavior, position/risk management, or market-state inference over UI-only repairs.",
    "- Prefer repairs that also strengthen FORMAL_METHODS.md, haskell/app/Trader/Formal/*, or Haskell tests with a concrete invariant/property/proof sketch.",
    `- Verification commands must be chosen from this allowlist: ${Array.from(SAFE_VERIFICATION_COMMANDS).join(" | ")}`,
    "",
    `Failed branch: ${failureContext.branchName}`,
    `Failed head: ${failureContext.headSha}`,
    `Failed run: ${failureContext.runUrl}`,
    `Changed paths on the current branch: ${failureContext.changedPaths.join(", ")}`,
    `Failure-targeted editable files: ${failureRepairPaths.join(", ") || "(none)"}`,
    syntaxRepairRequired ? `Parser-failing editable files: ${parserFailurePaths.join(", ")}` : "",
    automaticRepairFailure ? `Automatic repair failure: ${clampText(automaticRepairFailure, 4000)}` : "",
    "Failed log excerpt:",
    clampText(failureContext.failedLog, 20000),
    "",
    repoContextText(repoContext),
  ].join("\n");

  return normalizeIdeaSelection(await callModelJson({ prompt, maxOutputTokens: 2200 }));
}

async function readEditableFiles(paths) {
  const out = [];
  for (const rawPath of paths) {
    const rel = sanitizeRelativePath(rawPath);
    if (!allowedEditPath(rel)) throw new Error(`Path is outside the autoloop allowlist: ${rel}`);
    const abs = path.join(ROOT, rel);
    const content = await fs.readFile(abs, "utf8");
    out.push({ path: rel, content });
  }
  return out;
}

async function requestPatchPlan(_repoContext, idea, editableFiles, failureContext) {
  const fileSections = editableFiles
    .map((file) => `FILE: ${file.path}\n<<<FILE\n${file.content}\nFILE;`)
    .join("\n\n");
  const parserFailurePaths = deriveParserFailurePaths(failureContext?.failedLog);
  const syntaxRepairRequired = parserFailurePaths.length > 0 && hasHaskellParserFailure(failureContext?.failedLog || "");

  const promptLines = [
    "You are implementing a single repository change.",
    "Keep the change centered on a backend Haskell trading improvement and its formal-methods coverage.",
    "Use the selected file contents below as the complete source of truth for editing this patch-plan step.",
    "Respond in JSON with keys: noChange, title, summary, commitMessage, algorithmReviewSummary, formalMethodsSummary, verificationCommands, changes.",
    "Each entry in changes must be an object with path, content, and optional reason.",
    "The content field must contain the complete replacement file content for that path.",
    "Do not include markdown fences or prose outside JSON.",
    "Constraints:",
    "- Only modify the provided files.",
    "- Preserve unrelated content.",
    "- Keep the change minimal and focused.",
    "- If the failed CI log shows parser-level Haskell errors, first restore valid Haskell syntax/module/import/test structure in the named files before attempting formatter-only cleanup or broader semantic edits.",
    "- Explicitly complete the required phases inside this plan: the chosen backend algorithm change, a review of the selected Haskell algorithm file, and a formal-methods review with an invariant/property/proof-sketch update.",
    "- algorithmReviewSummary must say what backend algorithm file was reviewed and what algorithmic change or no-change decision followed.",
    "- formalMethodsSummary must name the invariant/property/test or FORMAL_METHODS / Trader.Formal proof sketch that now covers the change.",
    "- If behavior changes, prefer updating FORMAL_METHODS.md, haskell/app/Trader/Formal/*, or Haskell tests within the selected files.",
    "- Prefer verification commands that exercise Haskell build/test coverage for the touched trading logic.",
    "- Use ASCII unless the file already requires Unicode.",
    `- Verification commands must be chosen from this allowlist: ${Array.from(SAFE_VERIFICATION_COMMANDS).join(" | ")}`,
    idea.title ? `Selected idea: ${idea.title}` : "",
    idea.rationale ? `Rationale: ${idea.rationale}` : "",
    idea.algorithmReviewPath ? `Algorithm review file: ${idea.algorithmReviewPath}` : "",
    idea.algorithmReviewFocus ? `Algorithm review focus: ${idea.algorithmReviewFocus}` : "",
    idea.formalMethodsPath ? `Formal methods review file: ${idea.formalMethodsPath}` : "",
    idea.formalMethodsFocus ? `Formal methods review focus: ${idea.formalMethodsFocus}` : "",
    syntaxRepairRequired ? `Parser-failing files that must be made parseable first: ${parserFailurePaths.join(", ")}` : "",
  ].filter(Boolean);

  let failedLogChars = failureContext ? 18000 : 0;
  let prompt = "";
  for (;;) {
    prompt = [
      ...promptLines,
      failureContext ? `Failed CI log excerpt:\n${clampText(failureContext.failedLog, failedLogChars)}` : "",
      "",
      "Editable file contents:",
      fileSections,
    ]
      .filter(Boolean)
      .join("\n");

    if (prompt.length <= PATCH_PLAN_PROMPT_MAX_CHARS || !failureContext || failedLogChars <= 2000) {
      break;
    }
    failedLogChars = Math.max(2000, Math.floor(failedLogChars / 2));
  }

  if (prompt.length > PATCH_PLAN_PROMPT_MAX_CHARS) {
    const fileSizes = editableFiles.map((file) => `${file.path} (${file.content.length} chars)`).join(", ");
    throw new Error(
      `Patch-plan prompt is ${prompt.length} chars, above AUTOLOOP_PATCH_PLAN_MAX_CHARS=${PATCH_PLAN_PROMPT_MAX_CHARS}. Selected files: ${fileSizes}`,
    );
  }

  return normalizePatchPlan(await callModelJson({ prompt, maxOutputTokens: 12000, timeoutMs: CODEX_PATCH_TIMEOUT_MS }));
}

function applyAutomaticRepair(repair) {
  if (repair.type === "fourmolu") {
    const relPaths = repair.changedPaths.map((filePath) => {
      const rel = sanitizeRelativePath(filePath);
      if (!rel.startsWith("haskell/")) throw new Error(`fourmolu repair expected a Haskell path: ${rel}`);
      return rel.slice("haskell/".length);
    });
    runCommand("fourmolu", ["-i", ...relPaths], {
      cwd: path.join(ROOT, "haskell"),
      capture: false,
    });
    return;
  }

  throw new Error(`Unsupported automatic repair type: ${repair.type}`);
}

function applyFileChanges(changes) {
  for (const change of changes) {
    if (!allowedEditPath(change.path)) throw new Error(`Autoloop cannot edit ${change.path}`);
    const abs = path.join(ROOT, change.path);
    if (change.delete) throw new Error(`Autoloop does not allow deletes: ${change.path}`);
    mkdirSync(path.dirname(abs), { recursive: true });
    writeFileSync(abs, change.content, "utf8");
  }
}

function assertPlanMatchesEditableFiles(changes, editableFiles) {
  const allowed = new Set(editableFiles.map((file) => sanitizeRelativePath(file.path)));
  const unexpected = uniqueStrings(
    changes
      .map((change) => sanitizeRelativePath(change.path))
      .filter((filePath) => !allowed.has(filePath)),
  );
  if (unexpected.length > 0) {
    throw new Error(`Patch plan touched files outside the requested context: ${unexpected.join(", ")}`);
  }
}

function listTrackedWorktreeChanges() {
  return uniqueStrings(runGit(["diff", "--name-only"]).split(/\r?\n/).filter(Boolean).map(sanitizeRelativePath));
}

function listUntrackedWorktreeChanges() {
  return uniqueStrings(
    runGit(["ls-files", "--others", "--exclude-standard"])
      .split(/\r?\n/)
      .filter(Boolean)
      .map(sanitizeRelativePath),
  );
}

function collectChangedPlanPaths(plannedPaths) {
  const planned = new Set(plannedPaths.map(sanitizeRelativePath));
  return uniqueStrings([...listTrackedWorktreeChanges(), ...listUntrackedWorktreeChanges()]).filter((filePath) =>
    planned.has(filePath),
  );
}

function collectUnexpectedWorktreeChanges(plannedPaths) {
  const planned = new Set(plannedPaths.map(sanitizeRelativePath));
  return uniqueStrings([...listTrackedWorktreeChanges(), ...listUntrackedWorktreeChanges()]).filter(
    (filePath) => !planned.has(filePath),
  );
}

function planVerificationCommands(changedPaths, suggestedCommands) {
  const planned = new Set();
  const needsWeb = changedPaths.some((file) => file.startsWith("haskell/web/"));
  const needsHsBuild = changedPaths.some((file) => file.startsWith("haskell/app/") || file.startsWith("haskell/test/") || file.startsWith("haskell/scripts/"));
  const needsHsTest = changedPaths.some((file) => file.startsWith("haskell/app/") || file.startsWith("haskell/test/"));

  if (needsWeb) {
    planned.add("cd haskell/web && npm --workspaces=false run typecheck");
    planned.add("cd haskell/web && npm --workspaces=false run test");
    planned.add("cd haskell/web && npm --workspaces=false run build");
  }
  if (needsHsBuild) planned.add("cd haskell && cabal build");
  if (needsHsTest) planned.add("cd haskell && cabal test");

  for (const command of suggestedCommands) {
    const normalized = String(command || "").trim();
    if (SAFE_VERIFICATION_COMMANDS.has(normalized)) planned.add(normalized);
  }
  return Array.from(planned);
}

async function runVerificationCommands(commands) {
  for (const command of commands) {
    console.log(`Running verification: ${command}`);
    runBash(command, { capture: false });
  }
}

function commitBranch(message, changedPaths) {
  runGit(["config", "user.name", "autoloop[bot]"], { capture: false });
  runGit(["config", "user.email", "autoloop[bot]@users.noreply.github.com"], { capture: false });
  runGit(["add", "--", ...changedPaths], { capture: false });
  runGit(["commit", "-m", message], { capture: false });
}

function pushBranch() {
  runGit(["push", "-u", "origin", `${LOOP_BRANCH}:refs/heads/${LOOP_BRANCH}`], { capture: false });
}

function isSuccessfulWorkflowConclusion(conclusion) {
  return conclusion === "success" || conclusion === "neutral" || conclusion === "skipped";
}

function isFailedWorkflowConclusion(conclusion) {
  return Boolean(conclusion) && !isSuccessfulWorkflowConclusion(conclusion);
}

function listCommitChangedPaths(headSha) {
  try {
    const response = JSON.parse(runGh(["api", `repos/:owner/:repo/commits/${headSha}`]));
    return uniqueStrings(
      (Array.isArray(response?.files) ? response.files : [])
        .map((file) => String(file?.filename || "").trim())
        .filter(Boolean),
    );
  } catch {
    return [];
  }
}

function readFailedWorkflowRunLog(runId) {
  try {
    return runGh(["run", "view", String(runId), "--log-failed"]);
  } catch {
    return runGh(["run", "view", String(runId), "--log"]);
  }
}

function collectFailedWorkflowDiagnostics(failedRuns) {
  const logs = failedRuns.map((run) => {
    const runId = run?.id ? String(run.id) : "";
    const name = String(run?.name || run?.display_title || "(unnamed workflow)");
    const url = String(run?.html_url || run?.url || "");
    const failedLog = clampText(readFailedWorkflowRunLog(runId), 18000);
    return [`Workflow: ${name}`, `Run URL: ${url}`, failedLog].filter(Boolean).join("\n");
  });

  return {
    runId: failedRuns[0]?.id ?? null,
    runUrl: failedRuns[0]?.html_url || failedRuns[0]?.url || "",
    failedLog: logs.join("\n\n---\n\n"),
  };
}

function pollGitHubActionsForHead(headSha, branchName, { requireWorkflowRun }) {
  const deadline = Date.now() + CI_DISCOVERY_TIMEOUT_SECONDS * 1000;

  while (true) {
    const runs = listWorkflowRunsForHead(headSha, branchName).filter((run) => run.head_sha === headSha);
    const failedRuns = runs.filter(
      (run) => run.status === "completed" && isFailedWorkflowConclusion(run.conclusion),
    );
    if (failedRuns.length > 0) {
      return {
        ok: false,
        headSha,
        branchName,
        workflowRuns: runs,
        ...collectFailedWorkflowDiagnostics(failedRuns),
      };
    }

    const pendingRuns = runs.filter((run) => run.status !== "completed");
    if (runs.length > 0 && pendingRuns.length === 0) {
      return {
        ok: true,
        headSha,
        branchName,
        workflowRuns: runs,
      };
    }

    if (Date.now() >= deadline) {
      if (!requireWorkflowRun && runs.length === 0) {
        return {
          ok: true,
          headSha,
          branchName,
          workflowRuns: [],
          missing: true,
        };
      }

      const suiteSummary = summarizeCheckSuitesForHead(headSha);
      throw new Error(
        `No completed GitHub Actions workflow run found for branch ${branchName} and head ${headSha} after ${CI_DISCOVERY_TIMEOUT_SECONDS}s.` +
          `${suiteSummary ? ` Check suites: ${suiteSummary}.` : " Check suites: none."}`,
      );
    }

    runCommand("sleep", [String(CI_DISCOVERY_POLL_SECONDS)], { capture: false });
  }
}

function waitForBranchCi(headSha, branchName) {
  return pollGitHubActionsForHead(headSha, branchName, { requireWorkflowRun: true });
}

async function inspectLatestRemoteBranchFailureContext() {
  const latestHeadSha = readRemoteBranchHead(LOOP_BRANCH) || readRemoteBranchHead(BASE_BRANCH);
  if (!latestHeadSha) return null;

  const ci = pollGitHubActionsForHead(latestHeadSha, LOOP_BRANCH, { requireWorkflowRun: false });
  if (ci.ok) return null;

  return {
    iteration: 0,
    branchName: LOOP_BRANCH,
    headSha: latestHeadSha,
    runId: ci.runId,
    runUrl: ci.runUrl,
    failedLog: ci.failedLog,
    changedPaths: listCommitChangedPaths(latestHeadSha),
  };
}

function listWorkflowRunsForHead(headSha, branchName) {
  const response = JSON.parse(runGh(["api", buildActionsRunsApiPath(headSha, branchName, 50)]));
  return Array.isArray(response?.workflow_runs) ? response.workflow_runs : [];
}

function summarizeCheckSuitesForHead(headSha) {
  const response = JSON.parse(runGh(["api", `repos/:owner/:repo/commits/${headSha}/check-suites`]));
  const suites = Array.isArray(response?.check_suites) ? response.check_suites : [];
  return uniqueStrings(
    suites.map((suite) => {
      const app = suite?.app?.slug || suite?.app?.name || "unknown";
      const status = suite?.status || "unknown";
      const conclusion = suite?.conclusion ? `/${suite.conclusion}` : "";
      return `${app}:${status}${conclusion}`;
    }),
  ).join(", ");
}

main().catch(async (err) => {
  if (err?.skipAutoloop) {
    const message = err instanceof Error ? err.message : String(err);
    await updateStatus({ phase: "skipped", outcome: "skipped_openai_api_error", message });
    console.warn(`Autoloop skipped: ${message}`);
    return;
  }
  const message = err instanceof Error ? err.stack || err.message : String(err);
  await updateStatus({ phase: "error", outcome: "error", message });
  console.error(message);
  process.exitCode = 1;
});

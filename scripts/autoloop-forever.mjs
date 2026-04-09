#!/usr/bin/env node

import fs from "node:fs/promises";
import { createWriteStream } from "node:fs";
import path from "node:path";
import process from "node:process";
import { execFileSync, spawn } from "node:child_process";
import {
  buildBranchMergeCandidates,
  normalizeGitBranchShortName,
  parseGitStatusPaths,
  uniqueStrings,
  writeJsonFileAtomic,
} from "./autoloop-lib.mjs";

const ROOT = process.cwd();
const STATE_DIR = path.join(ROOT, ".tmp", "autoloop");
const CYCLES_DIR = path.join(STATE_DIR, "cycles");
const STATUS_FILE = path.join(STATE_DIR, "status.json");
const CURRENT_CYCLE_STATUS_FILE = path.join(STATE_DIR, "current-cycle.json");
const PID_FILE = path.join(STATE_DIR, "runner.pid");
const STOP_FILE = path.join(STATE_DIR, "stop");
const RUNNER_LOG_FILE = path.join(STATE_DIR, "runner.log");
const BASE_BRANCH = normalizeGitBranchShortName(process.env.AUTOLOOP_BASE_BRANCH || "main") || "main";
const LOOP_INTERVAL_SECONDS = clampInt(process.env.AUTOLOOP_FOREVER_INTERVAL_SECONDS, 300, 15, 86400);
const STOP_POLL_SECONDS = clampInt(process.env.AUTOLOOP_FOREVER_STOP_POLL_SECONDS, 5, 1, 60);
const STATUS_HEARTBEAT_SECONDS = clampInt(process.env.AUTOLOOP_FOREVER_STATUS_HEARTBEAT_SECONDS, 15, 5, 300);
const CHILD_ARGS = process.argv.slice(2);

let runnerState = {
  mode: "forever",
  pid: process.pid,
  state: "starting",
  intervalSeconds: LOOP_INTERVAL_SECONDS,
  stopPollSeconds: STOP_POLL_SECONDS,
  statusHeartbeatSeconds: STATUS_HEARTBEAT_SECONDS,
  baseBranch: BASE_BRANCH,
  root: ROOT,
  pidFile: relativePath(PID_FILE),
  stopFile: relativePath(STOP_FILE),
  runnerLogFile: relativePath(RUNNER_LOG_FILE),
  statusFile: relativePath(STATUS_FILE),
  currentCycleStatusFile: relativePath(CURRENT_CYCLE_STATUS_FILE),
  startedAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
  cycleCount: 0,
  childArgs: CHILD_ARGS,
};

let shutdownRequest = null;
let activeChild = null;
let statusHeartbeatTimer = null;

async function main() {
  await ensureStateDir();
  await ensureSingleRunner();
  await clearLaunchArtifacts();
  await fs.writeFile(PID_FILE, `${process.pid}\n`, "utf8");
  installSignalHandlers();
  startStatusHeartbeat();
  await logRunner(`started persistent runner with interval=${LOOP_INTERVAL_SECONDS}s`);
  await updateRunnerStatus({ state: "idle", heartbeatAt: new Date().toISOString() });

  try {
    while (true) {
      if (await syncStopFileState()) break;

      const block = await detectPreflightBlock();
      if (block) {
        await logRunner(`preflight blocked: ${block.reason}`);
        await updateRunnerStatus({
          state: "blocked",
          blockReason: block.reason,
          blockDetails: block.details,
          nextRunAt: futureIso(LOOP_INTERVAL_SECONDS),
        });
        if (await sleepWithStopPolling(LOOP_INTERVAL_SECONDS)) break;
        continue;
      }

      const branchSweep = await reconcileUnmergedBranchesOntoBaseBranch();
      if (!branchSweep.ok) {
        await logRunner(`branch reconciliation blocked: ${branchSweep.reason}`);
        await updateRunnerStatus({
          state: "blocked",
          blockReason: branchSweep.reason,
          blockDetails: branchSweep.details ?? [],
          lastBranchSweep: branchSweep.summary ?? null,
          nextRunAt: futureIso(LOOP_INTERVAL_SECONDS),
        });
        if (await sleepWithStopPolling(LOOP_INTERVAL_SECONDS)) break;
        continue;
      }

      const cycleIndex = runnerState.cycleCount + 1;
      const cycleStamp = new Date().toISOString().replace(/[:.]/g, "-");
      const cycleLogFile = path.join(CYCLES_DIR, `cycle-${String(cycleIndex).padStart(4, "0")}-${cycleStamp}.log`);
      await fs.writeFile(CURRENT_CYCLE_STATUS_FILE, "", "utf8");
      await updateRunnerStatus({
        state: "running",
        cycleCount: cycleIndex,
        lastBranchSweep: branchSweep.summary ?? null,
        currentLogFile: relativePath(cycleLogFile),
        nextRunAt: null,
        blockReason: null,
        blockDetails: null,
      });
      await logRunner(`cycle ${cycleIndex} starting -> ${relativePath(cycleLogFile)}`);

      const result = await runBoundedCycle(cycleIndex, cycleLogFile);
      const cycleStatus = await readJsonIfPresent(CURRENT_CYCLE_STATUS_FILE);
      const dirtyRecovery = await tryAutoSnapshotDirtyCycle();
      await logRunner(
        `cycle ${cycleIndex} finished code=${result.exitCode ?? "null"} signal=${result.signal ?? "none"} outcome=${cycleStatus?.outcome ?? "unknown"} recovery=${dirtyRecovery?.recovered ? dirtyRecovery.branch : "none"}`,
      );

      await updateRunnerStatus({
        state: shutdownRequest ? "stopping" : "sleeping",
        currentLogFile: relativePath(cycleLogFile),
        lastCycle: {
          index: cycleIndex,
          startedAt: result.startedAt,
          endedAt: result.endedAt,
          exitCode: result.exitCode,
          signal: result.signal,
          logFile: relativePath(cycleLogFile),
          phase: cycleStatus?.phase ?? null,
          outcome: cycleStatus?.outcome ?? null,
          message: cycleStatus?.message ?? null,
        },
        recovery: dirtyRecovery
          ? {
              recovered: dirtyRecovery.recovered,
              branch: dirtyRecovery.branch ?? null,
              commit: dirtyRecovery.commit ?? null,
              reason: dirtyRecovery.reason ?? null,
              paths: dirtyRecovery.paths ?? [],
            }
          : null,
        nextRunAt: shutdownRequest ? null : futureIso(LOOP_INTERVAL_SECONDS),
      });

      if (await syncStopFileState()) break;
      if (await sleepWithStopPolling(LOOP_INTERVAL_SECONDS)) break;
    }
  } finally {
    stopStatusHeartbeat();
    const shutdown = shutdownRequest || {
      reason: "completed",
      requestedAt: new Date().toISOString(),
    };
    await updateRunnerStatus({
      state: "stopped",
      nextRunAt: null,
      shutdown,
      finishedAt: new Date().toISOString(),
    });
    await logRunner(`stopped (${shutdown.reason})`);
    await safeUnlink(PID_FILE);
  }
}

function clampInt(raw, fallback, min, max) {
  const n = Number(raw);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, Math.trunc(n)));
}

function relativePath(target) {
  return path.relative(ROOT, target) || ".";
}

function futureIso(seconds) {
  return new Date(Date.now() + seconds * 1000).toISOString();
}

async function ensureStateDir() {
  await fs.mkdir(CYCLES_DIR, { recursive: true });
}

async function clearLaunchArtifacts() {
  await safeUnlink(STOP_FILE);
}

async function ensureSingleRunner() {
  const rawPid = await fs.readFile(PID_FILE, "utf8").catch(() => "");
  const pid = Number.parseInt(rawPid.trim(), 10);
  if (!Number.isFinite(pid) || pid <= 0) return;
  try {
    process.kill(pid, 0);
    throw new Error(`Autoloop runner is already active with PID ${pid}.`);
  } catch (err) {
    if (err && typeof err === "object" && "code" in err && err.code === "ESRCH") return;
    if (err instanceof Error) throw err;
    throw new Error(String(err));
  }
}

async function updateRunnerStatus(patch) {
  runnerState = {
    ...runnerState,
    ...patch,
    updatedAt: new Date().toISOString(),
  };
  await writeJsonFileAtomic(STATUS_FILE, runnerState);
}

function startStatusHeartbeat() {
  stopStatusHeartbeat();
  statusHeartbeatTimer = setInterval(() => {
    void updateRunnerStatus({ heartbeatAt: new Date().toISOString(), pid: process.pid });
  }, STATUS_HEARTBEAT_SECONDS * 1000);
  statusHeartbeatTimer.unref?.();
}

function stopStatusHeartbeat() {
  if (!statusHeartbeatTimer) return;
  clearInterval(statusHeartbeatTimer);
  statusHeartbeatTimer = null;
}

async function logRunner(message) {
  const line = `[${new Date().toISOString()}] ${message}\n`;
  await fs.appendFile(RUNNER_LOG_FILE, line, "utf8");
}

async function detectPreflightBlock() {
  const requestedBackend = String(process.env.AUTOLOOP_BACKEND || "auto").trim().toLowerCase();
  const hasOpenAiKey = Boolean(process.env.OPENAI_API_KEY);
  const hasCodex = commandExists("codex");
  const backendAvailable =
    requestedBackend === "openai" || requestedBackend === "responses"
      ? hasOpenAiKey
      : requestedBackend === "codex"
        ? hasCodex
        : hasOpenAiKey || hasCodex;

  if (!backendAvailable) {
    return {
      reason:
        requestedBackend && requestedBackend !== "auto"
          ? `requested autoloop backend \"${requestedBackend}\" is unavailable; waiting before the next bounded cycle`
          : "neither OPENAI_API_KEY nor Codex CLI is available; waiting before the next bounded cycle",
      details: [],
    };
  }

  let status = runCommand("git", ["status", "--porcelain"], { trimOutput: false });
  if (!status) return null;

  const dirtyRecovery = await tryAutoSnapshotDirtyCycle();
  status = runCommand("git", ["status", "--porcelain"], { trimOutput: false });
  if (!status) return null;

  const dirtyCheckpoint = await tryAutoCheckpointDirtyWorktree();
  status = runCommand("git", ["status", "--porcelain"], { trimOutput: false });
  if (status) {
    return {
      reason:
        dirtyCheckpoint && dirtyCheckpoint.recovered
          ? "dirty worktree persisted after auto-checkpoint recovery; waiting for operator cleanup before bounded autoloop runs"
          : dirtyRecovery && dirtyRecovery.recovered
            ? "dirty worktree persisted after auto-snapshot recovery; waiting for operator cleanup before bounded autoloop runs"
            : "dirty worktree; waiting for operator cleanup before bounded autoloop runs",
      details: status.split(/\r?\n/).filter(Boolean).slice(0, 40),
    };
  }

  return null;
}

function splitNonEmptyLines(raw) {
  return String(raw ?? "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
}

function listUnmergedConflictPaths() {
  return uniqueStrings(runCommand("git", ["diff", "--name-only", "--diff-filter=U"], { trimOutput: false }).split(/\r?\n/))
    .map((line) => line.trim())
    .filter(Boolean);
}

function buildMergeCommitMessage(shortName = "", branchRef = "") {
  if (shortName === BASE_BRANCH || branchRef === `origin/${BASE_BRANCH}`) {
    return `autoloop: sync ${BASE_BRANCH} with origin/${BASE_BRANCH}`;
  }
  return `autoloop: merge ${shortName || branchRef} into ${BASE_BRANCH}`;
}

function buildConflictResolutionCommitMessage(shortName = "", branchRef = "") {
  if (shortName === BASE_BRANCH || branchRef === `origin/${BASE_BRANCH}`) {
    return `autoloop: resolve ${BASE_BRANCH} sync conflicts`;
  }
  return `autoloop: resolve conflicts while merging ${shortName || branchRef} into ${BASE_BRANCH}`;
}

function mergeRefOntoBaseBranch(branchRef, shortName = normalizeGitBranchShortName(branchRef)) {
  const headBefore = runCommand("git", ["rev-parse", "HEAD"]);
  const mergeMessage = buildMergeCommitMessage(shortName, branchRef);
  const conflictMessage = buildConflictResolutionCommitMessage(shortName, branchRef);

  try {
    runCommand("git", ["merge", "--no-ff", "-m", mergeMessage, branchRef], { capture: false });
  } catch (err) {
    const conflicts = listUnmergedConflictPaths();
    if (conflicts.length === 0) throw err;

    // Conflict resolution stays fail-closed toward the current main branch.
    runCommand("git", ["restore", "--source=HEAD", "--staged", "--worktree", "--", ...conflicts], { capture: false });
    runCommand("git", ["commit", "-m", conflictMessage], { capture: false });

    return {
      outcome: "conflict-resolved",
      ref: branchRef,
      shortName,
      headBefore,
      headAfter: runCommand("git", ["rev-parse", "HEAD"]),
      conflicts,
    };
  }

  const headAfter = runCommand("git", ["rev-parse", "HEAD"]);
  return {
    outcome: headAfter === headBefore ? "noop" : "merged",
    ref: branchRef,
    shortName,
    headBefore,
    headAfter,
    conflicts: [],
  };
}

function syncBaseBranchToOrigin() {
  const remoteRef = `origin/${BASE_BRANCH}`;
  const localHead = runCommand("git", ["rev-parse", "HEAD"]);
  const remoteHead = runCommand("git", ["rev-parse", remoteRef]);
  if (localHead === remoteHead) {
    return {
      outcome: "noop",
      ref: remoteRef,
      shortName: BASE_BRANCH,
      headBefore: localHead,
      headAfter: localHead,
      conflicts: [],
    };
  }
  return mergeRefOntoBaseBranch(remoteRef, BASE_BRANCH);
}

function pushBaseBranchWithRetry() {
  try {
    runCommand("git", ["push", "origin", `${BASE_BRANCH}:refs/heads/${BASE_BRANCH}`], { capture: false });
    return { pushed: true, retried: false, retrySync: null };
  } catch {
    runCommand("git", ["fetch", "origin", "--prune"], { capture: false });
    const retrySync = syncBaseBranchToOrigin();
    runCommand("git", ["push", "origin", `${BASE_BRANCH}:refs/heads/${BASE_BRANCH}`], { capture: false });
    return { pushed: true, retried: true, retrySync };
  }
}

function isIgnorablePruneError(message) {
  const text = String(message ?? "");
  return (
    /remote ref does not exist/i.test(text) ||
    /remote ref not found/i.test(text) ||
    /unable to delete .*remote ref does not exist/i.test(text) ||
    /branch .* not found/i.test(text)
  );
}

function pruneMergedRefsOnBaseBranch(baseBranch) {
  const localBranches = splitNonEmptyLines(
    runCommand("git", ["branch", "--format=%(refname:short)", "--merged", baseBranch], { trimOutput: false }),
  );
  const remoteBranches = splitNonEmptyLines(
    runCommand("git", ["branch", "-r", "--format=%(refname:short)", "--merged", baseBranch], { trimOutput: false }),
  );
  const candidates = buildBranchMergeCandidates({ localBranches, remoteBranches, baseBranch });
  const prunedLocalBranches = [];
  const prunedRemoteBranches = [];
  const pruneErrors = [];

  for (const candidate of candidates) {
    if (candidate.remoteRef) {
      try {
        runCommand("git", ["push", "origin", "--delete", candidate.shortName], { capture: false });
        prunedRemoteBranches.push(candidate.shortName);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        if (!isIgnorablePruneError(message)) {
          pruneErrors.push({ branch: candidate.shortName, scope: "remote", error: message });
        }
      }
    }

    if (candidate.localRef) {
      try {
        runCommand("git", ["branch", "-D", candidate.shortName], { capture: false });
        prunedLocalBranches.push(candidate.shortName);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        if (!isIgnorablePruneError(message)) {
          pruneErrors.push({ branch: candidate.shortName, scope: "local", error: message });
        }
      }
    }
  }

  if (prunedRemoteBranches.length > 0) {
    runCommand("git", ["fetch", "origin", "--prune"], { capture: false });
  }

  return {
    candidateBranches: candidates.map((candidate) => candidate.shortName),
    prunedLocalBranches,
    prunedRemoteBranches,
    pruneErrors,
  };
}

async function reconcileUnmergedBranchesOntoBaseBranch() {
  const startedAt = new Date().toISOString();

  try {
    runCommand("git", ["fetch", "origin", "--prune"], { capture: false });
    const currentBranch = normalizeGitBranchShortName(runCommand("git", ["branch", "--show-current"])) || BASE_BRANCH;
    if (currentBranch !== BASE_BRANCH) {
      runCommand("git", ["checkout", BASE_BRANCH], { capture: false });
    }

    const originSync = syncBaseBranchToOrigin();
    const localBranches = splitNonEmptyLines(
      runCommand("git", ["branch", "--format=%(refname:short)", "--no-merged", BASE_BRANCH], { trimOutput: false }),
    );
    const remoteBranches = splitNonEmptyLines(
      runCommand("git", ["branch", "-r", "--format=%(refname:short)", "--no-merged", BASE_BRANCH], { trimOutput: false }),
    );
    const candidates = buildBranchMergeCandidates({ localBranches, remoteBranches, baseBranch: BASE_BRANCH });

    const mergedBranches = [];
    const conflictResolvedBranches = [];

    for (const candidate of candidates) {
      const mergeResult = mergeRefOntoBaseBranch(candidate.ref, candidate.shortName);
      if (mergeResult.outcome === "merged") {
        mergedBranches.push(candidate.shortName);
        continue;
      }
      if (mergeResult.outcome === "conflict-resolved") {
        conflictResolvedBranches.push({
          branch: candidate.shortName,
          conflicts: mergeResult.conflicts,
        });
      }
    }

    const shouldPush =
      originSync.outcome !== "noop" || mergedBranches.length > 0 || conflictResolvedBranches.length > 0;
    const pushResult = shouldPush ? pushBaseBranchWithRetry() : { pushed: false, retried: false, retrySync: null };
    const pruneResult = pruneMergedRefsOnBaseBranch(BASE_BRANCH);
    const summary = {
      startedAt,
      endedAt: new Date().toISOString(),
      baseBranch: BASE_BRANCH,
      syncedOrigin: originSync.outcome,
      candidateBranches: candidates.map((candidate) => candidate.shortName),
      mergedBranches,
      conflictResolvedBranches,
      pushed: pushResult.pushed,
      pushRetried: pushResult.retried,
      retrySyncOutcome: pushResult.retrySync?.outcome ?? null,
      pruneCandidateBranches: pruneResult.candidateBranches,
      prunedLocalBranches: pruneResult.prunedLocalBranches,
      prunedRemoteBranches: pruneResult.prunedRemoteBranches,
      pruneErrors: pruneResult.pruneErrors,
      head: runCommand("git", ["rev-parse", "HEAD"]),
    };

    if (pruneResult.pruneErrors.length > 0) {
      return {
        ok: false,
        reason: `merged ref pruning on ${BASE_BRANCH} failed`,
        details: pruneResult.pruneErrors.map((item) => `${item.scope}:${item.branch}: ${item.error}`).slice(0, 40),
        summary,
      };
    }

    if (mergedBranches.length > 0 || conflictResolvedBranches.length > 0) {
      await logRunner(
        `branch reconciliation merged ${mergedBranches.length + conflictResolvedBranches.length} branch(es) into ${BASE_BRANCH}`,
      );
    }
    if (pruneResult.prunedLocalBranches.length > 0 || pruneResult.prunedRemoteBranches.length > 0) {
      await logRunner(
        `branch reconciliation pruned ${pruneResult.prunedLocalBranches.length} local and ${pruneResult.prunedRemoteBranches.length} remote merged ref(s)`,
      );
    } else if (originSync.outcome !== "noop" && pushResult.pushed) {
      await logRunner(`branch reconciliation refreshed ${BASE_BRANCH} from origin/${BASE_BRANCH}`);
    }

    return { ok: true, summary };
  } catch (err) {
    try {
      runCommand("git", ["merge", "--abort"], { capture: false });
    } catch {}

    const message = err instanceof Error ? err.message : String(err);
    return {
      ok: false,
      reason: `branch reconciliation onto ${BASE_BRANCH} failed`,
      details: splitNonEmptyLines(message).slice(0, 40),
      summary: {
        startedAt,
        endedAt: new Date().toISOString(),
        baseBranch: BASE_BRANCH,
        error: message,
      },
    };
  }
}

function runCommand(command, args, opts = {}) {
  const capture = opts.capture !== false;
  const trimOutput = opts.trimOutput !== false;
  try {
    const out = execFileSync(command, args, {
      cwd: ROOT,
      encoding: "utf8",
      stdio: capture ? ["ignore", "pipe", "pipe"] : "inherit",
    });
    return capture ? (trimOutput ? out.trim() : out) : "";
  } catch (err) {
    const stdout = err?.stdout ? String(err.stdout) : "";
    const stderr = err?.stderr ? String(err.stderr) : "";
    throw new Error(`${command} ${args.join(" ")} failed.\n${stdout}${stderr}`.trim());
  }
}

function pushHeadToBaseBranchWithRetry() {
  try {
    runCommand("git", ["push", "origin", `HEAD:refs/heads/${BASE_BRANCH}`], { capture: false });
    return { pushed: true, retried: false, retrySync: null };
  } catch {
    runCommand("git", ["fetch", "origin", "--prune"], { capture: false });
    const retrySync = syncBaseBranchToOrigin();
    runCommand("git", ["push", "origin", `HEAD:refs/heads/${BASE_BRANCH}`], { capture: false });
    return { pushed: true, retried: true, retrySync };
  }
}

async function tryAutoSnapshotDirtyCycle() {
  const dirtyStatus = runCommand("git", ["status", "--porcelain"], { trimOutput: false });
  const dirtyPaths = parseGitStatusPaths(dirtyStatus);
  if (dirtyPaths.length === 0) return null;

  const cycleStatus = await readJsonIfPresent(CURRENT_CYCLE_STATUS_FILE);
  const changedPaths = Array.isArray(cycleStatus?.changedPaths)
    ? cycleStatus.changedPaths.map((value) => String(value))
    : [];
  if (cycleStatus?.phase !== "error" || changedPaths.length === 0) {
    return {
      recovered: false,
      reason: "dirty worktree is not a recoverable failed autoloop cycle",
      paths: dirtyPaths,
    };
  }

  const expected = new Set(changedPaths);
  const exactlyMatchesFailedCycle =
    dirtyPaths.length === expected.size && dirtyPaths.every((item) => expected.has(item));
  if (!exactlyMatchesFailedCycle) {
    return {
      recovered: false,
      reason: "dirty worktree does not exactly match the last failed cycle changedPaths",
      paths: dirtyPaths,
    };
  }

  const commitMessage = `WIP: save failed autoloop cycle ${cycleStatus?.runId || "cycle"}`;

  try {
    runCommand("git", ["config", "user.name", "autoloop[bot]"], { capture: false });
    runCommand("git", ["config", "user.email", "autoloop[bot]@users.noreply.github.com"], { capture: false });
    runCommand("git", ["add", "--", ...dirtyPaths], { capture: false });
    runCommand("git", ["commit", "-m", commitMessage], { capture: false });
    const commit = runCommand("git", ["rev-parse", "HEAD"]);
    const pushResult = pushHeadToBaseBranchWithRetry();
    await writeJsonFileAtomic(CURRENT_CYCLE_STATUS_FILE, {
      ...(cycleStatus || {}),
      recoveryBranch: BASE_BRANCH,
      recoveryMode: "direct-main",
      recoveryCommit: commit,
      recoveredDirtyPaths: dirtyPaths,
      recoveredAt: new Date().toISOString(),
      recoveryPushed: pushResult.pushed,
      recoveryPushRetried: pushResult.retried,
      recoveryPushRetrySync: pushResult.retrySync,
    });
    await logRunner(
      pushResult.retried
        ? `auto-committed failed dirty cycle directly to ${BASE_BRANCH} (${commit}) after syncing origin/${BASE_BRANCH}`
        : `auto-committed failed dirty cycle directly to ${BASE_BRANCH} (${commit})`,
    );
    return {
      recovered: true,
      branch: BASE_BRANCH,
      commit,
      pushed: pushResult.pushed,
      retried: pushResult.retried,
      retrySync: pushResult.retrySync,
      paths: dirtyPaths,
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    await logRunner(`auto-snapshot recovery failed: ${message}`);
    return {
      recovered: false,
      reason: message,
      paths: dirtyPaths,
    };
  }
}

function analyzeDirtyWorktree(paths, currentBranch) {
  const segments = paths
    .map((item) => String(item).split("/")[0] || String(item))
    .filter(Boolean);
  const topScopes = Array.from(new Set(segments)).slice(0, 4);
  const scopeSummary = topScopes.length > 0 ? topScopes.join(", ") : "misc";
  return {
    commitMessage: "chore(autoloop): checkpoint dirty worktree",
    summary: `Checkpoint dirty worktree from ${currentBranch || "main"} covering ${scopeSummary}.`,
  };
}

async function tryAutoCheckpointDirtyWorktree() {
  const dirtyStatus = runCommand("git", ["status", "--porcelain"], { trimOutput: false });
  const dirtyPaths = parseGitStatusPaths(dirtyStatus);
  if (dirtyPaths.length === 0) return null;

  const currentBranch = runCommand("git", ["branch", "--show-current"]) || BASE_BRANCH;
  const analysis = analyzeDirtyWorktree(dirtyPaths, currentBranch);
  const commitBody = [
    analysis.summary,
    "",
    "Dirty worktree analysis:",
    `- source branch: ${currentBranch}`,
    `- paths: ${dirtyPaths.join(", ")}`,
  ].join("\n");
  try {
    runCommand("git", ["config", "user.name", "autoloop[bot]"], { capture: false });
    runCommand("git", ["config", "user.email", "autoloop[bot]@users.noreply.github.com"], { capture: false });
    runCommand("git", ["add", "-A"], { capture: false });
    runCommand("git", ["commit", "-m", analysis.commitMessage, "-m", commitBody], { capture: false });
    const commit = runCommand("git", ["rev-parse", "HEAD"]);
    const pushResult = pushHeadToBaseBranchWithRetry();
    await logRunner(
      pushResult.retried
        ? `auto-committed dirty worktree directly to ${BASE_BRANCH} (${commit}) after syncing origin/${BASE_BRANCH}`
        : `auto-committed dirty worktree directly to ${BASE_BRANCH} (${commit})`,
    );
    return {
      recovered: true,
      branch: BASE_BRANCH,
      commit,
      pushed: pushResult.pushed,
      retried: pushResult.retried,
      retrySync: pushResult.retrySync,
      paths: dirtyPaths,
      commitMessage: analysis.commitMessage,
      reason: analysis.summary,
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    await logRunner(`auto-checkpoint failed: ${message}`);
    return {
      recovered: false,
      reason: message,
      paths: dirtyPaths,
      commitMessage: analysis.commitMessage,
    };
  }
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

async function runBoundedCycle(cycleIndex, cycleLogFile) {
  const startedAt = new Date().toISOString();
  const logStream = createWriteStream(cycleLogFile, { flags: "a" });
  logStream.write(`[${startedAt}] cycle ${cycleIndex} starting\n`);

  return await new Promise((resolve, reject) => {
    const child = spawn(process.execPath, ["scripts/autoloop.mjs", ...CHILD_ARGS], {
      cwd: ROOT,
      env: {
        ...process.env,
        AUTOLOOP_RUN_MODE: "bounded",
        AUTOLOOP_RUN_ID: `cycle-${cycleIndex}`,
        AUTOLOOP_STATUS_FILE: CURRENT_CYCLE_STATUS_FILE,
      },
      stdio: ["ignore", "pipe", "pipe"],
    });

    activeChild = { child, cycleIndex };

    const stopTimer = setInterval(() => {
      void syncStopFileState();
    }, STOP_POLL_SECONDS * 1000);

    child.stdout.on("data", (chunk) => {
      logStream.write(chunk);
    });
    child.stderr.on("data", (chunk) => {
      logStream.write(chunk);
    });
    child.on("error", (err) => {
      clearInterval(stopTimer);
      activeChild = null;
      logStream.end();
      reject(err);
    });
    child.on("close", (exitCode, signal) => {
      clearInterval(stopTimer);
      activeChild = null;
      const endedAt = new Date().toISOString();
      logStream.write(`[${endedAt}] cycle ${cycleIndex} finished code=${exitCode ?? "null"} signal=${signal ?? "none"}\n`);
      logStream.end();
      resolve({ startedAt, endedAt, exitCode, signal });
    });
  });
}

async function readJsonIfPresent(filePath) {
  const raw = await fs.readFile(filePath, "utf8").catch(() => "");
  const trimmed = raw.trim();
  if (!trimmed) return null;
  try {
    return JSON.parse(trimmed);
  } catch {
    return null;
  }
}

async function stopFileExists() {
  try {
    await fs.access(STOP_FILE);
    return true;
  } catch {
    return false;
  }
}

async function syncStopFileState() {
  if (!(await stopFileExists())) return Boolean(shutdownRequest);
  requestShutdown("stop file requested by operator");
  return true;
}

function requestShutdown(reason, signalName = "SIGTERM") {
  if (shutdownRequest) return;
  shutdownRequest = {
    reason,
    signal: signalName,
    requestedAt: new Date().toISOString(),
  };
  void fs.writeFile(STOP_FILE, `${shutdownRequest.requestedAt} ${reason}\n`, "utf8");
  void updateRunnerStatus({ state: "stopping", shutdown: shutdownRequest, nextRunAt: null });
  void logRunner(`shutdown requested (${reason})`);

  if (!activeChild) return;
  activeChild.child.kill(signalName);
  setTimeout(() => {
    if (!activeChild) return;
    activeChild.child.kill("SIGKILL");
  }, 30000).unref();
}

function installSignalHandlers() {
  process.on("SIGINT", () => {
    requestShutdown("received SIGINT", "SIGINT");
  });
  process.on("SIGTERM", () => {
    requestShutdown("received SIGTERM", "SIGTERM");
  });
}

async function sleepWithStopPolling(seconds) {
  const deadline = Date.now() + seconds * 1000;
  while (Date.now() < deadline) {
    if (await syncStopFileState()) return true;
    const remainingMs = deadline - Date.now();
    if (remainingMs <= 0) break;
    await wait(Math.min(remainingMs, STOP_POLL_SECONDS * 1000));
  }
  return await syncStopFileState();
}

function wait(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

async function safeUnlink(filePath) {
  await fs.unlink(filePath).catch(() => {});
}

main().catch(async (err) => {
  const message = err instanceof Error ? err.stack || err.message : String(err);
  await updateRunnerStatus({
    state: "error",
    nextRunAt: null,
    error: message,
    finishedAt: new Date().toISOString(),
  }).catch(() => {});
  await logRunner(`fatal error: ${message}`).catch(() => {});
  console.error(message);
  process.exitCode = 1;
});

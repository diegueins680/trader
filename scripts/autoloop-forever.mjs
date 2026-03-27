#!/usr/bin/env node

import fs from "node:fs/promises";
import { createWriteStream } from "node:fs";
import path from "node:path";
import process from "node:process";
import { execFileSync, spawn } from "node:child_process";
import { writeJsonFileAtomic } from "./autoloop-lib.mjs";

const ROOT = process.cwd();
const STATE_DIR = path.join(ROOT, ".tmp", "autoloop");
const CYCLES_DIR = path.join(STATE_DIR, "cycles");
const STATUS_FILE = path.join(STATE_DIR, "status.json");
const CURRENT_CYCLE_STATUS_FILE = path.join(STATE_DIR, "current-cycle.json");
const PID_FILE = path.join(STATE_DIR, "runner.pid");
const STOP_FILE = path.join(STATE_DIR, "stop");
const RUNNER_LOG_FILE = path.join(STATE_DIR, "runner.log");
const LOOP_INTERVAL_SECONDS = clampInt(process.env.AUTOLOOP_FOREVER_INTERVAL_SECONDS, 300, 15, 86400);
const STOP_POLL_SECONDS = clampInt(process.env.AUTOLOOP_FOREVER_STOP_POLL_SECONDS, 5, 1, 60);
const CHILD_ARGS = process.argv.slice(2);

let runnerState = {
  mode: "forever",
  pid: process.pid,
  state: "starting",
  intervalSeconds: LOOP_INTERVAL_SECONDS,
  stopPollSeconds: STOP_POLL_SECONDS,
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

async function main() {
  await ensureStateDir();
  await ensureSingleRunner();
  await clearLaunchArtifacts();
  await fs.writeFile(PID_FILE, `${process.pid}\n`, "utf8");
  installSignalHandlers();
  await logRunner(`started persistent runner with interval=${LOOP_INTERVAL_SECONDS}s`);
  await updateRunnerStatus({ state: "idle" });

  try {
    while (true) {
      if (await syncStopFileState()) break;

      const block = detectPreflightBlock();
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

      const cycleIndex = runnerState.cycleCount + 1;
      const cycleStamp = new Date().toISOString().replace(/[:.]/g, "-");
      const cycleLogFile = path.join(CYCLES_DIR, `cycle-${String(cycleIndex).padStart(4, "0")}-${cycleStamp}.log`);
      await fs.writeFile(CURRENT_CYCLE_STATUS_FILE, "", "utf8");
      await updateRunnerStatus({
        state: "running",
        cycleCount: cycleIndex,
        currentLogFile: relativePath(cycleLogFile),
        nextRunAt: null,
        blockReason: null,
        blockDetails: null,
      });
      await logRunner(`cycle ${cycleIndex} starting -> ${relativePath(cycleLogFile)}`);

      const result = await runBoundedCycle(cycleIndex, cycleLogFile);
      const cycleStatus = await readJsonIfPresent(CURRENT_CYCLE_STATUS_FILE);
      await logRunner(
        `cycle ${cycleIndex} finished code=${result.exitCode ?? "null"} signal=${result.signal ?? "none"} outcome=${cycleStatus?.outcome ?? "unknown"}`,
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
        nextRunAt: shutdownRequest ? null : futureIso(LOOP_INTERVAL_SECONDS),
      });

      if (await syncStopFileState()) break;
      if (await sleepWithStopPolling(LOOP_INTERVAL_SECONDS)) break;
    }
  } finally {
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

async function logRunner(message) {
  const line = `[${new Date().toISOString()}] ${message}\n`;
  await fs.appendFile(RUNNER_LOG_FILE, line, "utf8");
}

function detectPreflightBlock() {
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

  const status = runCommand("git", ["status", "--porcelain"]);
  if (status) {
    return {
      reason: "dirty worktree; waiting for operator cleanup before bounded autoloop runs",
      details: status.split(/\r?\n/).filter(Boolean).slice(0, 40),
    };
  }

  return null;
}

function runCommand(command, args) {
  try {
    return execFileSync(command, args, {
      cwd: ROOT,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
    }).trim();
  } catch (err) {
    const stdout = err?.stdout ? String(err.stdout) : "";
    const stderr = err?.stderr ? String(err.stderr) : "";
    throw new Error(`${command} ${args.join(" ")} failed.\n${stdout}${stderr}`.trim());
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

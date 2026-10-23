#!/usr/bin/env node

import { execFile } from "node:child_process";
import { readFile } from "node:fs/promises";
import { promisify } from "node:util";
import { pathToFileURL } from "node:url";

const execFileAsync = promisify(execFile);
const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export function parseArgs(argv) {
  const config = { rounds: 20, clients: 4, intervalMs: 1000, timeoutMs: 30000, expectedWorkers: 18 };
  const numeric = new Map([
    ["--rounds", ["rounds", 1, 10000]], ["--clients", ["clients", 1, 16]],
    ["--interval-ms", ["intervalMs", 0, 3600000]], ["--timeout-ms", ["timeoutMs", 1, 120000]],
    ["--expected-workers", ["expectedWorkers", 0, 256]], ["--rss-pid", ["rssPid", 1, 2147483647]],
    ["--max-memory-bytes", ["maxMemoryBytes", 1, Number.MAX_SAFE_INTEGER]],
  ]);
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    if (flag === "--help") return { help: true };
    const value = argv[++i];
    if (!value) throw new Error(`Missing value for ${flag}`);
    if (numeric.has(flag)) {
      const [key, min, max] = numeric.get(flag);
      const number = Number(value);
      if (!Number.isSafeInteger(number) || number < min || number > max) throw new Error(`Invalid ${flag}`);
      config[key] = number;
    } else if (flag === "--base-url") config.baseUrl = value;
    else if (flag === "--memory-file") config.memoryFile = value;
    else throw new Error(`Unknown argument: ${flag}`);
  }
  const url = new URL(config.baseUrl);
  if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.search || url.hash || url.pathname !== "/") {
    throw new Error("--base-url must be an HTTP(S) origin without credentials, query, or path");
  }
  config.baseUrl = url.origin;
  if (config.memoryFile && config.rssPid) throw new Error("Choose --memory-file or --rss-pid");
  if (config.maxMemoryBytes !== undefined && !config.memoryFile && !config.rssPid) {
    throw new Error("--max-memory-bytes requires --memory-file or --rss-pid");
  }
  return config;
}

export async function readMemoryBytes(config) {
  let bytes;
  if (config.memoryFile) bytes = Number((await readFile(config.memoryFile, "utf8")).trim());
  else if (config.rssPid) {
    const { stdout } = await execFileAsync("ps", ["-o", "rss=", "-p", String(config.rssPid)], { timeout: 2000 });
    bytes = Number(stdout.trim()) * 1024;
  } else return null;
  if (!Number.isSafeInteger(bytes) || bytes <= 0) throw new Error("Memory measurement unavailable");
  return bytes;
}

function summarize(samples) {
  if (samples.length === 0) return null;
  const sorted = [...samples].sort((a, b) => a - b);
  const percentile = (p) => sorted[Math.max(0, Math.ceil(p * sorted.length) - 1)];
  return { count: sorted.length, first: samples[0], last: samples.at(-1), max: sorted.at(-1), p95: percentile(0.95), p99: percentile(0.99) };
}

// Only GET probes are issued. Scheduled sync and any independently authorized
// startup/recovery run normally while this tool records their observed effects.
export async function runBenchmark(config, {
  fetchImpl = fetch, memorySample = () => readMemoryBytes(config),
  token = process.env.TRADER_API_TOKEN, tenantKey = process.env.TRADER_BENCH_TENANT_KEY,
} = {}) {
  const headers = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  if (tenantKey) headers["X-Tenant-Key"] = tenantKey;
  const requests = [];
  const memory = [];
  const workers = [];
  let memoryErrors = 0;
  let etag;
  let stopped = false;
  const startedAt = new Date().toISOString();
  const monitor = (async () => {
    while (!stopped) {
      try {
        const bytes = await memorySample();
        if (bytes !== null || config.memoryFile || config.rssPid) {
          if (!Number.isSafeInteger(bytes) || bytes <= 0) throw new Error("Memory measurement unavailable");
          memory.push(bytes);
        }
      } catch { memoryErrors += 1; }
      if (!stopped) await pause(100);
    }
  })();
  const probe = async (path, conditional = false) => {
    const start = performance.now();
    let status = 0;
    let httpStatus = 0;
    let failureReason = "request-failed";
    let byteCount = 0;
    let body;
    try {
      const requestHeaders = { ...headers };
      const sentEtag = conditional ? etag : undefined;
      if (sentEtag) requestHeaders["If-None-Match"] = sentEtag;
      const response = await fetchImpl(`${config.baseUrl}${path}`, {
        method: "GET", headers: requestHeaders, redirect: "error", signal: AbortSignal.timeout(config.timeoutMs),
      });
      status = response.status;
      httpStatus = status;
      if (status === 304 && !sentEtag) {
        failureReason = "unexpected-304";
        throw new Error("Unexpected conditional response");
      }
      if (status === 200 && !response.headers.get("content-type")?.includes("application/json")) {
        failureReason = "non-json";
        await response.body?.cancel();
        throw new Error("Expected API JSON");
      }
      // Consume every body to measure full request cost; do not retain leaderboard JSON.
      const parseBody = !conditional && status === 200;
      const chunks = parseBody ? [] : null;
      failureReason = "body-read-failed";
      if (response.body) {
        for await (const chunk of response.body) {
          byteCount += chunk.length;
          if (chunks) {
            if (byteCount > 16 * 1024 * 1024) {
              failureReason = "status-too-large";
              throw new Error("Status response too large");
            }
            chunks.push(chunk);
          }
        }
      }
      if (parseBody) {
        failureReason = "invalid-json";
        body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
      }
      if ((path === "/health" || path === "/ready") && status === 200 && body?.ready !== true) {
        failureReason = body?.ready === false ? "readiness-false" : "readiness-missing";
        throw new Error("Readiness evidence missing or false");
      }
      if (conditional && status === 200) etag = response.headers.get("etag") || undefined;
      if (status === 200 && path === "/bot/status") {
        const bots = body?.multi === true ? body.bots : [body];
        if (!Array.isArray(bots) || !bots.every((bot) => bot && typeof bot.running === "boolean")) {
          failureReason = "worker-evidence-invalid";
          throw new Error("Worker status evidence missing");
        }
        workers.push(bots.filter((bot) => bot.running === true).length);
      }
      failureReason = status === 200 || status === 304 ? null : "http-status";
    } catch {
      // Errors can embed URLs, response bodies, or authorization details.
      status = 0;
    }
    requests.push({ path, status, httpStatus, failureReason, bytes: byteCount, latencyMs: performance.now() - start });
  };
  try {
    for (let round = 0; round < config.rounds; round += 1) {
      await Promise.all([
        ...Array.from({ length: config.clients }, () => probe("/optimizer/combos", true)),
        probe("/health"), probe("/ready"), probe("/bot/status"),
      ]);
      if (round + 1 < config.rounds) await pause(config.intervalMs);
    }
  } finally {
    stopped = true;
    await monitor;
  }
  const failures = requests.filter(({ status }) => status !== 200 && status !== 304).length;
  const memoryBytes = summarize(memory);
  const memoryFailed = memoryErrors > 0 || ((config.memoryFile || config.rssPid) && memoryBytes === null);
  const memoryLimit = config.maxMemoryBytes ?? null;
  const memoryAssessment = memoryFailed ? "measurement-failed"
    : memoryBytes === null ? "unmeasured"
    : memoryLimit === null ? "observed"
    : memoryBytes.max > memoryLimit ? "limit-exceeded" : "within-limit";
  const paths = [...new Set(requests.map(({ path }) => path))];
  const endpoints = Object.fromEntries(paths.map((path) => {
    const matching = requests.filter((request) => request.path === path);
    return [path, {
      latencyMs: summarize(matching.map((request) => request.latencyMs)),
      statuses: Object.fromEntries([...new Set(matching.map(({ status }) => status))].map((status) => [status, matching.filter((r) => r.status === status).length])),
      httpStatuses: Object.fromEntries([...new Set(matching.map(({ httpStatus }) => httpStatus))].map((status) => [status, matching.filter((r) => r.httpStatus === status).length])),
      failureReasons: Object.fromEntries([...new Set(matching.map(({ failureReason }) => failureReason).filter(Boolean))].map((reason) => [reason, matching.filter((r) => r.failureReason === reason).length])),
      bytes: matching.reduce((sum, request) => sum + request.bytes, 0),
    }];
  }));
  return {
    schemaVersion: 1, startedAt, finishedAt: new Date().toISOString(),
    rounds: config.rounds, clients: config.clients, expectedWorkers: config.expectedWorkers,
    passed: failures === 0 && !memoryFailed && memoryAssessment !== "limit-exceeded"
      && workers.length === config.rounds && workers.every((count) => count === config.expectedWorkers),
    failures, endpoints, workers: summarize(workers),
    memory: {
      source: config.memoryFile ? "cgroup" : config.rssPid ? "rss" : "unmeasured",
      bytes: memoryBytes, errors: memoryErrors, assessment: memoryAssessment, maxAllowedBytes: memoryLimit,
    },
    coverage: { syncTriggered: false, startupTriggered: false, note: "Correlate timestamps with server sync/startup logs; GET probes do not establish that those events occurred." },
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    const config = parseArgs(process.argv.slice(2));
    if (config.help) {
      console.log("Usage: node scripts/benchmark-live-fleet.mjs --base-url ORIGIN [--rounds 20] [--clients 4] [--interval-ms 1000] [--timeout-ms 30000] [--expected-workers 18] [--rss-pid PID | --memory-file PATH] [--max-memory-bytes BYTES]\nCredentials: TRADER_API_TOKEN and optional TRADER_BENCH_TENANT_KEY. Only GET requests are sent. A memory ceiling requires a local memory source; otherwise the report only observes sampled memory or marks it unmeasured.");
    } else {
      const result = await runBenchmark(config);
      console.log(JSON.stringify(result, null, 2));
      process.exitCode = result.passed ? 0 : 1;
    }
  } catch {
    console.error("Fleet benchmark failed. Check arguments, origin, and local memory access; use --help for usage.");
    process.exitCode = 1;
  }
}

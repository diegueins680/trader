#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { pathToFileURL } from "node:url";

const DEFAULT_APP = "tdf-hq-db";
const DEFAULT_WARN_PERCENT = 75;
const DEFAULT_CRITICAL_PERCENT = 85;

function usage() {
  return [
    "Usage: node scripts/check-fly-postgres-disk.mjs [options]",
    "",
    "Options:",
    "  --app <name>          Fly Postgres app name (default: FLY_POSTGRES_APP or tdf-hq-db)",
    "  --warn <percent>      Warning disk-used threshold (default: 75)",
    "  --critical <percent>  Critical disk-used threshold (default: 85)",
    "  --webhook-url <url>   Optional JSON webhook for warning/critical alerts",
    "  --flyctl <path>       flyctl executable path (default: flyctl)",
    "  --json               Print the full result as JSON",
    "  -h, --help           Show this help",
  ].join("\n");
}

export function parseArgs(argv, env = process.env) {
  const args = {
    app: env.FLY_POSTGRES_APP || DEFAULT_APP,
    warnPercent: DEFAULT_WARN_PERCENT,
    criticalPercent: DEFAULT_CRITICAL_PERCENT,
    webhookUrl: env.TRADER_DISK_ALERT_WEBHOOK_URL || env.SLACK_WEBHOOK_URL || "",
    flyctl: "flyctl",
    json: false,
    help: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const readValue = () => {
      i += 1;
      if (i >= argv.length || argv[i].startsWith("-")) {
        throw new Error(`Missing value for ${arg}`);
      }
      return argv[i];
    };

    if (arg === "--app") {
      args.app = readValue();
    } else if (arg === "--warn") {
      args.warnPercent = parsePercent(readValue(), arg);
    } else if (arg === "--critical") {
      args.criticalPercent = parsePercent(readValue(), arg);
    } else if (arg === "--webhook-url") {
      args.webhookUrl = readValue();
    } else if (arg === "--flyctl") {
      args.flyctl = readValue();
    } else if (arg === "--json") {
      args.json = true;
    } else if (arg === "-h" || arg === "--help") {
      args.help = true;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  if (!args.app) {
    throw new Error("Fly app name must not be empty");
  }
  if (args.warnPercent >= args.criticalPercent) {
    throw new Error("--warn must be lower than --critical");
  }

  return args;
}

function parsePercent(value, label) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0 || parsed > 100) {
    throw new Error(`${label} must be a percentage from 0 to 100`);
  }
  return parsed;
}

export function parseFlyChecksJson(checksJson) {
  const machines = Object.entries(checksJson || {});
  const checks = machines.flatMap(([machineId, entries]) =>
    Array.isArray(entries) ? entries.map((entry) => ({ machineId, ...entry })) : [],
  );

  const pgCheck = checks.find((check) => check.name === "pg");
  const vmCheck = checks.find((check) => check.name === "vm");
  const diskCapacityPercent = parseFirstNumber(pgCheck?.output, /disk-capacity:\s*([0-9]+(?:\.[0-9]+)?)%/i);
  const readonlyThresholdPercent = parseFirstNumber(
    pgCheck?.output,
    /readonly mode will be enabled at\s*([0-9]+(?:\.[0-9]+)?)%/i,
  );
  const freeDataPercent = parseFirstNumber(vmCheck?.output, /checkDisk:.*\(([0-9]+(?:\.[0-9]+)?)%\)\s*free space on \/data\//i);

  return {
    checks,
    diskCapacityPercent,
    readonlyThresholdPercent,
    freeDataPercent,
    nonPassingChecks: checks.filter((check) => check.status !== "passing"),
  };
}

function parseFirstNumber(text, pattern) {
  const match = typeof text === "string" ? text.match(pattern) : null;
  return match ? Number(match[1]) : null;
}

export function evaluateDisk(metrics, thresholds) {
  if (metrics.nonPassingChecks.length > 0) {
    return {
      level: "critical",
      exitCode: 2,
      reason: `non-passing checks: ${metrics.nonPassingChecks.map((check) => check.name).join(", ")}`,
    };
  }
  if (!Number.isFinite(metrics.diskCapacityPercent)) {
    return {
      level: "critical",
      exitCode: 2,
      reason: "missing pg disk-capacity check",
    };
  }
  if (metrics.diskCapacityPercent >= thresholds.criticalPercent) {
    return {
      level: "critical",
      exitCode: 2,
      reason: `disk-capacity ${metrics.diskCapacityPercent}% >= critical ${thresholds.criticalPercent}%`,
    };
  }
  if (metrics.diskCapacityPercent >= thresholds.warnPercent) {
    return {
      level: "warning",
      exitCode: 1,
      reason: `disk-capacity ${metrics.diskCapacityPercent}% >= warning ${thresholds.warnPercent}%`,
    };
  }
  return {
    level: "ok",
    exitCode: 0,
    reason: `disk-capacity ${metrics.diskCapacityPercent}% below warning ${thresholds.warnPercent}%`,
  };
}

function runFlyChecks({ flyctl, app }) {
  const stdout = execFileSync(flyctl, ["checks", "list", "-a", app, "--json"], {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  return JSON.parse(stdout);
}

function formatResult({ app, metrics, evaluation }) {
  const parts = [
    `fly-postgres-disk ${evaluation.level.toUpperCase()}`,
    `app=${app}`,
    `used=${formatPercent(metrics.diskCapacityPercent)}`,
    `free=${formatPercent(metrics.freeDataPercent)}`,
  ];
  if (Number.isFinite(metrics.readonlyThresholdPercent)) {
    parts.push(`readonly_at=${formatPercent(metrics.readonlyThresholdPercent)}`);
  }
  parts.push(`reason="${evaluation.reason}"`);
  return parts.join(" ");
}

function formatPercent(value) {
  return Number.isFinite(value) ? `${value.toFixed(1)}%` : "unknown";
}

async function postWebhook(webhookUrl, payload) {
  if (!webhookUrl || payload.evaluation.level === "ok") {
    return;
  }
  const response = await fetch(webhookUrl, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      text: payload.summary,
      app: payload.app,
      level: payload.evaluation.level,
      reason: payload.evaluation.reason,
      metrics: payload.metrics,
    }),
  });
  if (!response.ok) {
    throw new Error(`Webhook returned HTTP ${response.status}`);
  }
}

async function main() {
  let args;
  try {
    args = parseArgs(process.argv.slice(2));
  } catch (error) {
    console.error(error.message);
    console.error("");
    console.error(usage());
    process.exitCode = 2;
    return;
  }

  if (args.help) {
    console.log(usage());
    return;
  }

  try {
    const checksJson = runFlyChecks(args);
    const metrics = parseFlyChecksJson(checksJson);
    const evaluation = evaluateDisk(metrics, args);
    const payload = {
      app: args.app,
      metrics,
      evaluation,
      summary: formatResult({ app: args.app, metrics, evaluation }),
    };
    await postWebhook(args.webhookUrl, payload);
    if (args.json) {
      console.log(JSON.stringify(payload, null, 2));
    } else {
      console.log(payload.summary);
    }
    process.exitCode = evaluation.exitCode;
  } catch (error) {
    console.error(`fly-postgres-disk CRITICAL app=${args.app} reason="${error.message}"`);
    process.exitCode = 2;
  }
}

if (import.meta.url === pathToFileURL(process.argv[1] || "").href) {
  await main();
}

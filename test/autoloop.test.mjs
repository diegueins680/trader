import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs/promises";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import {
  evaluateDisk,
  parseArgs as parseFlyPostgresDiskArgs,
  parseFlyChecksJson,
} from "../scripts/check-fly-postgres-disk.mjs";
import {
  checkStationLive,
  maintainStations,
  normalizeStation,
  parseArgs as parseRadioStationArgs,
} from "../scripts/maintain-radio-stations.mjs";
import {
  buildAutoloopScratchBranchCandidates,
  buildBranchMergeCandidates,
  buildActionsRunsApiPath,
  buildForceWithLeaseFlag,
  buildRemoteTrackingRefspec,
  buildAnthropicApiError,
  buildOpenAiApiError,
  clampText,
  extractAnthropicResponseText,
  extractCodexExecLastMessage,
  extractResponseText,
  isAutoloopRecoveryBranch,
  normalizeGitBranchShortName,
  normalizeIdeaSelection,
  normalizePatchPlan,
  parseGitStatusPaths,
  parseLsRemoteBranchHead,
  parseJsonResponse,
  prepareShellCommand,
  resolveAutoloopBackend,
  sanitizeRelativePath,
  selectMergeVerificationTarget,
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

const SCORECARD_SCRIPT = fileURLToPath(new URL("../haskell/scripts/volatility_gating_scorecard.py", import.meta.url));
const VALIDATE_TOML_SCRIPT = fileURLToPath(new URL("../scripts/validate-toml.py", import.meta.url));
const EXTRACT_EDGES_SCRIPT = fileURLToPath(new URL("../scripts/extract_edges.py", import.meta.url));
const RESEARCH_DIR = fileURLToPath(new URL("../scripts/research/", import.meta.url));
const PYTHON_RESEARCH_DEPS =
  spawnSync("python3", ["-c", "import numpy, pandas"], { encoding: "utf8" }).status === 0;

function runScorecard(args) {
  return spawnSync("python3", [SCORECARD_SCRIPT, ...args], {
    encoding: "utf8",
  });
}

async function writeScorecardJson(dir, name, payload) {
  const filePath = path.join(dir, name);
  await fs.writeFile(filePath, JSON.stringify(payload));
  return filePath;
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

test("parseFlyChecksJson extracts Postgres disk metrics", () => {
  const metrics = parseFlyChecksJson({
    "machine-1": [
      {
        name: "pg",
        status: "passing",
        output:
          "[OK] connections: 13 used, 3 reserved, 300 max\n[OK] disk-capacity: 32.0% - readonly mode will be enabled at 90.0%",
      },
      {
        name: "vm",
        status: "passing",
        output: "[OK] checkDisk: 1.92 GB (65.8%) free space on /data/",
      },
    ],
  });

  assert.equal(metrics.diskCapacityPercent, 32);
  assert.equal(metrics.readonlyThresholdPercent, 90);
  assert.equal(metrics.freeDataPercent, 65.8);
  assert.deepEqual(metrics.nonPassingChecks, []);
});

test("evaluateDisk fails before Postgres readonly mode", () => {
  const warning = evaluateDisk(
    { diskCapacityPercent: 80, readonlyThresholdPercent: 90, freeDataPercent: 20, nonPassingChecks: [] },
    { warnPercent: 75, criticalPercent: 85 },
  );
  assert.equal(warning.level, "warning");
  assert.equal(warning.exitCode, 1);

  const critical = evaluateDisk(
    { diskCapacityPercent: 86, readonlyThresholdPercent: 90, freeDataPercent: 14, nonPassingChecks: [] },
    { warnPercent: 75, criticalPercent: 85 },
  );
  assert.equal(critical.level, "critical");
  assert.equal(critical.exitCode, 2);
});

test("parseFlyPostgresDiskArgs validates alert thresholds", () => {
  assert.deepEqual(parseFlyPostgresDiskArgs(["--app", "db", "--warn", "70", "--critical", "80"], {}), {
    app: "db",
    warnPercent: 70,
    criticalPercent: 80,
    webhookUrl: "",
    flyctl: "flyctl",
    json: false,
    help: false,
  });
  assert.throws(
    () => parseFlyPostgresDiskArgs(["--warn", "90", "--critical", "80"], {}),
    /--warn must be lower than --critical/,
  );
});

test("parseRadioStationArgs accepts cron-oriented discovery settings", () => {
  const args = parseRadioStationArgs(
    ["--stations-file", "state/radio.json", "--discovery-url", "https://example.test/more.json", "--max-failures", "1"],
    {
      RADIO_STATION_DISCOVERY_URLS: "https://example.test/base.json",
      RADIO_STATION_CHECK_TIMEOUT_MS: "1500",
      RADIO_STATION_CHECK_CONCURRENCY: "3",
    },
  );

  assert.equal(args.stationsFile, "state/radio.json");
  assert.deepEqual(args.discoveryUrls, ["https://example.test/base.json", "https://example.test/more.json"]);
  assert.equal(args.timeoutMs, 1500);
  assert.equal(args.concurrency, 3);
  assert.equal(args.maxFailures, 1);
});

test("normalizeStation accepts radio-browser style fields", () => {
  const station = normalizeStation(
    {
      stationuuid: "ABC 123",
      name: " Example FM ",
      url_resolved: "https://stream.example.test/live.mp3",
      homepage: "https://example.test",
      tags: "news, talk,news",
      bitrate: "128",
    },
    "fixture",
  );

  assert.equal(station.id, "abc-123");
  assert.equal(station.name, "Example FM");
  assert.equal(station.url, "https://stream.example.test/live.mp3");
  assert.deepEqual(station.tags, ["news", "talk"]);
  assert.equal(station.bitrate, 128);
  assert.equal(normalizeStation({ name: "bad", url: "ftp://example.test/stream" }, "fixture"), null);
});

test("checkStationLive probes streams with native HTTP by default", async () => {
  const server = http.createServer((request, response) => {
    assert.equal(request.headers.range, "bytes=0-1023");
    response.writeHead(200, { "content-type": "audio/mpeg" });
    response.write(Buffer.from([1, 2, 3]));
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));

  try {
    const { port } = server.address();
    const result = await checkStationLive({ url: `http://127.0.0.1:${port}/stream` }, { timeoutMs: 1000 });
    assert.deepEqual(result, { live: true, status: 200 });
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test("checkStationLive accepts stream headers without closing the response body", async () => {
  let signal;
  let bodyRead = false;
  const result = await checkStationLive(
    { url: "https://stream.example.test/live.mp3" },
    {
      fetchImpl: async (_url, options) => {
        signal = options.signal;
        return {
          status: 200,
          headers: {
            get: (name) => (name.toLowerCase() === "content-type" ? "audio/mpeg" : ""),
          },
          body: {
            getReader: () => {
              bodyRead = true;
              throw new Error("body should not be read when stream headers identify the response");
            },
          },
        };
      },
    },
  );

  assert.deepEqual(result, { live: true, status: 200 });
  assert.equal(bodyRead, false);
  assert.equal(signal.aborted, false);
});

test("checkStationLive leaves first-chunk probes open instead of canceling the reader", async () => {
  let signal;
  let cancelCalled = false;
  const result = await checkStationLive(
    { url: "https://stream.example.test/live" },
    {
      fetchImpl: async (_url, options) => {
        signal = options.signal;
        return {
          status: 200,
          headers: {
            get: () => "text/plain",
          },
          body: {
            getReader: () => ({
              read: async () => ({ done: false, value: new Uint8Array([1, 2, 3]) }),
              cancel: async () => {
                cancelCalled = true;
              },
            }),
          },
        };
      },
    },
  );

  assert.deepEqual(result, { live: true, status: 200 });
  assert.equal(cancelCalled, false);
  assert.equal(signal.aborted, false);
});

test("radio station maintenance purges failed stations and adds live discoveries", async () => {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "radio-stations-test-"));
  try {
    const stationsFile = path.join(dir, "radio-stations.json");
    const discoveryFile = path.join(dir, "discovery.json");
    await fs.writeFile(
      stationsFile,
      JSON.stringify({
        stations: [
          { id: "keep", name: "Keep Live", url: "https://stream.example.test/keep-live.mp3" },
          { id: "retry", name: "Retry Later", url: "https://stream.example.test/retry-down.mp3" },
          {
            id: "purge",
            name: "Purge Down",
            url: "https://stream.example.test/purge-down.mp3",
            consecutiveFailures: 1,
          },
        ],
      }),
    );
    await fs.writeFile(
      discoveryFile,
      JSON.stringify({
        stations: [
          { id: "keep-dupe", name: "Keep Duplicate", url: "https://stream.example.test/keep-live.mp3" },
          { stationuuid: "new-live", name: "New Live", url_resolved: "https://stream.example.test/new-live.mp3" },
          { stationuuid: "new-down", name: "New Down", url_resolved: "https://stream.example.test/new-down.mp3" },
        ],
      }),
    );

    const result = await maintainStations({
      stationsFile,
      discoveryFiles: [discoveryFile],
      maxFailures: 2,
      concurrency: 2,
      now: new Date("2026-07-06T12:00:00.000Z"),
      checkLive: async (station) => ({ live: station.url.includes("live"), status: station.url.includes("live") ? 200 : 503 }),
    });

    assert.equal(result.checked, 3);
    assert.equal(result.live, 1);
    assert.equal(result.failed, 2);
    assert.deepEqual(
      result.purged.map((station) => station.id),
      ["purge"],
    );
    assert.equal(result.discovered, 3);
    assert.equal(result.newCandidates, 2);
    assert.deepEqual(
      result.added.map((station) => station.id),
      ["new-live"],
    );
    assert.deepEqual(
      result.skippedDiscoveryDown.map((station) => station.id),
      ["new-down"],
    );

    const persisted = JSON.parse(await fs.readFile(stationsFile, "utf8"));
    assert.deepEqual(
      persisted.stations.map((station) => station.id),
      ["keep", "retry", "new-live"],
    );
    assert.equal(persisted.stations.find((station) => station.id === "retry").consecutiveFailures, 1);
    assert.equal(persisted.lastMaintenance.purged, 1);
    assert.equal(persisted.lastMaintenance.added, 1);
  } finally {
    await fs.rm(dir, { recursive: true, force: true });
  }
});

test("Fly config installs the radio maintenance process group", async () => {
  const flyConfig = await fs.readFile(new URL("../fly.toml", import.meta.url), "utf8");
  const dockerfile = await fs.readFile(new URL("../Dockerfile", import.meta.url), "utf8");
  const radioLoop = await fs.readFile(new URL("../scripts/fly-radio-stations-loop.sh", import.meta.url), "utf8");

  assert.equal(flyConfig.match(/^\[processes\]$/gm)?.length, 1, "Fly config must declare [processes] once");
  assert.match(flyConfig, /\[processes\][\s\S]*app = "trader-hs --serve --port 8080 --platform binance --futures --trade-log \.tmp\/trader\/live_trades\.ndjson"/);
  assert.match(flyConfig, /TRADER_BOT_TRADE = "false"/);
  assert.doesNotMatch(flyConfig, /app = "[^"]*--binance-live/);
  assert.match(flyConfig, /\[processes\][\s\S]*radio = "sh \/usr\/local\/bin\/fly-radio-stations-loop"/);
  assert.match(flyConfig, /RADIO_STATIONS_FILE = "\/var\/lib\/trader\/state\/radio-stations\.json"/);
  assert.match(flyConfig, /RADIO_STATION_DISCOVERY_URLS = "https:\/\/all\.api\.radio-browser\.info\/json\/stations\/topclick\/500\?hidebroken=true"/);
  assert.match(flyConfig, /\[\[vm\]\][\s\S]*size = "performance-8x"[\s\S]*processes = \["app"\]/);
  assert.match(flyConfig, /\[\[vm\]\][\s\S]*size = "shared-cpu-1x"[\s\S]*processes = \["radio"\]/);

  assert.match(dockerfile, /apt-get install -y --no-install-recommends[^\n]*nodejs/);
  assert.match(dockerfile, /COPY scripts\/autoloop-lib\.mjs scripts\/maintain-radio-stations\.mjs \/opt\/trader\/scripts\//);
  assert.match(dockerfile, /COPY scripts\/fly-radio-stations-loop\.sh \/usr\/local\/bin\/fly-radio-stations-loop/);

  assert.match(radioLoop, /maintain-radio-stations\.mjs --stations-file "\$stations_file"/);
  assert.doesNotMatch(radioLoop, /--json/);
});

test("deploy config verifier parses every Fly TOML and rejects duplicate tables", async () => {
  const valid = spawnSync("python3", [VALIDATE_TOML_SCRIPT], { encoding: "utf8" });
  assert.equal(valid.status, 0, valid.stderr);
  assert.match(valid.stdout, /validated TOML: fly\.toml/);
  assert.match(valid.stdout, /validated TOML: fly\.research\.toml/);
  assert.match(valid.stdout, /validated TOML: haskell\/web\/fly\.frontend\.toml/);

  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "invalid-toml-test-"));
  try {
    const invalid = path.join(dir, "duplicate.toml");
    await fs.writeFile(invalid, "[processes]\napp = 'one'\n[processes]\napp = 'two'\n");
    const rejected = spawnSync("python3", [VALIDATE_TOML_SCRIPT, invalid], { encoding: "utf8" });
    assert.notEqual(rejected.status, 0);
    assert.match(rejected.stderr, /invalid TOML .*duplicate\.toml/i);
  } finally {
    await fs.rm(dir, { recursive: true, force: true });
  }

  const verify = await fs.readFile(new URL("../scripts/verify.sh", import.meta.url), "utf8");
  const deployVerifier = await fs.readFile(new URL("../scripts/verify-deploy-config.sh", import.meta.url), "utf8");
  assert.match(verify, /bash scripts\/verify-deploy-config\.sh/);
  assert.match(deployVerifier, /docker compose[\s\S]*config --quiet/);
  assert.match(deployVerifier, /trader\.trading\.env\.example/);
  assert.match(deployVerifier, /trader\.research\.env\.example/);
});

test("buildBranchMergeCandidates ignores autoloop recovery branches", () => {
  const candidates = buildBranchMergeCandidates({
    baseBranch: "main",
    localBranches: [
      "main",
      "feature/live-fix",
      "autoloop/recovery/main/cycle-41-2026-04-10t09-34-23-000z",
      "autoloop/checkpoint/main/main-2026-04-10t09-28-48-000z",
    ],
    remoteBranches: [
      "origin/main",
      "origin/feature/live-fix",
      "origin/autoloop/recovery/main/cycle-41-2026-04-10t09-34-23-000z",
    ],
  });

  assert.deepEqual(candidates, [
    {
      shortName: "feature/live-fix",
      ref: "feature/live-fix",
      localRef: "feature/live-fix",
      remoteRef: "origin/feature/live-fix",
    },
  ]);
  assert.equal(isAutoloopRecoveryBranch("refs/heads/autoloop/recovery/main/cycle-41"), true);
  assert.equal(isAutoloopRecoveryBranch("origin/autoloop/checkpoint/main/main-2026"), true);
  assert.equal(isAutoloopRecoveryBranch("feature/live-fix"), false);
});

test("buildAutoloopScratchBranchCandidates keeps only autoloop recovery and checkpoint refs", () => {
  const candidates = buildAutoloopScratchBranchCandidates({
    baseBranch: "main",
    localBranches: [
      "main",
      "feature/live-fix",
      "autoloop/recovery/main/cycle-41-2026-04-10t09-34-23-000z",
      "autoloop/checkpoint/main/main-2026-04-10t09-28-48-000z",
    ],
    remoteBranches: [
      "origin/main",
      "origin/autoloop/recovery/main/cycle-41-2026-04-10t09-34-23-000z",
      "origin/autoloop/checkpoint/main/main-2026-04-10t09-28-48-000z",
      "origin/feature/live-fix",
    ],
  });

  assert.deepEqual(candidates, [
    {
      shortName: "autoloop/checkpoint/main/main-2026-04-10t09-28-48-000z",
      ref: "autoloop/checkpoint/main/main-2026-04-10t09-28-48-000z",
      localRef: "autoloop/checkpoint/main/main-2026-04-10t09-28-48-000z",
      remoteRef: "origin/autoloop/checkpoint/main/main-2026-04-10t09-28-48-000z",
    },
    {
      shortName: "autoloop/recovery/main/cycle-41-2026-04-10t09-34-23-000z",
      ref: "autoloop/recovery/main/cycle-41-2026-04-10t09-34-23-000z",
      localRef: "autoloop/recovery/main/cycle-41-2026-04-10t09-34-23-000z",
      remoteRef: "origin/autoloop/recovery/main/cycle-41-2026-04-10t09-34-23-000z",
    },
  ]);
});

test("selectMergeVerificationTarget chooses the narrowest canonical gate", () => {
  assert.equal(selectMergeVerificationTarget(["haskell/app/Main.hs"]), null);
  assert.equal(selectMergeVerificationTarget(["haskell/web/src/App.tsx", "README.md"]), "web");
  assert.equal(selectMergeVerificationTarget(["scripts/autoloop-lib.mjs", "CHANGELOG.md"]), "automation");
  assert.equal(selectMergeVerificationTarget(["haskell/web/src/App.tsx", "scripts/autoloop-lib.mjs"]), "full");
  assert.equal(selectMergeVerificationTarget(["deploy/hetzner/docker-compose.trading.yml"]), "full");
  assert.equal(selectMergeVerificationTarget(["FORMAL_METHODS.md"]), "full");
  assert.equal(selectMergeVerificationTarget([]), null);
});

test("extract_edges keeps ex-ante decision edges separate from realized outcomes", async () => {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "extract-edges-test-"));
  try {
    const input = path.join(dir, "backtest.json");
    const decisionOutput = path.join(dir, "decision.csv");
    const outcomeOutput = path.join(dir, "outcome.csv");
    await fs.writeFile(
      input,
      JSON.stringify({
        trades: [
          { entryPrice: 100, exitPrice: 110, return: -0.05 },
          { return: 0 },
          { entryPrice: 100, exitPrice: 90, side: "SHORT" },
        ],
        decisionTraces: [{ edge: 0.02 }, { entryEdge: 0.025 }, { edge: Number.NaN }],
        gateTelemetry: { recentRejections: [{ edge: 0.03 }] },
      }),
    );

    const decisionRun = spawnSync(
      "python3",
      [EXTRACT_EDGES_SCRIPT, "--backtest-json", input, "--output", decisionOutput],
      { encoding: "utf8" },
    );
    assert.equal(decisionRun.status, 0, decisionRun.stderr);
    assert.deepEqual((await fs.readFile(decisionOutput, "utf8")).trim().split("\n"), [
      "0.02000000",
      "0.02500000",
      "0.03000000",
    ]);

    const outcomeRun = spawnSync(
      "python3",
      [
        EXTRACT_EDGES_SCRIPT,
        "--backtest-json",
        input,
        "--series",
        "realized-return",
        "--output",
        outcomeOutput,
      ],
      { encoding: "utf8" },
    );
    assert.equal(outcomeRun.status, 0, outcomeRun.stderr);
    assert.deepEqual((await fs.readFile(outcomeOutput, "utf8")).trim().split("\n"), [
      "-0.05000000",
      "0.00000000",
      "0.10000000",
    ]);
  } finally {
    await fs.rm(dir, { recursive: true, force: true });
  }
});

test(
  "research cache overlap and normalization remain point-in-time",
  { skip: !PYTHON_RESEARCH_DEPS },
  () => {
    const program = String.raw`
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1])
import datafeed
import run_example

old = pd.DataFrame([
    {"openTime": 1, "close": 10.0, "funding": 0.01, "oi": 100.0},
    {"openTime": 2, "close": 20.0, "funding": 0.02, "oi": 200.0},
])
fresh = pd.DataFrame([
    {"openTime": 1, "close": 11.0, "funding": np.nan, "oi": np.nan},
    {"openTime": 3, "close": 30.0, "funding": 0.03, "oi": 300.0},
])
merged = datafeed.merge_cache_frames(old, fresh)
overlap = merged.loc[merged["openTime"] == 1].iloc[0]

prefix = np.arange(1.0, 31.0)
prefix_score = run_example.expanding_past_zscore(prefix)
extended_score = run_example.expanding_past_zscore(np.concatenate([prefix, [1e12, -1e12]]))
print(json.dumps({
    "openTimes": merged["openTime"].tolist(),
    "overlapClose": overlap["close"],
    "overlapFunding": overlap["funding"],
    "overlapOi": overlap["oi"],
    "prefixInvariant": bool(np.allclose(prefix_score, extended_score[:len(prefix)], equal_nan=True)),
}))
`;
    const run = spawnSync("python3", ["-c", program, RESEARCH_DIR], { encoding: "utf8" });
    assert.equal(run.status, 0, run.stderr);
    const result = JSON.parse(run.stdout);
    assert.deepEqual(result.openTimes, [1, 2, 3]);
    assert.equal(result.overlapClose, 11);
    assert.equal(result.overlapFunding, 0.01);
    assert.equal(result.overlapOi, 100);
    assert.equal(result.prefixInvariant, true);
  },
);

test("research and web proxy correctness contracts are explicit in source", async () => {
  const datafeed = await fs.readFile(new URL("../scripts/research/datafeed.py", import.meta.url), "utf8");
  const example = await fs.readFile(new URL("../scripts/research/run_example.py", import.meta.url), "utf8");
  const nginx = await fs.readFile(new URL("../haskell/web/nginx/default.conf.template", import.meta.url), "utf8");
  assert.match(datafeed, /fresh_by_time\.combine_first\(old_by_time\)/);
  assert.match(example, /def expanding_past_zscore\(/);
  assert.doesNotMatch(example, /np\.nanmean|np\.nanstd/);
  assert.match(nginx, /proxy_pass \$\{TRADER_API_ORIGIN\}\//);
  assert.match(nginx, /proxy_set_header Host \$proxy_host;/);
  assert.match(nginx, /proxy_ssl_name \$proxy_host;/);
  assert.doesNotMatch(nginx, /trader-hs\.fly\.dev/);
});

test("parseJsonResponse rejects invalid JSON", () => {
  assert.throws(() => parseJsonResponse("not-json"), /invalid JSON/);
});

test("parseJsonResponse extracts first JSON object when model emits a stray preamble", () => {
  // Newer Anthropic models (e.g. claude-opus-4-8) reject assistant-message
  // prefill, so we can no longer force the response to start with '{'. The
  // parser must tolerate optional whitespace, prose, or a leading newline
  // before the JSON object.
  assert.deepEqual(parseJsonResponse('Sure, here is the result:\n{"ok":true}'), { ok: true });
  assert.deepEqual(parseJsonResponse('  \n {"noChange": false, "title": "x"}  \n'), { noChange: false, title: "x" });
  assert.deepEqual(parseJsonResponse('```json\n{"a":1}\n```'), { a: 1 });
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
  const reviewIdea = normalizeIdeaSelection(
    {
      noChange: false,
      title: "Handle valid automation review",
      rationale: "The review points at a concrete autoloop behavior issue.",
      algorithmReviewPath: "scripts/autoloop.mjs",
      algorithmReviewFocus: "Validate the review against the autoloop planner flow.",
      formalMethodsPath: "test/autoloop.test.mjs",
      formalMethodsFocus: "Cover the review-driven behavior in automation tests.",
      filesNeeded: ["scripts/autoloop.mjs", "test/autoloop.test.mjs"],
      verificationCommands: ["bash scripts/verify.sh automation"],
    },
    { algorithmReviewPrefixes: ["scripts/"] },
  );
  assert.equal(reviewIdea.algorithmReviewPath, "scripts/autoloop.mjs");
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
  const coercedPlan = normalizePatchPlan({
    noChange: false,
    title: { text: "Patch planner text fields" },
    summary: ["Tolerate planner metadata arrays.", "Keep file changes validated separately."],
    commitMessage: { message: "fix: tolerate planner text metadata" },
    algorithmReviewSummary: { text: "Reviewed the planner normalizer." },
    formalMethodsSummary: { text: "Covered metadata coercion in automation tests." },
    changes: [{ path: "README.md", content: "# hi" }],
    verificationCommands: [],
  });
  assert.equal(coercedPlan.title, "Patch planner text fields");
  assert.equal(coercedPlan.summary, "Tolerate planner metadata arrays.\nKeep file changes validated separately.");
  assert.equal(coercedPlan.commitMessage, "fix: tolerate planner text metadata");
  const replacementPlan = normalizePatchPlan({
    noChange: false,
    title: "Patch large file",
    summary: "Use targeted replacements",
    commitMessage: "Patch large file",
    algorithmReviewSummary: "Reviewed the backend trading file.",
    formalMethodsSummary: "The contract is unchanged.",
    changes: [
      {
        path: "haskell/app/Trader/Trading.hs",
        replacements: [{ find: "oldThreshold", replace: "newThreshold", expectedCount: 1 }],
      },
    ],
  });
  assert.deepEqual(replacementPlan.changes[0]?.replacements, [
    { find: "oldThreshold", replace: "newThreshold", expectedCount: 1, reason: "" },
  ]);
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
  assert.throws(
    () =>
      normalizePatchPlan({
        noChange: false,
        title: "Patch marker payload",
        summary: "Patch marker payload",
        commitMessage: "Patch marker payload",
        algorithmReviewSummary: "Reviewed the backend trading file.",
        formalMethodsSummary: "The contract is unchanged.",
        changes: [
          {
            path: "haskell/app/Trader/Trading.hs",
            content: "*** Begin Patch\n*** Update File: haskell/app/Trader/Trading.hs\n",
          },
        ],
      }),
    /looks like a patch\/diff payload/,
  );
  assert.throws(
    () =>
      normalizePatchPlan({
        noChange: false,
        title: "Diff payload",
        summary: "Diff payload",
        commitMessage: "Diff payload",
        algorithmReviewSummary: "Reviewed the backend trading file.",
        formalMethodsSummary: "The contract is unchanged.",
        changes: [
          {
            path: "README.md",
            content: "diff --git a/README.md b/README.md\n",
          },
        ],
      }),
    /looks like a patch\/diff payload/,
  );
  assert.throws(
    () =>
      normalizePatchPlan({
        noChange: false,
        title: "Instruction payload",
        summary: "Instruction payload",
        commitMessage: "Instruction payload",
        algorithmReviewSummary: "Reviewed the backend trading file.",
        formalMethodsSummary: "The contract is unchanged.",
        changes: [
          {
            path: "haskell/app/Trader/Trading.hs",
            content: "In haskell/app/Trader/Trading.hs, replace this Kelly-lite binding block:\nold\nwith this:\nnew\n",
          },
        ],
      }),
    /looks like edit instructions/,
  );
  assert.throws(
    () =>
      normalizePatchPlan({
        noChange: false,
        title: "Ambiguous patch",
        summary: "Ambiguous patch",
        commitMessage: "Ambiguous patch",
        algorithmReviewSummary: "Reviewed the backend trading file.",
        formalMethodsSummary: "The contract is unchanged.",
        changes: [
          {
            path: "haskell/app/Trader/Trading.hs",
            content: "module Trader.Trading where\n",
            replacements: [{ find: "old", replace: "new" }],
          },
        ],
      }),
    /either content or replacements/,
  );
  assert.throws(
    () =>
      normalizePatchPlan({
        noChange: false,
        title: "Bad replacement count",
        summary: "Bad replacement count",
        commitMessage: "Bad replacement count",
        algorithmReviewSummary: "Reviewed the backend trading file.",
        formalMethodsSummary: "The contract is unchanged.",
        changes: [
          {
            path: "haskell/app/Trader/Trading.hs",
            replacements: [{ find: "old", replace: "new", expectedCount: 0 }],
          },
        ],
      }),
    /expectedCount must be a positive integer/,
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

test("normalizeGitBranchShortName strips origin and refs prefixes", () => {
  assert.equal(normalizeGitBranchShortName("origin/feature/test"), "feature/test");
  assert.equal(normalizeGitBranchShortName("refs/heads/main"), "main");
  assert.equal(normalizeGitBranchShortName("refs/remotes/origin/topic"), "topic");
  assert.equal(normalizeGitBranchShortName("origin"), "");
  assert.equal(normalizeGitBranchShortName("origin/HEAD"), "");
  assert.equal(normalizeGitBranchShortName(""), "");
});

test("buildBranchMergeCandidates prefers local heads while deduping remote matches", () => {
  assert.deepEqual(
    buildBranchMergeCandidates({
      localBranches: ["feature/local", "topic/only-local", "main"],
      remoteBranches: ["origin", "origin/feature/local", "origin/topic/only-remote", "origin/main", "origin/HEAD"],
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

test("resolveAutoloopBackend prefers Anthropic, then OpenAI, then Codex in auto mode", () => {
  assert.equal(
    resolveAutoloopBackend("auto", { hasAnthropicKey: true, hasOpenAiKey: true, hasCodex: true }),
    "anthropic",
  );
  assert.equal(resolveAutoloopBackend("auto", { hasOpenAiKey: true, hasCodex: true }), "openai");
  assert.equal(resolveAutoloopBackend("", { hasOpenAiKey: false, hasCodex: true }), "codex");
  assert.equal(resolveAutoloopBackend("", { hasOpenAiKey: false, hasCodex: false }), "");
});

test("resolveAutoloopBackend respects explicit backend requests", () => {
  assert.equal(
    resolveAutoloopBackend("anthropic", { hasAnthropicKey: true, hasOpenAiKey: true, hasCodex: true }),
    "anthropic",
  );
  assert.equal(resolveAutoloopBackend("claude", { hasAnthropicKey: true }), "anthropic");
  assert.equal(resolveAutoloopBackend("anthropic", { hasAnthropicKey: false, hasOpenAiKey: true }), "");
  assert.equal(resolveAutoloopBackend("openai", { hasOpenAiKey: true, hasCodex: true }), "openai");
  assert.equal(resolveAutoloopBackend("codex", { hasOpenAiKey: true, hasCodex: true }), "codex");
  assert.equal(resolveAutoloopBackend("codex", { hasOpenAiKey: true, hasCodex: false }), "");
  assert.throws(
    () => resolveAutoloopBackend("mystery", { hasOpenAiKey: true, hasCodex: true }),
    /Unknown autoloop backend/,
  );
});

test("extractAnthropicResponseText concatenates text blocks", () => {
  const response = {
    content: [
      { type: "text", text: "one" },
      { type: "thinking", thinking: "ignored" },
      { type: "text", text: "two" },
    ],
  };
  assert.equal(extractAnthropicResponseText(response), "one\ntwo");
  assert.equal(extractAnthropicResponseText({}), "");
});

test("buildAnthropicApiError marks auth and billing failures as skippable", () => {
  const authErr = buildAnthropicApiError(401, { error: { type: "authentication_error", message: "bad key" } });
  const billingErr = buildAnthropicApiError(400, {
    error: { type: "invalid_request_error", message: "Your credit balance is too low" },
  });
  const overloadedErr = buildAnthropicApiError(529, { error: { type: "overloaded_error", message: "busy" } });
  assert.equal(authErr.skipAutoloop, true);
  assert.equal(billingErr.skipAutoloop, true);
  assert.equal(overloadedErr.skipAutoloop, false);
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
  assert.match(script, /const CI_DISCOVERY_TIMEOUT_SECONDS = clampInt\(process\.env\.AUTOLOOP_CI_DISCOVERY_TIMEOUT_SECONDS, 3000, 60, 7200\);/);
  assert.match(script, /const FAILURE_DISCOVERY_TIMEOUT_SECONDS = clampInt\(\s*process\.env\.AUTOLOOP_FAILURE_DISCOVERY_TIMEOUT_SECONDS,\s*60,\s*5,\s*CI_DISCOVERY_TIMEOUT_SECONDS,\s*\);/);
  assert.match(script, /const pushedHeadSha = runGit\(\["rev-parse", "HEAD"\]\);/);
  assert.match(script, /phase: "ci-wait",\s*[\s\S]*headSha: pushedHeadSha/);
  assert.match(script, /const ci = waitForBranchCi\(pushedHeadSha, LOOP_BRANCH\);/);
  assert.match(script, /function selectLatestWorkflowRunsByWorkflow\(runs\)/);
  assert.match(script, /function workflowRunIdentity\(run\)/);
  assert.match(
    script,
    /function pollGitHubActionsForHead\(\s*headSha,\s*branchName,\s*\{ requireWorkflowRun, timeoutSeconds = CI_DISCOVERY_TIMEOUT_SECONDS \},\s*\)/,
  );
  assert.match(script, /const runs = listWorkflowRunsForHead\(headSha, branchName\)\.filter\(\(run\) => run\.head_sha === headSha\);/);
  assert.match(script, /const latestRuns = selectLatestWorkflowRunsByWorkflow\(runs\);/);
  assert.match(script, /const failedRuns = latestRuns\.filter\(/);
  assert.match(script, /const pendingRuns = latestRuns\.filter\(/);
  assert.match(script, /String\(Math\.min\(CI_DISCOVERY_POLL_SECONDS, remainingSeconds\)\)/);
  assert.match(script, /pending: pendingRuns\.length > 0,/);
  assert.match(script, /return \{\s*ok: true,\s*headSha,\s*branchName,\s*workflowRuns: latestRuns,/);
  assert.match(script, /missing: latestRuns\.length === 0,/);
});

test("autoloop script feeds failed CI logs back into codex repair prompts", async () => {
  const script = await fs.readFile(new URL("../scripts/autoloop.mjs", import.meta.url), "utf8");
  assert.match(script, /failureContext = \{\s*[\s\S]*failedLog: ci\.failedLog,/);
  assert.match(
    script,
    /if \(failureContext\) \{\s*selectedIdea = await requestFixIdea\(repoContext, failureContext, failureRepairPaths, automaticRepairFailure\);/,
  );
  assert.match(script, /"Failed log excerpt:",\s*clampText\(failureContext\.failedLog, 20000\)/);
  assert.match(script, /let failedLogChars = failureContext \? 18000 : 0;/);
  assert.match(script, /failureContext \? `Failed CI log excerpt:\\n\$\{clampText\(failureContext\.failedLog, failedLogChars\)\}` : ""/);
  assert.match(script, /function readFailedWorkflowRunLog\(runId\)/);
  assert.match(script, /const failedLog = clampText\(readFailedWorkflowRunLog\(runId\), 18000\);/);
});

test("autoloop polls unresolved Copilot review feedback before autonomous ideas", async () => {
  const script = await fs.readFile(new URL("../scripts/autoloop.mjs", import.meta.url), "utf8");
  const workflow = await fs.readFile(new URL("../.github/workflows/autoloop.yml", import.meta.url), "utf8");

  assert.match(script, /const AI_REVIEW_POLL_ENABLED = !readBooleanEnv\(process\.env\.AUTOLOOP_DISABLE_AI_REVIEW_POLL\);/);
  assert.match(script, /const AI_REVIEW_LOOKBACK_PRS = clampInt\(process\.env\.AUTOLOOP_AI_REVIEW_LOOKBACK_PRS, 20, 1, 50\);/);
  assert.match(script, /phase: "copilot-review-poll"/);
  assert.match(script, /reviewFeedbackContext = pollGitHubAiReviewFeedback\(\);/);
  assert.match(script, /let reviewFeedbackContext = null;[\s\S]*phase: "copilot-review-poll"[\s\S]*if \(automaticRepair\)/);
  assert.match(script, /phase: "copilot-review-analysis"/);
  assert.match(script, /selectedIdea = await requestReviewFeedbackSelection\(repoContext, actionableReviewFeedbackContext\);/);
  assert.match(script, /No actionable Copilot\/Codex review feedback selected/);
  assert.match(script, /actionableReviewFeedbackContext = null;/);
  assert.match(script, /selectedIdea = await requestIdeaSelection\(repoContext\);/);
  assert.match(script, /function pollGitHubAiReviewFeedback\(\)/);
  assert.match(script, /reviewThreads\(first: 50\)/);
  assert.match(script, /if \(thread\?\.isResolved \|\| thread\?\.isOutdated\) return null;/);
  assert.match(script, /function isAiReviewAuthor\(login\)/);
  assert.match(script, /normalized\.includes\("copilot"\) \|\| normalized\.includes\("chatgpt-codex"\)/);
  assert.match(script, /function requestReviewFeedbackSelection\(repoContext, reviewFeedbackContext\)/);
  assert.match(script, /You must analyze whether a thread is correct before selecting it\./);
  assert.match(script, /algorithmReviewPath must be a reviewed editable file or another directly relevant editable file/);
  assert.match(script, /algorithmReviewPrefixes: ALLOWED_EDIT_PREFIXES/);
  assert.match(script, /function deriveAiReviewRepairPaths\(reviewFeedbackContext\)/);
  assert.match(script, /requestPatchPlan\(repoContext, idea, editableFiles, failureContext, actionableReviewFeedbackContext\)/);
  assert.match(script, /GitHub AI review feedback being considered/);
  assert.match(script, /Keep the change centered on the selected GitHub AI review feedback/);
  assert.match(workflow, /pull-requests: read/);
});

test("autoloop script repairs the latest remote branch head before proposing new work", async () => {
  const script = await fs.readFile(new URL("../scripts/autoloop.mjs", import.meta.url), "utf8");
  assert.match(script, /let failureContext = await inspectLatestRemoteBranchFailureContext\(\);/);
  assert.match(script, /Latest remote \$\{failureContext\.branchName\} commit \$\{failureContext\.headSha\} has failing GitHub Actions\./);
  assert.match(script, /logFailureRepairContext\("Repairing latest failing GitHub Actions", failureContext\);/);
  assert.match(script, /async function inspectLatestRemoteBranchFailureContext\(\)/);
  assert.match(script, /const latestHeadSha = readRemoteBranchHead\(LOOP_BRANCH\) \|\| readRemoteBranchHead\(BASE_BRANCH\);/);
  assert.match(
    script,
    /const ci = pollGitHubActionsForHead\(latestHeadSha, LOOP_BRANCH, \{\s*requireWorkflowRun: false,\s*timeoutSeconds: FAILURE_DISCOVERY_TIMEOUT_SECONDS,\s*\}\);/,
  );
  assert.match(script, /if \(ci\.pending\) \{\s*return \{\s*pendingCi: true,\s*branchName: LOOP_BRANCH,\s*headSha: latestHeadSha,/);
  assert.match(script, /outcome: "skipped_pending_ci"/);
  assert.match(script, /changedPaths: listCommitChangedPaths\(latestHeadSha\),/);
  assert.match(script, /function logFailureRepairContext\(prefix, failureContext\)/);
  assert.match(script, /console\.log\(`\$\{prefix\} for \$\{branchName\} @ \$\{headSha\}\$\{runUrl\} \$\{logState\}`\);/);
});

test("autoloop script auto-heals formatting-only CI failures on editable Haskell files", async () => {
  const script = await fs.readFile(new URL("../scripts/autoloop.mjs", import.meta.url), "utf8");
  assert.match(script, /const FOURMOLU_CHECK_COMMAND = "cd haskell && find app test bench -name '\*\.hs' -print0 \| xargs -0 fourmolu --mode check";/);
  assert.match(script, /const failureRepairPaths = deriveFailureRepairPaths\(failureContext\);/);
  assert.match(script, /const automaticRepair = failureContext \? detectAutomaticRepair\(failureContext\) : null;/);
  assert.match(script, /phase: "auto-repair"/);
  assert.match(script, /let automaticRepairFailure = "";/);
  assert.match(script, /try \{\s*applyAutomaticRepair\(automaticRepair\);/);
  assert.match(script, /Automatic .* repair failed; falling back to semantic repair\./);
  assert.match(script, /phase: "auto-repair-fallback"/);
  assert.match(script, /if \(!automaticRepair \|\| automaticRepairFailure\)/);
  assert.match(script, /function parseFourmoluFailurePaths\(failedLog\)/);
  assert.match(script, /const isFourmoluFailure = \/\\bfourmolu --mode check\\b\/\.test\(failedLog\);/);
  assert.match(
    script,
    /Trust the failed formatter log over commit changed-path metadata because CI[\s\S]*const fourmoluPaths = parseFourmoluFailurePaths\(failedLog\);/,
  );
  assert.match(script, /type: "fourmolu"/);
  assert.match(script, /function applyAutomaticRepair\(repair\)/);
  assert.match(script, /runCommand\("fourmolu", \["-i", \.\.\.relPaths\], \{/);
  assert.match(script, /verificationCommands: planVerificationCommands\(fourmoluPaths, \[FOURMOLU_CHECK_COMMAND\]\),/);
  assert.match(script, /function hasHaskellParserFailure\(logText\)/);
  assert.match(script, /function deriveParserFailurePaths\(\.\.\.logTexts\)/);
});

test("autoloop script auto-heals hlint-only CI failures on editable Haskell files", async () => {
  const script = await fs.readFile(new URL("../scripts/autoloop.mjs", import.meta.url), "utf8");
  assert.match(script, /const HLINT_CHECK_COMMAND = "cd haskell && bash scripts\/hlint_check\.sh";/);
  assert.match(script, /SAFE_VERIFICATION_COMMANDS = new Set\(\[[\s\S]*HLINT_CHECK_COMMAND/);
  assert.match(script, /function stripHlintBlockIndent\(blockText\)/);
  assert.match(script, /function parseHlintFailureEntries\(failedLog\)/);
  assert.ok(
    script.includes(
      'const isHlintFailure = /\\bhlint\\b/.test(failedLog) && /\\b(?:app|test|bench)\\/[A-Za-z0-9_./-]+\\.hs:/.test(failedLog);',
    ),
  );
  assert.match(script, /type: "hlint"/);
  assert.match(script, /verificationCommands: planVerificationCommands\(hlintPaths, \[HLINT_CHECK_COMMAND\]\),/);
  assert.match(script, /suggestions: hlintEntries/);
  assert.match(script, /if \(repair.type === "hlint"\)/);
  assert.match(script, /applyHlintSuggestions\(repair\.suggestions \|\| \[\]\);/);
  assert.match(script, /function applyHlintSuggestions\(suggestions\)/);
  assert.match(script, /function replaceHlintSuggestion\(content, suggestion\)/);
  assert.match(script, /function findSnippetNearLine\(content, snippet, startLine\)/);
  assert.match(script, /Unable to apply hlint suggestion/);
});

test("autoloop script promotes failing log paths into generic self-heal scope", async () => {
  const script = await fs.readFile(new URL("../scripts/autoloop.mjs", import.meta.url), "utf8");
  assert.match(script, /const MAX_EDITABLE_FILE_BYTES = clampInt\(process\.env\.AUTOLOOP_MAX_FILE_BYTES, 1000000, 4000, 5000000\);/);
  assert.match(script, /const PATCH_PLAN_PROMPT_MAX_CHARS = clampInt\(process\.env\.AUTOLOOP_PATCH_PLAN_MAX_CHARS, 2000000, 200000, 3000000\);/);
  assert.match(script, /function parseFailureReferencedPaths\(failedLog\)/);
  assert.match(script, /function deriveFailureRepairPaths\(failureContext\)/);
  assert.match(script, /const reviewRepairPaths = deriveAiReviewRepairPaths\(actionableReviewFeedbackContext\);/);
  assert.match(script, /const repoContext = await buildRepoContext\(uniqueStrings\(\[\.\.\.failureRepairPaths, \.\.\.reviewRepairPaths\]\)\);/);
  assert.match(script, /await requestFixIdea\(repoContext, failureContext, failureRepairPaths, automaticRepairFailure\)/);
  assert.match(script, /buildFailureRepairIdea\(failureContext, failureRepairPaths, automaticRepairFailure\) \|\| selectedIdea/);
  assert.match(script, /const editableFiles = await readEditableFiles\(idea\.filesNeeded\);/);
  assert.match(script, /Self-heal is required for any actionable failure or error when the failed log names editable files\./);
  assert.match(script, /Failure-targeted editable files: \$\{failureRepairPaths\.join\(", "\) \|\| "\([^"]+\)"\}/);
  assert.match(script, /Automatic repair failure: \$\{clampText\(automaticRepairFailure, 4000\)\}/);
  assert.match(script, /if \(prompt\.length > PATCH_PLAN_PROMPT_MAX_CHARS\)/);
  assert.match(script, /Patch-plan prompt is \$\{prompt\.length\} chars, above AUTOLOOP_PATCH_PLAN_MAX_CHARS=/);
  assert.match(script, /The failed CI log shows parser-level Haskell errors in editable files, so the loop must restore valid syntax\/module structure/);
  assert.match(script, /If the failed log shows parser-level Haskell errors, restore valid syntax, module headers, import\/export structure, and declaration shape/);
  assert.match(script, /When parser-failing files are named, filesNeeded must include those parser-failing files first/);
  assert.match(script, /Parser-failing editable files: \$\{parserFailurePaths\.join\(", "\)\}/);
  assert.match(script, /Parser-failing files that must be made parseable first: \$\{parserFailurePaths\.join\(", "\)\}/);
  assert.match(script, /verificationCommands: planVerificationCommands\(filesNeeded, syntaxRepairRequired \? \["cd haskell && cabal build", FOURMOLU_CHECK_COMMAND\] : \[\]\),/);
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
  assert.match(script, /const CODEX_EXEC_TIMEOUT_MS = clampInt\(process\.env\.AUTOLOOP_CODEX_TIMEOUT_MS, 420000, 10000, 1800000\);/);
  assert.match(script, /const CODEX_PATCH_TIMEOUT_MS = clampInt\(\s*process\.env\.AUTOLOOP_CODEX_PATCH_TIMEOUT_MS,\s*1800000,\s*CODEX_EXEC_TIMEOUT_MS,\s*3600000,\s*\);/);
  assert.match(script, /const CODEX_RETRY_MAX_ATTEMPTS = clampInt\(process\.env\.AUTOLOOP_CODEX_RETRY_MAX_ATTEMPTS, 2, 1, 5\);/);
  assert.match(script, /const CODEX_RETRY_BACKOFF_MS = clampInt\(process\.env\.AUTOLOOP_CODEX_RETRY_BACKOFF_MS, 30000, 1000, 300000\);/);
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
  assert.match(script, /Model returned invalid JSON/);
  assert.match(script, /function isModelJsonParseError\(err\)/);
  assert.match(script, /Patch plan returned invalid JSON after retry/);
  assert.match(script, /callModelJson\(\{ prompt, maxOutputTokens: 12000, timeoutMs: CODEX_PATCH_TIMEOUT_MS \}\)/);
  assert.match(script, /parseJsonResponse\(extractCodexExecLastMessage\(rawEvents\)\)/);
  assert.doesNotMatch(script, /--output-last-message/);
});

test("autoloop main loop gracefully degrades on retryable codex exec errors", async () => {
  const script = await fs.readFile(new URL("../scripts/autoloop.mjs", import.meta.url), "utf8");
  assert.match(script, /try \{\s*if \(failureContext\)/);
  assert.match(script, /\} catch \(err\) \{/);
  assert.match(script, /if \(isRetryableCodexExecError\(err\)\)/);
  assert.match(script, /outcome: "no_patch_plan"/);
  assert.match(script, /Skipping cycle gracefully/);
  assert.match(script, /return;/);
  assert.match(script, /throw err;/);
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

test("logical-correctness loop starts with a backend trading algorithm audit", async () => {
  const script = await fs.readFile(new URL("../scripts/codex-logical-correctness-loop.sh", import.meta.url), "utf8");
  assert.match(script, /Audit the Haskell trading algorithm and detect anything that could be made more logical and correct\./);
  assert.match(script, /Start with the backend trading algorithm: signal gates, predictors, optimizer behavior, position\/risk management, market-state inference, backtest\/live parity, and cost\/risk accounting\./);
  assert.match(script, /Implement fixes for all of the trading-algorithm logic\/correctness issues listed below\./);
  assert.match(script, /Keep the implementation centered on backend Haskell trading correctness unless the finding explicitly proves another file is required\./);
});

test("autoloop forever script auto-snapshots recoverable dirty cycles before blocking", async () => {
  const script = await fs.readFile(new URL("../scripts/autoloop-forever.mjs", import.meta.url), "utf8");
  assert.match(script, /const dirtyRecovery = await tryAutoSnapshotDirtyCycle\(\);/);
  assert.match(script, /const dirtyCheckpoint = await tryAutoCheckpointDirtyWorktree\(\);/);
  assert.match(script, /runCommand\("git", \["status", "--porcelain"\], \{ trimOutput: false \}\)/);
  assert.match(script, /function buildDirtyRecoveryBranchName\(kind, rawLabel = ""\)/);
  assert.match(script, /function pushHeadToRecoveryBranch\(branchName\)/);
  assert.match(script, /function restoreBaseBranchAfterRecovery\(recoveryBranch\)/);
  assert.match(script, /runCommand\("git", \["push", "-u", "origin", `HEAD:refs\/heads\/\$\{branchName\}`\], \{ capture: false \}\)/);
  assert.match(script, /runCommand\("git", \["checkout", "-b", recoveryBranch\], \{ capture: false \}\)/);
  assert.match(script, /runCommand\("git", \["checkout", "-b", checkpointBranch\], \{ capture: false \}\)/);
  assert.match(script, /cycle [^`]*recovery=\$\{dirtyRecovery\?\.recovered \? dirtyRecovery\.branch : "none"\}/);
  assert.match(script, /cycleStatus\?\.phase !== "error" \|\| changedPaths\.length === 0/);
  assert.match(script, /dirty worktree does not exactly match the last failed cycle changedPaths/);
  assert.match(script, /recoveryBranch,\s*recoveryMode: "recovery-branch"/);
  assert.match(script, /auto-committed failed dirty cycle to/);
  assert.match(script, /recoveryPushed: pushResult\.pushed/);
  assert.match(script, /recoveryBaseSync: baseSync/);
  assert.match(script, /auto-committed dirty worktree to/);
  assert.match(script, /branch: checkpointBranch/);
  assert.match(script, /baseSync,/);
});

test("autoloop forever script reconciles every unmerged branch onto main before bounded cycles", async () => {
  const script = await fs.readFile(new URL("../scripts/autoloop-forever.mjs", import.meta.url), "utf8");
  assert.match(script, /const BASE_BRANCH = normalizeGitBranchShortName\(process\.env\.AUTOLOOP_BASE_BRANCH \|\| "main"\) \|\| "main";/);
  assert.match(script, /const branchSweep = await reconcileUnmergedBranchesOntoBaseBranch\(\);/);
  assert.match(script, /runCommand\("git", \["fetch", "origin", "--prune"\], \{ capture: false \}\);/);
  assert.match(
    script,
    /buildBranchMergeCandidates\(\{\s*localBranches,\s*remoteBranches,\s*baseBranch: BASE_BRANCH,\s*\}\)/,
  );
  assert.match(script, /buildAutoloopScratchBranchCandidates\(\{/);
  assert.match(script, /runCommand\("git", \["branch", "--format=%\(refname:short\)", "--no-merged", BASE_BRANCH\], \{ trimOutput: false \}\)/);
  assert.match(script, /runCommand\("git", \["branch", "-r", "--format=%\(refname:short\)", "--no-merged", BASE_BRANCH\], \{ trimOutput: false \}\)/);
  assert.match(script, /function buildMergeCommitMessage\(shortName = "", branchRef = ""\)/);
  assert.match(script, /return `autoloop: sync \$\{BASE_BRANCH\} with origin\/\$\{BASE_BRANCH\}`;/);
  assert.match(script, /return `autoloop: merge \$\{shortName \|\| branchRef\} into \$\{BASE_BRANCH\}`;/);
  assert.match(script, /function rebaseBaseBranchOntoOrigin\(\)/);
  assert.match(script, /runCommand\("git", \["rebase", remoteRef\], \{ capture: false \}\)/);
  assert.match(script, /runCommand\("git", \["rebase", "--abort"\], \{ capture: false \}\)/);
  assert.match(script, /outcome: "rebased"/);
  assert.match(script, /const mergeArgs = \["merge", "--no-ff", "-m", mergeMessage, branchRef\];/);
  assert.match(script, /runCommand\("git", mergeArgs, \{ capture: false \}\)/);
  assert.doesNotMatch(script, /"-s", "ours"/);
  // Conflicts are NEVER auto-resolved (the per-file restore --source=HEAD
  // policy produced semantically torn trees that broke main twice: PR #147,
  // PR #150). The merge must abort and flag the branch for an operator.
  assert.doesNotMatch(script, /restore", "--source=HEAD", "--staged", "--worktree", "--", \.\.\.conflicts/);
  assert.doesNotMatch(script, /buildConflictResolutionCommitMessage/);
  assert.match(script, /runCommand\("git", \["merge", "--abort"\], \{ capture: false \}\);/);
  assert.match(script, /outcome: "conflict-aborted"/);
  assert.match(script, /const conflictAbortedBranches = \[\];/);
  assert.match(script, /if \(originSync\.outcome === "conflict-aborted"\) \{/);
  assert.match(script, /automated resolution is disabled, operator must reconcile/);
  assert.match(script, /unmerged due to conflicts — operator review required/);
  // The push retry must also stop instead of resolving: when the retry sync
  // conflict-aborts, nothing may be pushed on top of a diverged base, and the
  // reconciliation must fail the cycle (ok: false) so the loop halts now
  // rather than on the next cycle.
  assert.match(script, /if \(retrySync\.outcome === "conflict-aborted"\) \{/);
  assert.match(script, /return \{ pushed: false, retried: true, retrySync \};/);
  assert.match(script, /could not push \$\{BASE_BRANCH\}: origin moved with conflicting changes/);
  assert.match(script, /after a push retry; automated resolution is disabled, operator must reconcile/);
  // The push-conflict guard must run BEFORE the merged-ref prune: pruning on
  // a base whose merges never reached origin deletes the only branch refs
  // for that work.
  assert.ok(
    script.indexOf('if (pushResult.retrySync?.outcome === "conflict-aborted") {') <
      script.indexOf("const pruneResult = pruneMergedRefsOnBaseBranch(BASE_BRANCH);"),
    "push-conflict guard must precede pruneMergedRefsOnBaseBranch",
  );
  assert.match(script, /runCommand\("git", \["push", "origin", `\$\{BASE_BRANCH\}:refs\/heads\/\$\{BASE_BRANCH\}`\], \{ capture: false \}\)/);
  assert.match(script, /const pruneResult = pruneMergedRefsOnBaseBranch\(BASE_BRANCH\);/);
  assert.match(script, /runCommand\("git", \["worktree", "prune"\], \{ capture: false \}\);/);
  assert.match(script, /runCommand\("git", \["branch", "--format=%\(refname:short\)", "--merged", baseBranch\], \{ trimOutput: false \}\)/);
  assert.match(script, /runCommand\("git", \["branch", "-r", "--format=%\(refname:short\)", "--merged", baseBranch\], \{ trimOutput: false \}\)/);
  assert.match(
    script,
    /buildAutoloopScratchBranchCandidates\(\{\s*localBranches,\s*remoteBranches,\s*baseBranch,\s*\}\)/,
  );
  assert.doesNotMatch(script, /const allLocalBranches =/);
  assert.doesNotMatch(script, /const allRemoteBranches =/);
  assert.match(script, /runCommand\("git", \["worktree", "list", "--porcelain"\], \{ trimOutput: false \}\)/);
  assert.match(script, /const worktreeBranches = listWorktreeBranches\(\);/);
  assert.match(script, /scratchCandidateBranches: pruneResult\.scratchCandidateBranches/);
  assert.match(script, /if \(worktreeBranches\.has\(candidate\.shortName\)\) \{/);
  assert.match(script, /skippedWorktreeBranches\.push\(candidate\.shortName\);/);
  assert.match(script, /runCommand\("git", \["push", "origin", "--delete", candidate\.shortName\], \{ capture: false \}\)/);
  assert.match(script, /runCommand\("git", \["branch", "-D", candidate\.shortName\], \{ capture: false \}\)/);
  assert.match(script, /prunedLocalBranches: pruneResult\.prunedLocalBranches/);
  assert.match(script, /prunedRemoteBranches: pruneResult\.prunedRemoteBranches/);
  assert.match(script, /skippedWorktreeBranches: pruneResult\.skippedWorktreeBranches/);
  assert.match(script, /pruneErrors: pruneResult\.pruneErrors/);
  assert.match(script, /branch reconciliation pruned \$\{pruneResult\.prunedLocalBranches\.length\} local and \$\{pruneResult\.prunedRemoteBranches\.length\} remote merged ref\(s\)/);
  assert.match(script, /branch reconciliation skipped \$\{pruneResult\.skippedWorktreeBranches\.length\} merged local ref\(s\) still attached to worktrees/);
  assert.match(script, /runCommand\("bash", \["scripts\/verify\.sh", target\], \{ capture: false \}\)/);
  assert.match(script, /canonical merge verification failed; rolled back/);
  assert.ok(
    script.indexOf("const canonicalVerification =") < script.indexOf("const shouldPush ="),
    "canonical verification must precede pushing the reconciled base branch",
  );
});

test("autoloop workflow requires a dedicated push token and never skips post-push CI polling", async () => {
  const workflow = await fs.readFile(new URL("../.github/workflows/autoloop.yml", import.meta.url), "utf8");
  assert.match(workflow, /contents:\s+write/);
  assert.doesNotMatch(workflow, /pull-requests:\s+write/);
  assert.match(workflow, /name:\s+Require dedicated push token/);
  assert.match(workflow, /AUTOLOOP_PUSH_TOKEN is required so autoloop pushes trigger downstream CI and can poll GitHub Actions to repair failures\./);
  assert.match(workflow, /name:\s+Install HLint/);
  assert.match(workflow, /sudo apt-get update && sudo apt-get install -y hlint/);
  assert.match(workflow, /name:\s+Install fourmolu/);
  assert.match(workflow, /curl --fail --silent --show-error --location/);
  assert.match(workflow, /https:\/\/github\.com\/fourmolu\/fourmolu\/releases\/download\/v0\.15\.0\.0\/fourmolu-0\.15\.0\.0-linux-x86_64/);
  assert.match(workflow, /chmod \+x "\$HOME\/\.local\/bin\/fourmolu"/);
  assert.match(workflow, /name:\s+Show fourmolu version/);
  assert.match(workflow, /run:\s+fourmolu --version/);
  assert.match(workflow, /token:\s+\$\{\{\s*secrets\.AUTOLOOP_PUSH_TOKEN\s*\}\}/);
  assert.match(workflow, /GITHUB_TOKEN:\s+\$\{\{\s*secrets\.AUTOLOOP_PUSH_TOKEN\s*\}\}/);
  assert.doesNotMatch(workflow, /AUTOLOOP_SKIP_CI_WAIT:/);
  assert.doesNotMatch(workflow, /github\.token/);
});

test("CI Fly deploy skips external billing blockers", async () => {
  const workflow = await fs.readFile(new URL("../.github/workflows/ci.yml", import.meta.url), "utf8");
  assert.match(workflow, /overdue invoices\|update your payment information/);
  assert.match(workflow, /blocked by Fly billing state/);
  assert.match(workflow, /Skipping \$\{label\} deploy for this run/);
});

test("CI Fly deploy gives live position recovery one uninterrupted window", async () => {
  const workflow = await fs.readFile(new URL("../.github/workflows/ci.yml", import.meta.url), "utf8");
  assert.match(workflow, /--wait-timeout 60m/);
  assert.match(workflow, /timeout reached waiting for health checks to pass/);
  assert.match(workflow, /leaving the recovering machine in place without retrying/);
  assert.ok(
    workflow.indexOf("timeout reached waiting for health checks to pass") <
      workflow.indexOf("503 Service Unavailable|502 Bad Gateway|504 Gateway Timeout|failed to set release status"),
    "application readiness timeouts must be handled before transient transport failures",
  );
});

test("CI pins the checked-in GHC and Cabal toolchain", async () => {
  for (const relativePath of ["../.github/workflows/ci.yml", "../.github/workflows/autoloop.yml"]) {
    const workflow = await fs.readFile(new URL(relativePath, import.meta.url), "utf8");
    assert.match(workflow, /ghc:\s*"9\.4\.8"/);
    assert.match(workflow, /cabal:\s*"3\.12\.1\.0"/);
    assert.doesNotMatch(workflow, /cabal:\s*latest/);
  }
});

test("CI Fly deploy reports app-scoped token fixes for unauthorized apps", async () => {
  const workflow = await fs.readFile(new URL("../.github/workflows/ci.yml", import.meta.url), "utf8");
  assert.match(workflow, /unauthorized\|not authorized\|permission denied\|access denied/);
  assert.match(workflow, /token validation error\|no verified tokens/);
  assert.match(workflow, /FLY_API_TOKEN_FRONTEND failed Fly authentication; retrying frontend deploy with FLY_API_TOKEN/);
  assert.match(workflow, /Set GitHub secret \$\{token_secret_name\} to a Fly deploy token scoped to that app/);
  assert.match(workflow, /FLY_API_TOKEN_RESEARCH/);
  assert.match(workflow, /FLY_API_TOKEN_FRONTEND/);
});

test("Hetzner deploy retries SSH failures and deploys only green commits", async () => {
  const workflow = await fs.readFile(new URL("../.github/workflows/deploy-hetzner.yml", import.meta.url), "utf8");
  const deployScript = await fs.readFile(new URL("../deploy/hetzner/deploy-remote.sh", import.meta.url), "utf8");
  const dockerfile = await fs.readFile(new URL("../Dockerfile", import.meta.url), "utf8");
  const rollupScript = await fs.readFile(
    new URL("../haskell/scripts/rollup_performance.sh", import.meta.url),
    "utf8",
  );

  assert.match(workflow, /is_transient_ssh_failure\(\)/);
  assert.match(workflow, /workflow_dispatch:/);
  assert.match(workflow, /name:\s+Resolve deploy commit/);
  assert.match(workflow, /--workflow "CI"/);
  assert.match(workflow, /--status success/);
  assert.match(workflow, /No successful CI run found for \$\{branch\}; refusing to deploy an unverified commit\./);
  assert.match(workflow, /ref:\s+\$\{\{\s*needs\.resolve-deploy\.outputs\.deploy_sha\s*\}\}/);
  assert.match(workflow, /TRADER_GIT_COMMIT:\s+\$\{\{\s*needs\.resolve-deploy\.outputs\.deploy_sha\s*\}\}/);
  assert.doesNotMatch(workflow, /github\.event\.workflow_run\.head_sha \|\| github\.sha/);
  assert.match(workflow, /for attempt in 1 2 3; do/);
  assert.match(workflow, /Transient \$\{ROLE\} Hetzner SSH failure on attempt \$\{attempt\}; retrying\./);
  assert.match(workflow, /Hetzner \$\{ROLE\} deploy failed after \$\{last_attempt\} attempt\(s\)\./);
  assert.match(workflow, /deploy exited successfully without remote commit attestation/);
  assert.match(workflow, /Deploy healthy and commit-attested \(\$\{TRADER_GIT_COMMIT\}\)/);
  assert.doesNotMatch(workflow, /HETZNER_RESEARCH_REQUIRED/);
  assert.doesNotMatch(workflow, /skipping research deploy/);
  assert.match(workflow, /HETZNER_\$\{ROLE\^\^\}_KNOWN_HOSTS is required; host-key TOFU is not permitted/);
  assert.doesNotMatch(workflow, /ssh-keyscan/);

  assert.match(deployScript, /TRADER_HETZNER_SSH_CONNECT_TIMEOUT/);
  assert.match(deployScript, /-o StrictHostKeyChecking=yes/);
  assert.match(deployScript, /-o "ConnectTimeout=\$\{ssh_connect_timeout\}"/);
  assert.match(deployScript, /-o "ConnectionAttempts=\$\{ssh_connection_attempts\}"/);
  assert.match(deployScript, /--exclude '\.cabal\/'/);
  assert.match(deployScript, /--exclude '\.git'/);
  assert.match(deployScript, /if \[\[ -f \.git \]\]; then[\s\S]*?rm -f \.git/);
  assert.match(deployScript, /--exclude 'haskell\/\.stack-root\/'/);
  assert.match(deployScript, /--exclude 'haskell\/\.stack-work\/'/);
  assert.match(deployScript, /--exclude '\.venv\/'/);
  assert.match(deployScript, /rsync -az --delete --human-readable/);
  assert.match(deployScript, /haskell\/web\/dist\//);
  assert.match(deployScript, /docker image tag "\$previous_image_id" "\$ROLLBACK_IMAGE"/);
  assert.match(
    deployScript,
    /exec -T api curl -fsS --max-time 5 http:\/\/127\.0\.0\.1:8080\/health <\/dev\/null/,
  );
  assert.match(deployScript, /wait_for_api_health/);
  assert.match(deployScript, /\/health reported commit/);
  assert.match(deployScript, /Remote deployment started for \$\{TRADER_GIT_COMMIT\}/);
  assert.match(deployScript, /\"\$\{compose\[@\]\}\" build api/);
  assert.match(deployScript, /run --rm --no-deps caddy caddy validate --config \/etc\/caddy\/Caddyfile --adapter caddyfile/);
  assert.match(deployScript, /--force-recreate api/);
  assert.match(deployScript, /--force-recreate caddy/);
  assert.match(deployScript, /exec -T caddy caddy validate --config \/etc\/caddy\/Caddyfile --adapter caddyfile/);
  assert.match(deployScript, /Rolling API back to/);
  assert.match(deployScript, /TRADER_API_IMAGE="\$ROLLBACK_IMAGE"/);
  assert.match(dockerfile, /postgresql-client/);
  assert.match(dockerfile, /COPY haskell\/scripts\/rollup_performance\.sh \/usr\/local\/bin\/rollup-performance/);
  assert.match(deployScript, /exec rollup-performance/);
  assert.match(deployScript, /TRADER_OPS_ROLLUP_ON_DEPLOY/);
  assert.match(deployScript, /until "\$\{compose\[@\]\}" exec -T api/);
  assert.match(deployScript, /rollup_attempt >= 3/);
  assert.match(deployScript, /retrying the transaction/);
  assert.match(rollupScript, /BEGIN;/);
  assert.match(rollupScript, /pg_advisory_xact_lock/);
  assert.match(rollupScript, /LOCK TABLE platform_symbols IN SHARE ROW EXCLUSIVE MODE;/);
  assert.match(rollupScript, /COMMIT;/);
});

test("docs pin the mandatory Hetzner deploy contract for both roles", async () => {
  const readme = await fs.readFile(new URL("../README.md", import.meta.url), "utf8");
  const changelog = await fs.readFile(new URL("../CHANGELOG.md", import.meta.url), "utf8");

  assert.match(readme, /both trading and research boxes are mandatory/);
  assert.match(readme, /latest green commit/);
  assert.match(changelog, /both trading and research deploys mandatory/);
  assert.match(changelog, /latest green commit/);
  assert.doesNotMatch(readme, /HETZNER_RESEARCH_REQUIRED/);
  assert.doesNotMatch(changelog, /HETZNER_RESEARCH_REQUIRED/);

  // Mandatory research deploys must not relax live risk: the adoption
  // maxPositionSize cap stays at 0.25.
  assert.match(changelog, /maxPositionSize` at 0\.25/);
});

test("top-combo sync retention is independent from optimizer retention", async () => {
  const main = await fs.readFile(new URL("../haskell/app/Main.hs", import.meta.url), "utf8");
  const sourceSection = (startNeedle, length, from = 0) => {
    const start = main.indexOf(startNeedle, from);
    assert.notEqual(start, -1, `expected Main.hs to contain ${startNeedle}`);
    return main.slice(start, start + length);
  };

  assert.match(main, /defaultTopCombosSyncMaxCombos\s*::\s*Int\s+defaultTopCombosSyncMaxCombos\s*=\s*5000/);
  assert.match(main, /lookupEnv "TRADER_TOP_COMBOS_SYNC_MAX_COMBOS"/);
  assert.match(main, /max defaultTopCombosSyncMaxCombos optimizerMaxCombos/);
  assert.match(main, /lookupEnv "TRADER_TOP_COMBOS_SYNC_INITIAL_DELAY_SEC"/);

  const syncLoop = sourceSection("topCombosSyncLoop ::", 4500);
  assert.match(syncLoop, /syncMaxCombos <- topCombosSyncMaxCombosFromEnv/);
  assert.match(syncLoop, /mergeTopCombosPayloads syncMaxCombos now candidates/);
  assert.doesNotMatch(syncLoop, /optimizerMaxCombosFromEnv/);

  const serverStartup = sourceSection("runRestApi ::", 50000);
  const botWorkerStart = serverStartup.indexOf('forkSupervisedWorker workers "bot-auto-start"');
  const syncWorkerStart = serverStartup.indexOf('forkSupervisedWorker workers "top-combos-sync"');
  assert.notEqual(botWorkerStart, -1, "expected server startup to launch the bot auto-start worker");
  assert.notEqual(syncWorkerStart, -1, "expected server startup to launch the top-combo sync worker");
  assert.ok(botWorkerStart < syncWorkerStart, "bot auto-start must launch before the delayed replica sync worker");

  const importHandlerStart = main.indexOf("handleStateSyncImport ::");
  assert.notEqual(importHandlerStart, -1, "expected Main.hs to contain handleStateSyncImport");
  const topCombosImport = sourceSection("case sspTopCombos payload of", 4500, importHandlerStart);
  assert.match(topCombosImport, /maxCombos <- topCombosSyncMaxCombosFromEnv/);
  assert.match(topCombosImport, /mergeTopCombosPayloads maxCombos now \[localVal, incomingSanitized\]/);
  assert.doesNotMatch(topCombosImport, /optimizerMaxCombosFromEnv/);

  const retentionConfigPaths = [
    "../.env.example",
    "../fly.toml",
    "../fly.research.toml",
    "../deploy/hetzner/trader.research.env.example",
    "../deploy/hetzner/trader.research.env.managed",
    "../deploy/hetzner/trader.trading.env.example",
    "../deploy/hetzner/trader.trading.env.managed",
  ];
  for (const relativePath of retentionConfigPaths) {
    const config = await fs.readFile(new URL(relativePath, import.meta.url), "utf8");
    assert.match(config, /TRADER_OPTIMIZER_MAX_COMBOS\s*(?:=|:)\s*"?5000"?/, `${relativePath} optimizer retention`);
    assert.match(
      config,
      /TRADER_TOP_COMBOS_SYNC_MAX_COMBOS\s*(?:=|:)\s*"?5000"?/,
      `${relativePath} sync retention`,
    );
  }

  const productionConfigPaths = [
    "../fly.toml",
    "../fly.research.toml",
    "../deploy/hetzner/trader.research.env.example",
    "../deploy/hetzner/trader.research.env.managed",
    "../deploy/hetzner/trader.trading.env.example",
    "../deploy/hetzner/trader.trading.env.managed",
  ];
  for (const relativePath of productionConfigPaths) {
    const config = await fs.readFile(new URL(relativePath, import.meta.url), "utf8");
    assert.match(
      config,
      /TRADER_TOP_COMBOS_SYNC_INITIAL_DELAY_SEC\s*(?:=|:)\s*"?30"?/,
      `${relativePath} initial sync delay`,
    );
  }

  const hetznerCompose = await fs.readFile(new URL("../deploy/hetzner/docker-compose.yml", import.meta.url), "utf8");
  assert.match(hetznerCompose, /TRADER_TOP_COMBOS_SYNC_INITIAL_DELAY_SEC: \$\{TRADER_TOP_COMBOS_SYNC_INITIAL_DELAY_SEC:-0\}/);
  assert.match(hetznerCompose, /TRADER_TOP_COMBOS_SYNC_MAX_COMBOS: \$\{TRADER_TOP_COMBOS_SYNC_MAX_COMBOS:-\}/);
});

test("top-combo PostgreSQL replication preserves UUID typing and portfolio evidence", async () => {
  const main = await fs.readFile(new URL("../haskell/app/Main.hs", import.meta.url), "utf8");

  const uuidArrayPredicates = main.match(/combo_uuid = ANY\(\?::uuid\[\]\)/g) ?? [];
  assert.ok(
    uuidArrayPredicates.length >= 2,
    "top-combo and live-evidence UUID array queries must bind uuid[] instead of driver-inferred text[]",
  );
  assert.doesNotMatch(main, /combo_uuid = ANY\(\?\)/, "combo UUID arrays must never rely on inferred PostgreSQL types");

  const persistBegin = main.indexOf("persistTopCombosToDbBulk ::");
  assert.notEqual(persistBegin, -1, "expected Main.hs to contain top-combo DB persistence");
  const persistSection = main.slice(persistBegin, persistBegin + 9000);
  assert.match(persistSection, /tcPortfolioEvidence combo/);
  assert.match(
    persistSection,
    /KM\.insert \(AK\.fromString "portfolioEvidence"\) \(toJSON evidence\) metricsWithFreshness/,
    "DB persistence must retain the OOS portfolio evidence used by canary selection",
  );
});

test("trading auto-start prioritizes recoverable positions without pinning later targets", async () => {
  const main = await fs.readFile(new URL("../haskell/app/Main.hs", import.meta.url), "utf8");
  const autoStartBegin = main.indexOf("botAutoStartLoop ::");
  assert.notEqual(autoStartBegin, -1, "expected Main.hs to contain botAutoStartLoop");
  const autoStartLoop = main.slice(autoStartBegin, autoStartBegin + 60000);

  assert.match(main, /lookupTrimmedEnv "TRADER_BOT_START_ADOPTION_RELAX_GATES"/);
  assert.match(main, /lookupTrimmedEnv "TRADER_BOT_START_ADOPTION_RELAX_TARGET_COUNT"/);
  assert.match(main, /lookupEnv "TRADER_BOT_ONLINE_OPTIMIZER_ENABLED"/);
  assert.match(main, /readBoundedIntEnv "TRADER_PORTFOLIO_SELECTOR_MAX_BOTS" 1 5/);
  assert.match(autoStartLoop, /targetSymbolsBase = dedupeStable \(topSymbols \+\+ liveBaseSymbols\)/);
  assert.match(autoStartLoop, /capBotStartSymbolsPreservingOrphans maxBots targetSymbolsBase orphanSymbols/);
  assert.match(autoStartLoop, /filterBotStartAttemptsPreservingOrphans\s+circuitOpen[\s\S]*?orphanSymbols\s+missingAll/);
  assert.match(autoStartLoop, /throttleBotStartSymbolsPreservingOrphans maxStartsPerCycle orphanSymbols eligibleMissing/);
  assert.ok(
    autoStartLoop.indexOf("filterBotStartAttemptsPreservingOrphans") <
      autoStartLoop.indexOf("throttleBotStartSymbolsPreservingOrphans maxStartsPerCycle"),
    "backoff filtering must happen before the start throttle so one backed-off symbol cannot pin the queue",
  );
  assert.match(autoStartLoop, /if not topComboAdoptionEnabled\s+then pure \(argsSym, Nothing\)/);
  assert.match(autoStartLoop, /forceEnvPreset <- botStartForceEnvPresetFromEnv[\s\S]*?applyBotStartupEnvPreset argsCombo/);
  assert.match(
    autoStartLoop,
    /if bsTradeEnabled settings\s+then resolveOrphanOpenPositionActions mOps argsWithKeys tenantMap0\s+else pure \(Right \(\[\], \[\]\)\)/,
  );
  const orphanScanIndex = autoStartLoop.indexOf("resolveOrphanOpenPositionActions mOps argsWithKeys tenantMap0");
  const portfolioSelectionIndex = autoStartLoop.indexOf("topTargets <-\n                                    if adoptionPriority");
  assert.ok(
    orphanScanIndex >= 0 && portfolioSelectionIndex > orphanScanIndex,
    "open positions must be inspected before portfolio selection",
  );
  assert.match(autoStartLoop, /if adoptionPriority\s+then pure \[\]\s+else loadTopTargets topComboTargetCount/);
  assert.match(autoStartLoop, /adoptionPrioritySymbols = dedupeStable \(orphanSymbols \+\+ adoptionStartingSymbols\)/);
  assert.match(autoStartLoop, /targetSymbols \+\+ orphanSymbols \+\+ adoptionStartingSymbols \+\+ locallyOpenSymbols/);
  assert.match(autoStartLoop, /startupPhase && orphanScanReady && not adoptionPriority/);
  assert.match(
    autoStartLoop,
    /writeIORef recoveryReadyRef \(orphanScanReady && null orphanSymbols && null adoptionStartingSymbols\)/,
  );
  assert.match(
    autoStartLoop,
    /writeIORef recoveryReadyRef \(orphanScanReady && null adoptionStartingSymbols && and registered\)/,
  );
  assert.match(
    autoStartLoop,
    /case effectivePortfolioMode of\s+PortfolioShadow -> do\s+writeIORef portfolioSelectionFailureRef Nothing\s+pure Nothing/,
  );
  assert.match(autoStartLoop, /\(PortfolioShadow, _\) -> independentTargets/);
  assert.match(autoStartLoop, /symbol `elem` map normalizeSymbol baseSymbols/);

  assert.match(main, /ON CONFLICT \(bot_id\) WHERE bot_id IS NOT NULL DO UPDATE/);

  const startupGuardBegin = main.indexOf("runTopComboStartupBacktestGuard ::");
  assert.notEqual(startupGuardBegin, -1, "expected Main.hs to contain the startup backtest guard");
  const startupGuard = main.slice(startupGuardBegin, startupGuardBegin + 2500);
  const disabledReturn = startupGuard.indexOf("not (tcbcEnabled ctx) = pure (Right ())");
  const comboLookup = startupGuard.indexOf("lookupTopComboValueByUuid");
  assert.ok(
    disabledReturn >= 0 && comboLookup > disabledReturn,
    "disabled startup guards must return before combo-store and database work",
  );
  assert.doesNotMatch(startupGuard, /bot\.start_combo_backtest_skipped/);

  for (const relativePath of [
    "../deploy/hetzner/trader.trading.env.example",
    "../deploy/hetzner/trader.trading.env.managed",
  ]) {
    const config = await fs.readFile(new URL(relativePath, import.meta.url), "utf8");
    assert.match(config, /TRADER_PORTFOLIO_SELECTOR_ROLLOUT_MODE=shadow/);
    assert.match(config, /TRADER_BOT_START_ADOPTION_RELAX_GATES=true/);
    assert.match(config, /TRADER_BOT_START_ADOPTION_MAX_POSITION_SIZE_CAP=0\.05/);
    assert.match(config, /TRADER_TOP_COMBO_DEPLOYABLE_OVERRIDE_UUIDS=/);
  }

  const compose = await fs.readFile(new URL("../deploy/hetzner/docker-compose.yml", import.meta.url), "utf8");
  assert.match(compose, /TRADER_BOT_START_ADOPTION_RELAX_GATES: \$\{TRADER_BOT_START_ADOPTION_RELAX_GATES:-\}/);
  assert.match(
    compose,
    /TRADER_BOT_START_ADOPTION_RELAX_TARGET_COUNT: \$\{TRADER_BOT_START_ADOPTION_RELAX_TARGET_COUNT:-\}/,
  );
  assert.match(
    compose,
    /TRADER_BOT_START_ADOPTION_MAX_POSITION_SIZE_CAP: \$\{TRADER_BOT_START_ADOPTION_MAX_POSITION_SIZE_CAP:-\}/,
  );
  assert.match(
    compose,
    /TRADER_TOP_COMBO_DEPLOYABLE_OVERRIDE_UUIDS: \$\{TRADER_TOP_COMBO_DEPLOYABLE_OVERRIDE_UUIDS:-\}/,
  );
  assert.match(
    compose,
    /TRADER_BOT_ONLINE_OPTIMIZER_ENABLED: \$\{TRADER_BOT_ONLINE_OPTIMIZER_ENABLED:-true\}/,
  );
  assert.match(
    compose,
    /TRADER_PORTFOLIO_AUTO_GRADUATE_ENABLED: \$\{TRADER_PORTFOLIO_AUTO_GRADUATE_ENABLED:-false\}/,
  );
  assert.match(compose, /TRADER_LSTM_REUSE_PERSISTED: \$\{TRADER_LSTM_REUSE_PERSISTED:-false\}/);
  assert.match(compose, /TRADER_EXECUTION_MAKER_FIRST: \$\{TRADER_EXECUTION_MAKER_FIRST:-true\}/);
  assert.match(compose, /TRADER_EXECUTION_MAKER_TIMEOUT_SEC: \$\{TRADER_EXECUTION_MAKER_TIMEOUT_SEC:-3\}/);
  assert.match(main, /lookupTrimmedEnv "TRADER_EXECUTION_MAKER_FIRST"/);
  assert.match(main, /"executionPath"/);

  const hetznerTrading = await fs.readFile(
    new URL("../deploy/hetzner/trader.trading.env.managed", import.meta.url),
    "utf8",
  );
  assert.match(hetznerTrading, /TRADER_BINANCE_LIVE=true/);
  assert.match(hetznerTrading, /TRADER_BOT_TRADE=true/);
  assert.match(hetznerTrading, /TRADER_BOT_AUTOSTART=true/);
  assert.match(
    hetznerTrading,
    /TRADER_BOT_SYMBOLS=AVAXUSDT,UNIUSDT,SUIUSDT,ETCUSDT,ADAUSDT/,
  );
  assert.match(hetznerTrading, /TRADER_BOT_TOP_COMBO_BOTS=5/);
  assert.match(hetznerTrading, /TRADER_BOT_TOP_COMBO_BOTS_STARTUP=5/);
  assert.match(hetznerTrading, /TRADER_BOT_AUTOSTART_MAX_BOTS=5/);
  assert.match(hetznerTrading, /TRADER_BOT_START_MAX_SYMBOLS=5/);
  assert.match(hetznerTrading, /TRADER_PORTFOLIO_SELECTOR_ROLLOUT_MODE=shadow/);
  assert.match(hetznerTrading, /TRADER_BOT_START_ADOPTION_RELAX_TARGET_COUNT=5/);
  assert.match(hetznerTrading, /TRADER_BOT_START_METHOD=ta_best/);
  assert.match(hetznerTrading, /TRADER_BOT_START_ADOPTION_MAX_POSITION_SIZE_CAP=0\.05/);
  assert.match(hetznerTrading, /TRADER_PORTFOLIO_SELECTOR_MAX_BOT_WEIGHT=0\.05/);
  assert.match(hetznerTrading, /TRADER_PORTFOLIO_SELECTOR_MAX_GROSS_WEIGHT=0\.25/);
  assert.match(hetznerTrading, /TRADER_PORTFOLIO_SELECTOR_MIN_DAYS=30/);
  assert.match(hetznerTrading, /TRADER_PORTFOLIO_AUTO_GRADUATE_ENABLED=true/);
  assert.match(hetznerTrading, /TRADER_PORTFOLIO_AUTO_GRADUATE_STARTED_AT_MS=1787322000000/);
  assert.match(hetznerTrading, /TRADER_PORTFOLIO_AUTO_GRADUATE_MIN_DAILY_OBSERVATIONS=30/);
  assert.match(hetznerTrading, /TRADER_PORTFOLIO_AUTO_GRADUATE_MIN_EXECUTION_RELIABILITY=0\.95/);
  assert.match(hetznerTrading, /TRADER_PORTFOLIO_AUTO_GRADUATE_MIN_STATUS_RELIABILITY=0\.99/);
  assert.match(hetznerTrading, /TRADER_TOP_COMBO_DEPLOYABLE_OVERRIDE_UUIDS=/);
  assert.match(hetznerTrading, /TRADER_BOT_ONLINE_OPTIMIZER_ENABLED=false/);
  assert.match(hetznerTrading, /TRADER_BOT_START_TOP_COMBO_ADOPTION=true/);
  assert.match(hetznerTrading, /TRADER_BOT_START_ALLOW_STALE_INCOMPLETE_COMBOS=true/);
  assert.match(hetznerTrading, /TRADER_BOT_START_FORCE_ENV_PRESET=false/);
  assert.match(hetznerTrading, /TRADER_EXECUTION_MAKER_FIRST=true/);
  assert.match(hetznerTrading, /TRADER_EXECUTION_MAKER_FALLBACK_MARKET=true/);
  assert.match(compose, /TRADER_BOT_START_ALLOW_STALE_INCOMPLETE_COMBOS: \$\{TRADER_BOT_START_ALLOW_STALE_INCOMPLETE_COMBOS:-false\}/);
  assert.match(main, /lookupEnv "TRADER_BOT_START_ALLOW_STALE_INCOMPLETE_COMBOS"/);
  assert.match(main, /resolvePortfolioGraduationMode/);
  assert.match(main, /"portfolio\.graduated"/);

  const fly = await fs.readFile(new URL("../fly.toml", import.meta.url), "utf8");
  assert.match(fly, /kill_signal = "SIGTERM"/);
  assert.match(fly, /kill_timeout = 30/);
  assert.match(fly, /path = "\/ready"/);
  assert.match(fly, /\[\[mounts\]\][\s\S]*?source = "trader_data"[\s\S]*?destination = "\/var\/lib\/trader"/);
  assert.match(fly, /\[\[mounts\]\][\s\S]*?processes = \["app"\]/);
  assert.match(fly, /TRADER_LSTM_REUSE_PERSISTED = "true"/);
  assert.match(fly, /TRADER_BOT_TRADE = "false"/);
  assert.doesNotMatch(fly, /app = "[^"]*--binance-live/);
  assert.match(main, /lookupEnv "TRADER_LSTM_REUSE_PERSISTED"/);
  assert.match(main, /Just seedModel \| reusePersisted -> \(seedModel, \[\]\)/);
  assert.match(main, /botRecoveryRequired = argBinanceLive baseArgs && botTradeEnabled/);
  assert.match(main, /readyLabel[\s\S]*?recovering_positions/);
  assert.match(main, /"botRecoveryReady" \.= botRecoveryReady/);
});

test("production research scopes can satisfy the portfolio evidence floor", async () => {
  const flyResearch = await fs.readFile(new URL("../fly.research.toml", import.meta.url), "utf8");
  const hetznerResearch = await fs.readFile(
    new URL("../deploy/hetzner/trader.research.env.managed", import.meta.url),
    "utf8",
  );
  const hetznerCompose = await fs.readFile(
    new URL("../deploy/hetzner/docker-compose.yml", import.meta.url),
    "utf8",
  );
  const flyTrading = await fs.readFile(new URL("../fly.toml", import.meta.url), "utf8");
  const backend = await fs.readFile(new URL("../haskell/app/Main.hs", import.meta.url), "utf8");

  assert.match(flyResearch, /TRADER_OPTIMIZER_LOOKBACK_WINDOWS = "1100d"/);
  assert.match(flyResearch, /TRADER_OPTIMIZER_INTERVALS = "6h,12h,1d"/);
  assert.match(flyResearch, /TRADER_OPTIMIZER_TRIALS = "4"/);
  assert.match(flyResearch, /TRADER_OPTIMIZER_DISCOVERY_RECOVERY_TRIALS = "6"/);
  assert.match(flyResearch, /TRADER_OPTIMIZER_DISCOVERY_MIN_EDGE_MIN = "0\.0018"/);
  assert.match(flyResearch, /TRADER_OPTIMIZER_DISCOVERY_MIN_EDGE_MAX = "0\.0024"/);
  assert.match(flyResearch, /TRADER_OPTIMIZER_METHOD_WEIGHT_01 = "3"/);
  assert.match(flyResearch, /TRADER_OPTIMIZER_METHOD_WEIGHT_TA_REGIME_SWITCH = "3"/);
  assert.match(hetznerResearch, /TRADER_OPTIMIZER_LOOKBACK_WINDOWS=1100d/);
  assert.match(hetznerResearch, /TRADER_OPTIMIZER_LOOKBACK_HEADROOM_POINTS=64/);
  assert.match(hetznerResearch, /TRADER_OPTIMIZER_INTERVALS=6h/);
  assert.match(hetznerResearch, /TRADER_OPTIMIZER_TRIALS=4/);
  assert.match(hetznerResearch, /TRADER_OPTIMIZER_MIN_ROUND_TRIPS=20/);
  assert.match(hetznerResearch, /TRADER_OPTIMIZER_P_COST_AWARE_EDGE=1\.0/);
  assert.match(hetznerResearch, /TRADER_OPTIMIZER_DISCOVERY_RECOVERY_TRIALS=6/);
  assert.match(hetznerResearch, /TRADER_OPTIMIZER_DISCOVERY_MIN_EDGE_MIN=0\.0018/);
  assert.match(hetznerResearch, /TRADER_OPTIMIZER_DISCOVERY_MIN_EDGE_MAX=0\.0024/);
  assert.match(hetznerResearch, /TRADER_OPTIMIZER_METHOD_WEIGHT_01=1/);
  assert.match(hetznerResearch, /TRADER_OPTIMIZER_METHOD_WEIGHT_TA_BEST=4/);
  assert.match(hetznerResearch, /TRADER_OPTIMIZER_METHOD_WEIGHT_TA_REGIME_SWITCH=5/);
  assert.match(hetznerResearch, /TRADER_TOP_COMBOS_BACKTEST_TOP_N=20/);
  assert.match(hetznerResearch, /TRADER_TOP_COMBOS_BACKTEST_EVERY_SEC=86400/);
  assert.match(hetznerResearch, /TRADER_TOP_COMBOS_BACKTEST_STALE_DAYS=7/);
  assert.match(hetznerCompose, /TRADER_OPTIMIZER_DISCOVERY_MIN_EDGE_MIN: \$\{TRADER_OPTIMIZER_DISCOVERY_MIN_EDGE_MIN:-\}/);
  assert.match(hetznerCompose, /TRADER_OPTIMIZER_DISCOVERY_MIN_EDGE_MAX: \$\{TRADER_OPTIMIZER_DISCOVERY_MIN_EDGE_MAX:-\}/);
  assert.match(hetznerCompose, /TRADER_OPTIMIZER_LOOKBACK_HEADROOM_POINTS: \$\{TRADER_OPTIMIZER_LOOKBACK_HEADROOM_POINTS:-64\}/);
  assert.match(backend, /lookupEnv "TRADER_OPTIMIZER_DISCOVERY_MIN_EDGE_MIN"/);
  assert.match(backend, /lookupEnv "TRADER_OPTIMIZER_DISCOVERY_MIN_EDGE_MAX"/);
  assert.match(backend, /lookupEnv "TRADER_OPTIMIZER_LOOKBACK_HEADROOM_POINTS"/);
  assert.match(backend, /optimizer\.auto\.no_admission/);
  assert.match(backend, /max venueMinEdgeFloor \(readNonNegativeDouble discoveryRecoveryMinEdgeMinEnv venueMinEdgeFloor\)/);
  assert.match(flyTrading, /TRADER_PORTFOLIO_SELECTOR_MIN_DAYS = "180"/);

  const lookbackDays = 1100;
  const backtestRatio = 0.2;
  const selectorFloorDays = 180;
  const sixHourBarsPerDay = 4;
  const maxPoints = 5000;
  assert.ok(lookbackDays * backtestRatio >= selectorFloorDays);
  assert.ok(lookbackDays * sixHourBarsPerDay <= maxPoints);
});

test("repo root package exposes all root automation and formal verifier tests", async () => {
  const pkgRaw = await fs.readFile(new URL("../package.json", import.meta.url), "utf8");
  const pkg = JSON.parse(pkgRaw);
  const testScript = pkg?.scripts?.["test:autoloop"];
  assert.equal(typeof testScript, "string");
  assert.equal(testScript, "node --test test/*.test.mjs");
});

test("repo root test command includes the autoloop verifier", async () => {
  const pkgRaw = await fs.readFile(new URL("../package.json", import.meta.url), "utf8");
  const pkg = JSON.parse(pkgRaw);
  const testScript = pkg?.scripts?.test;
  assert.equal(typeof testScript, "string");
  assert.match(testScript, /\bnpm run test:autoloop\b/);
});

test("top-combo optimizer wrapper supports no-optimization audit binaries", async () => {
  const script = await fs.readFile(new URL("../haskell/scripts/run_optimize_equity_top5.sh", import.meta.url), "utf8");
  assert.match(script, /NOOPT="\$\{NOOPT:-0\}"/);
  assert.match(script, /noopt=\$NOOPT/);
  assert.match(script, /if \[\[ "\$NOOPT" == "1" \]\]; then/);
  assert.match(script, /cabal build optimize-equity trader-hs --disable-optimization/);
  assert.match(script, /cabal list-bin optimize-equity --disable-optimization/);
  assert.match(script, /cabal list-bin trader-hs --disable-optimization/);
  assert.match(script, /--binary "\$trader_bin"/);
});

test("top-combo optimizer wrapper retries activity-only short audits with deeper bars", async () => {
  const script = await fs.readFile(new URL("../haskell/scripts/run_optimize_equity_top5.sh", import.meta.url), "utf8");
  assert.match(script, /ACTIVITY_RETRY_BARS="\$\{ACTIVITY_RETRY_BARS:-700,1000\}"/);
  assert.match(script, /DEFAULT_OPEN_THRESHOLD_MAX="2e-2"/);
  assert.match(script, /OPEN_THRESHOLD_MAX_WAS_SET="\$\{OPEN_THRESHOLD_MAX:\+1\}"/);
  assert.match(script, /ACTIVITY_RECOVERY="\$\{ACTIVITY_RECOVERY:-1\}"/);
  assert.match(script, /ACTIVITY_RECOVERY_OPEN_THRESHOLD_MAX="\$\{ACTIVITY_RECOVERY_OPEN_THRESHOLD_MAX:-6e-3\}"/);
  assert.match(script, /NEUTRAL_RECOVERY="\$\{NEUTRAL_RECOVERY:-1\}"/);
  assert.match(script, /NEUTRAL_RECOVERY_TIMEOUT_SEC="\$\{NEUTRAL_RECOVERY_TIMEOUT_SEC:-\}"/);
  assert.match(script, /NEUTRAL_RECOVERY_METHOD_WEIGHT_10="\$\{NEUTRAL_RECOVERY_METHOD_WEIGHT_10:-0\.0\}"/);
  assert.match(script, /NEUTRAL_RECOVERY_METHOD_WEIGHT_EDGE_PICK="\$\{NEUTRAL_RECOVERY_METHOD_WEIGHT_EDGE_PICK:-1\.0\}"/);
  assert.match(script, /activity_retry_bars=\$ACTIVITY_RETRY_BARS/);
  assert.match(script, /activity_recovery=\$ACTIVITY_RECOVERY/);
  assert.match(script, /neutral_recovery=\$NEUTRAL_RECOVERY/);
  assert.match(script, /neutral_recovery_timeout=\$NEUTRAL_RECOVERY_TIMEOUT_SEC/);
  assert.match(script, /quality_effective_threshold_max\(\)/);
  assert.match(script, /open_threshold_range_requested=\$OPEN_THRESHOLD_MIN:\$OPEN_THRESHOLD_MAX/);
  assert.match(script, /open_threshold_range_effective=\$OPEN_THRESHOLD_MIN:\$EFFECTIVE_OPEN_THRESHOLD_MAX/);
  assert.match(script, /should_retry_activity_bars\(\)/);
  assert.match(script, /should_retry_neutral_recovery\(\)/);
  assert.match(script, /No eligible trials\./);
  assert.match(script, /activityCount</);
  assert.match(script, /neutral_diagnostic_count = 0/);
  assert.match(script, /latest=([^ ]+)/);
  assert.match(script, /NEUTRAL/);
  assert.match(script, /timeout>/);
  assert.ok(script.includes('diagnostic_match = re.search(r"\\(diagnostic: ([^)]+)\\)", line)'));
  assert.match(script, /for retry_bars in \$\{ACTIVITY_RETRY_BARS\/\/,\/ \}; do/);
  assert.match(script, /\(\(retry_bars > bars_attempt\)\)/);
  assert.match(script, /--bars-min "\$bars_attempt"/);
  assert.match(script, /--bars-max "\$bars_attempt"/);
  assert.match(script, /activity_recovery_attempt=0/);
  assert.match(script, /open_threshold_max_explicit="1"/);
  assert.match(script, /cmd\+=\(--open-threshold-max "\$open_threshold_max"\)/);
  assert.match(script, /cmd\+=\(--close-threshold-max "\$close_threshold_max"\)/);
  assert.match(script, /--min-hold-bars-min "\$min_hold_bars_min"/);
  assert.match(script, /timeout_sec="\$NEUTRAL_RECOVERY_TIMEOUT_SEC"/);
  assert.match(script, /--timeout-sec "\$timeout_sec"/);
  assert.match(script, /--method-weight-edge-pick "\$method_weight_edge_pick"/);
  assert.match(script, /--method-weight-regime-switch "\$method_weight_regime_switch"/);
  assert.match(script, /Retrying \$label: \$sym \$interval with bars=\$next_bars after activityCount-only skips/);
  assert.match(script, /Retrying \$label: \$sym \$interval with activity-recovery thresholds after activityCount-only skips/);
  assert.match(script, /Retrying \$label: \$sym \$interval with neutral-recovery method weights after neutral activity diagnostics/);
  assert.ok(script.includes('run_match = re.search(r"\\(trials=\\d+ bars=(\\d+)(?: [^)]*)?\\)", line)'));
  assert.match(script, /"status": status/);
  assert.match(script, /"filterReasons": filter_reasons/);
  assert.match(script, /"failureReasons": failure_reasons/);
    assert.match(script, /"activityDiagnostics": activity_diagnostics/);
});

test("top-combo optimizer wrapper centralizes quality-mode threshold max logging", async () => {
  const script = await fs.readFile(new URL("../haskell/scripts/run_optimize_equity_top5.sh", import.meta.url), "utf8");
  const readAssignedValue = (name) => {
    const match = script.match(new RegExp(`^${name}="([^"]*)"$`, "m"));
    assert.ok(match, `expected ${name} assignment`);
    return match[1];
  };

  const qualityDefault = readAssignedValue("QUALITY_DEFAULT_THRESHOLD_MAX");
  const productionDefault = readAssignedValue("DEFAULT_OPEN_THRESHOLD_MAX");
  const loggedEffectiveThresholdMax = ({ requestedMax, explicit, quality = "1" }) =>
    quality === "1" && explicit !== "1" && requestedMax === productionDefault ? qualityDefault : requestedMax;

  assert.match(script, /QUALITY_DEFAULT_THRESHOLD_MAX="5e-2"/);
  assert.match(script, /local quality_default="\$4"/);
  assert.match(
    script,
    /EFFECTIVE_OPEN_THRESHOLD_MAX="\$\(quality_effective_threshold_max "\$OPEN_THRESHOLD_MAX" "\$OPEN_THRESHOLD_MAX_WAS_SET" "\$DEFAULT_OPEN_THRESHOLD_MAX" "\$QUALITY_DEFAULT_THRESHOLD_MAX"\)"/,
  );
  assert.match(
    script,
    /EFFECTIVE_CLOSE_THRESHOLD_MAX="\$\(quality_effective_threshold_max "\$CLOSE_THRESHOLD_MAX" "\$CLOSE_THRESHOLD_MAX_WAS_SET" "\$DEFAULT_CLOSE_THRESHOLD_MAX" "\$QUALITY_DEFAULT_THRESHOLD_MAX"\)"/,
  );
  assert.equal(loggedEffectiveThresholdMax({ requestedMax: productionDefault, explicit: "" }), qualityDefault);
  assert.equal(loggedEffectiveThresholdMax({ requestedMax: productionDefault, explicit: "1" }), productionDefault);
});

test("volatility scorecard fails Kelly-lite rows without material exposure reduction", async () => {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "scorecard-test-"));
  try {
    const baseline = await writeScorecardJson(dir, "baseline.json", {
      backtest: {
        metrics: {
          sharpe: 1.0,
          maxDrawdown: 0.1,
          avgTradeReturn: 0.01,
          tradeCount: 100,
          roundTrips: 100,
        },
      },
    });
    const missingReport = await writeScorecardJson(dir, "missing-report.json", {
      params: { kellyLiteSizing: true },
      backtest: {
        metrics: {
          sharpe: 1.2,
          maxDrawdown: 0.08,
          avgTradeReturn: 0.012,
          tradeCount: 90,
          roundTrips: 90,
        },
      },
    });
    const weakReduction = await writeScorecardJson(dir, "weak-reduction.json", {
      backtest: {
        metrics: {
          sharpe: 1.2,
          maxDrawdown: 0.08,
          avgTradeReturn: 0.012,
          tradeCount: 90,
          roundTrips: 90,
        },
        kellyLite: {
          enabled: true,
          realizedExposure: 0.485,
          uncappedExposure: 0.5,
          exposureRatio: 0.97,
          exposureReduction: 0.015,
        },
      },
    });
    const noUncappedExposure = await writeScorecardJson(dir, "no-uncapped-exposure.json", {
      backtest: {
        metrics: {
          sharpe: 1.2,
          maxDrawdown: 0.08,
          avgTradeReturn: 0.012,
          tradeCount: 90,
          roundTrips: 90,
        },
        kellyLite: {
          enabled: true,
          realizedExposure: 0,
          uncappedExposure: 0,
          exposureRatio: 0,
          exposureReduction: 0,
        },
      },
    });

    const result = runScorecard([
      "--row",
      `baseline=${baseline}`,
      "--row",
      `missing-kelly=${missingReport}`,
      "--row",
      `weak-kelly=${weakReduction}`,
      "--row",
      `no-uncapped=${noUncappedExposure}`,
      "--min-sharpe-delta",
      "0",
      "--max-dd-regression",
      "1",
      "--min-trade-retention",
      "0",
      "--min-closed-trades",
      "1",
      "--min-kelly-lite-exposure-reduction",
      "0.05",
      "--max-kelly-lite-exposure-ratio",
      "0.90",
    ]);

    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stdout, /missing-kelly[\s\S]*fail \(kellyLiteExposureMissing\)/);
    assert.match(result.stdout, /weak-kelly[\s\S]*kellyLiteExposureReduction<0\.050/);
    assert.match(result.stdout, /weak-kelly[\s\S]*kellyLiteExposureRatio>0\.900/);
    assert.match(result.stdout, /no-uncapped[\s\S]*kellyLiteUncappedExposure<=0/);
  } finally {
    await fs.rm(dir, { recursive: true, force: true });
  }
});

test("volatility scorecard renders passing Kelly-lite exposure reductions", async () => {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "scorecard-test-"));
  try {
    const baseline = await writeScorecardJson(dir, "baseline.json", {
      backtest: {
        metrics: {
          sharpe: 1.0,
          maxDrawdown: 0.1,
          avgTradeReturn: 0.01,
          tradeCount: 100,
          roundTrips: 100,
        },
      },
    });
    const candidate = await writeScorecardJson(dir, "candidate.json", {
      backtest: {
        metrics: {
          sharpe: 1.2,
          maxDrawdown: 0.08,
          avgTradeReturn: 0.012,
          tradeCount: 90,
          roundTrips: 90,
        },
        kellyLite: {
          enabled: true,
          realizedExposure: 0.4,
          uncappedExposure: 0.5,
          exposureRatio: 0.8,
          exposureReduction: 0.1,
        },
      },
    });

    const result = runScorecard([
      "--row",
      `baseline=${baseline}`,
      "--row",
      `candidate=${candidate}`,
      "--min-sharpe-delta",
      "0",
      "--max-dd-regression",
      "1",
      "--min-trade-retention",
      "0",
      "--min-closed-trades",
      "1",
      "--min-kelly-lite-exposure-reduction",
      "0.05",
      "--max-kelly-lite-exposure-ratio",
      "0.90",
    ]);

    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stdout, /candidate[\s\S]*80\.0% \(0\.4000\/0\.5000, -0\.1000\)[\s\S]*pass/);
  } finally {
    await fs.rm(dir, { recursive: true, force: true });
  }
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

test("autoloop calls the local stack refresh after a green CI push", async () => {
  const script = await fs.readFile(new URL("../scripts/autoloop.mjs", import.meta.url), "utf8");
  // The green-CI branch must invoke refreshLocalStack and surface the result in status.
  assert.match(script, /const localRefresh = refreshLocalStack\(\{ headSha: pushedHeadSha \}\);/);
  assert.match(script, /localRefresh,/);
  // The helper itself must be opt-outable and call the dedicated shell script.
  assert.match(script, /function refreshLocalStack\(\{ headSha \}\) \{/);
  assert.match(script, /AUTOLOOP_SKIP_LOCAL_REFRESH/);
  assert.match(script, /scripts\/restart-local-stack\.sh/);
});

test("restart-local-stack rewrites build-commit and kicks API + Web LaunchAgents", async () => {
  const script = await fs.readFile(new URL("../scripts/restart-local-stack.sh", import.meta.url), "utf8");
  assert.match(script, /TRADER_API_LAUNCHD_LABEL:-ai\.openclaw\.trader\.api/);
  assert.match(script, /TRADER_WEB_LAUNCHD_LABEL:-ai\.openclaw\.trader\.web/);
  // Best-effort: never bail out on missing tools / non-Darwin hosts.
  assert.match(script, /uname -s/);
  assert.match(script, /command -v launchctl/);
  // Refresh the build-commit marker so /health reports the new SHA.
  assert.match(script, /haskell\/\.build-commit/);
  assert.match(script, /git rev-parse HEAD/);
  // Kick both services.
  assert.match(script, /launchctl kickstart -k/);
});

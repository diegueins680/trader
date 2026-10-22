import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { parseArgs, runBenchmark } from "../scripts/benchmark-live-fleet.mjs";

test("fleet benchmark bounds concurrency and rejects credential-bearing URLs", () => {
  for (const url of ["https://secret@example.com", "https://example.com?token=secret", "file:///tmp/test", "https://example.com/api"]) {
    assert.throws(() => parseArgs(["--base-url", url]));
  }
  assert.throws(() => parseArgs(["--base-url", "https://example.com", "--clients", "17"]));
  assert.throws(() => parseArgs(["--base-url", "https://example.com", "--rounds", "0"]));
  assert.throws(() => parseArgs(["--base-url", "https://example.com", "--max-memory-bytes", "4096"]));
  for (const limit of ["0", "-1", "1.5", "Infinity", "9007199254740992"]) {
    assert.throws(() => parseArgs(["--base-url", "https://example.com", "--rss-pid", "1", "--max-memory-bytes", limit]));
  }
});

test("fleet benchmark measures concurrent cold requests and conditional cache hits without retaining secrets", async () => {
  const config = parseArgs(["--base-url", "https://example.com", "--rounds", "3", "--clients", "4", "--interval-ms", "0"]);
  let cold = 0;
  let warm = 0;
  let active = 0;
  let peakActive = 0;
  const result = await runBenchmark(config, {
    token: "test-secret", tenantKey: "private-tenant", memorySample: async () => 4096,
    fetchImpl: async (url, options) => {
      assert.equal(options.method, "GET");
      assert.equal(options.redirect, "error");
      assert.equal(options.headers.Authorization, "Bearer test-secret");
      active += 1;
      peakActive = Math.max(peakActive, active);
      await new Promise((resolve) => setImmediate(resolve));
      active -= 1;
      if (url.endsWith("/optimizer/combos")) {
        if (options.headers["If-None-Match"] === 'W/"stable"') {
          warm += 1;
          return new Response(null, { status: 304 });
        }
        cold += 1;
        return Response.json({ combos: [] }, { headers: { etag: 'W/"stable"' } });
      }
      if (url.endsWith("/bot/status")) return Response.json({ multi: true, bots: Array.from({ length: 18 }, () => ({ running: true })) });
      return Response.json({ ready: true });
    },
  });
  assert.equal(cold, 4);
  assert.equal(warm, 8);
  assert.equal(peakActive, 7);
  assert.equal(result.passed, true);
  assert.equal(result.memory.bytes.max, 4096);
  assert.equal(result.memory.assessment, "observed");
  assert.equal(result.memory.maxAllowedBytes, null);
  assert.equal(result.workers.last, 18);
  assert.equal(result.coverage.startupTriggered, false);
  assert.doesNotMatch(JSON.stringify(result), /test-secret|private-tenant/);
});

test("fleet benchmark reports outages, worker loss, and measurement failure without leaking exception text", async () => {
  const config = parseArgs(["--base-url", "https://example.com", "--rounds", "1", "--clients", "1"]);
  const result = await runBenchmark(config, {
    token: "", tenantKey: "", memorySample: async () => { throw new Error("private-memory-path"); },
    fetchImpl: async (url) => {
      if (url.endsWith("/optimizer/combos")) throw new Error("Authorization: secret");
      if (url.endsWith("/ready")) return new Response(null, { status: 503 });
      if (url.endsWith("/health")) return Response.json({ ready: true });
      return Response.json({ running: false });
    },
  });
  assert.equal(result.passed, false);
  assert.equal(result.failures, 2);
  assert.equal(result.workers.last, 0);
  assert.equal(result.memory.errors, 1);
  assert.equal(result.memory.assessment, "measurement-failed");
  assert.doesNotMatch(JSON.stringify(result), /secret|private-memory-path/);
});

test("fleet benchmark rejects HTML proxy fallbacks and false readiness despite HTTP 200", async () => {
  const config = parseArgs(["--base-url", "https://example.com", "--rounds", "1", "--clients", "1", "--expected-workers", "0"]);
  const result = await runBenchmark(config, {
    token: "", tenantKey: "", memorySample: async () => null,
    fetchImpl: async (url) => {
      if (url.endsWith("/optimizer/combos") || url.endsWith("/ready")) return new Response("<html>SPA fallback</html>");
      if (url.endsWith("/health")) return Response.json({ ready: false });
      return Response.json({ running: false });
    },
  });
  assert.equal(result.passed, false);
  assert.equal(result.failures, 3);
});

const emptyFleetResponse = async (url) => url.endsWith("/bot/status")
  ? Response.json({ multi: true, bots: [] }) : Response.json({ ready: true });

test("fleet benchmark distinguishes request success from unmeasured memory", async () => {
  const config = parseArgs(["--base-url", "https://example.com", "--rounds", "1", "--clients", "1", "--expected-workers", "0"]);
  const result = await runBenchmark(config, {
    token: "", tenantKey: "", fetchImpl: emptyFleetResponse, memorySample: async () => null,
  });
  assert.equal(result.passed, true);
  assert.equal(result.memory.assessment, "unmeasured");
  assert.equal(result.memory.bytes, null);
});

test("fleet memory ceiling accepts equality and rejects unavailable or invalid measurements", async () => {
  const config = parseArgs(["--base-url", "https://example.com", "--rounds", "1", "--clients", "1", "--expected-workers", "0", "--memory-file", "/not-read-by-fixture", "--max-memory-bytes", "4096"]);
  for (const sample of [4096, null, NaN, Infinity, -1, 0, 1.5]) {
    const result = await runBenchmark(config, {
      token: "", tenantKey: "", fetchImpl: emptyFleetResponse, memorySample: async () => sample,
    });
    assert.equal(result.passed, sample === 4096);
    assert.equal(result.memory.assessment, sample === 4096 ? "within-limit" : "measurement-failed");
    assert.equal(result.memory.maxAllowedBytes, 4096);
  }
});

test("fleet memory ceiling retains a transient peak even after memory returns below budget", async () => {
  const config = parseArgs(["--base-url", "https://example.com", "--rounds", "1", "--clients", "1", "--expected-workers", "0", "--rss-pid", "1", "--max-memory-bytes", "4096"]);
  let samples = 0;
  const result = await runBenchmark(config, {
    token: "", tenantKey: "", memorySample: async () => samples++ === 0 ? 4097 : 4096,
    fetchImpl: async (url) => {
      await new Promise((resolve) => setTimeout(resolve, 150));
      return emptyFleetResponse(url);
    },
  });
  assert.equal(result.failures, 0);
  assert.equal(result.memory.bytes.last, 4096);
  assert.equal(result.memory.bytes.max, 4097);
  assert.equal(result.memory.assessment, "limit-exceeded");
  assert.equal(result.passed, false);
});

test("fleet benchmark rejects unsolicited 304 responses and malformed empty-fleet evidence", async () => {
  const config = parseArgs(["--base-url", "https://example.com", "--rounds", "1", "--clients", "1", "--expected-workers", "0"]);
  const invalidResponses = [
    ["/health", () => new Response(null, { status: 304 })],
    ["/ready", () => new Response(null, { status: 304 })],
    ["/optimizer/combos", () => new Response(null, { status: 304 })],
    ["/bot/status", () => Response.json({})],
    ["/bot/status", () => Response.json({ multi: true })],
    ["/bot/status", () => Response.json({ multi: true, bots: [null] })],
    ["/bot/status", () => Response.json({ multi: true, bots: [{ running: "false" }] })],
    ["/bot/status", () => new Response('{"running":false}')],
  ];
  for (const [path, response] of invalidResponses) {
    const result = await runBenchmark(config, {
      token: "", tenantKey: "", memorySample: async () => null,
      fetchImpl: async (url) => url.endsWith(path) ? response() : emptyFleetResponse(url),
    });
    assert.equal(result.passed, false, path);
    assert.equal(result.failures, 1, path);
  }
});

test("research and trading retain the complete pinned fleet independently of eligibility overrides", async () => {
  const values = async (role, key) => {
    const content = await readFile(new URL(`../deploy/hetzner/trader.${role}.env.managed`, import.meta.url), "utf8");
    const line = content.split("\n").find((row) => row.startsWith(`${key}=`));
    assert.ok(line, `${role} must declare ${key}`);
    return new Set(line.slice(key.length + 1).split(","));
  };
  const pins = await values("trading", "TRADER_TOP_COMBO_DEPLOYABLE_OVERRIDE_UUIDS");
  const retained = await values("trading", "TRADER_TOP_COMBOS_RETAIN_UUIDS");
  for (const pin of pins) assert.ok(retained.has(pin), `missing retained live pin ${pin}`);
  assert.ok(retained.size <= 256);
  assert.deepEqual(await values("research", "TRADER_TOP_COMBOS_RETAIN_UUIDS"), retained);
  assert.equal((await values("research", "TRADER_TOP_COMBO_DEPLOYABLE_OVERRIDE_UUIDS")).size, 5);
  for (const file of ["fly.toml", "fly.research.toml"]) {
    const config = await readFile(new URL(`../${file}`, import.meta.url), "utf8");
    const match = config.match(/^\s*TRADER_TOP_COMBOS_RETAIN_UUIDS = "([^"]*)"$/m);
    assert.ok(match, `${file} must retain shared live records`);
    assert.deepEqual(new Set(match[1].split(",")), retained);
  }
});

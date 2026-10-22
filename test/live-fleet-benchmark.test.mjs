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

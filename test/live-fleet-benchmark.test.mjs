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
  assert.equal(result.workers.min, 18);
  assert.equal(result.workerCountMismatches, 0);
  assert.equal(result.missingWorkerSamples, 0);
  assert.equal(result.timing.deadlineMs, 30000);
  assert.equal(result.timing.deadlineOverruns, 0);
  assert.equal(result.timing.overrunMs, null);
  const leaderboard = result.endpoints["/optimizer/combos"];
  assert.equal(leaderboard.conditionalRequests, 8);
  assert.equal(leaderboard.conditionalMisses, 0);
  assert.equal(leaderboard.byHttpStatus[200].latencyMs.count, 4);
  assert.equal(leaderboard.byHttpStatus[304].latencyMs.count, 8);
  assert.equal(leaderboard.byHttpStatus[304].bytes, 0);
  assert.equal(result.coverage.startupTriggered, false);
  assert.deepEqual(result.endpoints["/health"].failureReasons, {});
  assert.deepEqual(result.endpoints["/health"].readinessStates, { ready: 3 });
  assert.equal(Object.hasOwn(leaderboard, "readinessStates"), false);
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
  assert.deepEqual(result.endpoints["/optimizer/combos"].failureReasons, { "request-failed": 1 });
  assert.equal(result.endpoints["/optimizer/combos"].byHttpStatus[0].timeToHeadersMs, null);
  assert.equal(result.endpoints["/optimizer/combos"].byHttpStatus[0].failedProbes, 1);
  assert.deepEqual(result.endpoints["/ready"].httpStatuses, { 503: 1 });
  assert.deepEqual(result.endpoints["/ready"].failureReasons, { "http-status": 1 });
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
  assert.deepEqual(result.endpoints["/health"].statuses, { 0: 1 });
  assert.deepEqual(result.endpoints["/health"].httpStatuses, { 200: 1 });
  assert.deepEqual(result.endpoints["/health"].failureReasons, { "readiness-false": 1 });
  assert.equal(result.endpoints["/health"].byHttpStatus[200].failedProbes, 1);
  assert.deepEqual(result.endpoints["/ready"].failureReasons, { "non-json": 1 });
});

test("fleet benchmark categorizes malformed JSON and missing readiness without copying response details", async () => {
  const config = parseArgs(["--base-url", "https://example.com", "--rounds", "1", "--clients", "1", "--expected-workers", "0"]);
  const result = await runBenchmark(config, {
    token: "", tenantKey: "", memorySample: async () => null,
    fetchImpl: async (url) => {
      if (url.endsWith("/health")) return new Response('{"secret":"private-response",', { headers: { "content-type": "application/json" } });
      if (url.endsWith("/ready")) return Response.json({ error: "private-response" });
      if (url.endsWith("/bot/status")) return Response.json({ running: "private-response" });
      return Response.json({ combos: [] });
    },
  });
  assert.equal(result.failures, 3);
  assert.deepEqual(result.endpoints["/health"].failureReasons, { "invalid-json": 1 });
  assert.deepEqual(result.endpoints["/ready"].failureReasons, { "readiness-missing": 1 });
  assert.deepEqual(result.endpoints["/bot/status"].failureReasons, { "worker-evidence-invalid": 1 });
  assert.doesNotMatch(JSON.stringify(result), /private-response|secret/);
});

const emptyFleetResponse = async (url) => url.endsWith("/bot/status")
  ? Response.json({ multi: true, bots: [] }) : Response.json({ ready: true });

test("fleet benchmark fails successful late responses and exposes probe scheduling stalls", async () => {
  const config = parseArgs(["--base-url", "https://example.com", "--rounds", "1", "--clients", "1", "--timeout-ms", "5", "--expected-workers", "0"]);
  let blocked = false;
  const result = await runBenchmark(config, {
    token: "private-token", tenantKey: "private-tenant", memorySample: async () => null,
    fetchImpl: async (url) => {
      // Model a transport completing before a stalled abort timer is serviced.
      // Give the event-loop monitor time to start before blocking this process.
      await new Promise((resolve) => setTimeout(resolve, 40));
      if (!blocked) {
        blocked = true;
        Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 80);
      }
      return emptyFleetResponse(url);
    },
  });
  assert.equal(result.failures, 0);
  assert.equal(result.passed, false);
  assert.equal(result.timing.deadlineOverruns, 4);
  assert.equal(result.timing.overrunMs.count, 4);
  assert.ok(result.timing.overrunMs.min >= 25);
  assert.ok(result.timing.eventLoopDelayMs.max >= 40);
  for (const endpoint of Object.values(result.endpoints)) {
    assert.deepEqual(endpoint.statuses, { 200: 1 });
    assert.deepEqual(endpoint.httpStatuses, { 200: 1 });
    assert.deepEqual(endpoint.failureReasons, {});
    assert.equal(endpoint.deadlineOverruns, 1);
    assert.equal(endpoint.byHttpStatus[200].deadlineOverruns, 1);
    assert.equal(endpoint.byHttpStatus[200].failedProbes, 0);
  }
  assert.doesNotMatch(JSON.stringify(result), /private-token|private-tenant/);
});

test("fleet deadline includes consuming a body after headers arrive on time", async () => {
  const config = parseArgs(["--base-url", "https://example.com", "--rounds", "1", "--clients", "1", "--timeout-ms", "10", "--expected-workers", "0"]);
  const result = await runBenchmark(config, {
    token: "", tenantKey: "", memorySample: async () => null,
    fetchImpl: async (url) => {
      if (!url.endsWith("/optimizer/combos")) return emptyFleetResponse(url);
      const body = new ReadableStream({
        async pull(controller) {
          await new Promise((resolve) => setTimeout(resolve, 50));
          controller.enqueue(new TextEncoder().encode('{"combos":[]}'));
          controller.close();
        },
      }, { highWaterMark: 0 });
      return new Response(body, { headers: { "content-type": "application/json" } });
    },
  });
  assert.equal(result.passed, false);
  assert.equal(result.failures, 0);
  assert.equal(result.endpoints["/optimizer/combos"].deadlineOverruns, 1);
  assert.equal(result.endpoints["/optimizer/combos"].byHttpStatus[200].failedProbes, 0);
  assert.ok(result.endpoints["/optimizer/combos"].latencyMs.first >= 40);
});

test("fleet deadline also rejects a late validated cache hit", async () => {
  const config = parseArgs(["--base-url", "https://example.com", "--rounds", "2", "--clients", "1", "--interval-ms", "0", "--timeout-ms", "10", "--expected-workers", "0"]);
  const result = await runBenchmark(config, {
    token: "", tenantKey: "", memorySample: async () => null,
    fetchImpl: async (url, options) => {
      if (!url.endsWith("/optimizer/combos")) return emptyFleetResponse(url);
      if (!options.headers["If-None-Match"]) return Response.json({ combos: [] }, { headers: { etag: 'W/"stable"' } });
      await new Promise((resolve) => setTimeout(resolve, 50));
      return new Response(null, { status: 304 });
    },
  });
  assert.equal(result.passed, false);
  assert.equal(result.failures, 0);
  const endpoint = result.endpoints["/optimizer/combos"];
  assert.equal(endpoint.byHttpStatus[304].deadlineOverruns, 1);
  assert.equal(endpoint.byHttpStatus[304].failedProbes, 0);
  assert.equal(endpoint.conditionalRequests, 1);
  assert.equal(endpoint.conditionalMisses, 0);
});

test("fleet benchmark separates full bodies, ETag misses, cache hits and time to headers", async () => {
  const config = parseArgs(["--base-url", "https://example.com", "--rounds", "3", "--clients", "1", "--interval-ms", "0", "--expected-workers", "0"]);
  const payload = new TextEncoder().encode('{"combos":[]}');
  let leaderboardRequests = 0;
  const result = await runBenchmark(config, {
    token: "", tenantKey: "", memorySample: async () => null,
    fetchImpl: async (url, options) => {
      if (!url.endsWith("/optimizer/combos")) return emptyFleetResponse(url);
      leaderboardRequests += 1;
      if (leaderboardRequests === 1) {
        assert.equal(options.headers["If-None-Match"], undefined);
        // Do not prefetch: defer the body after the consumer receives headers.
        const body = new ReadableStream({
          async pull(controller) {
            await new Promise((resolve) => setTimeout(resolve, 50));
            controller.enqueue(payload);
            controller.close();
          },
        }, { highWaterMark: 0 });
        return new Response(body, { headers: { "content-type": "application/json", etag: 'W/"v1"' } });
      }
      if (leaderboardRequests === 2) {
        assert.equal(options.headers["If-None-Match"], 'W/"v1"');
        return Response.json({ combos: [] }, { headers: { etag: 'W/"v2"' } });
      }
      assert.equal(options.headers["If-None-Match"], 'W/"v2"');
      return new Response(null, { status: 304 });
    },
  });
  const endpoint = result.endpoints["/optimizer/combos"];
  assert.equal(result.passed, true);
  assert.equal(endpoint.conditionalRequests, 2);
  assert.equal(endpoint.conditionalMisses, 1);
  assert.equal(endpoint.byHttpStatus[200].latencyMs.count, 2);
  assert.equal(endpoint.byHttpStatus[304].latencyMs.count, 1);
  assert.equal(endpoint.byHttpStatus[200].bytes, 2 * payload.length);
  assert.equal(endpoint.byHttpStatus[304].bytes, 0);
  assert.equal(endpoint.timeToHeadersMs.count, 3);
  assert.ok(endpoint.byHttpStatus[200].latencyMs.first - endpoint.byHttpStatus[200].timeToHeadersMs.first >= 25);
});

test("fleet benchmark exposes a temporary worker loss hidden by identical first and last counts", async () => {
  const config = parseArgs(["--base-url", "https://example.com", "--rounds", "3", "--clients", "1", "--interval-ms", "0"]);
  const counts = [18, 17, 18];
  const result = await runBenchmark(config, {
    token: "", tenantKey: "", memorySample: async () => null,
    fetchImpl: async (url) => url.endsWith("/bot/status")
      ? Response.json({ multi: true, bots: Array.from({ length: counts.shift() }, () => ({ running: true })) })
      : emptyFleetResponse(url),
  });
  assert.equal(result.passed, false);
  assert.equal(result.failures, 0);
  assert.equal(result.workers.first, 18);
  assert.equal(result.workers.last, 18);
  assert.equal(result.workers.max, 18);
  assert.equal(result.workers.min, 17);
  assert.equal(result.workerCountMismatches, 1);
  assert.equal(result.missingWorkerSamples, 0);
});

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
    if (path === "/bot/status") {
      assert.equal(result.workers, null);
      assert.equal(result.missingWorkerSamples, 1);
      assert.equal(result.workerCountMismatches, 0);
    }
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


test("fleet benchmark reports readiness signals while preserving HTTP and false-readiness failures", async () => {
  const config = parseArgs(["--base-url", "https://example.com", "--rounds", "1", "--clients", "1", "--expected-workers", "0"]);
  const cases = [
    [{ ready: true }, "ready"],
    [{ ready: false, draining: false, botRecoveryReady: false }, "recovery-not-ready"],
    [{ ready: false, draining: true, botRecoveryReady: false }, "draining"],
    [{ ready: false }, "not-ready"],
    [{ ready: false, draining: "false", botRecoveryReady: false }, "not-ready"],
    [{ ready: false, draining: false, botRecoveryReady: true }, "not-ready"],
    [{ ready: "false", draining: false, botRecoveryReady: false }, "unavailable"],
    [{ draining: true }, "unavailable"],
    [null, "unavailable"],
  ];
  for (const [body, state] of cases) {
    const payload = body === null ? null : { ...body, status: "private-readiness-reason", error: "private-response" };
    const result = await runBenchmark(config, {
      token: "private-token", tenantKey: "private-tenant", memorySample: async () => null,
      fetchImpl: async (url) => {
        if (url.endsWith("/health")) return Response.json(payload);
        if (url.endsWith("/ready")) return Response.json(payload, { status: 503 });
        return emptyFleetResponse(url);
      },
    });
    assert.equal(result.passed, false);
    assert.equal(result.failures, body?.ready === true ? 1 : 2);
    const health = result.endpoints["/health"], ready = result.endpoints["/ready"];
    assert.deepEqual(health.readinessStates, { [state]: 1 });
    assert.deepEqual(ready.readinessStates, { [state]: 1 });
    assert.deepEqual(health.httpStatuses, { 200: 1 });
    assert.deepEqual(ready.statuses, { 503: 1 });
    assert.deepEqual(ready.httpStatuses, { 503: 1 });
    assert.deepEqual(ready.failureReasons, { "http-status": 1 });
    assert.equal(ready.byHttpStatus[503].failedProbes, 1);
    assert.doesNotMatch(JSON.stringify(result), /private-token|private-tenant|private-readiness-reason|private-response/);
  }
});

test("fleet benchmark bounds readiness error diagnostics and consumes malformed or non-JSON errors", async () => {
  const config = parseArgs(["--base-url", "https://example.com", "--rounds", "1", "--clients", "1", "--expected-workers", "0"]);
  const valid = JSON.stringify({ ready: false, draining: false, botRecoveryReady: false });
  const limit = 64 * 1024;
  const bodies = [
    [valid + " ".repeat(limit - valid.length), "application/json", "recovery-not-ready"],
    [valid + " ".repeat(limit + 1 - valid.length), "application/json", "unavailable"],
    ['{"ready":false,"private-response":', "application/json", "unavailable"],
    [valid, "text/html", "unavailable"],
  ];
  for (const [text, contentType, state] of bodies) {
    let consumed = false;
    const encoded = new TextEncoder().encode(text);
    const result = await runBenchmark(config, {
      token: "", tenantKey: "", memorySample: async () => null,
      fetchImpl: async (url) => {
        if (!url.endsWith("/ready")) return emptyFleetResponse(url);
        const body = new ReadableStream({
          start(controller) {
            controller.enqueue(encoded.subarray(0, encoded.length - 1));
          },
          pull(controller) {
            controller.enqueue(encoded.subarray(encoded.length - 1));
            consumed = true;
            controller.close();
          },
        }, { highWaterMark: 0 });
        return new Response(body, { status: 503, headers: { "content-type": contentType } });
      },
    });
    const ready = result.endpoints["/ready"];
    assert.equal(consumed, true);
    assert.equal(result.passed, false);
    assert.equal(result.failures, 1);
    assert.deepEqual(ready.readinessStates, { [state]: 1 });
    assert.deepEqual(ready.httpStatuses, { 503: 1 });
    assert.deepEqual(ready.failureReasons, { "http-status": 1 });
    assert.equal(ready.bytes, encoded.length);
    assert.doesNotMatch(JSON.stringify(result), /private-response/);
  }
});

test("fleet readiness diagnostics preserve a truncated error body's transport failure", async () => {
  const config = parseArgs(["--base-url", "https://example.com", "--rounds", "1", "--clients", "1", "--expected-workers", "0"]);
  const result = await runBenchmark(config, {
    token: "", tenantKey: "", memorySample: async () => null,
    fetchImpl: async (url) => {
      if (!url.endsWith("/ready")) return emptyFleetResponse(url);
      const body = new ReadableStream({
        start(controller) { controller.enqueue(new TextEncoder().encode('{"ready":false,')); },
        pull(controller) { controller.error(new Error("private-stream-error")); },
      }, { highWaterMark: 0 });
      return new Response(body, { status: 503, headers: { "content-type": "application/json" } });
    },
  });
  assert.equal(result.passed, false);
  assert.equal(result.failures, 1);
  assert.deepEqual(result.endpoints["/ready"].statuses, { 0: 1 });
  assert.deepEqual(result.endpoints["/ready"].httpStatuses, { 503: 1 });
  assert.deepEqual(result.endpoints["/ready"].failureReasons, { "body-read-failed": 1 });
  assert.deepEqual(result.endpoints["/ready"].readinessStates, { unavailable: 1 });
  assert.doesNotMatch(JSON.stringify(result), /private-stream-error/);
});

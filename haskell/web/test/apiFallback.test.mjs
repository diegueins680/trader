import assert from "node:assert/strict";
import { test } from "node:test";

const apiBundleUrl = new URL("../.tmp/web-tests/api.js", import.meta.url);

function jsonResponse(status, body) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

async function loadApiModule(config, fetchImpl) {
  globalThis.window = {
    location: {
      origin: "https://ui.example.com",
      hostname: "ui.example.com",
    },
    setTimeout,
    clearTimeout,
    __TRADER_CONFIG__: config,
  };
  globalThis.fetch = fetchImpl;
  const modUrl = new URL(apiBundleUrl);
  modUrl.searchParams.set("cachebust", `${Date.now()}-${Math.random()}`);
  return import(modUrl.href);
}

function restoreGlobal(name, priorValue) {
  if (priorValue === undefined) {
    delete globalThis[name];
    return;
  }
  globalThis[name] = priorValue;
}

async function withApiModule(config, fetchImpl, run) {
  const priorWindow = globalThis.window;
  const priorFetch = globalThis.fetch;
  try {
    const api = await loadApiModule(config, fetchImpl);
    return await run(api);
  } finally {
    restoreGlobal("window", priorWindow);
    restoreGlobal("fetch", priorFetch);
  }
}

test("api fallback does not use 401 for explicit direct hosts", async () => {
  const calls = [];
  await withApiModule(
    {
      apiBaseUrl: "https://api.example.com",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "/api",
      apiToken: "",
    },
    async (url) => {
      calls.push(String(url));
      return jsonResponse(401, { error: "Unauthorized" });
    },
    async (api) => {
      await assert.rejects(
        () => api.health("https://api.example.com", { timeoutMs: 5_000 }),
        (err) => err?.name === "HttpError" && err.status === 401,
      );
    },
  );
  assert.deepEqual(calls, ["https://api.example.com/health"]);
});

test("api fallback uses 401 for inferred direct hosts with /api fallback", async () => {
  const calls = [];
  await withApiModule(
    {
      apiBaseUrl: "https://api.example.com",
      apiBaseUrlInferred: true,
      apiFallbackUrl: "/api",
      apiToken: "",
    },
    async (url) => {
      const href = String(url);
      calls.push(href);
      if (href === "https://api.example.com/health") {
        return jsonResponse(401, { error: "Unauthorized" });
      }
      return jsonResponse(200, { status: "ok" });
    },
    async (api) => {
      const out = await api.health("https://api.example.com", { timeoutMs: 5_000 });
      assert.equal(out.status, "ok");
    },
  );
  assert.deepEqual(calls, ["https://api.example.com/health", "/api/health"]);
});

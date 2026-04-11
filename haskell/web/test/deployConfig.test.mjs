import test from "node:test";
import assert from "node:assert/strict";

const deployConfigModuleUrl = new URL("../.tmp/web-tests/deployConfig.js", import.meta.url);
let deployConfigImportSeq = 0;

function compactTimeouts(timeouts) {
  return Object.fromEntries(Object.entries(timeouts ?? {}).filter(([, value]) => value !== undefined));
}

async function loadDeployConfig(rawConfig) {
  const prevWindow = globalThis.window;
  try {
    globalThis.window = { __TRADER_CONFIG__: rawConfig };
    const specifier = new URL(deployConfigModuleUrl);
    specifier.searchParams.set("case", String(deployConfigImportSeq++));
    const mod = await import(specifier.href);
    return mod.TRADER_UI_CONFIG;
  } finally {
    if (prevWindow === undefined) delete globalThis.window;
    else globalThis.window = prevWindow;
  }
}

test("deploy-config defaults missing global config to inferred /api", async () => {
  const config = await loadDeployConfig(undefined);

  assert.equal(config.apiBaseUrl, "/api");
  assert.equal(config.apiBaseUrlInferred, true);
  assert.equal(config.apiFallbackUrl, undefined);
  assert.equal(config.apiToken, "");
});

test("deploy-config infers /api primary when apiBaseUrl is blank and preserves distinct fallback", async () => {
  const config = await loadDeployConfig({
    apiBaseUrl: "  ",
    apiBaseUrlInferred: false,
    apiFallbackUrl: "https://api.example.com",
    apiToken: "token",
  });

  assert.equal(config.apiBaseUrl, "/api");
  assert.equal(config.apiBaseUrlInferred, true);
  assert.equal(config.apiFallbackUrl, "https://api.example.com");
  assert.equal(config.apiToken, "token");
});

test("deploy-config keeps exact safe integer millisecond timeouts and clamps only after validation", async () => {
  const config = await loadDeployConfig({
    apiBaseUrl: "/api",
    apiToken: "",
    timeoutsMs: {
      requestMs: "1000",
      signalMs: "1000.0",
      backtestMs: "1e3",
      tradeMs: 86_400_001,
    },
  });

  assert.deepEqual(compactTimeouts(config.timeoutsMs), {
    requestMs: 1000,
    signalMs: 1000,
    backtestMs: 1000,
    tradeMs: 86_400_000,
  });
});

test("deploy-config rejects fractional and unsafe integer-like timeout values instead of rounding them", async () => {
  const config = await loadDeployConfig({
    apiBaseUrl: "/api",
    apiToken: "",
    timeoutsMs: {
      requestMs: 2_000,
      signalMs: "1000.4",
      backtestMs: "9007199254740993",
      tradeMs: Number.MAX_SAFE_INTEGER + 1,
    },
  });

  assert.deepEqual(compactTimeouts(config.timeoutsMs), {
    requestMs: 2_000,
  });
});
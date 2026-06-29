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

test("deploy-config derives backend Fly fallback from valid backend env input when explicit fallback is blank and host is blank or missing", async () => {
  const missingHostConfig = await loadDeployConfig({
    apiBaseUrl: " ",
    apiFallbackUrl: " ",
    BACKEND_FLY_APP: " Trader-Api ",
    FRONTEND_FLY_APP: " ",
    FLY_DOMAIN: undefined,
    apiToken: "token",
  });

  assert.equal(missingHostConfig.apiBaseUrl, "/api");
  assert.equal(missingHostConfig.apiBaseUrlInferred, true);
  assert.equal(missingHostConfig.apiFallbackUrl, "https://trader-api.fly.dev/api");
  assert.equal(missingHostConfig.apiToken, "token");

  const blankHostConfig = await loadDeployConfig({
    apiBaseUrl: " ",
    apiFallbackUrl: " ",
    BACKEND_FLY_APP: " Trader-Api ",
    FLY_DOMAIN: " ",
    apiToken: "token",
  });

  assert.equal(blankHostConfig.apiBaseUrl, "/api");
  assert.equal(blankHostConfig.apiBaseUrlInferred, true);
  assert.equal(blankHostConfig.apiFallbackUrl, "https://trader-api.fly.dev/api");
  assert.equal(blankHostConfig.apiToken, "token");
});

test("deploy-config ignores blank Fly inputs and rejects malformed Fly overrides instead of synthesizing a fallback", async () => {
  const blankConfig = await loadDeployConfig({
    apiBaseUrl: " ",
    apiFallbackUrl: "",
    BACKEND_FLY_APP: " ",
    FLY_DOMAIN: " ",
    apiToken: "",
  });
  assert.equal(blankConfig.apiBaseUrl, "/api");
  assert.equal(blankConfig.apiFallbackUrl, undefined);

  const malformedHostConfig = await loadDeployConfig({
    apiBaseUrl: " ",
    apiFallbackUrl: "",
    BACKEND_FLY_APP: "trader-api",
    FLY_DOMAIN: "https://fly.dev/",
    apiToken: "",
  });
  assert.equal(malformedHostConfig.apiBaseUrl, "/api");
  assert.equal(malformedHostConfig.apiFallbackUrl, undefined);

  const malformedHostLabelConfig = await loadDeployConfig({
    apiBaseUrl: " ",
    apiFallbackUrl: "",
    BACKEND_FLY_APP: "trader-api",
    FLY_DOMAIN: ".fly..dev",
    apiToken: "",
  });
  assert.equal(malformedHostLabelConfig.apiBaseUrl, "/api");
  assert.equal(malformedHostLabelConfig.apiFallbackUrl, undefined);

  const malformedNonStringHostConfig = await loadDeployConfig({
    apiBaseUrl: " ",
    apiFallbackUrl: "",
    BACKEND_FLY_APP: "trader-api",
    FLY_DOMAIN: 42,
    apiToken: "",
  });
  assert.equal(malformedNonStringHostConfig.apiBaseUrl, "/api");
  assert.equal(malformedNonStringHostConfig.apiFallbackUrl, undefined);

  const malformedAppConfig = await loadDeployConfig({
    apiBaseUrl: " ",
    apiFallbackUrl: "",
    BACKEND_FLY_APP: "trader/api",
    FLY_DOMAIN: "fly.dev",
    apiToken: "",
  });
  assert.equal(malformedAppConfig.apiBaseUrl, "/api");
  assert.equal(malformedAppConfig.apiFallbackUrl, undefined);

  const malformedNonStringAppConfig = await loadDeployConfig({
    apiBaseUrl: " ",
    apiFallbackUrl: "",
    BACKEND_FLY_APP: { name: "trader-api" },
    FLY_DOMAIN: "fly.dev",
    apiToken: "",
  });
  assert.equal(malformedNonStringAppConfig.apiBaseUrl, "/api");
  assert.equal(malformedNonStringAppConfig.apiFallbackUrl, undefined);
});

test("deploy-config lowercases valid Fly host and app inputs before deriving a fallback", async () => {
  const config = await loadDeployConfig({
    apiBaseUrl: " ",
    apiFallbackUrl: "",
    BACKEND_FLY_APP: "Trader-Api",
    FLY_DOMAIN: "Internal-Fly.Dev/",
    apiToken: "",
  });

  assert.equal(config.apiBaseUrl, "/api");
  assert.equal(config.apiBaseUrlInferred, true);
  assert.equal(config.apiFallbackUrl, "https://trader-api.internal-fly.dev/api");
});

test("deploy-config deduplicates derived backend Fly fallback against an explicit absolute base URL", async () => {
  const config = await loadDeployConfig({
    apiBaseUrl: "HTTPS://Trader-Api.FLY.dev/api/",
    BACKEND_FLY_APP: "trader-api",
    apiToken: "",
  });

  assert.equal(config.apiBaseUrl, "HTTPS://Trader-Api.FLY.dev/api/");
  assert.equal(config.apiFallbackUrl, undefined);
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
      binanceTradesMs: 180_000,
    },
  });

  assert.deepEqual(compactTimeouts(config.timeoutsMs), {
    requestMs: 1000,
    signalMs: 1000,
    backtestMs: 1000,
    tradeMs: 86_400_000,
    binanceTradesMs: 180_000,
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
      binanceTradesMs: "999",
    },
  });

  assert.deepEqual(compactTimeouts(config.timeoutsMs), {
    requestMs: 2_000,
  });
});

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";
import { transformSync } from "esbuild";

const cabalConfig = readFileSync(new URL("../../../.cabal/config", import.meta.url), "utf8");
const appSource = readFileSync(new URL("../src/App.tsx", import.meta.url), "utf8");
const frontendConstantsSource = readFileSync(new URL("../src/app/constants.ts", import.meta.url), "utf8");
const frontendUtilsSource = readFileSync(new URL("../src/app/utils.ts", import.meta.url), "utf8");
const configLayoutSource = readFileSync(new URL("../src/app/configLayout.ts", import.meta.url), "utf8");
const backendBinanceIntervalsSource = readFileSync(new URL("../../app/Trader/BinanceIntervals.hs", import.meta.url), "utf8");

function parseStringLiterals(source) {
  return Array.from(source.matchAll(/"([^"]+)"/g), ([, value]) => value);
}

function parseTsConstStringArray(source, constName) {
  const match = source.match(new RegExp(`export const ${constName} = \\[(.*?)\\] as const;`, "s"));
  assert.ok(match, `expected ${constName} const array in frontend constants`);
  return parseStringLiterals(match[1]);
}

function parseTsConstObjectBody(source, constName) {
  const match = source.match(new RegExp(`export const ${constName}[^=]*= \\{(.*?)\\};`, "s"));
  assert.ok(match, `expected ${constName} const object in TypeScript source`);
  return match[1];
}

function parseTsConstObjectKeys(source, constName) {
  return Array.from(parseTsConstObjectBody(source, constName).matchAll(/"([^"]+)":/g), ([, value]) => value);
}

function parseHsStringList(source, bindingName) {
  const match = source.match(
    new RegExp(`${bindingName}\\s*=\\s*\\n(.*?)(?=\\n[a-zA-Z_][\\w']*\\s*::|\\n[a-zA-Z_][\\w']*\\s*=|$)`, "s"),
  );
  assert.ok(match, `expected ${bindingName} binding in backend BinanceIntervals module`);
  return parseStringLiterals(match[1]);
}

function parseTargetIdLiterals(source) {
  return Array.from(
    source.matchAll(/\b(?:targetId|[A-Za-z]+TargetId)\b\s*(?:\??:|=)\s*([^\n;]+)/g),
    ([, expression]) => parseStringLiterals(expression),
  ).flat();
}

function parseUiTargetIdLiterals(...sources) {
  return Array.from(new Set(sources.flatMap((source) => parseTargetIdLiterals(source))));
}

function loadConfigLayoutModule() {
  const { code } = transformSync(configLayoutSource, {
    loader: "ts",
    format: "cjs",
    platform: "node",
    target: "node20",
  });
  const module = { exports: {} };
  const context = { module, exports: module.exports };
  vm.runInNewContext(code, context, { filename: "configLayout.js" });
  return module.exports;
}

function assertUnique(values, label) {
  assert.equal(new Set(values).size, values.length, `${label} must stay duplicate-free`);
}

function assertContainsExactly(values, expected, label) {
  assertUnique(values, label);
  assert.deepEqual(
    [...values].sort(),
    [...expected].sort(),
    `${label} must contain each expected value exactly once`,
  );
}

const emittedUiTargetPageMap = {
  "section-api": "section-api",
  symbol: "section-market",
  platform: "section-market",
  market: "section-market",
  interval: "section-market",
  bars: "section-market",
  lookbackWindow: "section-lookback",
  lookbackBars: "section-lookback",
  positioning: "section-thresholds",
  backtestRatio: "section-thresholds",
  tuneRatio: "section-thresholds",
  epochs: "section-risk",
  hiddenSize: "section-risk",
  botSymbols: "section-livebot",
  orderQuote: "section-trade",
  orderQuoteFraction: "section-trade",
  platformKeys: "section-api",
  "section-trade": "section-trade",
};

const configLayoutModule = loadConfigLayoutModule();
const configPageIds = Array.from(configLayoutModule.CONFIG_PAGE_IDS);
const configPanelIds = Array.from(configLayoutModule.CONFIG_PANEL_IDS);
const configPanelDefaultPage = { ...configLayoutModule.CONFIG_PANEL_DEFAULT_PAGE };
const configTargetPageMap = { ...configLayoutModule.CONFIG_TARGET_PAGE_MAP };

test("tracked cabal config leaves machine-specific path overrides disabled", () => {
  const activeLines = cabalConfig
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith("--"));

  for (const key of ["remote-repo-cache", "build-summary", "installdir"]) {
    assert.equal(activeLines.some((line) => line.startsWith(`${key}:`)), false);
  }
});

test("repo contract keeps frontend Binance intervals aligned with backend validation", () => {
  const frontendBinanceIntervals = parseTsConstStringArray(frontendConstantsSource, "BINANCE_INTERVALS");
  const frontendBinanceIntervalSecondsKeys = parseTsConstObjectKeys(frontendConstantsSource, "BINANCE_INTERVAL_SECONDS");
  const backendBinanceIntervals = parseHsStringList(backendBinanceIntervalsSource, "binanceIntervals");

  assertUnique(frontendBinanceIntervals, "frontend BINANCE_INTERVALS");
  assertUnique(backendBinanceIntervals, "backend binanceIntervals");
  assert.deepEqual(frontendBinanceIntervals, backendBinanceIntervals);
  assert.deepEqual(frontendBinanceIntervalSecondsKeys, frontendBinanceIntervals);
});

test("repo contract routes lookback and live-bot validation targets to their config pages", () => {
  const targetPageMapBody = parseTsConstObjectBody(configLayoutSource, "CONFIG_TARGET_PAGE_MAP");

  assert.match(
    targetPageMapBody,
    /platformKeys:\s*"section-api"/,
    "platformKeys validation issues must route to the API config page",
  );
  assert.match(
    targetPageMapBody,
    /lookbackWindow:\s*"section-lookback"/,
    "lookbackWindow validation issues must route to the Lookback config page",
  );
  assert.match(
    targetPageMapBody,
    /lookbackBars:\s*"section-lookback"/,
    "lookbackBars validation issues must route to the Lookback config page",
  );
  assert.match(
    targetPageMapBody,
    /botSymbols:\s*"section-livebot"/,
    "botSymbols validation issues must route to the Live bot config page",
  );
});

test("repo contract routes trade sizing validation targets to trade config page", () => {
  const targetPageMapBody = parseTsConstObjectBody(configLayoutSource, "CONFIG_TARGET_PAGE_MAP");

  assert.match(
    targetPageMapBody,
    /orderQuote:\s*"section-trade"/,
    "orderQuote sizing issues must route to the Trade config page",
  );
  assert.match(
    targetPageMapBody,
    /orderQuoteFraction:\s*"section-trade"/,
    "orderQuoteFraction sizing issues must route to the Trade config page",
  );
});

test("repo contract routes every emitted UI target id to the intended config page", () => {
  const emittedUiTargetIds = parseUiTargetIdLiterals(appSource, frontendUtilsSource);

  assertContainsExactly(emittedUiTargetIds, Object.keys(emittedUiTargetPageMap), "emitted UI target ids");

  for (const [targetId, expectedPageId] of Object.entries(emittedUiTargetPageMap)) {
    assert.ok(configPageIds.includes(expectedPageId), `${expectedPageId} must remain a valid config page`);
    if (configPageIds.includes(targetId)) {
      assert.equal(targetId, expectedPageId, `${targetId} must stay a direct config page id`);
    } else {
      assert.equal(
        configTargetPageMap[targetId],
        expectedPageId,
        `${targetId} must stay routed by CONFIG_TARGET_PAGE_MAP`,
      );
    }
    assert.equal(configLayoutModule.resolveConfigPageForTarget(targetId), expectedPageId, `${targetId} must resolve to ${expectedPageId}`);
  }
});

test("repo contract keeps config page normalization total and deterministic", () => {
  assertUnique(configPageIds, "config page ids");
  assertUnique(configPanelIds, "config panel ids");
  assertContainsExactly(Object.keys(configPanelDefaultPage), configPanelIds, "config panel default-page keys");

  for (const pageId of Object.values(configPanelDefaultPage)) {
    assert.ok(configPageIds.includes(pageId), `${pageId} must remain a valid config page`);
  }
  for (const pageId of configPageIds) {
    assert.equal(configLayoutModule.normalizeConfigPage(pageId), pageId);
  }
  for (const [panelId, pageId] of Object.entries(configPanelDefaultPage)) {
    assert.equal(configLayoutModule.normalizeConfigPage(panelId), pageId);
  }
  for (const value of [undefined, null, "", "section-missing", "config-missing", 7, {}, []]) {
    assert.equal(configLayoutModule.normalizeConfigPage(value), "section-api");
  }
});

test("repo contract normalizes config panel restore order without duplicates or dropped panels", () => {
  const normalized = Array.from(
    configLayoutModule.normalizeConfigPanelOrder([
      configPanelIds[2],
      "invalid-panel",
      configPanelIds[0],
      configPanelIds[2],
      configPanelIds[4],
    ]),
  );

  assert.deepEqual(normalized, [
    configPanelIds[2],
    configPanelIds[0],
    configPanelIds[4],
    configPanelIds[1],
    configPanelIds[3],
  ]);
  assertContainsExactly(normalized, configPanelIds, "normalized config panel order");
  assert.deepEqual(Array.from(configLayoutModule.normalizeConfigPanelOrder(null)), configPanelIds);
});

test("repo contract resolves config page ids and validation target ids only", () => {
  for (const pageId of Object.values(configTargetPageMap)) {
    assert.ok(configPageIds.includes(pageId), `${pageId} must remain a valid config page`);
  }
  for (const pageId of configPageIds) {
    assert.equal(configLayoutModule.resolveConfigPageForTarget(pageId), pageId);
  }
  for (const [targetId, pageId] of Object.entries(configTargetPageMap)) {
    assert.equal(configLayoutModule.resolveConfigPageForTarget(targetId), pageId);
  }
  for (const targetId of ["", "section-missing", "config-market", "unknown-field"]) {
    assert.equal(configLayoutModule.resolveConfigPageForTarget(targetId), null);
  }
});
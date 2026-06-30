import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";
import { transformSync } from "esbuild";

const webPackageJson = readJson(new URL("../package.json", import.meta.url));
const webPackageLock = readJson(new URL("../package-lock.json", import.meta.url));
const rootPackageLock = readJson(new URL("../../../package-lock.json", import.meta.url));
const cabalConfig = readFileSync(new URL("../../../.cabal/config", import.meta.url), "utf8");
const appSource = readFileSync(new URL("../src/App.tsx", import.meta.url), "utf8");
const contractsSource = readFileSync(new URL("../src/app/contracts.ts", import.meta.url), "utf8");
const frontendConstantsSource = readFileSync(new URL("../src/app/constants.ts", import.meta.url), "utf8");
const frontendUtilsSource = readFileSync(new URL("../src/app/utils.ts", import.meta.url), "utf8");
const configLayoutSource = readFileSync(new URL("../src/app/configLayout.ts", import.meta.url), "utf8");
const topCombosChartSource = readFileSync(new URL("../src/components/TopCombosChart.tsx", import.meta.url), "utf8");
const backendBinanceIntervalsSource = readFileSync(new URL("../../app/Trader/BinanceIntervals.hs", import.meta.url), "utf8");

function readJson(url) {
  return JSON.parse(readFileSync(url, "utf8"));
}

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

function assertExactRuntimeVersionSpec(value, label) {
  assert.equal(typeof value, "string", `${label} must be a string`);
  assert.match(value, /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/, `${label} must be an exact version, not a range`);
}

function assertReactRuntimePackageContract(packageMeta, label) {
  const deps = packageMeta.dependencies ?? {};
  assertExactRuntimeVersionSpec(deps.react, `${label} react dependency`);
  assertExactRuntimeVersionSpec(deps["react-dom"], `${label} react-dom dependency`);
  assert.equal(deps.react, deps["react-dom"], `${label} must pin react and react-dom to the same runtime version`);
}

function assertReactRuntimeLockContract(lockMeta, packageEntryPath, label) {
  const packageEntry = lockMeta.packages?.[packageEntryPath];
  assert.ok(packageEntry, `${label} must include the package entry`);
  assertReactRuntimePackageContract(packageEntry, label);

  const modulesPrefix = packageEntryPath ? `${packageEntryPath}/node_modules` : "node_modules";
  const reactVersion = lockMeta.packages?.[`${modulesPrefix}/react`]?.version;
  const reactDomVersion = lockMeta.packages?.[`${modulesPrefix}/react-dom`]?.version;

  assert.equal(reactVersion, packageEntry.dependencies.react, `${label} lockfile react package must match dependency pin`);
  assert.equal(reactDomVersion, packageEntry.dependencies["react-dom"], `${label} lockfile react-dom package must match dependency pin`);
  assert.equal(reactVersion, reactDomVersion, `${label} lockfile must install matching react and react-dom versions`);
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
  liveTradeQuoteBudget: "section-trade",
  liveTradeQuoteBalanceFraction: "section-trade",
  platformKeys: "section-api",
  "section-trade": "section-trade",
};

const configLayoutModule = loadConfigLayoutModule();
const configPageIds = Array.from(configLayoutModule.CONFIG_PAGE_IDS);
const configPanelIds = Array.from(configLayoutModule.CONFIG_PANEL_IDS);
const configPanelDefaultPage = { ...configLayoutModule.CONFIG_PANEL_DEFAULT_PAGE };
const configTargetPageMap = { ...configLayoutModule.CONFIG_TARGET_PAGE_MAP };

test("repo contract pins React runtime packages together", () => {
  assertReactRuntimePackageContract(webPackageJson, "haskell/web/package.json");
  assertReactRuntimeLockContract(webPackageLock, "", "haskell/web/package-lock.json");
  assertReactRuntimeLockContract(rootPackageLock, "haskell/web", "package-lock.json workspace entry");
});

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

test("repo contract keeps top-combo method sanitization aligned with shared method ids", () => {
  const contractMethods = parseTsConstStringArray(contractsSource, "METHOD_IDS");
  assert.ok(contractMethods.includes("kalman_physics_error"), "shared method contract must include kalman_physics_error");

  const usesSharedMethodIds = /const methods:\s*Method\[\]\s*=\s*\[\.\.\.METHOD_IDS\];/.test(appSource);
  if (usesSharedMethodIds) return;

  const appMethodsMatch = appSource.match(/const methods:\s*Method\[\]\s*=\s*\[(.*?)\];/s);
  assert.ok(appMethodsMatch, "expected sanitizeTopCombosPayload method allowlist in App.tsx");
  const appMethods = parseStringLiterals(appMethodsMatch[1]);
  assert.deepEqual(appMethods, contractMethods);
});

test("repo contract keeps TopCombosChart hooks ahead of empty-state early returns", () => {
  const loadingGuard = topCombosChartSource.indexOf("if (loading && combos.length === 0)");
  const emptyGuard = topCombosChartSource.indexOf("if (combos.length === 0)");
  const hoverStateHook = topCombosChartSource.indexOf("const [hoveredId, setHoveredId] = useState");
  const maxRatingHook = topCombosChartSource.indexOf("const maxRating = useMemo(");

  assert.ok(loadingGuard >= 0, "expected loading guard in TopCombosChart");
  assert.ok(emptyGuard >= 0, "expected empty guard in TopCombosChart");
  assert.ok(hoverStateHook >= 0, "expected hover state hook in TopCombosChart");
  assert.ok(maxRatingHook >= 0, "expected maxRating hook in TopCombosChart");
  assert.ok(
    hoverStateHook < loadingGuard && hoverStateHook < emptyGuard,
    "TopCombosChart must establish hook state before prop-dependent early returns.",
  );
  assert.ok(
    maxRatingHook < loadingGuard && maxRatingHook < emptyGuard,
    "TopCombosChart must establish memoized hook state before prop-dependent early returns.",
  );
});

test("repo contract routes top-combo integer imports through the exact safe-integer boundary", () => {
  for (const field of ["bars", "epochs", "hiddenSize", "patience", "volLookback", "walkForwardFolds"]) {
    assert.match(
      appSource,
      new RegExp(`readExactSafeInteger\\(params\\.${field}\\)`),
      `${field} combo imports must use readExactSafeInteger`,
    );
  }
  assert.match(
    appSource,
    /readNonNegativeExactSafeInteger\(params\.maxOrderErrors\)/,
    "maxOrderErrors combo imports must preserve the zero-disable sentinel via the non-negative exact-safe boundary",
  );
  assert.match(
    appSource,
    /sanitizeOptimizationComboOperation\(rawOp\)/,
    "combo operation imports must share the exact-safe discrete sanitizer",
  );
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
    /liveTradeQuoteBudget:\s*"section-trade"/,
    "liveTradeQuoteBudget sizing issues must route to the Trade config page",
  );
  assert.match(
    targetPageMapBody,
    /liveTradeQuoteBalanceFraction:\s*"section-trade"/,
    "liveTradeQuoteBalanceFraction sizing issues must route to the Trade config page",
  );
});

test("repo contract keeps open-position controls on the live-monitoring defaults", () => {
  assert.match(
    appSource,
    /const \[binancePositionsBars,\s*setBinancePositionsBars\] = useState\(20\);/,
    "open positions should default to 20 chart bars",
  );
  assert.match(
    appSource,
    /const \[binancePositionsSortKey,\s*setBinancePositionsSortKey\] = useState<BinancePositionSortKey>\("pnl"\);/,
    "open positions should default to PNL ordering",
  );
  assert.match(
    appSource,
    /const \[binancePositionsSortDirection,\s*setBinancePositionsSortDirection\] = useState<BinancePositionSortDirection>\("desc"\);/,
    "open positions should default to descending ordering",
  );
});

test("repo contract keeps frontend valRatio strictly below the backend upper bound", () => {
  assert.match(
    appSource,
    /clamp\(params\.valRatio,\s*0,\s*0\.999999\)/,
    "top-combo sanitization must keep valRatio below 1 before combo apply",
  );
  assert.match(
    appSource,
    /valRatio:\s*clamp\(form\.valRatio,\s*0,\s*0\.999999\)/,
    "request emission must keep valRatio below 1 to satisfy backend validation",
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

import assert from "node:assert/strict";
import { test } from "node:test";
import { buildSync } from "esbuild";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { applyComboToForm, buildDefaultOptimizerRunForm, invalidSymbolsForPlatform, sanitizeSymbolForPlatform } from "../.tmp/web-tests/appHelpers.js";
import { defaultForm, normalizeFormState } from "../.tmp/web-tests/formState.js";
import { methodLabel } from "../.tmp/web-tests/utils.js";

const testDir = dirname(fileURLToPath(import.meta.url));
const comboMarketBundle = buildSync({
  entryPoints: [resolve(testDir, "../src/app/comboMarket.ts")],
  bundle: true,
  format: "esm",
  platform: "node",
  write: false,
});
const { comboMarketLabel, comboMarketValue } = await import(
  `data:text/javascript;base64,${Buffer.from(comboMarketBundle.outputFiles[0].text).toString("base64")}`,
);


test("comboMarketValue keeps explicit params.platform precedence for filtering and display", () => {
  const cases = [
    { combo: { params: { platform: "coinbase" }, source: "binance" }, expectedValue: "coinbase", expectedLabel: "Coinbase" },
    { combo: { params: { platform: "kraken" }, source: "csv" }, expectedValue: "kraken", expectedLabel: "Kraken" },
    { combo: { params: { platform: "poloniex" }, source: null }, expectedValue: "poloniex", expectedLabel: "Poloniex" },
  ];

  for (const { combo, expectedValue, expectedLabel } of cases) {
    const market = comboMarketValue(combo);
    assert.equal(market, expectedValue);
    assert.equal(comboMarketLabel(market), expectedLabel);
  }
});

test("comboMarketValue falls back from non-csv source to csv to unknown", () => {
  const cases = [
    { combo: { params: {}, source: "binance" }, expectedValue: "binance", expectedLabel: "Binance" },
    { combo: { params: { platform: null }, source: "csv" }, expectedValue: "csv", expectedLabel: "CSV" },
    { combo: { params: {}, source: null }, expectedValue: "unknown", expectedLabel: "Unknown" },
  ];

  for (const { combo, expectedValue, expectedLabel } of cases) {
    const market = comboMarketValue(combo);
    assert.equal(market, expectedValue);
    assert.equal(comboMarketLabel(market), expectedLabel);
  }
});

test("normalizeFormState canonicalizes slash-delimited symbols per platform", () => {
  assert.equal(normalizeFormState({ platform: "binance", binanceSymbol: "eth/usdt" }).binanceSymbol, "ETHUSDT");
  assert.equal(normalizeFormState({ platform: "coinbase", binanceSymbol: "eth/usd" }).binanceSymbol, "ETH-USD");
  assert.equal(normalizeFormState({ platform: "poloniex", binanceSymbol: "eth/usdt" }).binanceSymbol, "ETH_USDT");
});

test("normalizeFormState falls back to platform defaults for unsanitizable cross-platform restores", () => {
  const cases = [
    { platform: "coinbase", rawSymbol: "BTCUSDT", expected: "BTC-USD" },
    { platform: "poloniex", rawSymbol: "BTCUSDT", expected: "BTC_USDT" },
  ];

  for (const { platform, rawSymbol, expected } of cases) {
    const restored = normalizeFormState({ platform, binanceSymbol: rawSymbol });
    assert.equal(restored.binanceSymbol, expected);
    assert.equal(sanitizeSymbolForPlatform(platform, restored.binanceSymbol), restored.binanceSymbol);
  }
});

test("normalizeFormState canonicalizes stale delimited restores for binance-like platforms", () => {
  const cases = [
    { platform: "binance", rawSymbol: "btc/usdt", expected: "BTCUSDT" },
    { platform: "kraken", rawSymbol: "eth-usdt", expected: "ETHUSDT" },
  ];

  for (const { platform, rawSymbol, expected } of cases) {
    const restored = normalizeFormState({ platform, binanceSymbol: rawSymbol });
    assert.equal(restored.binanceSymbol, expected);
    assert.equal(sanitizeSymbolForPlatform(platform, restored.binanceSymbol), restored.binanceSymbol);
  }
});

test("frontend symbol validation trims numeric combo suffixes for binance-like platforms", () => {
  const cases = [
    { platform: "binance", rawSymbol: "BTCUSDT_123", expected: "BTCUSDT" },
    { platform: "binance", rawSymbol: "btc/usdt_123", expected: "BTCUSDT" },
    { platform: "kraken", rawSymbol: "xbt/usd_99", expected: "XBTUSD" },
  ];

  for (const { platform, rawSymbol, expected } of cases) {
    assert.equal(sanitizeSymbolForPlatform(platform, rawSymbol), expected);
  }
});

test("normalizeFormState trims numeric combo suffixes during binance-like restore", () => {
  const cases = [
    { platform: "binance", rawSymbol: "BTCUSDT_123", expected: "BTCUSDT" },
    { platform: "binance", rawSymbol: "btc/usdt_123", expected: "BTCUSDT" },
    { platform: "kraken", rawSymbol: "xbt/usd_99", expected: "XBTUSD" },
  ];

  for (const { platform, rawSymbol, expected } of cases) {
    const restored = normalizeFormState({ platform, binanceSymbol: rawSymbol });
    assert.equal(restored.binanceSymbol, expected);
    assert.equal(sanitizeSymbolForPlatform(platform, restored.binanceSymbol), restored.binanceSymbol);
  }
});

test("frontend symbol validation accepts backend-compatible aliases", () => {
  assert.equal(sanitizeSymbolForPlatform("binance", "ETH/USDT"), "ETHUSDT");
  assert.equal(sanitizeSymbolForPlatform("coinbase", "ETH/USD"), "ETH-USD");
  assert.equal(sanitizeSymbolForPlatform("poloniex", "ETH-USDT"), "ETH_USDT");
  assert.deepEqual(invalidSymbolsForPlatform("coinbase", ["ETH/USD", "BTC-USD"]), []);
});

test("frontend symbol validation accepts spaced Coinbase and Poloniex delimiters without widening formats", () => {
  assert.equal(sanitizeSymbolForPlatform("coinbase", " eth - usd "), "ETH-USD");
  assert.equal(sanitizeSymbolForPlatform("poloniex", " eth _ usdt "), "ETH_USDT");
  assert.deepEqual(
    invalidSymbolsForPlatform("coinbase", ["ETH - USD", "BTC - USD", "ETH - USD - EXTRA"]),
    ["ETH - USD - EXTRA"],
  );
  assert.deepEqual(
    invalidSymbolsForPlatform("poloniex", ["ETH _ USDT", "BTC _ USDT", "ETH _ USDT _ EXTRA"]),
    ["ETH _ USDT _ EXTRA"],
  );
});

test("normalizeFormState canonicalizes spaced delimited symbols per platform", () => {
  assert.equal(normalizeFormState({ platform: "coinbase", binanceSymbol: " eth - usd " }).binanceSymbol, "ETH-USD");
  assert.equal(normalizeFormState({ platform: "poloniex", binanceSymbol: " eth _ usdt " }).binanceSymbol, "ETH_USDT");
});

test("applyComboToForm keeps valid slash-delimited combo symbols instead of falling back", () => {
  const combo = {
    id: 7,
    openThreshold: defaultForm.openThreshold,
    closeThreshold: defaultForm.closeThreshold,
    params: {
      platform: "coinbase",
      binanceSymbol: "ETH/USD",
      interval: defaultForm.interval,
      method: defaultForm.method,
      positioning: defaultForm.positioning,
      normalization: defaultForm.normalization,
      fee: defaultForm.fee,
      epochs: defaultForm.epochs,
      hiddenSize: defaultForm.hiddenSize,
    },
  };
  const next = applyComboToForm(defaultForm, combo, null);
  assert.equal(next.platform, "coinbase");
  assert.equal(next.binanceSymbol, "ETH-USD");
});

test("applyComboToForm keeps spaced delimited combo symbols instead of falling back", () => {
  const cases = [
    { id: 8, platform: "coinbase", raw: "ETH - USD", expected: "ETH-USD" },
    { id: 9, platform: "poloniex", raw: "ETH _ USDT", expected: "ETH_USDT" },
  ];

  for (const { id, platform, raw, expected } of cases) {
    const combo = {
      id,
      openThreshold: defaultForm.openThreshold,
      closeThreshold: defaultForm.closeThreshold,
      params: {
        platform,
        binanceSymbol: raw,
        interval: defaultForm.interval,
        method: defaultForm.method,
        positioning: defaultForm.positioning,
        normalization: defaultForm.normalization,
        fee: defaultForm.fee,
        epochs: defaultForm.epochs,
        hiddenSize: defaultForm.hiddenSize,
      },
    };
    const next = applyComboToForm(defaultForm, combo, null);
    assert.equal(next.platform, platform);
    assert.equal(next.binanceSymbol, expected);
  }
});

test("applyComboToForm trims numeric combo suffixes for binance-like combo imports", () => {
  const cases = [
    { id: 10, platform: "binance", raw: "BTCUSDT_123", expected: "BTCUSDT" },
    { id: 11, platform: "binance", raw: "btc/usdt_123", expected: "BTCUSDT" },
    { id: 12, platform: "kraken", raw: "xbt/usd_99", expected: "XBTUSD" },
  ];

  for (const { id, platform, raw, expected } of cases) {
    const combo = {
      id,
      openThreshold: defaultForm.openThreshold,
      closeThreshold: defaultForm.closeThreshold,
      params: {
        platform,
        binanceSymbol: raw,
        interval: defaultForm.interval,
        method: defaultForm.method,
        positioning: defaultForm.positioning,
        normalization: defaultForm.normalization,
        fee: defaultForm.fee,
        epochs: defaultForm.epochs,
        hiddenSize: defaultForm.hiddenSize,
      },
    };
    const next = applyComboToForm(defaultForm, combo, null);
    assert.equal(next.platform, platform);
    assert.equal(next.binanceSymbol, expected);
  }
});

test("coinbase restores keep live-order mode enabled", () => {
  const restored = normalizeFormState({
    platform: "coinbase",
    binanceLive: true,
    tradeArmed: true,
  });
  assert.equal(restored.platform, "coinbase");
  assert.equal(restored.binanceLive, true);
  assert.equal(restored.tradeArmed, true);
});

test("applyComboToForm preserves live-order toggles exactly on supported trade platforms", () => {
  const cases = [
    { id: 13, platform: "binance", symbol: "ETHUSDT", expectedLive: true, expectedTradeArmed: true },
    { id: 14, platform: "coinbase", symbol: "ETH-USD", expectedLive: true, expectedTradeArmed: true },
    { id: 15, platform: "kraken", symbol: "ETHUSD", expectedLive: false, expectedTradeArmed: false },
    { id: 16, platform: "poloniex", symbol: "ETH_USDT", expectedLive: false, expectedTradeArmed: false },
  ];

  for (const { id, platform, symbol, expectedLive, expectedTradeArmed } of cases) {
    const prev = {
      ...defaultForm,
      platform: "binance",
      binanceSymbol: "BTCUSDT",
      binanceLive: true,
      tradeArmed: true,
    };
    const combo = {
      id,
      openThreshold: prev.openThreshold,
      closeThreshold: prev.closeThreshold,
      params: {
        platform,
        binanceSymbol: symbol,
        interval: prev.interval,
        method: prev.method,
        positioning: prev.positioning,
        normalization: prev.normalization,
        fee: prev.fee,
        epochs: prev.epochs,
        hiddenSize: prev.hiddenSize,
      },
    };

    const next = applyComboToForm(prev, combo, null);

    assert.equal(next.platform, platform);
    assert.equal(next.binanceLive, expectedLive);
    assert.equal(next.tradeArmed, expectedTradeArmed);
  }
});

test("normalizeFormState keeps representative whole-number restores aligned with emitted request bounds", () => {
  const restored = normalizeFormState({
    trendLookback: "1000001",
    routerLookback: "1000001",
    minRoundTrips: "1000001",
    walkForwardFolds: "0",
    walkForwardEmbargoBars: "-5",
    botPollSeconds: "3605",
    botTrainBars: "5",
    botMaxPoints: "100001",
  });

  assert.equal(restored.trendLookback, 1_000_000);
  assert.equal(restored.routerLookback, 1_000_000);
  assert.equal(restored.minRoundTrips, 1_000_000);
  assert.equal(restored.walkForwardFolds, 1);
  assert.equal(restored.walkForwardEmbargoBars, 0);
  assert.equal(restored.botPollSeconds, 3600);
  assert.equal(restored.botTrainBars, 10);
  assert.equal(restored.botMaxPoints, 100_000);
});

test("normalizeFormState falls back from representative fractional legacy saved states on integer-backed fields", () => {
  const restored = normalizeFormState({
    trendLookback: "45.5",
    routerLookback: 12.25,
    minRoundTrips: "9.5",
    walkForwardFolds: 6.5,
    walkForwardEmbargoBars: "3.25",
    botPollSeconds: "60.5",
    botOnlineEpochs: 4.5,
    botTrainBars: "1200.25",
    botMaxPoints: 2500.5,
  });

  assert.equal(restored.trendLookback, defaultForm.trendLookback);
  assert.equal(restored.routerLookback, defaultForm.routerLookback);
  assert.equal(restored.minRoundTrips, defaultForm.minRoundTrips);
  assert.equal(restored.walkForwardFolds, defaultForm.walkForwardFolds);
  assert.equal(restored.walkForwardEmbargoBars, defaultForm.walkForwardEmbargoBars);
  assert.equal(restored.botPollSeconds, defaultForm.botPollSeconds);
  assert.equal(restored.botOnlineEpochs, defaultForm.botOnlineEpochs);
  assert.equal(restored.botTrainBars, defaultForm.botTrainBars);
  assert.equal(restored.botMaxPoints, defaultForm.botMaxPoints);
});

test("rebalance cost defaults stay aligned with backend defaults", () => {
  assert.equal(defaultForm.rebalanceCostMult, 0);
  const optimizerDefaults = buildDefaultOptimizerRunForm("BTCUSDT", "binance");
  assert.equal(optimizerDefaults.rebalanceCostMultMin, "");
  assert.equal(optimizerDefaults.rebalanceCostMultMax, "");
});

test("methodLabel includes kalman_physics_error", () => {
  assert.equal(methodLabel("kalman_physics_error"), "Kalman physics error");
});

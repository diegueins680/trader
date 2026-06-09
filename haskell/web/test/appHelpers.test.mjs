import assert from "node:assert/strict";
import { test } from "node:test";
import {
  adjustBacktestParamsForSplit,
  autoFitLookbackToBars,
  buildOpenBinancePositionSymbolSet,
  buildDefaultOptimizerRunForm,
  buildOptimizerCorrelationGuess,
  buildOptimizerRunRequest,
  findOptionalWholeNumberFieldError,
  formatDatetimeLocal,
  parseDatetimeLocal,
  isOpenBinancePosition,
  positionSideInfo,
  readExactSafeInteger,
  readNonNegativeExactSafeInteger,
  sanitizeOptimizationComboOperation,
  splitStats,
} from "../.tmp/web-tests/appHelpers.js";

function optimizerCombo(id, finalEquity, params) {
  return {
    id,
    finalEquity,
    openThreshold: null,
    closeThreshold: null,
    source: "binance",
    params: {
      binanceSymbol: "BTCUSDT",
      platform: "binance",
      interval: "1h",
      method: "10",
      normalization: "none",
      ...params,
    },
  };
}

function pad2(value) {
  return String(value).padStart(2, "0");
}

function isLeapYear(year) {
  return (year % 4 === 0 && year % 100 !== 0) || year % 400 === 0;
}

function daysInMonth(year, month) {
  switch (month) {
    case 2:
      return isLeapYear(year) ? 29 : 28;
    case 4:
    case 6:
    case 9:
    case 11:
      return 30;
    default:
      return 31;
  }
}

test("parseDatetimeLocal accepts exactly the valid month-end local timestamps in a bounded matrix", () => {
  const years = [2023, 2024];
  const days = [28, 29, 30, 31];

  for (const year of years) {
    for (let month = 1; month <= 12; month += 1) {
      for (const day of days) {
        const raw = `${year}-${pad2(month)}-${pad2(day)}T12:34`;
        const parsed = parseDatetimeLocal(raw);
        const valid = day <= daysInMonth(year, month);
        assert.equal(parsed !== null, valid, `${raw}: expected calendar validity ${valid}`);
        if (valid) {
          assert.equal(
            formatDatetimeLocal(parsed),
            raw,
            `${raw}: expected parse/format to preserve valid local timestamps`,
          );
        }
      }
    }
  }
});

test("parseDatetimeLocal rejects impossible local timestamps that Date.parse would otherwise normalize", () => {
  assert.equal(parseDatetimeLocal("2024-02-31T12:34"), null);
  assert.equal(parseDatetimeLocal("2023-04-31T00:00"), null);
  assert.equal(parseDatetimeLocal("2024-02-29 12:34"), parseDatetimeLocal("2024-02-29T12:34"));
});

test("formatDatetimeLocal suppresses finite out-of-range timestamps instead of rendering NaN fragments", () => {
  assert.equal(formatDatetimeLocal(1e20), "");
  assert.equal(formatDatetimeLocal(-1e20), "");
});

test("readExactSafeInteger accepts only exact safe integers from combo payloads", () => {
  const unsafe = Number.MAX_SAFE_INTEGER + 1;
  const cases = [
    { raw: 0, expected: 0 },
    { raw: -7, expected: -7 },
    { raw: 12.5, expected: null },
    { raw: Number.NaN, expected: null },
    { raw: Number.POSITIVE_INFINITY, expected: null },
    { raw: unsafe, expected: null },
    { raw: "9", expected: null },
  ];

  for (const { raw, expected } of cases) {
    assert.equal(
      readExactSafeInteger(raw),
      expected,
      `expected ${String(raw)} to normalize to ${String(expected)}`,
    );
  }
});

test("readNonNegativeExactSafeInteger preserves zero and rejects negative or inexact discrete payloads", () => {
  const unsafe = Number.MAX_SAFE_INTEGER + 1;
  const cases = [
    { raw: 0, expected: 0 },
    { raw: 7, expected: 7 },
    { raw: -1, expected: null },
    { raw: 2.5, expected: null },
    { raw: Number.NaN, expected: null },
    { raw: unsafe, expected: null },
  ];

  for (const { raw, expected } of cases) {
    assert.equal(
      readNonNegativeExactSafeInteger(raw),
      expected,
      `expected ${String(raw)} to normalize to ${String(expected)}`,
    );
  }
});

test("findOptionalWholeNumberFieldError accepts exact string overrides and rejects fractional string overrides", () => {
  assert.equal(
    findOptionalWholeNumberFieldError([{ label: "Trials", raw: "", override: " 12 " }]),
    null,
  );
  assert.equal(
    findOptionalWholeNumberFieldError([{ label: "Trials", raw: "", override: "12.5" }]),
    "Trials must be a whole number.",
  );
});

test("buildOptimizerRunRequest normalizes known integer extra overrides before merging", () => {
  const form = {
    ...buildDefaultOptimizerRunForm("BTCUSDT", "binance"),
    trials: "",
    barsMin: "",
  };
  const request = buildOptimizerRunRequest(form, {
    trials: "12",
    barsMin: "250",
    objective: "roi",
  });

  assert.equal(request.trials, 12);
  assert.equal(request.barsMin, 250);
  assert.equal(request.objective, "roi");
  assert.equal(typeof request.trials, "number");
  assert.equal(typeof request.barsMin, "number");
});

test("buildOptimizerRunRequest canonicalizes known source, symbol, and finite numeric extra overrides while enforcing source compatibility", () => {
  const form = {
    ...buildDefaultOptimizerRunForm("BTCUSDT", "binance"),
    timeoutSec: "",
    backtestRatio: "",
    tuneRatio: "",
  };
  const request = buildOptimizerRunRequest(form, {
    source: " Coinbase ",
    binanceSymbol: " btc-usd ",
    data: " ../data/sample_prices.csv ",
    timeoutSec: "12",
    backtestRatio: "0.2",
    tuneRatio: "0.25",
  });

  assert.equal(request.source, "coinbase");
  assert.equal(request.binanceSymbol, "BTC-USD");
  assert.equal("data" in request, false);
  assert.equal(request.timeoutSec, 12);
  assert.equal(request.backtestRatio, 0.2);
  assert.equal(request.tuneRatio, 0.25);
  assert.equal(typeof request.timeoutSec, "number");
  assert.equal(typeof request.backtestRatio, "number");
  assert.equal(typeof request.tuneRatio, "number");
});

test("buildOptimizerRunRequest drops invalid known integer extra overrides instead of emitting stringly payloads", () => {
  const unsafe = (BigInt(Number.MAX_SAFE_INTEGER) + 1n).toString();
  const form = {
    ...buildDefaultOptimizerRunForm("BTCUSDT", "binance"),
    trials: "",
    trendLookbackMin: "",
  };
  const request = buildOptimizerRunRequest(form, {
    trials: "12.5",
    trendLookbackMin: unsafe,
  });

  assert.equal("trials" in request, false);
  assert.equal("trendLookbackMin" in request, false);
});

test("buildOptimizerRunRequest keeps invalid known source and finite-number extra overrides inert", () => {
  const form = {
    ...buildDefaultOptimizerRunForm("BTCUSDT", "binance"),
    timeoutSec: "",
    backtestRatio: "",
    tuneRatio: "",
  };
  const request = buildOptimizerRunRequest(form, {
    source: "bogus",
    binanceSymbol: "   ",
    data: "   ",
    timeoutSec: "oops",
    backtestRatio: "nan",
    tuneRatio: "inf",
  });

  assert.equal(request.source, "binance");
  assert.equal(request.binanceSymbol, "BTCUSDT");
  assert.equal("data" in request, false);
  assert.equal("timeoutSec" in request, false);
  assert.equal("backtestRatio" in request, false);
  assert.equal("tuneRatio" in request, false);
});

test("buildOptimizerCorrelationGuess narrows visible optimizer ranges from high-ROI parameter correlations", () => {
  const combos = [
    optimizerCombo(1, 1.01, { stopLoss: 0.02 }),
    optimizerCombo(2, 1.02, { stopLoss: 0.04 }),
    optimizerCombo(3, 1.03, { stopLoss: 0.06 }),
    optimizerCombo(4, 1.08, { stopLoss: 0.1 }),
    optimizerCombo(5, 1.1, { stopLoss: 0.12 }),
  ];
  const guess = buildOptimizerCorrelationGuess(combos);

  assert.ok(guess);
  assert.equal(guess.patch.source, "binance");
  assert.equal(guess.patch.symbol, "BTCUSDT");
  assert.equal(guess.patch.intervals, "1h");
  assert.equal(guess.patch.minRoundTrips, "3");
  assert.equal(guess.patch.timeoutSec, "120");
  assert.ok(Number(guess.patch.stopMin) >= 0.09);
  assert.ok(Number(guess.patch.stopMax) <= 0.12);
  assert.ok(Number(guess.patch.stopMax) >= Number(guess.patch.stopMin));
  assert.ok(guess.basis.some((item) => item.startsWith("Stop Loss r ")));
});

test("buildOptimizerCorrelationGuess emits advanced optimizer knobs through extra JSON fields", () => {
  const combos = [
    optimizerCombo(1, 1.01, { maxHighVolProb: 0.2 }),
    optimizerCombo(2, 1.02, { maxHighVolProb: 0.4 }),
    optimizerCombo(3, 1.05, { maxHighVolProb: 0.6 }),
    optimizerCombo(4, 1.09, { maxHighVolProb: 0.8 }),
  ];
  const guess = buildOptimizerCorrelationGuess(combos);

  assert.ok(guess);
  assert.ok(guess.extras.maxHighVolProbMin >= 0.55);
  assert.ok(guess.extras.maxHighVolProbMax <= 0.8);
  assert.ok(guess.extras.maxHighVolProbMax >= guess.extras.maxHighVolProbMin);
});

test("buildOptimizerCorrelationGuess returns null when mapped parameters have no usable variation", () => {
  const combos = [
    optimizerCombo(1, 1.01, { stopLoss: 0.02 }),
    optimizerCombo(2, 1.02, { stopLoss: 0.02 }),
    optimizerCombo(3, 1.03, { stopLoss: 0.02 }),
    optimizerCombo(4, 1.04, { stopLoss: 0.02 }),
  ];

  assert.equal(buildOptimizerCorrelationGuess(combos), null);
});

test("buildOptimizerRunRequest drops CSV-only known overrides when the effective source is exchange data", () => {
  const form = {
    ...buildDefaultOptimizerRunForm("BTCUSDT", "binance"),
    source: "csv",
    dataPath: "../data/form.csv",
    priceColumn: "close",
    highColumn: "high",
    lowColumn: "low",
    symbol: "ignored-csv-symbol",
    platforms: "coinbase",
  };
  const request = buildOptimizerRunRequest(form, {
    source: " coinbase ",
    binanceSymbol: " eth-usd ",
    data: " ../data/override.csv ",
    priceColumn: " adjusted_close ",
    highColumn: " high_override ",
    lowColumn: " low_override ",
    platforms: " coinbase,kraken ",
  });

  assert.equal(request.source, "coinbase");
  assert.equal(request.binanceSymbol, "ETH-USD");
  assert.equal(request.platforms, "coinbase,kraken");
  assert.equal("data" in request, false);
  assert.equal("priceColumn" in request, false);
  assert.equal("highColumn" in request, false);
  assert.equal("lowColumn" in request, false);
});

test("adjustBacktestParamsForSplit preserves holdout ratio by raising bars when the cap allows it", () => {
  const adjusted = adjustBacktestParamsForSplit(
    {
      platform: "binance",
      method: "10",
      interval: "3m",
      bars: 500,
      lookbackBars: 499,
      backtestRatio: 0.2,
    },
    { platform: "binance", method: "10", apiLimits: null },
  );

  assert.equal(adjusted.params.bars, 625);
  assert.equal(adjusted.params.backtestRatio, 0.2);
  assert.equal(adjusted.changes?.bars, 625);
  assert.equal(adjusted.changes?.backtestRatio, undefined);
});

test("adjustBacktestParamsForSplit raises bars and lowers holdout when lookback would disable backtest", () => {
  const adjusted = adjustBacktestParamsForSplit(
    {
      platform: "binance",
      method: "10",
      interval: "3m",
      bars: 961,
      lookbackBars: 960,
      backtestRatio: 0.2,
    },
    { platform: "binance", method: "10", apiLimits: null },
  );

  assert.equal(adjusted.params.bars, 1000);
  assert.equal(adjusted.changes?.bars, 1000);
  assert.equal(adjusted.changes?.backtestRatio, adjusted.params.backtestRatio);
  assert.ok(Math.abs(adjusted.params.backtestRatio - 0.0385) < 1e-9);

  const stats = splitStats(adjusted.params.bars, adjusted.params.backtestRatio, 960, 0, false);
  assert.equal(stats.trainOk, true);
  assert.equal(stats.backtestOk, true);
});

test("adjustBacktestParamsForSplit shrinks an override lookback when the cap leaves no valid split", () => {
  const adjusted = adjustBacktestParamsForSplit(
    {
      platform: "binance",
      method: "10",
      interval: "3m",
      bars: 1000,
      lookbackBars: 999,
      backtestRatio: 0.2,
    },
    { platform: "binance", method: "10", apiLimits: null },
  );

  assert.equal(adjusted.changes?.lookbackBars, 799);
  assert.equal(adjusted.changes?.bars, undefined);
  assert.equal(adjusted.changes?.backtestRatio, undefined);
  assert.equal(adjusted.params.lookbackBars, 799);

  const stats = splitStats(1000, 0.2, adjusted.params.lookbackBars, 0, false);
  assert.equal(stats.trainOk, true);
  assert.equal(stats.backtestOk, true);
});

test("adjustBacktestParamsForSplit shrinks a window lookback when the cap leaves no valid split", () => {
  const adjusted = adjustBacktestParamsForSplit(
    {
      platform: "binance",
      method: "10",
      interval: "1m",
      bars: 1000,
      lookbackBars: 0,
      lookbackWindow: "999m",
      backtestRatio: 0.2,
    },
    { platform: "binance", method: "10", apiLimits: null },
  );

  assert.equal(adjusted.changes?.lookbackWindow, "799m");
  assert.equal(adjusted.params.lookbackWindow, "799m");
  assert.equal(adjusted.params.lookbackBars, 0);
});

test("adjustBacktestParamsForSplit prefers raising bars over shrinking lookback when the cap allows it", () => {
  const adjusted = adjustBacktestParamsForSplit(
    {
      platform: "binance",
      method: "10",
      interval: "3m",
      bars: 500,
      lookbackBars: 499,
      backtestRatio: 0.2,
    },
    { platform: "binance", method: "10", apiLimits: null },
  );

  assert.equal(adjusted.changes?.bars, 625);
  assert.equal(adjusted.changes?.lookbackBars, undefined);
  assert.equal(adjusted.changes?.lookbackWindow, undefined);
});

test("autoFitLookbackToBars leaves lookback alone when bars can be raised within the cap", () => {
  const fitted = autoFitLookbackToBars("binance", "1m", 0, "12h", 500, "11", {
    maxBarsLstm: 1000,
    maxEpochs: 100,
    maxHiddenSize: 50,
  });

  assert.equal(fitted, null);
});

test("autoFitLookbackToBars reduces a window lookback when it exceeds available capped bars", () => {
  const fitted = autoFitLookbackToBars("binance", "1m", 0, "7d", 978, "11", {
    maxBarsLstm: 1000,
    maxEpochs: 100,
    maxHiddenSize: 50,
  });

  assert.deepEqual(fitted, { action: "lookbackWindow", value: "977m" });
});

test("autoFitLookbackToBars reduces a bars override when it exceeds available capped bars", () => {
  const fitted = autoFitLookbackToBars("binance", "1m", 3360, "7d", 978, "11", {
    maxBarsLstm: 1000,
    maxEpochs: 100,
    maxHiddenSize: 50,
  });

  assert.deepEqual(fitted, { action: "lookbackBars", value: 977 });
});

test("buildOptimizerRunRequest drops exchange-only known overrides when the effective source is csv", () => {
  const form = {
    ...buildDefaultOptimizerRunForm("BTCUSDT", "binance"),
    source: "coinbase",
    symbol: "BTC-USD",
    platforms: "coinbase,kraken",
  };
  const request = buildOptimizerRunRequest(form, {
    source: " csv ",
    data: " ../data/override.csv ",
    priceColumn: " adjusted_close ",
    highColumn: " high_override ",
    lowColumn: " low_override ",
    binanceSymbol: " eth-usd ",
    platforms: " coinbase,kraken ",
  });

  assert.equal(request.source, "csv");
  assert.equal(request.data, "../data/override.csv");
  assert.equal(request.priceColumn, "adjusted_close");
  assert.equal(request.highColumn, "high_override");
  assert.equal(request.lowColumn, "low_override");
  assert.equal("binanceSymbol" in request, false);
  assert.equal("platforms" in request, false);
});

test("sanitizeOptimizationComboOperation preserves only exact non-negative discrete operation coordinates", () => {
  assert.deepEqual(
    sanitizeOptimizationComboOperation({
      entryIndex: 3,
      exitIndex: 8,
      entryEquity: 1000,
      exitEquity: 1025,
      return: 0.025,
      holdingPeriods: 5,
      exitReason: " target ",
    }),
    {
      entryIndex: 3,
      exitIndex: 8,
      entryEquity: 1000,
      exitEquity: 1025,
      return: 0.025,
      holdingPeriods: 5,
      exitReason: "target",
    },
  );
  assert.equal(sanitizeOptimizationComboOperation({ entryIndex: 1.5, exitIndex: 4 }), null);
  assert.equal(sanitizeOptimizationComboOperation({ entryIndex: 4, exitIndex: 3 }), null);
  assert.deepEqual(
    sanitizeOptimizationComboOperation({ entryIndex: 1, exitIndex: 4, holdingPeriods: 2.5 }),
    {
      entryIndex: 1,
      exitIndex: 4,
      entryEquity: null,
      exitEquity: null,
      return: null,
      holdingPeriods: null,
      exitReason: null,
    },
  );
});

test("positionSideInfo treats zero/dust amounts as flat before stale side metadata", () => {
  assert.deepEqual(positionSideInfo(0, "LONG"), { dir: 0, label: "FLAT", key: "FLAT" });
  assert.deepEqual(positionSideInfo(1e-13, "SHORT"), { dir: 0, label: "FLAT", key: "FLAT" });
  assert.deepEqual(positionSideInfo(2, "SHORT"), { dir: -1, label: "SHORT", key: "SHORT" });
});

test("positionSideInfo suppresses non-finite amounts before stale side metadata", () => {
  assert.deepEqual(positionSideInfo(Number.NaN, "LONG"), { dir: 0, label: "FLAT", key: "FLAT" });
  assert.deepEqual(positionSideInfo(Number.POSITIVE_INFINITY, "SHORT"), { dir: 0, label: "FLAT", key: "FLAT" });
});

test("open Binance position helpers ignore flat dust and normalize symbols", () => {
  const positions = [
    { symbol: " btcusdt ", positionAmt: 0, entryPrice: 0, markPrice: 0, unrealizedPnl: 0, positionSide: "LONG" },
    { symbol: "ethusdt", positionAmt: 1e-13, entryPrice: 0, markPrice: 0, unrealizedPnl: 0, positionSide: "SHORT" },
    { symbol: "filusdt", positionAmt: 3, entryPrice: 0, markPrice: 0, unrealizedPnl: 0, positionSide: "BOTH" },
    { symbol: "atomusdt", positionAmt: -2, entryPrice: 0, markPrice: 0, unrealizedPnl: 0, positionSide: null },
  ];

  assert.equal(isOpenBinancePosition(positions[0]), false);
  assert.equal(isOpenBinancePosition(positions[1]), false);
  assert.equal(isOpenBinancePosition(positions[2]), true);
  assert.equal(isOpenBinancePosition(positions[3]), true);
  assert.deepEqual([...buildOpenBinancePositionSymbolSet(positions)].sort(), ["ATOMUSDT", "FILUSDT"]);
});

import assert from "node:assert/strict";
import { test } from "node:test";
import { applyComboToForm, buildDefaultOptimizerRunForm, invalidSymbolsForPlatform, sanitizeSymbolForPlatform } from "../.tmp/web-tests/appHelpers.js";
import { defaultForm, normalizeFormState } from "../.tmp/web-tests/formState.js";
import { methodLabel } from "../.tmp/web-tests/utils.js";

test("normalizeFormState canonicalizes slash-delimited symbols per platform", () => {
  assert.equal(normalizeFormState({ platform: "binance", binanceSymbol: "eth/usdt" }).binanceSymbol, "ETHUSDT");
  assert.equal(normalizeFormState({ platform: "coinbase", binanceSymbol: "eth/usd" }).binanceSymbol, "ETH-USD");
  assert.equal(normalizeFormState({ platform: "poloniex", binanceSymbol: "eth/usdt" }).binanceSymbol, "ETH_USDT");
});

test("frontend symbol validation accepts backend-compatible aliases", () => {
  assert.equal(sanitizeSymbolForPlatform("binance", "ETH/USDT"), "ETHUSDT");
  assert.equal(sanitizeSymbolForPlatform("coinbase", "ETH/USD"), "ETH-USD");
  assert.equal(sanitizeSymbolForPlatform("poloniex", "ETH-USDT"), "ETH_USDT");
  assert.deepEqual(invalidSymbolsForPlatform("coinbase", ["ETH/USD", "BTC-USD"]), []);
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

test("rebalance cost defaults stay aligned with backend defaults", () => {
  assert.equal(defaultForm.rebalanceCostMult, 0);
  const optimizerDefaults = buildDefaultOptimizerRunForm("BTCUSDT", "binance");
  assert.equal(optimizerDefaults.rebalanceCostMultMin, "");
  assert.equal(optimizerDefaults.rebalanceCostMultMax, "");
});

test("methodLabel includes kalman_physics_error", () => {
  assert.equal(methodLabel("kalman_physics_error"), "Kalman physics error");
});

import assert from "node:assert/strict";
import { test } from "node:test";
import { buildOrphanedPositions, buildRequestIssueDetails, isLocalHostname, methodLabel, normalizeApiBaseUrlInput, numFromInput } from "../.tmp/web-tests/utils.js";
import { defaultForm, normalizeFormState } from "../.tmp/web-tests/formState.js";

test("buildRequestIssueDetails returns empty when clean", () => {
  assert.deepEqual(buildRequestIssueDetails({}), []);
});

test("buildRequestIssueDetails orders issues consistently", () => {
  const issues = buildRequestIssueDetails({
    rateLimitReason: "rate limit",
    apiStatusIssue: "api down",
    apiBlockedReason: "api down details",
    apiTargetId: "section-api",
    missingSymbol: true,
    symbolTargetId: "symbol",
    missingInterval: true,
    intervalTargetId: "interval",
    lookbackError: "lookback issue",
    lookbackTargetId: "lookbackWindow",
    apiLimitsReason: "limits issue",
    apiLimitsTargetId: "bars",
  });
  assert.deepEqual(
    issues.map((issue) => issue.message),
    [
      "rate limit",
      "api down",
      "Symbol is required.",
      "Interval is required.",
      "lookback issue",
      "limits issue",
    ],
  );
  assert.equal(issues[1]?.disabledMessage, "api down details");
});

test("buildRequestIssueDetails skips falsy inputs", () => {
  const issues = buildRequestIssueDetails({
    rateLimitReason: "",
    apiStatusIssue: null,
    apiBlockedReason: null,
    missingSymbol: false,
    missingInterval: false,
    lookbackError: undefined,
    apiLimitsReason: undefined,
  });
  assert.deepEqual(issues, []);
});

test("buildOrphanedPositions flags missing bots and matches hedge sides", () => {
  const positions = [
    { symbol: "BTCUSDT", positionAmt: 1.2, positionSide: "LONG" },
    { symbol: "BTCUSDT", positionAmt: -0.4, positionSide: "SHORT" },
    { symbol: "ETHUSDT", positionAmt: 0.7, positionSide: "BOTH" },
  ];
  const bots = [
    {
      symbol: "btcusdt",
      status: { running: true, market: "futures", positions: [0.5] },
    },
    {
      symbol: "ETHUSDT",
      status: { running: true, market: "futures", positions: [-0.2] },
    },
  ];
  const orphans = buildOrphanedPositions(positions, bots, { market: "futures" });
  assert.deepEqual(
    orphans.map((entry) => entry.pos.symbol),
    ["BTCUSDT", "ETHUSDT"],
  );
  assert.equal(orphans[0]?.pos.positionSide, "SHORT");
  assert.equal(orphans[1]?.pos.positionSide, "BOTH");
  assert.equal(orphans[0]?.reason, "side mismatch (bot LONG)");
  assert.equal(orphans[1]?.reason, "side mismatch (bot SHORT)");
});

test("buildOrphanedPositions treats flat bots as orphaned", () => {
  const positions = [{ symbol: "SOLUSDT", positionAmt: -2, positionSide: "BOTH" }];
  const bots = [{ symbol: "SOLUSDT", status: { running: true, market: "futures", positions: [0] } }];
  const orphans = buildOrphanedPositions(positions, bots, { market: "futures" });
  assert.equal(orphans.length, 1);
  assert.equal(orphans[0]?.reason, "bot side unknown");
});

test("buildOrphanedPositions treats starting bots as adopted", () => {
  const positions = [{ symbol: "FILUSDT", positionAmt: 1.2, positionSide: "LONG" }];
  const bots = [{ symbol: "FILUSDT", status: { running: false, starting: true, market: "futures" } }];
  const orphans = buildOrphanedPositions(positions, bots, { market: "futures" });
  assert.equal(orphans.length, 0);
});

test("buildOrphanedPositions flags market mismatch", () => {
  const positions = [{ symbol: "BTCUSDT", positionAmt: 1, positionSide: "LONG" }];
  const bots = [{ symbol: "BTCUSDT", status: { running: true, market: "spot", positions: [1] } }];
  const orphans = buildOrphanedPositions(positions, bots, { market: "futures" });
  assert.equal(orphans.length, 1);
  assert.equal(orphans[0]?.reason, "market mismatch");
});

test("normalizeApiBaseUrlInput supports bare loopback IPv6 with port", () => {
  assert.equal(normalizeApiBaseUrlInput("::1:8080"), "http://[::1]:8080");
  assert.equal(normalizeApiBaseUrlInput("[::1]:8080"), "http://[::1]:8080");
});

test("normalizeApiBaseUrlInput supports localhost host+path without explicit scheme", () => {
  assert.equal(normalizeApiBaseUrlInput("localhost/api"), "http://localhost/api");
});

test("isLocalHostname accepts bracketed IPv6 loopback", () => {
  assert.equal(isLocalHostname("[::1]"), true);
});

test("isLocalHostname accepts 0.0.0.0", () => {
  assert.equal(isLocalHostname("0.0.0.0"), true);
});

test("numFromInput parses thousands grouping and decimal comma consistently", () => {
  assert.equal(numFromInput("1,234", 99), 1234);
  assert.equal(numFromInput("12,345", 99), 12345);
  assert.equal(numFromInput("1,234,567", 0), 1234567);
  assert.equal(numFromInput("1,23", 0), 1.23);
  assert.equal(numFromInput("0,123", 0), 0.123);
});

test("normalizeFormState restores default minPositionSize for invalid input", () => {
  const fromInvalid = normalizeFormState({ minPositionSize: "not-a-number" });
  assert.equal(fromInvalid.minPositionSize, defaultForm.minPositionSize);
  const fromExplicitZero = normalizeFormState({ minPositionSize: 0 });
  assert.equal(fromExplicitZero.minPositionSize, 0);
});

test("defaultForm uses safe trade defaults", () => {
  assert.equal(defaultForm.binanceLive, false);
  assert.equal(defaultForm.tradeArmed, false);
});

test("normalizeFormState normalizes trade toggles and booleans from strings", () => {
  const out = normalizeFormState({
    binanceLive: "false",
    tradeArmed: "1",
    optimizeOperations: "true",
    sweepThreshold: "0",
    autoRefresh: "false",
  });
  assert.equal(out.binanceLive, false);
  assert.equal(out.tradeArmed, true);
  assert.equal(out.optimizeOperations, true);
  assert.equal(out.sweepThreshold, false);
  assert.equal(out.autoRefresh, false);
});

test("normalizeFormState forces non-binance platforms into spot + disables binance-only flags", () => {
  const out = normalizeFormState({
    platform: "coinbase",
    market: "futures",
    binanceTestnet: true,
    binanceLive: true,
    tradeArmed: true,
  });
  assert.equal(out.platform, "coinbase");
  assert.equal(out.market, "spot");
  assert.equal(out.binanceTestnet, false);
  assert.equal(out.binanceLive, false);
  assert.equal(out.tradeArmed, true);
});

test("normalizeFormState disables tradeArmed for non-trading platforms", () => {
  const out = normalizeFormState({ platform: "kraken", tradeArmed: true });
  assert.equal(out.platform, "kraken");
  assert.equal(out.tradeArmed, false);
});

test("normalizeFormState treats margin+non-live as spot (safe fallback)", () => {
  const out = normalizeFormState({
    market: "margin",
    binanceLive: false,
  });
  assert.equal(out.market, "spot");
  assert.equal(out.binanceLive, false);
});

test("normalizeFormState forces margin to disable testnet", () => {
  const out = normalizeFormState({
    market: "margin",
    binanceLive: true,
    binanceTestnet: true,
  });
  assert.equal(out.market, "margin");
  assert.equal(out.binanceLive, true);
  assert.equal(out.binanceTestnet, false);
});

test("methodLabel includes newly added backend methods", () => {
  assert.equal(methodLabel("divergence_gate"), "Divergence gate (shrinks blended return on disagreement)");
  assert.equal(methodLabel("smooth_softmax_blend"), "Smooth softmax blend (EMA-smooth softmax weights)");
  assert.equal(methodLabel("conformal_clip"), "Conformal clip (clips blended return to conformal/quantile band)");
  assert.equal(methodLabel("hedge_blend"), "Hedge blend (online exp-weights mix)");
});

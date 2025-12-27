import assert from "node:assert/strict";
import { test } from "node:test";
import { buildOrphanedPositions, buildRequestIssueDetails } from "../.tmp/web-tests/utils.js";

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

test("buildOrphanedPositions flags market mismatch", () => {
  const positions = [{ symbol: "BTCUSDT", positionAmt: 1, positionSide: "LONG" }];
  const bots = [{ symbol: "BTCUSDT", status: { running: true, market: "spot", positions: [1] } }];
  const orphans = buildOrphanedPositions(positions, bots, { market: "futures" });
  assert.equal(orphans.length, 1);
  assert.equal(orphans[0]?.reason, "market mismatch");
});

import assert from "node:assert/strict";
import { test } from "node:test";
import {
  buildOrphanedPositions,
  buildRequestIssueDetails,
  downsampleArray,
  downsampleIndices,
  downsampleOptionalArray,
  inferFlyApiAppName,
  inferFlyDirectApiBaseFromHostname,
  isLocalHostname,
  methodLabel,
  normalizeApiBaseUrlInput,
  numFromInput,
  remapIndexToSample,
  summarizeOrderSizing,
} from "../.tmp/web-tests/utils.js";
import { defaultForm, normalizeFormState, parseDurationSeconds } from "../.tmp/web-tests/formState.js";

function assertStrictlyIncreasing(values, context) {
  for (let i = 1; i < values.length; i += 1) {
    assert.ok(
      values[i - 1] < values[i],
      `${context}: expected strictly increasing indices, got ${values[i - 1]} then ${values[i]}`,
    );
  }
}

function expectedNearestSampleIndex(indices, idx) {
  if (indices.length === 0) return 0;
  let bestIndex = 0;
  let bestDistance = Math.abs(indices[0] - idx);
  for (let i = 1; i < indices.length; i += 1) {
    const distance = Math.abs(indices[i] - idx);
    if (distance < bestDistance) {
      bestIndex = i;
      bestDistance = distance;
    }
  }
  return bestIndex;
}

function assertSampledAlignment(source, indices, sampled, context) {
  assert.equal(sampled.length, indices.length, `${context}: expected sampled length ${indices.length}, got ${sampled.length}`);
  if (source.length === 0) {
    assert.deepEqual(sampled, [], `${context}: expected empty sampled data`);
    return;
  }

  for (let sampleIdx = 0; sampleIdx < indices.length; sampleIdx += 1) {
    const rawIdx = indices[sampleIdx];
    assert.equal(
      sampled[sampleIdx],
      source[rawIdx],
      `${context}: expected sample ${sampleIdx} to align with raw index ${rawIdx}`,
    );
    assert.equal(
      remapIndexToSample(indices, rawIdx),
      sampleIdx,
      `${context}: expected remap exact hit for raw index ${rawIdx}`,
    );
  }

  if (indices.length > 0) {
    assert.equal(sampled[0], source[0], `${context}: expected first sampled point to preserve the first endpoint`);
  }
  if (indices.length > 1) {
    assert.equal(
      sampled[indices.length - 1],
      source[source.length - 1],
      `${context}: expected final sampled point to preserve the last endpoint`,
    );
  }
}

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

test("buildOrphanedPositions treats flat running bots as adopted/reconciling", () => {
  const positions = [{ symbol: "SOLUSDT", positionAmt: -2, positionSide: "BOTH" }];
  const bots = [{ symbol: "SOLUSDT", status: { running: true, market: "futures", positions: [0] } }];
  const orphans = buildOrphanedPositions(positions, bots, { market: "futures" });
  assert.equal(orphans.length, 0);
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

test("buildOrphanedPositions keeps stopped bots with unknown market in scope", () => {
  const positions = [{ symbol: "BTCUSDT", positionAmt: 1, positionSide: "LONG" }];
  const bots = [{ symbol: "BTCUSDT", status: { running: false, symbol: "BTCUSDT" } }];
  const orphans = buildOrphanedPositions(positions, bots, { market: "futures" });
  assert.equal(orphans.length, 1);
  assert.equal(orphans[0]?.reason, "bot stopped");
});

test("normalizeApiBaseUrlInput supports bare loopback IPv6 with port", () => {
  assert.equal(normalizeApiBaseUrlInput("::1:8080"), "http://[::1]:8080");
  assert.equal(normalizeApiBaseUrlInput("[::1]:8080"), "http://[::1]:8080");
});

test("normalizeApiBaseUrlInput supports localhost host+path without explicit scheme", () => {
  assert.equal(normalizeApiBaseUrlInput("localhost/api"), "http://localhost/api");
});

test("inferFlyApiAppName resolves split -web-hs app names only", () => {
  assert.equal(inferFlyApiAppName("trader-web-hs"), "trader-hs");
  assert.equal(inferFlyApiAppName("alpha-web-api-web-hs"), "alpha-web-api-hs");
  assert.equal(inferFlyApiAppName("trader-web"), "");
  assert.equal(inferFlyApiAppName("news-web-api"), "");
  assert.equal(inferFlyApiAppName("trader-web-hs2"), "");
});

test("inferFlyDirectApiBaseFromHostname infers direct fly API host for split web apps", () => {
  assert.equal(inferFlyDirectApiBaseFromHostname("trader-web-hs.fly.dev"), "https://trader-hs.fly.dev");
  assert.equal(inferFlyDirectApiBaseFromHostname("ALPHA-web-api-web-hs.fly.dev"), "https://alpha-web-api-hs.fly.dev");
  assert.equal(inferFlyDirectApiBaseFromHostname("trader-web.fly.dev"), "");
  assert.equal(inferFlyDirectApiBaseFromHostname("price-webhook.fly.dev"), "");
  assert.equal(inferFlyDirectApiBaseFromHostname("trader-web-hs2.fly.dev"), "");
  assert.equal(inferFlyDirectApiBaseFromHostname("example.com"), "");
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

test("parseDurationSeconds keeps minute and month units distinct", () => {
  assert.equal(parseDurationSeconds("1m"), 60);
  assert.equal(parseDurationSeconds("1M"), 30 * 24 * 60 * 60);
  assert.notEqual(parseDurationSeconds("1m"), parseDurationSeconds("1M"));
});

test("parseDurationSeconds rejects malformed duration strings", () => {
  for (const raw of ["", " ", "1", "m", "1mm", "1MM", "1.5h", "-1h", "1 h", "1Q"]) {
    assert.equal(parseDurationSeconds(raw), null, `expected ${JSON.stringify(raw)} to be rejected`);
  }
});

test("downsampleIndices preserves bounded chart sampling invariants", () => {
  for (let total = 0; total <= 257; total += 1) {
    for (let maxPoints = 0; maxPoints <= 65; maxPoints += 1) {
      const indices = downsampleIndices(total, maxPoints);
      const budget = Math.max(1, Math.trunc(maxPoints));

      if (total === 0) {
        assert.deepEqual(indices, []);
        continue;
      }

      assert.ok(indices.length >= 1, `expected a visible point for total=${total}, maxPoints=${maxPoints}`);
      assert.ok(
        indices.length <= Math.min(total, budget),
        `expected sample budget for total=${total}, maxPoints=${maxPoints}, got ${indices.length}`,
      );
      assert.equal(indices[0], 0, `expected first endpoint for total=${total}, maxPoints=${maxPoints}`);
      assertStrictlyIncreasing(indices, `total=${total}, maxPoints=${maxPoints}`);
      for (const idx of indices) {
        assert.ok(idx >= 0 && idx < total, `expected in-bounds sample ${idx} for total=${total}, maxPoints=${maxPoints}`);
      }
      if (indices.length > 1) {
        assert.equal(
          indices[indices.length - 1],
          total - 1,
          `expected final endpoint for total=${total}, maxPoints=${maxPoints}`,
        );
      }
      if (total <= budget) {
        assert.deepEqual(indices, Array.from({ length: total }, (_, i) => i));
      }
    }
  }
});

test("downsampleArray preserves sampled length and per-index alignment", () => {
  for (let total = 0; total <= 257; total += 1) {
    const source = Array.from({ length: total }, (_, rawIdx) => ({ rawIdx, label: `bar-${rawIdx}` }));
    for (let maxPoints = 0; maxPoints <= 65; maxPoints += 1) {
      const indices = downsampleIndices(total, maxPoints);
      const sampled = downsampleArray(source, indices);
      assertSampledAlignment(source, indices, sampled, `total=${total}, maxPoints=${maxPoints}`);
    }
  }
});

test("downsampleOptionalArray preserves aligned optional series and nullish absence", () => {
  const absenceIndices = downsampleIndices(12, 5);
  assert.equal(downsampleOptionalArray(undefined, absenceIndices), undefined);
  assert.equal(downsampleOptionalArray(null, absenceIndices), undefined);

  for (let total = 0; total <= 257; total += 1) {
    const source = Array.from({ length: total }, (_, rawIdx) =>
      rawIdx % 11 === 0 ? null : rawIdx % 7 === 0 ? undefined : `pred-${rawIdx}`
    );
    for (let maxPoints = 0; maxPoints <= 65; maxPoints += 1) {
      const indices = downsampleIndices(total, maxPoints);
      const sampled = downsampleOptionalArray(source, indices);
      assert.ok(Array.isArray(sampled), `total=${total}, maxPoints=${maxPoints}: expected sampled optional series array`);
      assertSampledAlignment(source, indices, sampled, `total=${total}, maxPoints=${maxPoints}`);
    }
  }
});

test("remapIndexToSample preserves exact sampled hits", () => {
  for (let total = 1; total <= 257; total += 1) {
    for (let maxPoints = 0; maxPoints <= 65; maxPoints += 1) {
      const indices = downsampleIndices(total, maxPoints);
      for (let sampleIdx = 0; sampleIdx < indices.length; sampleIdx += 1) {
        const rawIdx = indices[sampleIdx];
        assert.equal(
          remapIndexToSample(indices, rawIdx),
          sampleIdx,
          `expected exact sampled hit for total=${total}, maxPoints=${maxPoints}, rawIdx=${rawIdx}`,
        );
      }
    }
  }
});

test("remapIndexToSample chooses nearest visible points with deterministic left-biased ties", () => {
  assert.equal(remapIndexToSample([], 42), 0);
  assert.equal(remapIndexToSample([0, 4, 8], 2), 0);
  assert.equal(remapIndexToSample([0, 4, 8], 6), 1);

  for (let total = 1; total <= 257; total += 1) {
    for (let maxPoints = 0; maxPoints <= 65; maxPoints += 1) {
      const indices = downsampleIndices(total, maxPoints);
      for (let rawIdx = 0; rawIdx < total; rawIdx += 1) {
        const mapped = remapIndexToSample(indices, rawIdx);
        assert.ok(
          mapped >= 0 && mapped < indices.length,
          `expected mapped index in range for total=${total}, maxPoints=${maxPoints}, rawIdx=${rawIdx}`,
        );
        assert.equal(
          mapped,
          expectedNearestSampleIndex(indices, rawIdx),
          `expected nearest visible point for total=${total}, maxPoints=${maxPoints}, rawIdx=${rawIdx}`,
        );
      }
    }
  }
});

test("summarizeOrderSizing blocks trade when no effective size is configured", () => {
  const state = summarizeOrderSizing({
    orderQuantity: 0,
    orderQuote: 0,
    orderQuoteFraction: 0,
    maxOrderQuote: 0,
  });
  assert.equal(state.effective, "none");
  assert.equal(state.blockingError, "Set one sizing input: orderQuote, orderQuantity, or orderQuoteFraction.");
  assert.equal(state.blockingTargetId, "orderQuote");
  assert.equal(state.tone, "bad");
});

test("summarizeOrderSizing uses documented precedence when multiple sizing inputs are set", () => {
  const state = summarizeOrderSizing({
    orderQuantity: 0.25,
    orderQuote: 100,
    orderQuoteFraction: 0.1,
    maxOrderQuote: 50,
  });
  assert.deepEqual(state.active, ["orderQuantity", "orderQuote", "orderQuoteFraction"]);
  assert.equal(state.effective, "orderQuantity");
  assert.equal(state.conflicts, true);
  assert.equal(state.blockingError, null);
  assert.equal(state.tone, "warn");
});

test("summarizeOrderSizing keeps maxOrderQuote cap-only unless quote fraction is effective", () => {
  const cases = [
    { name: "no sizing", input: { orderQuantity: 0, orderQuote: 0, orderQuoteFraction: 0 } },
    { name: "quantity sizing", input: { orderQuantity: 0.25, orderQuote: 0, orderQuoteFraction: 0 } },
    { name: "quote sizing", input: { orderQuantity: 0, orderQuote: 100, orderQuoteFraction: 0 } },
    { name: "invalid fraction only", input: { orderQuantity: 0, orderQuote: 0, orderQuoteFraction: 1.25 } },
    { name: "quantity precedence", input: { orderQuantity: 0.25, orderQuote: 0, orderQuoteFraction: 0.1 } },
    { name: "fraction sizing", input: { orderQuantity: 0, orderQuote: 0, orderQuoteFraction: 0.1 } },
  ];

  for (const { name, input } of cases) {
    const withoutCap = summarizeOrderSizing({ ...input, maxOrderQuote: 0 });
    const withCap = summarizeOrderSizing({ ...input, maxOrderQuote: 50 });

    assert.deepEqual(withCap.active, withoutCap.active, `${name}: cap must not create an active sizing mode`);
    assert.equal(withCap.conflicts, withoutCap.conflicts, `${name}: cap must not change conflict detection`);
    assert.equal(withCap.effective, withoutCap.effective, `${name}: cap must not change effective sizing`);
    assert.equal(withCap.fractionError, withoutCap.fractionError, `${name}: cap must not change fraction validation`);
    assert.equal(withCap.blockingError, withoutCap.blockingError, `${name}: cap must not change trade readiness`);
    assert.equal(withCap.blockingTargetId, withoutCap.blockingTargetId, `${name}: cap must not change the blocking target`);
    assert.equal(withCap.statusLabel, withoutCap.statusLabel, `${name}: cap must not change status labeling`);
    assert.equal(withCap.tone, withoutCap.tone, `${name}: cap must not change severity`);

    if (withCap.effective === "orderQuoteFraction") {
      assert.notEqual(withCap.effectiveLabel, withoutCap.effectiveLabel, `${name}: active quote fraction should surface the cap label`);
      assert.notEqual(withCap.hint, withoutCap.hint, `${name}: active quote fraction should surface the cap hint`);
      assert.equal(withCap.effectiveLabel.includes("cap"), true, `${name}: capped quote fraction label should mention the cap`);
    } else {
      assert.equal(withCap.effectiveLabel, withoutCap.effectiveLabel, `${name}: cap must stay label-inert outside effective quote-fraction sizing`);
      assert.equal(withCap.hint, withoutCap.hint, `${name}: cap must stay hint-inert outside effective quote-fraction sizing`);
    }
  }
});

test("summarizeOrderSizing exhaustively preserves the modeled sizing and blocking-target contract", () => {
  const quantities = [0, 1];
  const quotes = [0, 1];
  const fractions = [-0.25, 0, 0.5, 1.25];
  const maxOrderQuotes = [0, 25];

  for (const orderQuantity of quantities) {
    for (const orderQuote of quotes) {
      for (const orderQuoteFraction of fractions) {
        for (const maxOrderQuote of maxOrderQuotes) {
          const state = summarizeOrderSizing({
            orderQuantity,
            orderQuote,
            orderQuoteFraction,
            maxOrderQuote,
          });

          const fractionError =
            orderQuoteFraction < 0
              ? "Order quote fraction must be >= 0 (use 0 to disable)."
              : orderQuoteFraction > 1
                ? "Order quote fraction must be <= 1 (use 0 to disable)."
                : null;
          const fractionOn = fractionError == null && orderQuoteFraction > 0;
          const expectedActive = [];
          if (orderQuantity > 0) expectedActive.push("orderQuantity");
          if (orderQuote > 0) expectedActive.push("orderQuote");
          if (fractionOn) expectedActive.push("orderQuoteFraction");
          const expectedEffective =
            orderQuantity > 0 ? "orderQuantity" : orderQuote > 0 ? "orderQuote" : fractionOn ? "orderQuoteFraction" : "none";
          const expectedBlocking =
            fractionError && orderQuantity <= 0 && orderQuote <= 0
              ? fractionError
              : expectedEffective === "none"
                ? "Set one sizing input: orderQuote, orderQuantity, or orderQuoteFraction."
                : null;
          const expectedBlockingTargetId = fractionError && orderQuantity <= 0 && orderQuote <= 0 ? "orderQuoteFraction" : "orderQuote";

          assert.deepEqual(state.active, expectedActive);
          assert.equal(state.effective, expectedEffective);
          assert.equal(state.conflicts, expectedActive.length > 1);
          assert.equal(state.blockingError, expectedBlocking);
          assert.equal(state.blockingTargetId, expectedBlockingTargetId);
          assert.equal(state.tone, expectedBlocking ? "bad" : expectedActive.length > 1 ? "warn" : "ok");
        }
      }
    }
  }
});

test("normalizeFormState preserves valid lookbackWindow minute and month units", () => {
  assert.equal(normalizeFormState({ lookbackWindow: "1m" }).lookbackWindow, "1m");
  assert.equal(normalizeFormState({ lookbackWindow: "1M" }).lookbackWindow, "1M");
});

test("normalizeFormState falls back to default lookbackWindow for malformed saved values", () => {
  for (const raw of [null, 123, "", " ", "1", "m", "1mm", "1.5h", "0m", "bad"]) {
    assert.equal(
      normalizeFormState({ lookbackWindow: raw }).lookbackWindow,
      defaultForm.lookbackWindow,
      `expected ${JSON.stringify(raw)} to fall back to the default lookbackWindow`,
    );
  }
});

test("normalizeFormState preserves restored whole-number counts", () => {
  const restored = normalizeFormState({
    lookbackBars: "24",
    minRoundTrips: 9,
    walkForwardFolds: "11.0",
    walkForwardEmbargoBars: "3",
  });
  assert.deepEqual(
    {
      lookbackBars: restored.lookbackBars,
      minRoundTrips: restored.minRoundTrips,
      walkForwardFolds: restored.walkForwardFolds,
      walkForwardEmbargoBars: restored.walkForwardEmbargoBars,
    },
    {
      lookbackBars: 24,
      minRoundTrips: 9,
      walkForwardFolds: 11,
      walkForwardEmbargoBars: 3,
    },
  );
});

test("normalizeFormState rejects fractional and non-finite restored whole-number counts", () => {
  const restored = normalizeFormState({
    lookbackBars: 24.5,
    minRoundTrips: "9.5",
    walkForwardFolds: Number.POSITIVE_INFINITY,
    walkForwardEmbargoBars: "NaN",
  });
  assert.deepEqual(
    {
      lookbackBars: restored.lookbackBars,
      minRoundTrips: restored.minRoundTrips,
      walkForwardFolds: restored.walkForwardFolds,
      walkForwardEmbargoBars: restored.walkForwardEmbargoBars,
    },
    {
      lookbackBars: defaultForm.lookbackBars,
      minRoundTrips: defaultForm.minRoundTrips,
      walkForwardFolds: defaultForm.walkForwardFolds,
      walkForwardEmbargoBars: defaultForm.walkForwardEmbargoBars,
    },
  );
});

test("normalizeFormState restores default minPositionSize for invalid input", () => {
  const fromInvalid = normalizeFormState({ minPositionSize: "not-a-number" });
  assert.equal(fromInvalid.minPositionSize, defaultForm.minPositionSize);
  const fromExplicitZero = normalizeFormState({ minPositionSize: 0 });
  assert.equal(fromExplicitZero.minPositionSize, 0);
});

test("normalizeFormState rehydrates manual sizing fields as finite numbers", () => {
  const restored = normalizeFormState({
    orderQuote: "125.5",
    orderQuantity: "0.25",
    orderQuoteFraction: "0.4",
    maxOrderQuote: "50",
  });
  assert.deepEqual(
    {
      orderQuote: restored.orderQuote,
      orderQuantity: restored.orderQuantity,
      orderQuoteFraction: restored.orderQuoteFraction,
      maxOrderQuote: restored.maxOrderQuote,
    },
    {
      orderQuote: 125.5,
      orderQuantity: 0.25,
      orderQuoteFraction: 0.4,
      maxOrderQuote: 50,
    },
  );
  for (const value of [restored.orderQuote, restored.orderQuantity, restored.orderQuoteFraction, restored.maxOrderQuote]) {
    assert.equal(typeof value, "number");
    assert.equal(Number.isFinite(value), true);
  }

  const fallback = normalizeFormState({
    orderQuote: "Infinity",
    orderQuantity: Number.NaN,
    orderQuoteFraction: "not-a-number",
    maxOrderQuote: "-Infinity",
  });
  assert.deepEqual(
    {
      orderQuote: fallback.orderQuote,
      orderQuantity: fallback.orderQuantity,
      orderQuoteFraction: fallback.orderQuoteFraction,
      maxOrderQuote: fallback.maxOrderQuote,
    },
    {
      orderQuote: defaultForm.orderQuote,
      orderQuantity: defaultForm.orderQuantity,
      orderQuoteFraction: defaultForm.orderQuoteFraction,
      maxOrderQuote: defaultForm.maxOrderQuote,
    },
  );
});

test("normalizeFormState keeps restored maxOrderQuote cap-only when quote fraction is inactive", () => {
  const restored = normalizeFormState({
    orderQuote: 0,
    orderQuantity: 0,
    orderQuoteFraction: 0,
    maxOrderQuote: "40",
  });
  const state = summarizeOrderSizing({
    orderQuantity: restored.orderQuantity,
    orderQuote: restored.orderQuote,
    orderQuoteFraction: restored.orderQuoteFraction,
    maxOrderQuote: restored.maxOrderQuote,
  });

  assert.equal(restored.maxOrderQuote, 40);
  assert.equal(state.effective, "none");
  assert.equal(state.blockingError, "Set one sizing input: orderQuote, orderQuantity, or orderQuoteFraction.");
  assert.equal(state.hint, "Set one sizing input: orderQuote, orderQuantity, or orderQuoteFraction.");
});

test("normalizeFormState preserves restored fraction validation and precedence", () => {
  const invalidFractionOnly = normalizeFormState({
    orderQuote: 0,
    orderQuantity: 0,
    orderQuoteFraction: "1.25",
    maxOrderQuote: "40",
  });
  const invalidFractionState = summarizeOrderSizing({
    orderQuantity: invalidFractionOnly.orderQuantity,
    orderQuote: invalidFractionOnly.orderQuote,
    orderQuoteFraction: invalidFractionOnly.orderQuoteFraction,
    maxOrderQuote: invalidFractionOnly.maxOrderQuote,
  });
  assert.equal(invalidFractionOnly.orderQuoteFraction, 1.25);
  assert.equal(invalidFractionOnly.maxOrderQuote, 40);
  assert.equal(invalidFractionState.fractionError, "Order quote fraction must be <= 1 (use 0 to disable).");
  assert.equal(invalidFractionState.blockingError, "Order quote fraction must be <= 1 (use 0 to disable).");
  assert.equal(invalidFractionState.blockingTargetId, "orderQuoteFraction");

  const precedenceRestored = normalizeFormState({
    orderQuantity: "0.25",
    orderQuote: "100",
    orderQuoteFraction: "1.25",
    maxOrderQuote: "40",
  });
  const precedenceState = summarizeOrderSizing({
    orderQuantity: precedenceRestored.orderQuantity,
    orderQuote: precedenceRestored.orderQuote,
    orderQuoteFraction: precedenceRestored.orderQuoteFraction,
    maxOrderQuote: precedenceRestored.maxOrderQuote,
  });
  assert.deepEqual(precedenceState.active, ["orderQuantity", "orderQuote"]);
  assert.equal(precedenceState.effective, "orderQuantity");
  assert.equal(precedenceState.fractionError, "Order quote fraction must be <= 1 (use 0 to disable).");
  assert.equal(precedenceState.blockingError, null);
  assert.equal(precedenceState.blockingTargetId, "orderQuote");
  assert.equal(precedenceState.tone, "warn");
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

test("normalizeFormState forces non-binance platforms into spot and preserves coinbase live mode", () => {
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
  assert.equal(out.binanceLive, true);
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
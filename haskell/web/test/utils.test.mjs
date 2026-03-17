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
function expectedOrderSizingFractionError(orderQuoteFraction) {
if (!Number.isFinite(orderQuoteFraction)) return "Order quote fraction must be a number.";
if (orderQuoteFraction < 0) return "Order quote fraction must be >= 0 (use 0 to disable).";
if (orderQuoteFraction > 1) return "Order quote fraction must be <= 1 (use 0 to disable).";
return null;
}
function expectedFmtPct(x, digits = 2) {
if (!Number.isFinite(x)) return "\u2014";
return `${(x * 100).toFixed(digits)}%`;
}
function expectedFmtNum(x, digits = 6) {
if (!Number.isFinite(x)) return "\u2014";
return x.toFixed(digits);
}
function expectedFmtMoney(x, digits = 2) {
if (!Number.isFinite(x)) return "\u2014";
return x.toFixed(digits);
}
function expectedOrderSizingModel({ orderQuantity, orderQuote, orderQuoteFraction, maxOrderQuote }) {
const quantityOn = Number.isFinite(orderQuantity) && orderQuantity > 0;
const quoteOn = Number.isFinite(orderQuote) && orderQuote > 0;
const fractionError = expectedOrderSizingFractionError(orderQuoteFraction);
const fractionOn = fractionError == null && orderQuoteFraction > 0;
const active = [];
if (quantityOn) active.push("orderQuantity");
if (quoteOn) active.push("orderQuote");
if (fractionOn) active.push("orderQuoteFraction");
const effective =
quantityOn ? "orderQuantity" : quoteOn ? "orderQuote" : fractionOn ? "orderQuoteFraction" : "none";
const conflicts = active.length > 1;
const blockingError =
fractionError && !quantityOn && !quoteOn
? fractionError
: effective === "none"
? "Set one sizing input: orderQuote, orderQuantity, or orderQuoteFraction."
: null;
const blockingTargetId = fractionError && !quantityOn && !quoteOn ? "orderQuoteFraction" : "orderQuote";
const effectiveLabel =
effective === "orderQuantity"
? `Quantity ${expectedFmtNum(orderQuantity, 8)}`
: effective === "orderQuote"
? `Quote ${expectedFmtMoney(orderQuote, 2)}`
: effective === "orderQuoteFraction"
? `Fraction ${expectedFmtPct(orderQuoteFraction, 2)}${maxOrderQuote > 0 ? ` cap ${expectedFmtMoney(maxOrderQuote, 2)}` : ""}`
: "No sizing selected";
const statusLabel =
blockingError
? "Sizing required"
: effective === "orderQuantity"
? "Using order quantity"
: effective === "orderQuote"
? "Using order quote"
: effective === "orderQuoteFraction"
? "Using quote fraction"
: "Sizing required";
const hint =
blockingError
? blockingError
: conflicts
? `${effective} takes precedence. Clear the other sizing inputs to avoid surprises.`
: `Effective sizing: ${effectiveLabel}.`;
const tone = blockingError ? "bad" : conflicts ? "warn" : "ok";
return {
active,
conflicts,
effective,
effectiveLabel,
fractionError,
blockingError,
blockingTargetId,
statusLabel,
hint,
tone,
};
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

function assertSizingStatusAndHintContract(state, context) {
  const expectedStatusLabel =
    state.blockingError
      ? "Sizing required"
      : state.effective === "orderQuantity"
        ? "Using order quantity"
        : state.effective === "orderQuote"
          ? "Using order quote"
          : state.effective === "orderQuoteFraction"
            ? "Using quote fraction"
            : "Sizing required";
  const expectedHint =
    state.blockingError
      ? state.blockingError
      : state.conflicts
        ? `${state.effective} takes precedence. Clear the other sizing inputs to avoid surprises.`
        : `Effective sizing: ${state.effectiveLabel}.`;
  assert.equal(state.statusLabel, expectedStatusLabel, `${context}: expected status label to follow the sizing contract`);
  assert.equal(state.hint, expectedHint, `${context}: expected hint to follow the sizing contract`);
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
test("buildOrphanedPositions prefers stopped unknown-market bots over other-market evidence", () => {
const stoppedStatus = { running: false, symbol: "BTCUSDT" };
const positions = [{ symbol: "BTCUSDT", positionAmt: 1, positionSide: "LONG" }];
const bots = [
{ symbol: "BTCUSDT", status: stoppedStatus },
{ symbol: "BTCUSDT", status: { running: true, market: "spot", positions: [1] } },
];
const orphans = buildOrphanedPositions(positions, bots, { market: "futures" });
assert.equal(orphans.length, 1);
assert.equal(orphans[0]?.reason, "bot stopped");
assert.equal(orphans[0]?.status, stoppedStatus);
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
test("numFromInput preserves the conservative comma-parsing contract for blank, signed, explicit, and ambiguous inputs", () => {
const cases = [
{ raw: "", fallback: 42, expected: 42, label: "empty input keeps fallback" },
{ raw: "   ", fallback: -12, expected: -12, label: "whitespace-only input keeps fallback" },
{ raw: "1,23", fallback: 0, expected: 1.23, label: "decimal comma parses" },
{ raw: "  -1,23  ", fallback: 0, expected: -1.23, label: "trimmed signed decimal comma parses" },
{ raw: " +0,125 ", fallback: 7, expected: 0.125, label: "signed leading-zero decimal comma parses" },
{ raw: "1234,567", fallback: 0, expected: 1234.567, label: "long-prefix decimal comma parses" },
{ raw: "1,234", fallback: 99, expected: 99, label: "ambiguous single-comma thousands keeps fallback" },
{ raw: "12,345", fallback: -7, expected: -7, label: "ambiguous two-digit thousands-like keeps fallback" },
{ raw: "-1,234", fallback: 5, expected: 5, label: "signed ambiguous thousands-like keeps fallback" },
{ raw: "1,234,567", fallback: 0, expected: 1234567, label: "explicit multi-group thousands parses" },
{ raw: "-1,234,567", fallback: 0, expected: -1234567, label: "signed multi-group thousands parses" },
];
for (const { raw, fallback, expected, label } of cases) {
assert.equal(
numFromInput(raw, fallback),
expected,
`${label}: expected ${expected} from ${JSON.stringify(raw)} with fallback ${fallback}`,
);
}
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
  assert.equal(state.statusLabel, "Sizing required");
  assert.equal(state.hint, "Set one sizing input: orderQuote, orderQuantity, or orderQuoteFraction.");
  assert.equal(state.tone, "bad");
  assertSizingStatusAndHintContract(state, "no effective sizing");
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
  assert.equal(state.statusLabel, "Using order quantity");
  assert.equal(state.hint, "orderQuantity takes precedence. Clear the other sizing inputs to avoid surprises.");
  assert.equal(state.tone, "warn");
  assertSizingStatusAndHintContract(state, "conflicting sizing");
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

    assertSizingStatusAndHintContract(withoutCap, `${name} without cap`);
    assertSizingStatusAndHintContract(withCap, `${name} with cap`);

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
      assert.equal(withCap.hint, `Effective sizing: ${withCap.effectiveLabel}.`, `${name}: capped quote fraction hint should mirror the effective label`);
    } else {
      assert.equal(withCap.effectiveLabel, withoutCap.effectiveLabel, `${name}: cap must stay label-inert outside effective quote-fraction sizing`);
      assert.equal(withCap.hint, withoutCap.hint, `${name}: cap must stay hint-inert outside effective quote-fraction sizing`);
    }
  }
});

test("summarizeOrderSizing exhaustively preserves the modeled sizing contract", () => {
  const quantities = [0, 1];
  const quotes = [0, 1];
  const fractions = [-0.25, 0, 0.5, 1.25];
  const maxOrderQuotes = [0, 25];

  for (const orderQuantity of quantities) {
    for (const orderQuote of quotes) {
      for (const orderQuoteFraction of fractions) {
        for (const maxOrderQuote of maxOrderQuotes) {
          const context = `orderQuantity=${orderQuantity}, orderQuote=${orderQuote}, orderQuoteFraction=${orderQuoteFraction}, maxOrderQuote=${maxOrderQuote}`;
          const state = summarizeOrderSizing({
            orderQuantity,
            orderQuote,
            orderQuoteFraction,
            maxOrderQuote,
          });
          const expected = expectedOrderSizingModel({
            orderQuantity,
            orderQuote,
            orderQuoteFraction,
            maxOrderQuote,
          });

          assert.deepEqual(state.active, expected.active, `${context}: active sizing modes should match the model`);
          assert.equal(state.effective, expected.effective, `${context}: effective sizing should match the model`);
          assert.equal(state.conflicts, expected.conflicts, `${context}: conflict flag should match the model`);
          assert.equal(state.fractionError, expected.fractionError, `${context}: fraction validation should match the model`);
          assert.equal(state.blockingError, expected.blockingError, `${context}: blocking error should match the model`);
          assert.equal(state.blockingTargetId, expected.blockingTargetId, `${context}: blocking target should match the model`);
          assert.equal(state.effectiveLabel, expected.effectiveLabel, `${context}: effective label should match the model`);
          assert.equal(state.statusLabel, expected.statusLabel, `${context}: status label should match the model`);
          assert.equal(state.hint, expected.hint, `${context}: hint should match the model`);
          assert.equal(state.tone, expected.tone, `${context}: severity should match the model`);
          assertSizingStatusAndHintContract(state, `${context}: derived copy contract`);
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
  for (const raw of [null, 123, "", " ", "1", "m", "1mm", "1.5h", "-1h", "0m", "bad"]) {
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
test("normalizeFormState rehydrates every numeric field from legacy string storage", () => {
const numericKeys = Object.entries(defaultForm)
.filter(([, value]) => typeof value === "number")
.map(([key]) => key);
const restored = normalizeFormState(
Object.fromEntries(numericKeys.map((key) => [key, String(defaultForm[key])])),
);
for (const key of numericKeys) {
assert.equal(typeof restored[key], "number", `${key} should rehydrate as a number`);
assert.equal(Number.isFinite(restored[key]), true, `${key} should rehydrate as a finite number`);
assert.equal(restored[key], defaultForm[key], `${key} should preserve its numeric value`);
}
});
test("normalizeFormState falls back to defaults for invalid restored numeric fields", () => {
const numericKeys = Object.entries(defaultForm)
.filter(([, value]) => typeof value === "number")
.map(([key]) => key);
const restored = normalizeFormState(
Object.fromEntries(numericKeys.map((key) => [key, "not-a-number"])),
);
for (const key of numericKeys) {
assert.equal(restored[key], defaultForm[key], `${key} should fall back to the default numeric value`);
}
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
test("normalizeFormState clamps restored manual sizing bounds without changing the sizing contract", () => {
const cases = [
{
label: "negative restores clamp non-fraction fields to zero and keep negative fraction finite",
raw: {
orderQuote: "-125.5",
orderQuantity: "-0.25",
orderQuoteFraction: "-0.4",
maxOrderQuote: "-50",
},
expectedFields: {
orderQuote: 0,
orderQuantity: 0,
orderQuoteFraction: -0.4,
maxOrderQuote: 0,
},
},
{
label: "oversized restores clamp non-fraction fields to the hard ceiling and keep finite fraction precedence",
raw: {
orderQuote: "1000000005",
orderQuantity: "1000000007",
orderQuoteFraction: "1.25",
maxOrderQuote: "1000000009",
},
expectedFields: {
orderQuote: 1e9,
orderQuantity: 1e9,
orderQuoteFraction: 1.25,
maxOrderQuote: 1e9,
},
},
];
for (const { label, raw, expectedFields } of cases) {
const restored = normalizeFormState(raw);
const restoredFields = {
orderQuote: restored.orderQuote,
orderQuantity: restored.orderQuantity,
orderQuoteFraction: restored.orderQuoteFraction,
maxOrderQuote: restored.maxOrderQuote,
};
assert.deepEqual(
restoredFields,
expectedFields,
label + ": restored sizing fields should clamp into supported bounds",
);
for (const [key, value] of Object.entries(restoredFields)) {
assert.equal(typeof value, "number", label + ": " + key + " should restore as a number");
assert.equal(Number.isFinite(value), true, label + ": " + key + " should restore as a finite number");
}
const state = summarizeOrderSizing({
orderQuantity: restored.orderQuantity,
orderQuote: restored.orderQuote,
orderQuoteFraction: restored.orderQuoteFraction,
maxOrderQuote: restored.maxOrderQuote,
});
const expected = expectedOrderSizingModel({
orderQuantity: expectedFields.orderQuantity,
orderQuote: expectedFields.orderQuote,
orderQuoteFraction: expectedFields.orderQuoteFraction,
maxOrderQuote: expectedFields.maxOrderQuote,
});
assert.deepEqual(state.active, expected.active, label + ": active sizing modes should match the model");
assert.equal(state.effective, expected.effective, label + ": effective sizing should match the model");
assert.equal(state.conflicts, expected.conflicts, label + ": conflict flag should match the model");
assert.equal(state.fractionError, expected.fractionError, label + ": fraction validation should match the model");
assert.equal(state.blockingError, expected.blockingError, label + ": blocking error should match the model");
assert.equal(state.blockingTargetId, expected.blockingTargetId, label + ": blocking target should match the model");
assert.equal(state.effectiveLabel, expected.effectiveLabel, label + ": effective label should match the model");
assert.equal(state.statusLabel, expected.statusLabel, label + ": status label should match the model");
assert.equal(state.hint, expected.hint, label + ": hint should match the model");
assert.equal(state.tone, expected.tone, label + ": severity should match the model");
}
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
assert.equal(state.statusLabel, "Sizing required");
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
  assert.equal(invalidFractionState.statusLabel, "Sizing required");

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
  assert.equal(precedenceState.statusLabel, "Using order quantity");
  assert.equal(precedenceState.tone, "warn");
});
test("defaultForm uses safe trade defaults", () => {
assert.equal(defaultForm.binanceLive, false);
assert.equal(defaultForm.tradeArmed, false);
assert.equal(defaultForm.botAdoptExistingPosition, true);
});
test("normalizeFormState preserves persisted restore safety invariants for live trading and bot adoption", () => {
const cases = [
{
label: "non-trading platforms keep the safe spot fallback",
raw: {
platform: "kraken",
market: "futures",
binanceLive: true,
binanceTestnet: true,
tradeArmed: true,
botAdoptExistingPosition: false,
},
expected: {
platform: "kraken",
market: "spot",
binanceLive: false,
binanceTestnet: false,
tradeArmed: false,
},
},
{
label: "coinbase restores supported live mode but remains spot-only",
raw: {
platform: "coinbase",
market: "margin",
binanceLive: true,
binanceTestnet: true,
tradeArmed: true,
botAdoptExistingPosition: false,
},
expected: {
platform: "coinbase",
market: "spot",
binanceLive: true,
binanceTestnet: false,
tradeArmed: true,
},
},
{
label: "binance margin without live falls back to spot without reviving the stale testnet toggle",
raw: {
platform: "binance",
market: "margin",
binanceLive: false,
binanceTestnet: true,
tradeArmed: true,
botAdoptExistingPosition: false,
},
expected: {
platform: "binance",
market: "spot",
binanceLive: false,
binanceTestnet: false,
tradeArmed: true,
},
},
];
for (const { label, raw, expected } of cases) {
const out = normalizeFormState(raw);
assert.equal(out.platform, expected.platform, label + ": platform should normalize deterministically");
assert.equal(out.market, expected.market, label + ": market should keep the safe restore fallback");
assert.equal(out.binanceLive, expected.binanceLive, label + ": live mode should stay inside supported restore states");
assert.equal(out.binanceTestnet, expected.binanceTestnet, label + ": testnet should stay inside supported restore states");
assert.equal(out.tradeArmed, expected.tradeArmed, label + ": trade arming should stay platform-safe");
assert.equal(out.botAdoptExistingPosition, true, label + ": existing-position adoption should always restore enabled");
}
});
test("normalizeFormState exhaustively gates restored binanceTestnet by platform and market", () => {
const platforms = ["binance", "coinbase", "kraken", "poloniex"];
const markets = ["spot", "margin", "futures"];
const bools = [false, true];
for (const platform of platforms) {
for (const market of markets) {
for (const binanceLive of bools) {
for (const binanceTestnet of bools) {
const label = `platform=${platform}, market=${market}, binanceLive=${binanceLive}, binanceTestnet=${binanceTestnet}`;
const out = normalizeFormState({
platform,
market,
binanceLive,
binanceTestnet,
botAdoptExistingPosition: false,
});
const expectedMarket =
platform !== "binance"
? "spot"
: market === "margin" && !binanceLive
? "spot"
: market;
const expectedBinanceLive =
platform === "binance"
? market === "margin" && !binanceLive
? false
: binanceLive
: platform === "coinbase"
? binanceLive
: false;
const expectedBinanceTestnet = platform === "binance" && market !== "margin" ? binanceTestnet : false;
assert.equal(out.platform, platform, label + ": platform should stay stable across restore gating");
assert.equal(out.market, expectedMarket, label + ": market should normalize safely");
assert.equal(out.binanceLive, expectedBinanceLive, label + ": live mode should normalize safely");
assert.equal(out.binanceTestnet, expectedBinanceTestnet, label + ": testnet should only restore for Binance spot/futures");
assert.equal(out.botAdoptExistingPosition, true, label + ": existing-position adoption should always restore enabled");
}
}
}
}
});
test("normalizeFormState trims and case-folds restored boolean-like strings", () => {
const out = normalizeFormState({
binanceLive: " TRUE ",
tradeArmed: " 1 ",
binanceTestnet: " False ",
optimizeOperations: " true ",
sweepThreshold: " FALSE ",
autoRefresh: " 1 ",
costAwareEdge: " 0 ",
botProtectionOrders: " TRUE ",
});
assert.equal(out.binanceLive, true);
assert.equal(out.tradeArmed, true);
assert.equal(out.binanceTestnet, false);
assert.equal(out.optimizeOperations, true);
assert.equal(out.sweepThreshold, false);
assert.equal(out.autoRefresh, true);
assert.equal(out.costAwareEdge, false);
assert.equal(out.botProtectionOrders, true);
});
test("normalizeFormState applies trimmed boolean-like strings before trading safety gates", () => {
const binanceMarginFallback = normalizeFormState({
platform: "binance",
market: "margin",
binanceLive: " FALSE ",
binanceTestnet: " TRUE ",
tradeArmed: " TRUE ",
});
assert.equal(binanceMarginFallback.market, "spot");
assert.equal(binanceMarginFallback.binanceLive, false);
assert.equal(binanceMarginFallback.binanceTestnet, false);
assert.equal(binanceMarginFallback.tradeArmed, true);
const binanceFuturesRestore = normalizeFormState({
platform: "binance",
market: "futures",
binanceLive: " TRUE ",
binanceTestnet: " TRUE ",
tradeArmed: " FALSE ",
});
assert.equal(binanceFuturesRestore.market, "futures");
assert.equal(binanceFuturesRestore.binanceLive, true);
assert.equal(binanceFuturesRestore.binanceTestnet, true);
assert.equal(binanceFuturesRestore.tradeArmed, false);
const coinbaseSpotRestore = normalizeFormState({
platform: "coinbase",
market: "margin",
binanceLive: " TRUE ",
binanceTestnet: " TRUE ",
tradeArmed: " TRUE ",
});
assert.equal(coinbaseSpotRestore.market, "spot");
assert.equal(coinbaseSpotRestore.binanceLive, true);
assert.equal(coinbaseSpotRestore.binanceTestnet, false);
assert.equal(coinbaseSpotRestore.tradeArmed, true);
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
binanceTestnet: true,
});
assert.equal(out.market, "spot");
assert.equal(out.binanceLive, false);
assert.equal(out.binanceTestnet, false);
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
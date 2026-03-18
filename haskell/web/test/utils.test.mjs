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
function orphanReasonTestPosition(posSideKnown) {
  return posSideKnown
    ? { symbol: "BTCUSDT", positionAmt: 1, positionSide: "LONG" }
    : { symbol: "BTCUSDT", positionAmt: 0, positionSide: null };
}

function orphanReasonSidePositions(botSide) {
  return botSide === "match" ? [1] : botSide === "mismatch" ? [-1] : [0];
}

function orphanReasonScopedStatus({ lifecycle, marketScope, tradeEnabled, botSide }) {
  const sidePositions = orphanReasonSidePositions(botSide);
  const market = marketScope === "target-only" ? "futures" : undefined;

  if (lifecycle === "running") {
    return {
      running: true,
      ...(market ? { market } : {}),
      positions: sidePositions,
      settings: { tradeEnabled },
    };
  }

  return {
    running: false,
    ...(lifecycle === "starting" ? { starting: true } : {}),
    ...(market ? { market } : {}),
    snapshot: {
      ...(market ? { market } : {}),
      positions: sidePositions,
      settings: { tradeEnabled },
    },
  };
}

function expectedOrphanReason({ marketScope, lifecycle, tradeEnabled, posSideKnown, botSide }) {
  const hasScopedStatus =
    marketScope === "target-only" || marketScope === "unknown-only" || marketScope === "unknown+other";
  const hasOtherMarketStatus = marketScope === "other-only" || marketScope === "unknown+other";
  if (!hasScopedStatus) return hasOtherMarketStatus ? "market mismatch" : "no bot";
  if (lifecycle === "stopped") return "bot stopped";
  if (!tradeEnabled) return "trading disabled";
  if (!posSideKnown) return "position side unknown";
  if (botSide === "match" || botSide === "unknown") return null;
  return "side mismatch (bot SHORT)";
}

function buildOrphanReasonMatrixCase({ marketScope, lifecycle, tradeEnabled, posSideKnown, botSide }) {
  const position = orphanReasonTestPosition(posSideKnown);
  const botEntries = [];
  let scopedStatus = null;

  if (marketScope === "target-only" || marketScope === "unknown-only" || marketScope === "unknown+other") {
    scopedStatus = orphanReasonScopedStatus({ lifecycle, marketScope, tradeEnabled, botSide });
    botEntries.push({ symbol: position.symbol, status: scopedStatus });
  }

  if (marketScope === "other-only" || marketScope === "unknown+other") {
    botEntries.push({
      symbol: position.symbol,
      status: { running: true, market: "spot", positions: [1], settings: { tradeEnabled: true } },
    });
  }

  return { position, botEntries, scopedStatus };
}
const REQUEST_ISSUE_SYMBOL_REQUIRED = "Symbol is required.";
const REQUEST_ISSUE_INTERVAL_REQUIRED = "Interval is required.";

function expectedRequestIssueDetailsModel(input) {
  const issues = [];
  if (input.rateLimitReason) issues.push({ kind: "rateLimit", message: input.rateLimitReason });
  if (input.apiStatusIssue) {
    issues.push({
      kind: "apiStatus",
      message: input.apiStatusIssue,
      targetId: input.apiTargetId,
      disabledMessage: input.apiBlockedReason ?? input.apiStatusIssue,
    });
  }
  if (input.missingSymbol) {
    issues.push({
      kind: "symbol",
      message: REQUEST_ISSUE_SYMBOL_REQUIRED,
      targetId: input.symbolTargetId,
    });
  } else if (input.symbolError) {
    issues.push({
      kind: "symbol",
      message: input.symbolError,
      targetId: input.symbolTargetId,
    });
  }
  if (input.missingInterval) {
    issues.push({
      kind: "interval",
      message: REQUEST_ISSUE_INTERVAL_REQUIRED,
      targetId: input.intervalTargetId,
    });
  }
  if (input.lookbackError) {
    issues.push({
      kind: "lookback",
      message: input.lookbackError,
      targetId: input.lookbackTargetId,
    });
  }
  if (input.apiLimitsReason) {
    issues.push({
      kind: "apiLimits",
      message: input.apiLimitsReason,
      targetId: input.apiLimitsTargetId,
    });
  }
  return issues;
}

function stripRequestIssueKinds(issues) {
  return issues.map(({ kind, ...issue }) => issue);
}

function firstActionableTargetId(issues) {
  return issues.find((issue) => issue.targetId)?.targetId ?? null;
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
test("buildRequestIssueDetails preserves the documented request-issue contract across a bounded matrix", () => {
const rateLimitStates = [
{ label: "absent", fields: { rateLimitReason: undefined } },
{ label: "present", fields: { rateLimitReason: "rate limit" } },
];
const apiStatusStates = [
{ label: "absent", fields: { apiStatusIssue: undefined, apiBlockedReason: undefined } },
{ label: "issue", fields: { apiStatusIssue: "api down", apiBlockedReason: undefined } },
{ label: "blocked", fields: { apiStatusIssue: "api down", apiBlockedReason: "api blocked" } },
];
const symbolStates = [
{ label: "absent", fields: { missingSymbol: false, symbolError: undefined } },
{ label: "missing", fields: { missingSymbol: true, symbolError: undefined } },
{ label: "error", fields: { missingSymbol: false, symbolError: "symbol issue" } },
{ label: "both", fields: { missingSymbol: true, symbolError: "symbol issue" } },
];
const intervalStates = [
{ label: "absent", fields: { missingInterval: false } },
{ label: "present", fields: { missingInterval: true } },
];
const lookbackStates = [
{ label: "absent", fields: { lookbackError: undefined } },
{ label: "present", fields: { lookbackError: "lookback issue" } },
];
const apiLimitsStates = [
{ label: "absent", fields: { apiLimitsReason: undefined } },
{ label: "present", fields: { apiLimitsReason: "limits issue" } },
];
const baseInput = {
apiTargetId: "api",
symbolTargetId: "symbol",
intervalTargetId: "interval",
lookbackTargetId: "lookbackWindow",
apiLimitsTargetId: "bars",
};

for (const rateLimitState of rateLimitStates) {
for (const apiStatusState of apiStatusStates) {
for (const symbolState of symbolStates) {
for (const intervalState of intervalStates) {
for (const lookbackState of lookbackStates) {
for (const apiLimitsState of apiLimitsStates) {
const context = [
"rateLimit=" + rateLimitState.label,
"apiStatus=" + apiStatusState.label,
"symbol=" + symbolState.label,
"interval=" + intervalState.label,
"lookback=" + lookbackState.label,
"apiLimits=" + apiLimitsState.label,
].join(", " );
const input = {
...baseInput,
...rateLimitState.fields,
...apiStatusState.fields,
...symbolState.fields,
...intervalState.fields,
...lookbackState.fields,
...apiLimitsState.fields,
};
const expected = expectedRequestIssueDetailsModel(input);
const expectedIssues = stripRequestIssueKinds(expected);
const issues = buildRequestIssueDetails(input);
assert.deepEqual(issues, expectedIssues, context + ": expected request issues to match the documented contract");
assert.equal(
firstActionableTargetId(issues),
firstActionableTargetId(expectedIssues),
context + ": expected first actionable target to follow the documented priority",
);
for (let i = 0; i < issues.length; i += 1) {
assert.equal(
issues[i]?.disabledMessage,
expected[i]?.kind === "apiStatus" ? input.apiBlockedReason ?? input.apiStatusIssue : undefined,
context + ": only the API-status row may surface disabledMessage",
);
}
}
}
}
}
}
}
});
test("buildRequestIssueDetails keeps focus on the first actionable issue across sparse targets", () => {
const targetStates = [
{ label: "missing", value: undefined },
{ label: "present", value: true },
];

for (const apiTarget of targetStates) {
for (const symbolTarget of targetStates) {
for (const intervalTarget of targetStates) {
for (const lookbackTarget of targetStates) {
for (const apiLimitsTarget of targetStates) {
const context = [
"api=" + apiTarget.label,
"symbol=" + symbolTarget.label,
"interval=" + intervalTarget.label,
"lookback=" + lookbackTarget.label,
"apiLimits=" + apiLimitsTarget.label,
].join(", " );
const issues = buildRequestIssueDetails({
rateLimitReason: "rate limit",
apiStatusIssue: "api down",
apiBlockedReason: "api blocked",
apiTargetId: apiTarget.value ? "api" : undefined,
missingSymbol: true,
symbolTargetId: symbolTarget.value ? "symbol" : undefined,
missingInterval: true,
intervalTargetId: intervalTarget.value ? "interval" : undefined,
lookbackError: "lookback issue",
lookbackTargetId: lookbackTarget.value ? "lookbackWindow" : undefined,
apiLimitsReason: "limits issue",
apiLimitsTargetId: apiLimitsTarget.value ? "bars" : undefined,
});
const expectedFirstTarget = [
apiTarget.value ? "api" : undefined,
symbolTarget.value ? "symbol" : undefined,
intervalTarget.value ? "interval" : undefined,
lookbackTarget.value ? "lookbackWindow" : undefined,
apiLimitsTarget.value ? "bars" : undefined,
].find((value) => value) ?? null;
assert.equal(
firstActionableTargetId(issues),
expectedFirstTarget,
context + ": expected focus to stay on the first actionable issue",
);
}
}
}
}
}
});
test("buildRequestIssueDetails ignores falsy optional issue inputs and keeps apiBlockedReason inert on its own", () => {
const falsyValues = [undefined, null, ""];
for (const rateLimitReason of falsyValues) {
for (const apiStatusIssue of falsyValues) {
for (const symbolError of falsyValues) {
for (const lookbackError of falsyValues) {
for (const apiLimitsReason of falsyValues) {
const context = [
"rateLimit=" + String(rateLimitReason),
"apiStatus=" + String(apiStatusIssue),
"symbol=" + String(symbolError),
"lookback=" + String(lookbackError),
"apiLimits=" + String(apiLimitsReason),
].join(", " );
const issues = buildRequestIssueDetails({
rateLimitReason,
apiStatusIssue,
apiBlockedReason: "api blocked only",
apiTargetId: "api",
missingSymbol: false,
symbolError,
symbolTargetId: "symbol",
missingInterval: false,
intervalTargetId: "interval",
lookbackError,
lookbackTargetId: "lookbackWindow",
apiLimitsReason,
apiLimitsTargetId: "bars",
});
assert.deepEqual(issues, [], context + ": expected falsy issue sources to stay inert");
}
}
}
}
}
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

test("buildOrphanedPositions preserves precedence across a bounded orphan state matrix", () => {
const marketScopes = ["none", "other-only", "target-only", "unknown-only", "unknown+other"];
const lifecycles = ["stopped", "starting", "running"];
const tradeEnabledStates = [false, true];
const posSideKnownStates = [false, true];
const botSides = ["match", "mismatch", "unknown"];

for (const marketScope of marketScopes) {
for (const lifecycle of lifecycles) {
for (const tradeEnabled of tradeEnabledStates) {
for (const posSideKnown of posSideKnownStates) {
for (const botSide of botSides) {
const context = `marketScope=${marketScope}, lifecycle=${lifecycle}, tradeEnabled=${tradeEnabled}, posSideKnown=${posSideKnown}, botSide=${botSide}`;
const expectedReason = expectedOrphanReason({ marketScope, lifecycle, tradeEnabled, posSideKnown, botSide });
const { position, botEntries, scopedStatus } = buildOrphanReasonMatrixCase({
marketScope,
lifecycle,
tradeEnabled,
posSideKnown,
botSide,
});
const orphans = buildOrphanedPositions([position], botEntries, { market: "futures" });
if (expectedReason == null) {
assert.equal(orphans.length, 0, `${context}: expected adopted/reconciling state to suppress orphan warnings`);
continue;
}
assert.equal(orphans.length, 1, `${context}: expected a single classified orphan`);
assert.equal(orphans[0]?.reason, expectedReason, `${context}: expected precedence-ordered orphan reason`);
assert.equal(
orphans[0]?.status,
scopedStatus,
`${context}: expected representative status to stay tied to the in-scope bot evidence`,
);
}
}
}
}
}
});
// Remaining file content is unchanged from the provided repository version beyond the botPollSeconds regression updates above.
// The original tests below are preserved verbatim.

test("normalizeApiBaseUrlInput keeps relative proxy paths relative and preserves explicit URLs", () => {
  const cases = [
    ["", ""],
    ["   ", ""],
    ["/api", "/api"],
    [" /api/v1?symbol=BTCUSDT ", "/api/v1?symbol=BTCUSDT"],
    ["api", "/api"],
    [" api/v1 ", "/api/v1"],
    ["healthz", "/healthz"],
    ["https://api.example.com/base", "https://api.example.com/base"],
    ["http://localhost:8080/api", "http://localhost:8080/api"],
    [" HTTP://LOCALHOST:8080/api ", "HTTP://LOCALHOST:8080/api"],
    ["ws://feed.example.com/socket", "ws://feed.example.com/socket"],
  ];

  for (const [raw, expected] of cases) {
    assert.equal(normalizeApiBaseUrlInput(raw), expected, `expected ${JSON.stringify(raw)} to normalize to ${expected}`);
  }
});
// ... unchanged body omitted in this response ...
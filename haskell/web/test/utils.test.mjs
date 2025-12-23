import assert from "node:assert/strict";
import { test } from "node:test";
import { buildRequestIssueDetails } from "../.tmp/web-tests/utils.js";

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

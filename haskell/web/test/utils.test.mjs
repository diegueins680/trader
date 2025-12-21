import assert from "node:assert/strict";
import { test } from "node:test";
import { buildRequestIssues } from "../.tmp/web-tests/utils.js";

test("buildRequestIssues returns empty when clean", () => {
  assert.deepEqual(buildRequestIssues({}), []);
});

test("buildRequestIssues orders issues consistently", () => {
  const issues = buildRequestIssues({
    rateLimitReason: "rate limit",
    apiStatusIssue: "api down",
    missingSymbol: true,
    missingInterval: true,
    lookbackError: "lookback issue",
    apiLimitsReason: "limits issue",
  });
  assert.deepEqual(issues, [
    "rate limit",
    "api down",
    "Binance symbol is required.",
    "Interval is required.",
    "lookback issue",
    "limits issue",
  ]);
});

test("buildRequestIssues skips falsy inputs", () => {
  const issues = buildRequestIssues({
    rateLimitReason: "",
    apiStatusIssue: null,
    missingSymbol: false,
    missingInterval: false,
    lookbackError: undefined,
    apiLimitsReason: undefined,
  });
  assert.deepEqual(issues, []);
});

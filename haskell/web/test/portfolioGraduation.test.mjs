import test from "node:test";
import assert from "node:assert/strict";
import { renderToStaticMarkup } from "react-dom/server";
import { PortfolioGraduationStatus } from "../.tmp/web-tests/PortfolioGraduationStatus.js";

const A = "11111111-1111-1111-1111-111111111111";
const B = "22222222-2222-2222-2222-222222222222";
function fixture() {
  return { mode: "shadow", selectionValid: false, configuration: { maxBots: 5, maxGrossWeight: 0.05 },
    graduation: { enabled: true, startedAtMs: 1787809701000, earliestReviewAtMs: 1788739200000,
      reviewedUuids: [A, B], configVersion: "current", requirements: { minimumDailyObservations: 10 },
      review: { reviewedUuids: [B, A], reviewedAtMs: Date.now() - 1000, configVersion: "current", decision: "pending",
        reasons: ["daily-observations-below-minimum", "graduation-evidence-unavailable"], evidence: { dailyObservationCount: 3 } } } };
}
const render = (selector) => renderToStaticMarkup(PortfolioGraduationStatus({ selector }));

test("fleet review exposes evidence progress, baseline blockers and the enforcing fleet cap", () => {
  const html = render(fixture());
  assert.match(html, /3\/10 complete UTC days/);
  assert.match(html, /boundary baselines/);
  assert.match(html, /5 bots and 5\.0% gross weight/);
  assert.match(html, /2 reviewed strategies/);
  assert.match(html, /2026-08-27 05:48:21 UTC/);
});

test("stale strategy sets and configurations cannot display a saved graduated review as current", () => {
  for (const mutate of [
    (s) => { s.graduation.review.reviewedUuids = [A]; },
    (s) => { s.graduation.review.configVersion = "old"; },
    (s) => { s.graduation.review.reviewedUuids = null; },
  ]) {
    const selector = fixture(); selector.mode = "enforce"; selector.graduation.review.decision = "graduated";
    mutate(selector);
    const html = render(selector);
    assert.doesNotMatch(html, />Graduated/);
    assert.doesNotMatch(html, /3\/10/);
    assert.match(html, /does not match/);
  }
});

test("graduated display requires the active mode, matching metadata, empty blockers and a valid review time", () => {
  const selector = fixture();
  selector.graduation.review.decision = "graduated"; selector.graduation.review.reasons = [];
  assert.doesNotMatch(render(selector), />Graduated/);
  selector.mode = "enforce";
  assert.match(render(selector), />Graduated/);
  selector.graduation.review.reviewedAtMs = NaN;
  assert.doesNotMatch(render(selector), />Graduated/);
});

test("missing and malformed supplemental review details render conservatively without echoing arbitrary reason text", () => {
  assert.match(render({}), /details unavailable/);
  const selector = fixture();
  selector.graduation.review.reasons = ["secret-canary", "__proto__"];
  selector.graduation.review.evidence.dailyObservationCount = NaN;
  const html = render(selector);
  assert.doesNotMatch(html, /secret-canary|NaN|__proto__/);
  assert.match(html, /Another review requirement/);
  assert.match(html, /Daily evidence count unavailable/);
});

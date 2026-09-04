import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const registrationUrls = [
  new URL("../research-notes/registrations/har-rv-risk-gate-v1.json", import.meta.url),
  new URL("../research-notes/registrations/depth-normalized-ofi-v1.json", import.meta.url),
  new URL("../research-notes/registrations/missingness-aware-calibrated-shallow-v1.json", import.meta.url),
];

async function readJson(url) {
  return JSON.parse(await readFile(url, "utf8"));
}

test("market-prediction registrations remain future-only disabled challengers", async () => {
  const registrations = await Promise.all(registrationUrls.map(readJson));
  assert.deepEqual(
    registrations.map((registration) => registration.campaign),
    ["har_rv_risk_gate_v1", "depth_normalized_ofi_v1", "missingness_aware_calibrated_shallow_v1"],
  );

  for (const registration of registrations) {
    assert.match(registration.status, /^preregistered_(waiting|blocked)_/);
    assert.deepEqual(registration.horizonBars, [1, 3, 6]);
    assert.equal(registration.dataset.startInclusiveUtc, "2027-01-21T00:00:00Z");
    assert.ok(Date.parse(registration.dataset.developmentEndInclusiveUtc) < Date.parse(registration.dataset.finalHoldoutStartInclusiveUtc));
    assert.ok(Date.parse(registration.dataset.finalHoldoutStartInclusiveUtc) <= Date.parse(registration.dataset.finalHoldoutEndInclusiveUtc));
    assert.equal(registration.promotionGates.disabledChallengerOnly, true);
    assert.equal(registration.promotionGates.automaticPromotion, false);
    assert.equal(registration.promotionGates.liveAuthorization, false);
    assert.equal(registration.computeBudget.noGpu, true);
    assert.ok(registration.hyperparameterSearch.totalExperimentBudget > 0);
    assert.ok(registration.failureConditions.length > 0);
  }
});

test("market-prediction audit preserves sealed evidence boundaries", async () => {
  const existingCarry = await readJson(new URL("../research-notes/registrations/cross-sectional-funding-carry-v1.json", import.meta.url));
  assert.equal(existingCarry.campaign, "cross_sectional_funding_carry_v1");
  assert.equal(existingCarry.prospectiveData.evaluationStartUtc, "2026-07-17T00:00:00Z");
  assert.equal(existingCarry.prospectiveData.minimumEvaluationTimeUtc, "2027-01-20T13:00:00Z");
  assert.equal(existingCarry.prospectiveData.returnRows, 4500);
  assert.equal(existingCarry.prospectiveEvaluationPolicy.oneShot, true);

  const registrations = await Promise.all(registrationUrls.map(readJson));
  for (const registration of registrations) {
    assert.ok(Date.parse(registration.dataset.startInclusiveUtc) > Date.parse(existingCarry.prospectiveData.minimumEvaluationTimeUtc));
  }
});

test("paper matrix has the complete structured review schema", async () => {
  const source = await readFile(new URL("../research-notes/market-prediction-2026-09-04/paper-matrix.csv", import.meta.url), "utf8");
  const lines = source.trimEnd().split("\n");
  assert.equal(lines.length, 51);
  for (const field of [
    "canonical_identifier",
    "prediction_target",
    "split_validation_design",
    "leakage_controls",
    "cost_assumptions",
    "independent_replication_evidence",
    "code_license",
    "data_license",
    "implementation_suitability_score",
    "final_disposition",
    "original_concise_summary",
  ]) {
    assert.match(lines[0], new RegExp(`"${field}"`));
  }
});

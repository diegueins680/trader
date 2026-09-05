import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const registrationUrls = [
  new URL("../research-notes/registrations/har-rv-risk-gate-v1.json", import.meta.url),
  new URL("../research-notes/registrations/depth-normalized-ofi-v1.json", import.meta.url),
  new URL("../research-notes/registrations/missingness-aware-calibrated-shallow-v1.json", import.meta.url),
];

const derivativesReceiptUrl = new URL(
  "../research-notes/market-prediction-2026-09-04/receipts/binance-derivatives-main-2026-09-05T040706Z.json",
  import.meta.url,
);

const rateLimitCircuitReceiptUrl = new URL(
  "../research-notes/market-prediction-2026-09-04/receipts/binance-rate-limit-circuit-2026-09-05T061022Z.json",
  import.meta.url,
);

const requiredRegistrationKeys = [
  "ablations",
  "academicOrigin",
  "assumedExecutionTimestamp",
  "baselines",
  "campaign",
  "computeBudget",
  "dataset",
  "decisionTimestamp",
  "deviationsFromPapers",
  "economicMetrics",
  "failureConditions",
  "featureAvailabilityRules",
  "forecastMetrics",
  "horizonBars",
  "hyperparameterSearch",
  "hypothesis",
  "implementationInterpretation",
  "promotionGates",
  "randomSeeds",
  "registeredOn",
  "registrationVersion",
  "robustnessTests",
  "statisticalTests",
  "status",
  "timeframes",
  "transactionCostModel",
  "universe",
  "validation",
];

async function readJson(url) {
  return JSON.parse(await readFile(url, "utf8"));
}

function assertNonEmptyArray(value, label) {
  assert.ok(Array.isArray(value) && value.length > 0, `${label} must be a non-empty array`);
}

function parseCsvRow(source) {
  const fields = [];
  let current = "";
  let quoted = false;

  for (let index = 0; index < source.length; index += 1) {
    const character = source[index];
    if (character === '"') {
      if (quoted && source[index + 1] === '"') {
        current += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (character === "," && !quoted) {
      fields.push(current);
      current = "";
    } else {
      current += character;
    }
  }

  assert.equal(quoted, false, "CSV row must close every quote");
  fields.push(current);
  return fields;
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

test("market-prediction registrations retain the complete frozen protocol", async () => {
  const registrations = await Promise.all(registrationUrls.map(readJson));

  for (const registration of registrations) {
    assert.deepEqual(Object.keys(registration).sort(), requiredRegistrationKeys);
    assert.equal(registration.registrationVersion, 1);
    assert.match(registration.registeredOn, /^\d{4}-\d{2}-\d{2}$/);
    assert.ok(registration.hypothesis.length > 40);
    assert.ok(Object.keys(registration.implementationInterpretation).length > 0);
    assertNonEmptyArray(registration.academicOrigin, "academicOrigin");
    assert.ok(registration.academicOrigin.every((source) => URL.canParse(source)));
    for (const field of [
      "deviationsFromPapers",
      "timeframes",
      "featureAvailabilityRules",
      "baselines",
      "randomSeeds",
      "forecastMetrics",
      "economicMetrics",
      "statisticalTests",
      "robustnessTests",
      "ablations",
      "failureConditions",
    ]) {
      assertNonEmptyArray(registration[field], field);
    }

    assert.ok(registration.universe.primaryBenchmarks.includes("BTCUSDT"));
    assert.ok(registration.universe.primaryBenchmarks.includes("ETHUSDT"));
    assert.ok(registration.validation.outerFolds >= 2);
    assert.ok(registration.validation.innerFolds >= 2);
    assert.ok(registration.validation.purgeBars);
    assert.ok(registration.validation.embargoBars >= Math.max(...registration.horizonBars));
    assert.match(registration.validation.finalHoldoutPolicy, /(once|single|one irreversible)/i);

    const search = registration.hyperparameterSearch;
    assert.equal(search.registeredCandidateConfigurations + search.registeredBaselineConfigurations, search.totalExperimentBudget);
    assert.ok(registration.randomSeeds.every((seed) => Number.isSafeInteger(seed)));
    assert.equal(new Set(registration.randomSeeds).size, registration.randomSeeds.length);

    const costContract = JSON.stringify(registration.transactionCostModel);
    assert.match(costContract, /funding/i);
    assert.match(costContract, /(delay|latency)/i);
    assert.ok(costContract.includes("1.5"));
    assert.ok(costContract.includes("2"));

    assert.equal(registration.promotionGates.minimumDeflatedSharpeProbability, 0.95);
    assert.equal(registration.promotionGates.maximumPbo, 0.2);
    assert.equal(registration.promotionGates.disabledChallengerOnly, true);
    assert.equal(registration.promotionGates.automaticPromotion, false);
    assert.equal(registration.promotionGates.liveAuthorization, false);
    assert.equal(registration.computeBudget.noGpu, true);
    assert.ok(registration.computeBudget.inferenceP99MillisecondsPerSymbol > 0);
    assert.ok(registration.computeBudget.residentMemoryMiBPerConcurrentSymbol > 0);
    assert.ok(registration.computeBudget.startupReloadMillisecondsPerArtifact > 0);
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

test("derivatives collection receipt binds metadata without opening outcomes", async () => {
  const receipt = await readJson(derivativesReceiptUrl);
  const symbols = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "LTCUSDT",
  ];
  const sha256Pattern = /^[0-9a-f]{64}$/;

  assert.equal(receipt.schemaVersion, 1);
  assert.equal(receipt.receiptType, "metadata_only_collection_artifact_receipt");
  assert.equal(receipt.collection.statusSchemaVersion, 3);
  assert.equal(receipt.collection.artifactSchema, "binance_derivatives_collection_artifacts_v3");
  assert.equal(receipt.collection.derivativesObservationSchema, "binance_derivatives_first_seen_v2");
  assert.equal(receipt.collection.featureAvailabilitySchema, "feature_availability_v2");
  assert.equal(receipt.collection.interval, "1h");
  assert.equal(receipt.collection.state, "pass");
  assert.deepEqual(receipt.collection.failedSymbols, []);
  assert.deepEqual(receipt.collection.provenanceIssues, []);
  assert.equal(receipt.collection.provenanceTrackedClean, true);
  assert.match(receipt.collection.codeCommit, /^[0-9a-f]{40}$/);
  assert.ok(Date.parse(receipt.collection.startedAt) < Date.parse(receipt.collection.finishedAt));
  assert.equal(receipt.source.access, "public_read_only");
  assert.equal(receipt.source.licenseManifest, "research-notes/market-prediction-2026-09-04/data-source-license-manifest.json");
  assert.match(receipt.status.sha256, sha256Pattern);
  assert.deepEqual(receipt.universe.symbols, symbols);
  assert.deepEqual(Object.keys(receipt.artifacts).sort(), [...symbols].sort());

  const logicalPaths = new Set();
  let artifactCount = 0;
  for (const symbol of symbols) {
    const artifact = receipt.artifacts[symbol];
    assert.deepEqual(Object.keys(artifact.observations).sort(), ["basis", "funding", "oi", "taker"]);
    assert.equal(artifact.cache.logicalPath, `${symbol}_1h.csv`);
    for (const item of [artifact.cache, ...Object.values(artifact.observations)]) {
      assert.ok(Number.isSafeInteger(item.rows) && item.rows > 0);
      assert.match(item.sha256, sha256Pattern);
      assert.equal(item.logicalPath.startsWith("/"), false);
      assert.equal(item.logicalPath.includes(".."), false);
      assert.equal(logicalPaths.has(item.logicalPath), false);
      logicalPaths.add(item.logicalPath);
      artifactCount += 1;
    }
  }
  assert.equal(artifactCount, 50);
  assert.equal(receipt.verification.archiveFileCount, artifactCount + 1);
  assert.equal(receipt.verification.result, "verified_in_place_and_relocated");
  assert.deepEqual(receipt.outcomeBoundary, {
    admission: "acquisition_metadata_only",
    returnsComputed: false,
    ranksComputed: false,
    weightsComputed: false,
    pnlComputed: false,
    riskMetricsComputed: false,
    forecastMetricsComputed: false,
    economicMetricsComputed: false,
    holdoutsOpened: 0,
    ordersPlaced: 0,
    modelInputsChanged: false,
    liveAuthorizationChanged: false,
  });
});

test("rate-limit circuit receipt preserves sanitized operational evidence only", async () => {
  const receiptText = await readFile(rateLimitCircuitReceiptUrl, "utf8");
  const receipt = await readJson(rateLimitCircuitReceiptUrl);
  const symbols = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "LTCUSDT",
  ];

  assert.equal(receipt.schemaVersion, 1);
  assert.equal(receipt.receiptType, "metadata_only_operational_circuit_receipt");
  assert.equal(receipt.collection.statusSchemaVersion, 3);
  assert.equal(receipt.collection.state, "partial_failure");
  assert.equal(receipt.collection.provenanceTrackedClean, true);
  assert.deepEqual(receipt.collection.provenanceIssues, []);
  assert.ok(Date.parse(receipt.collection.startedAt) < Date.parse(receipt.collection.finishedAt));
  assert.equal(receipt.collection.durationSeconds, 21.761);
  for (const field of ["codeCommit", "mergedHeadCommit", "mergeCommit"]) {
    assert.match(receipt.collection[field], /^[0-9a-f]{40}$/);
  }
  assert.equal(receipt.collection.mergedPullRequest, 218);
  assert.equal(receipt.source.access, "public_read_only");
  assert.match(receipt.status.sha256, /^[0-9a-f]{64}$/);
  assert.equal(receipt.status.bytes, 7743);
  assert.deepEqual(receipt.universe.symbols, symbols);
  assert.deepEqual(receipt.circuitObservation.successfulBeforeThrottle, ["BTCUSDT"]);
  assert.equal(receipt.circuitObservation.throttleSymbol, "ETHUSDT");
  assert.equal(receipt.circuitObservation.failureKind, "provider_rate_limit");
  assert.equal(receipt.circuitObservation.httpStatus, 429);
  assert.equal(receipt.circuitObservation.bannedUntilMs, null);
  assert.equal(receipt.circuitObservation.retryAfterSeconds, null);
  assert.deepEqual(receipt.circuitObservation.remainingSymbols, symbols.slice(2));
  assert.equal(receipt.circuitObservation.remainingStatus, "skipped");
  assert.equal(receipt.circuitObservation.remainingFailureKind, "provider_rate_limit_circuit_open");
  assert.equal(receipt.circuitObservation.providerErrorMessageRetained, false);
  assert.equal(receipt.circuitObservation.providerReturnedIpRetained, false);
  assert.equal(receipt.circuitObservation.result, "circuit_observed");
  assert.doesNotMatch(receiptText, /\b(?:\d{1,3}\.){3}\d{1,3}\b/);
  assert.doesNotMatch(receiptText, /"(?:errorMessage|rawError|providerMessage)"/);
  assert.deepEqual(receipt.evidenceBoundary, {
    rawStatusCommitted: false,
    artifactAdmission: false,
    outcomeAdmission: "acquisition_metadata_only",
    returnsComputed: false,
    ranksComputed: false,
    weightsComputed: false,
    pnlComputed: false,
    riskMetricsComputed: false,
    forecastMetricsComputed: false,
    economicMetricsComputed: false,
    holdoutsOpened: 0,
    ordersPlaced: 0,
    modelInputsChanged: false,
    liveAuthorizationChanged: false,
  });
});

test("experiment registry accounts for every trial and matches new budgets", async () => {
  const registry = await readJson(new URL("../research-notes/market-prediction-2026-09-04/experiment-registry.json", import.meta.url));
  const registrations = await Promise.all(registrationUrls.map(readJson));
  const campaigns = registry.relatedResidualFundingFamily.campaigns;
  const priorCampaigns = campaigns.filter((campaign) => campaign.campaign !== "cross_sectional_funding_carry_v1");
  assert.equal(priorCampaigns.reduce((total, campaign) => total + campaign.count, 0), 45);
  assert.equal(registry.relatedResidualFundingFamily.executedOrPlannedLifetimeCountBeforeProspectiveCarry, 45);

  for (const campaign of campaigns) {
    const [first, last] = campaign.lifetimeTrialNumbers;
    assert.equal(last - first + 1, campaign.count);
    const namedTrials = campaign.trialIds?.length ?? campaign.primaryTrialIds?.length ?? 0;
    assert.equal(namedTrials + (campaign.derivedRegisteredPhaseVariants ?? 0), campaign.count);
    assert.match(campaign.holdout ?? "not applicable", /(unopened|not applicable)/);
  }

  const prospectiveCarry = campaigns.find((campaign) => campaign.campaign === "cross_sectional_funding_carry_v1");
  assert.deepEqual(prospectiveCarry.lifetimeTrialNumbers, [46, 46]);
  assert.equal(prospectiveCarry.performanceAvailable, false);
  assert.equal(registry.auditRun.performanceTrials, 0);
  assert.equal(registry.auditRun.holdoutsOpened, 0);
  assert.equal(registry.auditRun.ordersPlaced, 0);

  for (const registration of registrations) {
    const registryEntry = registry.newIndependentProspectiveFamilies.find((entry) => entry.campaign === registration.campaign);
    assert.ok(registryEntry, `missing registry entry for ${registration.campaign}`);
    assert.equal(registryEntry.registeredCandidateConfigurations, registration.hyperparameterSearch.registeredCandidateConfigurations);
    assert.equal(registryEntry.registeredBaselineConfigurations, registration.hyperparameterSearch.registeredBaselineConfigurations);
    assert.equal(registryEntry.totalBudget, registration.hyperparameterSearch.totalExperimentBudget);
    assert.equal(registryEntry.executedConfigurations, 0);
  }
});

test("paper matrix has the complete structured review schema", async () => {
  const source = await readFile(new URL("../research-notes/market-prediction-2026-09-04/paper-matrix.csv", import.meta.url), "utf8");
  const lines = source.trimEnd().split("\n");
  assert.equal(lines.length, 51);
  const rows = lines.map(parseCsvRow);
  const header = rows[0];
  assert.equal(header.length, 27);
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
    assert.ok(header.includes(field));
  }

  const fieldIndex = Object.fromEntries(header.map((field, index) => [field, index]));
  const identifiers = new Set();
  for (const row of rows.slice(1)) {
    assert.equal(row.length, header.length);
    assert.ok(row[fieldIndex.title].length > 0);
    assert.ok(URL.canParse(row[fieldIndex.canonical_identifier]));
    assert.equal(identifiers.has(row[fieldIndex.canonical_identifier]), false);
    identifiers.add(row[fieldIndex.canonical_identifier]);
    assert.ok(Number.isInteger(Number(row[fieldIndex.year])));
    assert.ok(Number(row[fieldIndex.year]) <= 2026);
    const score = Number(row[fieldIndex.implementation_suitability_score]);
    assert.ok(Number.isInteger(score) && score >= 0 && score <= 100);
    assert.ok(["reject", "monitor", "prototype", "integrate"].includes(row[fieldIndex.final_disposition]));
    assert.ok(row[fieldIndex.original_concise_summary].length > 20);
  }
});

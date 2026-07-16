import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

const RESEARCH_DIR = fileURLToPath(
  new URL("../scripts/research/", import.meta.url),
);
const hasResearchPython =
  spawnSync("python3", ["-c", "import numpy, pandas"], { encoding: "utf8" })
    .status === 0;

test(
  "risk-controlled v2 pins lower gross, stable tie order, and adaptive lineage",
  { skip: !hasResearchPython },
  () => {
    const program = String.raw`
import copy
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1])
import campaign_runner as common
import historical_datafeed as feed
import run_historical_risk_controlled_reversal_campaign_v2 as runner

registration = runner._registration()
assert registration["campaign"] == runner.CAMPAIGN_ID
assert runner.REGISTRATION_SHA256 == common._file_digest(runner.REGISTRATION_PATH)
assert registration["strategy"]["grossExposure"] == 0.25
assert registration["strategy"]["longTargetWeight"] == 0.125
assert registration["strategy"]["shortTargetWeight"] == -0.125
assert registration["validation"]["priorTrialCount"] == 39
assert registration["validation"]["newTrialCount"] == 6
assert registration["validation"]["lifetimeTrialCount"] == 45
v2_config = runner._strategy_config(registration)
assert v2_config.gross_exposure == 0.25
assert v2_config.registered_gross_exposure == 0.25
assert runner.HOLDOUT_REGISTRY_DIR == common.SHARED_HOLDOUT_REGISTRY_DIR
assert runner.REGISTERED_HOLDOUT_REGISTRY_DIR == (
    common.CANONICAL_SHARED_HOLDOUT_REGISTRY_DIR
)
runner._validate_registry_policy(registration["holdoutPolicy"])
assert runner._validate_runtime_paths(runner.parse_args([])).is_absolute()
protected_args = runner.parse_args([])
protected_output = (
    common.SHARED_REPOSITORY_ROOT
    / runner.ADAPTIVE_PREDECESSOR_CAMPAIGN_DIRECTORY
    / "must-not-be-created"
)
protected_args.output_dir = str(protected_output)
try:
    runner._validate_runtime_paths(protected_args)
except ValueError as error:
    assert "adaptive predecessor" in str(error)
else:
    raise AssertionError("output paths may not alias immutable v1 evidence")
assert not protected_output.exists()

with tempfile.TemporaryDirectory() as invocation_temporary:
    invocation_output = Path(invocation_temporary)
    terminal_schema_output = invocation_output / "terminal-schema"
    terminal_schema_output.mkdir()
    terminal_index = {
        "campaign": runner.CAMPAIGN_ID,
        "status": "risk_invalid",
        "registrationSha256": runner.REGISTRATION_SHA256,
        "campaignManifestSha256": "1" * 64,
        "artifacts": {},
    }
    terminal_index_path = terminal_schema_output / "evidence-index.json"
    terminal_index_path.write_text(json.dumps(terminal_index))
    try:
        runner._existing_terminal_result(terminal_schema_output)
    except ValueError as error:
        assert "index bytes changed" in str(error)
    else:
        raise AssertionError("noncanonical terminal index bytes must fail closed")
    common._write_json(
        terminal_index_path, {**terminal_index, "unexpectedField": True}
    )
    try:
        runner._existing_terminal_result(terminal_schema_output)
    except ValueError as error:
        assert "index fields changed" in str(error)
    else:
        raise AssertionError("extra terminal index fields must fail closed")

    risk_terminal_output = invocation_output / "risk-terminal"
    risk_terminal_output.mkdir()
    (risk_terminal_output / ".campaign.lock").write_text("")
    risk_manifest_path = risk_terminal_output / "campaign-manifest.json"
    common._write_json(
        risk_manifest_path,
        {
            "campaign": runner.CAMPAIGN_ID,
            "registrationSha256": runner.REGISTRATION_SHA256,
        },
    )
    risk_manifest_sha = common._file_digest(risk_manifest_path)
    risk_failure = {"reason": "synthetic_risk_breach"}
    reserved_holdout = {"status": "reserved", "openRequested": False}
    risk_failure_path = risk_terminal_output / "risk-failure.json"
    common._write_json(
        risk_failure_path,
        {
            "campaign": runner.CAMPAIGN_ID,
            "status": "risk_invalid",
            "registrationSha256": runner.REGISTRATION_SHA256,
            "campaignManifestSha256": risk_manifest_sha,
            "riskFailure": risk_failure,
            "finalHoldout": reserved_holdout,
        },
    )
    risk_ledger_path = risk_terminal_output / "risk-ledger.json"
    common._write_json(
        risk_ledger_path,
        {"status": "risk_invalid", "primaryFailure": risk_failure},
    )
    risk_summary_path = risk_terminal_output / "summary.json"
    risk_summary = {
        "campaign": runner.CAMPAIGN_ID,
        "status": "risk_invalid",
        "registrationSha256": runner.REGISTRATION_SHA256,
        "campaignManifestSha256": risk_manifest_sha,
        "riskFailure": risk_failure,
        "promotionGates": {"everyPrimaryPathRiskSafeAndComplete": False},
        "finalHoldout": reserved_holdout,
        "evidence": {
            "riskFailure": str(risk_failure_path.resolve()),
            "riskFailureSha256": common._file_digest(risk_failure_path),
            "riskLedger": str(risk_ledger_path.resolve()),
            "riskLedgerSha256": common._file_digest(risk_ledger_path),
        },
    }
    common._write_json(risk_summary_path, risk_summary)

    def terminal_artifact_record(path):
        return {
            "path": str(path.resolve()),
            "sha256": common._file_digest(path),
        }

    risk_terminal_index_path = risk_terminal_output / "evidence-index.json"
    risk_terminal_index = {
        "campaign": runner.CAMPAIGN_ID,
        "status": "risk_invalid",
        "registrationSha256": runner.REGISTRATION_SHA256,
        "campaignManifestSha256": risk_manifest_sha,
        "artifacts": {
            "campaignManifest": terminal_artifact_record(risk_manifest_path),
            "risk-failure": terminal_artifact_record(risk_failure_path),
            "risk-ledger": terminal_artifact_record(risk_ledger_path),
            "summary": terminal_artifact_record(risk_summary_path),
        },
    }
    common._write_json(risk_terminal_index_path, risk_terminal_index)
    assert runner._existing_terminal_result(risk_terminal_output) == risk_summary

    incompatible_summary = copy.deepcopy(risk_summary)
    incompatible_summary["finalHoldout"] = {
        "status": "pass",
        "openRequested": True,
    }
    common._write_json(risk_summary_path, incompatible_summary)
    incompatible_index = copy.deepcopy(risk_terminal_index)
    incompatible_index["artifacts"]["summary"] = terminal_artifact_record(
        risk_summary_path
    )
    common._write_json(risk_terminal_index_path, incompatible_index)
    try:
        runner._existing_terminal_result(risk_terminal_output)
    except ValueError as error:
        assert "non-final terminal holdout state is invalid" in str(error)
    else:
        raise AssertionError("incompatible non-final holdout state was accepted")
    common._write_json(risk_summary_path, risk_summary)

    renamed_index = copy.deepcopy(risk_terminal_index)
    renamed_index["artifacts"]["unexpected-ledger-key"] = renamed_index[
        "artifacts"
    ].pop("risk-ledger")
    common._write_json(risk_terminal_index_path, renamed_index)
    try:
        runner._existing_terminal_result(risk_terminal_output)
    except ValueError as error:
        assert "artifact key changed" in str(error)
    else:
        raise AssertionError("renamed terminal artifact keys must fail closed")

    common._write_json(
        risk_ledger_path,
        {"status": "risk_invalid", "primaryFailure": {"changed": True}},
    )
    risk_summary["evidence"]["riskLedgerSha256"] = common._file_digest(
        risk_ledger_path
    )
    common._write_json(risk_summary_path, risk_summary)
    tampered_chain = copy.deepcopy(risk_terminal_index)
    tampered_chain["artifacts"]["risk-ledger"] = terminal_artifact_record(
        risk_ledger_path
    )
    tampered_chain["artifacts"]["summary"] = terminal_artifact_record(
        risk_summary_path
    )
    common._write_json(risk_terminal_index_path, tampered_chain)
    try:
        runner._existing_terminal_result(risk_terminal_output)
    except ValueError as error:
        assert "risk-invalid terminal evidence cross-links changed" in str(error)
    else:
        raise AssertionError("rehashed inconsistent risk evidence must fail closed")

    for label, registered_directory in (
        ("registered predecessor", runner.PREDECESSOR_CAMPAIGN_DIRECTORY),
        ("registered source campaign", runner.SOURCE_CAMPAIGN_DIRECTORY),
        ("registered snapshot", runner.SNAPSHOT_DIRECTORY),
    ):
        alias_args = runner.parse_args([])
        alias_args.predecessor_campaign_dir = str(invocation_output / "input-a")
        alias_args.source_campaign_dir = str(invocation_output / "input-b")
        alias_args.snapshot_dir = str(invocation_output / "input-c")
        alias_output = (
            common.SHARED_REPOSITORY_ROOT
            / registered_directory
            / "must-not-be-created-v2-path-guard-test"
        )
        alias_args.output_dir = str(alias_output)
        try:
            runner._validate_runtime_paths(alias_args)
        except ValueError as error:
            assert label in str(error)
        else:
            raise AssertionError(
                f"overridden inputs may not expose {label} to output writes"
            )
        assert not alias_output.exists()

    git_output = common.GIT_COMMON_DIR / "must-not-be-created-v2-test"
    git_args = runner.parse_args(["--output-dir", str(git_output)])
    try:
        runner._validate_runtime_paths(git_args)
    except ValueError as error:
        assert "common Git metadata" in str(error)
    else:
        raise AssertionError("output may not overlap common Git metadata")
    assert not git_output.exists()

    for label, sensitive_root in (
        (
            "shared registration directory",
            common.SHARED_REPOSITORY_ROOT / "research-notes/registrations",
        ),
        (
            "shared research implementation",
            common.SHARED_REPOSITORY_ROOT / "scripts/research",
        ),
    ):
        sensitive_output = sensitive_root / "must-not-be-created-v2-test"
        sensitive_args = runner.parse_args(
            ["--output-dir", str(sensitive_output)]
        )
        try:
            runner._validate_runtime_paths(sensitive_args)
        except ValueError as error:
            assert label in str(error)
        else:
            raise AssertionError(f"output may not overlap {label}")
        assert not sensitive_output.exists()

    missing_ready_args = runner.parse_args(
        [
            "--output-dir",
            str(invocation_output / "missing-ready"),
            "--open-final-holdout",
            "--development-audit-sha256",
            "0" * 64,
        ]
    )
    try:
        runner.run(missing_ready_args)
    except ValueError as error:
        assert "prior immutable no-flag" in str(error)
    else:
        raise AssertionError("a first-invocation holdout request must fail closed")

    missing_audit_args = runner.parse_args(["--open-final-holdout"])
    try:
        runner._validate_open_authorization_args(missing_audit_args)
    except ValueError as error:
        assert "requires --development-audit-sha256" in str(error)
    else:
        raise AssertionError("opening without an audit digest must fail closed")
    stray_audit_args = runner.parse_args(
        ["--development-audit-sha256", "0" * 64]
    )
    try:
        runner._validate_open_authorization_args(stray_audit_args)
    except ValueError as error:
        assert "valid only with" in str(error)
    else:
        raise AssertionError("an audit digest without opening must fail closed")

    ready_output = invocation_output / "ready"
    ready_output.mkdir()
    (ready_output / ".campaign.lock").write_text("")
    implementation_artifacts = runner._implementation_artifacts()
    implementation_sha = common._json_digest(implementation_artifacts)

    manifest_validation_output = invocation_output / "manifest-validation"
    manifest_validation_output.mkdir()
    source_evidence = {"synthetic": True}
    exact_manifest = runner._campaign_manifest_value(
        registration,
        runner.REGISTRATION_SHA256,
        implementation_artifacts,
        source_evidence,
    )
    manifest_validation_path = (
        manifest_validation_output / "campaign-manifest.json"
    )
    common._write_json(manifest_validation_path, exact_manifest)
    manifest_validation_summary = {
        "registrationSha256": runner.REGISTRATION_SHA256,
        "campaignManifestSha256": common._file_digest(manifest_validation_path),
    }
    runner._validate_current_terminal_manifest(
        manifest_validation_output,
        manifest_validation_summary,
        registration,
        runner.REGISTRATION_SHA256,
        implementation_artifacts,
        source_evidence,
    )
    inconsistent_manifest = copy.deepcopy(exact_manifest)
    inconsistent_manifest["riskPolicy"] = {"synthetic": "mutation"}
    common._write_json(manifest_validation_path, inconsistent_manifest)
    manifest_validation_summary["campaignManifestSha256"] = common._file_digest(
        manifest_validation_path
    )
    try:
        runner._validate_current_terminal_manifest(
            manifest_validation_output,
            manifest_validation_summary,
            registration,
            runner.REGISTRATION_SHA256,
            implementation_artifacts,
            source_evidence,
        )
    except ValueError as error:
        assert "manifest semantics changed" in str(error)
    else:
        raise AssertionError("rehashed inconsistent manifests must fail closed")
    common._write_json(manifest_validation_path, exact_manifest)
    manifest_validation_summary["campaignManifestSha256"] = common._file_digest(
        manifest_validation_path
    )
    manifest_validation_summary["registrationSha256"] = "f" * 64
    try:
        runner._validate_current_terminal_manifest(
            manifest_validation_output,
            manifest_validation_summary,
            registration,
            runner.REGISTRATION_SHA256,
            implementation_artifacts,
            source_evidence,
        )
    except ValueError as error:
        assert "terminal evidence registration changed" in str(error)
    else:
        raise AssertionError("stale terminal registration pins must fail closed")

    manifest = {
        "campaign": runner.CAMPAIGN_ID,
        "registrationSha256": runner.REGISTRATION_SHA256,
        "implementationArtifacts": implementation_artifacts,
        "implementationSha256": implementation_sha,
    }
    common._write_json(ready_output / "campaign-manifest.json", manifest)
    for name in runner._development_analysis_artifact_names() - {
        "campaign-manifest.json"
    }:
        if name == "risk-ledger.json":
            common._write_json(ready_output / name, {"synthetic": True})
        else:
            (ready_output / name).write_text(f"synthetic {name}\n")
    champion = runner._eligible_names(
        runner.R.campaign_specs(feed.CONTRACT_INTERVAL_MS)
    )[0]
    ready_summary = {
        "campaign": runner.CAMPAIGN_ID,
        "registrationSha256": runner.REGISTRATION_SHA256,
        "campaignManifestSha256": common._file_digest(
            ready_output / "campaign-manifest.json"
        ),
        "status": "ready_for_final_holdout",
        "champion": champion,
        "primaryPathsRiskSafeAndComplete": True,
        "derivedPathsRiskSafeAndComplete": True,
        "promotionGates": {
            name: True for name in runner.PROMOTION_GATE_NAMES
        },
        "finalHoldout": {"status": "reserved", "openRequested": False},
        "evidence": {
            "auditablePaths": {
                Path(name).stem: {
                    "path": str((ready_output / name).resolve()),
                    "sha256": common._file_digest(ready_output / name),
                }
                for name in runner._auditable_development_artifact_names()
            }
        },
    }
    runner._write_development_ready_evidence(
        ready_output,
        ready_summary,
        registration,
        runner.REGISTRATION_SHA256,
        implementation_artifacts,
    )
    audit_sha = common._file_digest(
        ready_output / runner.DEVELOPMENT_READY_INDEX
    )
    validated_summary, validated_index = (
        runner._validate_ready_development_evidence(
            ready_output,
            registration,
            runner.REGISTRATION_SHA256,
            implementation_artifacts,
            expected_audit_sha256=audit_sha,
        )
    )
    assert validated_summary == ready_summary
    assert validated_index["promotionGateNames"] == list(
        runner.PROMOTION_GATE_NAMES
    )
    assert set(validated_index["artifacts"]) == (
        runner._ready_index_artifact_names()
    )
    ready_index_path = ready_output / runner.DEVELOPMENT_READY_INDEX
    ready_index_value = json.loads(ready_index_path.read_text())
    ready_index_path.write_text(json.dumps(ready_index_value))
    try:
        runner._validate_ready_development_evidence(
            ready_output,
            registration,
            runner.REGISTRATION_SHA256,
            implementation_artifacts,
            expected_audit_sha256=None,
        )
    except ValueError as error:
        assert "ready index bytes changed" in str(error)
    else:
        raise AssertionError("noncanonical ready-index bytes must fail closed")
    common._write_json(ready_index_path, ready_index_value)
    assert common._file_digest(ready_index_path) == audit_sha

    terminal_summary = {
        **ready_summary,
        "status": "final_holdout_failed",
        "developmentAuditSha256": audit_sha,
        "finalHoldout": {"status": "fail", "openRequested": True},
    }
    common._write_json(ready_output / "summary.json", terminal_summary)
    common._write_json(ready_output / "final-holdout-opened.json", {})
    common._write_json(ready_output / "final-holdout-result.json", {})
    original_registry_validation = runner._validate_completed_holdout_registry
    runner._validate_completed_holdout_registry = lambda *_args, **_kwargs: None
    try:
        runner._validate_terminal_ready_chain(
            ready_output,
            terminal_summary,
            registration,
            runner.REGISTRATION_SHA256,
            implementation_artifacts,
        )
        rehashed_terminal_summary = copy.deepcopy(terminal_summary)
        rehashed_terminal_summary["promotionGates"][
            runner.PROMOTION_GATE_NAMES[0]
        ] = False
        common._write_json(
            ready_output / "summary.json", rehashed_terminal_summary
        )
        try:
            runner._validate_terminal_ready_chain(
                ready_output,
                rehashed_terminal_summary,
                registration,
                runner.REGISTRATION_SHA256,
                implementation_artifacts,
            )
        except ValueError as error:
            assert "differs from frozen receipt" in str(error)
        else:
            raise AssertionError(
                "coordinated terminal development mutations must fail closed"
            )
    finally:
        runner._validate_completed_holdout_registry = original_registry_validation
        (ready_output / "final-holdout-opened.json").unlink()
        (ready_output / "final-holdout-result.json").unlink()
        (ready_output / "summary.json").write_bytes(
            (ready_output / runner.DEVELOPMENT_READY_SUMMARY).read_bytes()
        )

    tampered_summary = copy.deepcopy(ready_summary)
    tampered_summary["finalHoldout"]["openRequested"] = True
    common._write_json(ready_output / "summary.json", tampered_summary)
    try:
        runner._validate_ready_development_evidence(
            ready_output,
            registration,
            runner.REGISTRATION_SHA256,
            implementation_artifacts,
            expected_audit_sha256=audit_sha,
        )
    except ValueError as error:
        assert "mutable summary differs" in str(error)
    else:
        raise AssertionError("mutable ready summaries must be rejected")

    (ready_output / "summary.json").write_bytes(
        (ready_output / runner.DEVELOPMENT_READY_SUMMARY).read_bytes()
    )
    escaped_index = invocation_output / "escaped-ready-index.json"
    escaped_index.write_bytes(ready_index_path.read_bytes())
    ready_index_path.unlink()
    ready_index_path.symlink_to(escaped_index)
    try:
        runner._validate_ready_development_evidence(
            ready_output,
            registration,
            runner.REGISTRATION_SHA256,
            implementation_artifacts,
            expected_audit_sha256=audit_sha,
        )
    except ValueError as error:
        assert "complete immutable development-ready evidence" in str(error)
    else:
        raise AssertionError("symlinked ready indexes must fail closed")

    runner.HOLDOUT_REGISTRY_DIR = invocation_output / "noncanonical-registry"
    try:
        runner._validate_registry_policy(registration["holdoutPolicy"])
    except ValueError as error:
        assert "registered shared registry" in str(error)
    else:
        raise AssertionError("official noncanonical registries must fail closed")
    finally:
        runner.HOLDOUT_REGISTRY_DIR = common.SHARED_HOLDOUT_REGISTRY_DIR

    ordering_registry = invocation_output / "ordering-registry"
    audit_digest = "a" * 64
    opening_args = runner.parse_args(
        [
            "--open-final-holdout",
            "--development-audit-sha256",
            audit_digest,
        ]
    )

    def make_holdout_output(name):
        output = invocation_output / name
        output.mkdir()
        (output / ".campaign.lock").write_text("")
        manifest = {
            "campaign": runner.CAMPAIGN_ID,
            "registrationSha256": runner.REGISTRATION_SHA256,
            "registeredData": {
                "fullPanelDigestSha256": registration["registeredData"][
                    "fullPanelDigestSha256"
                ]
            },
        }
        common._write_json(output / "campaign-manifest.json", manifest)
        return output, common._file_digest(output / "campaign-manifest.json")

    ordering_output, ordering_manifest_sha = make_holdout_output("ordering")
    original_registry = runner.HOLDOUT_REGISTRY_DIR
    original_test_override = runner.TEST_ONLY_ALLOW_REGISTRY_OVERRIDE
    original_input_assertion = runner._assert_inputs_unchanged
    original_ready_validation = runner._validate_ready_development_evidence
    original_snapshot_loader = runner.V1._load_full_registered_inputs
    snapshot_loader_calls = [0]

    def reservation_spy(_snapshot_dir, _registration):
        snapshot_loader_calls[0] += 1
        markers = list(ordering_registry.glob("*.json"))
        assert len(markers) == 1
        shared = json.loads(markers[0].read_text())
        local = json.loads(
            (ordering_output / "final-holdout-opened.json").read_text()
        )
        assert shared == local
        assert shared["status"] == "opening"
        assert shared["developmentAuditSha256"] == audit_digest
        raise RuntimeError("snapshot loader sentinel")

    runner.HOLDOUT_REGISTRY_DIR = ordering_registry
    runner.TEST_ONLY_ALLOW_REGISTRY_OVERRIDE = True
    runner._assert_inputs_unchanged = lambda *_args, **_kwargs: None
    runner._validate_ready_development_evidence = (
        lambda *_args, **_kwargs: ({}, {})
    )
    runner.V1._load_full_registered_inputs = reservation_spy
    try:
        ordered_result = runner._open_final_holdout(
            opening_args,
            ordering_output,
            invocation_output / "predecessor",
            invocation_output / "source",
            invocation_output / "snapshot",
            registration,
            runner.REGISTRATION_SHA256,
            implementation_artifacts,
            implementation_sha,
            ordering_manifest_sha,
            audit_digest,
            champion,
            runner.R.campaign_specs(feed.CONTRACT_INTERVAL_MS),
            common._periods_per_year(feed.CONTRACT_INTERVAL_MS),
        )
        assert snapshot_loader_calls[0] == 1
        assert ordered_result["status"] == "fail"
        assert ordered_result["failure"]["reason"] == "holdout_execution_error"
        completed_local = json.loads(
            (ordering_output / "final-holdout-opened.json").read_text()
        )
        assert completed_local["status"] == "completed"
        assert completed_local["result"] == ordered_result

        second_output, second_manifest_sha = make_holdout_output("ordering-two")
        try:
            runner._open_final_holdout(
                opening_args,
                second_output,
                invocation_output / "predecessor",
                invocation_output / "source",
                invocation_output / "snapshot",
                registration,
                runner.REGISTRATION_SHA256,
                implementation_artifacts,
                implementation_sha,
                second_manifest_sha,
                audit_digest,
                champion,
                runner.R.campaign_specs(feed.CONTRACT_INTERVAL_MS),
                common._periods_per_year(feed.CONTRACT_INTERVAL_MS),
            )
        except ValueError as error:
            assert "overlaps an already consumed" in str(error)
        else:
            raise AssertionError("an overlapping holdout must remain consumed")
        assert snapshot_loader_calls[0] == 1
        assert not (second_output / "final-holdout-opened.json").exists()
    finally:
        runner.HOLDOUT_REGISTRY_DIR = original_registry
        runner.TEST_ONLY_ALLOW_REGISTRY_OVERRIDE = original_test_override
        runner._assert_inputs_unchanged = original_input_assertion
        runner._validate_ready_development_evidence = original_ready_validation
        runner.V1._load_full_registered_inputs = original_snapshot_loader

stable_order = tuple(registration["strategy"]["stableTieOrder"])
assert stable_order == runner.REGISTERED_STABLE_TIE_ORDER
for changed_order in (
    "not-a-list",
    list(stable_order[:-1]),
    [*stable_order[:-1], stable_order[0]],
    [*stable_order[:-1], "UNKNOWN"],
    list(reversed(stable_order)),
):
    changed = copy.deepcopy(registration)
    changed["strategy"]["stableTieOrder"] = changed_order
    try:
        runner._registered_stable_tie_order(changed)
    except ValueError as error:
        assert "stableTieOrder" in str(error)
    else:
        raise AssertionError("invalid stable tie orders must fail closed")

changed = copy.deepcopy(registration)
changed["strategy"]["grossExposure"] = 0.3
try:
    runner._validate_registration(changed)
except ValueError as error:
    assert "grossExposure" in str(error)
else:
    raise AssertionError("the registered v2 gross target must remain fixed")

times = (
    runner.REGISTERED_START_OPEN_TIME
    + np.arange(80, dtype=np.int64) * feed.CONTRACT_INTERVAL_MS
)
shared_close = 100.0 * np.exp(np.linspace(0.0, 0.1, len(times)))
panel = {
    symbol: pd.DataFrame(
        {
            "openTime": times,
            "closeTime": times + feed.CONTRACT_INTERVAL_MS - 1,
            "close": shared_close,
        }
    )
    for symbol in sorted(stable_order)
}
observed_residual_inputs = []
original_residual = runner.H._residual_momentum

def residual_spy(close, lookback, horizons):
    observed_residual_inputs.append(tuple(close.columns))
    return original_residual(close, lookback, horizons)

runner.H._residual_momentum = residual_spy
try:
    close, residual = runner._trial_inputs(panel, registration)
finally:
    runner.H._residual_momentum = original_residual
assert observed_residual_inputs == [stable_order]
assert tuple(close.columns) == stable_order
assert all(tuple(frame.columns) == stable_order for frame in residual.values())
synthetic_momentum = pd.DataFrame(
    np.tile(np.linspace(-5.0, 5.0, len(stable_order)), (3, 1)),
    index=close.index[:3],
    columns=stable_order,
)
v2_decisions = runner.R.decision_weights_for_trial(
    synthetic_momentum,
    runner.R.campaign_specs(feed.CONTRACT_INTERVAL_MS)[0],
    v2_config,
)
assert np.isclose(v2_decisions.iloc[0].abs().sum(), 0.25)
assert np.isclose(v2_decisions.iloc[0].max(), 0.125)
assert np.isclose(v2_decisions.iloc[0].min(), -0.125)

missing_panel = dict(panel)
missing_panel.pop(stable_order[-1])
try:
    runner._trial_inputs(missing_panel, registration)
except ValueError as error:
    assert "panel symbols" in str(error)
else:
    raise AssertionError("incomplete symbol coverage must fail before scoring")

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    synthetic = copy.deepcopy(registration)
    policy = synthetic["adaptivePredecessorEvidence"]
    predecessor_dir = root / policy["directory"]
    predecessor_dir.mkdir(parents=True)
    (predecessor_dir / ".campaign.lock").write_text("")

    manifest = {
        "campaign": runner.PRIOR.CAMPAIGN_ID,
        "registrationSha256": policy["registrationSha256"],
        "implementationSha256": policy["implementationSha256"],
        "implementationArtifacts": policy["implementationArtifacts"],
    }
    manifest_path = predecessor_dir / policy["campaignManifest"]
    manifest_path.write_text(json.dumps(manifest))
    policy["campaignManifestSha256"] = common._file_digest(manifest_path)

    required = policy["requiredResult"]
    risk_failure = {
        **required["riskFailure"],
        "immediateCloseLiquidationEvidence": {
            "field": "drawdown",
            "observed": 0.21,
            "limit": 0.2,
        },
    }
    final_holdout = {
        **required["finalHoldout"],
        "startOpenTime": 1,
        "endOpenTime": 2,
    }
    failure = {
        "campaign": runner.PRIOR.CAMPAIGN_ID,
        "status": required["status"],
        "registrationSha256": policy["registrationSha256"],
        "campaignManifestSha256": policy["campaignManifestSha256"],
        "riskFailure": risk_failure,
        "finalHoldout": final_holdout,
    }
    failure_path = predecessor_dir / policy["riskFailure"]
    failure_path.write_text(json.dumps(failure))
    policy["riskFailureSha256"] = common._file_digest(failure_path)

    ledger_path = predecessor_dir / policy["riskLedger"]
    ledger_path.write_text(
        json.dumps({"status": required["status"], "primaryFailure": risk_failure})
    )
    policy["riskLedgerSha256"] = common._file_digest(ledger_path)

    summary = {
        **failure,
        "promotionGates": required["promotionGates"],
        "evidence": {
            "riskFailureSha256": policy["riskFailureSha256"],
            "riskLedgerSha256": policy["riskLedgerSha256"],
        },
    }
    summary_path = predecessor_dir / policy["summary"]
    summary_path.write_text(json.dumps(summary))
    policy["summarySha256"] = common._file_digest(summary_path)

    index = {
        "campaign": runner.PRIOR.CAMPAIGN_ID,
        "status": required["status"],
        "registrationSha256": policy["registrationSha256"],
        "campaignManifestSha256": policy["campaignManifestSha256"],
        "artifacts": {
            "campaignManifest": {"sha256": policy["campaignManifestSha256"]},
            "risk-failure": {"sha256": policy["riskFailureSha256"]},
            "risk-ledger": {"sha256": policy["riskLedgerSha256"]},
            "summary": {"sha256": policy["summarySha256"]},
        },
    }
    index_path = predecessor_dir / policy["evidenceIndex"]
    index_path.write_text(json.dumps(index))
    policy["evidenceIndexSha256"] = common._file_digest(index_path)

    original_shared_root = common.SHARED_REPOSITORY_ROOT
    common.SHARED_REPOSITORY_ROOT = root
    try:
        evidence = runner._validate_adaptive_predecessor_evidence(synthetic)
        assert evidence["terminalStatus"] == "risk_invalid"
        summary_path.write_text("{}")
        try:
            runner._validate_adaptive_predecessor_evidence(synthetic)
        except ValueError as error:
            assert "hash mismatch" in str(error)
        else:
            raise AssertionError("changed adaptive evidence must fail closed")
    finally:
        common.SHARED_REPOSITORY_ROOT = original_shared_root
`;
    const run = spawnSync("python3", ["-c", program, RESEARCH_DIR], {
      encoding: "utf8",
    });
    assert.equal(run.status, 0, run.stderr);
  },
);

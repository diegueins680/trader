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
  "strict holdout registry binds marker identity and campaign evidence",
  { skip: !hasResearchPython },
  () => {
    const run = spawnSync(
      "python3",
      [
        "-c",
        String.raw`
import hashlib
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, sys.argv[1])
import campaign_runner as runner

with tempfile.TemporaryDirectory() as layout_temporary:
    layout = runner.Path(layout_temporary)
    canonical = layout / "canonical repo"
    linked = layout / "linked worktree"
    stale = layout / "stale worktree"
    subprocess.run(["git", "init", str(canonical)], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(
        ["git", "-C", str(canonical), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(canonical), "config", "user.name", "Registry Test"],
        check=True,
    )
    (canonical / "tracked").write_text("test\n")
    subprocess.run(["git", "-C", str(canonical), "add", "tracked"], check=True)
    subprocess.run(
        ["git", "-C", str(canonical), "commit", "-m", "fixture"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    for worktree in (linked, stale):
        subprocess.run(
            [
                "git",
                "-C",
                str(canonical),
                "worktree",
                "add",
                "--detach",
                str(worktree),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )

    assert runner._shared_repository_root(canonical) == canonical.resolve()
    assert runner._shared_repository_root(linked) == canonical.resolve()
    assert runner._git_common_directory(canonical) == (canonical / ".git").resolve()
    assert runner._git_common_directory(linked) == (canonical / ".git").resolve()
    canonical_registry = runner._configured_shared_holdout_registry(
        runner._shared_repository_root(canonical), {}
    )
    linked_registry = runner._configured_shared_holdout_registry(
        runner._shared_repository_root(linked), {}
    )
    assert canonical_registry == linked_registry
    previous_override = os.environ.get("TRADER_EDGE_HOLDOUT_REGISTRY")
    os.environ["TRADER_EDGE_HOLDOUT_REGISTRY"] = str(layout / "ignored")
    try:
        assert runner._configured_shared_holdout_registry(canonical) == (
            canonical / runner.SHARED_HOLDOUT_REGISTRY_RELATIVE
        ).resolve()
    finally:
        if previous_override is None:
            del os.environ["TRADER_EDGE_HOLDOUT_REGISTRY"]
        else:
            os.environ["TRADER_EDGE_HOLDOUT_REGISTRY"] = previous_override
    absolute_override = layout / "absolute-registry"
    assert runner._configured_shared_holdout_registry(
        canonical,
        {"TRADER_EDGE_HOLDOUT_REGISTRY": str(absolute_override)},
    ) == absolute_override.resolve()

    try:
        runner._configured_shared_holdout_registry(
            canonical, {"TRADER_EDGE_HOLDOUT_REGISTRY": "relative/registry"}
        )
    except ValueError as error:
        assert "must be absolute" in str(error)
    else:
        raise AssertionError("relative shared-registry overrides must fail closed")

    malformed = layout / "malformed"
    malformed.mkdir()
    (malformed / ".git").write_text("not-gitdir-metadata\n")
    try:
        runner._shared_repository_root(malformed)
    except ValueError as error:
        assert "unable to resolve Git worktree metadata" in str(error)
    else:
        raise AssertionError("malformed linked-worktree metadata must fail closed")

    legacy_registry = linked / runner.SHARED_HOLDOUT_REGISTRY_RELATIVE
    legacy_registry.mkdir(parents=True)
    (legacy_registry / ("a" * 64 + ".json")).write_text("{}")
    try:
        runner._assert_shared_registry_reconciled(
            canonical, canonical, canonical_registry
        )
    except ValueError as error:
        assert "require reconciliation" in str(error)
    else:
        raise AssertionError("legacy worktree-local markers must fail closed")

    original_repository_root = runner.REPOSITORY_ROOT
    original_shared_root = runner.SHARED_REPOSITORY_ROOT
    original_canonical_registry = runner.CANONICAL_SHARED_HOLDOUT_REGISTRY_DIR
    original_shared_registry = runner.SHARED_HOLDOUT_REGISTRY_DIR
    original_resolution_error = runner.SHARED_REPOSITORY_RESOLUTION_ERROR
    guard_output = layout / "guard-output"
    guard_output.mkdir()
    guard_marker = canonical_registry / ("c" * 64 + ".json")
    guard_output_record = guard_output / "final-holdout-opened.json"
    runner.REPOSITORY_ROOT = canonical
    runner.SHARED_REPOSITORY_ROOT = canonical
    runner.CANONICAL_SHARED_HOLDOUT_REGISTRY_DIR = canonical_registry
    runner.SHARED_HOLDOUT_REGISTRY_DIR = canonical_registry
    runner.SHARED_REPOSITORY_RESOLUTION_ERROR = None
    try:
        try:
            runner._reserve_holdout(
                canonical_registry,
                guard_marker,
                runner._holdout_window(["AAA"], "8h", 100, 200),
                guard_output_record,
                {},
            )
        except ValueError as error:
            assert "require reconciliation" in str(error)
        else:
            raise AssertionError(
                "shared reservation must reconcile every linked worktree"
            )
        assert not guard_marker.exists()
        assert not guard_output_record.exists()
        runner.SHARED_REPOSITORY_RESOLUTION_ERROR = "synthetic failure"
        runner._assert_output_holdout_not_consumed(
            canonical_registry, guard_output
        )
        runner._reserve_holdout(
            canonical_registry,
            guard_marker,
            runner._holdout_window(["AAA"], "8h", 100, 200),
            guard_output_record,
            {},
        )
        assert guard_marker.exists()
        assert guard_output_record.exists()
        guard_marker.unlink()
        guard_output_record.unlink()
        try:
            runner._assert_output_holdout_not_consumed(
                canonical_registry, guard_output, strict_identity=True
            )
        except ValueError as error:
            assert "official shared holdout registry cannot be resolved" in str(error)
        else:
            raise AssertionError("strict official fallback registries must fail closed")
        try:
            runner._reserve_holdout(
                canonical_registry,
                guard_marker,
                runner._holdout_window(["AAA"], "8h", 100, 200),
                guard_output_record,
                {},
                strict_identity=True,
            )
        except ValueError as error:
            assert "official shared holdout registry cannot be resolved" in str(error)
        else:
            raise AssertionError("strict fallback reservations must fail closed")
        assert not guard_marker.exists()
        assert not guard_output_record.exists()
    finally:
        runner.REPOSITORY_ROOT = original_repository_root
        runner.SHARED_REPOSITORY_ROOT = original_shared_root
        runner.CANONICAL_SHARED_HOLDOUT_REGISTRY_DIR = original_canonical_registry
        runner.SHARED_HOLDOUT_REGISTRY_DIR = original_shared_registry
        runner.SHARED_REPOSITORY_RESOLUTION_ERROR = original_resolution_error
    next(legacy_registry.glob("*.json")).unlink()

    # Git still reports prunable paths. Their remaining directory must be
    # scanned even if the .git indirection was deleted.
    (stale / ".git").unlink()
    stale_registry = stale / runner.SHARED_HOLDOUT_REGISTRY_RELATIVE
    stale_registry.mkdir(parents=True)
    (stale_registry / ("b" * 64 + ".json")).write_text("{}")
    try:
        runner._assert_shared_registry_reconciled(
            canonical, canonical, canonical_registry
        )
    except ValueError as error:
        assert str(stale_registry.resolve()) in str(error)
    else:
        raise AssertionError("prunable worktree markers must fail closed")

    separate_root = layout / "separate root"
    separate_git = layout / "separate metadata"
    subprocess.run(
        [
            "git",
            "init",
            "--separate-git-dir",
            str(separate_git),
            str(separate_root),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    assert runner._shared_repository_root(separate_root) == separate_root.resolve()
    assert runner._git_common_directory(separate_root) == separate_git.resolve()
    subprocess.run(
        ["git", "-C", str(separate_root), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(separate_root), "config", "user.name", "Registry Test"],
        check=True,
    )
    (separate_root / "tracked").write_text("test\n")
    subprocess.run(
        ["git", "-C", str(separate_root), "add", "tracked"], check=True
    )
    subprocess.run(
        ["git", "-C", str(separate_root), "commit", "-m", "fixture"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    separate_linked = layout / "separate linked"
    subprocess.run(
        [
            "git",
            "-C",
            str(separate_root),
            "worktree",
            "add",
            "--detach",
            str(separate_linked),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    for checkout in (separate_root, separate_linked):
        try:
            runner._shared_repository_root(checkout)
        except ValueError as error:
            assert "separate-git-dir with linked worktrees" in str(error)
        else:
            raise AssertionError(
                "separate-git-dir linked worktrees must fail closed"
            )

with tempfile.TemporaryDirectory() as temporary:
    root = runner.Path(temporary)
    unsafe_output = root / "unsafe-output"
    unsafe_output.mkdir()
    campaign_lock_target = root / "campaign-lock-target"
    (unsafe_output / ".campaign.lock").symlink_to(campaign_lock_target)
    try:
        with runner._campaign_output_lock(unsafe_output):
            raise AssertionError("unsafe campaign lock unexpectedly acquired")
    except ValueError as error:
        assert "lock path is unsafe" in str(error)
    assert not campaign_lock_target.exists()

    unsafe_registry = root / "unsafe-registry"
    unsafe_registry.mkdir()
    registry_lock_target = root / "registry-lock-target"
    (unsafe_registry / ".registry.lock").symlink_to(registry_lock_target)
    try:
        with runner._holdout_registry_lock(unsafe_registry):
            raise AssertionError("unsafe registry lock unexpectedly acquired")
    except ValueError as error:
        assert "lock path is unsafe" in str(error)
    assert not registry_lock_target.exists()

    dangling_output = root / "dangling-output"
    dangling_output.mkdir()
    dangling_target = root / "missing-output-record-target"
    dangling_record = dangling_output / "final-holdout-opened.json"
    dangling_record.symlink_to(dangling_target)
    dangling_registry = root / "dangling-registry"
    dangling_marker = dangling_registry / ("d" * 64 + ".json")
    try:
        runner._assert_output_holdout_not_consumed(
            dangling_registry, dangling_output
        )
    except ValueError as error:
        assert "output record path is unsafe" in str(error)
    else:
        raise AssertionError("dangling output-record symlinks must fail closed")
    try:
        runner._reserve_holdout(
            dangling_registry,
            dangling_marker,
            runner._holdout_window(["AAA"], "8h", 100, 200),
            dangling_record,
            {},
        )
    except ValueError as error:
        assert "output record path is unsafe" in str(error)
    else:
        raise AssertionError("reservation followed a dangling output record")
    assert not dangling_marker.exists()
    assert not dangling_target.exists()

    locked_registry = root / "locked-registry"
    escaped_registry = root / "escaped-registry"
    escaped_registry.mkdir()
    escaped_marker = escaped_registry / ("e" * 64 + ".json")
    escaped_output = root / "escaped-output"
    escaped_output.mkdir()
    try:
        runner._reserve_holdout(
            locked_registry,
            escaped_marker,
            runner._holdout_window(["AAA"], "8h", 100, 200),
            escaped_output / "final-holdout-opened.json",
            {},
        )
    except ValueError as error:
        assert "marker escaped its locked registry" in str(error)
    else:
        raise AssertionError("holdout marker escaped the registry lock domain")
    assert not escaped_marker.exists()
    assert not locked_registry.exists()

    window = runner._holdout_window(["AAA", "BBB"], "8h", 100, 200)
    campaign = "residual_reversal_rank_hysteresis_risk_v1"
    panel_sha = "3" * 64
    identity = runner._json_digest({
        "campaign": campaign,
        "panelSha256": panel_sha,
        "window": window,
    })
    output_path = root / "output"
    output_path.mkdir()
    strict_manifest = {
        "campaign": campaign,
        "registrationSha256": "1" * 64,
        "registeredData": {"fullPanelDigestSha256": panel_sha},
    }
    strict_manifest_path = output_path / "campaign-manifest.json"
    strict_manifest_path.write_text(json.dumps(strict_manifest, indent=2))
    output_directory = str(output_path.resolve())
    record = {
        "registryVersion": runner.HOLDOUT_REGISTRY_VERSION,
        "status": "opening",
        "campaign": campaign,
        "registrationSha256": "1" * 64,
        "campaignManifestSha256": hashlib.sha256(
            strict_manifest_path.read_bytes()
        ).hexdigest(),
        "panelSha256": panel_sha,
        "holdoutIdentitySha256": identity,
        "outputBindingSha256": runner._json_digest({
            "holdoutIdentitySha256": identity,
            "outputDirectory": output_directory,
        }),
        "window": window,
        "artifacts": {"outputDirectory": output_directory},
    }
    registry = root / "registry"
    marker = registry / f"{identity}.json"
    output_record = root / "output" / "final-holdout-opened.json"
    runner._reserve_holdout(
        registry,
        marker,
        window,
        output_record,
        record,
        strict_identity=True,
    )
    assert json.loads(marker.read_text()) == record

    overlap_failure = None
    try:
        runner._assert_holdout_available(
            registry,
            runner._holdout_window(["BBB", "CCC"], "8h", 150, 250),
            root / "other-output" / "final-holdout-opened.json",
            strict_identity=True,
        )
    except ValueError as error:
        overlap_failure = str(error)

    legacy_campaign = "residual_momentum_funding_only_v1"
    legacy_identity = runner._json_digest({
        "campaign": legacy_campaign,
        "panelSha256": panel_sha,
        "window": window,
    })
    legacy_output = root / "legacy-owned-output"
    legacy_output.mkdir()
    legacy_manifest = {
        "campaign": legacy_campaign,
        "registrationSha256": "1" * 64,
        "registeredData": {"panelSha256": panel_sha},
    }
    legacy_manifest_path = legacy_output / "campaign-manifest.json"
    legacy_manifest_path.write_text(json.dumps(legacy_manifest))
    legacy_record = {
        **record,
        "campaign": legacy_campaign,
        "campaignManifestSha256": runner._json_digest(legacy_manifest),
        "holdoutIdentitySha256": legacy_identity,
        "artifacts": {"outputDirectory": str(legacy_output.resolve())},
    }
    del legacy_record["panelSha256"]
    del legacy_record["outputBindingSha256"]
    legacy_registry = root / "legacy-registry"
    legacy_registry.mkdir()
    (legacy_registry / f"{legacy_identity}.json").write_text(
        json.dumps(legacy_record)
    )
    runner._assert_holdout_available(
        legacy_registry,
        runner._holdout_window(["ZZZ"], "8h", 500, 600),
        root / "legacy-unrelated-output.json",
        strict_identity=True,
    )

    reversal_campaign = "residual_reversal_turnover_v1"
    reversal_identity = runner._json_digest({
        "campaign": reversal_campaign,
        "panelSha256": panel_sha,
        "window": window,
    })
    reversal_output = root / "legacy-reversal-output"
    reversal_output.mkdir()
    reversal_manifest = {
        "campaign": reversal_campaign,
        "registrationSha256": "1" * 64,
        "registeredData": {"fullPanelDigestSha256": panel_sha},
    }
    reversal_manifest_path = reversal_output / "campaign-manifest.json"
    reversal_manifest_path.write_text(json.dumps(reversal_manifest, indent=2))
    reversal_record = {
        **legacy_record,
        "campaign": reversal_campaign,
        "campaignManifestSha256": hashlib.sha256(
            reversal_manifest_path.read_bytes()
        ).hexdigest(),
        "holdoutIdentitySha256": reversal_identity,
        "artifacts": {"outputDirectory": str(reversal_output.resolve())},
    }
    reversal_registry = root / "legacy-reversal-registry"
    reversal_registry.mkdir()
    (reversal_registry / f"{reversal_identity}.json").write_text(
        json.dumps(reversal_record)
    )
    runner._assert_holdout_available(
        reversal_registry,
        runner._holdout_window(["ZZZ"], "8h", 500, 600),
        root / "legacy-reversal-unrelated-output.json",
        strict_identity=True,
    )

    edge_campaign = "residual_momentum_derivatives_ablation_v1"
    edge_identity = runner._json_digest({
        "campaign": edge_campaign,
        "panelSha256": panel_sha,
        "window": window,
    })
    edge_output = root / "legacy-edge-output"
    edge_output.mkdir()
    edge_manifest = {
        "campaign": edge_campaign,
        "registeredData": {"panelSha256": panel_sha},
    }
    (edge_output / "campaign-manifest.json").write_text(
        json.dumps(edge_manifest, indent=2)
    )
    edge_record = {
        **legacy_record,
        "campaign": edge_campaign,
        "registrationSha256": runner._json_digest(edge_manifest),
        "holdoutIdentitySha256": edge_identity,
        "artifacts": {"outputDirectory": str(edge_output.resolve())},
    }
    del edge_record["campaignManifestSha256"]
    edge_registry = root / "legacy-edge-registry"
    edge_registry.mkdir()
    (edge_registry / f"{edge_identity}.json").write_text(
        json.dumps(edge_record)
    )
    runner._assert_holdout_available(
        edge_registry,
        runner._holdout_window(["ZZZ"], "8h", 500, 600),
        root / "legacy-edge-unrelated-output.json",
        strict_identity=True,
    )
    legacy_overlap_failure = None
    try:
        runner._assert_holdout_available(
            legacy_registry,
            runner._holdout_window(["BBB", "ZZZ"], "8h", 150, 250),
            root / "legacy-overlap-output.json",
            strict_identity=True,
        )
    except ValueError as error:
        legacy_overlap_failure = str(error)
    legacy_reservation_consumed_failure = None
    try:
        runner._assert_holdout_available(
            legacy_registry,
            runner._holdout_window(["ZZZ"], "8h", 500, 600),
            legacy_output / "final-holdout-opened.json",
            strict_identity=True,
        )
    except ValueError as error:
        legacy_reservation_consumed_failure = str(error)
    legacy_consumed_failure = None
    try:
        runner._assert_output_holdout_not_consumed(
            legacy_registry,
            legacy_output,
            strict_identity=True,
        )
    except ValueError as error:
        legacy_consumed_failure = str(error)

    legacy_missing_artifacts = root / "legacy-missing-artifacts"
    legacy_missing_artifacts.mkdir()
    (legacy_missing_artifacts / f"{legacy_identity}.json").write_text(
        json.dumps({
            key: value
            for key, value in legacy_record.items()
            if key != "artifacts"
        })
    )
    legacy_artifacts_failure = None
    try:
        runner._assert_output_holdout_not_consumed(
            legacy_missing_artifacts,
            root / "legacy-unmapped-output",
            strict_identity=True,
        )
    except ValueError as error:
        legacy_artifacts_failure = str(error)

    failures = []
    malformed_cases = (
        ("wrong-name.json", record),
        (f"{identity}.json", {**record, "registrationSha256": "not-a-sha"}),
        (f"{identity}.json", {**record, "registrationSha256": "4" * 64}),
        (f"{identity}.json", {**record, "campaignManifestSha256": "4" * 64}),
        (f"{identity}.json", {**record, "campaign": ""}),
        (f"{identity}.json", {**record, "panelSha256": "4" * 64}),
        (
            f"{identity}.json",
            {key: value for key, value in record.items() if key != "panelSha256"},
        ),
        (
            f"{identity}.json",
            {
                **{
                    key: value
                    for key, value in record.items()
                    if key != "panelSha256"
                },
                "campaign": legacy_campaign,
            },
        ),
        (
            f"{identity}.json",
            {
                **record,
                "artifacts": {
                    "outputDirectory": str((root / "changed-output").resolve())
                },
            },
        ),
        (
            f"{legacy_identity}.json",
            {**legacy_record, "registrationSha256": "not-a-sha"},
        ),
    )
    for position, (name, malformed) in enumerate(malformed_cases):
        bad_registry = root / f"bad-registry-{position}"
        bad_registry.mkdir()
        (bad_registry / name).write_text(json.dumps(malformed))
        try:
            runner._assert_holdout_available(
                bad_registry,
                runner._holdout_window(["ZZZ"], "8h", 500, 600),
                root / f"unused-{position}.json",
                strict_identity=True,
            )
        except ValueError as error:
            failures.append(str(error))

    mismatched_marker = root / "fresh" / "wrong.json"
    try:
        runner._reserve_holdout(
            mismatched_marker.parent,
            mismatched_marker,
            window,
            root / "unused-output.json",
            record,
            strict_identity=True,
        )
    except ValueError as error:
        failures.append(str(error))

    correct_marker = root / "wrong-output" / f"{identity}.json"
    try:
        runner._reserve_holdout(
            correct_marker.parent,
            correct_marker,
            window,
            root / "different-output" / "final-holdout-opened.json",
            record,
            strict_identity=True,
        )
    except ValueError as error:
        failures.append(str(error))

    print(json.dumps({
        "failures": failures,
        "legacyArtifactsFailure": legacy_artifacts_failure,
        "legacyConsumedFailure": legacy_consumed_failure,
        "legacyOverlapFailure": legacy_overlap_failure,
        "legacyReservationConsumedFailure": (
            legacy_reservation_consumed_failure
        ),
        "overlapFailure": overlap_failure,
    }))
`,
        RESEARCH_DIR,
      ],
      { encoding: "utf8" },
    );

    assert.equal(run.status, 0, run.stderr);
    const result = JSON.parse(run.stdout);
    assert.equal(result.failures.length, 12);
    assert.ok(result.failures.every((message) => /registry entry/.test(message)));
    assert.match(result.legacyArtifactsFailure, /registry entry/);
    assert.match(result.legacyConsumedFailure, /already consumed/);
    assert.match(
      result.legacyOverlapFailure,
      /overlaps an already consumed/,
    );
    assert.match(
      result.legacyReservationConsumedFailure,
      /already consumed/,
    );
    assert.match(result.overlapFailure, /overlaps an already consumed/);
  },
);

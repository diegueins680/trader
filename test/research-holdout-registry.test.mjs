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
import json
import sys
import tempfile

sys.path.insert(0, sys.argv[1])
import campaign_runner as runner

with tempfile.TemporaryDirectory() as temporary:
    root = runner.Path(temporary)
    window = runner._holdout_window(["AAA", "BBB"], "8h", 100, 200)
    campaign = "strict_campaign_v1"
    panel_sha = "3" * 64
    identity = runner._json_digest({
        "campaign": campaign,
        "panelSha256": panel_sha,
        "window": window,
    })
    record = {
        "registryVersion": runner.HOLDOUT_REGISTRY_VERSION,
        "status": "opening",
        "campaign": campaign,
        "registrationSha256": "1" * 64,
        "campaignManifestSha256": "2" * 64,
        "panelSha256": panel_sha,
        "holdoutIdentitySha256": identity,
        "window": window,
        "artifacts": {"outputDirectory": str((root / "output").resolve())},
    }
    registry = root / "registry"
    marker = registry / f"{identity}.json"
    output_record = root / "output" / "final-holdout-opened.json"
    output_record.parent.mkdir()
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

    failures = []
    malformed_cases = (
        ("wrong-name.json", record),
        (f"{identity}.json", {**record, "registrationSha256": "not-a-sha"}),
        (f"{identity}.json", {**record, "campaign": ""}),
        (f"{identity}.json", {**record, "panelSha256": "4" * 64}),
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

    print(json.dumps({
        "failures": failures,
        "overlapFailure": overlap_failure,
    }))
`,
        RESEARCH_DIR,
      ],
      { encoding: "utf8" },
    );

    assert.equal(run.status, 0, run.stderr);
    const result = JSON.parse(run.stdout);
    assert.equal(result.failures.length, 5);
    assert.ok(result.failures.every((message) => /registry entry/.test(message)));
    assert.match(result.overlapFailure, /overlaps an already consumed/);
  },
);

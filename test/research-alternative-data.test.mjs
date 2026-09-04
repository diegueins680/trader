import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

const RESEARCH_DIR = fileURLToPath(new URL("../scripts/research/", import.meta.url));

test("alternative data is cached, PIT aligned, normalized, and failure isolated", () => {
  const program = String.raw`
import csv
import json
from pathlib import Path
import sys
import tempfile
import urllib.error

sys.path.insert(0, sys.argv[1])
import alternative_data as alt

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    sources = []
    delayed_family = "policy"
    flow_family = "security"
    for index, family in enumerate(alt.FAMILIES):
        feed = root / f"{family}.csv"
        with feed.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["eventTime", "availableTime", "value", "entity", "revision"],
            )
            writer.writeheader()
            available = 120000 if family == delayed_family else 0
            writer.writerow({
                "eventTime": 0,
                "availableTime": available,
                "value": index + 1,
                "entity": "BTC",
                "revision": "v1",
            })
        sources.append({
            "id": f"fixture-{family}",
            "family": family,
            "adapter": "csv",
            "location": str(feed),
            "metric": "fixture",
            "eventTimeField": "eventTime",
            "availableTimeField": "availableTime",
            "valueField": "value",
            "entityField": "entity",
            "revisionField": "revision",
            "aggregation": "sum" if family == flow_family else "last",
            "transform": "raw",
            "minHistory": 1,
        })
    sources.append({
        "id": "unavailable-provider",
        "family": "news",
        "adapter": "json",
        "location": str(root / "missing.json"),
        "recordsPath": "data",
        "eventTimeField": "timestamp",
        "valueField": "value",
        "metric": "unavailable",
    })
    config_path = root / "config.json"
    cache_path = root / "observations.csv"
    config_path.write_text(json.dumps({
        "schemaVersion": 1,
        "cache": str(cache_path),
        "sources": sources,
    }), encoding="utf-8")

    config = alt.load_config(config_path)
    results, total = alt.collect(config)
    assert total == len(alt.FAMILIES)
    assert sum(result.status == "ok" for result in results) == len(alt.FAMILIES)
    assert results[-1].status == "error"
    assert "missing.json" in results[-1].error

    rows = alt.read_cache(cache_path)
    assert {row.family for row in rows} == set(alt.FAMILIES)
    assert all(row.timestamp >= row.eventTime for row in rows)

    bars_path = root / "bars.csv"
    with bars_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["openTime", "close"])
        writer.writeheader()
        for open_time in [0, 60000, 120000, 180000]:
            writer.writerow({"openTime": open_time, "close": 100})
    panel_path = root / "panel.csv"
    manifest_path = root / "panel.json"
    manifest = alt.build_panel(
        cache_path,
        bars_path,
        panel_path,
        interval_ms=60000,
        manifest_path=manifest_path,
        symbol="BTCUSDT",
    )
    with panel_path.open(newline="", encoding="utf-8") as handle:
        panel = list(csv.DictReader(handle))
    assert len(panel) == 4
    assert set(alt.FAMILIES).issubset(panel[0])
    assert {row["symbol"] for row in panel} == {"BTCUSDT"}
    assert float(panel[0][delayed_family]) == 0
    assert float(panel[1][delayed_family]) == 0
    assert float(panel[2][delayed_family]) == min(5, alt.FAMILIES.index(delayed_family) + 1)
    assert float(panel[0][flow_family]) == min(5, alt.FAMILIES.index(flow_family) + 1)
    assert float(panel[1][flow_family]) == 0
    assert manifest["pointInTime"] is True
    assert manifest["barsCount"] == 4
    assert manifest["symbol"] == "BTCUSDT"
    persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted_manifest["intervalMs"] == 60000
    assert persisted_manifest["panelSchema"] == {
        "id": alt.PANEL_SCHEMA_ID,
        "version": alt.PANEL_SCHEMA_VERSION,
        "featureAvailabilitySchemaId": alt.FEATURE_AVAILABILITY_SCHEMA_ID,
        "columns": alt._panel_fields(),
        "coverage": {
            "suffix": "_coverage",
            "minimum": 0.0,
            "maximum": 1.0,
            "zeroMeansUnavailable": True,
            "unavailableValue": 0.0,
        },
    }
    assert set(persisted_manifest["artifacts"]) == {"cache", "bars", "panel"}
    for name, path in {
        "cache": cache_path,
        "bars": bars_path,
        "panel": panel_path,
    }.items():
        artifact = persisted_manifest["artifacts"][name]
        assert artifact["path"] == str(path.resolve())
        assert artifact["sha256"] == alt._file_sha256(path)
    verified = alt.verify_panel_artifact(manifest_path)
    assert verified["state"] == "verified"
    assert verified["barsCount"] == 4
    assert verified["panelSha256"] == alt._file_sha256(panel_path)
    assert verified["reproduced"] is True
    assert alt.main(["verify-panel", "--manifest", str(manifest_path)]) == 0

    # Panel bytes are deterministic even though manifests have a generatedAt
    # timestamp and absolute output path.
    second_panel_path = root / "panel-second.csv"
    second_manifest_path = root / "panel-second.json"
    second_manifest = alt.build_panel(
        cache_path,
        bars_path,
        second_panel_path,
        interval_ms=60000,
        manifest_path=second_manifest_path,
        symbol="BTCUSDT",
    )
    assert second_panel_path.read_bytes() == panel_path.read_bytes()
    assert second_manifest["artifacts"]["panel"]["sha256"] == verified["panelSha256"]

    # A byte mutation fails the digest. Rehashing semantically invalid bytes
    # still cannot bypass coverage validation.
    original_panel = panel_path.read_bytes()
    original_manifest = manifest_path.read_text(encoding="utf-8")
    corrupt_rows = panel.copy()
    corrupt_rows[0][f"{delayed_family}_coverage"] = "2"
    with panel_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=alt._panel_fields())
        writer.writeheader()
        writer.writerows(corrupt_rows)
    try:
        alt.verify_panel_artifact(manifest_path)
    except ValueError as error:
        assert "panel artifact sha256 mismatch" in str(error)
    else:
        raise AssertionError("a panel byte mutation must fail its artifact digest")
    rehashed_manifest = json.loads(original_manifest)
    rehashed_manifest["artifacts"]["panel"]["sha256"] = alt._file_sha256(panel_path)
    manifest_path.write_text(json.dumps(rehashed_manifest), encoding="utf-8")
    try:
        alt.verify_panel_artifact(manifest_path)
    except ValueError as error:
        assert "coverage is outside [0, 1]" in str(error)
    else:
        raise AssertionError("rehashing cannot authorize invalid panel semantics")
    corrupt_rows[0][f"{delayed_family}_coverage"] = "0"
    corrupt_rows[0][delayed_family] = "1"
    with panel_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=alt._panel_fields())
        writer.writeheader()
        writer.writerows(corrupt_rows)
    rehashed_manifest["artifacts"]["panel"]["sha256"] = alt._file_sha256(panel_path)
    manifest_path.write_text(json.dumps(rehashed_manifest), encoding="utf-8")
    try:
        alt.verify_panel_artifact(manifest_path)
    except ValueError as error:
        assert "actionable value without coverage" in str(error)
    else:
        raise AssertionError("zero coverage cannot authorize a non-zero feature")
    panel_path.write_bytes(original_panel)
    manifest_path.write_text(original_manifest, encoding="utf-8")

    with panel_path.open(newline="", encoding="utf-8") as handle:
        valid_but_changed_rows = list(csv.DictReader(handle))
    valid_but_changed_rows[0][flow_family] = str(
        float(valid_but_changed_rows[0][flow_family]) + 0.125
    )
    with panel_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=alt._panel_fields())
        writer.writeheader()
        writer.writerows(valid_but_changed_rows)
    rehashed_manifest = json.loads(original_manifest)
    rehashed_manifest["artifacts"]["panel"]["sha256"] = alt._file_sha256(panel_path)
    manifest_path.write_text(json.dumps(rehashed_manifest), encoding="utf-8")
    try:
        alt.verify_panel_artifact(manifest_path)
    except ValueError as error:
        assert "do not reproduce from the bound inputs" in str(error)
    else:
        raise AssertionError("a rehashed finite value must fail deterministic reconstruction")
    panel_path.write_bytes(original_panel)
    manifest_path.write_text(original_manifest, encoding="utf-8")

    incompatible_manifest = json.loads(original_manifest)
    incompatible_manifest["panelSchema"]["id"] = "unknown_panel_schema"
    manifest_path.write_text(json.dumps(incompatible_manifest), encoding="utf-8")
    try:
        alt.verify_panel_artifact(manifest_path)
    except ValueError as error:
        assert "unsupported alternative-data panel schema id" in str(error)
    else:
        raise AssertionError("an unsupported panel schema must fail closed")
    manifest_path.write_text(original_manifest, encoding="utf-8")

    inaccurate_manifest = json.loads(original_manifest)
    inaccurate_manifest["observationsCount"] += 1
    manifest_path.write_text(json.dumps(inaccurate_manifest), encoding="utf-8")
    try:
        alt.verify_panel_artifact(manifest_path)
    except ValueError as error:
        assert "observationsCount disagrees with bound cache" in str(error)
    else:
        raise AssertionError("manifest observation counts must be recomputed")
    inaccurate_manifest = json.loads(original_manifest)
    inaccurate_manifest["metricsByFamily"][flow_family] += 1
    manifest_path.write_text(json.dumps(inaccurate_manifest), encoding="utf-8")
    try:
        alt.verify_panel_artifact(manifest_path)
    except ValueError as error:
        assert "metricsByFamily disagrees with bound cache" in str(error)
    else:
        raise AssertionError("manifest family counts must be recomputed")
    manifest_path.write_text(original_manifest, encoding="utf-8")

    original_bars = bars_path.read_bytes()
    with bars_path.open(newline="", encoding="utf-8") as handle:
        corrupt_bars = list(csv.DictReader(handle))
    corrupt_bars[0]["openTime"] = "1"
    with bars_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["openTime", "close"])
        writer.writeheader()
        writer.writerows(corrupt_bars)
    rehashed_manifest = json.loads(original_manifest)
    rehashed_manifest["artifacts"]["bars"]["sha256"] = alt._file_sha256(bars_path)
    manifest_path.write_text(json.dumps(rehashed_manifest), encoding="utf-8")
    try:
        alt.verify_panel_artifact(manifest_path)
    except ValueError as error:
        assert "panel timestamps do not match the bound bar grid" in str(error)
    else:
        raise AssertionError("rehashing a mismatched bar grid must fail closed")
    bars_path.write_bytes(original_bars)
    manifest_path.write_text(original_manifest, encoding="utf-8")

    original_cache = cache_path.read_bytes()
    with cache_path.open(newline="", encoding="utf-8") as handle:
        corrupt_cache = list(csv.DictReader(handle))
    corrupt_cache[0]["eventTime"] = str(int(corrupt_cache[0]["timestamp"]) + 1)
    with cache_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=alt.CANONICAL_FIELDS)
        writer.writeheader()
        writer.writerows(corrupt_cache)
    rehashed_manifest = json.loads(original_manifest)
    rehashed_manifest["artifacts"]["cache"]["sha256"] = alt._file_sha256(cache_path)
    manifest_path.write_text(json.dumps(rehashed_manifest), encoding="utf-8")
    try:
        alt.verify_panel_artifact(manifest_path)
    except ValueError as error:
        assert "incoherent observation timestamps" in str(error)
    else:
        raise AssertionError("rehashing a time-incoherent cache must fail closed")
    cache_path.write_bytes(original_cache)
    manifest_path.write_text(original_manifest, encoding="utf-8")

    # A changed payload cannot reuse an explicit provider release identity:
    # preserving the original row prevents the correction from being backdated.
    original = rows[0]
    replacement = alt.Observation(**{
        **original.__dict__,
        "value": original.value + 10,
        "ingestedAt": original.ingestedAt + 1,
    })
    try:
        alt.merge_cache(cache_path, [replacement])
    except ValueError as error:
        assert "distinct revision or availability timestamp" in str(error)
    else:
        raise AssertionError("ambiguous explicit correction must fail closed")
    preserved = {row.key(): row for row in alt.read_cache(cache_path)}[original.key()]
    assert preserved.value == original.value
    distinct_release = alt.Observation(**{
        **replacement.__dict__,
        "timestamp": replacement.timestamp + 1,
    })
    assert alt.merge_cache(cache_path, [distinct_release]) == len(alt.FAMILIES) + 1
    released = {row.key(): row for row in alt.read_cache(cache_path)}[distinct_release.key()]
    assert released.value == distinct_release.value

    # A provider without release/vintage timestamps becomes available only
    # when first observed. Later revisions cannot inherit the historical event
    # time, while unchanged re-fetches preserve the original first-seen row.
    revision_feed = root / "revision.csv"
    revision_epoch = 1700000000000
    def write_revision(value):
        with revision_feed.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["eventTime", "value"])
            writer.writeheader()
            writer.writerow({"eventTime": revision_epoch, "value": value})
    revision_config_path = root / "revision-config.json"
    revision_cache_path = root / "revision-observations.csv"
    revision_config_path.write_text(json.dumps({
        "schemaVersion": 1,
        "cache": str(revision_cache_path),
        "sources": [{
            "id": "revision-provider",
            "family": "macro",
            "adapter": "csv",
            "location": str(revision_feed),
            "eventTimeField": "eventTime",
            "valueField": "value",
            "metric": "revised_history",
            "aggregation": "last",
            "transform": "raw",
            "minHistory": 1,
        }],
    }), encoding="utf-8")
    revision_config = alt.load_config(revision_config_path)
    revision_source = revision_config.sources[0]
    write_revision(1)
    first_seen = alt.fetch_source(revision_config, revision_source, now_ms=revision_epoch + 60000)
    assert first_seen[0].timestamp == revision_epoch + 60000
    assert first_seen[0].availabilityMode == "first_seen"
    assert alt.merge_cache(revision_cache_path, first_seen) == 1
    write_revision(2)
    revised = alt.fetch_source(revision_config, revision_source, now_ms=revision_epoch + 180000)
    assert revised[0].timestamp == revision_epoch + 180000
    assert alt.merge_cache(revision_cache_path, revised) == 2
    unchanged = alt.fetch_source(revision_config, revision_source, now_ms=revision_epoch + 240000)
    assert alt.merge_cache(revision_cache_path, unchanged) == 2
    write_revision(1)
    reverted = alt.fetch_source(revision_config, revision_source, now_ms=revision_epoch + 300000)
    assert alt.merge_cache(revision_cache_path, reverted) == 3
    revision_rows = alt.read_cache(revision_cache_path)
    assert [(row.timestamp, row.value) for row in revision_rows] == [
        (revision_epoch + 60000, 1),
        (revision_epoch + 180000, 2),
        (revision_epoch + 300000, 1),
    ]
    direct_values, direct_present = alt._metric_values_for_bars(
        revision_rows,
        [revision_epoch + offset for offset in [59999, 119999, 179999, 239999, 299999, 359999]],
    )
    assert direct_values == [None, 1, 1, 2, 2, 1], direct_values
    assert direct_present == [False, True, True, True, True, True]
    revision_bars_path = root / "revision-bars.csv"
    with revision_bars_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["openTime", "close"])
        writer.writeheader()
        for open_time in [revision_epoch + offset for offset in [0, 60000, 120000, 180000, 240000, 300000]]:
            writer.writerow({"openTime": open_time, "close": 100})
    revision_panel_path = root / "revision-panel.csv"
    alt.build_panel(
        revision_cache_path,
        revision_bars_path,
        revision_panel_path,
        interval_ms=60000,
        symbol="BTCUSDT",
    )
    with revision_panel_path.open(newline="", encoding="utf-8") as handle:
        revision_panel = list(csv.DictReader(handle))
    revision_values = [float(row["macro"]) for row in revision_panel]
    assert revision_values == [0, 1, 1, 2, 2, 1], revision_values

    secret_error = urllib.error.HTTPError(
        "https://provider.invalid/data?api_key=do-not-log",
        401,
        "Unauthorized",
        None,
        None,
    )
    assert "do-not-log" not in alt._safe_error(secret_error)

    # A z-score at an update uses only earlier releases, not future values or
    # repeated hourly forward fills.
    zrows = [
        alt.Observation(
            timestamp=index * 60000,
            eventTime=index * 60000,
            source="z",
            family="onchain",
            metric="activity",
            entity="BTC",
            value=value,
            unit="",
            revision="",
            ingestedAt=1,
            aggregation="last",
            transform="zscore",
            polarity=1,
            maxAgeMs=None,
            minHistory=2,
        )
        for index, value in enumerate([1.0, 2.0, 4.0])
    ]
    values, present = alt._metric_values_for_bars(zrows, [59999, 119999, 179999, 239999])
    assert values[:2] == [0, 0]
    assert values[2] > 3
    assert values[3] == values[2]
    assert all(present)

    # Empty mean buckets remain missing through transformation: they neither
    # manufacture a signal nor reset the previous observed value.
    mean_rows = [
        alt.Observation(
            timestamp=timestamp,
            eventTime=timestamp,
            source="mean",
            family="news",
            metric="sentiment",
            entity="BTC",
            value=value,
            unit="",
            revision="",
            ingestedAt=1,
            aggregation="mean",
            transform="delta",
            polarity=1,
            maxAgeMs=None,
            minHistory=1,
        )
        for timestamp, value in [(1000, 1.0), (121000, 3.0)]
    ]
    mean_values, mean_present = alt._metric_values_for_bars(
        mean_rows,
        [59999, 119999, 179999],
        [0, 60000, 120000],
    )
    assert mean_values == [0, None, 2], mean_values
    assert mean_present == [True, False, True]
`;

  const result = spawnSync("python3", ["-c", program, RESEARCH_DIR], {
    encoding: "utf8",
  });
  assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
});

test("alternative data config rejects unknown families and implicit field values", () => {
  const program = String.raw`
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, sys.argv[1])
import alternative_data as alt

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    for source, expected in [
        ({"id": "bad", "family": "unknown", "adapter": "csv", "location": "x", "metric": "x", "valueField": "x"}, "unknown family"),
        ({"id": "bad", "family": "macro", "adapter": "csv", "location": "x", "metric": "x"}, "valueField is required"),
    ]:
        path = root / "bad.json"
        path.write_text(json.dumps({"schemaVersion": 1, "sources": [source]}), encoding="utf-8")
        try:
            alt.load_config(path)
        except alt.ConfigError as error:
            assert expected in str(error)
        else:
            raise AssertionError("invalid source configuration must fail closed")
`;
  const result = spawnSync("python3", ["-c", program, RESEARCH_DIR], {
    encoding: "utf8",
  });
  assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
});

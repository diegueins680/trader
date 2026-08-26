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
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["intervalMs"] == 60000

    # Identical PIT keys are idempotent; a later collection replaces only the
    # ingestion copy rather than manufacturing another historical observation.
    original = rows[0]
    replacement = alt.Observation(**{
        **original.__dict__,
        "value": original.value + 10,
        "ingestedAt": original.ingestedAt + 1,
    })
    assert alt.merge_cache(cache_path, [replacement]) == len(alt.FAMILIES)
    replaced = {row.key(): row for row in alt.read_cache(cache_path)}[replacement.key()]
    assert replaced.value == replacement.value

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

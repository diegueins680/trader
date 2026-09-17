#!/usr/bin/env python3
"""Explicit opt-in runner restricted to already contaminated registered data."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import time
import numpy as np
import pandas as pd

from sequential_env import Execution, Scale
from sequential_learning import infer, save_policy, train_ppo, train_q
from sequential_evaluation import Baselines, replay_policy, short_ope

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = ROOT / "research-notes/registrations/sequential-control-screen-v1.json"
ALGORITHMS = ("ppo", "double_dqn", "cql", "cql_no_inventory_penalty")
SOURCES = [Path(__file__), *[Path(__file__).with_name(n) for n in
           ("sequential_env.py", "sequential_learning.py", "sequential_evaluation.py")], REGISTRATION]
STRESSES = {"base": Execution(), "cost1_5x": Execution(cost_multiplier=1.5),
            "cost2x": Execution(cost_multiplier=2), "extreme25bp": Execution(cost_multiplier=2.5),
            "delay1bar": Execution(extra_delay=1), "partial50pct": Execution(fill_fraction=0.5),
            "missed10pct": Execution(miss_every=10), "impact10bp": Execution(impact_bps=10),
            "funding2x": Execution(funding_multiplier=2)}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False, indent=2)
        stream.write("\n")


def source_commit() -> str:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for path in SOURCES:
        committed = subprocess.check_output(["git", "show", f"{commit}:{path.relative_to(ROOT)}"], cwd=ROOT)
        if committed != path.read_bytes():
            raise ValueError(f"uncommitted experiment source: {path.name}")
    return commit


def load_development(panel: Path, settlements: Path, registration: dict):
    spec = registration["data"]
    # Verify both files before parsing any value; no access to full source snapshots.
    if digest(panel) != spec["panelSha256"] or digest(settlements) != spec["settlementsSha256"]:
        raise ValueError("only exact registered development bytes are permitted")
    bars = pd.read_csv(panel)
    events = pd.read_csv(settlements)
    if list(bars.columns) != ["symbol", "openTime", "closeTime", "close"]:
        raise ValueError("panel schema")
    if sorted(bars.symbol.unique()) != spec["symbols"] or sorted(events.symbol.unique()) != spec["symbols"]:
        raise ValueError("symbol coverage")
    prices, funding = {}, {}
    times = np.arange(spec["startOpenTime"], spec["endOpenTime"] + 1, spec["intervalMilliseconds"], dtype=np.int64)
    for symbol in spec["symbols"]:
        rows = bars[bars.symbol == symbol]
        if len(rows) != spec["rowsPerSymbol"] or not np.array_equal(rows.openTime, times):
            raise ValueError("registered grid")
        closes = times + spec["intervalMilliseconds"] - 1
        if not np.array_equal(rows.closeTime, closes):
            raise ValueError("close timestamps")
        p = rows.close.to_numpy(dtype=float)
        if not np.isfinite(p).all() or np.any(p <= 0):
            raise ValueError("invalid price")
        f = np.zeros(len(p))
        es = events[events.symbol == symbol]
        if not np.isfinite(es[["fundingTime", "fundingRate", "resolvedMarkPrice"]].to_numpy()).all():
            raise ValueError("invalid funding")
        if np.any(es.resolvedMarkPrice <= 0) or np.any(np.diff(es.fundingTime) <= 0):
            raise ValueError("funding identity/order")
        # Settlement exactly at a close belongs to that endpoint. Before the
        # first close there is no held inventory and those events are not used.
        for event in es.itertuples():
            j = int(np.searchsorted(closes, event.fundingTime, side="left"))
            if j >= len(f):
                raise ValueError("funding beyond development")
            f[j] += event.fundingRate * event.resolvedMarkPrice
        p.setflags(write=False); f.setflags(write=False)
        prices[symbol], funding[symbol] = p, f
    return prices, funding, times


def summary(records: list[dict]) -> list[dict]:
    groups = {}
    for r in records:
        key = (r["algorithm"], r["horizon"], r["seed"], r["stress"])
        groups.setdefault(key, []).append(r)
    out = []
    for (algorithm, horizon, seed, stress), rows in groups.items():
        complete = [r for r in rows if r["result"].get("status") == "complete"]
        finite_rows = [r["result"] for r in rows if "netReturn" in r["result"]]
        out.append({"algorithm": algorithm, "horizon": horizon, "seed": seed, "stress": stress,
                    "paths": len(rows), "completePaths": len(complete),
                    "failedPaths": len(rows) - len(complete),
                    "failureRate": (len(rows) - len(complete)) / len(rows),
                    "meanTerminalOrStoppedReturn": float(np.mean([r["netReturn"] for r in finite_rows])) if finite_rows else None,
                    "worstTerminalOrStoppedReturn": min((r["netReturn"] for r in finite_rows), default=None),
                    "worstDrawdown": max((r["maxDrawdown"] for r in finite_rows), default=None),
                    "worstES95": max((r["expectedShortfall95"] for r in finite_rows), default=None),
                    "medianPathSharpe": float(np.median([r["sharpe"] for r in finite_rows if r["sharpe"] is not None])) if any(r["sharpe"] is not None for r in finite_rows) else None,
                    "meanFees": float(np.mean([r["costsOverInitialEquity"]["fee"] for r in finite_rows])) if finite_rows else None,
                    "meanFunding": float(np.mean([r["fundingPnlOverInitialEquity"] for r in finite_rows])) if finite_rows else None,
                    "comparisonWarning": "Stopped paths have different endpoints; these descriptive aggregates cannot establish relative economic superiority."})
    return out


def run(panel: Path, settlements: Path, output: Path) -> None:
    commit = source_commit()
    registration = json.loads(REGISTRATION.read_text())
    prices, funding, times = load_development(panel, settlements, registration)
    output.mkdir(parents=True, exist_ok=False)
    (output / "policies").mkdir()
    started = time.perf_counter()
    write_json(output / "manifest.json", {"campaign": registration["campaign"], "codeCommit": commit,
               "registrationCommit": "dfc6b27d", "registrationSha256": digest(REGISTRATION),
               "sources": {str(p.relative_to(ROOT)): digest(p) for p in SOURCES},
               "panelSha256": digest(panel), "settlementsSha256": digest(settlements),
               "firstOpenUtc": datetime.fromtimestamp(int(times[0])/1000, timezone.utc).isoformat(),
               "lastOpenUtc": datetime.fromtimestamp(int(times[-1])/1000, timezone.utc).isoformat(),
               "createdAtUtc": datetime.now(timezone.utc).isoformat(),
               "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
               "holdoutOpened": False, "liveAuthorization": False, "networkUsed": False,
               "evidenceClass": "contaminated_development_only", "promotionAllowed": False})
    records, training, ope = [], [], []
    # Complete planned roster is persisted before the first training or evaluation.
    planned = []
    for h in registration["data"]["decisionHorizonBars"]:
        for fold in range(3):
            for alg in (*ALGORITHMS, *Baselines.names):
                seeds = registration["seeds"] if alg in ALGORITHMS else [20260917]
                for seed in seeds:
                    trial = f"{alg}/h{h}/f{fold}/s{seed}"
                    if alg in ALGORITHMS:
                        planned.append({"id": trial, "kind": "training", "status": "planned"})
                    for stress in STRESSES:
                        for symbol in sorted(prices):
                            planned.append({"id": f"{trial}/{stress}/{symbol}", "kind": "replay", "status": "planned"})
    write_json(output / "planned-registry.json", planned)
    with (output / "events.jsonl").open("x") as ledger, (output / "returns.csv").open("x") as paths:
        writer = csv.writer(paths)
        writer.writerow(["trial", "symbol", "outcomeIndex", "netReturn", "equity", "gross", "funding", "fee", "spread", "slippage", "impact", "exposure"])
        def event(value):
            ledger.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n"); ledger.flush()
        for h in registration["data"]["decisionHorizonBars"]:
            for fi, split in enumerate(registration["validation"]["outerFolds"]):
                train = {s: p[:split["trainStop"]] for s, p in prices.items()}
                funds = {s: p[:split["trainStop"]] for s, p in funding.items()}
                scale = Scale.fit(list(train.values()))
                controls = Baselines(train, funds, scale, h)
                for alg in (*ALGORITHMS, *Baselines.names):
                    for seed in (registration["seeds"] if alg in ALGORITHMS else [20260917]):
                        trial = f"{alg}/h{h}/f{fi}/s{seed}"
                        net = None
                        if alg in ALGORITHMS:
                            event({"id": trial, "status": "started"})
                            then = time.perf_counter()
                            try:
                                if alg == "ppo":
                                    net, info = train_ppo(train, funds, scale, h, seed)
                                else:
                                    net, info = train_q(train, funds, scale, h, seed, offline=alg != "double_dqn",
                                                       risk_penalty=0 if alg == "cql_no_inventory_penalty" else 0.01)
                                info.update({"id": trial, "seconds": time.perf_counter() - then,
                                             "algorithm": alg, "seed": seed, "fold": fi, "horizon": h})
                                provenance = {"codeCommit": commit, "registrationSha256": digest(REGISTRATION),
                                              "dataSha256": digest(panel), "seed": seed, "horizon": h,
                                              "algorithm": alg, "fold": fi, "split": split,
                                              "fundingSha256": digest(settlements), "scale": {k: getattr(scale, k).tolist() for k in ("mean", "std", "low", "high")}}
                                artifact = output / "policies" / (trial.replace("/", "_") + ".json")
                                info["artifactSha256"] = save_policy(artifact, net, provenance)
                                info["artifactBytes"] = artifact.stat().st_size
                                training.append(info)
                                event({"id": trial, "status": "complete", "seconds": info["seconds"], "artifactSha256": info["artifactSha256"]})
                                try:
                                    ope_result = short_ope(prices, funding, scale, h, split["testStart"], split["testStop"], net, seed)
                                except Exception as exc:
                                    ope_result = {"status": "failed", "reason": str(exc)}
                                ope.append({"id": trial, "result": ope_result})
                            except Exception as exc:
                                event({"id": trial, "status": "failed", "reason": type(exc).__name__ + ": " + str(exc)})
                                training.append({"id": trial, "status": "failed", "reason": str(exc)})
                                continue
                        for stress, cfg in STRESSES.items():
                            if alg == "cql_no_inventory_penalty":
                                cfg = replace(cfg, risk_penalty=0)
                            for symbol in sorted(prices):
                                replay_id = f"{trial}/{stress}/{symbol}"
                                event({"id": replay_id, "status": "started"})
                                rng = np.random.default_rng(seed)
                                def choose(obs):
                                    if net is not None:
                                        return infer(net, obs, enabled=True)
                                    tick = time.perf_counter_ns()
                                    result = controls.action(alg, obs, rng)
                                    return result, (time.perf_counter_ns() - tick) / 1e6
                                try:
                                    env, result = replay_policy(prices[symbol], funding[symbol], split["testStart"], split["testStop"], h, scale, choose, cfg)
                                    records.append({"id": replay_id, "algorithm": alg, "seed": seed, "fold": fi,
                                                    "horizon": h, "symbol": symbol, "stress": stress, "result": result})
                                    for row in env.rows:
                                        writer.writerow([trial + "/" + stress, symbol, row["t"], *[row[k] for k in ("net", "equity", "gross", "funding", "fee", "spread", "slippage", "impact", "exposure")]])
                                    event({"id": replay_id, "status": result["status"], "reason": result.get("reason"), "observations": result["observations"]})
                                except Exception as exc:
                                    result = {"status": "failed", "reason": str(exc), "observations": 0}
                                    records.append({"id": replay_id, "algorithm": alg, "seed": seed, "fold": fi, "horizon": h, "symbol": symbol, "stress": stress, "result": result})
                                    event({"id": replay_id, **result})
                        print(f"completed {trial}", flush=True)
    write_json(output / "training.json", training)
    write_json(output / "evaluation.json", records)
    write_json(output / "ope.json", ope)
    write_json(output / "summary.json", {"decision": "no_candidate_passed", "promotionAllowed": False,
               "holdoutOpened": False, "evidenceClass": "contaminated_development_only",
               "trainingFits": len(training), "replayPaths": len(records), "plannedEntries": len(planned),
               "seconds": time.perf_counter() - started,
               "processPeakRssPlatformUnits": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "statistics": {"DSR": None, "PBO": None, "SPA": None, "pairedConfidence": None,
                   "reason": "No promotion inference: contaminated panel, no matched champion, constant cash baseline and possibly stopped/incomplete paths. Do not discard required paths to manufacture a complete selection matrix."},
               "groups": summary(records)})
    files = sorted(p for p in output.rglob("*") if p.is_file())
    write_json(output / "evidence-index.json", {str(p.relative_to(output)): digest(p) for p in files})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-registered-development-v1", action="store_true", required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--settlements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.panel, args.settlements, args.output)

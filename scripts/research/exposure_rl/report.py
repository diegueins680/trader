"""Aggregate a registered run into the committed evidence tables.

Reads the job records and per-bar series of one run directory and writes
results.json, all-seed-results.csv, experiment-registry.csv and
comparison-report.md. Nothing is selected by test outcome here: every seed,
fold, policy and failure is reported, and learned policies are summarised by
the mean of their seeds' per-bar returns as registered.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import diagnostics  # noqa: E402
from exposure_rl import evaluate as V  # noqa: E402
from exposure_rl.env import STRESSES  # noqa: E402

RL = ("ppo", "td3", "iql")
EXTENSIONS = ("ppo_risk", "ppo_no_context")
LEARNED_NON_RL = ("behavior_cloning", "contextual_bandit")
DETERMINISTIC = ("cash", "buy_and_hold", "regime_rule", "ridge_forecast", "champion_method")
GATE_COMPARATORS = ("cash", "buy_and_hold", "regime_rule", "ridge_forecast", "champion_method")
ALL = RL + EXTENSIONS + LEARNED_NON_RL + DETERMINISTIC
BENCHMARK = "BTCUSDT"


def load_run(run_dir: Path):
    jobs = [json.loads(path.read_text()) for path in sorted((run_dir / "jobs").glob("*.json")) if not path.name.endswith(".policy.json")]
    series: dict = {}  # (policy, stress) -> fold -> seed -> (times, returns)
    symbols: dict = {}  # policy -> symbol -> fold -> seed -> (times, returns)
    for job in jobs:
        if job["status"] != "completed":
            continue
        with np.load(run_dir / "jobs" / f"{job['id']}.npz") as archive:
            for key in archive.files:
                parts = key.split("__")
                if parts[-1] == "fleet":
                    policy, stress = parts[0], parts[1]
                    series.setdefault((policy, stress), {}).setdefault(job["fold"], {})[job["seed"]] = (archive[f"{policy}__{stress}__times"], archive[key])
                elif len(parts) == 4 and parts[1] == "base" and parts[3] == "net":
                    policy, symbol = parts[0], parts[2]
                    symbols.setdefault(policy, {}).setdefault(symbol, {}).setdefault(job["fold"], {})[job["seed"]] = (archive[f"{policy}__base__{symbol}__open_time"], archive[key])
    return jobs, series, symbols


def pooled(folds: dict, seed=None):
    """Chronological concatenation over folds of one seed or of the seed mean."""
    times, values = [], []
    for fold in sorted(folds):
        runs = folds[fold]
        chosen = [runs[seed]] if seed is not None and seed in runs else list(runs.values()) if seed is None else []
        if not chosen:
            return None
        grid = chosen[0][0]
        if any(len(t) != len(grid) or np.any(t != grid) for t, _ in chosen):
            raise ValueError("seed series are not aligned")
        times.append(grid)
        values.append(np.mean([v for _, v in chosen], axis=0))
    return np.concatenate(times), np.concatenate(values)


def seeds_of(folds: dict) -> list:
    return sorted({seed for runs in folds.values() for seed in runs}, key=lambda s: (s is None, s))


def summarise(run_dir: Path) -> dict:
    jobs, series, symbols = load_run(run_dir)
    planned = json.loads((run_dir / "planned-registry.json").read_text())
    out: dict = {
        "campaign": planned["campaign"],
        "registrationSha256": planned["registrationSha256"],
        "source": planned["source"],
        "run": run_dir.name,
        "jobs": {"planned": len(planned["jobs"]), "completed": sum(j["status"] == "completed" for j in jobs), "failed": [{"id": j["id"], "error": j.get("error")} for j in jobs if j["status"] != "completed"], "cpuSecondsByKind": {}},
        "policies": {},
    }
    for job in jobs:
        out["jobs"]["cpuSecondsByKind"][job["kind"]] = round(out["jobs"]["cpuSecondsByKind"].get(job["kind"], 0.0) + job.get("seconds", 0.0), 1)
    base = {}
    for policy in ALL:
        if (policy, "base") not in series:
            continue
        entry: dict = {"stress": {}, "perSeed": {}, "perFold": {}, "perSymbol": {}}
        for stress in list(STRESSES) + ["unshielded_diagnostic"]:
            if (policy, stress) in series:
                times, values = pooled(series[(policy, stress)])
                entry["stress"][stress] = V.metrics(values)
                if stress == "base":
                    base[policy] = (times, values)
        folds = series[(policy, "base")]
        for seed in seeds_of(folds):
            one = pooled(folds, seed)
            if one is not None:
                entry["perSeed"][str(seed)] = V.metrics(one[1])
        for fold in sorted(folds):
            entry["perFold"][str(fold)] = V.metrics(np.mean([v for _, v in folds[fold].values()], axis=0))
        for symbol, by_fold in symbols.get(policy, {}).items():
            one = pooled(by_fold)
            entry["perSymbol"][symbol] = V.metrics(one[1])
        out["policies"][policy] = entry
    # Activity, shield interventions, checkpoints and OPE from the job summaries.
    activity: dict = {}
    for job in jobs:
        for policy, record in job.get("policies", {}).items():
            row = activity.setdefault(policy, {"activity": [], "selected": [], "ope": []})
            row["activity"].append(record["test"]["base"]["activity"])
            if record.get("selectedStep") is not None:
                row["selected"].append(record["selectedStep"])
            if "ope" in record:
                row["ope"].append(record["ope"])
    for policy, row in activity.items():
        if policy not in out["policies"]:
            continue
        keys = [k for k, v in row["activity"][0].items() if isinstance(v, (int, float)) and v is not None]
        out["policies"][policy]["activity"] = {k: float(np.mean([a[k] for a in row["activity"] if a.get(k) is not None])) for k in keys}
        out["policies"][policy]["activity"]["maxAbsExposure"] = float(max(a["maxAbsExposure"] for a in row["activity"]))
        if row["selected"]:
            out["policies"][policy]["selectedCheckpointSteps"] = sorted(row["selected"])
        valid = [o for o in row["ope"] if "fqeMean" in o]
        if row["ope"]:
            ope = {"batches": len(row["ope"]), "valid": len(valid)}
            if valid:
                ope.update({k: float(np.mean([o[k] for o in valid])) for k in ("fqeMean", "simulatorTruthMean", "meanError", "rmse", "rankCorrelation", "truthStd")})
                for h in ("1", "4", "16", "64", "256"):
                    fractions = [o["importanceSamplingEssByHorizon"][h].get("fraction") for o in valid]
                    ope[f"essFractionH{h}"] = None if any(f is None for f in fractions) else float(np.mean(fractions))
            out["policies"][policy]["ope"] = ope
    # Paired comparisons on the common time grid.
    out["paired"] = {}
    for candidate in RL + EXTENSIONS:
        if candidate not in base:
            continue
        for comparator in GATE_COMPARATORS + LEARNED_NON_RL:
            if comparator not in base:
                continue
            for stress in ("base", "cost2x", "delay1bar"):
                a, b = pooled(series[(candidate, stress)]), pooled(series[(comparator, stress)])
                if len(a[0]) != len(b[0]) or np.any(a[0] != b[0]):
                    raise ValueError("candidate and comparator grids differ")
                out["paired"].setdefault(candidate, {}).setdefault(comparator, {})[stress] = V.paired_difference(a[1], b[1], seed=20261111)
        # Per-seed paired differences against the best comparator by annualised mean.
    # Multiple-testing diagnostics over every evaluated variant.
    names = [p for p in ALL if p in base and p != "cash"]
    frame = pd.DataFrame({name: base[name][1] for name in names})
    out["multipleTesting"] = {"variants": names, "observations": int(len(frame))}
    for candidate in RL + EXTENSIONS:
        if candidate not in frame:
            continue
        try:
            result = diagnostics.deflated_sharpe_ratio(frame, selected_trial=candidate, periods_per_year=V.BARS_PER_YEAR)
            out["multipleTesting"].setdefault("dsr", {})[candidate] = {"probability": result.probability, "selectedSharpeAnnualized": result.selected_sharpe_annualized, "benchmarkSharpeAnnualized": result.benchmark_sharpe_annualized, "trials": result.trials, "independentTrials": result.independent_trials}
        except Exception as error:  # an undefined statistic is reported, not replaced
            out["multipleTesting"].setdefault("dsr", {})[candidate] = {"status": "insufficient evidence", "reason": f"{type(error).__name__}: {error}"}
    try:
        trimmed = frame.iloc[len(frame) % 16 :].reset_index(drop=True)
        pbo = diagnostics.cscv_pbo(trimmed, n_slices=16)
        out["multipleTesting"]["pbo"] = {"probability": pbo.probability if hasattr(pbo, "probability") else pbo.pbo, "observationsUsed": int(len(trimmed)), "slices": 16}
    except Exception as error:
        out["multipleTesting"]["pbo"] = {"status": "insufficient evidence", "reason": f"{type(error).__name__}: {error}"}
    # Benchmark-only symbol.
    out["benchmark"] = {policy: out["policies"][policy]["perSymbol"].get(BENCHMARK) for policy in out["policies"]}
    return out


def gates(summary: dict, registration: dict) -> dict:
    """Evaluate the registered development gates for each RL learner and extension."""
    materiality = registration["metrics"]["materiality"]
    margin = registration["metrics"]["nonInferiority"]["maxDrawdownExcessOverBestComparator"]
    result = {}
    for candidate in RL + EXTENSIONS:
        if candidate not in summary["paired"]:
            continue
        paired, checks = summary["paired"][candidate], {}
        comparators = [c for c in GATE_COMPARATORS if c in paired]
        checks["allComparatorsAvailable"] = len(comparators) == len(GATE_COMPARATORS)
        checks["lowerBoundPositive"] = all(paired[c]["base"]["ci95"][0] > 0 for c in comparators)
        checks["materialPointEstimate"] = all(paired[c]["base"]["annualizedDifference"] >= materiality for c in comparators)
        checks["positiveAtCost2x"] = all(paired[c]["cost2x"]["annualizedDifference"] > 0 for c in comparators)
        checks["positiveAtDelay1bar"] = all(paired[c]["delay1bar"]["annualizedDifference"] > 0 for c in comparators)
        best_drawdown = min(summary["policies"][c]["stress"]["base"]["maxDrawdown"] for c in comparators if c != "cash")
        checks["drawdownNonInferior"] = summary["policies"][candidate]["stress"]["base"]["maxDrawdown"] <= best_drawdown + margin
        dsr = summary["multipleTesting"].get("dsr", {}).get(candidate, {})
        checks["dsr"] = dsr.get("probability") is not None and dsr["probability"] >= 0.95
        pbo = summary["multipleTesting"].get("pbo", {})
        checks["pbo"] = pbo.get("probability") is not None and pbo["probability"] <= 0.20
        best = max((c for c in comparators), key=lambda c: summary["policies"][c]["stress"]["base"]["annualizedMean"])
        seeds = summary["policies"][candidate]["perSeed"]
        positive = sum(1 for m in seeds.values() if m["annualizedMean"] > summary["policies"][best]["stress"]["base"]["annualizedMean"])
        checks["seedsPositiveVsBestComparator"] = positive >= 4
        result[candidate] = {"checks": checks, "bestComparator": best, "seedsAboveBestComparator": positive, "seeds": len(seeds), "passesDevelopmentGates": all(checks.values())}
    return result


def fmt(value, digits=3, pct=False):
    if value is None:
        return "n/a"
    return f"{100 * value:.1f}%" if pct else f"{value:.{digits}f}"


def write_outputs(summary: dict, run_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    jobs, _, _ = load_run(run_dir)
    with (out_dir / "experiment-registry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["id", "kind", "fold", "seed", "status", "seconds", "error"])
        for job in sorted(jobs, key=lambda j: j["id"]):
            writer.writerow([job["id"], job["kind"], job["fold"], job.get("seed"), job["status"], job.get("seconds"), job.get("error") or ""])
    with (out_dir / "all-seed-results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["policy", "fold", "seed", "stress", "bars", "totalReturn", "annualizedMean", "annualizedVol", "sharpe", "maxDrawdown", "expectedShortfall95", "annualTurnover", "meanAbsExposure", "projectedRate", "costPaid", "fundingPaid"])
        for job in sorted(jobs, key=lambda j: j["id"]):
            for policy, record in sorted(job.get("policies", {}).items()):
                for stress, cell in sorted(record["test"].items()):
                    m, a = cell["fleet"], cell["activity"]
                    writer.writerow([policy, job["fold"], job.get("seed"), stress, m.get("bars"), m.get("totalReturn"), m.get("annualizedMean"), m.get("annualizedVol"), m.get("sharpe"), m.get("maxDrawdown"), m.get("expectedShortfall95"), a["annualTurnover"], a["meanAbsExposure"], a["projectedRate"], a["costPaid"], a["fundingPaid"]])
    lines = [f"# Development comparison: run `{summary['run']}`", "", "Generated by `scripts/research/exposure_rl/report.py`. Fleet aggregate of six symbols, four walk-forward test folds pooled, base cost unless stated. Learned policies are the mean of five seeds' per-bar returns. Development data only (previously exposed).", ""]
    lines += ["## Pooled out-of-sample results", "", "| Policy | Ann. mean | Ann. vol | Sharpe | Max DD | ES95 (bar) | Turnover/yr | Mean abs exposure | Sharpe 2x cost | Sharpe 1-bar delay | Sharpe severe |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for policy in ALL:
        if policy not in summary["policies"]:
            continue
        p = summary["policies"][policy]
        b, act = p["stress"]["base"], p.get("activity", {})
        lines.append(f"| {policy} | {fmt(b.get('annualizedMean'), pct=True)} | {fmt(b.get('annualizedVol'), pct=True)} | {fmt(b.get('sharpe'), 2)} | {fmt(b.get('maxDrawdown'), pct=True)} | {fmt(b.get('expectedShortfall95'), pct=True)} | {fmt(act.get('annualTurnover'), 1)} | {fmt(act.get('meanAbsExposure'), pct=True)} | {fmt(p['stress'].get('cost2x', {}).get('sharpe'), 2)} | {fmt(p['stress'].get('delay1bar', {}).get('sharpe'), 2)} | {fmt(p['stress'].get('severe25bp_delay', {}).get('sharpe'), 2)} |")
    lines += ["", "## Paired annualised return difference (candidate minus comparator), 95% stationary bootstrap", "", "| Candidate | Comparator | Base | 95% interval | 2x cost | 1-bar delay |", "|---|---|---|---|---|---|"]
    for candidate, rows in summary["paired"].items():
        for comparator, cells in rows.items():
            b = cells["base"]
            lines.append(f"| {candidate} | {comparator} | {fmt(b['annualizedDifference'], pct=True)} | [{fmt(b['ci95'][0], pct=True)}, {fmt(b['ci95'][1], pct=True)}] | {fmt(cells['cost2x']['annualizedDifference'], pct=True)} | {fmt(cells['delay1bar']['annualizedDifference'], pct=True)} |")
    lines += ["", "## Every seed (pooled folds, base cost)", "", "| Policy | Seed | Ann. mean | Sharpe | Max DD |", "|---|---|---|---|---|"]
    for policy in RL + EXTENSIONS + LEARNED_NON_RL:
        for seed, m in summary["policies"].get(policy, {}).get("perSeed", {}).items():
            lines.append(f"| {policy} | {seed} | {fmt(m.get('annualizedMean'), pct=True)} | {fmt(m.get('sharpe'), 2)} | {fmt(m.get('maxDrawdown'), pct=True)} |")
    lines += ["", "## By fold (seed mean, base cost): annualised mean / Sharpe", "", "| Policy | Fold 1 | Fold 2 | Fold 3 | Fold 4 |", "|---|---|---|---|---|"]
    for policy in ALL:
        folds = summary["policies"].get(policy, {}).get("perFold")
        if folds:
            lines.append(f"| {policy} | " + " | ".join(f"{fmt(folds[f].get('annualizedMean'), pct=True)} / {fmt(folds[f].get('sharpe'), 2)}" if f in folds else "n/a" for f in "1234") + " |")
    lines += ["", "## By symbol (pooled folds, seed mean, base cost): total return of the symbol account", ""]
    names = sorted({s for p in summary["policies"].values() for s in p["perSymbol"]})
    lines += ["| Policy | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for policy in ALL:
        if policy in summary["policies"]:
            lines.append(f"| {policy} | " + " | ".join(fmt(summary["policies"][policy]["perSymbol"].get(s, {}).get("totalReturn"), pct=True) for s in names) + " |")
    lines += ["", "## Shield interventions (mean over folds and seeds)", "", "| Policy | Permitted | Abstain | Rejected | Recovery | Projected | Reduce-only | Max abs exposure |", "|---|---|---|---|---|---|---|---|"]
    for policy in ALL:
        act = summary["policies"].get(policy, {}).get("activity")
        if act:
            lines.append(f"| {policy} | " + " | ".join(fmt(act[k], pct=True) for k in ("permittedRate", "abstainRate", "rejectedRate", "recoveryRate", "projectedRate", "riskOffRate", "maxAbsExposure")) + " |")
    (out_dir / "comparison-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    run_dir = ROOT / ".tmp" / "research" / "exposure-rl-v1" / "runs" / args.run
    summary = summarise(run_dir)
    registration = json.loads((ROOT / "research-notes" / "registrations" / "exposure-rl-control-v1.json").read_text())
    summary["gates"] = gates(summary, registration)
    write_outputs(summary, run_dir, args.output)
    print(json.dumps({"gates": summary["gates"], "jobs": summary["jobs"]}, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

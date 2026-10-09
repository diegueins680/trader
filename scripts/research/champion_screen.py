#!/usr/bin/env python3
"""Adopted-champion screen: replay the frozen live fleet causally on lawful data.

Registration: research-notes/registrations/adopted-champion-screen-v1.json.

Each frozen combo is replayed with ``trader-hs --adopt-combo-file`` so the
backtest applies exactly the parameters, venue cost floors and adoption caps a
live bot would.  The evaluation window is cut into consecutive chunks; every
chunk is the backtest slice of a run whose input is the combo's own ``bars``
window ending at the chunk end, so the model is refit only on data strictly
before each chunk (the optimizer's own protocol, rolled forward).  Baselines
are computed on the identical rows with the identical per-side cost.

Subcommands:
  fetch  - download Binance USD-M klines and write a hash manifest
  run    - replay one registered phase and write per-bar returns + summary
  selftest - synthetic checks of chunking, chaining and baselines (no network)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = ROOT / "research-notes/registrations/adopted-champion-screen-v1.json"
FAPI_KLINES = "https://fapi.binance.com/fapi/v1/klines"
INTERVAL_MS = {"1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000, "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000}
CSV_FIELDS = ["openTimeMs", "open", "high", "low", "close", "volume", "closeTimeMs"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_registration(path: Path | None = None) -> dict:
    return json.loads((path or REGISTRATION).read_text())


# ---------------------------------------------------------------- data


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[list]:
    """All klines with start_ms <= openTime and closeTime < end_ms (closed bars only)."""
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        query = urllib.parse.urlencode({"symbol": symbol, "interval": interval, "startTime": cursor, "endTime": end_ms - 1, "limit": 1500})
        with urllib.request.urlopen(f"{FAPI_KLINES}?{query}", timeout=30) as resp:
            batch = json.loads(resp.read())
        if not batch:
            break
        rows.extend(k for k in batch if int(k[6]) < end_ms)
        nxt = int(batch[-1][0]) + INTERVAL_MS[interval]
        if nxt <= cursor:
            break
        cursor = nxt
        time.sleep(0.2)
    return rows


def check_contiguous(open_times: list[int], step: int) -> None:
    for a, b in zip(open_times, open_times[1:]):
        if b - a != step:
            raise ValueError(f"kline gap or duplicate between {a} and {b}")


def cmd_fetch(args: argparse.Namespace) -> None:
    reg = load_registration()
    data = reg["data"]
    out_dir = ROOT / data["directory"]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"kind": "adopted_champion_screen_data_v1", "source": FAPI_KLINES, "files": {}}
    end_ms = int(args.end_ms) if args.end_ms else int(data["fetchEndExclusiveMs"])
    for combo in reg["combos"]:
        sym, interval = combo["symbol"], combo["interval"]
        rows = fetch_klines(sym, interval, int(data["fetchStartMs"]), end_ms)
        check_contiguous([int(r[0]) for r in rows], INTERVAL_MS[interval])
        path = out_dir / f"{sym}-{interval}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_FIELDS)
            for r in rows:
                writer.writerow([r[0], r[1], r[2], r[3], r[4], r[5], r[6]])
        manifest["files"][path.name] = {"sha256": sha256_file(path), "rows": len(rows), "firstOpenMs": int(rows[0][0]), "lastOpenMs": int(rows[-1][0])}
        print(f"{path.name}: {len(rows)} rows", file=sys.stderr)
    (out_dir / args.manifest).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def read_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------- protocol


def chunk_plan(open_times: list[int], start_ms: int, end_ms: int, window: int, test_len: int) -> list[tuple[int, int]]:
    """Consecutive [s, e) row-index chunks covering every bar opening in [start_ms, end_ms).

    Each chunk is at most ``test_len`` bars.  Its run's backtest slice is rows
    ``s-1 .. e-1`` (the extra leading row anchors the first return), so the run
    input is rows ``e-window .. e-1`` and the model never sees a row >= s-1
    while fitting.
    """
    idx = [i for i, t in enumerate(open_times) if start_ms <= t < end_ms]
    if not idx:
        raise ValueError("evaluation window has no bars")
    first, last = idx[0], idx[-1] + 1
    chunks = []
    s = first
    while s < last:
        e = min(s + test_len, last)
        if e - window < 0 or (e - s) + 1 >= window:
            raise ValueError("not enough history before the evaluation window")
        chunks.append((s, e))
        s = e
    return chunks


def equity_to_returns(equity: list[float]) -> list[float]:
    return [b / a - 1.0 for a, b in zip(equity, equity[1:])]


def chain(returns: list[float]) -> float:
    eq = 1.0
    for r in returns:
        eq *= 1.0 + r
    return eq - 1.0


def max_drawdown(returns: list[float]) -> float:
    eq, peak, worst = 1.0, 1.0, 0.0
    for r in returns:
        eq *= 1.0 + r
        peak = max(peak, eq)
        worst = max(worst, 1.0 - eq / peak)
    return worst


def sharpe_per_period(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    sd = statistics.stdev(returns)
    return 0.0 if sd == 0 else statistics.fmean(returns) / sd


def psr(returns: list[float]) -> float:
    """Probabilistic Sharpe ratio P(SR > 0) with skew/kurtosis correction (single trial)."""
    n = len(returns)
    if n < 3:
        return 0.5
    sr = sharpe_per_period(returns)
    mean = statistics.fmean(returns)
    sd = statistics.pstdev(returns)
    if sd == 0:
        return 0.5
    skew = sum(((r - mean) / sd) ** 3 for r in returns) / n
    kurt = sum(((r - mean) / sd) ** 4 for r in returns) / n
    var = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if var <= 0:
        return 0.5
    z = sr * math.sqrt(n - 1) / math.sqrt(var)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def baseline_returns(closes: list[float], s: int, e: int, size: float, cost_side: float, kind: str, long_short: bool, lookback: int) -> list[float]:
    """Per-bar net returns for bars s..e-1 (return of bar i = close[i]/close[i-1]-1)."""
    out: list[float] = []
    pos = 0.0
    for i in range(s, e):
        if kind == "flat":
            target = 0.0
        elif kind == "buy_and_hold":
            target = size
        elif kind == "momentum":
            j = i - 1 - lookback
            past = closes[i - 1] / closes[j] - 1.0 if j >= 0 else 0.0
            target = size if past > 0 else (-size if long_short and past < 0 else 0.0)
        else:
            raise ValueError(kind)
        # decide on information up to close[i-1], hold over bar i
        turnover = abs(target - pos)
        pos = target
        bar = closes[i] / closes[i - 1] - 1.0
        out.append(pos * bar - turnover * cost_side)
    if out and pos != 0.0:
        out[-1] -= abs(pos) * cost_side  # liquidate at window end
    return out


# ---------------------------------------------------------------- replay


def run_chunk(binary: Path, snapshot: Path, uuid: str, rows: list[dict], s: int, e: int, window: int, seed: int, cap: float, extra: list[str]) -> dict:
    lo = e - window
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "slice.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows[lo:e])
        ratio = (e - s + 1) / window
        cmd = [
            str(binary), "--data", str(path), "--price-column", "close", "--high-column", "high", "--low-column", "low",
            "--adopt-combo-file", str(snapshot), "--adopt-combo-uuid", uuid,
            "--adoption-max-position-size-cap", repr(cap), "--bars", str(window),
            "--backtest-ratio", repr(ratio), "--seed", str(seed), "--json", *extra,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0:
            raise RuntimeError(f"trader-hs failed ({uuid} seed {seed} chunk {s}-{e}): {proc.stderr[-2000:]}")
        return json.loads(proc.stdout)["backtest"]


def chunk_returns(out: dict, expected: int) -> tuple[list[float], dict]:
    """``expected`` net returns from a backtest slice of ``expected + 1`` bars."""
    split = out["split"]
    if int(split["backtest"]) != expected + 1:
        raise RuntimeError(f"backtest slice {split['backtest']} != registered chunk {expected} + 1 anchor bar")
    curve = [float(x) for x in out["equityCurve"]]
    if len(curve) != expected + 1:
        raise RuntimeError(f"equityCurve length {len(curve)} != chunk {expected} + 1")
    realized = out.get("costs", {}).get("attribution", {}).get("realized", {})
    return equity_to_returns(curve), {"metrics": out.get("metrics", {}), "realizedCosts": realized, "perSideCost": out.get("costs", {}).get("perSideCost"), "maxPositionSize": out.get("maxPositionSize")}


def chunk_gross_returns(out: dict, expected: int) -> list[float]:
    """Per-bar gross (pre-cost) returns of the same slice, from the engine's cost attribution."""
    curve = [float(x) for x in out["costs"]["attribution"]["gross"]["equityCurve"]]
    if len(curve) != expected + 1:
        raise RuntimeError(f"gross equityCurve length {len(curve)} != chunk {expected} + 1")
    return equity_to_returns(curve)


def doubled_cost_returns(net: list[float], gross: list[float]) -> list[float]:
    """Per bar, net - (gross - net): fee, slippage, spread and funding charged twice."""
    return [2.0 * n - g for n, g in zip(net, gross)]


def daily_returns(open_times: list[int], returns: list[float]) -> dict[int, float]:
    """Compound per-bar returns into UTC calendar days keyed by day index."""
    days: dict[int, float] = {}
    for t, r in zip(open_times, returns):
        day = t // 86_400_000
        days[day] = (1.0 + days.get(day, 0.0)) * (1.0 + r) - 1.0
    return days


def fleet_daily(per_combo: list[dict[int, float]]) -> list[float]:
    """Equal-weight daily fleet returns over the days every combo covers."""
    common = sorted(set.intersection(*(set(d) for d in per_combo)))
    return [statistics.fmean(d[day] for d in per_combo) for day in common]


def prospective_gates(combo_decisions: list[str], fleet: dict, trial_bound: float) -> dict:
    """Evaluate the amendment's pass gates; any failed gate is a rejection."""
    base = fleet["net"]
    lifetime = max(0.0, 1.0 - min(1.0, trial_bound * (1.0 - base["psr"])))
    gates = {
        "everyComboNotRejected": all(d == "NOT_REJECTED" for d in combo_decisions),
        "lifetimeAdjustedPsrAtLeast095": lifetime >= 0.95,
        "maxDrawdownAtMost20pct": base["maxDrawdown"] <= 0.20,
        "sharpeAboveBuyAndHold": base["sharpeAnnualized"] > fleet["buy_and_hold"]["sharpeAnnualized"],
        "sharpeAboveMomentum": base["sharpeAnnualized"] > fleet["momentum"]["sharpeAnnualized"],
        "doubledCostReturnPositive": fleet["doubledCosts"]["netReturn"] > 0.0,
        "oneBarDelayReturnPositive": fleet["oneBarDelay"]["netReturn"] > 0.0,
    }
    return {"lifetimeAdjustedPsr": lifetime, "gates": gates, "decision": "PASS" if all(gates.values()) else "REJECTED"}


def summarize(returns: list[float], periods_per_year: float) -> dict:
    sr = sharpe_per_period(returns)
    return {
        "bars": len(returns),
        "netReturn": chain(returns),
        "maxDrawdown": max_drawdown(returns),
        "sharpeAnnualized": sr * math.sqrt(periods_per_year),
        "psr": psr(returns),
    }


def check_embargo(phase: dict, now_ms: int) -> None:
    """Refuse a phase before its registered release time, before any row is read."""
    release = phase.get("minimumEvaluationTimeUtc")
    if release and now_ms < int(datetime.fromisoformat(release.replace("Z", "+00:00")).timestamp() * 1000):
        raise SystemExit(f"phase is embargoed until {release}; no rows were read")


def check_manifest(reg: dict, phase_name: str, manifest_path: Path) -> dict:
    """The retrospective manifest must be the preregistered one, byte for byte."""
    pinned = reg["data"].get(f"{phase_name}ManifestSha256")
    if pinned and sha256_file(manifest_path) != pinned:
        raise SystemExit(f"{manifest_path.name} does not match the registered sha256")
    return json.loads(manifest_path.read_text())


def check_coverage(open_times: list[int], step: int, first_needed_ms: int, last_open_ms: int) -> None:
    """Rows must be contiguous from the first training row through the window's final bar."""
    try:
        check_contiguous(open_times, step)
    except ValueError as err:
        raise SystemExit(f"phase dataset is not contiguous: {err}") from err
    if not open_times or open_times[0] > first_needed_ms or open_times[-1] < last_open_ms:
        raise SystemExit("phase dataset does not cover the registered window and its training history")


def cmd_run(args: argparse.Namespace) -> None:
    reg = load_registration()
    phase = reg["phases"][args.phase]
    check_embargo(phase, int(time.time() * 1000))
    out_dir = ROOT / phase["outputDirectory"]  # absolute paths (tests) override ROOT
    if phase.get("oneShot") and (out_dir / "results.json").exists():
        raise SystemExit(f"{args.phase} is one-shot and already has sealed results; no rows were read")
    stress = "passGates" in phase
    data_dir = ROOT / reg["data"]["directory"]
    manifest = check_manifest(reg, args.phase, data_dir / phase["dataManifest"])
    snapshot = ROOT / reg["snapshot"]["path"]
    if sha256_file(snapshot) != reg["snapshot"]["sha256"]:
        raise SystemExit("frozen champion snapshot drifted from registration")
    binary = Path(args.binary)
    results = {"registration": REGISTRATION.name, "registrationSha256": sha256_file(REGISTRATION), "phase": args.phase, "combos": {}}
    series_out: dict[str, dict] = {}
    fleet_inputs: list[dict[str, dict[int, float]]] = []
    for combo in reg["combos"]:
        if combo["symbol"] not in phase["symbols"]:
            continue
        fname = f"{combo['symbol']}-{combo['interval']}.csv"
        meta = manifest["files"][fname]
        if sha256_file(data_dir / fname) != meta["sha256"]:
            raise SystemExit(f"{fname} drifted from manifest")
        rows = read_rows(data_dir / fname)
        open_times = [int(r["openTimeMs"]) for r in rows]
        closes = [float(r["close"]) for r in rows]
        start_ms = max(int(phase["startMs"]), int(combo["createdAtMs"]))
        step = INTERVAL_MS[combo["interval"]]
        first_eval_ms = -(-start_ms // step) * step
        last_open_ms = (int(phase["endExclusiveMs"]) - 1) // step * step
        check_coverage(open_times, step, first_eval_ms - combo["windowBars"] * step, last_open_ms)
        chunks = chunk_plan(open_times, start_ms, int(phase["endExclusiveMs"]), combo["windowBars"], combo["chunkBars"])
        per_seed: dict[str, list[float]] = {}
        gross_seed: dict[str, list[float]] = {}
        delayed_seed: dict[str, list[float]] = {}
        diag: dict[str, list] = {}
        for seed in reg["protocol"]["seeds"]:
            rets: list[float] = []
            gross: list[float] = []
            delayed: list[float] = []
            diag[str(seed)] = []
            cap = reg["protocol"]["adoptionMaxPositionSizeCap"]
            for s, e in chunks:
                out = run_chunk(binary, snapshot, combo["uuid"], rows, s, e, combo["windowBars"], seed, cap, [])
                r, d = chunk_returns(out, e - s)
                rets.extend(r)
                diag[str(seed)].append({"chunk": [open_times[s], open_times[e - 1]], **d})
                if stress:
                    gross.extend(chunk_gross_returns(out, e - s))
                    late = run_chunk(binary, snapshot, combo["uuid"], rows, s, e, combo["windowBars"], seed, cap, ["--backtest-signal-delay-bars", "1"])
                    delayed.extend(chunk_returns(late, e - s)[0])
            per_seed[str(seed)] = rets
            gross_seed[str(seed)] = gross
            delayed_seed[str(seed)] = delayed
            print(f"{combo['symbol']} seed {seed}: {len(rets)} bars", file=sys.stderr)
        ppy = combo["periodsPerYear"]
        seed_summaries = {k: summarize(v, ppy) for k, v in per_seed.items()}
        for k, chunks_diag in diag.items():
            seed_summaries[k]["roundTrips"] = sum(int(c["metrics"].get("roundTrips") or 0) for c in chunks_diag)
        median_seed = sorted(seed_summaries, key=lambda k: seed_summaries[k]["netReturn"])[len(seed_summaries) // 2]
        s0, e_last = chunks[0][0], chunks[-1][1]
        adopted = diag[str(reg["protocol"]["liveSeed"])][0]
        size, cost_side = float(adopted["maxPositionSize"]), float(adopted["perSideCost"])
        base_returns = {
            kind: baseline_returns(closes, s0, e_last, size, cost_side, kind, combo["positioning"] == "long-short", reg["protocol"]["momentumLookbackBars"])
            for kind in ("flat", "buy_and_hold", "momentum")
        }
        bases = {kind: summarize(r, ppy) for kind, r in base_returns.items()}
        pin_idx = next(i for i, t in enumerate(open_times[s0:e_last]) if t >= int(reg["protocol"]["pinnedAtMs"]))
        med = per_seed[median_seed]
        rule = reg["protocol"]["rejectionRule"]
        rejected_return = seed_summaries[median_seed]["netReturn"] <= rule["maxNetReturnForRejection"]
        rejected_dd = seed_summaries[median_seed]["maxDrawdown"] > rule["maxDrawdown"]
        no_trades = seed_summaries[median_seed]["roundTrips"] == 0
        results["combos"][combo["symbol"]] = {
            "uuid": combo["uuid"],
            "window": [open_times[s0], open_times[e_last - 1]],
            "chunks": len(chunks),
            "adoptedMaxPositionSize": size,
            "adoptedPerSideCost": cost_side,
            "seeds": seed_summaries,
            "medianSeed": median_seed,
            "liveSeed": seed_summaries.get(str(reg["protocol"]["liveSeed"])),
            "postPinSubwindow": summarize(med[pin_idx:], ppy),
            "baselines": bases,
            "decision": "REJECTED" if (rejected_return or rejected_dd) else "NOT_REJECTED",
            "rejectionReasons": [n for n, hit in (("no_trades", no_trades), ("net_return_not_above_zero", rejected_return and not no_trades), ("drawdown_above_limit", rejected_dd)) if hit],
            "lifetimeAdjustedPsr": max(0.0, 1.0 - min(1.0, reg["protocol"]["optimizerTrialLowerBound"] * (1.0 - seed_summaries[median_seed]["psr"]))),
        }
        series_out[combo["symbol"]] = {"openTimeMs": open_times[s0:e_last], "seeds": per_seed}
        if stress:
            times = open_times[s0:e_last]
            med_net = per_seed[median_seed]
            fleet_inputs.append(
                {
                    "net": daily_returns(times, med_net),
                    "doubledCosts": daily_returns(times, doubled_cost_returns(med_net, gross_seed[median_seed])),
                    "oneBarDelay": daily_returns(times, delayed_seed[median_seed]),
                    "buy_and_hold": daily_returns(times, base_returns["buy_and_hold"]),
                    "momentum": daily_returns(times, base_returns["momentum"]),
                }
            )
            series_out[combo["symbol"]]["stress"] = {"gross": gross_seed[median_seed], "oneBarDelay": delayed_seed[median_seed], "medianSeed": median_seed}
        results.setdefault("diagnostics", {})[combo["symbol"]] = diag
    decisions = [c["decision"] for c in results["combos"].values()]
    rejected = decisions.count("REJECTED")
    results["fleetDecision"] = "REJECTED" if rejected >= reg["protocol"]["fleetRejectionMinCombos"] else "INCONCLUSIVE"
    if stress:
        fleet = {key: summarize(fleet_daily([c[key] for c in fleet_inputs]), 365.0) for key in fleet_inputs[0]}
        verdict = prospective_gates(decisions, fleet, reg["protocol"]["optimizerTrialLowerBound"])
        results["fleet"] = {**fleet, **verdict}
        results["fleetDecision"] = verdict["decision"]
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = "x" if phase.get("oneShot") else "w"
    for name, payload in (("results.json", json.dumps(results, indent=1, sort_keys=True) + "\n"), ("returns.json", json.dumps(series_out, sort_keys=True) + "\n")):
        with (out_dir / name).open(mode) as handle:
            handle.write(payload)
        if phase.get("oneShot"):
            os.chmod(out_dir / name, 0o444)
    print(json.dumps({k: {"decision": v["decision"], "median": v["seeds"][v["medianSeed"]]} for k, v in results["combos"].items()}, indent=1))
    print("fleet:", results["fleetDecision"])


# ---------------------------------------------------------------- selftest


def cmd_selftest(_: argparse.Namespace) -> None:
    times = [i * 10 for i in range(100)]
    plan = chunk_plan(times, 500, 1000, window=40, test_len=8)
    assert plan[0] == (50, 58) and plan[-1] == (98, 100), plan
    assert all(b[0] == a[1] for a, b in zip(plan, plan[1:]))
    try:
        chunk_plan(times, 0, 100, window=40, test_len=8)
        raise AssertionError("missing history must fail")
    except ValueError:
        pass
    assert abs(chain([0.1, -0.1]) - (-0.01)) < 1e-12
    assert abs(max_drawdown([0.1, -0.5, 0.2]) - 0.5) < 1e-12
    closes = [100.0, 110.0, 99.0]
    flat = baseline_returns(closes, 1, 3, 1.0, 0.001, "flat", False, 1)
    assert flat == [0.0, 0.0]
    bh = baseline_returns(closes, 1, 3, 1.0, 0.001, "buy_and_hold", False, 1)
    assert abs(bh[0] - (0.1 - 0.001)) < 1e-12 and abs(bh[1] - (-0.1 - 0.001)) < 1e-12
    assert psr([0.01, 0.02, 0.015, 0.012, 0.018]) > 0.9
    for call in (
        lambda: check_embargo({"minimumEvaluationTimeUtc": "2027-01-21T06:00:00Z"}, 1800489600000),
        lambda: check_coverage([0, 10, 20], 10, 0, 30),
        lambda: check_coverage([10, 20, 30], 10, 0, 30),
        lambda: check_coverage([0, 10, 30], 10, 0, 30),
    ):
        try:
            call()
            raise AssertionError("guard must refuse")
        except SystemExit:
            pass
    check_embargo({"minimumEvaluationTimeUtc": "2027-01-21T06:00:00Z"}, 1800511200000)
    assert doubled_cost_returns([0.01, -0.002], [0.012, 0.0]) == [0.008, -0.004]
    day = 86_400_000
    d = daily_returns([0, day // 2, day], [0.1, 0.1, -0.5])
    assert abs(d[0] - 0.21) < 1e-12 and d[1] == -0.5
    assert fleet_daily([{0: 0.02, 1: 0.0}, {0: 0.0, 1: 0.04, 2: 0.1}]) == [0.01, 0.02]
    good = {"psr": 1.0, "maxDrawdown": 0.05, "sharpeAnnualized": 3.0, "netReturn": 0.2}
    fleet = {"net": good, "buy_and_hold": {"sharpeAnnualized": 1.0}, "momentum": {"sharpeAnnualized": 1.0},
             "doubledCosts": {"netReturn": 0.1}, "oneBarDelay": {"netReturn": 0.05}}
    assert prospective_gates(["NOT_REJECTED"] * 3, fleet, 3261)["decision"] == "PASS"
    assert prospective_gates(["NOT_REJECTED", "REJECTED", "NOT_REJECTED"], fleet, 3261)["decision"] == "REJECTED"
    assert prospective_gates(["NOT_REJECTED"] * 3, {**fleet, "oneBarDelay": {"netReturn": -0.01}}, 3261)["decision"] == "REJECTED"
    assert prospective_gates(["NOT_REJECTED"] * 3, {**fleet, "net": {**good, "psr": 0.9999}}, 3261)["decision"] == "REJECTED"
    check_coverage([0, 10, 20, 30], 10, 0, 30)
    print("selftest ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--end-ms", help="override fetch end (exclusive), e.g. for the prospective phase")
    f.add_argument("--manifest", default="manifest.json")
    r = sub.add_parser("run")
    r.add_argument("--phase", required=True, choices=["retrospective", "prospective"])
    r.add_argument("--binary", default=os.environ.get("TRADER_HS_BIN", "trader-hs"))
    sub.add_parser("selftest")
    args = parser.parse_args()
    {"fetch": cmd_fetch, "run": cmd_run, "selftest": cmd_selftest}[args.cmd](args)


if __name__ == "__main__":
    main()

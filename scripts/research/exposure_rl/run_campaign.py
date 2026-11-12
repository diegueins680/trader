"""Registered runner for exposure_rl_control_v1.

Commands:
  champion      export champion-method positions through the trader-hs CLI
  development   run every registered development job (walk-forward folds x seeds)
  confirm       one-shot confirmation of a frozen candidate (guarded)

Everything is offline research: public data, simulated fills, no credentials,
no order-capable client. Artifacts are written under an ignored directory.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import multiprocessing.pool  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from exposure_rl import agents as A  # noqa: E402
from exposure_rl import baselines as B  # noqa: E402
from exposure_rl import evaluate as V  # noqa: E402
from exposure_rl import market as M  # noqa: E402
from exposure_rl.artifacts import save_policy  # noqa: E402
from exposure_rl.env import STRESSES, ExposureEnv, Execution, usable_symbols  # noqa: E402
from exposure_rl.shield import CAP, SCALE  # noqa: E402

REGISTRATION = ROOT / "research-notes" / "registrations" / "exposure-rl-control-v1.json"
ARTIFACTS = ROOT / ".tmp" / "research" / "exposure-rl-v1"
DAY_MS = 86_400_000
MIN_FIT_BARS = 300
STRESS_SEED = 20261111
LEARNERS = ("ppo", "ppo_risk", "ppo_no_context", "td3", "offline")
CHAMPION_FLAGS = [
    "--method", "ta_regime_switch", "--positioning", "long-short", "--interval", "6h",
    "--open-threshold", "0.001", "--close-threshold", "0.001", "--fee", "0.0005", "--slippage", "0.00015",
    "--spread", "0.0003", "--funding-rate", "0", "--max-position-size", "0.1666",
    "--stop-loss", "0.0753", "--take-profit", "0.0353",
]  # fmt: skip


def utc_ms(text: str) -> int:
    return int(datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp() * 1000)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def registration() -> tuple[dict, str]:
    payload = REGISTRATION.read_bytes()
    return json.loads(payload), sha256_bytes(payload)


def source_state() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "scripts/research/exposure_rl", str(REGISTRATION.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    return {"commit": head, "campaignSourcesClean": dirty == ""}


def load_development(reg: dict) -> dict:
    directory = ARTIFACTS / "data-dev"
    return M.load_panel(directory / "development-panel.csv", directory / "development-funding.csv", reg["data"]["development"])


def fold_ranges(panel: dict, reg: dict, fold: dict) -> dict:
    """Per-symbol bar index ranges for fit, inner validation and test."""
    purge = int(reg["validation"]["purgeBars"])
    inner = int(reg["validation"]["innerValidationDays"]) * DAY_MS
    start, end = utc_ms(fold["testStartUtc"]), utc_ms(fold["testEndUtcExclusive"])
    out = {"fit": {}, "val": {}, "test": {}}
    for symbol, series in panel.items():
        i0 = int(np.searchsorted(series.open_time, start))
        i1 = int(np.searchsorted(series.open_time, end))
        v0 = int(np.searchsorted(series.open_time, start - inner))
        train_stop = i0 - purge
        fit_stop = v0 - purge
        if i1 - i0 >= 2 or i1 - max(i0, M.BURN_IN) >= 2:
            out["test"][symbol] = (i0, i1)
        if fit_stop >= MIN_FIT_BARS:
            out["fit"][symbol] = (0, fit_stop)
            out["val"][symbol] = (v0, train_stop)
    return out


class Schedule:
    """Replays externally generated target exposures (the champion-method export)."""

    name = "champion_method"

    def __init__(self, positions: dict[str, dict[str, float]]):
        self.positions = {s: {int(k): float(v) for k, v in rows.items()} for s, rows in positions.items()}

    def act_at(self, symbols, open_times) -> np.ndarray:
        raw = [self.positions.get(s, {}).get(int(t), 0.0) * SCALE / CAP for s, t in zip(symbols, open_times)]
        return np.clip(np.asarray(raw, dtype=np.float64), -1.0, 1.0)


def fleet_sharpe(policy, panel, ranges) -> float:
    if not usable_symbols(panel, ranges):
        return float("-inf")
    _, series = V.portfolio_series(V.rollout(policy, panel, ranges, seed=STRESS_SEED))
    value = V.metrics(series).get("sharpe")
    return float("-inf") if value is None else float(value)


def evaluate_all(policy, panel, ranges, extra: dict | None = None) -> tuple[dict, dict]:
    """Evaluate a frozen policy under every registered stress. Returns (summary, arrays)."""
    summary, arrays = {}, {}
    stresses = dict(STRESSES)
    stresses.update(extra or {})
    for name, execution in stresses.items():
        result = V.rollout(policy, panel, ranges, execution, seed=STRESS_SEED)
        times, fleet = V.portfolio_series(result)
        summary[name] = {"fleet": V.metrics(fleet), "activity": V.activity(result)}
        arrays[f"{name}__times"], arrays[f"{name}__fleet"] = times, fleet
        if name == "base":
            for symbol, row in result.items():
                summary[name].setdefault("symbols", {})[symbol] = V.metrics(row["net"])
                for key in ("open_time", "net", "exposure", "turnover", "cost", "funding", "kind", "reasons"):
                    arrays[f"base__{symbol}__{key}"] = row[key]
    return summary, arrays


class Selector:
    """Checkpoint rule: highest inner-validation fleet Sharpe; earliest on ties."""

    def __init__(self, panel, val_ranges):
        self.panel, self.val = panel, val_ranges
        self.best, self.best_params, self.best_step, self.trace = float("-inf"), None, None, []

    def __call__(self, step: int, policy) -> None:
        score = fleet_sharpe(policy, self.panel, self.val)
        self.trace.append({"step": int(step), "validationSharpe": None if score == float("-inf") else score})
        if self.best_params is None or score > self.best:
            self.best, self.best_step = score, int(step)
            self.best_params = policy.actor.copy_params()

    def select(self, policy):
        if self.best_params is not None:
            policy.actor.params = self.best_params
        return policy


def run_job(spec: dict) -> dict:
    """One registered job. Runs in a worker process; never raises."""
    started = time.time()
    job_dir = Path(spec["runDir"]) / "jobs"
    try:
        reg, reg_sha = registration()
        if reg_sha != spec["registrationSha256"]:
            raise RuntimeError("registration changed after planning")
        panel = load_development(reg)
        fold = next(f for f in reg["validation"]["folds"] if f["id"] == spec["fold"])
        ranges = fold_ranges(panel, reg, fold)
        cfg, seed, kind = reg["policies"], spec.get("seed"), spec["kind"]
        outputs: dict[str, tuple] = {}
        record: dict = {"id": spec["id"], "kind": kind, "fold": spec["fold"], "seed": seed, "policies": {}}

        def finish(name: str, policy, extra_stress=None, detail=None):
            summary, arrays = evaluate_all(policy, panel, ranges["test"], extra_stress)
            outputs[name] = arrays
            record["policies"][name] = {"test": summary, **(detail or {})}
            if hasattr(policy, "actor"):
                digest = save_policy(job_dir / f"{spec['id']}__{name}.policy.json", policy, {"campaign": reg["campaign"], "registrationSha256": reg_sha, "fold": spec["fold"], "seed": seed, "sourceCommit": spec["source"]["commit"], "developmentPanelSha256": reg["data"]["development"]["panelSha256"]})
                record["policies"][name]["artifactSha256"] = digest

        if kind == "baselines":
            x, y = B.ridge_training_set(panel, ranges["fit"])
            champion = json.loads((ARTIFACTS / "champion" / "champion-positions.json").read_text())
            if champion["sha256"] != sha256_bytes(json.dumps(champion["positions"], sort_keys=True).encode()):
                raise RuntimeError("champion export hash mismatch")
            record["championExportSha256"] = champion["sha256"]
            record["championBinarySha256"] = champion.get("binarySha256")
            for policy in (B.Constant(0.0, "cash"), B.Constant(1.0, "buy_and_hold"), B.RegimeRule(), B.RidgeForecast(x, y), Schedule(champion["positions"])):
                finish(policy.name, policy)
        elif kind in ("ppo", "ppo_risk", "ppo_no_context"):
            c = cfg["rl"]["ppo"]
            mask = None
            if kind == "ppo_no_context":
                mask = np.ones(len(M.MARKET_FEATURES) + 4)
                for name in ("funding_last", "funding_mean_21", "btc_ret4_z", "btc_ret28_z"):
                    mask[M.MARKET_FEATURES.index(name)] = 0.0
            env = ExposureEnv(panel, ranges["fit"], c["lanes"], seed)
            selector = Selector(panel, ranges["val"])
            policy, aux = A.train_ppo(
                env, seed, c["totalSteps"], rollout=c["rollout"], epochs=c["epochs"], minibatch=c["minibatch"], lr=c["learningRate"],
                lam=c["gaeLambda"], clip=c["clip"], entropy_coef=c["entropyCoef"], checkpoint=selector, checkpoint_every=c["checkpointEvery"],
                risk_penalty=cfg["extension"]["ppo_risk"]["riskPenalty"] if kind == "ppo_risk" else 0.0, obs_mask=mask,
            )
            policy = selector.select(policy)
            policy.name = kind
            extra = {"unshielded_diagnostic": Execution(shield_enabled=False)} if kind == "ppo" else None
            finish(kind, policy, extra, {"checkpoints": selector.trace, "selectedStep": selector.best_step, "trainingTail": aux["history"][-3:], "trainingSteps": c["totalSteps"]})
        elif kind == "td3":
            c = cfg["rl"]["td3"]
            env = ExposureEnv(panel, ranges["fit"], c["lanes"], seed)
            selector = Selector(panel, ranges["val"])
            policy, aux = A.train_td3(
                env, seed, c["totalSteps"], warmup=c["warmupSteps"], batch=c["batch"], updates_per_step=c["updatesPerVectorStep"], lr=c["learningRate"],
                tau=c["tau"], explore_noise=c["exploreNoise"], target_noise=c["targetNoise"], noise_clip=c["noiseClip"], policy_delay=c["policyDelay"],
                checkpoint=selector, checkpoint_every=c["checkpointEvery"],
            )
            finish("td3", selector.select(policy), None, {"checkpoints": selector.trace, "selectedStep": selector.best_step, "updates": aux["updates"], "trainingSteps": c["totalSteps"]})
        elif kind == "offline":
            c, d = cfg["rl"]["iql"], cfg["offlineDataset"]
            x, y = B.ridge_training_set(panel, ranges["fit"])
            rule, ridge = B.RegimeRule(), B.RidgeForecast(x, y)
            env = ExposureEnv(panel, ranges["fit"], d["lanes"], seed)
            data = B.collect_dataset(env, B.BehaviorMixture(rule, ridge, d["lanes"], seed + 1), d["transitions"])
            record["dataset"] = {"origin": "simulated", "transitions": int(len(data["obs"])), "episodes": int(data["episode"].max() + 1), "meanReward": float(data["rew"].mean()), "actionStd": float(data["act"].std()), "terminated": int(data["term"].sum())}
            selector = Selector(panel, ranges["val"])
            iql, aux = A.train_iql(data, seed, c["updates"], batch=c["batch"], lr=c["learningRate"], expectile=c["expectile"], beta=c["beta"], weight_clip=c["weightClip"], tau=c["tau"], checkpoint=selector, checkpoint_every=c["checkpointEvery"])
            iql = selector.select(iql)
            bc, _ = A.train_bc(data, seed, d["bcUpdates"])
            bandit, _ = A.train_bandit(data, seed, d["banditUpdates"])
            ope = {name: offline_evaluation(panel, ranges["fit"], data, policy, seed) for name, policy in (("iql", iql), ("behavior_cloning", bc))}
            finish("iql", iql, None, {"checkpoints": selector.trace, "selectedStep": selector.best_step, "trainingTail": aux["history"][-2:], "ope": ope["iql"]})
            finish("behavior_cloning", bc, None, {"ope": ope["behavior_cloning"]})
            finish("contextual_bandit", bandit)
        else:
            raise ValueError(f"unknown job kind: {kind}")
        np.savez_compressed(job_dir / f"{spec['id']}.npz", **{f"{name}__{key}": value for name, arrays in outputs.items() for key, value in arrays.items()})
        record.update({"status": "completed", "seconds": round(time.time() - started, 1)})
    except Exception as error:  # recorded, never hidden
        record = {"id": spec["id"], "kind": spec["kind"], "fold": spec["fold"], "seed": spec.get("seed"), "status": "failed", "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()[-2000:], "seconds": round(time.time() - started, 1)}
    (job_dir / f"{spec['id']}.json").write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    return {key: record.get(key) for key in ("id", "kind", "fold", "seed", "status", "seconds", "error")}


def offline_evaluation(panel, fit_ranges, data: dict, policy, seed: int, starts: int = 512, horizon: int = 512) -> dict:
    """FQE versus simulator ground truth, and proposal-space importance-sampling support."""
    critic = A.fitted_q_evaluation(data, policy, seed + 7, 15_000)
    first = np.flatnonzero(data["step"] == 0)
    env = ExposureEnv(panel, fit_ranges, 1, seed + 9, episode_length=None)
    room = env.high[data["symbol"][first]] - data["bar"][first]
    first = first[room > horizon][:starts]
    if len(first) < 32:
        return {"status": "insufficient evidence", "startStates": int(len(first))}
    env = ExposureEnv(panel, fit_ranges, len(first), seed + 9, episode_length=None)
    obs = env.reset_at(data["symbol"][first], data["bar"][first])
    estimate = A.q_value(critic, obs, policy.act(obs))
    truth, discount = np.zeros(len(first)), 1.0
    for _ in range(horizon):
        obs, reward, _, _, _, _ = env.step(policy.act(obs), auto_reset=False)
        truth += discount * reward
        discount *= A.GAMMA
    # Importance weights for a stochastic form of the target (Gaussian, std 0.3)
    # against the Gaussian behaviour components, in proposal space.
    gaussian = data["component"] <= 1
    log_ratio = np.where(gaussian, -0.5 * ((data["raw"] - policy.act(data["obs"])) / B.BEHAVIOR_STD) ** 2 + 0.5 * ((data["raw"] - data["mean"]) / B.BEHAVIOR_STD) ** 2, np.nan)
    ess = {}
    episodes = data["episode"]
    order = np.lexsort((data["step"], episodes))
    for h in (1, 4, 16, 64, 256):
        keep = order[data["step"][order] < h]
        ids = episodes[keep]
        complete = np.bincount(ids, minlength=int(episodes.max()) + 1) == h
        valid = complete[ids] & gaussian[keep]
        sums = np.bincount(ids[valid], weights=log_ratio[keep][valid], minlength=int(episodes.max()) + 1)
        counts = np.bincount(ids[valid], minlength=int(episodes.max()) + 1)
        logs = sums[counts == h]
        weights = np.exp(logs - logs.max()) if len(logs) else logs  # scale-free: ESS is invariant
        if len(weights) < 30:
            ess[str(h)] = {"status": "insufficient evidence", "episodes": int(len(weights))}
        else:
            effective = float(weights.sum() ** 2 / np.sum(weights**2))
            ess[str(h)] = {"episodes": int(len(weights)), "effectiveSampleSize": effective, "fraction": effective / len(weights)}
    error = estimate - truth
    return {
        "startStates": int(len(first)),
        "horizonBars": horizon,
        "fqeMean": float(estimate.mean()),
        "simulatorTruthMean": float(truth.mean()),
        "meanError": float(error.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "rankCorrelation": float(np.corrcoef(np.argsort(np.argsort(estimate)), np.argsort(np.argsort(truth)))[0, 1]),
        "truthStd": float(truth.std()),
        "importanceSamplingEssByHorizon": ess,
        "note": "Simulated dataset and simulator truth on the fit region; says nothing about real logged behaviour.",
    }


def plan(reg: dict, reg_sha: str, run_dir: Path, source: dict) -> list[dict]:
    jobs = []
    for fold in reg["validation"]["folds"]:
        base = {"fold": fold["id"], "runDir": str(run_dir), "registrationSha256": reg_sha, "source": source}
        jobs.append({**base, "id": f"f{fold['id']}-baselines", "kind": "baselines"})
        for seed in reg["seeds"]:
            for kind in LEARNERS:
                jobs.append({**base, "id": f"f{fold['id']}-s{seed}-{kind}", "kind": kind, "seed": seed})
    return jobs


def command_development(args) -> int:
    reg, reg_sha = registration()
    source = source_state()
    if not source["campaignSourcesClean"] and not args.allow_dirty:
        raise SystemExit("campaign sources or registration are uncommitted; commit them first")
    load_development(reg)  # verifies registered hashes before anything is planned
    run_dir = ARTIFACTS / "runs" / args.run
    (run_dir / "jobs").mkdir(parents=True, exist_ok=False)
    jobs = plan(reg, reg_sha, run_dir, source)
    if args.only:
        jobs = [job for job in jobs if any(job["id"].startswith(prefix) or job["kind"] == prefix for prefix in args.only)]
    (run_dir / "planned-registry.json").write_text(json.dumps({"campaign": reg["campaign"], "registrationSha256": reg_sha, "source": source, "plannedAt": datetime.now(timezone.utc).isoformat(), "pilot": bool(args.allow_dirty or args.only), "jobs": [{k: j[k] for k in ("id", "kind", "fold") if k in j} | {"seed": j.get("seed")} for j in jobs]}, indent=1), encoding="utf-8")
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as events, multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        for outcome in pool.imap_unordered(run_job, jobs):
            events.write(json.dumps(outcome, sort_keys=True) + "\n")
            events.flush()
            print(outcome, flush=True)
    return 0


def command_champion(args) -> int:
    reg, _ = registration()
    panel_path = ARTIFACTS / "data-dev" / "development-panel.csv"
    if M.sha256_file(panel_path) != reg["data"]["development"]["panelSha256"]:
        raise SystemExit("development panel hash mismatch")
    binary = Path(args.binary)
    out_dir = ARTIFACTS / "champion"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list] = {}
    with panel_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.setdefault(row["symbol"], []).append(row)
    first_test = utc_ms(reg["validation"]["folds"][0]["testStartUtc"])
    positions, runs = {}, {}

    def export(item):
        symbol, bars = item
        index = next((i for i, bar in enumerate(bars) if int(bar["openTime"]) >= first_test), len(bars))
        ratio = min(0.9, max(0.05, (len(bars) - index + 8) / len(bars)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{symbol}.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["openTimeMs", "open", "high", "low", "close", "volume", "closeTimeMs"])
                writer.writerows([b["openTime"], b["open"], b["high"], b["low"], b["close"], b["volume"], b["closeTime"]] for b in bars)
            command = [str(binary), "--data", str(path), "--price-column", "close", "--high-column", "high", "--low-column", "low", *CHAMPION_FLAGS, "--backtest-ratio", f"{ratio:.6f}", "--json"]
            done = subprocess.run(command, capture_output=True, text=True, timeout=3600)
        if done.returncode != 0:
            raise SystemExit(f"{symbol}: trader-hs failed: {done.stderr[-500:]}")
        backtest = json.loads(done.stdout)["backtest"]
        times, held = backtest["openTimes"], backtest["positions"]
        summary = {"backtestRatio": ratio, "firstOpenTime": int(times[0]), "bars": len(held), "nonZeroShare": float(np.mean(np.asarray(held) != 0)), "nativeMetrics": {k: backtest["metrics"].get(k) for k in ("totalReturn", "sharpe", "maxDrawdown", "tradeCount")}}
        print(symbol, summary, flush=True)
        return symbol, {str(int(t)): float(p) for t, p in zip(times, held)}, summary

    with multiprocessing.pool.ThreadPool(args.workers) as pool:
        for symbol, held, summary in pool.map(export, sorted(rows.items())):
            positions[symbol], runs[symbol] = held, summary
    document = {"schema": "champion_method_export_v1", "flags": CHAMPION_FLAGS, "binarySha256": M.sha256_file(binary), "source": source_state(), "runs": runs, "positions": positions, "sha256": sha256_bytes(json.dumps(positions, sort_keys=True).encode())}
    (out_dir / "champion-positions.json").write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    champion = commands.add_parser("champion")
    champion.add_argument("--binary", required=True)
    champion.add_argument("--workers", type=int, default=7)
    development = commands.add_parser("development")
    development.add_argument("--run", required=True)
    development.add_argument("--workers", type=int, default=8)
    development.add_argument("--allow-dirty", action="store_true", help="pilot only; results are not registered evidence")
    development.add_argument("--only", nargs="*", help="pilot only; job id prefixes or kinds")
    args = parser.parse_args(argv)
    return {"champion": command_champion, "development": command_development}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


BINANCE_INTERVALS = [
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1M",
]


def _parse_int_prefix(s: str) -> Tuple[int, str]:
    i = 0
    while i < len(s) and s[i].isdigit():
        i += 1
    if i == 0:
        raise ValueError(f"expected integer prefix in {s!r}")
    return int(s[:i]), s[i:]


def duration_seconds(raw: str) -> int:
    s = raw.strip()
    n, unit = _parse_int_prefix(s)
    if unit == "":
        raise ValueError(f"missing duration unit in {raw!r}")
    if len(unit) != 1:
        raise ValueError(f"invalid duration unit in {raw!r}")
    u = unit
    if u == "s":
        mult = 1
    elif u == "m":
        mult = 60
    elif u == "h":
        mult = 60 * 60
    elif u == "d":
        mult = 24 * 60 * 60
    elif u == "w":
        mult = 7 * 24 * 60 * 60
    elif u == "M":
        mult = 30 * 24 * 60 * 60
    else:
        u2 = u.lower()
        if u2 in ("s", "m", "h", "d", "w"):
            return duration_seconds(f"{n}{u2}")
        raise ValueError(f"invalid duration unit in {raw!r}")
    return n * mult


def lookback_bars(interval: str, lookback_window: str) -> int:
    interval_sec = duration_seconds(interval)
    lookback_sec = duration_seconds(lookback_window)
    if interval_sec <= 0 or lookback_sec <= 0:
        raise ValueError("interval/lookback must be > 0")
    # Match Trader.Duration.lookbackBarsFrom: ceiling(lookback / interval), min 1.
    return max(1, (lookback_sec + interval_sec - 1) // interval_sec)


def count_csv_rows(path: Path) -> int:
    # Fast-ish line count; CSV has a header row.
    with path.open("rb") as f:
        return sum(1 for _ in f)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    if lo <= 0 or hi <= 0 or hi < lo:
        raise ValueError("log_uniform requires 0 < lo <= hi")
    if lo == hi:
        return lo
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def maybe(rng: random.Random, p_none: float, sampler) -> Optional[Any]:
    if rng.random() < p_none:
        return None
    return sampler()


@dataclass(frozen=True)
class TrialParams:
    interval: str
    bars: int  # 0 = auto/full; otherwise explicit bar count
    method: str  # "11" | "10" | "01"
    positioning: str  # "long-flat" | "long-short"
    normalization: str
    base_open_threshold: float
    base_close_threshold: float
    fee: float
    epochs: int
    hidden_size: int
    learning_rate: float
    val_ratio: float
    patience: int
    grad_clip: Optional[float]
    slippage: float
    spread: float
    intrabar_fill: str  # "stop-first" | "take-profit-first"
    stop_loss: Optional[float]
    take_profit: Optional[float]
    trailing_stop: Optional[float]
    max_drawdown: Optional[float]
    max_daily_loss: Optional[float]
    max_order_errors: Optional[int]
    kalman_dt: float
    kalman_process_var: float
    kalman_measurement_var: float
    kalman_z_min: float
    kalman_z_max: float
    max_high_vol_prob: Optional[float]
    max_conformal_width: Optional[float]
    max_quantile_width: Optional[float]
    confirm_conformal: bool
    confirm_quantiles: bool
    confidence_sizing: bool
    min_position_size: float


@dataclass(frozen=True)
class TrialResult:
    ok: bool
    reason: Optional[str]
    elapsed_sec: float
    params: TrialParams
    final_equity: Optional[float]
    metrics: Optional[Dict[str, Any]]
    open_threshold: Optional[float]
    close_threshold: Optional[float]
    stdout_json: Optional[Dict[str, Any]]


def discover_trader_bin(haskell_dir: Path) -> Path:
    try:
        out = subprocess.check_output(
            ["cabal", "list-bin", "trader-hs"],
            cwd=str(haskell_dir),
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"failed to discover trader-hs binary via cabal: {e}") from e
    p = Path(out)
    if not p.exists():
        raise RuntimeError(f"cabal returned non-existent binary path: {p}")
    return p


def build_command(
    trader_bin: Path,
    base_args: List[str],
    params: TrialParams,
    tune_ratio: float,
    use_sweep_threshold: bool,
) -> List[str]:
    cmd = [str(trader_bin)]
    cmd += base_args
    cmd += ["--interval", params.interval]
    cmd += ["--bars", str(params.bars)]
    cmd += ["--positioning", params.positioning]
    cmd += ["--method", params.method]
    cmd += ["--normalization", params.normalization]
    cmd += ["--open-threshold", f"{max(0.0, params.base_open_threshold):.12g}"]
    cmd += ["--close-threshold", f"{max(0.0, params.base_close_threshold):.12g}"]
    cmd += ["--fee", f"{max(0.0, params.fee):.12g}"]
    cmd += ["--epochs", str(params.epochs)]
    cmd += ["--hidden-size", str(params.hidden_size)]
    cmd += ["--lr", f"{params.learning_rate:.8f}"]
    cmd += ["--val-ratio", f"{params.val_ratio:.6f}"]
    cmd += ["--patience", str(params.patience)]
    if params.grad_clip is not None:
        cmd += ["--grad-clip", f"{params.grad_clip:.8f}"]
    cmd += ["--slippage", f"{params.slippage:.8f}"]
    cmd += ["--spread", f"{params.spread:.8f}"]
    cmd += ["--intrabar-fill", params.intrabar_fill]
    if params.stop_loss is not None:
        cmd += ["--stop-loss", f"{params.stop_loss:.8f}"]
    if params.take_profit is not None:
        cmd += ["--take-profit", f"{params.take_profit:.8f}"]
    if params.trailing_stop is not None:
        cmd += ["--trailing-stop", f"{params.trailing_stop:.8f}"]
    if params.max_drawdown is not None:
        cmd += ["--max-drawdown", f"{params.max_drawdown:.8f}"]
    if params.max_daily_loss is not None:
        cmd += ["--max-daily-loss", f"{params.max_daily_loss:.8f}"]
    if params.max_order_errors is not None:
        cmd += ["--max-order-errors", str(params.max_order_errors)]
    cmd += ["--kalman-dt", f"{max(1e-12, params.kalman_dt):.12g}"]
    cmd += ["--kalman-process-var", f"{max(1e-12, params.kalman_process_var):.12g}"]
    cmd += ["--kalman-measurement-var", f"{max(1e-12, params.kalman_measurement_var):.12g}"]
    cmd += ["--kalman-z-min", f"{max(0.0, params.kalman_z_min):.12g}"]
    cmd += ["--kalman-z-max", f"{max(max(0.0, params.kalman_z_min), params.kalman_z_max):.12g}"]
    if params.max_high_vol_prob is not None:
        cmd += ["--max-high-vol-prob", f"{clamp(float(params.max_high_vol_prob), 0.0, 1.0):.12g}"]
    if params.max_conformal_width is not None:
        cmd += ["--max-conformal-width", f"{max(0.0, float(params.max_conformal_width)):.12g}"]
    if params.max_quantile_width is not None:
        cmd += ["--max-quantile-width", f"{max(0.0, float(params.max_quantile_width)):.12g}"]
    if params.confirm_conformal:
        cmd += ["--confirm-conformal"]
    if params.confirm_quantiles:
        cmd += ["--confirm-quantiles"]
    if params.confidence_sizing:
        cmd += ["--confidence-sizing"]
    cmd += ["--min-position-size", f"{clamp(float(params.min_position_size), 0.0, 1.0):.12g}"]
    if use_sweep_threshold:
        cmd += ["--sweep-threshold", "--tune-ratio", f"{tune_ratio:.6f}"]
    cmd += ["--json"]
    return cmd


def run_trial(
    trader_bin: Path,
    base_args: List[str],
    params: TrialParams,
    tune_ratio: float,
    use_sweep_threshold: bool,
    timeout_sec: float,
    disable_lstm_persistence: bool,
) -> TrialResult:
    cmd = build_command(trader_bin, base_args, params, tune_ratio, use_sweep_threshold)
    env = os.environ.copy()
    if disable_lstm_persistence:
        env["TRADER_LSTM_WEIGHTS_DIR"] = ""

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(trader_bin.parent),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        elapsed = time.time() - t0
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return TrialResult(
            ok=False,
            reason=f"timeout>{timeout_sec}s",
            elapsed_sec=elapsed,
            params=params,
            final_equity=None,
            metrics=None,
            open_threshold=None,
            close_threshold=None,
            stdout_json=None,
        )

    if proc.returncode != 0:
        reason = (proc.stderr or proc.stdout or "").strip()
        if len(reason) > 300:
            reason = reason[:300] + "…"
        return TrialResult(
            ok=False,
            reason=reason or f"exit={proc.returncode}",
            elapsed_sec=elapsed,
            params=params,
            final_equity=None,
            metrics=None,
            open_threshold=None,
            close_threshold=None,
            stdout_json=None,
        )

    try:
        out = json.loads(proc.stdout)
    except Exception as e:  # noqa: BLE001
        reason = f"json parse error: {e}"
        return TrialResult(
            ok=False,
            reason=reason,
            elapsed_sec=elapsed,
            params=params,
            final_equity=None,
            metrics=None,
            open_threshold=None,
            close_threshold=None,
            stdout_json=None,
        )

    try:
        bt = out["backtest"]
        metrics = bt["metrics"]
        final_eq = float(metrics["finalEquity"])
        open_thr = float(bt.get("openThreshold")) if bt.get("openThreshold") is not None else None
        close_thr = float(bt.get("closeThreshold")) if bt.get("closeThreshold") is not None else None
    except Exception as e:  # noqa: BLE001
        reason = f"unexpected json schema: {e}"
        return TrialResult(
            ok=False,
            reason=reason,
            elapsed_sec=elapsed,
            params=params,
            final_equity=None,
            metrics=None,
            open_threshold=None,
            close_threshold=None,
            stdout_json=out,
        )

    return TrialResult(
        ok=True,
        reason=None,
        elapsed_sec=elapsed,
        params=params,
        final_equity=final_eq,
        metrics=metrics,
        open_threshold=open_thr,
        close_threshold=close_thr,
        stdout_json=out,
    )


def fmt_opt_float(v: Optional[float]) -> str:
    if v is None:
        return "null"
    return f"{v:.8f}"


def fmt_opt_int(v: Optional[int]) -> str:
    if v is None:
        return "null"
    return str(v)


def trial_to_record(tr: TrialResult) -> Dict[str, Any]:
    r: Dict[str, Any] = {
        "ok": tr.ok,
        "reason": tr.reason,
        "elapsedSec": tr.elapsed_sec,
        "finalEquity": tr.final_equity,
        "openThreshold": tr.open_threshold,
        "closeThreshold": tr.close_threshold,
        "params": {
            "interval": tr.params.interval,
            "bars": tr.params.bars,
            "method": tr.params.method,
            "positioning": tr.params.positioning,
            "normalization": tr.params.normalization,
            "baseOpenThreshold": tr.params.base_open_threshold,
            "baseCloseThreshold": tr.params.base_close_threshold,
            "fee": tr.params.fee,
            "epochs": tr.params.epochs,
            "hiddenSize": tr.params.hidden_size,
            "learningRate": tr.params.learning_rate,
            "valRatio": tr.params.val_ratio,
            "patience": tr.params.patience,
            "gradClip": tr.params.grad_clip,
            "slippage": tr.params.slippage,
            "spread": tr.params.spread,
            "intrabarFill": tr.params.intrabar_fill,
            "stopLoss": tr.params.stop_loss,
            "takeProfit": tr.params.take_profit,
            "trailingStop": tr.params.trailing_stop,
            "maxDrawdown": tr.params.max_drawdown,
            "maxDailyLoss": tr.params.max_daily_loss,
            "maxOrderErrors": tr.params.max_order_errors,
            "kalmanDt": tr.params.kalman_dt,
            "kalmanProcessVar": tr.params.kalman_process_var,
            "kalmanMeasurementVar": tr.params.kalman_measurement_var,
            "kalmanZMin": tr.params.kalman_z_min,
            "kalmanZMax": tr.params.kalman_z_max,
            "maxHighVolProb": tr.params.max_high_vol_prob,
            "maxConformalWidth": tr.params.max_conformal_width,
            "maxQuantileWidth": tr.params.max_quantile_width,
            "confirmConformal": tr.params.confirm_conformal,
            "confirmQuantiles": tr.params.confirm_quantiles,
            "confidenceSizing": tr.params.confidence_sizing,
            "minPositionSize": tr.params.min_position_size,
        },
    }
    if tr.metrics is not None:
        r["metrics"] = tr.metrics
    return r


def pick_intervals(
    intervals: List[str],
    lookback_window: str,
    max_bars_cap: int,
) -> List[str]:
    out: List[str] = []
    for itv in intervals:
        try:
            lb = lookback_bars(itv, lookback_window)
        except Exception:
            continue
        if lb >= 2 and (lb + 3) <= max_bars_cap:
            out.append(itv)
    return out


def sample_params(
    rng: random.Random,
    intervals: List[str],
    bars_min: int,
    bars_max: int,
    open_threshold_min: float,
    open_threshold_max: float,
    close_threshold_min: float,
    close_threshold_max: float,
    fee_min: float,
    fee_max: float,
    p_long_short: float,
    p_intrabar_take_profit_first: float,
    epochs_min: int,
    epochs_max: int,
    hidden_min: int,
    hidden_max: int,
    lr_min: float,
    lr_max: float,
    val_min: float,
    val_max: float,
    patience_max: int,
    grad_clip_min: float,
    grad_clip_max: float,
    slippage_max: float,
    spread_max: float,
    kalman_dt_min: float,
    kalman_dt_max: float,
    kalman_process_var_min: float,
    kalman_process_var_max: float,
    kalman_measurement_var_min: float,
    kalman_measurement_var_max: float,
    kalman_z_min_min: float,
    kalman_z_min_max: float,
    kalman_z_max_min: float,
    kalman_z_max_max: float,
    p_disable_max_high_vol_prob: float,
    max_high_vol_prob_range: Tuple[float, float],
    p_disable_max_conformal_width: float,
    max_conformal_width_range: Tuple[float, float],
    p_disable_max_quantile_width: float,
    max_quantile_width_range: Tuple[float, float],
    p_confirm_conformal: float,
    p_confirm_quantiles: float,
    p_confidence_sizing: float,
    min_position_size_range: Tuple[float, float],
    stop_range: Tuple[float, float],
    take_range: Tuple[float, float],
    trail_range: Tuple[float, float],
    method_weights: Dict[str, float],
    normalization_choices: List[str],
    p_disable_stop: float,
    p_disable_tp: float,
    p_disable_trail: float,
    p_disable_max_dd: float,
    p_disable_max_dl: float,
    p_disable_max_oe: float,
    p_disable_grad_clip: float,
    max_dd_range: Tuple[float, float],
    max_dl_range: Tuple[float, float],
    max_oe_range: Tuple[int, int],
) -> TrialParams:
    interval = rng.choice(intervals)

    def sample_bars() -> int:
        # 0 means auto/all; otherwise explicit number within the configured range.
        if rng.random() < 0.25:
            return 0
        return rng.randint(bars_min, bars_max)

    # Weighted categorical draw for method.
    methods = list(method_weights.keys())
    weights = [max(0.0, float(method_weights[m])) for m in methods]
    s = sum(weights)
    if s <= 0:
        method = rng.choice(methods)
    else:
        r = rng.random() * s
        acc = 0.0
        method = methods[-1]
        for m, w in zip(methods, weights):
            acc += w
            if r <= acc:
                method = m
                break

    normalization = rng.choice(normalization_choices)
    base_open_threshold = log_uniform(rng, max(1e-12, open_threshold_min), max(1e-12, open_threshold_max))
    base_close_threshold = log_uniform(rng, max(1e-12, close_threshold_min), max(1e-12, close_threshold_max))
    fee = rng.uniform(max(0.0, fee_min), max(0.0, fee_max))
    epochs = rng.randint(epochs_min, epochs_max)
    hidden_size = rng.randint(hidden_min, hidden_max)
    learning_rate = log_uniform(rng, lr_min, lr_max)
    val_ratio = rng.uniform(val_min, val_max)
    patience = rng.randint(0, patience_max)
    grad_clip = maybe(rng, p_disable_grad_clip, lambda: log_uniform(rng, grad_clip_min, grad_clip_max))

    slippage = rng.uniform(0.0, max(0.0, slippage_max))
    spread = rng.uniform(0.0, max(0.0, spread_max))

    positioning = "long-short" if rng.random() < clamp(p_long_short, 0.0, 1.0) else "long-flat"
    intrabar_fill = (
        "take-profit-first" if rng.random() < clamp(p_intrabar_take_profit_first, 0.0, 1.0) else "stop-first"
    )

    kalman_dt = rng.uniform(max(1e-12, kalman_dt_min), max(1e-12, kalman_dt_max))
    kalman_process_var = log_uniform(rng, max(1e-12, kalman_process_var_min), max(1e-12, kalman_process_var_max))
    kalman_measurement_var = log_uniform(
        rng, max(1e-12, kalman_measurement_var_min), max(1e-12, kalman_measurement_var_max)
    )
    kalman_z_min = rng.uniform(max(0.0, kalman_z_min_min), max(0.0, kalman_z_min_max))
    kalman_z_max_lo = max(kalman_z_min, max(0.0, kalman_z_max_min))
    kalman_z_max = rng.uniform(kalman_z_max_lo, max(kalman_z_max_lo, kalman_z_max_max))

    hv_lo, hv_hi = max_high_vol_prob_range
    hv_lo, hv_hi = (min(hv_lo, hv_hi), max(hv_lo, hv_hi))
    max_high_vol_prob = maybe(
        rng,
        p_disable_max_high_vol_prob,
        lambda: clamp(rng.uniform(hv_lo, hv_hi), 0.0, 1.0),
    )
    cw_lo, cw_hi = max_conformal_width_range
    cw_lo, cw_hi = (min(cw_lo, cw_hi), max(cw_lo, cw_hi))
    max_conformal_width = maybe(
        rng,
        p_disable_max_conformal_width,
        lambda: log_uniform(rng, max(1e-12, cw_lo), max(1e-12, cw_hi)),
    )
    qw_lo, qw_hi = max_quantile_width_range
    qw_lo, qw_hi = (min(qw_lo, qw_hi), max(qw_lo, qw_hi))
    max_quantile_width = maybe(
        rng,
        p_disable_max_quantile_width,
        lambda: log_uniform(rng, max(1e-12, qw_lo), max(1e-12, qw_hi)),
    )
    confirm_conformal = rng.random() < clamp(p_confirm_conformal, 0.0, 1.0)
    confirm_quantiles = rng.random() < clamp(p_confirm_quantiles, 0.0, 1.0)
    confidence_sizing = rng.random() < clamp(p_confidence_sizing, 0.0, 1.0)
    mps_lo, mps_hi = min_position_size_range
    mps_lo, mps_hi = (min(mps_lo, mps_hi), max(mps_lo, mps_hi))
    min_position_size = clamp(rng.uniform(mps_lo, mps_hi), 0.0, 1.0) if confidence_sizing else 0.0

    stop_loss = maybe(rng, p_disable_stop, lambda: log_uniform(rng, stop_range[0], stop_range[1]))
    take_profit = maybe(rng, p_disable_tp, lambda: log_uniform(rng, take_range[0], take_range[1]))
    trailing_stop = maybe(rng, p_disable_trail, lambda: log_uniform(rng, trail_range[0], trail_range[1]))

    max_drawdown = maybe(rng, p_disable_max_dd, lambda: rng.uniform(max_dd_range[0], max_dd_range[1]))
    max_daily_loss = maybe(rng, p_disable_max_dl, lambda: rng.uniform(max_dl_range[0], max_dl_range[1]))

    def sample_oe() -> int:
        lo, hi = max_oe_range
        return rng.randint(lo, hi)

    max_order_errors = maybe(rng, p_disable_max_oe, sample_oe)

    return TrialParams(
        interval=interval,
        bars=sample_bars(),
        method=method,
        positioning=positioning,
        normalization=normalization,
        base_open_threshold=base_open_threshold,
        base_close_threshold=base_close_threshold,
        fee=fee,
        epochs=epochs,
        slippage=slippage,
        spread=spread,
        intrabar_fill=intrabar_fill,
        stop_loss=stop_loss,
        take_profit=take_profit,
        trailing_stop=trailing_stop,
        max_drawdown=max_drawdown,
        max_daily_loss=max_daily_loss,
        max_order_errors=max_order_errors,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        val_ratio=val_ratio,
        patience=patience,
        grad_clip=grad_clip,
        kalman_dt=kalman_dt,
        kalman_process_var=kalman_process_var,
        kalman_measurement_var=kalman_measurement_var,
        kalman_z_min=kalman_z_min,
        kalman_z_max=kalman_z_max,
        max_high_vol_prob=max_high_vol_prob,
        max_conformal_width=max_conformal_width,
        max_quantile_width=max_quantile_width,
        confirm_conformal=confirm_conformal,
        confirm_quantiles=confirm_quantiles,
        confidence_sizing=confidence_sizing,
        min_position_size=min_position_size,
    )


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Random-search optimizer for trader-hs cumulative equity (finalEquity).")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=str, help="CSV path for backtest (recommended for optimization).")
    src.add_argument("--binance-symbol", type=str, help="Binance symbol (requires network; slower).")
    parser.add_argument("--price-column", type=str, default="close", help="CSV column name for price (default: close).")
    parser.add_argument("--lookback-window", type=str, default="24h", help="Lookback window (default: 24h).")
    parser.add_argument("--backtest-ratio", type=float, default=0.2, help="Backtest holdout ratio (default: 0.2).")
    parser.add_argument("--tune-ratio", type=float, default=0.2, help="Tune ratio for --sweep-threshold (default: 0.2).")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials (default: 50).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42).")
    parser.add_argument("--timeout-sec", type=float, default=60.0, help="Per-trial timeout in seconds (default: 60).")
    parser.add_argument("--output", type=str, default="", help="Write JSONL trial records to this path.")
    parser.add_argument("--binary", type=str, default="", help="Path to trader-hs binary (auto-discovered via cabal if omitted).")
    parser.add_argument("--no-sweep-threshold", action="store_true", help="Disable internal threshold sweep (not recommended).")
    parser.add_argument("--disable-lstm-persistence", action="store_true", help="Disable LSTM weight caching (more reproducible).")
    parser.add_argument("--top-json", type=str, default="", help="Write the top-performing combos (JSON) for the UI chart.")

    parser.add_argument("--intervals", type=str, default=",".join(BINANCE_INTERVALS), help="Comma-separated intervals to sample.")
    parser.add_argument("--bars-min", type=int, default=0, help="Min bars when sampling explicit bars (default: 0=auto).")
    parser.add_argument("--bars-max", type=int, default=0, help="Max bars when sampling explicit bars (default: 0=auto-detect for CSV).")
    parser.add_argument("--epochs-min", type=int, default=0, help="Min epochs (default: 0).")
    parser.add_argument("--epochs-max", type=int, default=10, help="Max epochs (default: 10).")
    parser.add_argument("--slippage-max", type=float, default=0.001, help="Max slippage per side (default: 0.001).")
    parser.add_argument("--spread-max", type=float, default=0.001, help="Max total spread (default: 0.001).")
    parser.add_argument("--fee-min", type=float, default=0.0, help="Min fee when sampling (default: 0).")
    parser.add_argument("--fee-max", type=float, default=0.001, help="Max fee when sampling (default: 0.001).")

    parser.add_argument(
        "--open-threshold-min",
        type=float,
        default=1e-5,
        help="Min open-threshold used as the base (and as the value when --no-sweep-threshold is set) (default: 1e-5).",
    )
    parser.add_argument(
        "--open-threshold-max",
        type=float,
        default=1e-2,
        help="Max open-threshold used as the base (and as the value when --no-sweep-threshold is set) (default: 1e-2).",
    )
    parser.add_argument(
        "--close-threshold-min",
        type=float,
        default=1e-5,
        help="Min close-threshold used as the base (and as the value when --no-sweep-threshold is set) (default: 1e-5).",
    )
    parser.add_argument(
        "--close-threshold-max",
        type=float,
        default=1e-2,
        help="Max close-threshold used as the base (and as the value when --no-sweep-threshold is set) (default: 1e-2).",
    )

    parser.add_argument("--p-long-short", type=float, default=0.2, help="Probability of long-short positioning (default: 0.2).")
    parser.add_argument(
        "--p-intrabar-take-profit-first",
        type=float,
        default=0.2,
        help="Probability intrabar-fill is take-profit-first (default: 0.2).",
    )

    parser.add_argument("--kalman-dt-min", type=float, default=0.5, help="Min kalman-dt (default: 0.5).")
    parser.add_argument("--kalman-dt-max", type=float, default=2.0, help="Max kalman-dt (default: 2.0).")
    parser.add_argument("--kalman-process-var-min", type=float, default=1e-7, help="Min kalman-process-var (default: 1e-7).")
    parser.add_argument("--kalman-process-var-max", type=float, default=1e-3, help="Max kalman-process-var (default: 1e-3).")
    parser.add_argument(
        "--kalman-measurement-var-min", type=float, default=1e-6, help="Min kalman-measurement-var (default: 1e-6)."
    )
    parser.add_argument(
        "--kalman-measurement-var-max", type=float, default=1e-1, help="Max kalman-measurement-var (default: 1e-1)."
    )

    parser.add_argument("--kalman-z-min-min", type=float, default=0.0, help="Min kalman-z-min (default: 0).")
    parser.add_argument("--kalman-z-min-max", type=float, default=2.0, help="Max kalman-z-min (default: 2).")
    parser.add_argument("--kalman-z-max-min", type=float, default=0.0, help="Min kalman-z-max (default: 0).")
    parser.add_argument("--kalman-z-max-max", type=float, default=6.0, help="Max kalman-z-max (default: 6).")

    parser.add_argument(
        "--p-disable-max-high-vol-prob",
        type=float,
        default=0.9,
        help="Probability max-high-vol-prob is disabled (default: 0.9).",
    )
    parser.add_argument("--max-high-vol-prob-min", type=float, default=0.2, help="Min max-high-vol-prob when enabled (default: 0.2).")
    parser.add_argument("--max-high-vol-prob-max", type=float, default=0.95, help="Max max-high-vol-prob when enabled (default: 0.95).")

    parser.add_argument(
        "--p-disable-max-conformal-width",
        type=float,
        default=0.95,
        help="Probability max-conformal-width is disabled (default: 0.95).",
    )
    parser.add_argument(
        "--max-conformal-width-min", type=float, default=0.002, help="Min max-conformal-width when enabled (default: 0.002)."
    )
    parser.add_argument(
        "--max-conformal-width-max", type=float, default=0.20, help="Max max-conformal-width when enabled (default: 0.20)."
    )

    parser.add_argument(
        "--p-disable-max-quantile-width",
        type=float,
        default=0.95,
        help="Probability max-quantile-width is disabled (default: 0.95).",
    )
    parser.add_argument(
        "--max-quantile-width-min", type=float, default=0.002, help="Min max-quantile-width when enabled (default: 0.002)."
    )
    parser.add_argument(
        "--max-quantile-width-max", type=float, default=0.20, help="Max max-quantile-width when enabled (default: 0.20)."
    )

    parser.add_argument("--p-confirm-conformal", type=float, default=0.1, help="Probability confirm-conformal is enabled (default: 0.1).")
    parser.add_argument("--p-confirm-quantiles", type=float, default=0.1, help="Probability confirm-quantiles is enabled (default: 0.1).")
    parser.add_argument("--p-confidence-sizing", type=float, default=0.15, help="Probability confidence-sizing is enabled (default: 0.15).")
    parser.add_argument("--min-position-size-min", type=float, default=0.0, help="Min min-position-size when enabled (default: 0).")
    parser.add_argument("--min-position-size-max", type=float, default=0.5, help="Max min-position-size when enabled (default: 0.5).")

    parser.add_argument("--stop-min", type=float, default=0.002, help="Min stop-loss when enabled (default: 0.002).")
    parser.add_argument("--stop-max", type=float, default=0.20, help="Max stop-loss when enabled (default: 0.20).")
    parser.add_argument("--tp-min", type=float, default=0.002, help="Min take-profit when enabled (default: 0.002).")
    parser.add_argument("--tp-max", type=float, default=0.20, help="Max take-profit when enabled (default: 0.20).")
    parser.add_argument("--trail-min", type=float, default=0.002, help="Min trailing-stop when enabled (default: 0.002).")
    parser.add_argument("--trail-max", type=float, default=0.20, help="Max trailing-stop when enabled (default: 0.20).")

    parser.add_argument("--p-disable-stop", type=float, default=0.5, help="Probability stop-loss is disabled (default: 0.5).")
    parser.add_argument("--p-disable-tp", type=float, default=0.5, help="Probability take-profit is disabled (default: 0.5).")
    parser.add_argument("--p-disable-trail", type=float, default=0.6, help="Probability trailing-stop is disabled (default: 0.6).")

    parser.add_argument("--p-disable-max-dd", type=float, default=0.9, help="Probability max-drawdown is disabled (default: 0.9).")
    parser.add_argument("--p-disable-max-dl", type=float, default=0.9, help="Probability max-daily-loss is disabled (default: 0.9).")
    parser.add_argument("--p-disable-max-oe", type=float, default=0.95, help="Probability max-order-errors is disabled (default: 0.95).")
    parser.add_argument("--max-dd-min", type=float, default=0.05, help="Min max-drawdown when enabled (default: 0.05).")
    parser.add_argument("--max-dd-max", type=float, default=0.50, help="Max max-drawdown when enabled (default: 0.50).")
    parser.add_argument("--max-dl-min", type=float, default=0.02, help="Min max-daily-loss when enabled (default: 0.02).")
    parser.add_argument("--max-dl-max", type=float, default=0.30, help="Max max-daily-loss when enabled (default: 0.30).")
    parser.add_argument("--max-oe-min", type=int, default=1, help="Min max-order-errors when enabled (default: 1).")
    parser.add_argument("--max-oe-max", type=int, default=10, help="Max max-order-errors when enabled (default: 10).")

    parser.add_argument("--method-weight-11", type=float, default=1.0, help="Sampling weight for method=11 (default: 1.0).")
    parser.add_argument("--method-weight-10", type=float, default=2.0, help="Sampling weight for method=10 (default: 2.0).")
    parser.add_argument("--method-weight-01", type=float, default=1.0, help="Sampling weight for method=01 (default: 1.0).")
    parser.add_argument(
        "--normalizations",
        type=str,
        default="none,minmax,standard,log",
        help="Comma-separated normalization choices (default: none,minmax,standard,log).",
    )

    parser.add_argument("--hidden-size-min", type=int, default=8, help="Min LSTM hidden size to sample (default: 8).")
    parser.add_argument("--hidden-size-max", type=int, default=64, help="Max LSTM hidden size to sample (default: 64).")
    parser.add_argument("--lr-min", type=float, default=1e-4, help="Min learning rate (default: 1e-4).")
    parser.add_argument("--lr-max", type=float, default=1e-2, help="Max learning rate (default: 1e-2).")
    parser.add_argument("--val-ratio-min", type=float, default=0.1, help="Min validation ratio (default: 0.1).")
    parser.add_argument("--val-ratio-max", type=float, default=0.4, help="Max validation ratio (default: 0.4).")
    parser.add_argument("--patience-max", type=int, default=20, help="Max early-stopping patience to sample (default: 20).")
    parser.add_argument("--grad-clip-min", type=float, default=0.001, help="Min grad clip when enabled (default: 0.001).")
    parser.add_argument("--grad-clip-max", type=float, default=1.0, help="Max grad clip when enabled (default: 1.0).")
    parser.add_argument("--p-disable-grad-clip", type=float, default=0.7, help="Probability grad clipping is disabled (default: 0.7).")

    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    haskell_dir = root

    trader_bin = Path(args.binary).expanduser() if args.binary else discover_trader_bin(haskell_dir)

    intervals_in = [s.strip() for s in str(args.intervals).split(",") if s.strip()]
    if not intervals_in:
        print("No intervals provided.", file=sys.stderr)
        return 2

    max_bars_cap = 1000
    csv_n = None
    if args.data:
        p = Path(args.data)
        if not p.exists():
            print(f"--data not found: {p}", file=sys.stderr)
            return 2
        # csv rows includes header; bars are data rows.
        csv_n = max(0, count_csv_rows(p) - 1)
        max_bars_cap = max(2, csv_n)

    intervals = pick_intervals(intervals_in, args.lookback_window, max_bars_cap=max_bars_cap)
    if not intervals:
        print(
            f"No feasible intervals for lookback-window={args.lookback_window!r} and max bars={max_bars_cap}.",
            file=sys.stderr,
        )
        return 2

    bars_max = int(args.bars_max)
    if bars_max <= 0:
        bars_max = max_bars_cap
    bars_min = int(args.bars_min)
    if bars_min <= 0:
        # Ensure we can at least run splitTrainBacktest for the worst-case feasible interval.
        # (Still allow bars=0 = auto/all, handled separately.)
        worst_lb = max(lookback_bars(itv, args.lookback_window) for itv in intervals)
        bars_min = min(bars_max, max(10, worst_lb + 5))
    bars_min = max(2, min(bars_min, bars_max))

    epochs_min = max(0, int(args.epochs_min))
    epochs_max = max(epochs_min, int(args.epochs_max))

    hidden_min = max(1, int(args.hidden_size_min))
    hidden_max = max(hidden_min, int(args.hidden_size_max))
    lr_min = max(1e-9, float(args.lr_min))
    lr_max = max(lr_min, float(args.lr_max))
    val_min = clamp(float(args.val_ratio_min), 0.0, 0.9)
    val_max = max(val_min, float(args.val_ratio_max))
    patience_max = max(0, int(args.patience_max))
    grad_clip_min = max(1e-9, float(args.grad_clip_min))
    grad_clip_max = max(grad_clip_min, float(args.grad_clip_max))

    stop_min = float(args.stop_min)
    stop_max = float(args.stop_max)
    tp_min = float(args.tp_min)
    tp_max = float(args.tp_max)
    trail_min = float(args.trail_min)
    trail_max = float(args.trail_max)

    fee_min = max(0.0, float(args.fee_min))
    fee_max = max(fee_min, float(args.fee_max))

    open_threshold_min = max(1e-12, float(args.open_threshold_min))
    open_threshold_max = max(open_threshold_min, float(args.open_threshold_max))
    close_threshold_min = max(1e-12, float(args.close_threshold_min))
    close_threshold_max = max(close_threshold_min, float(args.close_threshold_max))

    p_long_short = clamp(float(args.p_long_short), 0.0, 1.0)
    p_intrabar_take_profit_first = clamp(float(args.p_intrabar_take_profit_first), 0.0, 1.0)

    kalman_dt_min = max(1e-12, float(args.kalman_dt_min))
    kalman_dt_max = max(kalman_dt_min, float(args.kalman_dt_max))
    kalman_process_var_min = max(1e-12, float(args.kalman_process_var_min))
    kalman_process_var_max = max(kalman_process_var_min, float(args.kalman_process_var_max))
    kalman_measurement_var_min = max(1e-12, float(args.kalman_measurement_var_min))
    kalman_measurement_var_max = max(kalman_measurement_var_min, float(args.kalman_measurement_var_max))

    kalman_z_min_min = max(0.0, float(args.kalman_z_min_min))
    kalman_z_min_max = max(kalman_z_min_min, float(args.kalman_z_min_max))
    kalman_z_max_min = max(0.0, float(args.kalman_z_max_min))
    kalman_z_max_max = max(kalman_z_max_min, float(args.kalman_z_max_max))

    p_disable_max_high_vol_prob = clamp(float(args.p_disable_max_high_vol_prob), 0.0, 1.0)
    max_high_vol_prob_range = (float(args.max_high_vol_prob_min), float(args.max_high_vol_prob_max))
    p_disable_max_conformal_width = clamp(float(args.p_disable_max_conformal_width), 0.0, 1.0)
    max_conformal_width_range = (float(args.max_conformal_width_min), float(args.max_conformal_width_max))
    p_disable_max_quantile_width = clamp(float(args.p_disable_max_quantile_width), 0.0, 1.0)
    max_quantile_width_range = (float(args.max_quantile_width_min), float(args.max_quantile_width_max))

    p_confirm_conformal = clamp(float(args.p_confirm_conformal), 0.0, 1.0)
    p_confirm_quantiles = clamp(float(args.p_confirm_quantiles), 0.0, 1.0)
    p_confidence_sizing = clamp(float(args.p_confidence_sizing), 0.0, 1.0)
    min_position_size_range = (float(args.min_position_size_min), float(args.min_position_size_max))

    method_weights = {
        "11": float(args.method_weight_11),
        "10": float(args.method_weight_10),
        "01": float(args.method_weight_01),
    }
    normalization_choices = [s.strip() for s in str(args.normalizations).split(",") if s.strip()]
    if not normalization_choices:
        print("No normalizations provided.", file=sys.stderr)
        return 2

    base_args: List[str] = []
    if args.data:
        data_path = Path(args.data).expanduser().resolve()
        base_args += ["--data", str(data_path)]
        base_args += ["--price-column", args.price_column]
    else:
        base_args += ["--binance-symbol", args.binance_symbol]

    base_args += ["--lookback-window", args.lookback_window]
    base_args += ["--backtest-ratio", f"{float(args.backtest_ratio):.6f}"]
    # Fix seed so model init is comparable across trials.
    base_args += ["--seed", str(int(args.seed))]

    rng = random.Random(int(args.seed))

    out_path = Path(args.output).expanduser() if args.output else None
    out_fh = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_fh = out_path.open("w", encoding="utf-8")

    best: Optional[TrialResult] = None

    use_sweep_threshold = not bool(args.no_sweep_threshold)
    disable_lstm_persistence = bool(args.disable_lstm_persistence)

    trials = int(args.trials)
    records: List[TrialResult] = []
    for i in range(1, trials + 1):
        params = sample_params(
            rng=rng,
            intervals=intervals,
            bars_min=bars_min,
            bars_max=bars_max,
            open_threshold_min=open_threshold_min,
            open_threshold_max=open_threshold_max,
            close_threshold_min=close_threshold_min,
            close_threshold_max=close_threshold_max,
            fee_min=fee_min,
            fee_max=fee_max,
            p_long_short=p_long_short,
            p_intrabar_take_profit_first=p_intrabar_take_profit_first,
            epochs_min=epochs_min,
            epochs_max=epochs_max,
            hidden_min=hidden_min,
            hidden_max=hidden_max,
            lr_min=lr_min,
            lr_max=lr_max,
            val_min=val_min,
            val_max=val_max,
            patience_max=patience_max,
            grad_clip_min=grad_clip_min,
            grad_clip_max=grad_clip_max,
            slippage_max=float(args.slippage_max),
            spread_max=float(args.spread_max),
            kalman_dt_min=kalman_dt_min,
            kalman_dt_max=kalman_dt_max,
            kalman_process_var_min=kalman_process_var_min,
            kalman_process_var_max=kalman_process_var_max,
            kalman_measurement_var_min=kalman_measurement_var_min,
            kalman_measurement_var_max=kalman_measurement_var_max,
            kalman_z_min_min=kalman_z_min_min,
            kalman_z_min_max=kalman_z_min_max,
            kalman_z_max_min=kalman_z_max_min,
            kalman_z_max_max=kalman_z_max_max,
            p_disable_max_high_vol_prob=p_disable_max_high_vol_prob,
            max_high_vol_prob_range=max_high_vol_prob_range,
            p_disable_max_conformal_width=p_disable_max_conformal_width,
            max_conformal_width_range=max_conformal_width_range,
            p_disable_max_quantile_width=p_disable_max_quantile_width,
            max_quantile_width_range=max_quantile_width_range,
            p_confirm_conformal=p_confirm_conformal,
            p_confirm_quantiles=p_confirm_quantiles,
            p_confidence_sizing=p_confidence_sizing,
            min_position_size_range=min_position_size_range,
            stop_range=(stop_min, stop_max),
            take_range=(tp_min, tp_max),
            trail_range=(trail_min, trail_max),
            method_weights=method_weights,
            normalization_choices=normalization_choices,
            p_disable_stop=clamp(float(args.p_disable_stop), 0.0, 1.0),
            p_disable_tp=clamp(float(args.p_disable_tp), 0.0, 1.0),
            p_disable_trail=clamp(float(args.p_disable_trail), 0.0, 1.0),
            p_disable_max_dd=clamp(float(args.p_disable_max_dd), 0.0, 1.0),
            p_disable_max_dl=clamp(float(args.p_disable_max_dl), 0.0, 1.0),
            p_disable_max_oe=clamp(float(args.p_disable_max_oe), 0.0, 1.0),
            p_disable_grad_clip=clamp(float(args.p_disable_grad_clip), 0.0, 1.0),
            max_dd_range=(float(args.max_dd_min), float(args.max_dd_max)),
            max_dl_range=(float(args.max_dl_min), float(args.max_dl_max)),
            max_oe_range=(int(args.max_oe_min), int(args.max_oe_max)),
        )

        tr = run_trial(
            trader_bin=trader_bin,
            base_args=base_args,
            params=params,
            tune_ratio=float(args.tune_ratio),
            use_sweep_threshold=use_sweep_threshold,
            timeout_sec=float(args.timeout_sec),
            disable_lstm_persistence=disable_lstm_persistence,
        )

        if out_fh is not None:
            out_fh.write(json.dumps(trial_to_record(tr), sort_keys=True) + "\n")
            out_fh.flush()

        records.append(tr)

        if tr.ok and tr.final_equity is not None:
            if best is None or tr.final_equity > (best.final_equity or -1e18):
                best = tr

        status = "OK" if tr.ok else "FAIL"
        eq = f"{tr.final_equity:.6f}x" if tr.final_equity is not None else "-"
        print(
            f"[{i:>4}/{trials}] {status} eq={eq} t={tr.elapsed_sec:.2f}s "
            f"interval={params.interval} bars={params.bars} method={params.method} "
            f"norm={params.normalization} epochs={params.epochs} "
            f"slip={params.slippage:.6f} spr={params.spread:.6f} "
            f"sl={fmt_opt_float(params.stop_loss)} tp={fmt_opt_float(params.take_profit)} trail={fmt_opt_float(params.trailing_stop)} "
            f"maxDD={fmt_opt_float(params.max_drawdown)} maxDL={fmt_opt_float(params.max_daily_loss)} maxOE={fmt_opt_int(params.max_order_errors)}"
            + (f" ({tr.reason})" if tr.reason else ""),
            flush=True,
        )

    if out_fh is not None:
        out_fh.close()

    if best is None:
        print("No successful trials.", file=sys.stderr)
        return 1

    b = best
    assert b.final_equity is not None
    print("\nBest:")
    print(f"  finalEquity: {b.final_equity:.8f}x")
    print(f"  interval:    {b.params.interval}")
    print(f"  bars:        {b.params.bars}")
    print(f"  method:      {b.params.method}")
    print(f"  positioning: {b.params.positioning}")
    print(f"  thresholds:  open={b.open_threshold} close={b.close_threshold} (from sweep)")
    print(f"  base thresholds: open={b.params.base_open_threshold} close={b.params.base_close_threshold}")
    print(f"  normalization: {b.params.normalization}")
    print(f"  epochs:        {b.params.epochs}")
    print(f"  hiddenSize:    {b.params.hidden_size}")
    print(f"  lr:            {b.params.learning_rate}")
    print(f"  valRatio:      {b.params.val_ratio}")
    print(f"  patience:      {b.params.patience}")
    print(f"  gradClip:      {b.params.grad_clip}")
    print(f"  fee:           {b.params.fee}")
    print(f"  slippage:      {b.params.slippage}")
    print(f"  spread:        {b.params.spread}")
    print(f"  intrabarFill:  {b.params.intrabar_fill}")
    print(f"  stopLoss:      {b.params.stop_loss}")
    print(f"  takeProfit:    {b.params.take_profit}")
    print(f"  trailingStop:  {b.params.trailing_stop}")
    print(f"  maxDrawdown:   {b.params.max_drawdown}")
    print(f"  maxDailyLoss:  {b.params.max_daily_loss}")
    print(f"  maxOrderErrors:{b.params.max_order_errors}")
    print(f"  kalmanDt:            {b.params.kalman_dt}")
    print(f"  kalmanProcessVar:    {b.params.kalman_process_var}")
    print(f"  kalmanMeasurementVar:{b.params.kalman_measurement_var}")
    print(f"  kalmanZMin:          {b.params.kalman_z_min}")
    print(f"  kalmanZMax:          {b.params.kalman_z_max}")
    print(f"  maxHighVolProb:      {b.params.max_high_vol_prob}")
    print(f"  maxConformalWidth:   {b.params.max_conformal_width}")
    print(f"  maxQuantileWidth:    {b.params.max_quantile_width}")
    print(f"  confirmConformal:    {b.params.confirm_conformal}")
    print(f"  confirmQuantiles:    {b.params.confirm_quantiles}")
    print(f"  confidenceSizing:    {b.params.confidence_sizing}")
    print(f"  minPositionSize:     {b.params.min_position_size}")

    print("\nRepro command:")
    env_prefix = "TRADER_LSTM_WEIGHTS_DIR='' " if disable_lstm_persistence else ""
    cmd = build_command(
        trader_bin=trader_bin,
        base_args=base_args,
        params=b.params,
        tune_ratio=float(args.tune_ratio),
        use_sweep_threshold=use_sweep_threshold,
    )
    print("  " + env_prefix + " ".join(cmd))
    if args.top_json:
        successful = [tr for tr in records if tr.ok and tr.final_equity is not None]
        successful_sorted = sorted(successful, key=lambda tr: tr.final_equity or 0, reverse=True)
        source = "binance" if args.binance_symbol else "csv"
        combos = []
        for rank, tr in enumerate(successful_sorted[:10], start=1):
            combos.append(
                {
                    "rank": rank,
                    "finalEquity": tr.final_equity,
                    "openThreshold": tr.open_threshold,
                    "closeThreshold": tr.close_threshold,
                    "source": source,
                    "params": {
                        "interval": tr.params.interval,
                        "bars": tr.params.bars,
                        "method": tr.params.method,
                        "positioning": tr.params.positioning,
                        "normalization": tr.params.normalization,
                        "baseOpenThreshold": tr.params.base_open_threshold,
                        "baseCloseThreshold": tr.params.base_close_threshold,
                        "fee": tr.params.fee,
                        "epochs": tr.params.epochs,
                        "hiddenSize": tr.params.hidden_size,
                        "learningRate": tr.params.learning_rate,
                        "valRatio": tr.params.val_ratio,
                        "patience": tr.params.patience,
                        "gradClip": tr.params.grad_clip,
                        "slippage": tr.params.slippage,
                        "spread": tr.params.spread,
                        "intrabarFill": tr.params.intrabar_fill,
                        "stopLoss": tr.params.stop_loss,
                        "takeProfit": tr.params.take_profit,
                        "trailingStop": tr.params.trailing_stop,
                        "maxDrawdown": tr.params.max_drawdown,
                        "maxDailyLoss": tr.params.max_daily_loss,
                        "maxOrderErrors": tr.params.max_order_errors,
                        "kalmanDt": tr.params.kalman_dt,
                        "kalmanProcessVar": tr.params.kalman_process_var,
                        "kalmanMeasurementVar": tr.params.kalman_measurement_var,
                        "kalmanZMin": tr.params.kalman_z_min,
                        "kalmanZMax": tr.params.kalman_z_max,
                        "maxHighVolProb": tr.params.max_high_vol_prob,
                        "maxConformalWidth": tr.params.max_conformal_width,
                        "maxQuantileWidth": tr.params.max_quantile_width,
                        "confirmConformal": tr.params.confirm_conformal,
                        "confirmQuantiles": tr.params.confirm_quantiles,
                        "confidenceSizing": tr.params.confidence_sizing,
                        "minPositionSize": tr.params.min_position_size,
                    },
                }
            )
        export = {
            "generatedAtMs": int(time.time() * 1000),
            "source": "optimize_equity.py",
            "combos": combos,
        }
        top_path = Path(args.top_json).expanduser()
        top_path.parent.mkdir(parents=True, exist_ok=True)
        top_path.write_text(json.dumps(export, indent=2), encoding="utf-8")
        print(f"Wrote top combos JSON: {top_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

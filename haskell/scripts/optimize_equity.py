#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, replace
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


def normalize_header_name(raw: str) -> str:
    return "".join(ch for ch in raw.strip().lower() if ch.isalnum())


def detect_high_low_columns(path: Path) -> Tuple[Optional[str], Optional[str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
    except Exception:
        return None, None
    if not header:
        return None, None
    normalized: Dict[str, str] = {}
    for col in header:
        key = normalize_header_name(str(col))
        if key and key not in normalized:
            normalized[key] = str(col)
    high = next((normalized.get(k) for k in ("high", "highprice", "highpx") if k in normalized), None)
    low = next((normalized.get(k) for k in ("low", "lowprice", "lowpx") if k in normalized), None)
    return high, low


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


def metric_float(metrics: Optional[Dict[str, Any]], key: str, default: float = 0.0) -> float:
    if not metrics:
        return default
    v = metrics.get(key)
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return default


def metric_int(metrics: Optional[Dict[str, Any]], key: str, default: int = 0) -> int:
    if not metrics:
        return default
    v = metrics.get(key)
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float) and math.isfinite(v):
        return int(v)
    return default


def metric_profit_factor(metrics: Optional[Dict[str, Any]]) -> float:
    if not metrics:
        return 0.0
    v = metrics.get("profitFactor")
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    gross_profit = metric_float(metrics, "grossProfit", 0.0)
    gross_loss = metric_float(metrics, "grossLoss", 0.0)
    if gross_loss > 0:
        return gross_profit / gross_loss
    if gross_profit > 0:
        return float("inf")
    return 0.0


def extract_walk_forward_summary(stdout_json: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(stdout_json, dict):
        return None
    bt = stdout_json.get("backtest")
    if not isinstance(bt, dict):
        return None
    wf = bt.get("walkForward")
    if not isinstance(wf, dict):
        return None
    summary = wf.get("summary")
    if not isinstance(summary, dict):
        return None
    return summary


def coerce_float_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            v = float(s)
        except ValueError:
            return None
        return v if math.isfinite(v) else None
    return None


def coerce_int_value(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value) if math.isfinite(value) else None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return int(float(s))
        except ValueError:
            return None
    return None


def apply_quality_preset(args: argparse.Namespace) -> None:
    args.trials = max(int(args.trials), 500)
    args.min_round_trips = max(int(args.min_round_trips), 5)
    args.open_threshold_max = max(float(args.open_threshold_max), 5e-2)
    args.close_threshold_max = max(float(args.close_threshold_max), 5e-2)
    args.min_win_rate = max(float(args.min_win_rate), 0.45)
    args.min_profit_factor = max(float(args.min_profit_factor), 1.1)
    args.min_exposure = max(float(args.min_exposure), 0.05)
    args.min_sharpe = max(float(args.min_sharpe), 0.25)
    args.min_wf_sharpe_mean = max(float(args.min_wf_sharpe_mean), 0.2)
    if float(args.max_wf_sharpe_std) <= 0:
        args.max_wf_sharpe_std = 1.5
    args.min_signal_to_noise_min = max(float(args.min_signal_to_noise_min), 0.2)
    args.min_signal_to_noise_max = max(float(args.min_signal_to_noise_max), 1.0)
    args.epochs_max = max(int(args.epochs_max), 50)
    args.hidden_size_max = max(int(args.hidden_size_max), 128)
    args.lr_max = max(float(args.lr_max), 5e-2)
    args.backtest_ratio = min(float(args.backtest_ratio), 0.10)
    args.tune_ratio = min(float(args.tune_ratio), 0.15)
    if str(args.objective).strip().lower() in ("final-equity", "final_equity", "finalequity"):
        args.objective = "equity-dd-turnover"
    args.penalty_turnover = max(float(args.penalty_turnover), 0.1)
    if int(args.bars_max) > 0:
        args.bars_max = 0
    args.auto_high_low = True
    args.walk_forward_folds_min = max(int(args.walk_forward_folds_min), 3)
    args.walk_forward_folds_max = max(int(args.walk_forward_folds_max), int(args.walk_forward_folds_min))
    if str(args.interval).strip():
        args.interval = ""
        args.intervals = ",".join(BINANCE_INTERVALS)


def objective_score(
    metrics: Dict[str, Any],
    objective: str,
    penalty_max_drawdown: float,
    penalty_turnover: float,
) -> float:
    final_eq = metric_float(metrics, "finalEquity", 0.0)
    max_dd = metric_float(metrics, "maxDrawdown", 0.0)
    sharpe = metric_float(metrics, "sharpe", 0.0)
    ann_ret = metric_float(metrics, "annualizedReturn", 0.0)
    turnover = metric_float(metrics, "turnover", 0.0)

    obj = str(objective).strip().lower()
    if obj in ("final-equity", "final_equity", "finalequity"):
        return final_eq
    if obj == "sharpe":
        return sharpe
    if obj == "calmar":
        return ann_ret / max(1e-12, max_dd)
    if obj in ("equity-dd", "equity_maxdd", "equity-dd-only"):
        return final_eq - float(penalty_max_drawdown) * max_dd
    if obj in ("equity-dd-turnover", "equity-dd-ops", "equity-dd-turn"):
        return final_eq - float(penalty_max_drawdown) * max_dd - float(penalty_turnover) * turnover
    raise ValueError(f"unknown objective: {objective!r}")


@dataclass(frozen=True)
class TrialParams:
    interval: str
    bars: int  # 0 = auto/full; otherwise explicit bar count
    method: str  # "11" | "10" | "01"
    blend_weight: float
    positioning: str  # "long-flat" | "long-short"
    normalization: str
    base_open_threshold: float
    base_close_threshold: float
    min_hold_bars: int
    cooldown_bars: int
    max_hold_bars: Optional[int]
    min_edge: float
    min_signal_to_noise: float
    edge_buffer: float
    cost_aware_edge: bool
    trend_lookback: int
    max_position_size: float
    vol_target: Optional[float]
    vol_lookback: int
    vol_ewma_alpha: Optional[float]
    vol_floor: float
    vol_scale_max: float
    max_volatility: Optional[float]
    periods_per_year: Optional[float]
    kalman_market_top_n: int
    fee: float
    epochs: int
    hidden_size: int
    learning_rate: float
    val_ratio: float
    patience: int
    walk_forward_folds: int
    tune_stress_vol_mult: float
    tune_stress_shock: float
    tune_stress_weight: float
    grad_clip: Optional[float]
    slippage: float
    spread: float
    intrabar_fill: str  # "stop-first" | "take-profit-first"
    stop_loss: Optional[float]
    take_profit: Optional[float]
    trailing_stop: Optional[float]
    stop_loss_vol_mult: Optional[float]
    take_profit_vol_mult: Optional[float]
    trailing_stop_vol_mult: Optional[float]
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
    eligible: bool = False
    filter_reason: Optional[str] = None
    objective: str = "final-equity"
    score: Optional[float] = None


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
    is_binance = "--binance-symbol" in base_args
    if params.bars <= 0:
        cmd += ["--bars", "auto" if is_binance else "0"]
    else:
        cmd += ["--bars", str(params.bars)]
    cmd += ["--positioning", params.positioning]
    cmd += ["--method", params.method]
    cmd += ["--blend-weight", f"{clamp(float(params.blend_weight), 0.0, 1.0):.6f}"]
    cmd += ["--normalization", params.normalization]
    cmd += ["--open-threshold", f"{max(0.0, params.base_open_threshold):.12g}"]
    cmd += ["--close-threshold", f"{max(0.0, params.base_close_threshold):.12g}"]
    cmd += ["--min-hold-bars", str(max(0, params.min_hold_bars))]
    cmd += ["--cooldown-bars", str(max(0, params.cooldown_bars))]
    if params.max_hold_bars is not None:
        cmd += ["--max-hold-bars", str(max(1, params.max_hold_bars))]
    cmd += ["--min-edge", f"{max(0.0, params.min_edge):.12g}"]
    cmd += ["--min-signal-to-noise", f"{max(0.0, params.min_signal_to_noise):.12g}"]
    cmd += ["--edge-buffer", f"{max(0.0, params.edge_buffer):.12g}"]
    if params.cost_aware_edge:
        cmd += ["--cost-aware-edge"]
    cmd += ["--trend-lookback", str(max(0, params.trend_lookback))]
    cmd += ["--max-position-size", f"{max(0.0, params.max_position_size):.12g}"]
    if params.vol_target is not None:
        cmd += ["--vol-target", f"{max(1e-12, params.vol_target):.12g}"]
    cmd += ["--vol-lookback", str(max(0, params.vol_lookback))]
    if params.vol_ewma_alpha is not None:
        cmd += ["--vol-ewma-alpha", f"{clamp(float(params.vol_ewma_alpha), 0.0, 1.0):.6f}"]
    cmd += ["--vol-floor", f"{max(0.0, params.vol_floor):.12g}"]
    cmd += ["--vol-scale-max", f"{max(0.0, params.vol_scale_max):.12g}"]
    if params.max_volatility is not None:
        cmd += ["--max-volatility", f"{max(1e-12, params.max_volatility):.12g}"]
    if params.periods_per_year is not None:
        cmd += ["--periods-per-year", f"{max(1e-12, params.periods_per_year):.6f}"]
    cmd += ["--fee", f"{max(0.0, params.fee):.12g}"]
    cmd += ["--epochs", str(params.epochs)]
    cmd += ["--hidden-size", str(params.hidden_size)]
    cmd += ["--lr", f"{params.learning_rate:.8f}"]
    cmd += ["--val-ratio", f"{params.val_ratio:.6f}"]
    cmd += ["--patience", str(params.patience)]
    cmd += ["--walk-forward-folds", str(max(1, params.walk_forward_folds))]
    cmd += ["--tune-stress-vol-mult", f"{max(1e-12, params.tune_stress_vol_mult):.6f}"]
    cmd += ["--tune-stress-shock", f"{params.tune_stress_shock:.6f}"]
    cmd += ["--tune-stress-weight", f"{max(0.0, params.tune_stress_weight):.6f}"]
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
    if params.stop_loss_vol_mult is not None:
        cmd += ["--stop-loss-vol-mult", f"{params.stop_loss_vol_mult:.8f}"]
    if params.take_profit_vol_mult is not None:
        cmd += ["--take-profit-vol-mult", f"{params.take_profit_vol_mult:.8f}"]
    if params.trailing_stop_vol_mult is not None:
        cmd += ["--trailing-stop-vol-mult", f"{params.trailing_stop_vol_mult:.8f}"]
    if params.max_drawdown is not None:
        cmd += ["--max-drawdown", f"{params.max_drawdown:.8f}"]
    if params.max_daily_loss is not None:
        cmd += ["--max-daily-loss", f"{params.max_daily_loss:.8f}"]
    if params.max_order_errors is not None:
        cmd += ["--max-order-errors", str(params.max_order_errors)]
    cmd += ["--kalman-market-top-n", str(max(0, params.kalman_market_top_n))]
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


def normalize_symbol(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = str(raw).strip().upper()
    return s if s else None


def trial_to_record(tr: TrialResult, symbol_label: Optional[str]) -> Dict[str, Any]:
    r: Dict[str, Any] = {
        "ok": tr.ok,
        "eligible": tr.eligible,
        "objective": tr.objective,
        "score": tr.score,
        "filterReason": tr.filter_reason,
        "reason": tr.reason,
        "elapsedSec": tr.elapsed_sec,
        "finalEquity": tr.final_equity,
        "openThreshold": tr.open_threshold,
        "closeThreshold": tr.close_threshold,
        "params": {
            "interval": tr.params.interval,
            "bars": tr.params.bars,
            "method": tr.params.method,
            "blendWeight": tr.params.blend_weight,
            "positioning": tr.params.positioning,
            "normalization": tr.params.normalization,
            "baseOpenThreshold": tr.params.base_open_threshold,
            "baseCloseThreshold": tr.params.base_close_threshold,
            "minHoldBars": tr.params.min_hold_bars,
            "cooldownBars": tr.params.cooldown_bars,
            "maxHoldBars": tr.params.max_hold_bars,
            "minEdge": tr.params.min_edge,
            "minSignalToNoise": tr.params.min_signal_to_noise,
            "edgeBuffer": tr.params.edge_buffer,
            "costAwareEdge": tr.params.cost_aware_edge,
            "trendLookback": tr.params.trend_lookback,
            "maxPositionSize": tr.params.max_position_size,
            "volTarget": tr.params.vol_target,
            "volLookback": tr.params.vol_lookback,
            "volEwmaAlpha": tr.params.vol_ewma_alpha,
            "volFloor": tr.params.vol_floor,
            "volScaleMax": tr.params.vol_scale_max,
            "maxVolatility": tr.params.max_volatility,
            "periodsPerYear": tr.params.periods_per_year,
            "kalmanMarketTopN": tr.params.kalman_market_top_n,
            "fee": tr.params.fee,
            "epochs": tr.params.epochs,
            "hiddenSize": tr.params.hidden_size,
            "learningRate": tr.params.learning_rate,
            "valRatio": tr.params.val_ratio,
            "patience": tr.params.patience,
            "walkForwardFolds": tr.params.walk_forward_folds,
            "tuneStressVolMult": tr.params.tune_stress_vol_mult,
            "tuneStressShock": tr.params.tune_stress_shock,
            "tuneStressWeight": tr.params.tune_stress_weight,
            "gradClip": tr.params.grad_clip,
            "slippage": tr.params.slippage,
            "spread": tr.params.spread,
            "intrabarFill": tr.params.intrabar_fill,
            "stopLoss": tr.params.stop_loss,
            "takeProfit": tr.params.take_profit,
            "trailingStop": tr.params.trailing_stop,
            "stopLossVolMult": tr.params.stop_loss_vol_mult,
            "takeProfitVolMult": tr.params.take_profit_vol_mult,
            "trailingStopVolMult": tr.params.trailing_stop_vol_mult,
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
    symbol = normalize_symbol(symbol_label)
    if symbol is not None:
        r["params"]["binanceSymbol"] = symbol
    if tr.metrics is not None:
        r["metrics"] = tr.metrics
    ops = extract_operations(tr.stdout_json)
    if ops is not None:
        r["operations"] = ops
    return r


def extract_operations(stdout_json: Optional[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(stdout_json, dict):
        return None
    bt = stdout_json.get("backtest")
    if not isinstance(bt, dict):
        return None
    trades = bt.get("trades")
    if not isinstance(trades, list):
        return None
    ops: List[Dict[str, Any]] = []
    for raw in trades:
        if not isinstance(raw, dict):
            continue
        entry_idx = coerce_int_value(raw.get("entryIndex"))
        exit_idx = coerce_int_value(raw.get("exitIndex"))
        if entry_idx is None or exit_idx is None:
            continue
        ops.append(
            {
                "entryIndex": entry_idx,
                "exitIndex": exit_idx,
                "entryEquity": coerce_float_value(raw.get("entryEquity")),
                "exitEquity": coerce_float_value(raw.get("exitEquity")),
                "return": coerce_float_value(raw.get("return")),
                "holdingPeriods": coerce_int_value(raw.get("holdingPeriods")),
                "exitReason": raw.get("exitReason") if isinstance(raw.get("exitReason"), str) else None,
            }
        )
    return ops if ops else None


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
    p_auto_bars: float,
    bars_min: int,
    bars_max: int,
    bars_distribution: str,
    open_threshold_min: float,
    open_threshold_max: float,
    close_threshold_min: float,
    close_threshold_max: float,
    min_hold_bars_range: Tuple[int, int],
    cooldown_bars_range: Tuple[int, int],
    max_hold_bars_range: Tuple[int, int],
    min_edge_range: Tuple[float, float],
    min_signal_to_noise_range: Tuple[float, float],
    edge_buffer_range: Tuple[float, float],
    trend_lookback_range: Tuple[int, int],
    max_position_size_range: Tuple[float, float],
    vol_target_range: Tuple[float, float],
    vol_lookback_range: Tuple[int, int],
    vol_ewma_alpha_range: Tuple[float, float],
    p_disable_vol_ewma_alpha: float,
    vol_floor_range: Tuple[float, float],
    vol_scale_max_range: Tuple[float, float],
    max_volatility_range: Tuple[float, float],
    periods_per_year_range: Tuple[float, float],
    kalman_market_top_n_range: Tuple[int, int],
    p_cost_aware_edge: float,
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
    walk_forward_folds_range: Tuple[int, int],
    tune_stress_vol_mult_range: Tuple[float, float],
    tune_stress_shock_range: Tuple[float, float],
    tune_stress_weight_range: Tuple[float, float],
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
    stop_vol_mult_range: Tuple[float, float],
    take_vol_mult_range: Tuple[float, float],
    trail_vol_mult_range: Tuple[float, float],
    method_weights: Dict[str, float],
    normalization_choices: List[str],
    blend_weight_range: Tuple[float, float],
    p_disable_stop: float,
    p_disable_tp: float,
    p_disable_trail: float,
    p_disable_stop_vol_mult: float,
    p_disable_tp_vol_mult: float,
    p_disable_trail_vol_mult: float,
    p_disable_max_dd: float,
    p_disable_max_dl: float,
    p_disable_max_oe: float,
    p_disable_grad_clip: float,
    p_disable_vol_target: float,
    p_disable_max_volatility: float,
    max_dd_range: Tuple[float, float],
    max_dl_range: Tuple[float, float],
    max_oe_range: Tuple[int, int],
) -> TrialParams:
    interval = rng.choice(intervals)

    def sample_bars() -> int:
        # 0 means auto/all; otherwise explicit number within the configured range.
        if rng.random() < clamp(p_auto_bars, 0.0, 1.0):
            return 0
        if str(bars_distribution).strip().lower() == "log":
            lo = max(2, bars_min)
            hi = max(lo, bars_max)
            if lo == hi:
                return lo
            return int(round(log_uniform(rng, float(lo), float(hi))))
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

    bw_lo, bw_hi = blend_weight_range
    bw_lo, bw_hi = (min(bw_lo, bw_hi), max(bw_lo, bw_hi))
    blend_weight = clamp(rng.uniform(bw_lo, bw_hi), 0.0, 1.0)

    normalization = rng.choice(normalization_choices)
    base_open_threshold = log_uniform(rng, max(1e-12, open_threshold_min), max(1e-12, open_threshold_max))
    base_close_threshold = log_uniform(rng, max(1e-12, close_threshold_min), max(1e-12, close_threshold_max))
    min_hold_lo, min_hold_hi = min_hold_bars_range
    min_hold_bars = rng.randint(min_hold_lo, min_hold_hi)
    cooldown_lo, cooldown_hi = cooldown_bars_range
    cooldown_bars = rng.randint(cooldown_lo, cooldown_hi)
    max_hold_lo, max_hold_hi = max_hold_bars_range
    max_hold_bars = None
    if max_hold_hi > 0:
        max_hold_sample = rng.randint(max_hold_lo, max_hold_hi)
        if max_hold_sample > 0:
            max_hold_bars = max_hold_sample
    min_edge_lo, min_edge_hi = min_edge_range
    min_edge = rng.uniform(min_edge_lo, min_edge_hi)
    min_sn_lo, min_sn_hi = min_signal_to_noise_range
    min_sn_lo, min_sn_hi = (min(min_sn_lo, min_sn_hi), max(min_sn_lo, min_sn_hi))
    min_signal_to_noise = rng.uniform(min_sn_lo, min_sn_hi)
    edge_buffer_lo, edge_buffer_hi = edge_buffer_range
    edge_buffer = rng.uniform(edge_buffer_lo, edge_buffer_hi)
    if p_cost_aware_edge < 0:
        cost_aware_edge = edge_buffer > 0
    else:
        cost_aware_edge = rng.random() < clamp(p_cost_aware_edge, 0.0, 1.0)
        if not cost_aware_edge:
            edge_buffer = 0.0
    trend_lookback_lo, trend_lookback_hi = trend_lookback_range
    trend_lookback = rng.randint(trend_lookback_lo, trend_lookback_hi)
    max_pos_lo, max_pos_hi = max_position_size_range
    max_position_size = max(0.0, rng.uniform(max_pos_lo, max_pos_hi))
    vol_target_lo, vol_target_hi = vol_target_range
    vol_target = None
    if max(vol_target_lo, vol_target_hi) > 0 and rng.random() >= clamp(p_disable_vol_target, 0.0, 1.0):
        vt_lo = max(1e-12, min(vol_target_lo, vol_target_hi))
        vt_hi = max(vt_lo, max(vol_target_lo, vol_target_hi))
        vol_target = rng.uniform(vt_lo, vt_hi)
    vol_alpha_lo, vol_alpha_hi = vol_ewma_alpha_range
    vol_ewma_alpha = None
    if max(vol_alpha_lo, vol_alpha_hi) > 0 and rng.random() >= clamp(p_disable_vol_ewma_alpha, 0.0, 1.0):
        va_lo = max(1e-6, min(vol_alpha_lo, vol_alpha_hi))
        va_hi = min(0.999, max(vol_alpha_lo, vol_alpha_hi))
        if va_hi >= va_lo:
            vol_ewma_alpha = rng.uniform(va_lo, va_hi)
    vol_lb_lo, vol_lb_hi = vol_lookback_range
    vol_lookback = rng.randint(vol_lb_lo, vol_lb_hi)
    if vol_target is not None and vol_ewma_alpha is None:
        vol_lookback = max(2, vol_lookback)
    vol_floor_lo, vol_floor_hi = vol_floor_range
    vol_floor = max(0.0, rng.uniform(vol_floor_lo, vol_floor_hi))
    vol_scale_lo, vol_scale_hi = vol_scale_max_range
    vol_scale_max = max(0.0, rng.uniform(vol_scale_lo, vol_scale_hi))
    ppy_lo, ppy_hi = periods_per_year_range
    periods_per_year = None
    if max(ppy_lo, ppy_hi) > 0:
        ppy_min = max(1e-12, min(ppy_lo, ppy_hi))
        ppy_max = max(ppy_min, max(ppy_lo, ppy_hi))
        periods_per_year = rng.uniform(ppy_min, ppy_max)
    km_lo, km_hi = kalman_market_top_n_range
    km_lo, km_hi = (min(km_lo, km_hi), max(km_lo, km_hi))
    kalman_market_top_n = rng.randint(km_lo, km_hi)
    max_vol_lo, max_vol_hi = max_volatility_range
    max_volatility = None
    if max(max_vol_lo, max_vol_hi) > 0 and rng.random() >= clamp(p_disable_max_volatility, 0.0, 1.0):
        mv_lo = max(1e-12, min(max_vol_lo, max_vol_hi))
        mv_hi = max(mv_lo, max(max_vol_lo, max_vol_hi))
        max_volatility = rng.uniform(mv_lo, mv_hi)
    fee = rng.uniform(max(0.0, fee_min), max(0.0, fee_max))
    epochs = rng.randint(epochs_min, epochs_max)
    hidden_size = rng.randint(hidden_min, hidden_max)
    learning_rate = log_uniform(rng, lr_min, lr_max)
    val_ratio = rng.uniform(val_min, val_max)
    patience = rng.randint(0, patience_max)
    wf_lo, wf_hi = walk_forward_folds_range
    wf_lo, wf_hi = (min(wf_lo, wf_hi), max(wf_lo, wf_hi))
    walk_forward_folds = rng.randint(max(1, wf_lo), max(1, wf_hi))
    tsvm_lo, tsvm_hi = tune_stress_vol_mult_range
    tsvm_lo = max(1e-12, min(tsvm_lo, tsvm_hi))
    tsvm_hi = max(tsvm_lo, max(tsvm_lo, tsvm_hi))
    tune_stress_vol_mult = rng.uniform(tsvm_lo, tsvm_hi)
    tss_lo, tss_hi = tune_stress_shock_range
    tss_lo, tss_hi = (min(tss_lo, tss_hi), max(tss_lo, tss_hi))
    tune_stress_shock = rng.uniform(tss_lo, tss_hi)
    tsw_lo, tsw_hi = tune_stress_weight_range
    tsw_lo = max(0.0, min(tsw_lo, tsw_hi))
    tsw_hi = max(tsw_lo, max(tsw_lo, tsw_hi))
    tune_stress_weight = rng.uniform(tsw_lo, tsw_hi)
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

    svm_lo, svm_hi = stop_vol_mult_range
    stop_vol_mult = None
    if max(svm_lo, svm_hi) > 0:
        svm_min = max(1e-6, min(svm_lo, svm_hi))
        svm_max = max(svm_min, max(svm_lo, svm_hi))
        stop_vol_mult = maybe(rng, p_disable_stop_vol_mult, lambda: log_uniform(rng, svm_min, svm_max))

    tvm_lo, tvm_hi = take_vol_mult_range
    take_vol_mult = None
    if max(tvm_lo, tvm_hi) > 0:
        tvm_min = max(1e-6, min(tvm_lo, tvm_hi))
        tvm_max = max(tvm_min, max(tvm_lo, tvm_hi))
        take_vol_mult = maybe(rng, p_disable_tp_vol_mult, lambda: log_uniform(rng, tvm_min, tvm_max))

    trvm_lo, trvm_hi = trail_vol_mult_range
    trail_vol_mult = None
    if max(trvm_lo, trvm_hi) > 0:
        trvm_min = max(1e-6, min(trvm_lo, trvm_hi))
        trvm_max = max(trvm_min, max(trvm_lo, trvm_hi))
        trail_vol_mult = maybe(rng, p_disable_trail_vol_mult, lambda: log_uniform(rng, trvm_min, trvm_max))

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
        blend_weight=blend_weight,
        positioning=positioning,
        normalization=normalization,
        base_open_threshold=base_open_threshold,
        base_close_threshold=base_close_threshold,
        min_hold_bars=min_hold_bars,
        cooldown_bars=cooldown_bars,
        max_hold_bars=max_hold_bars,
        min_edge=min_edge,
        min_signal_to_noise=min_signal_to_noise,
        edge_buffer=edge_buffer,
        cost_aware_edge=cost_aware_edge,
        trend_lookback=trend_lookback,
        max_position_size=max_position_size,
        vol_target=vol_target,
        vol_lookback=vol_lookback,
        vol_ewma_alpha=vol_ewma_alpha,
        vol_floor=vol_floor,
        vol_scale_max=vol_scale_max,
        max_volatility=max_volatility,
        periods_per_year=periods_per_year,
        kalman_market_top_n=kalman_market_top_n,
        fee=fee,
        epochs=epochs,
        slippage=slippage,
        spread=spread,
        intrabar_fill=intrabar_fill,
        stop_loss=stop_loss,
        take_profit=take_profit,
        trailing_stop=trailing_stop,
        stop_loss_vol_mult=stop_vol_mult,
        take_profit_vol_mult=take_vol_mult,
        trailing_stop_vol_mult=trail_vol_mult,
        max_drawdown=max_drawdown,
        max_daily_loss=max_daily_loss,
        max_order_errors=max_order_errors,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        val_ratio=val_ratio,
        patience=patience,
        walk_forward_folds=walk_forward_folds,
        tune_stress_vol_mult=tune_stress_vol_mult,
        tune_stress_shock=tune_stress_shock,
        tune_stress_weight=tune_stress_weight,
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
    parser.add_argument(
        "--symbol-label",
        type=str,
        default="",
        help="Optional symbol label to attach to params (useful for CSV runs).",
    )
    parser.add_argument(
        "--source-label",
        type=str,
        default="",
        help="Optional source label override for outputs (e.g., binance).",
    )
    parser.add_argument("--price-column", type=str, default="close", help="CSV column name for price (default: close).")
    parser.add_argument(
        "--high-column",
        type=str,
        default="",
        help="CSV column name for high (optional; requires --low-column; enables intrabar stops/TP/trailing).",
    )
    parser.add_argument(
        "--low-column",
        type=str,
        default="",
        help="CSV column name for low (optional; requires --high-column; enables intrabar stops/TP/trailing).",
    )
    parser.add_argument("--lookback-window", type=str, default="24h", help="Lookback window (default: 24h).")
    parser.add_argument("--backtest-ratio", type=float, default=0.2, help="Backtest holdout ratio (default: 0.2).")
    parser.add_argument("--tune-ratio", type=float, default=0.2, help="Tune ratio for --sweep-threshold (default: 0.2).")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials (default: 50).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42).")
    parser.add_argument("--timeout-sec", type=float, default=60.0, help="Per-trial timeout in seconds (default: 60).")
    parser.add_argument("--output", type=str, default="", help="Write JSONL trial records to this path.")
    parser.add_argument("--append", action="store_true", help="Append to --output instead of overwriting it.")
    parser.add_argument("--binary", type=str, default="", help="Path to trader-hs binary (auto-discovered via cabal if omitted).")
    parser.add_argument("--no-sweep-threshold", action="store_true", help="Disable internal threshold sweep (not recommended).")
    parser.add_argument("--disable-lstm-persistence", action="store_true", help="Disable LSTM weight caching (more reproducible).")
    parser.add_argument("--top-json", type=str, default="", help="Write the top-performing combos (JSON) for the UI chart.")
    parser.add_argument(
        "--quality",
        action="store_true",
        help="Apply a deeper-search preset (more trials, wider ranges, stricter filters, equity-dd-turnover, smaller splits).",
    )
    parser.add_argument(
        "--auto-high-low",
        action="store_true",
        help="Auto-detect CSV high/low columns when not explicitly provided (enables intrabar stops/TP/trailing).",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="final-equity",
        choices=["final-equity", "sharpe", "calmar", "equity-dd", "equity-dd-turnover"],
        help="Optimization objective used to select the best params/top-json (default: final-equity).",
    )
    parser.add_argument(
        "--penalty-max-drawdown",
        type=float,
        default=1.0,
        help="Penalty weight for maxDrawdown (used by equity-dd* objectives; default: 1.0).",
    )
    parser.add_argument(
        "--penalty-turnover",
        type=float,
        default=0.0,
        help="Penalty weight for turnover (used by equity-dd-turnover; default: 0.0).",
    )
    parser.add_argument(
        "--min-round-trips",
        type=int,
        default=0,
        help="Skip candidates with fewer than this many roundTrips (default: 0).",
    )
    parser.add_argument(
        "--min-win-rate",
        type=float,
        default=0.0,
        help="Skip candidates with winRate below this threshold (default: 0).",
    )
    parser.add_argument(
        "--min-profit-factor",
        type=float,
        default=0.0,
        help="Skip candidates with profitFactor below this threshold (default: 0).",
    )
    parser.add_argument(
        "--min-exposure",
        type=float,
        default=0.0,
        help="Skip candidates with exposure below this threshold (default: 0).",
    )
    parser.add_argument(
        "--min-sharpe",
        type=float,
        default=0.0,
        help="Skip candidates with Sharpe below this threshold (default: 0).",
    )
    parser.add_argument(
        "--min-wf-sharpe-mean",
        type=float,
        default=0.0,
        help="Skip candidates with walk-forward sharpeMean below this threshold (default: 0).",
    )
    parser.add_argument(
        "--max-wf-sharpe-std",
        type=float,
        default=0.0,
        help="Skip candidates with walk-forward sharpeStd above this threshold (default: 0=disabled).",
    )
    parser.add_argument(
        "--tune-objective",
        type=str,
        default="equity-dd-turnover",
        choices=["final-equity", "sharpe", "calmar", "equity-dd", "equity-dd-turnover"],
        help="Tune objective used by --sweep-threshold inside trader-hs (default: equity-dd-turnover).",
    )
    parser.add_argument(
        "--tune-penalty-max-drawdown",
        type=float,
        default=1.0,
        help="Penalty weight for max drawdown in tune scoring (default: 1.0).",
    )
    parser.add_argument(
        "--tune-penalty-turnover",
        type=float,
        default=0.1,
        help="Penalty weight for turnover in tune scoring (default: 0.1).",
    )
    parser.add_argument(
        "--tune-stress-vol-mult",
        type=float,
        default=1.0,
        help="Stress volatility multiplier for tune scoring (default: 1.0).",
    )
    parser.add_argument(
        "--tune-stress-shock",
        type=float,
        default=0.0,
        help="Stress shock added to returns for tune scoring (default: 0.0).",
    )
    parser.add_argument(
        "--tune-stress-weight",
        type=float,
        default=0.0,
        help="Penalty weight for stress scenario in tune scoring (default: 0.0).",
    )
    parser.add_argument(
        "--tune-stress-vol-mult-min",
        type=float,
        default=None,
        help="Min stress volatility multiplier when sampling (default: tune-stress-vol-mult).",
    )
    parser.add_argument(
        "--tune-stress-vol-mult-max",
        type=float,
        default=None,
        help="Max stress volatility multiplier when sampling (default: tune-stress-vol-mult).",
    )
    parser.add_argument(
        "--tune-stress-shock-min",
        type=float,
        default=None,
        help="Min stress shock when sampling (default: tune-stress-shock).",
    )
    parser.add_argument(
        "--tune-stress-shock-max",
        type=float,
        default=None,
        help="Max stress shock when sampling (default: tune-stress-shock).",
    )
    parser.add_argument(
        "--tune-stress-weight-min",
        type=float,
        default=None,
        help="Min stress weight when sampling (default: tune-stress-weight).",
    )
    parser.add_argument(
        "--tune-stress-weight-max",
        type=float,
        default=None,
        help="Max stress weight when sampling (default: tune-stress-weight).",
    )
    parser.add_argument(
        "--walk-forward-folds-min",
        type=int,
        default=5,
        help="Min walk-forward folds when sampling (default: 5).",
    )
    parser.add_argument(
        "--walk-forward-folds-max",
        type=int,
        default=5,
        help="Max walk-forward folds when sampling (default: 5).",
    )

    interval_group = parser.add_mutually_exclusive_group()
    interval_group.add_argument("--interval", type=str, default="", help="Single interval to sample (alias for --intervals).")
    interval_group.add_argument(
        "--intervals",
        type=str,
        default=",".join(BINANCE_INTERVALS),
        help="Comma-separated intervals to sample.",
    )
    parser.add_argument("--bars-min", type=int, default=0, help="Min bars when sampling explicit bars (default: 0=auto).")
    parser.add_argument("--bars-max", type=int, default=0, help="Max bars when sampling explicit bars (default: 0=auto-detect for CSV).")
    parser.add_argument("--bars-auto-prob", type=float, default=0.25, help="Probability of sampling bars=auto/all (default: 0.25).")
    parser.add_argument(
        "--bars-distribution",
        type=str,
        default="uniform",
        choices=["uniform", "log"],
        help="Sampling distribution for explicit bars (default: uniform).",
    )
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
    parser.add_argument("--min-hold-bars-min", type=int, default=0, help="Min min-hold-bars when sampling (default: 0).")
    parser.add_argument("--min-hold-bars-max", type=int, default=0, help="Max min-hold-bars when sampling (default: 0).")
    parser.add_argument("--cooldown-bars-min", type=int, default=0, help="Min cooldown-bars when sampling (default: 0).")
    parser.add_argument("--cooldown-bars-max", type=int, default=0, help="Max cooldown-bars when sampling (default: 0).")
    parser.add_argument("--max-hold-bars-min", type=int, default=0, help="Min max-hold-bars when sampling (default: 0=disabled).")
    parser.add_argument("--max-hold-bars-max", type=int, default=0, help="Max max-hold-bars when sampling (default: 0=disabled).")
    parser.add_argument("--min-edge-min", type=float, default=0.0, help="Min min-edge when sampling (default: 0).")
    parser.add_argument("--min-edge-max", type=float, default=0.0, help="Max min-edge when sampling (default: 0).")
    parser.add_argument(
        "--min-signal-to-noise-min",
        type=float,
        default=0.0,
        help="Min min-signal-to-noise when sampling (default: 0=disabled).",
    )
    parser.add_argument(
        "--min-signal-to-noise-max",
        type=float,
        default=0.0,
        help="Max min-signal-to-noise when sampling (default: 0=disabled).",
    )
    parser.add_argument("--edge-buffer-min", type=float, default=0.0, help="Min edge-buffer when sampling (default: 0).")
    parser.add_argument("--edge-buffer-max", type=float, default=0.0, help="Max edge-buffer when sampling (default: 0).")
    parser.add_argument(
        "--p-cost-aware-edge",
        type=float,
        default=-1.0,
        help="Probability cost-aware-edge is enabled (-1=enable when edge-buffer>0).",
    )
    parser.add_argument("--trend-lookback-min", type=int, default=0, help="Min trend-lookback when sampling (default: 0).")
    parser.add_argument("--trend-lookback-max", type=int, default=0, help="Max trend-lookback when sampling (default: 0).")

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
    parser.add_argument("--kalman-market-top-n-min", type=int, default=50, help="Min kalman-market-top-n when sampling (default: 50).")
    parser.add_argument("--kalman-market-top-n-max", type=int, default=50, help="Max kalman-market-top-n when sampling (default: 50).")

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
    parser.add_argument("--max-position-size-min", type=float, default=1.0, help="Min max-position-size when sampling (default: 1).")
    parser.add_argument("--max-position-size-max", type=float, default=1.0, help="Max max-position-size when sampling (default: 1).")
    parser.add_argument("--vol-target-min", type=float, default=0.0, help="Min vol-target when sampling (default: 0=disabled).")
    parser.add_argument("--vol-target-max", type=float, default=0.0, help="Max vol-target when sampling (default: 0=disabled).")
    parser.add_argument(
        "--p-disable-vol-target",
        type=float,
        default=0.0,
        help="Probability vol-target is disabled when sampling (default: 0).",
    )
    parser.add_argument("--vol-lookback-min", type=int, default=20, help="Min vol-lookback when sampling (default: 20).")
    parser.add_argument("--vol-lookback-max", type=int, default=20, help="Max vol-lookback when sampling (default: 20).")
    parser.add_argument("--vol-ewma-alpha-min", type=float, default=0.0, help="Min vol-ewma-alpha when sampling (default: 0=disabled).")
    parser.add_argument("--vol-ewma-alpha-max", type=float, default=0.0, help="Max vol-ewma-alpha when sampling (default: 0=disabled).")
    parser.add_argument(
        "--p-disable-vol-ewma-alpha",
        type=float,
        default=0.0,
        help="Probability vol-ewma-alpha is disabled when sampling (default: 0).",
    )
    parser.add_argument("--vol-floor-min", type=float, default=0.0, help="Min vol-floor when sampling (default: 0).")
    parser.add_argument("--vol-floor-max", type=float, default=0.0, help="Max vol-floor when sampling (default: 0).")
    parser.add_argument("--vol-scale-max-min", type=float, default=1.0, help="Min vol-scale-max when sampling (default: 1).")
    parser.add_argument("--vol-scale-max-max", type=float, default=1.0, help="Max vol-scale-max when sampling (default: 1).")
    parser.add_argument(
        "--max-volatility-min", type=float, default=0.0, help="Min max-volatility when sampling (default: 0=disabled)."
    )
    parser.add_argument(
        "--max-volatility-max", type=float, default=0.0, help="Max max-volatility when sampling (default: 0=disabled)."
    )
    parser.add_argument(
        "--p-disable-max-volatility",
        type=float,
        default=0.0,
        help="Probability max-volatility is disabled when sampling (default: 0).",
    )
    parser.add_argument(
        "--periods-per-year-min",
        type=float,
        default=0.0,
        help="Min periods-per-year when sampling (default: 0=auto).",
    )
    parser.add_argument(
        "--periods-per-year-max",
        type=float,
        default=0.0,
        help="Max periods-per-year when sampling (default: 0=auto).",
    )

    parser.add_argument("--stop-min", type=float, default=0.002, help="Min stop-loss when enabled (default: 0.002).")
    parser.add_argument("--stop-max", type=float, default=0.20, help="Max stop-loss when enabled (default: 0.20).")
    parser.add_argument("--tp-min", type=float, default=0.002, help="Min take-profit when enabled (default: 0.002).")
    parser.add_argument("--tp-max", type=float, default=0.20, help="Max take-profit when enabled (default: 0.20).")
    parser.add_argument("--trail-min", type=float, default=0.002, help="Min trailing-stop when enabled (default: 0.002).")
    parser.add_argument("--trail-max", type=float, default=0.20, help="Max trailing-stop when enabled (default: 0.20).")

    parser.add_argument("--p-disable-stop", type=float, default=0.5, help="Probability stop-loss is disabled (default: 0.5).")
    parser.add_argument("--p-disable-tp", type=float, default=0.5, help="Probability take-profit is disabled (default: 0.5).")
    parser.add_argument("--p-disable-trail", type=float, default=0.6, help="Probability trailing-stop is disabled (default: 0.6).")

    parser.add_argument("--stop-vol-mult-min", type=float, default=0.0, help="Min stop-loss vol-mult when enabled (default: 0=disabled).")
    parser.add_argument("--stop-vol-mult-max", type=float, default=0.0, help="Max stop-loss vol-mult when enabled (default: 0=disabled).")
    parser.add_argument("--tp-vol-mult-min", type=float, default=0.0, help="Min take-profit vol-mult when enabled (default: 0=disabled).")
    parser.add_argument("--tp-vol-mult-max", type=float, default=0.0, help="Max take-profit vol-mult when enabled (default: 0=disabled).")
    parser.add_argument("--trail-vol-mult-min", type=float, default=0.0, help="Min trailing-stop vol-mult when enabled (default: 0=disabled).")
    parser.add_argument("--trail-vol-mult-max", type=float, default=0.0, help="Max trailing-stop vol-mult when enabled (default: 0=disabled).")

    parser.add_argument(
        "--p-disable-stop-vol-mult", type=float, default=0.5, help="Probability stop-loss vol-mult is disabled (default: 0.5)."
    )
    parser.add_argument(
        "--p-disable-tp-vol-mult", type=float, default=0.5, help="Probability take-profit vol-mult is disabled (default: 0.5)."
    )
    parser.add_argument(
        "--p-disable-trail-vol-mult", type=float, default=0.6, help="Probability trailing-stop vol-mult is disabled (default: 0.6)."
    )

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
    parser.add_argument("--method-weight-blend", type=float, default=0.0, help="Sampling weight for method=blend (default: 0).")
    parser.add_argument("--blend-weight-min", type=float, default=0.5, help="Min blend-weight when sampling (default: 0.5).")
    parser.add_argument("--blend-weight-max", type=float, default=0.5, help="Max blend-weight when sampling (default: 0.5).")
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
    if args.quality:
        apply_quality_preset(args)

    root = Path(__file__).resolve().parents[1]
    haskell_dir = root

    trader_bin = Path(args.binary).expanduser() if args.binary else discover_trader_bin(haskell_dir)

    if str(args.interval).strip():
        intervals_in = [str(args.interval).strip()]
    else:
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
        if args.auto_high_low and not args.high_column and not args.low_column:
            high_col, low_col = detect_high_low_columns(p)
            if high_col and low_col:
                args.high_column = high_col
                args.low_column = low_col
                print(f"Auto-detected high/low columns: {high_col}/{low_col}", file=sys.stderr)
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

    use_sweep_threshold = not bool(args.no_sweep_threshold)

    bars_max = int(args.bars_max)
    if bars_max <= 0:
        bars_max = max_bars_cap
    bars_min = int(args.bars_min)
    if bars_min <= 0:
        # Ensure we can at least run:
        # - splitTrainBacktest (needs lookback+3 prices)
        # - the fit/tune split when --sweep-threshold is enabled.
        # (Still allow bars=0 = auto/all, handled separately.)
        worst_lb = max(lookback_bars(itv, args.lookback_window) for itv in intervals)
        min_required = worst_lb + 3
        if use_sweep_threshold:
            br = float(args.backtest_ratio)
            tr = float(args.tune_ratio)
            if not (0.0 < br < 1.0):
                print("--backtest-ratio must be between 0 and 1.", file=sys.stderr)
                return 2
            if not (0.0 < tr < 1.0):
                print("--tune-ratio must be between 0 and 1 when sweep-threshold is enabled.", file=sys.stderr)
                return 2

            denom = max(1e-12, (1.0 - br) * (1.0 - tr))
            min_required = max(min_required, int(math.ceil((worst_lb + 1) / denom)) + 2)

            min_train = int(math.ceil(2.0 / tr))
            min_required = max(min_required, int(math.ceil(min_train / max(1e-12, (1.0 - br)))) + 2)

            auto_bars = 500 if args.binance_symbol else max_bars_cap
            if min_required > max(bars_max, auto_bars):
                print(
                    f"Not enough bars for lookback={worst_lb} with backtest-ratio={br} and tune-ratio={tr}. "
                    f"Need bars >= {min_required}. Increase --bars-max, reduce --tune-ratio/--backtest-ratio, "
                    "reduce --lookback-window, or pass --no-sweep-threshold.",
                    file=sys.stderr,
                )
                return 2

        bars_min = min(bars_max, max(10, min_required))
    bars_min = max(2, min(bars_min, bars_max))
    bars_auto_prob = clamp(float(args.bars_auto_prob), 0.0, 1.0)
    bars_distribution = str(args.bars_distribution).strip().lower()

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
    walk_forward_folds_min = max(1, int(args.walk_forward_folds_min))
    walk_forward_folds_max = max(walk_forward_folds_min, int(args.walk_forward_folds_max))
    walk_forward_folds_range = (walk_forward_folds_min, walk_forward_folds_max)
    tune_stress_vol_mult_base = max(1e-12, float(args.tune_stress_vol_mult))
    tune_stress_shock_base = float(args.tune_stress_shock)
    tune_stress_weight_base = max(0.0, float(args.tune_stress_weight))
    tsvm_min = args.tune_stress_vol_mult_min
    tsvm_max = args.tune_stress_vol_mult_max
    if tsvm_min is None and tsvm_max is None:
        tsvm_min = tune_stress_vol_mult_base
        tsvm_max = tune_stress_vol_mult_base
    else:
        if tsvm_min is None:
            tsvm_min = tsvm_max
        if tsvm_max is None:
            tsvm_max = tsvm_min
    tsvm_min = max(1e-12, float(tsvm_min))
    tsvm_max = max(1e-12, float(tsvm_max))
    tune_stress_vol_mult_range = (min(tsvm_min, tsvm_max), max(tsvm_min, tsvm_max))
    tss_min = args.tune_stress_shock_min
    tss_max = args.tune_stress_shock_max
    if tss_min is None and tss_max is None:
        tss_min = tune_stress_shock_base
        tss_max = tune_stress_shock_base
    else:
        if tss_min is None:
            tss_min = tss_max
        if tss_max is None:
            tss_max = tss_min
    tune_stress_shock_range = (float(min(tss_min, tss_max)), float(max(tss_min, tss_max)))
    tsw_min = args.tune_stress_weight_min
    tsw_max = args.tune_stress_weight_max
    if tsw_min is None and tsw_max is None:
        tsw_min = tune_stress_weight_base
        tsw_max = tune_stress_weight_base
    else:
        if tsw_min is None:
            tsw_min = tsw_max
        if tsw_max is None:
            tsw_max = tsw_min
    tsw_min = max(0.0, float(tsw_min))
    tsw_max = max(0.0, float(tsw_max))
    tune_stress_weight_range = (min(tsw_min, tsw_max), max(tsw_min, tsw_max))

    stop_min = float(args.stop_min)
    stop_max = float(args.stop_max)
    tp_min = float(args.tp_min)
    tp_max = float(args.tp_max)
    trail_min = float(args.trail_min)
    trail_max = float(args.trail_max)
    stop_vol_mult_min = max(0.0, float(args.stop_vol_mult_min))
    stop_vol_mult_max = max(stop_vol_mult_min, float(args.stop_vol_mult_max))
    stop_vol_mult_range = (stop_vol_mult_min, stop_vol_mult_max)
    tp_vol_mult_min = max(0.0, float(args.tp_vol_mult_min))
    tp_vol_mult_max = max(tp_vol_mult_min, float(args.tp_vol_mult_max))
    tp_vol_mult_range = (tp_vol_mult_min, tp_vol_mult_max)
    trail_vol_mult_min = max(0.0, float(args.trail_vol_mult_min))
    trail_vol_mult_max = max(trail_vol_mult_min, float(args.trail_vol_mult_max))
    trail_vol_mult_range = (trail_vol_mult_min, trail_vol_mult_max)

    fee_min = max(0.0, float(args.fee_min))
    fee_max = max(fee_min, float(args.fee_max))

    open_threshold_min = max(1e-12, float(args.open_threshold_min))
    open_threshold_max = max(open_threshold_min, float(args.open_threshold_max))
    close_threshold_min = max(1e-12, float(args.close_threshold_min))
    close_threshold_max = max(close_threshold_min, float(args.close_threshold_max))
    min_hold_min = max(0, int(args.min_hold_bars_min))
    min_hold_max = max(min_hold_min, int(args.min_hold_bars_max))
    cooldown_min = max(0, int(args.cooldown_bars_min))
    cooldown_max = max(cooldown_min, int(args.cooldown_bars_max))
    max_hold_min = max(0, int(args.max_hold_bars_min))
    max_hold_max = max(max_hold_min, int(args.max_hold_bars_max))
    min_edge_min = max(0.0, float(args.min_edge_min))
    min_edge_max = max(min_edge_min, float(args.min_edge_max))
    min_signal_to_noise_min = max(0.0, float(args.min_signal_to_noise_min))
    min_signal_to_noise_max = max(min_signal_to_noise_min, float(args.min_signal_to_noise_max))
    edge_buffer_min = max(0.0, float(args.edge_buffer_min))
    edge_buffer_max = max(edge_buffer_min, float(args.edge_buffer_max))
    p_cost_aware_edge = float(args.p_cost_aware_edge)
    trend_lookback_min = max(0, int(args.trend_lookback_min))
    trend_lookback_max = max(trend_lookback_min, int(args.trend_lookback_max))

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
    kalman_market_top_n_min = max(0, int(args.kalman_market_top_n_min))
    kalman_market_top_n_max = max(kalman_market_top_n_min, int(args.kalman_market_top_n_max))
    kalman_market_top_n_range = (kalman_market_top_n_min, kalman_market_top_n_max)

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
    max_position_size_min = max(0.0, float(args.max_position_size_min))
    max_position_size_max = max(max_position_size_min, float(args.max_position_size_max))
    max_position_size_range = (max_position_size_min, max_position_size_max)
    vol_target_min = max(0.0, float(args.vol_target_min))
    vol_target_max = max(vol_target_min, float(args.vol_target_max))
    vol_target_range = (vol_target_min, vol_target_max)
    p_disable_vol_target = clamp(float(args.p_disable_vol_target), 0.0, 1.0)
    vol_lookback_min = max(0, int(args.vol_lookback_min))
    vol_lookback_max = max(vol_lookback_min, int(args.vol_lookback_max))
    vol_lookback_range = (vol_lookback_min, vol_lookback_max)
    vol_ewma_alpha_min = max(0.0, float(args.vol_ewma_alpha_min))
    vol_ewma_alpha_max = max(vol_ewma_alpha_min, float(args.vol_ewma_alpha_max))
    vol_ewma_alpha_range = (vol_ewma_alpha_min, vol_ewma_alpha_max)
    p_disable_vol_ewma_alpha = clamp(float(args.p_disable_vol_ewma_alpha), 0.0, 1.0)
    vol_floor_min = max(0.0, float(args.vol_floor_min))
    vol_floor_max = max(vol_floor_min, float(args.vol_floor_max))
    vol_floor_range = (vol_floor_min, vol_floor_max)
    vol_scale_max_min = max(0.0, float(args.vol_scale_max_min))
    vol_scale_max_max = max(vol_scale_max_min, float(args.vol_scale_max_max))
    vol_scale_max_range = (vol_scale_max_min, vol_scale_max_max)
    max_volatility_min = max(0.0, float(args.max_volatility_min))
    max_volatility_max = max(max_volatility_min, float(args.max_volatility_max))
    max_volatility_range = (max_volatility_min, max_volatility_max)
    p_disable_max_volatility = clamp(float(args.p_disable_max_volatility), 0.0, 1.0)
    periods_per_year_min = max(0.0, float(args.periods_per_year_min))
    periods_per_year_max = max(periods_per_year_min, float(args.periods_per_year_max))
    periods_per_year_range = (periods_per_year_min, periods_per_year_max)

    method_weights = {
        "11": float(args.method_weight_11),
        "10": float(args.method_weight_10),
        "01": float(args.method_weight_01),
        "blend": float(args.method_weight_blend),
    }
    blend_weight_min = clamp(float(args.blend_weight_min), 0.0, 1.0)
    blend_weight_max = clamp(float(args.blend_weight_max), 0.0, 1.0)
    blend_weight_range = (min(blend_weight_min, blend_weight_max), max(blend_weight_min, blend_weight_max))
    normalization_choices = [s.strip() for s in str(args.normalizations).split(",") if s.strip()]
    if not normalization_choices:
        print("No normalizations provided.", file=sys.stderr)
        return 2

    base_args: List[str] = []
    if args.data:
        data_path = Path(args.data).expanduser().resolve()
        base_args += ["--data", str(data_path)]
        base_args += ["--price-column", args.price_column]
        high_col = (args.high_column or "").strip()
        low_col = (args.low_column or "").strip()
        if bool(high_col) != bool(low_col):
            print("When using --high-column/--low-column, you must provide both.", file=sys.stderr)
            return 2
        if high_col and low_col:
            base_args += ["--high-column", high_col, "--low-column", low_col]
    else:
        base_args += ["--binance-symbol", args.binance_symbol]

    base_args += ["--lookback-window", args.lookback_window]
    base_args += ["--backtest-ratio", f"{float(args.backtest_ratio):.6f}"]
    base_args += ["--tune-objective", str(args.tune_objective)]
    base_args += ["--tune-penalty-max-drawdown", f"{max(0.0, float(args.tune_penalty_max_drawdown)):.6f}"]
    base_args += ["--tune-penalty-turnover", f"{max(0.0, float(args.tune_penalty_turnover)):.6f}"]
    # Fix seed so model init is comparable across trials.
    base_args += ["--seed", str(int(args.seed))]

    rng = random.Random(int(args.seed))
    data_source = "csv" if args.data else "binance"
    source_override = str(args.source_label).strip().lower()
    if source_override:
        data_source = source_override
    symbol_label = normalize_symbol(args.symbol_label) or normalize_symbol(args.binance_symbol)

    out_path = Path(args.output).expanduser() if args.output else None
    out_fh = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_fh = out_path.open("a" if bool(args.append) else "w", encoding="utf-8")

    best: Optional[TrialResult] = None

    disable_lstm_persistence = bool(args.disable_lstm_persistence)

    trials = int(args.trials)
    records: List[TrialResult] = []
    min_round_trips = max(0, int(args.min_round_trips))
    min_win_rate = max(0.0, float(args.min_win_rate))
    min_profit_factor = max(0.0, float(args.min_profit_factor))
    min_exposure = max(0.0, float(args.min_exposure))
    min_sharpe = max(0.0, float(args.min_sharpe))
    min_wf_sharpe_mean = max(0.0, float(args.min_wf_sharpe_mean))
    max_wf_sharpe_std = max(0.0, float(args.max_wf_sharpe_std))
    for i in range(1, trials + 1):
        params = sample_params(
            rng=rng,
            intervals=intervals,
            p_auto_bars=bars_auto_prob,
            bars_min=bars_min,
            bars_max=bars_max,
            bars_distribution=bars_distribution,
            open_threshold_min=open_threshold_min,
            open_threshold_max=open_threshold_max,
            close_threshold_min=close_threshold_min,
            close_threshold_max=close_threshold_max,
            min_hold_bars_range=(min_hold_min, min_hold_max),
            cooldown_bars_range=(cooldown_min, cooldown_max),
            max_hold_bars_range=(max_hold_min, max_hold_max),
            min_edge_range=(min_edge_min, min_edge_max),
            min_signal_to_noise_range=(min_signal_to_noise_min, min_signal_to_noise_max),
            edge_buffer_range=(edge_buffer_min, edge_buffer_max),
            trend_lookback_range=(trend_lookback_min, trend_lookback_max),
            max_position_size_range=max_position_size_range,
            vol_target_range=vol_target_range,
            vol_lookback_range=vol_lookback_range,
            vol_ewma_alpha_range=vol_ewma_alpha_range,
            p_disable_vol_ewma_alpha=p_disable_vol_ewma_alpha,
            vol_floor_range=vol_floor_range,
            vol_scale_max_range=vol_scale_max_range,
            max_volatility_range=max_volatility_range,
            periods_per_year_range=periods_per_year_range,
            kalman_market_top_n_range=kalman_market_top_n_range,
            p_cost_aware_edge=p_cost_aware_edge,
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
            walk_forward_folds_range=walk_forward_folds_range,
            tune_stress_vol_mult_range=tune_stress_vol_mult_range,
            tune_stress_shock_range=tune_stress_shock_range,
            tune_stress_weight_range=tune_stress_weight_range,
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
            stop_vol_mult_range=stop_vol_mult_range,
            take_vol_mult_range=tp_vol_mult_range,
            trail_vol_mult_range=trail_vol_mult_range,
            method_weights=method_weights,
            normalization_choices=normalization_choices,
            blend_weight_range=blend_weight_range,
            p_disable_stop=clamp(float(args.p_disable_stop), 0.0, 1.0),
            p_disable_tp=clamp(float(args.p_disable_tp), 0.0, 1.0),
            p_disable_trail=clamp(float(args.p_disable_trail), 0.0, 1.0),
            p_disable_stop_vol_mult=clamp(float(args.p_disable_stop_vol_mult), 0.0, 1.0),
            p_disable_tp_vol_mult=clamp(float(args.p_disable_tp_vol_mult), 0.0, 1.0),
            p_disable_trail_vol_mult=clamp(float(args.p_disable_trail_vol_mult), 0.0, 1.0),
            p_disable_max_dd=clamp(float(args.p_disable_max_dd), 0.0, 1.0),
            p_disable_max_dl=clamp(float(args.p_disable_max_dl), 0.0, 1.0),
            p_disable_max_oe=clamp(float(args.p_disable_max_oe), 0.0, 1.0),
            p_disable_grad_clip=clamp(float(args.p_disable_grad_clip), 0.0, 1.0),
            p_disable_vol_target=p_disable_vol_target,
            p_disable_max_volatility=p_disable_max_volatility,
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

        objective = str(args.objective)
        eligible = tr.ok and tr.final_equity is not None and tr.metrics is not None
        filter_reason: Optional[str] = None
        score: Optional[float] = None
        if eligible:
            rts = metric_int(tr.metrics, "roundTrips", 0)
            if min_round_trips > 0 and rts < min_round_trips:
                eligible = False
                filter_reason = f"roundTrips<{min_round_trips}"
            else:
                win_rate = metric_float(tr.metrics, "winRate", 0.0)
                if min_win_rate > 0 and win_rate < min_win_rate:
                    eligible = False
                    filter_reason = f"winRate<{min_win_rate:.3f}"
                else:
                    profit_factor = metric_profit_factor(tr.metrics)
                    if min_profit_factor > 0 and profit_factor < min_profit_factor:
                        eligible = False
                        filter_reason = f"profitFactor<{min_profit_factor:.3f}"
                    else:
                        exposure = metric_float(tr.metrics, "exposure", 0.0)
                        if min_exposure > 0 and exposure < min_exposure:
                            eligible = False
                            filter_reason = f"exposure<{min_exposure:.3f}"
                        else:
                            sharpe = metric_float(tr.metrics, "sharpe", 0.0)
                            if min_sharpe > 0 and sharpe < min_sharpe:
                                eligible = False
                                filter_reason = f"sharpe<{min_sharpe:.3f}"
                            else:
                                if min_wf_sharpe_mean > 0 or max_wf_sharpe_std > 0:
                                    wf_summary = extract_walk_forward_summary(tr.stdout_json)
                                    if not wf_summary:
                                        eligible = False
                                        filter_reason = "walkForwardMissing"
                                    else:
                                        wf_sharpe_mean = metric_float(wf_summary, "sharpeMean", 0.0)
                                        wf_sharpe_std = metric_float(wf_summary, "sharpeStd", 0.0)
                                        if min_wf_sharpe_mean > 0 and wf_sharpe_mean < min_wf_sharpe_mean:
                                            eligible = False
                                            filter_reason = f"wfSharpeMean<{min_wf_sharpe_mean:.3f}"
                                        elif max_wf_sharpe_std > 0 and wf_sharpe_std > max_wf_sharpe_std:
                                            eligible = False
                                            filter_reason = f"wfSharpeStd>{max_wf_sharpe_std:.3f}"
                                        else:
                                            score = objective_score(
                                                tr.metrics or {},
                                                objective=objective,
                                                penalty_max_drawdown=float(args.penalty_max_drawdown),
                                                penalty_turnover=float(args.penalty_turnover),
                                            )
                                else:
                                    score = objective_score(
                                        tr.metrics or {},
                                        objective=objective,
                                        penalty_max_drawdown=float(args.penalty_max_drawdown),
                                        penalty_turnover=float(args.penalty_turnover),
                                    )
        tr = replace(tr, eligible=eligible, filter_reason=filter_reason, objective=objective, score=score)

        if out_fh is not None:
            rec = trial_to_record(tr, symbol_label)
            rec["source"] = data_source
            out_fh.write(json.dumps(rec, sort_keys=True) + "\n")
            out_fh.flush()

        records.append(tr)

        if tr.eligible and tr.score is not None:
            if best is None or tr.score > (best.score if best.score is not None else -1e18):
                best = tr

        status = "OK" if tr.eligible else "SKIP" if tr.ok else "FAIL"
        eq = f"{tr.final_equity:.6f}x" if tr.final_equity is not None else "-"
        scoreLabel = f"{tr.score:.6f}" if tr.score is not None else "-"
        print(
            f"[{i:>4}/{trials}] {status} score={scoreLabel} eq={eq} t={tr.elapsed_sec:.2f}s "
            f"interval={params.interval} bars={params.bars} method={params.method} "
            f"norm={params.normalization} epochs={params.epochs} "
            f"slip={params.slippage:.6f} spr={params.spread:.6f} "
            f"sl={fmt_opt_float(params.stop_loss)} tp={fmt_opt_float(params.take_profit)} trail={fmt_opt_float(params.trailing_stop)} "
            f"maxDD={fmt_opt_float(params.max_drawdown)} maxDL={fmt_opt_float(params.max_daily_loss)} maxOE={fmt_opt_int(params.max_order_errors)}"
            + (f" (filter: {tr.filter_reason})" if tr.filter_reason else (f" ({tr.reason})" if tr.reason else "")),
            flush=True,
        )

    if out_fh is not None:
        out_fh.close()

    if best is None:
        msg = "No eligible trials."
        hints = []
        if min_round_trips > 0:
            hints.append("--min-round-trips")
        if min_win_rate > 0:
            hints.append("--min-win-rate")
        if min_profit_factor > 0:
            hints.append("--min-profit-factor")
        if min_exposure > 0:
            hints.append("--min-exposure")
        if min_sharpe > 0:
            hints.append("--min-sharpe")
        if min_wf_sharpe_mean > 0:
            hints.append("--min-wf-sharpe-mean")
        if max_wf_sharpe_std > 0:
            hints.append("--max-wf-sharpe-std")
        if hints:
            msg += " (Try lowering " + ", ".join(hints) + ".)"
        print(msg, file=sys.stderr)
        return 1

    b = best
    assert b.final_equity is not None
    print("\nBest:")
    if b.score is not None:
        print(f"  objective:   {b.objective} (score={b.score:.8f})")
    print(f"  finalEquity: {b.final_equity:.8f}x")
    print(f"  interval:    {b.params.interval}")
    print(f"  bars:        {b.params.bars}")
    print(f"  method:      {b.params.method}")
    print(f"  positioning: {b.params.positioning}")
    print(f"  thresholds:  open={b.open_threshold} close={b.close_threshold} (from sweep)")
    print(f"  base thresholds: open={b.params.base_open_threshold} close={b.params.base_close_threshold}")
    print(f"  blendWeight:  {b.params.blend_weight}")
    print(f"  minHoldBars:  {b.params.min_hold_bars}")
    print(f"  cooldownBars: {b.params.cooldown_bars}")
    print(f"  maxHoldBars:  {b.params.max_hold_bars}")
    print(f"  minEdge:      {b.params.min_edge}")
    print(f"  minSignalToNoise:{b.params.min_signal_to_noise}")
    print(f"  edgeBuffer:   {b.params.edge_buffer}")
    print(f"  costAwareEdge:{b.params.cost_aware_edge}")
    print(f"  trendLookback:{b.params.trend_lookback}")
    print(f"  maxPositionSize:{b.params.max_position_size}")
    print(f"  volTarget:    {b.params.vol_target}")
    print(f"  volLookback:  {b.params.vol_lookback}")
    print(f"  volEwmaAlpha: {b.params.vol_ewma_alpha}")
    print(f"  volFloor:     {b.params.vol_floor}")
    print(f"  volScaleMax:  {b.params.vol_scale_max}")
    print(f"  maxVolatility: {b.params.max_volatility}")
    print(f"  periodsPerYear:{b.params.periods_per_year}")
    print(f"  kalmanMarketTopN:{b.params.kalman_market_top_n}")
    print(f"  normalization: {b.params.normalization}")
    print(f"  epochs:        {b.params.epochs}")
    print(f"  hiddenSize:    {b.params.hidden_size}")
    print(f"  lr:            {b.params.learning_rate}")
    print(f"  valRatio:      {b.params.val_ratio}")
    print(f"  patience:      {b.params.patience}")
    print(f"  walkForwardFolds:{b.params.walk_forward_folds}")
    print(f"  tuneStressVolMult:{b.params.tune_stress_vol_mult}")
    print(f"  tuneStressShock: {b.params.tune_stress_shock}")
    print(f"  tuneStressWeight:{b.params.tune_stress_weight}")
    print(f"  gradClip:      {b.params.grad_clip}")
    print(f"  fee:           {b.params.fee}")
    print(f"  slippage:      {b.params.slippage}")
    print(f"  spread:        {b.params.spread}")
    print(f"  intrabarFill:  {b.params.intrabar_fill}")
    print(f"  stopLoss:      {b.params.stop_loss}")
    print(f"  takeProfit:    {b.params.take_profit}")
    print(f"  trailingStop:  {b.params.trailing_stop}")
    print(f"  stopLossVolMult:   {b.params.stop_loss_vol_mult}")
    print(f"  takeProfitVolMult: {b.params.take_profit_vol_mult}")
    print(f"  trailingStopVolMult:{b.params.trailing_stop_vol_mult}")
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
        successful = [tr for tr in records if tr.eligible and tr.final_equity is not None and tr.score is not None]
        successful_sorted = sorted(successful, key=lambda tr: tr.score or 0, reverse=True)
        source = data_source
        combos = []
        for rank, tr in enumerate(successful_sorted[:10], start=1):
            sharpe = metric_float(tr.metrics, "sharpe", 0.0)
            max_dd = metric_float(tr.metrics, "maxDrawdown", 0.0)
            turnover = metric_float(tr.metrics, "turnover", 0.0)
            round_trips = metric_int(tr.metrics, "roundTrips", 0)
            symbol = symbol_label
            combo = {
                "rank": rank,
                "finalEquity": tr.final_equity,
                "objective": tr.objective,
                "score": tr.score,
                "openThreshold": tr.open_threshold,
                "closeThreshold": tr.close_threshold,
                "source": source,
                "metrics": {
                    "sharpe": sharpe,
                    "maxDrawdown": max_dd,
                    "turnover": turnover,
                    "roundTrips": round_trips,
                },
                "params": {
                    "interval": tr.params.interval,
                    "bars": tr.params.bars,
                    "method": tr.params.method,
                    "blendWeight": tr.params.blend_weight,
                    "positioning": tr.params.positioning,
                    "normalization": tr.params.normalization,
                    "baseOpenThreshold": tr.params.base_open_threshold,
                    "baseCloseThreshold": tr.params.base_close_threshold,
                    "minHoldBars": tr.params.min_hold_bars,
                    "cooldownBars": tr.params.cooldown_bars,
                    "maxHoldBars": tr.params.max_hold_bars,
                    "minEdge": tr.params.min_edge,
                    "minSignalToNoise": tr.params.min_signal_to_noise,
                    "edgeBuffer": tr.params.edge_buffer,
                    "costAwareEdge": tr.params.cost_aware_edge,
                    "trendLookback": tr.params.trend_lookback,
                    "maxPositionSize": tr.params.max_position_size,
                    "volTarget": tr.params.vol_target,
                    "volLookback": tr.params.vol_lookback,
                    "volEwmaAlpha": tr.params.vol_ewma_alpha,
                    "volFloor": tr.params.vol_floor,
                    "volScaleMax": tr.params.vol_scale_max,
                    "maxVolatility": tr.params.max_volatility,
                    "periodsPerYear": tr.params.periods_per_year,
                    "kalmanMarketTopN": tr.params.kalman_market_top_n,
                    "fee": tr.params.fee,
                    "epochs": tr.params.epochs,
                    "hiddenSize": tr.params.hidden_size,
                    "learningRate": tr.params.learning_rate,
                    "valRatio": tr.params.val_ratio,
                    "patience": tr.params.patience,
                    "walkForwardFolds": tr.params.walk_forward_folds,
                    "tuneStressVolMult": tr.params.tune_stress_vol_mult,
                    "tuneStressShock": tr.params.tune_stress_shock,
                    "tuneStressWeight": tr.params.tune_stress_weight,
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
                    "binanceSymbol": symbol,
                },
            }
            ops = extract_operations(tr.stdout_json)
            if ops is not None:
                combo["operations"] = ops
            combos.append(combo)
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

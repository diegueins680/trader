#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from functools import cmp_to_key
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return int(float(s))
        except ValueError:
            return None
    return None


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("1", "true", "t", "yes", "y", "on"):
            return True
        if s in ("0", "false", "f", "no", "n", "off"):
            return False
    return None


def _normalize_symbol(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip().upper()
        return s if s else None
    return _normalize_symbol(str(value))


def _normalize_objective(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip().lower()
        return s if s else None
    return _normalize_objective(str(value))


def _coerce_operations(value: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(value, list):
        return None
    ops: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            ops.append(item)
    return ops if ops else None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_top_combos(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = _read_json(path)
    combos = data.get("combos")
    if not isinstance(combos, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in combos:
        if isinstance(item, dict):
            out.append(item)
    return out


def _load_combos_from_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict):
            continue
        if not rec.get("ok"):
            continue
        final_equity = _coerce_float(rec.get("finalEquity"))
        if final_equity is None:
            continue
        source_raw = rec.get("source")
        source = source_raw if isinstance(source_raw, str) else None
        params = rec.get("params")
        if not isinstance(params, dict):
            params = {}
        out.append(
            {
                "finalEquity": final_equity,
                "objective": _normalize_objective(rec.get("objective")),
                "score": _coerce_float(rec.get("score")),
                "openThreshold": _coerce_float(rec.get("openThreshold")),
                "closeThreshold": _coerce_float(rec.get("closeThreshold")),
                "source": source,
                "metrics": rec.get("metrics") if isinstance(rec.get("metrics"), dict) else None,
                "operations": _coerce_operations(rec.get("operations")),
                "params": params,
            }
        )
    return out


def _normalize_combo(combo: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    final_equity = _coerce_float(combo.get("finalEquity"))
    if final_equity is None:
        return None

    source_raw = combo.get("source")
    source_s = source_raw.strip().lower() if isinstance(source_raw, str) else ""
    source_map = {
        "binance": "binance",
        "coinbase": "coinbase",
        "kraken": "kraken",
        "poloniex": "poloniex",
        "csv": "csv",
    }
    source = source_map.get(source_s)

    params_raw = combo.get("params")
    if not isinstance(params_raw, dict):
        params_raw = {}

    platform_raw = params_raw.get("platform")
    platform_s = platform_raw.strip().lower() if isinstance(platform_raw, str) else ""
    if platform_s not in ("binance", "coinbase", "kraken", "poloniex"):
        platform_s = source if source in ("binance", "coinbase", "kraken", "poloniex") else None

    objective = _normalize_objective(combo.get("objective"))
    score = _coerce_float(combo.get("score"))
    metrics = combo.get("metrics") if isinstance(combo.get("metrics"), dict) else None
    operations = _coerce_operations(combo.get("operations"))

    interval = params_raw.get("interval")
    interval_s = interval if isinstance(interval, str) else (str(interval) if interval is not None else "")

    bars = _coerce_int(params_raw.get("bars"))
    if bars is None:
        bars = 0

    method = params_raw.get("method")
    method_s = method if isinstance(method, str) else (str(method) if method is not None else "")

    normalization = params_raw.get("normalization")
    normalization_s = normalization if isinstance(normalization, str) else (str(normalization) if normalization is not None else "")

    epochs = _coerce_int(params_raw.get("epochs"))
    if epochs is None:
        epochs = 0

    hidden_size = _coerce_int(params_raw.get("hiddenSize"))
    learning_rate = _coerce_float(params_raw.get("learningRate"))
    val_ratio = _coerce_float(params_raw.get("valRatio"))
    patience = _coerce_int(params_raw.get("patience"))
    grad_clip = _coerce_float(params_raw.get("gradClip"))

    positioning = params_raw.get("positioning")
    positioning_s = positioning if isinstance(positioning, str) else (str(positioning) if positioning is not None else "")

    slippage = _coerce_float(params_raw.get("slippage"))
    spread = _coerce_float(params_raw.get("spread"))

    cost_aware_edge = _coerce_bool(params_raw.get("costAwareEdge")) or False
    confirm_conformal = _coerce_bool(params_raw.get("confirmConformal")) or False
    confirm_quantiles = _coerce_bool(params_raw.get("confirmQuantiles")) or False
    confidence_sizing = _coerce_bool(params_raw.get("confidenceSizing")) or False
    symbol = _normalize_symbol(params_raw.get("binanceSymbol") or params_raw.get("symbol"))

    normalized_params: Dict[str, Any] = dict(params_raw)
    normalized_params.update(
        {
            "platform": platform_s,
            "interval": interval_s,
            "bars": bars,
            "method": method_s,
            "normalization": normalization_s,
            "positioning": positioning_s,
            "baseOpenThreshold": _coerce_float(params_raw.get("baseOpenThreshold")),
            "baseCloseThreshold": _coerce_float(params_raw.get("baseCloseThreshold")),
            "blendWeight": _coerce_float(params_raw.get("blendWeight")),
            "minHoldBars": _coerce_int(params_raw.get("minHoldBars")),
            "cooldownBars": _coerce_int(params_raw.get("cooldownBars")),
            "maxHoldBars": _coerce_int(params_raw.get("maxHoldBars")),
            "minEdge": _coerce_float(params_raw.get("minEdge")),
            "edgeBuffer": _coerce_float(params_raw.get("edgeBuffer")),
            "costAwareEdge": cost_aware_edge,
            "trendLookback": _coerce_int(params_raw.get("trendLookback")),
            "maxPositionSize": _coerce_float(params_raw.get("maxPositionSize")),
            "volTarget": _coerce_float(params_raw.get("volTarget")),
            "volLookback": _coerce_int(params_raw.get("volLookback")),
            "volEwmaAlpha": _coerce_float(params_raw.get("volEwmaAlpha")),
            "volFloor": _coerce_float(params_raw.get("volFloor")),
            "volScaleMax": _coerce_float(params_raw.get("volScaleMax")),
            "maxVolatility": _coerce_float(params_raw.get("maxVolatility")),
            "periodsPerYear": _coerce_float(params_raw.get("periodsPerYear")),
            "kalmanMarketTopN": _coerce_int(params_raw.get("kalmanMarketTopN")),
            "fee": _coerce_float(params_raw.get("fee")),
            "epochs": epochs,
            "hiddenSize": hidden_size,
            "learningRate": learning_rate,
            "valRatio": val_ratio,
            "patience": patience,
            "walkForwardFolds": _coerce_int(params_raw.get("walkForwardFolds")),
            "tuneStressVolMult": _coerce_float(params_raw.get("tuneStressVolMult")),
            "tuneStressShock": _coerce_float(params_raw.get("tuneStressShock")),
            "tuneStressWeight": _coerce_float(params_raw.get("tuneStressWeight")),
            "gradClip": grad_clip,
            "slippage": slippage,
            "spread": spread,
            "intrabarFill": params_raw.get("intrabarFill")
            if isinstance(params_raw.get("intrabarFill"), str)
            else (str(params_raw.get("intrabarFill")) if params_raw.get("intrabarFill") is not None else None),
            "stopLoss": _coerce_float(params_raw.get("stopLoss")),
            "takeProfit": _coerce_float(params_raw.get("takeProfit")),
            "trailingStop": _coerce_float(params_raw.get("trailingStop")),
            "maxDrawdown": _coerce_float(params_raw.get("maxDrawdown")),
            "maxDailyLoss": _coerce_float(params_raw.get("maxDailyLoss")),
            "maxOrderErrors": _coerce_int(params_raw.get("maxOrderErrors")),
            "kalmanDt": _coerce_float(params_raw.get("kalmanDt")),
            "kalmanProcessVar": _coerce_float(params_raw.get("kalmanProcessVar")),
            "kalmanMeasurementVar": _coerce_float(params_raw.get("kalmanMeasurementVar")),
            "kalmanZMin": _coerce_float(params_raw.get("kalmanZMin")),
            "kalmanZMax": _coerce_float(params_raw.get("kalmanZMax")),
            "maxHighVolProb": _coerce_float(params_raw.get("maxHighVolProb")),
            "maxConformalWidth": _coerce_float(params_raw.get("maxConformalWidth")),
            "maxQuantileWidth": _coerce_float(params_raw.get("maxQuantileWidth")),
            "confirmConformal": confirm_conformal,
            "confirmQuantiles": confirm_quantiles,
            "confidenceSizing": confidence_sizing,
            "minPositionSize": _coerce_float(params_raw.get("minPositionSize")),
            "binanceSymbol": symbol,
        }
    )

    return {
        "finalEquity": final_equity,
        "objective": objective,
        "score": score,
        "openThreshold": _coerce_float(combo.get("openThreshold")),
        "closeThreshold": _coerce_float(combo.get("closeThreshold")),
        "source": source,
        "metrics": metrics,
        "operations": operations,
        "params": normalized_params,
    }


def _signature(combo: Dict[str, Any]) -> Tuple[Any, ...]:
    params = combo.get("params") if isinstance(combo.get("params"), dict) else {}

    def p(name: str) -> Any:
        return params.get(name)

    return (
        combo.get("source"),
        p("platform"),
        p("interval"),
        p("bars"),
        p("binanceSymbol"),
        p("method"),
        p("blendWeight"),
        p("normalization"),
        p("positioning"),
        p("baseOpenThreshold"),
        p("baseCloseThreshold"),
        p("minHoldBars"),
        p("cooldownBars"),
        p("maxHoldBars"),
        p("minEdge"),
        p("edgeBuffer"),
        p("costAwareEdge"),
        p("trendLookback"),
        p("maxPositionSize"),
        p("volTarget"),
        p("volLookback"),
        p("volEwmaAlpha"),
        p("volFloor"),
        p("volScaleMax"),
        p("maxVolatility"),
        p("periodsPerYear"),
        p("walkForwardFolds"),
        p("tuneStressVolMult"),
        p("tuneStressShock"),
        p("tuneStressWeight"),
        p("fee"),
        p("epochs"),
        p("hiddenSize"),
        p("learningRate"),
        p("valRatio"),
        p("patience"),
        p("gradClip"),
        p("slippage"),
        p("spread"),
        p("intrabarFill"),
        p("stopLoss"),
        p("takeProfit"),
        p("trailingStop"),
        p("maxDrawdown"),
        p("maxDailyLoss"),
        p("maxOrderErrors"),
        p("kalmanDt"),
        p("kalmanProcessVar"),
        p("kalmanMeasurementVar"),
        p("kalmanZMin"),
        p("kalmanZMax"),
        p("kalmanMarketTopN"),
        p("maxHighVolProb"),
        p("maxConformalWidth"),
        p("maxQuantileWidth"),
        p("confirmConformal"),
        p("confirmQuantiles"),
        p("confidenceSizing"),
        p("minPositionSize"),
        combo.get("openThreshold"),
        combo.get("closeThreshold"),
    )


def _merge_combos(sources: Iterable[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    best_by_key: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for src in sources:
        for raw in src:
            norm = _normalize_combo(raw)
            if norm is None:
                continue
            key = _signature(norm)
            prev = best_by_key.get(key)
            score = _coerce_float(norm.get("score"))
            prev_score = _coerce_float(prev.get("score")) if prev else None
            objective = norm.get("objective")
            prev_objective = prev.get("objective") if prev else None
            if prev is None:
                best_by_key[key] = norm
            elif objective == prev_objective and (score is not None or prev_score is not None):
                if (score if score is not None else float("-inf")) > (prev_score if prev_score is not None else float("-inf")):
                    best_by_key[key] = norm
            elif float(norm["finalEquity"]) > float(prev["finalEquity"]):
                best_by_key[key] = norm
    return list(best_by_key.values())


def _compare_desc(a: Optional[float], b: Optional[float]) -> int:
    a_val = float("-inf") if a is None else float(a)
    b_val = float("-inf") if b is None else float(b)
    if a_val > b_val:
        return -1
    if a_val < b_val:
        return 1
    return 0


def _compare_combos(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    obj_a = a.get("objective")
    obj_b = b.get("objective")
    score_a = _coerce_float(a.get("score"))
    score_b = _coerce_float(b.get("score"))
    eq_a = _coerce_float(a.get("finalEquity"))
    eq_b = _coerce_float(b.get("finalEquity"))

    if obj_a == obj_b:
        score_cmp = _compare_desc(score_a, score_b)
        if score_cmp != 0:
            return score_cmp
        return _compare_desc(eq_a, eq_b)

    return _compare_desc(eq_a, eq_b)


def _write_top_json(path: Path, combos: List[Dict[str, Any]], max_items: int) -> None:
    combos_sorted = sorted(combos, key=cmp_to_key(_compare_combos))[:max_items]

    export: Dict[str, Any] = {
        "generatedAtMs": int(time.time() * 1000),
        "source": "merge_top_combos.py",
        "combos": [],
    }

    for rank, combo in enumerate(combos_sorted, start=1):
        metrics = combo.get("metrics")
        metrics_out = metrics if isinstance(metrics, dict) else None
        combo_out: Dict[str, Any] = {
            "rank": rank,
            "finalEquity": combo["finalEquity"],
            "objective": combo.get("objective"),
            "score": combo.get("score"),
            "openThreshold": combo.get("openThreshold"),
            "closeThreshold": combo.get("closeThreshold"),
            "source": combo.get("source"),
            "metrics": metrics_out,
            "params": combo.get("params", {}),
        }
        operations = combo.get("operations")
        if isinstance(operations, list):
            combo_out["operations"] = operations
        export["combos"].append(combo_out)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(export, indent=2) + "\n", encoding="utf-8")


def _archive_top_json(history_dir: Optional[Path], out_path: Path) -> None:
    if history_dir is None:
        return
    try:
        history_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    stamp = int(time.time() * 1000)
    archive_path = history_dir / f"top-combos-{stamp}.json"
    try:
        shutil.copy2(out_path, archive_path)
    except OSError:
        return


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Merge optimizer outputs into a single top-combos.json for the web UI.")
    parser.add_argument(
        "--top-json",
        type=str,
        default="haskell/web/public/top-combos.json",
        help="Existing top-combos.json path (default: haskell/web/public/top-combos.json).",
    )
    parser.add_argument(
        "--from-jsonl",
        action="append",
        default=[],
        help="JSONL trial output(s) to merge (repeatable). If omitted, auto-discovers .tmp/optimize_equity*.jsonl and haskell/.tmp/optimize_equity*.jsonl.",
    )
    parser.add_argument(
        "--from-top-json",
        action="append",
        default=[],
        help="Additional top-combos JSON files to merge (repeatable).",
    )
    parser.add_argument("--out", type=str, default="", help="Output path (default: overwrite --top-json).")
    parser.add_argument("--max", type=int, default=200, help="Max combos to keep (default: 200).")
    parser.add_argument(
        "--history-dir",
        type=str,
        default="",
        help="Optional directory to write timestamped history snapshots.",
    )
    parser.add_argument(
        "--copy-to-dist",
        action="store_true",
        help="If haskell/web/dist exists, also copy the merged file to haskell/web/dist/top-combos.json.",
    )

    args = parser.parse_args(argv)

    top_json_path = Path(args.top_json).expanduser()
    out_path = Path(args.out).expanduser() if args.out else top_json_path

    jsonl_paths: List[Path] = []
    for raw in args.from_jsonl:
        jsonl_paths.append(Path(raw).expanduser())
    if not jsonl_paths:
        jsonl_paths.extend(sorted(Path(".tmp").glob("optimize_equity*.jsonl")))
        jsonl_paths.extend(sorted(Path("haskell/.tmp").glob("optimize_equity*.jsonl")))

    other_top_json_paths: List[Path] = [Path(p).expanduser() for p in args.from_top_json]

    sources: List[List[Dict[str, Any]]] = []
    sources.append(_load_top_combos(top_json_path))
    for p in other_top_json_paths:
        sources.append(_load_top_combos(p))
    for p in jsonl_paths:
        sources.append(_load_combos_from_jsonl(p))

    merged = _merge_combos(sources)
    _write_top_json(out_path, merged, max_items=max(1, int(args.max)))
    history_dir = Path(args.history_dir).expanduser() if args.history_dir else None
    _archive_top_json(history_dir, out_path)

    if args.copy_to_dist:
        dist_dir = Path("haskell/web/dist")
        if dist_dir.exists() and dist_dir.is_dir():
            (dist_dir / "top-combos.json").write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"Merged {sum(len(s) for s in sources)} candidates into {len(merged)} unique combos.")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

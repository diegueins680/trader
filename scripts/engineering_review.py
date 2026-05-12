#!/usr/bin/env python3
"""
Engineering Review Script — 2026-05-11
======================================

This script demonstrates the correct engineering response to zero-trade behavior:
1. Collect gate-level telemetry
2. Identify the binding gate
3. Calibrate thresholds from historical edge distributions
4. Run sensitivity analysis on parameters
5. Apply ONLY the validated change

Usage:
    python scripts/engineering_review.py --backtest-json backtest.json \
        --edges-csv edges.csv --output report.json

Design invariants:
- No parameter change without data.
- All changes must include sensitivity analysis.
- Thresholds must be calibrated, not guessed.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import statistics


@dataclass
class GateRejection:
    gate: str
    reason: str
    count: int


@dataclass
class TelemetrySummary:
    total_bars: int
    total_candidates: int
    total_rejections: int
    rejection_rate: float
    candidate_rate: float
    diagnosis: str
    binding_gate: Optional[str]


@dataclass
class EdgeDistribution:
    sample_size: int
    mean: float
    median: float
    stddev: float
    min_edge: float
    max_edge: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float


@dataclass
class CalibrationResult:
    method: str
    suggested_threshold: float
    headroom_threshold: float
    fee_buffer_threshold: float
    confidence_interval: Tuple[float, float]
    recommendation: str


@dataclass
class SensitivityPoint:
    parameter_value: float
    sharpe: float
    max_drawdown: float
    trade_count: int
    win_rate: float
    profit_factor: float


@dataclass
class SensitivityReport:
    parameter_name: str
    baseline: float
    elasticity: float
    recommendation: str
    points: List[SensitivityPoint]


def compute_edge_distribution(edges: List[float]) -> Optional[EdgeDistribution]:
    """Compute statistical distribution of edge values."""
    if not edges or any(not (0 <= e < float('inf')) for e in edges):
        return None
    
    sorted_edges = sorted(edges)
    n = len(sorted_edges)
    
    def percentile(p: float) -> float:
        idx = (p / 100.0) * (n - 1)
        lower = int(idx)
        upper = min(lower + 1, n - 1)
        frac = idx - lower
        return sorted_edges[lower] + frac * (sorted_edges[upper] - sorted_edges[lower])
    
    mean = statistics.mean(edges)
    stddev = statistics.stdev(edges) if n > 1 else 0.0
    
    return EdgeDistribution(
        sample_size=n,
        mean=mean,
        median=percentile(50),
        stddev=stddev,
        min_edge=min(edges),
        max_edge=max(edges),
        p10=percentile(10),
        p25=percentile(25),
        p50=percentile(50),
        p75=percentile(75),
        p90=percentile(90),
        p95=percentile(95),
        p99=percentile(99),
    )


def calibrate_threshold(
    edges: List[float],
    method: str = "percentile",
    percentile_target: float = 75.0,
    stddev_multiplier: float = 1.0,
    fee_floor: float = 0.001
) -> Optional[CalibrationResult]:
    """Calibrate open threshold from edge distribution."""
    dist = compute_edge_distribution(edges)
    if dist is None:
        return None
    
    if method == "percentile":
        threshold = getattr(dist, f"p{int(percentile_target)}")
    elif method == "stddev":
        threshold = dist.mean + stddev_multiplier * dist.stddev
    elif method == "hybrid":
        threshold = max(
            getattr(dist, f"p{int(percentile_target)}"),
            dist.mean + stddev_multiplier * dist.stddev
        )
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    
    headroom = threshold / 1.5
    fee_buffer = threshold + fee_floor
    
    # 95% confidence interval for threshold
    if dist.sample_size > 1:
        se = dist.stddev / (dist.sample_size ** 0.5)
        ci_lower = max(0, threshold - 1.96 * se)
        ci_upper = threshold + 1.96 * se
    else:
        ci_lower = ci_upper = threshold
    
    if dist.sample_size < 100:
        rec = f"INSUFFICIENT_SAMPLE: Need >= 100 edges (got {dist.sample_size})"
    elif threshold > dist.p95:
        rec = "CONSERVATIVE: Threshold above 95th percentile — may produce very few trades"
    elif threshold < dist.p25:
        rec = "AGGRESSIVE: Threshold below 25th percentile — may produce excessive trades"
    else:
        rec = "BALANCED: Threshold within interquartile range — recommended"
    
    return CalibrationResult(
        method=method,
        suggested_threshold=threshold,
        headroom_threshold=headroom,
        fee_buffer_threshold=fee_buffer,
        confidence_interval=(ci_lower, ci_upper),
        recommendation=rec
    )


def analyze_telemetry(telemetry: dict) -> TelemetrySummary:
    """Analyze gate telemetry to identify binding constraints."""
    total_bars = telemetry.get("totalBars", 0)
    total_candidates = telemetry.get("totalCandidates", 0)
    total_rejections = telemetry.get("totalRejections", 0)
    
    rejection_rate = total_rejections / max(1, total_candidates)
    candidate_rate = total_candidates / max(1, total_bars)
    
    if total_candidates == 0:
        diagnosis = "NO_CANDIDATES"
    elif total_rejections == total_candidates:
        diagnosis = "ALL_REJECTED"
    elif rejection_rate > 0.9:
        diagnosis = "MOSTLY_REJECTED"
    else:
        diagnosis = "NORMAL"
    
    # Find binding gate
    per_gate = telemetry.get("perGateCounts", {})
    binding_gate = max(per_gate, key=per_gate.get) if per_gate else None
    
    return TelemetrySummary(
        total_bars=total_bars,
        total_candidates=total_candidates,
        total_rejections=total_rejections,
        rejection_rate=rejection_rate,
        candidate_rate=candidate_rate,
        diagnosis=diagnosis,
        binding_gate=binding_gate
    )


def generate_report(
    telemetry: Optional[dict],
    edges: Optional[List[float]],
    sensitivity: Optional[List[dict]]
) -> dict:
    """Generate a comprehensive engineering review report."""
    report = {
        "review_date": "2026-05-11T23:23:00Z",
        "review_type": "daily_engineering_review",
        "findings": [],
        "recommendations": [],
        "actions": []
    }
    
    # Telemetry analysis
    if telemetry:
        summary = analyze_telemetry(telemetry)
        report["telemetry_summary"] = asdict(summary)
        
        if summary.diagnosis == "NO_CANDIDATES":
            report["findings"].append(
                "CRITICAL: Zero candidates generated. Strategy logic or data issue."
            )
            report["actions"].append("Audit strategy candidate generation logic")
        elif summary.diagnosis == "ALL_REJECTED":
            report["findings"].append(
                f"CRITICAL: All {summary.total_candidates} candidates rejected. "
                f"Binding gate: {summary.binding_gate}"
            )
            report["actions"].append(
                f"Investigate {summary.binding_gate} gate thresholds and calibration"
            )
        elif summary.diagnosis == "MOSTLY_REJECTED":
            report["findings"].append(
                f"WARNING: {summary.rejection_rate:.1%} rejection rate. "
                f"Binding gate: {summary.binding_gate}"
            )
    
    # Threshold calibration
    if edges:
        calib = calibrate_threshold(edges, method="hybrid", percentile_target=75, stddev_multiplier=1.0)
        if calib:
            report["threshold_calibration"] = asdict(calib)
            report["findings"].append(
                f"THRESHOLD: Calibrated threshold = {calib.suggested_threshold:.6f} "
                f"({calib.recommendation})"
            )
            if "INSUFFICIENT" not in calib.recommendation:
                report["recommendations"].append(
                    f"Set --open-threshold to {calib.suggested_threshold:.6f} "
                    f"(95% CI: [{calib.confidence_interval[0]:.6f}, {calib.confidence_interval[1]:.6f}])"
                )
    
    # Sensitivity analysis
    if sensitivity:
        reports = []
        for param in sensitivity:
            points = [SensitivityPoint(**p) for p in param.get("points", [])]
            # Compute elasticity from points
            baseline = param.get("baseline", 0)
            baseline_point = next((p for p in points if abs(p.parameter_value - baseline) < 1e-9), None)
            if baseline_point and baseline > 0:
                elasticity = sum(
                    abs((p.sharpe - baseline_point.sharpe) / max(0.001, baseline_point.sharpe))
                    / abs((p.parameter_value - baseline) / baseline)
                    for p in points if p != baseline_point and baseline_point.sharpe > 0
                ) / max(1, len(points) - 1)
            else:
                elasticity = 0
            
            reports.append(SensitivityReport(
                parameter_name=param["name"],
                baseline=baseline,
                elasticity=elasticity,
                recommendation=param.get("recommendation", ""),
                points=points
            ))
        
        reports.sort(key=lambda r: r.elasticity, reverse=True)
        report["sensitivity_analysis"] = [asdict(r) for r in reports]
        
        if reports:
            top = reports[0]
            report["findings"].append(
                f"SENSITIVITY: Most sensitive parameter is '{top.parameter_name}' "
                f"with elasticity {top.elasticity:.4f}"
            )
            report["recommendations"].append(
                f"Prioritize tuning '{top.parameter_name}' — it has the highest "
                f"impact on Sharpe ratio"
            )
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Trader Engineering Review")
    parser.add_argument("--telemetry-json", help="Path to gate telemetry JSON")
    parser.add_argument("--edges-csv", help="Path to edge values CSV")
    parser.add_argument("--sensitivity-json", help="Path to sensitivity analysis JSON")
    parser.add_argument("--output", default="engineering_review.json", help="Output report path")
    args = parser.parse_args()
    
    telemetry = None
    if args.telemetry_json:
        with open(args.telemetry_json) as f:
            telemetry = json.load(f)
    
    edges = None
    if args.edges_csv:
        with open(args.edges_csv) as f:
            edges = [float(line.strip()) for line in f if line.strip()]
    
    sensitivity = None
    if args.sensitivity_json:
        with open(args.sensitivity_json) as f:
            sensitivity = json.load(f)
    
    report = generate_report(telemetry, edges, sensitivity)
    
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"Engineering review written to {args.output}")
    print(f"\nKey findings: {len(report['findings'])}")
    for finding in report["findings"]:
        print(f"  • {finding}")
    
    print(f"\nRecommendations: {len(report['recommendations'])}")
    for rec in report["recommendations"]:
        print(f"  • {rec}")
    
    print(f"\nActions required: {len(report['actions'])}")
    for action in report["actions"]:
        print(f"  • {action}")


if __name__ == "__main__":
    main()

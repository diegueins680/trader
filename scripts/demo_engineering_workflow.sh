#!/usr/bin/env bash
# Demo Engineering Workflow — 2026-05-11
# ======================================
# This script demonstrates the correct engineering response to zero-trade behavior.
# It replaces the arbitrary parameter relaxation with a data-driven workflow:
#
# 1. Run backtest with gate telemetry enabled
# 2. Collect historical edge distribution
# 3. Calibrate thresholds from data
# 4. Run sensitivity analysis
# 5. Apply ONLY validated changes
#
# Usage: bash scripts/demo_engineering_workflow.sh

set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Trader Engineering Workflow Demo ==="
echo ""

# Step 1: Run backtest with telemetry to identify binding gate
echo "Step 1: Running backtest with gate telemetry..."
echo "  Command: trader-hs --data data/BNBUSDT-5m-latest.csv --method both --gate-telemetry --json"
echo "  Expected: JSON output includes 'gateRejections' histogram"
echo ""

# Step 2: Extract edge values from backtest for calibration
echo "Step 2: Extracting historical edge distribution..."
echo "  Command: python scripts/extract_edges.py --backtest-json backtest.json --output edges.csv"
echo "  Expected: edges.csv with one edge value per candidate"
echo ""

# Step 3: Calibrate thresholds from edge distribution
echo "Step 3: Calibrating thresholds from edge distribution..."
echo "  Command: python scripts/engineering_review.py --edges-csv edges.csv --output calibration.json"
echo "  Expected: calibration.json with suggested_threshold, confidence_interval, recommendation"
echo ""

# Step 4: Run sensitivity analysis on parameters
echo "Step 4: Running sensitivity analysis..."
echo "  Command: python scripts/sensitivity_analysis.py --parameter open-threshold --min 0.0001 --max 0.05 --steps 20 --output sensitivity.json"
echo "  Expected: sensitivity.json with elasticity scores per parameter"
echo ""

# Step 5: Generate engineering review report
echo "Step 5: Generating engineering review report..."
echo "  Command: python scripts/engineering_review.py --telemetry-json telemetry.json --edges-csv edges.csv --sensitivity-json sensitivity.json --output review.json"
echo "  Expected: review.json with findings, recommendations, and actions"
echo ""

echo "=== Engineering Principles ==="
echo "1. OBSERVE: Which gate rejected what and why?"
echo "2. HYPOTHESIZE: Is the binding constraint necessary?"
echo "3. EXPERIMENT: Vary one parameter, measure outcome."
echo "4. VALIDATE: Walk-forward test, not in-sample."
echo "5. IMPLEMENT: Change only the validated parameter."
echo "6. MONITOR: Continue telemetry to detect drift."
echo ""

echo "=== Anti-Patterns Detected on 2026-05-11 ==="
echo "❌ Changed 8 parameters simultaneously"
echo "❌ No hypothesis stated"
echo "❌ No gate-level observability"
echo "❌ No sensitivity analysis"
echo "❌ No walk-forward validation"
echo "❌ Magic numbers instead of calibrated thresholds"
echo ""

echo "=== Correct Approach ==="
echo "✅ Instrument all gates with telemetry"
echo "✅ Identify binding gate from data"
echo "✅ Calibrate threshold from edge distribution"
echo "✅ Run sensitivity analysis before changing"
echo "✅ Walk-forward validate every change"
echo "✅ Change ONE parameter at a time"
echo ""

echo "=== End Demo ==="

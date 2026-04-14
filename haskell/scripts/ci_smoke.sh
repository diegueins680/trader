#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
haskell_dir="$(cd "$script_dir/.." && pwd)"
repo_root="$(cd "$haskell_dir/.." && pwd)"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

cd "$haskell_dir"

public_surface_witness="$tmpdir/trader-trading-public-surface.hs"
cat >"$public_surface_witness" <<'EOF'
module PublicSurfaceWitness where

import qualified Data.Vector as V
import Trader.SignalGates (signalEntryHeadroomThresholdCap)
import Trader.Trading (
    BacktestResult,
    EnsembleConfig,
    ExitReason,
    IntrabarFill (..),
    Positioning (..),
    StepMeta,
    Trade,
    simulateEnsembleVWithHLChecked
 )

publicSurfaceShim ::
    EnsembleConfig ->
    Int ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    Maybe (V.Vector StepMeta) ->
    Either String BacktestResult
publicSurfaceShim = simulateEnsembleVWithHLChecked

publicSurfaceReachable :: Bool
publicSurfaceReachable =
    case (Nothing :: Maybe ExitReason, Nothing :: Maybe Trade) of
        (Nothing, Nothing) ->
            signalEntryHeadroomThresholdCap 0.03 `seq`
                (LongFlat `seq`
                    (StopFirst `seq`
                        (publicSurfaceShim `seq` True)))
        _ -> False
EOF

cabal exec ghc -- -fno-code -iapp "$public_surface_witness" >/dev/null

trader_bin="$(cabal list-bin exe:trader-hs)"
merge_bin="$(cabal list-bin exe:merge-top-combos)"
optimize_bin="$(cabal list-bin exe:optimize-equity)"

trader_out="$tmpdir/trader-smoke.json"
"$trader_bin" \
  --data "$repo_root/data/sample_prices.csv" \
  --price-column close \
  --epochs 1 \
  --hidden-size 4 \
  --json >"$trader_out"

grep -q '"backtest"' "$trader_out"
grep -q '"finalEquity"' "$trader_out"

merge_in="$tmpdir/merge-input.jsonl"
cat >"$merge_in" <<'EOF'
{"ok":true,"finalEquity":123.45,"params":{"platform":"binance","symbol":"BTCUSDT","method":"kalman"}}
EOF

merge_out="$tmpdir/top-combos.json"
"$merge_bin" \
  --top-json "$tmpdir/base-top-combos.json" \
  --from-jsonl "$merge_in" \
  --out "$merge_out" >/dev/null

grep -q '"finalEquity": 123.45' "$merge_out"
grep -q '"platform": "binance"' "$merge_out"
grep -q '"symbol": "BTCUSDT"' "$merge_out"

optimize_out="$tmpdir/optimizer-smoke.jsonl"
optimize_log="$tmpdir/optimizer-smoke.log"
if ! "$optimize_bin" \
  --data "$repo_root/data/sample_prices.csv" \
  --price-column close \
  --binary "$trader_bin" \
  --output "$optimize_out" \
  --trials 1 \
  --seed-trials 0 \
  --seed 7 \
  --bars-min 220 \
  --bars-max 220 \
  --epochs-min 1 \
  --epochs-max 1 \
  --hidden-size-min 4 \
  --hidden-size-max 4 \
  --timeout-sec 10 \
  --no-sweep-threshold \
  --min-round-trips 0 \
  --min-exposure 0 \
  --min-sharpe 0 \
  --min-calmar 0 \
  --min-wf-sharpe-mean 0 \
  --max-wf-sharpe-std 999999 \
  --min-annualized-return -999999 \
  --min-win-rate 0 \
  --min-profit-factor 0 >"$optimize_log" 2>&1; then
  cat "$optimize_log"
  exit 1
fi

grep -q '"ok":true' "$optimize_out"
grep -q '"source":"csv"' "$optimize_out"

echo "Smoke checks passed."
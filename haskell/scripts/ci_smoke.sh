#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
haskell_dir="$(cd "$script_dir/.." && pwd)"
repo_root="$(cd "$haskell_dir/.." && pwd)"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

cd "$haskell_dir"

trader_bin="$(cabal list-bin exe:trader-hs)"
merge_bin="$(cabal list-bin exe:merge-top-combos)"
optimize_bin="$(cabal list-bin exe:optimize-equity)"

public_surface_probe="$tmpdir/trading-public-surface-probe.hs"
cat >"$public_surface_probe" <<'EOF'
module Main (main) where

import qualified Data.Vector as V
import Trader.Trading
    ( BacktestResult
    , EnsembleConfig
    , EntryGateState
    , ExitReason (ExitSignal)
    , IntrabarFill (StopFirst)
    , PositionSide (SideLong)
    , Positioning (LongFlat)
    , StepMeta
    , Trade
    , TradeEntrySource (TradeEntrySignal)
    , TradingEntryGateInputs
    , exitReasonFromCode
    , mkEntryGateState
    , mkTradingEntryGateInputs
    , simulateEnsemble
    , simulateEnsembleVWithHLChecked
    , simulateEnsembleWithHLChecked
    , tradeEntrySourceCode
    )

emptySeries :: V.Vector Double
emptySeries = V.empty

emptyMeta :: Maybe (V.Vector StepMeta)
emptyMeta = Nothing

simulateVWitness :: Either String BacktestResult
simulateVWitness =
    simulateEnsembleVWithHLChecked
        undefined
        0
        emptySeries
        emptySeries
        emptySeries
        emptySeries
        emptySeries
        emptyMeta

simulateCheckedWitness :: Either String BacktestResult
simulateCheckedWitness =
    simulateEnsembleWithHLChecked
        undefined
        0
        emptySeries
        emptySeries
        emptySeries
        emptySeries
        emptySeries
        emptyMeta

simulateWitness :: BacktestResult
simulateWitness =
    simulateEnsemble
        undefined
        0
        emptySeries
        emptySeries
        emptySeries
        emptySeries
        emptySeries
        emptyMeta

surfaceTypesWitness ::
    ( EnsembleConfig -> ()
    , BacktestResult -> ()
    , StepMeta -> ()
    , Trade -> ()
    , TradingEntryGateInputs -> ()
    , EntryGateState -> ()
    )
surfaceTypesWitness =
    (const (), const (), const (), const (), const (), const ())

surfaceValuesWitness ::
    ( IntrabarFill
    , PositionSide
    , Positioning
    , ExitReason
    , TradeEntrySource
    , String
    , Maybe ExitReason
    )
surfaceValuesWitness =
    ( StopFirst
    , SideLong
    , LongFlat
    , ExitSignal
    , TradeEntrySignal
    , tradeEntrySourceCode TradeEntrySignal
    , exitReasonFromCode "SIGNAL"
    )

entryGateWitness :: EntryGateState
entryGateWitness = mkEntryGateState (mkTradingEntryGateInputs 0 0 Nothing)

main :: IO ()
main = pure ()
EOF

echo "Trader.Trading public surface probe"
cabal exec ghc -- -fno-code -iapp "$public_surface_probe" >/dev/null

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
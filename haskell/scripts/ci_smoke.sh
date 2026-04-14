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

test -x "$optimize_bin"

public_surface_src="$tmpdir/trading-public-surface-smoke.hs"
cat >"$public_surface_src" <<'EOF'
module Main (main) where

import qualified Data.Vector as V
import Trader.Trading
    ( BacktestResult (..)
    , EnsembleConfig (..)
    , ExitReason (..)
    , IntrabarFill (..)
    , Positioning (..)
    , StepMeta (..)
    , Trade (..)
    , simulateEnsembleVWithHLChecked
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

touchTradeExitReason :: Trade -> Maybe ExitReason
touchTradeExitReason = trExitReason

main :: IO ()
main = do
    let _ = publicSurfaceShim
        _ = touchTradeExitReason
        _ = StopFirst
        _ = LongFlat
        _ = ExitSignal
    putStrLn "Trader.Trading public surface ok."
EOF

public_surface_log="$tmpdir/trading-public-surface-smoke.log"
public_surface_bin="$tmpdir/trading-public-surface-smoke"
# Keep this witness import-only so CI checks the optimizer-facing module seam
# without changing or evaluating trading simulation behavior.
if ! cabal exec ghc -- -iapp -odir "$tmpdir" -hidir "$tmpdir" "$public_surface_src" -o "$public_surface_bin" >"$public_surface_log" 2>&1; then
  cat "$public_surface_log"
  exit 1
fi

"$public_surface_bin" >"$public_surface_log"
grep -q 'Trader.Trading public surface ok.' "$public_surface_log"

echo "Smoke checks passed."
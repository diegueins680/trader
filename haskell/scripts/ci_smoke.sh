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
cabal list-bin exe:optimize-equity >/dev/null

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

# Keep the smoke check aligned with the maintained Trader.Trading public surface
# without forcing the runtime shim.
public_surface_smoke="$tmpdir/trading-public-surface-smoke.hs"
cat >"$public_surface_smoke" <<'EOF'
module Main (main) where

import qualified Data.Vector as V
import Trader.Trading (
    BacktestResult,
    EnsembleConfig,
    ExitReason,
    StepMeta,
    Trade,
    simulateEnsembleVWithHLChecked
 )

publicSurfaceReachable ::
    ( EnsembleConfig ->
      Int ->
      V.Vector Double ->
      V.Vector Double ->
      V.Vector Double ->
      V.Vector Double ->
      V.Vector Double ->
      Maybe (V.Vector StepMeta) ->
      Either String BacktestResult
    ) ->
    Bool
publicSurfaceReachable entrypoint =
    case
        ( Nothing :: Maybe EnsembleConfig
        , Nothing :: Maybe ExitReason
        , Nothing :: Maybe Trade
        )
    of
        (Nothing, Nothing, Nothing) -> entrypoint `seq` True
        _ -> False

main :: IO ()
main
    | publicSurfaceReachable simulateEnsembleVWithHLChecked = pure ()
    | otherwise = error "Trader.Trading public surface witness unreachable"
EOF

cabal exec ghc -- -iapp -fno-code "$public_surface_smoke" >/dev/null

echo "Trader.Trading.simulateEnsembleVWithHLChecked: public surface shim"
echo "Smoke checks passed."
#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GENERATOR="${ROOT_DIR}/scripts/fetch-backtest-data.py"

if [[ -n "${VERIFY_MANIFEST:-}" ]]; then
  exec python3 "${GENERATOR}" --verify-manifest "${VERIFY_MANIFEST}"
fi

if [[ -z "${END_TIME_MS:-}" ]]; then
  printf 'END_TIME_MS is required; moving latest-window acquisition is not reproducible.\n' >&2
  exit 2
fi

read -r -a SYMBOL_ARGS <<< "${SYMBOLS:-BTCUSDT ETHUSDT SOLUSDT}"

ARGS=(
  --end-time-ms "${END_TIME_MS}"
  --data-dir "${DATA_DIR:-data}"
  --interval "${KLINE_INTERVAL:-4h}"
  --limit "${KLINE_LIMIT:-1000}"
  --base-url "${BINANCE_KLINES_URL:-https://api.binance.com/api/v3/klines}"
  --symbols "${SYMBOL_ARGS[@]}"
)

if [[ -n "${EXPECTED_MANIFEST:-}" ]]; then
  ARGS+=(--expected-manifest "${EXPECTED_MANIFEST}")
fi

exec python3 "${GENERATOR}" "${ARGS[@]}"

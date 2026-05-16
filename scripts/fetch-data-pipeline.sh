#!/bin/bash
# Repeatable data fetch pipeline for BTCUSDT/ETHUSDT/SOLUSDT 4h 1000 bars
set -euo pipefail
DATA_DIR="${DATA_DIR:-data}"
mkdir -p "$DATA_DIR"
for sym in BTCUSDT ETHUSDT SOLUSDT; do
  out="$DATA_DIR/${sym}-4h-1000.csv"
  tmp="$out.tmp.$$"
  # Fetch 1000 klines from Binance spot
  curl -s -H 'User-Agent: Mozilla/5.0' \
    "https://api.binance.com/api/v3/klines?symbol=${sym}&interval=4h&limit=1000" \
    | python3 -c '
import sys, json, csv
ks = json.load(sys.stdin)
w = csv.writer(sys.stdout)
w.writerow(["openTimeMs","open","high","low","close","volume","closeTimeMs","quoteAssetVolume","tradeCount","takerBuyBaseVolume","takerBuyQuoteVolume","ignore"])
for k in ks:
    w.writerow(k)
' > "$tmp"
  mv "$tmp" "$out"
  md5 "$out"
done

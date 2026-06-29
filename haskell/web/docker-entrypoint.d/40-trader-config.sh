#!/bin/sh
set -eu

config_path="${TRADER_CONFIG_PATH:-/usr/share/nginx/html/trader-config.js}"

json_escape() {
  awk '
    BEGIN { ORS = "" }
    {
      gsub(/\\/, "\\\\")
      gsub(/"/, "\\\"")
      gsub(/\r/, "\\r")
      gsub(/\t/, "\\t")
      if (NR > 1) {
        printf "\\n"
      }
      printf "%s", $0
    }
  '
}

json_string() {
  printf '%s' "$1" | json_escape
}

timeout_ms() {
  value="$1"
  fallback="$2"
  case "$value" in
    ""|*[!0-9]*)
      printf '%s' "$fallback"
      ;;
    *)
      if [ "$value" -lt 1000 ]; then
        printf '%s' "$fallback"
      else
        printf '%s' "$value"
      fi
      ;;
  esac
}

seconds_to_ms_min() {
  value="$1"
  min_ms="$2"
  case "$value" in
    ""|*[!0-9]*)
      printf '%s' "$min_ms"
      ;;
    *)
      computed=$((value * 1000))
      if [ "$computed" -lt "$min_ms" ]; then
        printf '%s' "$min_ms"
      else
        printf '%s' "$computed"
      fi
      ;;
  esac
}

api_base_url="${TRADER_UI_API_BASE_URL:-${TRADER_API_BASE_URL:-/api}}"
api_fallback_url="${TRADER_UI_API_FALLBACK_URL:-}"
api_token="${TRADER_UI_API_TOKEN:-${TRADER_API_TOKEN:-}}"
request_timeout_ms="$(timeout_ms "${TRADER_UI_REQUEST_TIMEOUT_MS:-}" 60000)"
signal_timeout_ms="$(timeout_ms "${TRADER_UI_SIGNAL_TIMEOUT_MS:-}" 1800000)"
backtest_default_ms="$(seconds_to_ms_min "${TRADER_API_BACKTEST_TIMEOUT_SEC:-}" 1800000)"
backtest_timeout_ms="$(timeout_ms "${TRADER_UI_BACKTEST_TIMEOUT_MS:-}" "$backtest_default_ms")"
trade_default_ms="$(seconds_to_ms_min "${TRADER_API_TRADE_TIMEOUT_SEC:-}" 1800000)"
trade_timeout_ms="$(timeout_ms "${TRADER_UI_TRADE_TIMEOUT_MS:-}" "$trade_default_ms")"
optimizer_timeout_ms="$(timeout_ms "${TRADER_UI_OPTIMIZER_TIMEOUT_MS:-}" 1800000)"
bot_start_timeout_ms="$(timeout_ms "${TRADER_UI_BOT_START_TIMEOUT_MS:-}" 1800000)"
bot_status_timeout_ms="$(timeout_ms "${TRADER_UI_BOT_STATUS_TIMEOUT_MS:-}" 120000)"
binance_trades_timeout_ms="$(timeout_ms "${TRADER_UI_BINANCE_TRADES_TIMEOUT_MS:-}" 180000)"

cat > "$config_path" <<EOF
globalThis.__TRADER_CONFIG__ = {
  apiBaseUrl: "$(json_string "$api_base_url")",
  apiBaseUrlInferred: false,
  apiFallbackUrl: "$(json_string "$api_fallback_url")",
  apiToken: "$(json_string "$api_token")",
  timeoutsMs: {
    requestMs: $request_timeout_ms,
    signalMs: $signal_timeout_ms,
    backtestMs: $backtest_timeout_ms,
    tradeMs: $trade_timeout_ms,
    optimizerMs: $optimizer_timeout_ms,
    botStartMs: $bot_start_timeout_ms,
    botStatusMs: $bot_status_timeout_ms,
    binanceTradesMs: $binance_trades_timeout_ms,
  },
};
EOF

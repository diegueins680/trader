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

api_base_url="${TRADER_UI_API_BASE_URL:-${TRADER_API_BASE_URL:-/api}}"
api_fallback_url="${TRADER_UI_API_FALLBACK_URL:-}"
api_token="${TRADER_UI_API_TOKEN:-${TRADER_API_TOKEN:-}}"

cat > "$config_path" <<EOF
globalThis.__TRADER_CONFIG__ = {
  apiBaseUrl: "$(json_string "$api_base_url")",
  apiBaseUrlInferred: false,
  apiFallbackUrl: "$(json_string "$api_fallback_url")",
  apiToken: "$(json_string "$api_token")",
  timeoutsMs: { requestMs: 60000, botStatusMs: 120000 },
};
EOF

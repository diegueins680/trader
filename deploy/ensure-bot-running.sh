#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

resolve_path() {
  local path="${1:-}"
  if [[ -z "$path" ]]; then
    return 1
  fi
  if [[ "$path" = /* ]]; then
    printf '%s' "$path"
  else
    printf '%s' "${repo_root}/${path}"
  fi
}

env_file_raw="${TRADER_ENV_FILE:-.env}"
env_file="$(resolve_path "$env_file_raw")"
if [[ -f "$env_file" ]]; then
  set -a
  # shellcheck source=/dev/null
  . "$env_file"
  set +a
fi

api_base="${TRADER_API_BASE_URL:-http://127.0.0.1:8080}"
api_base="${api_base%/}"
status_timeout="${TRADER_BOT_STATUS_TIMEOUT_SEC:-5}"
start_timeout="${TRADER_BOT_START_TIMEOUT_SEC:-10}"
python_bin="${TRADER_BOT_PYTHON_BIN:-python3}"

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "bot-watch: python3 is required (set TRADER_BOT_PYTHON_BIN if needed)" >&2
  exit 1
fi

curl_headers=()
if [[ -n "${TRADER_API_TOKEN:-}" ]]; then
  curl_headers+=(-H "Authorization: Bearer ${TRADER_API_TOKEN}")
fi

fetch_status() {
  local url="$1"
  local resp
  if ! resp=$(curl -fsS --max-time "$status_timeout" "${curl_headers[@]}" "$url"); then
    echo "bot-watch: failed to reach $url" >&2
    return 1
  fi
  printf '%s' "$resp"
}

status_state() {
  local resp="$1"
  local state
  if ! state=$(printf '%s' "$resp" | "$python_bin" -c 'import json,sys
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(1)

running = False
if isinstance(data, dict):
    running = bool(data.get("running")) or bool(data.get("starting"))

print("running" if running else "stopped")
'); then
    return 2
  fi

  if [[ "$state" == "running" ]]; then
    return 0
  fi

  return 1
}

status_symbols_running() {
  local resp="$1"
  local symbols_csv="$2"
  if printf '%s' "$resp" | "$python_bin" -c 'import json,sys
def norm(val):
    return val.strip().upper()

raw = sys.argv[1] if len(sys.argv) > 1 else ""
desired = []
for part in raw.split(","):
    sym = norm(part)
    if sym and sym not in desired:
        desired.append(sym)

try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(2)

def is_running(obj):
    if not isinstance(obj, dict):
        return False
    return bool(obj.get("running")) or bool(obj.get("starting"))

def symbol_from(obj):
    if not isinstance(obj, dict):
        return None
    sym = obj.get("symbol")
    if isinstance(sym, str):
        sym = norm(sym)
        if sym:
            return sym
    snap = obj.get("snapshot")
    if isinstance(snap, dict):
        sym = snap.get("symbol")
        if isinstance(sym, str):
            sym = norm(sym)
            if sym:
                return sym
    return None

statuses = {}
def add_status(obj):
    sym = symbol_from(obj)
    if not sym:
        return
    statuses[sym] = statuses.get(sym, False) or is_running(obj)

if isinstance(data, dict) and isinstance(data.get("bots"), list):
    for bot in data["bots"]:
        add_status(bot)
else:
    add_status(data)

if not desired:
    sys.exit(1)

missing = [sym for sym in desired if not statuses.get(sym)]
sys.exit(1 if missing else 0)
' "$symbols_csv"; then
    return 0
  fi
  local rc=$?
  if [[ $rc -eq 2 ]]; then
    return 2
  fi
  return 1
}

normalize_trade_flag() {
  if [[ -z "${TRADER_BOT_TRADE:-}" ]]; then
    return 1
  fi
  local trade_lc
  trade_lc="$(printf '%s' "${TRADER_BOT_TRADE}" | tr '[:upper:]' '[:lower:]')"
  case "$trade_lc" in
    true|false) printf '%s' "$trade_lc" ;;
    1) printf 'true' ;;
    0) printf 'false' ;;
    *)
      echo "bot-watch: TRADER_BOT_TRADE must be true/false/1/0" >&2
      return 1
      ;;
  esac
}

normalize_symbols_csv() {
  local symbols_csv="$1"
  local -a symbols=()
  IFS=',' read -r -a raw_symbols <<<"$symbols_csv"
  for raw in "${raw_symbols[@]}"; do
    local sym="${raw//[[:space:]]/}"
    if [[ -n "$sym" ]]; then
      symbols+=("$sym")
    fi
  done
  if [[ ${#symbols[@]} -eq 0 ]]; then
    return 1
  fi
  local joined
  joined=$(IFS=,; echo "${symbols[*]}")
  printf '%s' "$joined"
}

build_symbol_body() {
  local symbol_raw="$1"
  local symbol="${symbol_raw//[[:space:]]/}"
  if [[ -z "$symbol" ]]; then
    return 1
  fi
  local trade_flag=""
  if trade_flag=$(normalize_trade_flag); then
    printf '{"binanceSymbol":"%s","botTrade":%s}' "$symbol" "$trade_flag"
  else
    if [[ -n "${TRADER_BOT_TRADE:-}" ]]; then
      return 1
    fi
    printf '{"binanceSymbol":"%s"}' "$symbol"
  fi
}

build_symbols_body() {
  local symbols_csv="$1"
  local -a symbols=()
  local first_symbol=""
  IFS=',' read -r -a raw_symbols <<<"$symbols_csv"
  for raw in "${raw_symbols[@]}"; do
    local sym="${raw//[[:space:]]/}"
    if [[ -n "$sym" ]]; then
      if [[ -z "$first_symbol" ]]; then
        first_symbol="$sym"
      fi
      symbols+=("\"$sym\"")
    fi
  done
  if [[ ${#symbols[@]} -eq 0 ]]; then
    return 1
  fi
  if [[ -z "$first_symbol" ]]; then
    return 1
  fi
  local trade_flag=""
  local joined
  joined=$(IFS=,; echo "${symbols[*]}")
  if trade_flag=$(normalize_trade_flag); then
    printf '{"binanceSymbol":"%s","botSymbols":[%s],"botTrade":%s}' "$first_symbol" "$joined" "$trade_flag"
  else
    if [[ -n "${TRADER_BOT_TRADE:-}" ]]; then
      return 1
    fi
    printf '{"binanceSymbol":"%s","botSymbols":[%s]}' "$first_symbol" "$joined"
  fi
}

read_start_payload_optional() {
  if [[ -n "${TRADER_BOT_START_BODY:-}" ]]; then
    printf '%s' "$TRADER_BOT_START_BODY"
    return 0
  fi

  if [[ -n "${TRADER_BOT_START_FILE:-}" ]]; then
    local start_file
    start_file="$(resolve_path "$TRADER_BOT_START_FILE")"
    if [[ -f "$start_file" ]]; then
      cat "$start_file"
      return 0
    fi
    return 1
  fi

  local default_start_file="${repo_root}/bot-start.json"
  if [[ -f "$default_start_file" ]]; then
    cat "$default_start_file"
    return 0
  fi

  return 1
}

extract_symbols_from_body() {
  local body="$1"
  local symbols
  if ! symbols=$(printf '%s' "$body" | "$python_bin" -c 'import json,sys
def norm(val):
    return val.strip().upper()

try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(1)

symbols = []
if isinstance(data, dict):
    raw = data.get("botSymbols")
    if isinstance(raw, str):
        for part in raw.split(","):
            sym = norm(part)
            if sym:
                symbols.append(sym)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                sym = norm(item)
                if sym:
                    symbols.append(sym)

    sym = data.get("binanceSymbol")
    if isinstance(sym, str):
        sym = norm(sym)
        if sym:
            symbols.append(sym)

deduped = []
for sym in symbols:
    if sym not in deduped:
        deduped.append(sym)

print(",".join(deduped))
'); then
    return 1
  fi
  printf '%s' "$symbols"
}

resolve_symbols_csv() {
  local symbols=""
  local payload=""
  if [[ -n "${TRADER_BOT_SYMBOLS:-}" ]]; then
    if symbols=$(normalize_symbols_csv "${TRADER_BOT_SYMBOLS}"); then
      printf '%s' "$symbols"
      return 0
    else
      echo "bot-watch: TRADER_BOT_SYMBOLS is set but empty after trimming; checking start payload" >&2
    fi
  elif [[ -n "${TRADER_BOT_SYMBOL:-}" ]]; then
    if symbols=$(normalize_symbols_csv "${TRADER_BOT_SYMBOL}"); then
      printf '%s' "$symbols"
      return 0
    else
      echo "bot-watch: TRADER_BOT_SYMBOL is set but empty after trimming; checking start payload" >&2
    fi
  fi

  if payload=$(read_start_payload_optional); then
    if symbols=$(extract_symbols_from_body "$payload"); then
      if [[ -n "$symbols" ]]; then
        printf '%s' "$symbols"
        return 0
      fi
    else
      echo "bot-watch: start payload is not valid JSON; checking global status" >&2
    fi
    return 1
  fi

  return 1
}

resolve_start_body() {
  if [[ -n "${TRADER_BOT_START_BODY:-}" ]]; then
    printf '%s' "$TRADER_BOT_START_BODY"
    return 0
  fi

  if [[ -n "${TRADER_BOT_START_FILE:-}" ]]; then
    local start_file
    start_file="$(resolve_path "$TRADER_BOT_START_FILE")"
    if [[ ! -f "$start_file" ]]; then
      echo "bot-watch: start file not found: $start_file" >&2
      return 1
    fi
    cat "$start_file"
    return 0
  fi

  local default_start_file="${repo_root}/bot-start.json"
  if [[ -f "$default_start_file" ]]; then
    cat "$default_start_file"
    return 0
  fi

  if [[ -n "${TRADER_BOT_SYMBOLS:-}" ]]; then
    build_symbols_body "${TRADER_BOT_SYMBOLS}"
    return $?
  fi

  if [[ -n "${TRADER_BOT_SYMBOL:-}" ]]; then
    build_symbol_body "${TRADER_BOT_SYMBOL}"
    return $?
  fi

  return 1
}

needs_start=0
symbols_csv=""
if symbols_csv=$(resolve_symbols_csv); then
  :
fi

resp=$(fetch_status "${api_base}/bot/status") || exit 1
if [[ -n "$symbols_csv" ]]; then
  if status_symbols_running "$resp" "$symbols_csv"; then
    :
  else
    rc=$?
    if [[ $rc -eq 2 ]]; then
      echo "bot-watch: invalid status response" >&2
      exit 1
    fi
    needs_start=1
  fi
else
  if status_state "$resp"; then
    :
  else
    rc=$?
    if [[ $rc -eq 2 ]]; then
      echo "bot-watch: invalid status response" >&2
      exit 1
    fi
    needs_start=1
  fi
fi

if [[ "$needs_start" -eq 0 ]]; then
  exit 0
fi

start_body="$(resolve_start_body)" || {
  echo "bot-watch: missing start payload (set TRADER_BOT_START_BODY, TRADER_BOT_START_FILE, bot-start.json, or TRADER_BOT_SYMBOLS/TRADER_BOT_SYMBOL)" >&2
  exit 1
}

if ! curl -fsS --max-time "$start_timeout" -H "Content-Type: application/json" "${curl_headers[@]}" -X POST -d "$start_body" "${api_base}/bot/start" >/dev/null; then
  echo "bot-watch: failed to start bot via ${api_base}/bot/start" >&2
  exit 1
fi

echo "bot-watch: start requested"

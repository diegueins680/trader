#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HASKELL_DIR="$ROOT_DIR/haskell"

TOP_JSON="${TOP_JSON:-"$ROOT_DIR/top-combos.s3.json"}"
if [[ ! -f "$TOP_JSON" ]]; then
  TOP_JSON="$ROOT_DIR/haskell/web/public/top-combos.json"
fi

COUNT="${COUNT:-5}"
TRIALS="${TRIALS:-100}"
BARS="${BARS:-1000}"
TIMEOUT_SEC="${TIMEOUT_SEC:-60}"
PLATFORM="${PLATFORM:-binance}"
FUTURES="${FUTURES:-1}"
QUALITY="${QUALITY:-0}"
COMPARE="${COMPARE:-0}"
BASE_REF="${BASE_REF:-HEAD~1}"
OUT_ROOT="${OUT_ROOT:-"$ROOT_DIR/.tmp/opt-eq-top5-$(date +%Y%m%d-%H%M%S)"}"

mkdir -p "$OUT_ROOT"

COMBOS=()
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  COMBOS+=("$line")
done < <(
  python3 - <<'PY' "$TOP_JSON" "$COUNT"
import json, sys

path = sys.argv[1]
count = int(sys.argv[2])

with open(path) as f:
    data = json.load(f)

combos = []
for c in data.get("combos", []):
    params = c.get("params") or {}
    sym = params.get("binanceSymbol") or params.get("symbol") or params.get("binance_symbol")
    interval = params.get("interval")
    if sym and interval:
        combos.append((sym, interval))
    if len(combos) >= count:
        break

for sym, interval in combos:
    print(sym, interval)
PY
)

if [[ ${#COMBOS[@]} -eq 0 ]]; then
  echo "No combos found in $TOP_JSON"
  exit 1
fi

run_set() {
  local label="$1"
  local workdir="$2"
  local out_dir="$OUT_ROOT/$label"

  mkdir -p "$out_dir"
  (cd "$workdir/haskell" && cabal build optimize-equity trader-hs >/dev/null)

  local bin
  bin="$(cd "$workdir/haskell" && cabal list-bin optimize-equity)"

  for line in "${COMBOS[@]}"; do
    local sym interval safe log json
    sym="$(awk '{print $1}' <<<"$line")"
    interval="$(awk '{print $2}' <<<"$line")"
    safe="${sym}-${interval}"
    safe="${safe//\//_}"
    log="$out_dir/$safe.log"
    json="$out_dir/$safe.json"

    local cmd=("$bin"
      --symbol "$sym"
      --intervals "$interval"
      --platform "$PLATFORM"
      --bars-min "$BARS"
      --bars-max "$BARS"
      --bars-auto-prob 0
      --trials "$TRIALS"
      --timeout-sec "$TIMEOUT_SEC"
      --top-json "$json"
    )

    if [[ "$FUTURES" == "1" ]]; then
      cmd+=(--futures)
    fi
    if [[ "$QUALITY" == "1" ]]; then
      cmd+=(--quality)
    fi

    echo "Running $label: $sym $interval (trials=$TRIALS bars=$BARS)" | tee "$log"
    "${cmd[@]}" >>"$log" 2>&1
  done

  python3 - <<'PY' "$out_dir"
import json, os, sys

out_dir = sys.argv[1]
summary = []
for fn in sorted(os.listdir(out_dir)):
    if not fn.endswith(".json"):
        continue
    path = os.path.join(out_dir, fn)
    try:
        data = json.load(open(path))
        combos = data.get("combos") or []
        top = combos[0] if combos else {}
        metrics = top.get("metrics") or {}
        summary.append({
            "file": fn,
            "finalEquity": top.get("finalEquity"),
            "annualizedReturn": metrics.get("annualizedReturn"),
        })
    except Exception as e:
        summary.append({"file": fn, "error": str(e)})

with open(os.path.join(out_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("Wrote", os.path.join(out_dir, "summary.json"))
PY
}

run_set "current" "$ROOT_DIR"

if [[ "$COMPARE" == "1" ]]; then
  wt_dir="$(mktemp -d /tmp/trader-prev-XXXXXX)"
  git -C "$ROOT_DIR" worktree add "$wt_dir" "$BASE_REF"
  run_set "baseline" "$wt_dir"
  git -C "$ROOT_DIR" worktree remove "$wt_dir" --force
fi

echo "Done. Outputs in $OUT_ROOT"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HASKELL_DIR="$ROOT_DIR/haskell"

TOP_JSON="${TOP_JSON:-"$ROOT_DIR/top-combos.s3.json"}"
if [[ ! -f "$TOP_JSON" ]]; then
  TOP_JSON="$ROOT_DIR/haskell/web/public/top-combos.json"
fi

COUNT="${COUNT:-5}"
TRIALS="${TRIALS:-300}"
BARS="${BARS:-1000}"
ACTIVITY_RETRY_BARS="${ACTIVITY_RETRY_BARS:-700,1000}"
TIMEOUT_SEC="${TIMEOUT_SEC:-60}"
LOOKBACK_WINDOW="${LOOKBACK_WINDOW:-7d}"
TRADER_OPTIMIZER_MAX_POINTS="${TRADER_OPTIMIZER_MAX_POINTS:-1000}"
export TRADER_OPTIMIZER_MAX_POINTS
PLATFORM="${PLATFORM:-binance}"
FUTURES="${FUTURES:-1}"
QUALITY="${QUALITY:-1}"
QUALITY_MIN_TRIALS="${QUALITY_MIN_TRIALS:-$TRIALS}"
QUALITY_MAX_EPOCHS="${QUALITY_MAX_EPOCHS:-50}"
QUALITY_MAX_HIDDEN_SIZE="${QUALITY_MAX_HIDDEN_SIZE:-128}"
NOOPT="${NOOPT:-0}"
MIN_ROUND_TRIPS="${MIN_ROUND_TRIPS:-20}"
MIN_EXPOSURE="${MIN_EXPOSURE:-0.10}"
MIN_SHARPE="${MIN_SHARPE:-1.0}"
MIN_CALMAR="${MIN_CALMAR:-0.8}"
MIN_WF_SHARPE_MEAN="${MIN_WF_SHARPE_MEAN:-0.8}"
MAX_WF_SHARPE_STD="${MAX_WF_SHARPE_STD:-1.0}"
MIN_KELLY_LITE_EXPOSURE_REDUCTION="${MIN_KELLY_LITE_EXPOSURE_REDUCTION:-0.0}"
MAX_KELLY_LITE_EXPOSURE_RATIO="${MAX_KELLY_LITE_EXPOSURE_RATIO:-0.95}"
P_KELLY_LITE_SIZING="${P_KELLY_LITE_SIZING:-0.0}"
KELLY_LITE_FRACTION_MIN="${KELLY_LITE_FRACTION_MIN:-0.5}"
KELLY_LITE_FRACTION_MAX="${KELLY_LITE_FRACTION_MAX:-0.5}"
KELLY_LITE_FLOOR_MIN="${KELLY_LITE_FLOOR_MIN:-0.0}"
KELLY_LITE_FLOOR_MAX="${KELLY_LITE_FLOOR_MAX:-0.0}"
KELLY_LITE_CAP_MIN="${KELLY_LITE_CAP_MIN:-1.0}"
KELLY_LITE_CAP_MAX="${KELLY_LITE_CAP_MAX:-1.0}"
TUNE_STRESS_VOL_MULT="${TUNE_STRESS_VOL_MULT:-1.25}"
TUNE_STRESS_SHOCK="${TUNE_STRESS_SHOCK:-0.0}"
TUNE_STRESS_WEIGHT="${TUNE_STRESS_WEIGHT:-0.2}"
WF_EMBARGO_MIN="${WF_EMBARGO_MIN:-1}"
WF_EMBARGO_MAX="${WF_EMBARGO_MAX:-3}"
METHOD_WEIGHT_REGIME_SWITCH="${METHOD_WEIGHT_REGIME_SWITCH:-1.0}"
METHOD_WEIGHT_BANDIT_ROUTER="${METHOD_WEIGHT_BANDIT_ROUTER:-0.0}"
ROUTER_SCORE_PNL_WEIGHT_MIN="${ROUTER_SCORE_PNL_WEIGHT_MIN:-0.25}"
ROUTER_SCORE_PNL_WEIGHT_MAX="${ROUTER_SCORE_PNL_WEIGHT_MAX:-0.85}"
ROUTER_LOOKBACK_MIN="${ROUTER_LOOKBACK_MIN:-20}"
ROUTER_LOOKBACK_MAX="${ROUTER_LOOKBACK_MAX:-180}"
ROUTER_MIN_SCORE_MIN="${ROUTER_MIN_SCORE_MIN:-0.05}"
ROUTER_MIN_SCORE_MAX="${ROUTER_MIN_SCORE_MAX:-0.7}"
P_CONFIDENCE_SIZING="${P_CONFIDENCE_SIZING:-0.85}"
P_COST_AWARE_EDGE="${P_COST_AWARE_EDGE:-1.0}"
P_DISABLE_RISK_PER_TRADE="${P_DISABLE_RISK_PER_TRADE:-0.3}"
STOP_VOL_MULT_MIN="${STOP_VOL_MULT_MIN:-0.8}"
STOP_VOL_MULT_MAX="${STOP_VOL_MULT_MAX:-3.0}"
TP_VOL_MULT_MIN="${TP_VOL_MULT_MIN:-1.2}"
TP_VOL_MULT_MAX="${TP_VOL_MULT_MAX:-4.0}"
TRAIL_VOL_MULT_MIN="${TRAIL_VOL_MULT_MIN:-0.8}"
TRAIL_VOL_MULT_MAX="${TRAIL_VOL_MULT_MAX:-3.0}"
P_DISABLE_STOP_VOL_MULT="${P_DISABLE_STOP_VOL_MULT:-0.35}"
P_DISABLE_TP_VOL_MULT="${P_DISABLE_TP_VOL_MULT:-0.35}"
P_DISABLE_TRAIL_VOL_MULT="${P_DISABLE_TRAIL_VOL_MULT:-0.5}"
COMPARE="${COMPARE:-0}"
BASE_REF="${BASE_REF:-HEAD~1}"
OUT_ROOT="${OUT_ROOT:-"$ROOT_DIR/.tmp/opt-eq-top5-$(date +%Y%m%d-%H%M%S)"}"

mkdir -p "$OUT_ROOT"
RUN_LOG="$OUT_ROOT/run.log"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S%z')" "$*" | tee -a "$RUN_LOG"
}

on_exit() {
  local status=$?
  set +e
  log "Exit status=$status"
}

on_signal() {
  local sig="$1"
  local code="$2"
  set +e
  log "Received signal=$sig"
  exit "$code"
}

trap on_exit EXIT
trap 'on_signal HUP 129' HUP
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM

COMBOS=()
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  COMBOS+=("$line")
done < <(
  python3 - <<'PY' "$TOP_JSON" "$COUNT" "$MIN_ROUND_TRIPS" "$MIN_EXPOSURE" "$MIN_SHARPE" "$MIN_CALMAR" "$LOOKBACK_WINDOW" "$TRADER_OPTIMIZER_MAX_POINTS"
import json, sys

path = sys.argv[1]
count = int(sys.argv[2])
min_round_trips = float(sys.argv[3])
min_exposure = float(sys.argv[4])
min_sharpe = float(sys.argv[5])
min_calmar = float(sys.argv[6])
lookback_window = sys.argv[7]
try:
    max_points = min(5000, max(2, int(sys.argv[8])))
except Exception:
    max_points = 1000


def to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def parse_duration_seconds(raw):
    raw = str(raw).strip()
    if len(raw) < 2 or not raw[:-1].isdigit():
        return None
    unit_seconds = {
        "s": 1,
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60,
        "M": 30 * 24 * 60 * 60,
    }
    mult = unit_seconds.get(raw[-1])
    if mult is None:
        return None
    return int(raw[:-1]) * mult


def feasible_interval(interval):
    interval_sec = parse_duration_seconds(interval)
    lookback_sec = parse_duration_seconds(lookback_window)
    if not interval_sec or not lookback_sec:
        return False
    lookback_bars = (lookback_sec + interval_sec - 1) // interval_sec
    return lookback_bars >= 2 and lookback_bars + 3 <= max_points

with open(path) as f:
    data = json.load(f)

eligible = []
fallback = []
seen = set()
for c in data.get("combos", []):
    params = c.get("params") or {}
    metrics = c.get("metrics") or {}
    sym = params.get("binanceSymbol") or params.get("symbol") or params.get("binance_symbol")
    interval = params.get("interval")
    if not sym or not interval:
        continue
    if not feasible_interval(interval):
        continue
    key = (sym, interval)
    if key in seen:
        continue
    seen.add(key)

    annualized_return = to_float(metrics.get("annualizedReturn"))
    round_trips = to_float(metrics.get("roundTrips"))
    exposure = to_float(metrics.get("exposure"))
    sharpe = to_float(metrics.get("sharpe"))
    max_dd = to_float(metrics.get("maxDrawdown"))
    calmar_raw = metrics.get("calmar")
    if calmar_raw is None:
        calmar = annualized_return / max_dd if max_dd > 0 else annualized_return
    else:
        calmar = to_float(calmar_raw)
    score = (annualized_return, sharpe, exposure, round_trips)
    fallback.append((score, sym, interval))

    if (
        round_trips >= min_round_trips
        and exposure >= min_exposure
        and sharpe >= min_sharpe
        and calmar >= min_calmar
    ):
        eligible.append((score, sym, interval))

eligible.sort(reverse=True)
fallback.sort(reverse=True)

selected = []
selected_keys = set()
for _, sym, interval in eligible + fallback:
    key = (sym, interval)
    if key in selected_keys:
        continue
    selected.append((sym, interval))
    selected_keys.add(key)
    if len(selected) >= count:
        break

for sym, interval in selected:
    print(sym, interval)
PY
)

if [[ ${#COMBOS[@]} -eq 0 ]]; then
  log "No combos found in $TOP_JSON for lookback_window=$LOOKBACK_WINDOW max_points=$TRADER_OPTIMIZER_MAX_POINTS"
  exit 1
fi

log "Run start out=$OUT_ROOT top_json=$TOP_JSON count=$COUNT trials=$TRIALS bars=$BARS activity_retry_bars=$ACTIVITY_RETRY_BARS timeout=$TIMEOUT_SEC lookback_window=$LOOKBACK_WINDOW max_points=$TRADER_OPTIMIZER_MAX_POINTS platform=$PLATFORM futures=$FUTURES quality=$QUALITY quality_min_trials=$QUALITY_MIN_TRIALS quality_max_epochs=$QUALITY_MAX_EPOCHS quality_max_hidden_size=$QUALITY_MAX_HIDDEN_SIZE noopt=$NOOPT compare=$COMPARE min_round_trips=$MIN_ROUND_TRIPS min_exposure=$MIN_EXPOSURE min_sharpe=$MIN_SHARPE min_calmar=$MIN_CALMAR p_kelly_lite_sizing=$P_KELLY_LITE_SIZING min_kelly_lite_exposure_reduction=$MIN_KELLY_LITE_EXPOSURE_REDUCTION max_kelly_lite_exposure_ratio=$MAX_KELLY_LITE_EXPOSURE_RATIO"

should_retry_activity_bars() {
  local log_file="$1"
  python3 - "$log_file" <<'PY'
import re
import sys

path = sys.argv[1]
try:
    text = open(path, encoding="utf-8").read()
except OSError:
    sys.exit(1)

if "No eligible trials." not in text:
    sys.exit(1)

trial_lines = [line for line in text.splitlines() if re.match(r"\[\s*\d+/\d+\]", line)]
if not trial_lines:
    sys.exit(1)

for line in trial_lines:
    match = re.search(r"\(filter: ([^)]+)\)", line)
    if not match or not match.group(1).startswith("activityCount<"):
        sys.exit(1)

sys.exit(0)
PY
}

run_set() {
  local label="$1"
  local workdir="$2"
  local out_dir="$OUT_ROOT/$label"

  mkdir -p "$out_dir"

  local bin trader_bin
  if [[ "$NOOPT" == "1" ]]; then
    (cd "$workdir/haskell" && cabal build optimize-equity trader-hs --disable-optimization >/dev/null)
    bin="$(cd "$workdir/haskell" && cabal list-bin optimize-equity --disable-optimization)"
    trader_bin="$(cd "$workdir/haskell" && cabal list-bin trader-hs --disable-optimization)"
  else
    (cd "$workdir/haskell" && cabal build optimize-equity trader-hs >/dev/null)
    bin="$(cd "$workdir/haskell" && cabal list-bin optimize-equity)"
    trader_bin="$(cd "$workdir/haskell" && cabal list-bin trader-hs)"
  fi

  for line in "${COMBOS[@]}"; do
    local sym interval safe
    sym="$(awk '{print $1}' <<<"$line")"
    interval="$(awk '{print $2}' <<<"$line")"
    safe="${sym}-${interval}"
    safe="${safe//\//_}"

    local bars_attempt="$BARS"
    local attempt_index=0
    while :; do
      local suffix log json
      suffix=""
      if ((attempt_index > 0)); then
        suffix="-bars${bars_attempt}"
      fi
      log="$out_dir/$safe$suffix.log"
      json="$out_dir/$safe$suffix.json"

      local cmd=("$bin"
        --symbol "$sym"
        --intervals "$interval"
        --platform "$PLATFORM"
        --lookback-window "$LOOKBACK_WINDOW"
        --bars-min "$bars_attempt"
        --bars-max "$bars_attempt"
        --bars-auto-prob 0
        --trials "$TRIALS"
        --timeout-sec "$TIMEOUT_SEC"
        --binary "$trader_bin"
        --min-round-trips "$MIN_ROUND_TRIPS"
        --min-exposure "$MIN_EXPOSURE"
        --min-sharpe "$MIN_SHARPE"
        --min-calmar "$MIN_CALMAR"
        --min-wf-sharpe-mean "$MIN_WF_SHARPE_MEAN"
        --max-wf-sharpe-std "$MAX_WF_SHARPE_STD"
        --min-kelly-lite-exposure-reduction "$MIN_KELLY_LITE_EXPOSURE_REDUCTION"
        --max-kelly-lite-exposure-ratio "$MAX_KELLY_LITE_EXPOSURE_RATIO"
        --p-kelly-lite-sizing "$P_KELLY_LITE_SIZING"
        --kelly-lite-fraction-min "$KELLY_LITE_FRACTION_MIN"
        --kelly-lite-fraction-max "$KELLY_LITE_FRACTION_MAX"
        --kelly-lite-floor-min "$KELLY_LITE_FLOOR_MIN"
        --kelly-lite-floor-max "$KELLY_LITE_FLOOR_MAX"
        --kelly-lite-cap-min "$KELLY_LITE_CAP_MIN"
        --kelly-lite-cap-max "$KELLY_LITE_CAP_MAX"
        --tune-stress-vol-mult "$TUNE_STRESS_VOL_MULT"
        --tune-stress-shock "$TUNE_STRESS_SHOCK"
        --tune-stress-weight "$TUNE_STRESS_WEIGHT"
        --walk-forward-embargo-bars-min "$WF_EMBARGO_MIN"
        --walk-forward-embargo-bars-max "$WF_EMBARGO_MAX"
        --method-weight-regime-switch "$METHOD_WEIGHT_REGIME_SWITCH"
        --method-weight-bandit-router "$METHOD_WEIGHT_BANDIT_ROUTER"
        --router-score-pnl-weight-min "$ROUTER_SCORE_PNL_WEIGHT_MIN"
        --router-score-pnl-weight-max "$ROUTER_SCORE_PNL_WEIGHT_MAX"
        --router-lookback-min "$ROUTER_LOOKBACK_MIN"
        --router-lookback-max "$ROUTER_LOOKBACK_MAX"
        --router-min-score-min "$ROUTER_MIN_SCORE_MIN"
        --router-min-score-max "$ROUTER_MIN_SCORE_MAX"
        --p-confidence-sizing "$P_CONFIDENCE_SIZING"
        --p-cost-aware-edge "$P_COST_AWARE_EDGE"
        --p-disable-risk-per-trade "$P_DISABLE_RISK_PER_TRADE"
        --stop-vol-mult-min "$STOP_VOL_MULT_MIN"
        --stop-vol-mult-max "$STOP_VOL_MULT_MAX"
        --tp-vol-mult-min "$TP_VOL_MULT_MIN"
        --tp-vol-mult-max "$TP_VOL_MULT_MAX"
        --trail-vol-mult-min "$TRAIL_VOL_MULT_MIN"
        --trail-vol-mult-max "$TRAIL_VOL_MULT_MAX"
        --p-disable-stop-vol-mult "$P_DISABLE_STOP_VOL_MULT"
        --p-disable-tp-vol-mult "$P_DISABLE_TP_VOL_MULT"
        --p-disable-trail-vol-mult "$P_DISABLE_TRAIL_VOL_MULT"
        --top-json "$json"
      )

      if [[ "$FUTURES" == "1" ]]; then
        cmd+=(--futures)
      fi
      if [[ "$QUALITY" == "1" ]]; then
        cmd+=(
          --quality
          --quality-min-trials "$QUALITY_MIN_TRIALS"
          --quality-max-epochs "$QUALITY_MAX_EPOCHS"
          --quality-max-hidden-size "$QUALITY_MAX_HIDDEN_SIZE"
        )
      fi

      log "Running $label: $sym $interval (trials=$TRIALS bars=$bars_attempt)"
      echo "Running $label: $sym $interval (trials=$TRIALS bars=$bars_attempt)" | tee "$log"
      if "${cmd[@]}" >>"$log" 2>&1; then
        break
      fi

      log "Run failed for $sym $interval at bars=$bars_attempt; checking retry eligibility."
      echo "Run failed for $sym $interval at bars=$bars_attempt; checking retry eligibility." | tee -a "$log"

      if ! should_retry_activity_bars "$log"; then
        log "Run failed for $sym $interval; continuing."
        echo "Run failed for $sym $interval; continuing." | tee -a "$log"
        break
      fi

      local next_bars=""
      local retry_bars
      for retry_bars in ${ACTIVITY_RETRY_BARS//,/ }; do
        if [[ "$retry_bars" =~ ^[0-9]+$ ]] && ((retry_bars > bars_attempt)); then
          next_bars="$retry_bars"
          break
        fi
      done

      if [[ -z "$next_bars" ]]; then
        log "Run failed for $sym $interval after activityCount-only skips; no deeper retry bars remain."
        echo "Run failed for $sym $interval after activityCount-only skips; no deeper retry bars remain." | tee -a "$log"
        break
      fi

      log "Retrying $label: $sym $interval with bars=$next_bars after activityCount-only skips at bars=$bars_attempt."
      bars_attempt="$next_bars"
      attempt_index=$((attempt_index + 1))
    done
  done

  python3 - <<'PY' "$out_dir"
import json, os, re, sys

out_dir = sys.argv[1]
summary = []
json_files = set()
for fn in sorted(os.listdir(out_dir)):
    if not fn.endswith(".json"):
        continue
    json_files.add(fn)
    path = os.path.join(out_dir, fn)
    try:
        data = json.load(open(path))
        combos = data.get("combos") or []
        top = combos[0] if combos else {}
        metrics = top.get("metrics") or {}
        summary.append({
            "file": fn,
            "status": "ok",
            "finalEquity": top.get("finalEquity"),
            "annualizedReturn": metrics.get("annualizedReturn"),
        })
    except Exception as e:
        summary.append({"file": fn, "status": "error", "error": str(e)})

for fn in sorted(os.listdir(out_dir)):
    if not fn.endswith(".log") or fn[:-4] + ".json" in json_files:
        continue
    path = os.path.join(out_dir, fn)
    try:
        text = open(path, encoding="utf-8").read()
    except Exception as e:
        summary.append({"file": fn, "status": "error", "error": str(e)})
        continue

    bars = None
    trial_count = 0
    filter_reasons = {}
    for line in text.splitlines():
        run_match = re.search(r"\(trials=\d+ bars=(\d+)\)", line)
        if run_match:
            bars = int(run_match.group(1))
        filter_match = re.match(r"\[\s*\d+/\d+\].*\(filter: ([^)]+)\)", line)
        if filter_match:
            trial_count += 1
            reason = filter_match.group(1)
            filter_reasons[reason] = filter_reasons.get(reason, 0) + 1

    status = "noEligibleTrials" if "No eligible trials." in text else "failed"
    summary.append({
        "file": fn,
        "status": status,
        "bars": bars,
        "trialCount": trial_count,
        "filterReasons": filter_reasons,
    })

with open(os.path.join(out_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("Wrote", os.path.join(out_dir, "summary.json"))
PY
  log "Finished $label"
}

run_set "current" "$ROOT_DIR"

if [[ "$COMPARE" == "1" ]]; then
  wt_dir="$(mktemp -d /tmp/trader-prev-XXXXXX)"
  git -C "$ROOT_DIR" worktree add "$wt_dir" "$BASE_REF"
  run_set "baseline" "$wt_dir"
  git -C "$ROOT_DIR" worktree remove "$wt_dir" --force
fi

log "Done. Outputs in $OUT_ROOT"

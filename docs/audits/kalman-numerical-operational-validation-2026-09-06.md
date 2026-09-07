# Kalman numerical operational validation

Date: 2026-09-06 local / 2026-09-07 UTC

Risk: `KALMAN-NUMSTAB-001`

Disposition: closed after current deterministic and CLI liveness evidence

## Historical claim and implemented controls

The risk originated from reports that Kalman-only runs produced no trades and
might hang. Commit `8d41af11` raised the initial Kalman3 covariance and added a
configurable standard-deviation floor to z-score gating. Commit `9b57cf74`
subsequently made scalar updates ignore non-finite observations or invalid
measurement variance, reject invalid innovation variance, use the Joseph
covariance update, and retain the prior if a candidate state becomes
non-finite.

Current CLI validation also requires finite, positive time-step, process-noise,
and measurement-noise parameters and a finite, non-negative z-score standard
deviation floor. The formal optimization model already exhaustively checks its
bounded Kalman-fusion domain for finite positive posterior variance, malformed
measurement isolation, and non-increasing uncertainty from valid scalar
evidence.

## Long-run deterministic regression

`testKalmanLongRunNumericalStability` advances both implemented filter families
for 20,000 steps:

- Kalman3 receives a bounded trend-plus-cycle series with periodic `NaN` and
  positive-infinity observations. Every state vector, covariance element,
  covariance diagonal, predicted/filtered runner value, and terminal forecast
  must remain finite; covariance diagonals must remain non-negative.
- Kalman fusion receives two ordinary observations per valid step and periodic
  all-malformed observation sets. Every posterior mean, variance, and process
  variance must remain finite, and posterior variance must stay positive.

Completion is checked by the ordinary repository test process rather than a
flaky wall-clock assertion. The canonical local Haskell and full verification
wrappers and the hosted Haskell job provide bounded process-level witnesses.

## Current CLI witness

The witness ran from merged main `a2856b8ac1b275129198fb7cac5a44b36b7f58ff`
against the existing deterministic synthetic fixture
`data/stress-smooth-trend.csv`, SHA-256
`199c373d57e4523b35bddbff77df392ad878e3de682622f72b11a9b44aab7ea5`.

First, the default protected configuration completed with finite Kalman next
value `58839.3650` and z-score `0.413`, but made no trades. Its output named the
binding `CONFORMAL_CONFIRM` gate and showed an effective `0.240%` open
threshold, so zero activity in that run was not evidence of a numerical stall.

The isolated liveness run then removed costs and explicitly neutralized
cost-aware, volatility/confidence, conformal, quantile, confidence-sizing, and
minimum-edge gates:

```text
trader-hs \
  --data ../data/stress-smooth-trend.csv \
  --price-column close \
  --method 10 \
  --epochs 0 \
  --open-threshold 1e-6 \
  --fee 0 --slippage 0 --spread 0 \
  --no-cost-aware-edge \
  --vol-conf-gate disabled \
  --no-confidence-sizing \
  --kalman-z-min 0 \
  --min-edge 0 \
  --no-confirm-conformal \
  --no-confirm-quantiles
```

On the recorded machine it exited successfully in `0.72s` wall time. Output
was finite and reported three position changes, two trade records, one closed
round trip, `28.5%` exposure, and a `100%` Kalman signal rate.

These zero-cost synthetic results are deliberately unsuitable for return or
promotion claims. The extreme annualized metrics are artifacts of a smooth
synthetic fixture, a short 60-bar test segment, and disabled protections. The
witness proves only that current Kalman-only inference, simulation, and report
generation terminate and can produce actionable signals when unrelated gates
do not bind.

## Decision boundary

The historical numerical-instability risk now has current code-level,
long-run, malformed-input, process-completion, and nonzero-activity evidence.
It can close without changing production code. This does not resolve
`ZERO-VIABLE-SIGNAL-001`, validate Kalman alpha, justify disabling any admission
gate, or support champion/challenger promotion. No development or final
holdout was opened, no configuration or artifact changed, no deployment was
requested, and no order was placed.

Reproduce the checked-in regression from the repository root:

```bash
bash scripts/verify.sh haskell
bash scripts/verify.sh full
```

# Formal Methods

In a few words: maximize daily ROI without paying for fragility, churn, or inactivity.

## Formalized requirements

The repo already states the optimizer contract in prose. This document turns that into a small executable specification for the ROI objective used by `--tune-objective roi`.

Clauses:

1. Prefer higher annualized return as the repo's daily-ROI proxy.
2. Penalize drawdown and tail loss.
3. Penalize turnover.
4. Reward positive expectancy.
5. Reward faster payback.
6. Penalize low activity and idle capital.

Threshold-sweep tie-break contract:

1. Prefer higher final equity.
2. If equity ties, prefer lower turnover.
3. If turnover ties, prefer more round trips.
4. If still tied, prefer non-inverted hysteresis (`closeThreshold <= openThreshold`).
5. If still tied, prefer the lexicographically larger `(openThreshold, closeThreshold)` pair.

## Where it lives

- Spec + implementation mirror: `haskell/app/Trader/Formal/Optimization.hs`
- Production optimizer usage: `haskell/app/Trader/Optimization.hs`
- Verification harness: `haskell/test/TestMain.hs`

## What is proved

`verifyFormalOptimization` performs bounded exhaustive model checking over:

- `139,968` ROI states
- `11,664` ordered tie-break candidate pairs

The verifier checks:

1. The executable spec and the production ROI implementation return the same score on every modeled state.
2. ROI score is monotone in the intended direction:
   - non-decreasing in return and expectancy
   - non-increasing in drawdown, tail loss, turnover, and slower payback
3. Activity and exposure penalties are ordered as intended.
4. The threshold tie-break implementation matches the lexicographic spec on every modeled pair.

## Formal UI trade-sizing model

The web UI now treats manual `/trade` sizing as a finite-state contract instead of a loose collection of field-level hints.

Clauses:

1. A manual trade is only ready when there is an effective sizing mode.
2. Effective sizing follows a fixed precedence: `orderQuantity` > `orderQuote` > `orderQuoteFraction`.
3. Invalid quote-fraction ranges (`< 0` or `> 1`) block trading only when no higher-precedence valid size is present.
4. Multiple valid sizing inputs are allowed but must be reported as a conflict so the effective mode is explicit.
5. `maxOrderQuote` is cap-only metadata: it never becomes an active sizing mode, never creates trade readiness by itself, and only changes labeling when `orderQuoteFraction` is the effective mode.
6. The blocking-focus target is `orderQuoteFraction` only for fraction-only invalid states; every other state keeps the safe default target `orderQuote`.
7. `statusLabel` is derived from the same finite state: blocked states say `Sizing required`, and ready states name the effective sizing mode.

The verifier in `haskell/web/test/utils.test.mjs` performs bounded exhaustive enumeration over the modeled sizing state space (`32` states), plus paired cap/no-cap comparisons over every `(orderQuantity, orderQuote, orderQuoteFraction)` combination:

- `orderQuantity ∈ {0, 1}`
- `orderQuote ∈ {0, 1}`
- `orderQuoteFraction ∈ {-0.25, 0, 0.5, 1.25}`
- `maxOrderQuote ∈ {0, 25}`

For every state, it checks:

1. The active sizing modes match the executable spec.
2. The effective sizing mode matches the documented precedence.
3. Trade readiness, blocking reason, and `blockingTargetId` match the modeled state.
4. `statusLabel` matches the modeled readiness/effective-mode summary.
5. Conflict severity (`ok` / `warn` / `bad`) matches the modeled state.
6. The quote-cap metadata stays inert outside effective quote-fraction sizing and only changes `effectiveLabel` / `hint` when quote-fraction sizing is actually effective, including restored cap-only form states.

## Formal numeric input fallback contract

`numFromInput` in `haskell/web/src/app/utils.ts` is treated as a conservative parser for restored numeric form fields.

Clauses:

1. Empty or whitespace-only input keeps the supplied fallback unchanged.
2. Unambiguous decimal-comma forms parse as decimals, including signed values and long-prefix forms such as `1234,567`.
3. A single comma with a 1-3 digit prefix and an exactly 3-digit suffix (for example `1,234`, `12,345`, `-1,234`) is ambiguous between decimal-comma and thousands grouping, so the fallback is preserved.
4. Explicit multi-group thousands forms such as `1,234,567` remain parseable.
5. After normalization, only finite numeric results are accepted; non-finite results keep the fallback.

Proof sketch:

- `numFromInput` trims first and returns `fallback` on the empty string, so blank edits cannot overwrite a stored numeric value.
- In the single-comma branch, the `^[-+]?\d{1,3}$` plus `^\d{3}$` ambiguity check is the path that treats `1,234`-style inputs as undecidable, and it returns `fallback` instead of rewriting the saved number.
- Other two-part comma inputs normalize to decimal-comma forms, so explicit decimals like `1,23`, `-1,23`, `+0,125`, and `1234,567` remain parseable.
- The explicit multi-group branch keeps standard thousands-group forms such as `1,234,567` parseable.
- `haskell/web/test/utils.test.mjs` mirrors this contract with regression rows for blank input, decimal-comma parses, ambiguous single-comma fallback, and multi-group comma parses.

## Formal autoloop safety contract

The GitHub autoloop runner treats model output as untrusted input and checks a small executable contract before any commit is created.

Clauses:

1. Every proposed path must be relative, non-empty, and traversal-free.
2. A patch plan may not repeat the same path twice.
3. Every bounded cycle must name one local UI review file under `haskell/web/src/` and one correctness/formal review file under `test/`, `haskell/test/`, `haskell/web/test/`, or `FORMAL_METHODS.md`; both files must already be part of the inspected file set.
4. A patch plan may only modify files that were explicitly requested for inspection by the idea-selection phase.
5. The patch plan must report both review outcomes: `uiReviewSummary` for the UI/UX pass and `correctnessSummary` for the invariant/property/test or proof-sketch pass.
6. Verification commands must come from a fixed allowlist.
7. Git staging is restricted to the planned files that actually changed, so generated byproducts outside the plan are excluded from bot commits.
8. The repo-local forever runner may write only under the gitignored `.tmp/autoloop/` state directory, must honor a stop file or `SIGINT`/`SIGTERM`, and must wait between cycles instead of spinning.

The verifier in `test/autoloop.test.mjs` checks the JSON/path normalization and review-target clauses, `scripts/autoloop.mjs` enforces the subset/staging/phase clauses at runtime, and `scripts/autoloop-forever.mjs` enforces the local stop/sleep/state-directory clauses for the persistent supervisor.

## What is not proved

This does not prove that:

- the predictors forecast correctly
- the trading strategy is profitable on real markets
- exchange integrations, network I/O, or persistence layers are free of bugs
- floating-point arithmetic is globally free of all numerical issues

It proves that the repo's stated ROI contract, as encoded in the optimizer, matches the implementation over the modeled state space.

## Running the verifier

```bash
cd haskell
cabal test
```

## Formal close-timing model (combo-aware)

For each position with open time `ta` and realized close time `tc`, we define an optimization window:

- `tm ∈ [ta, ta + 2*(tc-ta)]`

`optimalCloseObservation` first filters the PnL path to finite candidates whose timestamps stay inside that window. If `tc <= ta` or no candidate survives, no observation is emitted. Otherwise `tm` is selected as the timestamp that maximizes path PnL inside the filtered window.

We then normalize by realized duration:

- `r = (tm-ta)/(tc-ta)`, so `r ∈ [0,2]`

Before fitting per-combo stats, `buildCloseTimingStats` drops any invalid stored observation:

- realized duration must be positive (`tc > ta`)
- optimal-close time must stay inside the modeled window, so normalized `r` remains in `[0,2]`

For stats emitted by `buildCloseTimingStats`, we estimate robust distribution statistics over `r`:

- median (`Q50`) as center
- MAD (`median |r-Q50|`) as robust dispersion
- interquartile band (`Q25`, `Q75`) as a policy interval
- `boundedPercentile` clamps every percentile back into `[0,2]`, and `orderQuartiles` sorts the fitted quartiles so `0 <= Q25 <= Q50 <= Q75 <= 2`

A risk-budgeted close policy is encoded as a convex blend target:

- `beta = clamp(0, 1, riskBudget)` when the supplied budget is finite; otherwise `beta = 0`
- `closeTimingDecision` re-clamps `Q50`, promotes `Q75` to at least `Q50`, and computes `target = clampRatio ((1-beta)*Q50 + beta*Q75)`
- therefore `target` stays finite and inside `[Q50, Q75] ⊆ [0,2]`

A live position is marked close-ready when its age ratio exceeds `target`.

The regression checks in `haskell/test/TestMain.hs` cover representative window selection and the boundary risk-budget decisions (`beta = 0` and `beta = 1`). This proof sketch now also makes the invalid-sample dropping (`tc <= ta`, `tm < ta`, and `tm > ta + 2*(tc-ta)`) and non-finite-budget contract (`NaN`, `+Infinity`, and `-Infinity` normalize to the same median-target policy as `beta = 0`) explicit, matching `observationRatioParts`, `validObservation`, `decisionTargetBand`, `normalizeRiskBudget`, and `clampRatio` in `haskell/app/Trader/Formal/CloseTiming.hs`.

### Implementation pointers

- Model + policy: `haskell/app/Trader/Formal/CloseTiming.hs`
- Unit tests: `haskell/test/TestMain.hs`
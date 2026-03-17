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
4. The blocking target is `orderQuoteFraction` only for standalone quote-fraction validation failures; every other blocked sizing state anchors on `orderQuote`.
5. Multiple valid sizing inputs are allowed but must be reported as a conflict so the effective mode is explicit.
6. `maxOrderQuote` is cap-only metadata: it never becomes an active sizing mode, never creates trade readiness by itself, and only changes labeling when `orderQuoteFraction` is the effective mode.
7. Every blocked sizing state must report `Sizing required` as the status label and repeat the blocking message as the operator hint.
8. Every non-blocked conflict must surface the precedence warning for the effective mode so operators can see which sizing input wins.
9. Every non-blocked single-mode state must surface `Effective sizing: <effectiveLabel>.` so the operator sees the exact active sizing label.

The verifier in `haskell/web/test/utils.test.mjs` performs bounded exhaustive enumeration over the modeled sizing state space:

- `orderQuantity ∈ {0, 1}`
- `orderQuote ∈ {0, 1}`
- `orderQuoteFraction ∈ {-0.25, 0, 0.5, 1.25}`
- `maxOrderQuote ∈ {0, 25}`

For every state, it checks:

1. The active sizing modes match the executable spec.
2. The effective sizing mode matches the documented precedence.
3. Trade readiness is blocked exactly when the model says no effective size exists.
4. The blocking target matches the documented anchor: standalone quote-fraction validation failures target `orderQuoteFraction`; all other states target `orderQuote`.
5. Conflict severity (`ok` / `warn` / `bad`) matches the modeled state.
6. The quote-cap metadata stays inert outside effective quote-fraction sizing, including restored cap-only form states.
7. Blocked states report `Sizing required` and mirror the blocking message into the hint.
8. Conflicting valid states surface the precedence warning, and single valid states surface the effective sizing label in the hint.

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

`tm` is selected as the timestamp that maximizes path PnL inside that window.

Close-timing selection invariant:

1. `tm` must maximize path PnL over all in-window observations.
2. If multiple in-window timestamps share the maximum path PnL, the model selects the earliest such timestamp.
3. The earliest-max rule must be invariant to input path order, so downstream stats and risk-budget targets do not depend on how equal-PnL samples arrive.

We then normalize by realized duration:

- `r = (tm-ta)/(tc-ta)`, so `r ∈ [0,2]`

Per combo, we estimate robust distribution statistics over `r`:

- median (`Q50`) as center
- MAD (`median |r-Q50|`) as robust dispersion
- interquartile band (`Q25`, `Q75`) as a policy interval

A risk-budgeted close policy is encoded as a convex blend target:

- `target = (1-β)*Q50 + β*Q75`, with `β ∈ [0,1]`

A live position is marked close-ready when its age ratio exceeds `target`.

Bounded-arithmetic invariant:

1. Window membership is evaluated in mathematical integer space before comparing timestamps, so a mathematically valid `tm` is not dropped when `ta + 2*(tc-ta)` exceeds `maxBound :: Int`.
2. Boundary windows such as `ta = minBound :: Int`, `tc = 0`, `tm = maxBound :: Int` remain admissible whenever they satisfy the mathematical window.
3. Observation validity and normalized `r` use mathematical integer deltas for `tm-ta` and `tc-ta`, so full-span `Int` observations such as `ta = minBound :: Int`, `tc = maxBound :: Int`, `tm = maxBound :: Int` retain `r = 1`.
4. Live age ratios use the same overflow-free delta arithmetic, so full-span `Int` close-readiness remains finite and depends on the modeled timestamps instead of machine-width wraparound.

### Implementation pointers

- Model + policy: `haskell/app/Trader/Formal/CloseTiming.hs`
- Unit tests: `haskell/test/TestMain.hs`
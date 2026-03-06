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

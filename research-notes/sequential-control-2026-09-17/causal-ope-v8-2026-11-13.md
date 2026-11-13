# Causal OPE v8 — engineering increment, 2026-11-13

Engineering evidence only: no market data was read, no financial trial was run
and no holdout was opened.

## What was added

`scripts/research/causal_ope_v8.py` is a default-disabled adapter that turns
the first H recorded transitions of causal-value-v6 and causal-ppo-v7 runs into
episodes for the exact estimator ope-rational-v2. It imports no frozen v1
module and neither trains nor samples.

## Findings

1. **The sampling-law gap is bounded.** The policy-gradient learner records the
   binary64 softmax probability of the drawn action but draws with one uniform
   k/2^53 against rounded cumulative thresholds. The two differ by at most
   2^-50 for every action. Above a 2^-10 floor that is at most 2^-40
   relatively, and at most 2^-34 on a 32-decision trajectory weight. The value
   learner's law is exact; only its binary64 conversion rounds.
2. **A floor is necessary.** The learners admit probabilities down to 2^-100.
   A recorded 2^-100 is drawn with probability 2^-53, about 2^47 times more
   often, so the adapter rejects evaluated probabilities below 2^-10.
3. **Exact evaluation has small capacity.** Every binary64 behavior probability
   puts up to 53 bits into the common denominator of the weights and 106 into
   the sum of squares. The estimator's 8192-bit limit therefore rejects six
   fixture runs at H = 32 (four are admitted; three policy-gradient runs alone
   reject, two are admitted). The rejection is all-or-nothing and never silent,
   but the adapter cannot evaluate a statistically meaningful number of long
   policy-gradient episodes.
4. **No run is dropped.** A run shorter than H, including one that terminated
   early, rejects the whole evaluation.

## Evidence

| Requirement | Method | Result |
|---|---|---|
| F-RL-OPE-V8-SOURCE | complete AST lock, transitive import closure, statement and prefix-access review; learner sampling statements re-read from their locks | eleven local modules, five external imports, no frozen module; 22 source mutants reject with a refreshed lock |
| F-RL-OPE-V8-ARITH | quantifier-free Z3 real arithmetic with exact rational evaluation | four implications unsat, one floor-removal control sat; (1 + 2^-40)^32 - 1 <= 2^-34 |
| F-RL-OPE-V8-FLOW | explicit-state stage model, 1..4 runs | 48 states, 52 transitions; three mutants reject |
| F-RL-OPE-V8-CONFORMANCE | exact-rational estimator oracle and integer-counted sampling-law oracle | 6 runs, 18 evaluations, 24 oracle episodes, 96 sampling-law checks, 12 prefix rewrites, 26 rejections, 1 capacity rejection |

## Not claimed

Unbiasedness, coverage, variance, independence across seeds, causal validity of
caller-supplied target probabilities and value models, data support,
profitability or deployability. The gap bounds concern trajectory weights, not
estimator error. The rounding and generator models are standard assumptions,
not machine-proved for this CPython or NumPy build. The published estimate
carries `reliable = False`.

## Obligation 10

Stays open. Remaining: network forward and optimizer-v5 arithmetic is guarded,
not bounded; exact evaluation needs more capacity (a separately versioned
estimator limit or dyadic recorded probabilities); and retiring the frozen
paths needs a new preregistration.

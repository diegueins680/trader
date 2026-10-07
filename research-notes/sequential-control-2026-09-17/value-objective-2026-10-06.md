# Value objective v2 — 2026-10-06

**Decision: retain a disabled, disconnected objective kernel; no adoption.** No
market data, archived results or holdout were read. Synthetic arrays only. The
frozen learner, champion and all reported value-based results are unchanged.

The [registration](../registrations/value-objective-v2-engineering.json) and
[contract](../../formal/research/value-objective-v2-contract.md) were committed
as `5626e92a` before the implementation.

## Why this increment

Obligation 10's blockers include CE-RL-014 and CE-RL-015 in the frozen
conservative Double-DQN helper `bellman_gradient`:

- **CE-RL-014:** finite `±max` Q-values with alpha = 0 give a NaN loss, because
  the penalty is evaluated eagerly and `0 · inf` follows.
- **CE-RL-015:** an exact common shift by 2^54 turns the 0.1·log 3 penalty into
  0, because `(m + log 3) − q_a` cancels at an ulp of 4.

## What changed

`scripts/research/value_objective_v2.py` (`objective_v2`, `enabled=False` by
default):

- Scalar IEEE binary64 operations with `math.fsum`; no NumPy/BLAS reductions.
- Admits three-action float rows, int actions, float targets with |value| ≤ 2^100,
  and 0 ≤ alpha ≤ 1.
- Skips the conservative penalty entirely when alpha = 0.
- Computes the penalty from differences only: `(m − q_a) + log Σ exp(q_j − m)`.
- Counts `exp` terms that underflow to zero or subnormal and publishes the count.
- Canonicalizes −0 to +0 on publication and rejects any non-finite result.

## Counterexample found during proof: CE-RL-024

The preregistered draft claimed **bitwise** invariance of differences under an
exact shift. A bit-blasted Z3 query over IEEE binary32 returned SAT: a = −0,
b = +0, s = −1.05·2^72 give (a+s) − (b+s) = +0 but a − b = −0. The same values
reproduce in binary64. Full binary64 bit-blasting returned `unknown` after
120 s, so it cannot be the gate. Resolution:

- Published floats are canonicalized with `x + 0.0`.
- The intermediate claim is restated as IEEE value equality and proved by a
  congruence lemma under the named assumption A-FP-ROUNDING.
- The witness is preserved in
  [value-v2-counterexamples.json](../../formal/research/value-v2-counterexamples.json)
  and re-evaluated in Z3 and in binary64 on every formal run.

## Evidence

| Requirement | Status | Scope |
|---|---|---|
| F-RL-VALUE-V2-SOURCE | exhaustively_checked | AST lock; scalar-only imports; alpha = 0 skip; difference-only penalty; fsum; zero canonicalization; finite guard; activation-first default. |
| F-RL-VALUE-V2-ARITH | smt_verified | 6 SAT-premise/UNSAT-violation pairs (congruence under A-FP-ROUNDING, max commutes with shift, exponent sign, 1 ≤ total ≤ 3, probability bounds, one-rounding magnitude lemma) plus an exact rational rounding chain: loss < 2^201, gradient < 2^101 on the admitted domain. Resolved counterexample: CE-RL-024. |
| F-RL-VALUE-V2-FLOW | model_checked | 6 states, 9 transitions; a mutant publishing before the finite guard is caught. |
| F-RL-VALUE-V2-CONFORMANCE | property_tested | 96 batches (scales 1, 50, 2^40, 2^90; alpha 0, 0.1, 1) agree with a 60-digit decimal oracle within 1e-12; 16 finite-difference gradient checks; underflow counts match; 96 ordinary batches agree with the frozen helper; CE-RL-014/015/024 witnesses. |

Five integrity tests cover the end-to-end check, eight source mutants, the model
mutant, solver refusal, 14 malformed inputs, injected failures, the alpha = 0
skip, published underflow, immutability, and a check that swapping the
frozen-style penalty back in reintroduces CE-RL-015 and fails the witness gate.

## Verification

| Command | Result |
|---|---|
| `scripts/formal/test_integrity.py` | 306/306 OK (251.9 s) |
| `scripts/formal/verify.py --record`, then plain `verify.py` | both exit 0; 12 closed / 26 unresolved; `missionComplete` false |
| `node scripts/verify-formal-specs.mjs` | valid |

## Limitations

A-VALUE-OBJECTIVE-V2 and A-FP-ROUNDING trust CPython's IEEE behavior, the stated
exp/log primitive bounds and the standard rounding model; none is machine-proved
for the interpreter build. The kernel is not composed into a learner, so frozen
CE-RL-014/015 remain in `train_q`. Obligation 10 stays **open**; its blockers now
name the unconnected successors explicitly.

## Recommendation

Unchanged: **no candidate passed**. Next step for obligation 10: register a
successor learner composing value-objective-v2, gae-targets-v2 and the exact
replay runner, so that retiring the frozen paths can be decided under a new
preregistration rather than by editing archived code.

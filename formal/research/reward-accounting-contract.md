# Reward and economic accounting audit — 2026-10-03

Preregistered engineering audit of unchanged Replay and economic reporting.
Classifications: accounting correctness, numeric correctness, statistical
interpretation and implementation conformance. No financial trial or data access.

Extract arithmetic expressions from pinned Replay.step, Replay._trade,
economic and _admit_economic_ledger ASTs. Bind the full function bodies to the
registration. Restricted translation rejects unsupported syntax. Hash checks
establish source identity, not complete interpreter refinement.

## Requirements

- F-RL-ROW-RECONCILE: in exact reals, E>0, old-position gross G, signed funding F,
  target-trade cost vector C and terminal-trade vector L, the sequential source
  updates yield E'=E+G+F-sum(C)-sum(L). The merged row reports C+L by cost category
  and net r=E'/E-1=(G+F-sum(C+L))/E. Missing calls contribute zero vectors.
  Cost formation, fill quantities and square-root impact estimates are not proved.
- F-RL-WEALTH-FOLD: starting wealth E0>0, accumulator P=1 at E=E0; with P=E/E0,
  E>0, E'>0 and source row r=E'/E-1, P'=P(1+r)=E'/E0. Check base and inductive
  step separately. Thus a finite positive-wealth row sequence compounds to its
  endpoint wealth ratio, subject to the stated induction argument. The current
  economic report subtracts one from final wealth when initial equity is one.
- F-RL-REWARD-RECONCILE: for one call starting at B>0, source reward
  R=100*((E'-B)/B-k*Q), Q=sum(w_i^2*v_i^2), k>=0. Source row penalties sum to
  100*k*Q. Check accumulator base and step and the final identity
  R+sum(rowPenalty)=100*(E'/B-1). Penalties are noncash and must not be debited
  from the economic ledger. No policy-invariance or CVaR theorem is claimed.
- F-RL-REWARD-ADDITIVE (claim to refute): even with zero costs, zero funding,
  zero inventory penalty and no discount, summing call rewards does not generally
  equal 100 times whole-episode net return. Preserve the registered rational
  witness with returns 0,+1/40,-1/40: sum reward=0, final wealth=1599/1600.
  This refutes an interpretation of the reward, not the existing economic reporter.

SMT uses exact real arithmetic, separate SAT-premise and UNSAT-violation queries,
seed 0 and 10-second limits. Refutation requires its prescribed SAT witness;
UNKNOWN always fails. The fold uses base/step verification, not bounded unrolling
presented as a general theorem. The inference from those obligations to finite
paths is stated explicitly; no Python induction/refinement theorem is claimed.

## Consistency and assumptions

The v1 registration and environment contract make each reward relative to that
call's starting equity. economic reports final equity minus one independently.
Preserve both semantics. Clarify that additive discounted learning reward and
compounded economic return are different objectives. Do not pool cadences or
reinterpret prior reward/OPE values as net trading return. No new log-return
reward or successor policy is introduced.

The existing phrase that equity reconciles “exactly” states the algebraic
contract. The actual reporter uses 1e-10 equity and 1e-12 row-return tolerances.
Document that distinction explicitly. This audit does not claim binary64 exact
equality, a universal rounding-error bound or finite outputs for arbitrary
admitted inputs. Overflow, missing data, incomplete failure paths, trusted cost
formation and full source-to-model refinement remain blockers.

Assume stable ordinary numeric state, positive denominators, nonnegative finite
cost vectors where required, standard Python/NumPy semantics, complete row
publication and returning helpers. Existing replay ordering/cutoff models provide
their previously scoped evidence; they do not prove all numeric helper semantics.
Early gate/market failures may have no new row and retained inventory. Insolvency
can invalidate the positive-wealth induction premise and learning admission.

## Conformance and boundaries

Use only deterministic synthetic arrays, causal prefix-fitted normalization and
the registered 1,944-case product grid: 3 targets × 3 horizons × 2 delays ×
3 remaining lengths × 3 cost multipliers × 3 funding signs × 2 price patterns ×
2 penalty coefficients = 1,944. Compare each call's returned reward with its
newly published penalties, each row's cash identity, compounded row returns,
final equity and actual economic report. Use registered 1e-11 absolute/relative
tolerances; these are tests, not formal floating-point bounds. Include the four
registered failure scenarios and the exact nonadditivity fixture. Retain failed
checks, not just passing cases. Mutations must expose wrong reward denominator,
cash penalty debit, missing cost category, double cost, and additive-return
interpretation. No policy training or original OPE execution is permitted.

Reuse Python 3.13.3, NumPy 2.3.5 and Z3 4.15.4. Reproduce offline after dependency
installation through formal/full wrappers. Add requirement/assumption/test/code/
CI links and results to the ledger. All 38 broader mission obligations remain
open/partial; this audit confers no candidate promotion or production authority.

# Gap-risk assurance contract — 2026-09-28

Specified before implementation. Base main: `dbd45e26`; verification dependency:
PR #281, `1817742d`. No new financial trial, policy, production adapter or risk
limit. This extends the existing rejected sequential-control research assurance.

## Requirements and intended meaning

`F-RL-GAP-BOUND` (numeric, accounting, safety): in exact real arithmetic, let E>0
be pre-bar equity, w the **drifted pre-bar exposure**, r the simple price return,
and c the total net debit divided by E (execution/liquidation costs plus signed
funding debit). E' = E(1+w*r-c). If |w|<=a, |r|<=b, 0<=c<=k,
0<=a<=1/4, b>=0, k>=0, 0<=d<1 and a*b+k<=d, then E'>=E(1-d)>0.
Funding credits (c<0) are outside this particular sufficient-condition theorem.
The inequalities are hypotheses, not facts established about future markets.

`F-RL-DRAWDOWN-COMPOSE` (numeric, safety): if peak P>=E>0,
E>=P(1-d0), E'>=E(1-d), E'>0 and d0,d in [0,1), then for
P'=max(P,E'), drawdown 1-E'/P' <= d0+d-d0*d. A one-bar loss budget is
not a fresh peak-to-trough budget at every step.

`F-RL-POSTCOST-EXPOSURE` (numeric, accounting): a full target fill made using
pre-cost equity gives post-cost absolute exposure |target|/(1-c). For
0<=c<=k<1, L>=0 and |target|<=L(1-k), post-cost exposure is <=L.
The target limit 1/4 and endpoint monitor 0.35 have different meanings. Partial
fills, drift and funding require separate treatment. No live cap is changed.

`F-RL-UNCONDITIONAL-FLOOR` (safety, **claim to refute**): an admitted target
in {-1/4,0,1/4}, positive prices and the existing replay shield suffice to keep
equity >=0.80 at every bar. Preserve exact rational counterexamples and replay
them against Python `Replay`, including its real fee, spread, slippage, delay,
pending-action and terminal-liquidation semantics. A later proposed exit cannot
undo old-inventory P&L before the next close. Refutation is a successful verifier
result only for this explicitly refuted claim; unexpected SAT remains failure.

`F-RL-GAP-CONFORMANCE` (implementation testing): deterministic real Replay
traces reproduce each stored witness within binary64 error tolerance, retain
the loss as failed evidence, close solvent terminal positions, cancel pending
targets, reject learning admission for unaccounted failures, and agree with an
independent exact-rational reference on a declared finite grid. Both future-data
perturbation and all three proposal targets are checked. This is finite
conformance testing, not universal refinement or a financial experiment.

## Abstraction and proof obligations

Abstract a bar immediately before an endpoint fill by
E=equity, w=units*leftPrice/E, r=rightPrice/leftPrice-1, and
c=(cash execution costs - funding cash)/E. The exact identity models existing
old-unit mark/funding-before-fill accounting. Concrete binary64 arithmetic,
NumPy, Python execution, fill realism and correctness of external market data
are not proved by a real-arithmetic theorem.

Z3 4.15.4, Python 3.13.3, and the existing hash-locked tooling are reused.
Negated sufficient-condition theorems must be UNSAT; their premises must each
be SAT. The explicitly false floor claim must have the prescribed SAT witnesses.
UNKNOWN, timeouts and missing witnesses fail. Source hashes, canonical clauses,
ledger status, tool versions and CI must agree. No skipped obligation is allowed.

Assumption `A-GAP`: bounded returns and bounded funding/execution debits are
external assumptions. They are **not** established for cryptocurrency prices,
the historical panel or this simulator's input domain. `A-ACCOUNT`, `A-FP`,
`A-SOLVER` and `A-GUARDS` retain their existing meanings. All assumptions map to
HIGH/OPEN `RL-OFFLINE-001`.

## Consistency resolution and limits

The authoritative existing `environment-contract.md` explicitly says gaps can
cross risk thresholds before detection. The stronger mission reading, that the
shield proves a pathwise loss ceiling for all positive future prices, conflicts
with those transition semantics. Preserve the actual monitor semantics; record
the stronger claim as refuted instead of adding an untracked bounded-jump axiom
or discarding failed paths. Solving this empirical/modeling gap is a prerequisite
for any future candidate admission. It is not permission to weaken acceptance.

The exact trace is a deterministic engineering counterexample, not new evidence
about frequency of market gaps, policy profitability or an existing production
bug. No probabilistic market guarantee, neural certificate, universal binary64
accounting proof, production concurrency proof or completed mission is claimed.
Whole-system obligations remain open/partial; protected holdouts stay sealed.

# Exact replay accounting, 2026-10-06

This is engineering progress, not mission completion or new economic evidence.
The 38 original scopes and closure criteria remain unchanged:12 scoped closures,
25 partial,1 open. No candidate is adopted. Continue offline research.

## Problem and implementation

Finite binary64 inputs can overflow funding products or lose very small debits.
The frozen v1 replay also cannot establish exact accounting identities from
real-number lemmas alone. The new `replay-accounting-v2` kernel admits bounded
native `Fraction` inputs and publishes an immutable exact one-bar receipt.
Both initial-state creation and transitions default to disabled and require the
explicit version. No runner, learner, policy, filesystem or production caller is
added. It is a numerical transition building block, not an execution-authorized
policy consumer. Caller-side data admission and policy shielding remain required
before future composition. The existing replay and experiment evidence are not
rewritten or reinterpreted.

Raw funding mark/rate products, signed inventory mark-to-market, fees, spread,
slippage, impact and liquidation are accounted without binary64 conversion.
Costs preserve the original5/0.5/4.5bps assumptions; they are assumptions, not
empirical calibration. Impact uses a conservatively rounded-up square root on a
2^-32 grid. This explicit semantic version differs from binary64 execution.
Partial fill is an input assumption; terminal close assumes full fill if solvent.
A gap can exhaust equity before a close: the receipt explicitly records failed
liquidation and outstanding inventory. Hard thresholds are breach detectors,
not guarantees that cryptocurrency prices cannot jump across them.

All rational intermediates are bounded to8192 numerator/denominator bits.
Oversize/arithmetic/resource failures return absent, preserving input state and
publishing no partial receipt. Allocation failure cannot guarantee process survival
or a physical deadline. Up to128 settlement events are accepted per tick;
tick4096 forces terminal accounting. Every continuing state has positive equity;
terminal states reject subsequent transitions.

## Specification and evidence

Preregistration:87f2c795, before implementation.
[Contract](../../formal/research/replay-accounting-v2-contract.md).
F-RL-ACCOUNT-V2-SOURCE checks17 complete reviewed definitions and their pure
imports, immutable types, checked primitives, exports and default/version guards.
Nine Z3 obligations have SAT premises and UNSAT violation queries. These cover
source-derived wealth/debit algebra, additive liquidation, target arithmetic,
integer intermediate bounds and conservative sqrt enclosure under the named
isqrt primitive contract. They are not a full Python interpreter refinement.

The finite publication abstraction checks28 states,52 transitions, maximum
shortest depth3, including rejected computation and insolvent termination.
It abstracts numeric values and funding-loop length; no probabilistic market or
wall-clock liveness claim follows. An incomplete-terminal-publication mutation
fails the checker. Source/debit/default mutations and SAT/UNKNOWN solver outcomes
are rejected by the integrity suite.

128 fixed-seed synthetic episodes produce1166 exact transitions. Each wealth
result matches the independent compiled Haskell Rational oracle, and each
transition repeats deterministically. Additional regressions exercise fractional
fills, close costs, gap insolvency, maximum ticks, turnover breach, oversized raw
funding, subnormal-sized exact debits, invalid inputs and injected resource failure.
The targeted8-test suite passed locally in21.101s; a separate full component run
including proof/model/Haskell conformance took8.227s on this host. These are
verification timings, not policy inference or trading performance benchmarks.

The delivered-source promotion inventory expands from15 to16 modules without a
new order, file or promotion effect. Its old human-readable ledger bounds were
stale at12 modules; they now match the reviewed16 modules,17 local imports,
2062 call sites and22 control fields. Existing12 scoped closures additionally
require reproduction of the new module's source-boundary certificate.

## Verification and limitations

Canonical formal/full and final-head CI results are recorded below after execution.
Original v1/v2 research inputs, artifacts, policies, learner arithmetic and
counterexamples remain frozen. This kernel does not establish total accounting
refinement of either the frozen replay or a future composed learner. Obligations
6/8/10/18/19 receive scoped evidence; none closes. Last open10 still includes
Q/CQL/PPO arithmetic, silent BLAS underflow, loading and whole-path publication.
No authentic historical availability witnesses are invented.

No market datasets or sealed holdouts were read. No financial training campaign,
OOS comparison, cost/delay stress campaign, OPE rerun, drawdown/tail comparison or
policy inference benchmark occurred. Existing108 fits and19,440 replays remain
contaminated development evidence; all108 OPE batches remain invalid.1,227 final
returns stay sealed and the prospective embargo remains2027-01-20T13:00Z.
Champion, reviewed fleet, exposure, credentials, live authorizations and deployment
configuration are unchanged. No live exploration or order occurred. No new
policy or production order authority exists. No proof placeholder is permitted.

Local full certificate recording failed before completion at
`python scripts/formal/verify.py --record`: creating the shutdown-conformance
temporary directory raised `OSError: [Errno 28] No space left on device`.
This was not a passed verification. Only this isolated worktree's338MiB generated
`haskell/dist-newstyle` cache was removed; other worktrees/services were untouched.
The separate7 promotion integrity tests passed in21.240s and the formal-specification
schema passed. Pinned CI must supply both complete canonical wrapper results.

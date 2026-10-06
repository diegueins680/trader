# Exact replay accounting, 2026-10-06

This is engineering progress, not mission completion or new economic evidence.
The 38 original scopes and closure criteria remain unchanged: 12 scoped closures,
25 partial, 1 open. No candidate is adopted. Continue offline research.

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
Costs preserve the original 5/0.5/4.5bps assumptions; they are assumptions, not
empirical calibration. Impact uses a conservatively rounded-up square root on a
2^-32 grid. This explicit semantic version differs from binary64 execution.
Partial fill is an input assumption; terminal close assumes full fill if solvent.
A gap can exhaust equity before a close: the receipt explicitly records failed
liquidation and outstanding inventory. Hard thresholds are breach detectors,
not guarantees that cryptocurrency prices cannot jump across them.

All rational intermediates are bounded to 8192 numerator/denominator bits.
Oversize/arithmetic/resource failures return absent, preserving input state and
publishing no partial receipt. Allocation failure cannot guarantee process survival
or a physical deadline. Up to 128 settlement events are accepted per tick;
tick 4096 forces terminal accounting. Every continuing state has positive equity;
terminal states reject subsequent transitions.

## Specification and evidence

Preregistration: 87f2c795, before implementation.
[Contract](../../formal/research/replay-accounting-v2-contract.md).
F-RL-ACCOUNT-V2-SOURCE checks 17 complete reviewed definitions and their pure
imports, immutable types, checked primitives, exports and default/version guards.
Nine Z3 obligations have SAT premises and UNSAT violation queries. These cover
source-derived wealth/debit algebra, additive liquidation, target arithmetic,
integer intermediate bounds and conservative sqrt enclosure under the named
isqrt primitive contract. They are not a full Python interpreter refinement.

The finite publication abstraction checks 28 states, 52 transitions, maximum
shortest depth 3, including rejected computation and insolvent termination.
It abstracts numeric values and funding-loop length; no probabilistic market or
wall-clock liveness claim follows. An incomplete-terminal-publication mutation
fails the checker. Source/debit/default mutations and SAT/UNKNOWN solver outcomes
are rejected by the integrity suite.

128 fixed-seed synthetic episodes produce 1166 exact transitions. Each wealth
result matches the independent compiled Haskell Rational oracle, and each
transition repeats deterministically. Additional regressions exercise fractional
fills, close costs, gap insolvency, maximum ticks, turnover breach, oversized raw
funding, subnormal-sized exact debits, invalid inputs and injected resource failure.
The targeted 8-test suite passed locally in 21.101s; a separate full component run
including proof/model/Haskell conformance took 8.227s on this host. These are
verification timings, not policy inference or trading performance benchmarks.

The delivered-source promotion inventory expands from 15 to 16 modules without a
new order, file or promotion effect. Its old human-readable ledger bounds were
stale at 12 modules; they now match the reviewed 16 modules, 17 local imports,
2062 call sites and 22 control fields. Existing 12 scoped closures additionally
require reproduction of the new module's source-boundary certificate.

## Verification and limitations

Canonical formal/full and final-head CI results are recorded below after execution.
Original v1/v2 research inputs, artifacts, policies, learner arithmetic and
counterexamples remain frozen. This kernel does not establish total accounting
refinement of either the frozen replay or a future composed learner. Obligations
6/8/10/18/19 receive scoped evidence; none closes. Last open 10 still includes
Q/CQL/PPO arithmetic, silent BLAS underflow, loading and whole-path publication.
No authentic historical availability witnesses are invented.

No market datasets or sealed holdouts were read. No financial training campaign,
OOS comparison, cost/delay stress campaign, OPE rerun, drawdown/tail comparison or
policy inference benchmark occurred. Existing 108 fits and 19,440 replays remain
contaminated development evidence; all 108 OPE batches remain invalid. 1,227 final
returns stay sealed and the prospective embargo remains 2027-01-20T13:00Z.
Champion, reviewed fleet, exposure, credentials, live authorizations and deployment
configuration are unchanged. No live exploration or order occurred. No new
policy or production order authority exists. No proof placeholder is permitted.

Local full certificate recording failed before completion at
`python scripts/formal/verify.py --record`: creating the shutdown-conformance
temporary directory raised `OSError: [Errno 28] No space left on device`.
This was not a passed verification. Only this isolated worktree's 338 MiB generated
`haskell/dist-newstyle` cache was removed; other worktrees/services were untouched.
The separate 7 promotion integrity tests passed in 21.240s and the formal-specification
schema passed. Pinned CI must supply both complete canonical wrapper results.

Pinned run 37434513361 reproduced the receipt in 77 s, then failed the canonical
formal wrapper: `LifecycleStageTests.test_actual_stage_composition` still asserted
15 research modules after adding the 16th. 289 other tests passed; 290 ran in 93.149s.
The assertion is updated to the reviewed 16-module inventory. No runtime or proof
predicate changes. Full was skipped. Ordinary stale-receipt CI 37434513485 was
canceled, not passed. This failed receipt is not imported as a successful run.

## Kernel benchmark

CPython 3.13.3, Darwin x86_64, 100 warmups and 1,000 repeated single transitions:
median 0.138058 ms, p99 0.325564 ms, maximum 0.403635 ms. This measures only the
exact arithmetic kernel under ordinary values; it is not policy inference,
worst-case arithmetic, concurrent throughput or an end-to-end timing guarantee.
Reproduce from the repository root with the pinned Python environment:

```python
import sys, time, statistics
from fractions import Fraction as F
sys.path.insert(0, 'scripts/research')
import replay_accounting_v2 as a
state = a.initial_v2(F(100), enabled=True)
times = []
for i in range(1100):
    start = time.perf_counter_ns()
    result = a.advance_v2(state, F(101), ((F(101), F(1, 10000)),),
                          F(1, 4), impact=F(1, 10000), enabled=True)
    elapsed = (time.perf_counter_ns() - start) / 1e6
    assert result is not None
    if i >= 100:
        times.append(elapsed)
print(statistics.median(times), sorted(times)[989], max(times))
```

## Passing canonical verification

Pinned [run 37435366856](https://github.com/diegueins680/trader/actions/runs/37435366856),
job 112175744427, source `196828f658ccb5e8ad28b3a2dd08f6f60efcc7c3`:
receipt generation 79 s; `bash scripts/verify.sh formal` 168 s;
`bash scripts/verify.sh full` 561 s. Both wrappers passed all 290 integrity tests
(87.572 s and 94.325 s). Haskell build/format/lint/smoke/tests passed; web 241/241
and automation 185/185 passed. All 81 SMT groups reproduced UNSAT violation
queries. The source/model/conformance receipt was imported verbatim, SHA256
`0832f7afed58358040f2d2524ac504be293b17e7ab973c9a15b2de9cd2c4f20f`.
Only exactReplayV2, the SMT roster, source hashes, and the reviewed module/call
counts in promotion/champion/stage/shield inventories changed. Existing model,
authorization and deployment results remain unchanged. Closure summary remains
12 closed /26 unresolved, with both research acceptance gates blocked.

The temporary reproduction workflow was removed after success. Ordinary
stale-receipt CI 37435366848 was canceled, not passed. The separate corrected
lifecycle-stage suite passed all 9 tests locally in 2.319 s. Final-head CI and
post-merge deployment observations are recorded on PR314; they cannot be known
inside the commit they attest to. No waiver, skipped proof or gate weakening was
used to obtain the passing canonical results.

# Exact OPE arithmetic successor: engineering evidence

Base main 2c37ae3f5d2d30d49aa6aedf44a3d9294f0848e5. Registration commit 6224c947
preceded implementation. No financial trial or historical data read occurred.

## Problem and change

The old OPE helper computes probability ratios, prefix products, discounts,
estimates and ESS in binary64. The existing exact ESS helper starts *after*
weights have already formed, so it cannot recover a positive weight lost upstream.
The separate default-disabled `ope_rational_v2.estimate_v2` converts validated
base floats to exact rationals before all estimator arithmetic. It computes the
complete IS, PDIS, WIS and DR batch means and trajectory-weight ESS, preserving
per-episode results for replay. This is an explicit successor, not a silent
replacement of frozen evidence or a new production consumer.

The [contract](../../formal/research/ope-rational-contract.md) defines exact formulas,
terminal-zero bootstrap, action/shape admission,256-episode / 32-decision limits,
8192-bit numerator/denominator bounds and whole-batch rejection. No weight clipping.
Zero aggregate support produces absent WIS and zero ESS, never a reliability claim.
The immutable envelope always has `reliable=False`; no confidence interval or
statistical inference is reported. All old OPE evidence remains invalid.

Authentic propensities, causal complete trajectories, correct value-function units,
and adequate policy support are caller evidence, not consequences of arithmetic.
No actual replay/learning reward production is certified by this successor. The
accepted domain can reject extreme rational growth; no saturation or clipping is
used to turn rejected results into favorable evidence. Rational bounds constrain
individual arithmetic objects, not end-to-end time or memory under host failure.

## Evidence and proof scope

F-RL-OPE-V2-ARITH extracts actual update expressions and checks 12 SAT-premise /
UNSAT-violation pairs: weights, discounts, IS/PDIS/DR recurrences, supported positive
weights, zero-target weights, aggregate recurrences, rational intermediate size,
ESS accumulation and bounds, final means and guarded WIS. CPython Fraction and
integer semantics are trusted; this is not compiler/runtime verification.

F-RL-OPE-V2-BOUNDARY checks 10 reviewed definitions, 3 checked arithmetic primitives,
3 frozen representations and 1 disabled/versioned public entry. The research effect
inventory expands from 13 to 14 modules; all 11 existing scoped closures require the
new boundary certificate in the same verification run. No order/persistence effect
is added. F-RL-OPE-V2-FLOW checks 23 reachable states, 24 edges, maximum shortest
depth 12 for 2 episodes × 3 decisions. Payload arithmetic is abstracted; source review
and conformance connect it to code, without a whole-language refinement theorem.

F-RL-OPE-V2-CONFORMANCE uses 96 seeded independent prefix-product oracle cases and 96
exact deterministic repeats. Twelve regression tests cover malformed batches,
all-input admission before computation, NaN/infinity, explicit activation, version
rejection, integer/boolean confusion, shape limits, zero support, allocation and
size refusal, immutable results, source mutants, refreshed-registry arithmetic
mutants, partial-publication mutants and UNKNOWN solver rejection.

CE-OPE-WEIGHT-UNDERFLOW preserves the synthetic two-step witness: positive target
probabilities 1e-200 under behavior probability 1 give a binary64 product 0. With
rewards (1,1) and gamma 1, v2 retains the positive exact weight, ESS 1 and WIS 2. This
fixture is not evidence about the frozen deterministic six-step policy domain.
Other extreme tests admit minimum-positive subnormal propensities/targets without
ratio overflow or lost support. The frozen v1 counterexample is deliberately not
removed or reinterpreted as successful financial evidence.

## Performance and reproduction

Local CPython 3.13.3 / macOS x86_64, nonisolated host, synthetic 256 identical episodes ×
32 decisions, gamma 0.99, behavior 0.5 / target 0.25, reward 0.125, zero Q/V:
1.014, 0.947, 1.029 seconds; process peak RSS 15,187,968 bytes; maximum stored rational
size 1648 bits. This measures offline estimator arithmetic, not inference, training,
execution latency or a worst-case wall-clock bound. No model artifact was produced.

```sh
PYTHONPATH=scripts/formal:scripts/research python3 -m unittest test_integrity.ExactOpeTests
bash scripts/verify.sh formal
bash scripts/verify.sh full
```

Use the existing pinned formal toolchain in the [runbook](../../formal/research/README.md).
The public API needs standard-library dependencies only. For a small example (run Python with `PYTHONPATH=scripts/research`):

```python
from ope_rational_v2 import Episode, estimate_v2
record = Episode((1.0,), (1,), (0.5,), (0.5,), (0.0,), (0.0, 0.0))
assert estimate_v2((record,), 1.0) is None
result = estimate_v2((record,), 1.0, enabled=True)
assert result.ordinary_is == 1 and result.effective_sample_size == 1
assert result.reliable is False
```

## Decision and remaining work

**Continue offline research; no candidate passed.** 11 scoped closures / 24 partial /
3 open remain. Obligation 10 remains open because frozen replay/accounting, funding,
Q/CQL and learning arithmetic are not all bounded or checked.37 crash recovery and
38 no permanent lockout are unchanged. All 38 titles/scope/closure criteria and
statuses are unchanged; component evidence is not substituted for broad closure.

No OOS, final-holdout, cost/stress, drawdown, tail-risk, all-seed or OPE acceptance
result is rerun. Existing 108 fits, 19,440 replays and 108 invalid OPE batches remain
contaminated development evidence. The 1,227-return holdout remains sealed and the
prospective embargo remains 2027-01-20T13:00Z. No champion, fleet, live setting, risk
limit, production ownership, exposure, credential, order or deployment is changed.

The pinned reproduction results below supersede the pending verification state. Initial
setup used the system interpreter without Z3 and was corrected to the pinned
formal interpreter. A documentation update initially selected the wrong risk JSON
field; it was corrected to `verificationLimitation` without changing risk status.

The first full local proof reproduction exposed an import-name collision between
legacy and successor `check_ope` functions. The successor import was explicitly
aliased; the failed attempt is not counted as passing verification.

The complete local integrity suite passed:255 tests in200.185 seconds. A subsequent
local full proof attempt failed the existing process-bridge timing-sensitive
requirement (`ValueError: PPO process bridge: no actual inference for trained
policy`) on the shared host. No timeout or risk limit was weakened; pinned CI must
reproduce that obligation before merge. This local failure is not a passing gate.


## Pinned verification receipt

Run [37406072370](https://github.com/diegueins680/trader/actions/runs/37406072370),
job112083820502, tested head491e5385f926e6f9ffc1b42d073e9fadbe540114:

- Receipt reproduction passed (49 seconds by job step timestamps).
- `bash scripts/verify.sh formal` passed (86 seconds);255 integrity tests in38.869 seconds.
- `bash scripts/verify.sh full` passed (301 seconds);255 integrity tests in39.388 seconds,
  Haskell build/format/lint/smoke/tests,241 web tests/build and185 automation tests passed.
- All77 SMT requirement groups reproduced, including the12 new OPE queries.
- Receipt SHA256: `92155b72818667a67fe52278f05cf470b48d85862fcbfffa87601d30bf051ed0`.
  Imported verbatim from the successful pinned job. Every source hash matches.
  Changed receipt sections only:exactOpeV2,sourceHashes,smt,promotionBoundary,
  shieldConsumers,championArchive. Previous non-surface model/SMT/conformance results
  in those boundary sections are unchanged. Broad closure counts are unchanged.
- Ordinary CI37406072351 on the pre-receipt revision was canceled because it had
  the prior committed receipt; it is not counted as a pass. The temporary reproduction
  workflow is removed from the delivered tree. Final-head CI is reported in PR310.

No verification limit was weakened, no new dependency added, and no proof placeholder
introduced. Passing these scoped checks does not make the research mission complete.

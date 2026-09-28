# Isolated GAE target v2 continuation — 2026-09-28

**No adoption or whole-learner correction.** This continuation introduces a small,
explicitly enabled raw-target kernel, disconnected from every existing training
and production caller. The frozen learner still has CE-RL-010/011. Historical
results, policies, artifacts, champion, financial registrations and holdouts are
unchanged. No market data, financial fit, OPE run or final holdout is accessed.

Latest main remains `dbd45e26`; branch `research/sequential-review-2026-09-28`, draft
#284 remains stacked on #281. Specification and engineering registration were
committed at `ea02097e` before kernel implementation. There are no new dependencies,
production services, CLI/API changes, saved-artifact changes or environment flags.

## Design and consistency resolution

The [contract](../../formal/research/target-v2-contract.md) and [engineering
registration](../registrations/gae-targets-v2-engineering.json) define
`gae-targets-v2`. Its defaults return `None`; activation requires exact Boolean
`True` and the exact native version string. This is a function argument for an
isolated research helper, not deployment authorization or a new policy flag.

[GAE](https://arxiv.org/abs/1506.02438) provides the advantage-estimation framework;
[PPO](https://arxiv.org/abs/1707.06347) is the existing screened policy family.
Primary sources were rechecked on 2026-09-28. Neither supplies a numerical certificate
for this custom implementation or financial efficacy. The prior focused paper
matrix remains applicable; this is not a new model-family selection.

For a terminal row the kernel computes raw advantage `reward-value` and assigns
target directly from `reward`. For a continuing row it uses the registered GAE
recurrence with lambda .95 and supplied discount [0,1], preserving separate
binary64 operations. Every input must have the exact native type and be finite.
Every returned pair must be finite. Overflow rejects the call rather than turning
it into a finite-looking reward or clipping it. Signed terminal reward zero is
preserved bit-for-bit.

The batch API takes immutable native tuples with 1..256 rows, walks backward from
zero carry and publishes a tuple only when every row succeeds. A late failure
returns no batch; local staging never mutates inputs or external state. Disabling
returns absence with no persisted state. The module imports only standard math
and annotations, and the source skeleton rejects added effects or unsupported
control flow. Stable runtime bindings and no monkey patching remain assumptions.

This deliberately excludes normalization, gradients, optimizer updates and training
integration. Returning raw values prevents them from being mistaken for the old
normalized-advantage API. No existing name is aliased or replaced. The earlier
contract's prohibition on silently changing frozen targets still holds: the new
engineering registration authorizes zero financial fits. Any training integration
requires a separately identified learner and financial registration, with adaptive
trial accounting and unchanged acceptance gates.

## Counterexamples and conformance

- CE-RL-010: the frozen learner still returns target 0 for reward 1 and critic
  1e16. The isolated v2 kernel returns target 1 exactly.
- CE-RL-011: v2 rejects the complete overflow fixture with `None`, including any
  already staged later rows. The original learner's NaN/infinity witness remains
  reproducible and is not reclassified as fixed there.

The [counterexample disposition](../../formal/research/target-v2-counterexamples.json)
links both requirements to the original exact-hex inputs and deterministic tests.
These are synthetic arithmetic findings, not evidence of historical prevalence,
profitability, cost resilience or successful retraining.

Eight added tests cover source mutants, witness removal, old/new counterexample
behavior, exact native types, wrong versions/enabling controls, hostile objects,
signed zero, subnormals, maximal floats, overflow, 256 rows and all 256 failure
positions. A 486-case scalar grid compares to independent exact rational GAE
expressions with absolute tolerance 1e-12; this tolerance is regression evidence,
not a proved error bound. Sixty-four seeded batches check deterministic replay,
finite publication and terminal separation from changed later rows. The seed is
20260928 for engineering fixtures only; registered financial seeds remain unchanged.

## Verification and scope

Three new SMT requirements add four SAT-premise/UNSAT-violation queries:
`F-RL-TARGET-V2-TERMINAL`, `F-RL-TARGET-V2-FINITE` and `F-RL-TARGET-V2-REAL`.
Exact-real recurrence, binary64 terminal bit preservation and finite/default-disabled
admission are separate claims. Full source skeletons bind the arithmetic slots,
native guards, version/default controls and batch publication order. The translator,
Python primitives/compiler and solver are trusted; this is not a general interpreter
proof. No nonterminal roundoff bound, normalized-advantage certificate or full
learner correction is supplied.

An initial generic floating-point premise check produced no certificate
(`unsatisfied/unknown premise`). The checker now uses a concrete valid witness for
non-vacuity, then removes all witness constraints before the universal violation
query. A dedicated regression catches accidental witness leakage. No theorem domain
or 10-second solver limit was weakened.

`F-RL-TARGET-V2-PUBLISH` explores **33,410 states and 66,562 transitions** across
batch sizes 1..256. Maximum shortest depth and progress bound are **258**. Every
nonterminal transition strictly decreases an integer rank; terminal states stutter.
Termination assumes terminating primitives and ordinary resource availability; this
is not a wall-clock deadline. The model has no authorization transition. Source
skeleton checks plus implementation regressions provide scoped conformance, not
universal Python or production-system refinement. Existing lifecycle/artifact models
remain separate and unchanged.

The scoped SMT total becomes **22** and the integrity suite **49 tests**. The
canonical registry, proof ledger, source lock, risk register and CI map the new
kernel/checker to their requirements. All 38 broader obligations remain open or
partially verified. `RL-OFFLINE-001` stays HIGH/OPEN. Both original learner defects
remain blockers for that learner; no economic acceptance gate is weakened.

## Synthetic kernel benchmark

Darwin x86_64, Python 3.13.3; 100 warmups and 1,000 timed calls per case. Inputs
are fixed synthetic tuples, not market observations. Measurements include the batch
wrapper; they exclude normalization, neural inference, training and deployment.

| Case | Median ms | p99 ms | Maximum ms |
| --- | ---: | ---: | ---: |
| One terminal | 0.003373 | 0.034864 | 0.207923 |
| 256 terminal rows | 0.589336 | 1.198564 | 1.354084 |
| 256 continuing rows | 0.684198 | 1.485787 | 2.438493 |
| Failure after 255 staged rows | 0.617400 | 1.118238 | 7.134481 |
| Default disabled | 0.000333 | 0.000437 | 0.000779 |

A separate 20-call tracemalloc probe measured peak tracked Python allocation of
16,680 bytes. This excludes interpreter/native allocations and is not process RSS.
Shared-host timing is noisy; no hard timeout or production inference budget is
certified. The failed path's measured maximum is retained, not discarded.

Reproduce from the repository root with the pinned formal Python environment:

```python
import hashlib,json,math,platform,statistics,sys,time,tracemalloc
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts/research'))
from gae_targets_v2 import batch_v2
valid=((.001,.1,.1,False),)*256
terminal=((.25,0.,0.,True),)*256
bad=((float('inf'),0.,0.,True),)+valid[1:]
scenarios={'one_terminal':lambda:batch_v2(terminal[:1],.99,enabled=True),
 '256_terminal':lambda:batch_v2(terminal,.99,enabled=True),
 '256_continuing':lambda:batch_v2(valid,.99,enabled=True),
 'late_failure':lambda:batch_v2(bad,.99,enabled=True),
 'disabled':lambda:batch_v2(valid,.99)}
results={}
for name,call in scenarios.items():
    for _ in range(100):call()
    timings=[]
    for _ in range(1000):
        start=time.perf_counter_ns();result=call();timings.append((time.perf_counter_ns()-start)/1e6)
    timings.sort()
    assert (result is None)==(name in ('late_failure','disabled'))
    results[name]={'medianMs':statistics.median(timings),'p99Ms':timings[989],'maximumMs':max(timings)}
tracemalloc.start()
for _ in range(20):batch_v2(valid,.99,enabled=True)
_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
print(json.dumps({'schemaVersion':1,'version':'gae-targets-v2','python':sys.version.split()[0],
 'platform':platform.system()+' '+platform.machine(),'samples':1000,'warmup':100,
 'sourceSha256':hashlib.sha256(Path('scripts/research/gae_targets_v2.py').read_bytes()).hexdigest(),
 'results':results,'tracemallocPeakBytes':peak,
 'scope':'synthetic microbenchmark only; no policy inference, training or deadline guarantee'},indent=2))
```

## Verification receipt

Implementation is frozen at `1f2047eb`. The formal wrapper passed: 49 tests
(30.172 seconds), 22 scoped SMT requirements and all scoped state/conformance
checks; the verifier reported 20.267 seconds. The required full wrapper is pending;
no full-verification claim yet. The initial
combined `verify.py --record` attempt failed the existing
`F-RL-TERMINAL-FP-MASK` check with `violating terminal claim: canceled` under the
pinned timeout. An unchanged retry passed in 21.060 seconds. The failed log is
preserved separately; no limit, domain, requirement or assertion was relaxed.


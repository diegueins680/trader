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

Implementation is frozen at `1f2047eb`; report revision `2b11d799`; specification
`ea02097e` precedes both. Subsequent receipt changes are documentation only.
Commands from the isolated worktree:

```bash
export PATH=/private/tmp/node-v20.19.0-darwin-x64/bin:$PATH
export TRADER_FORMAL_PYTHON=/private/tmp/trader-proof-20260928/bin/python
"$TRADER_FORMAL_PYTHON" scripts/formal/test_integrity.py TargetV2Tests
bash scripts/verify.sh formal
bash scripts/verify.sh full
bash scripts/verify.sh automation  # unchanged retry after full-run deadline failures
"$TRADER_FORMAL_PYTHON" scripts/formal/verify.py --require-complete
```

Pinned tools: Python 3.13.3, NumPy 2.3.5, z3-solver 4.15.4.0 (solver 4.15.4),
GHC 9.4.8, Cabal 3.12.1.0, fourmolu 0.15.0.0, hlint 3.8 and Node 20.19.0.

- Formal wrapper: **exit 0**, 49 tests (30.172 seconds), 22 scoped SMT requirements
  and all scoped state/conformance checks; verifier time 20.267 seconds.
- Full wrapper: **exit 1**. Formal, Haskell build/format/lint/smoke/tests and web
  typecheck/241 tests/build passed. Automation passed 182/185, failed three,
  none skipped, in 150.249 seconds. Exact failures:
  `research-datafeed-scheduler.test.mjs` receipt subprocess `TimeoutExpired`
  after 10 seconds; `research-edge-campaign.test.mjs` returned null status at
  its 60-second subprocess deadline; `sequential-screen.test.mjs` returned null
  status at its 120-second deadline. The scoped verifier reported 18.782 seconds.
- Unchanged `bash scripts/verify.sh automation` retry: **exit 1**, 183/185 passed,
  two failed, none skipped, in 121.255 seconds. The scheduler returned null status
  at its outer 30-second deadline and the edge campaign at 60 seconds; the
  sequential-contract suite passed on this retry. No test, timeout, environment
  concurrency setting, assertion or gate changed between attempts.
- Environmental evidence: one-minute host load averages were 236.804 and 271.855
  on 16 logical CPUs around these failures. This supports resource contention as
  an explanation, not a proof of root cause or a substitute for a passing local
  full command. Unrelated workloads were not stopped. At that receipt, local full
  verification remained an unresolved delivery limitation. The continuation below records
  subsequent attempts without replacing these failures. The PR is not ready for
  candidate integration, and the 38 broader obligations retain their open/partial status.
- Remote [CI run 36450058964](https://github.com/diegueins680/trader/actions/runs/36450058964)
  at `2b11d799`: formal, Haskell, web and automation passed. Docker build and
  deployment were skipped.
- Acceptance diagnostic: expected **exit 1**, `ValueError: research acceptance
  blocked by open obligations`, after reproducing the scoped certificates. All
  38 broader obligations remain open/partial.
- Existing separate models/conformance remain unchanged: lifecycle 75 states,
  349 transitions, two callers, depth 6; artifact path 27 states/40 transitions,
  depth 13; 20,480 Haskell conformance cases and 180 rational replay traces.
 The initial
combined `verify.py --record` attempt failed the existing
`F-RL-TERMINAL-FP-MASK` check with `violating terminal claim: canceled` under the
pinned timeout. An unchanged retry passed in 21.060 seconds. The failed log is
preserved separately; no limit, domain, requirement or assertion was relaxed.


Logs and benchmark output remain outside Git. SHA-256 receipts, prefix
`/private/tmp/trader-targetv2-`, suffix `-20260928.log` except benchmark `.json`:

| Artifact | SHA-256 |
| --- | --- |
| record (failed) | `e3aadb71399a5b933d8ce837d03409b592b9c01796a4660a5341441e446a1fe5` |
| record-retry | `d16c4066a67bb3a44597f918d2fa65e792053a3fdd591936ebb0df6861a308a8` |
| formal | `0249da047be84ec18318df5d698b6f1dcbe251eb4218c434147c3b14556ff587` |
| acceptance | `924e91e4939fe545e6f74ace1f4422bef5ed43c12deaa10d61f6e562262c907c` |
| benchmark | `d0ac9c51c31ecc57391f402b7524bfd57e8631fda09a2ebcfa6a225c05b3d792` |
| full (failed) | `7469c8dbd69c898247becdb01a8db0e90d241456c111cd7b9ce37332e294911a` |
| automation-retry (failed) | `7965f8f536fb8100f5b193931d0688adad935fa27a40367a9641080af6f894b4` |

## Verification continuation at `46f815e4`

No source, proof, test, timeout, tool version, Node worker scheduling or acceptance
gate changed. All 42 source hashes in the formal toolchain manifest still match.
An unchanged automation rerun again returned **exit 1**: 183/185 passed, two failed,
none skipped, in 107.765 seconds. Scheduler and edge-campaign subprocesses exceeded
their existing 30/60-second deadlines; sequential-contract tests passed.

A process-local resource experiment then capped numerical-library threads before
Python/NumPy initialization. NumPy reports Apple Accelerate as its BLAS/LAPACK
backend. Apple's installed `/usr/share/man/man7/Accelerate.7` documents
`VECLIB_MAXIMUM_THREADS` for controlling internal threading and avoiding contention.
The OpenBLAS/OpenMP variables also bound compatible libraries if loaded; this is
not evidence that those backends were used. Nothing was installed or written to
shell startup, deployment configuration or live trading environment variables.

```bash
PATH=/private/tmp/node-v20.19.0-darwin-x64/bin:$PATH \
TRADER_FORMAL_PYTHON=/private/tmp/trader-proof-20260928/bin/python \
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
bash scripts/verify.sh automation
# The same command prefix was used for the formal and full targets below.
```

- Automation with this prefix: **exit 0**, 185/185 passed, none skipped, in 73.975
  seconds. Scheduler took 26.439 seconds against its unchanged 30-second deadline.
  This single pass does not establish a causal fix or adequate timing headroom.
- Standalone formal with this prefix: **exit 1**, 49 tests in 87.100 seconds,
  one assertion failure and two errors. `F-RL-TARGET-V2-FINITE` returned
  `premise witness failed: unknown`; `F-RL-TERMINAL-FP-MASK` returned
  `violating terminal claim: canceled`. The latter also prevented a receipt-tamper
  regression from reaching its expected rejection. The pinned 10-second solver
  timeout was not increased. Unknown/canceled checks supply neither a certificate
  nor a counterexample and are not marked as passes.
- Remote [CI run 36453692641](https://github.com/diegueins680/trader/actions/runs/36453692641)
  at `46f815e4` passed formal, Haskell, web and automation; Docker/deployment were
  skipped. This is separate evidence, not a replacement for local failures.

The full wrapper with the same prefix returned **exit 0**. Its formal stage passed
49 tests in 42.356 seconds and all 22 scoped SMT requirements, bounded models and
conformance checks (verifier 30.224 seconds). Haskell build/format/lint/smoke/tests,
web typecheck/241 tests/build and all 185 automation tests passed, none skipped.
Automation took 54.769 seconds; the scheduler fixture took 17.955 seconds. This
supplies the previously missing local full-run pass for unchanged executable source.
It does not erase the earlier failures or certify reproducible timing headroom.

An unchanged standalone `bash scripts/verify.sh formal` retry with the same prefix
also returned **exit 0**: 49 tests in 39.231 seconds, all scoped checks and verifier
19.016 seconds. The previous standalone failure remains a separate receipt.

Host load varied during these attempts (including 61.511 on 16 logical CPUs around
formal failures), so thread caps and host availability are confounded. No unrelated workload was stopped.
The 38 broader obligations remain open/partial; no proof ledger status is promoted.
The acceptance command, using the same process prefix and pinned Python,
`python scripts/formal/verify.py --require-complete`, reproduced the scoped
certificates in 19.938 seconds, then returned expected **exit 1**:
`ValueError: research acceptance blocked by open obligations` (38 open/partial).
This is an acceptance refusal, not a successful whole-mission verification.

Continuation logs use prefix `/private/tmp/trader-targetv2-`, suffix
`-20260928.log` (CI receipt uses `.json`):

| Artifact | SHA-256 |
| --- | --- |
| resume-automation (failed) | `7ae818b6f5f62a0fadcce4a62270cee77941e19762f042f752bf2de680045296` |
| resume-automation-threadcap | `b107af5daf0cc5c7cb976055835cec0518d6822f648e8cae433480eb79465691` |
| resume-formal (failed) | `454decb7d45252c22d0c710980a58f2c75c4855f413e54df3a381f29611a11a8` |
| resume-formal-retry | `9d9de355450905cd2ae8f5c9e9cb921c87cf0c2f6c9c1733d6260158bfb0af57` |
| resume-acceptance (expected refusal) | `f9b84b3685d3c0b5303ef0088abf6ede0218bb3bbe285d54d363b56d4d13f482` |
| resume-ci | `90a0722171216f2d4523ac92505df0bc7ab474f25e38a1ea3dcaea246c180004` |
| resume-full | `1dd4ea1a42580b12f2d5eab932b5e30f40d8689dd1cde7c8028d954219ec0af5` |

No live authorization, order, authenticated trading experiment, live exploration,
holdout access, merge, deployment or champion change occurred. No proof placeholder
was introduced. No promotion, normalization, whole-learner or economic completion
claim follows from these scoped results. Recommendation: retain the isolated
engineering kernel for continued offline research; adopt no trading candidate.

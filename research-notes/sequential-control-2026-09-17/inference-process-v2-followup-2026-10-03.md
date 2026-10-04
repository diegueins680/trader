# Inference process v2: cancellation engineering follow-up

The separate Haskell process boundary can terminate a hung synthetic worker and
reject late replies. It is default-disabled, has no order or artifact-loading
interface and does not replace the frozen learner. **Continue offline engineering;
do not adopt a trading candidate.** This change supplies missing implementation
evidence for obligations 21/36; it does not close either broader obligation.

The [preregistration](../registrations/inference-process-v2-engineering.json)
was committed in `5e8f332c`, before implementation. Amendment `c8baa686` separates
initialization from request admission after cold startup exceeded 20 ms. The
[canonical contract](../../formal/research/inference-process-v2-contract.md) names
all timing and OS assumptions. `d13b07a4` preserves the first pre-initialized
implementation and its checked but economically irrelevant engineering evidence.
The source stays based on main `dbd45e2691cb37f1421a23306c43b676fa82e6fc`.

## Problem and corrected behavior

CE-RL-017 is still a valid counterexample for `sequential_learning.infer`: its
elapsed-time test runs after the forward call, so it cannot stop a nonreturning
call. No legacy identifier or saved artifact acquires new semantics here.

The new executable initializes its own trusted child before reading any request.
It allows one bounded synthetic 12–16–3 network request, with a 20 ms budget that
includes parsing, transfer, computation, cleanup and final admission. Initialization
has a separate 1 s ready wait. The child gets an empty environment and fixed
arguments; no shell, arbitrary command, descendant, exchange client or production
component is involved. Only quarter-exposure proposal codes can leave the parent.

Every session cleans up: TERM, up to 50 ms exit polling, KILL if needed, then up to
50 ms further polling. Failed cleanup rejects. Successful cleanup is followed by
another monotonic deadline check. Each process accepts at most one request, so
late output has no successor request to contaminate. Repeated async cancellation,
parent process death, a hostile binary and unbounded OS primitives are outside
the guarantee. Proposal delivery/consumer freshness and real-market observation
availability are not certified by this fixture.

## Engineering experiment register

These are engineering variants, not financial model trials or OOS evidence:

| Configuration | Result | Disposition |
|---|---|---|
| Cold launch included in 20 ms, default RTS tick | Five simple calls rejected; cold launch alone exceeded budget | Reject timing design |
| Cold launch included in 20 ms, 1 ms RTS tick | Five simple calls still rejected; three instrumented calls also rejected | Reject; tuning did not fix architecture |
| Initialize first, 1 ms tick, unbuffered text | Simple fixtures worked, but all 15 dense requests expired | Preserve failure; reject transfer implementation |
| Initialize first, bounded bulk bytes | 15/15 dense requests accepted; unchanged 20 ms guard | Retain offline engineering fixture only |

The intermediate bulk implementation was also measured (15/15 accepted) before
adding an explicit serialized-size guard. All three full benchmark runs, every
seed and every individual timing are in the [receipt](inference-process-v2-benchmarks-2026-10-03.json).
No seed or failed configuration was dropped. These checks do not establish an
advantage over a champion, a policy value, market significance or production readiness.

Final measurement: macOS 14.7.7, Intel, GHC 9.4.8, CPython 3.13.3, NumPy 2.3.5;
15 synthetic calls, seeds 11/23/47, five repeats each, nonisolated host:

| Quantity | Median | Maximum |
|---|---:|---:|
| Child startup to Ready | 13.802 ms | 418.533 ms |
| Direct worker request-to-reply | 3.854 ms | 5.306 ms |
| Direct worker exit/reap | 7.694 ms | 8.827 ms |
| Complete supervisor command, including startup | 48.471 ms | 55.448 ms |

The compiled executable was 2,747,416 bytes; macOS `/usr/bin/time -l` reported
8,400,896 bytes maximum resident set on one supervisor call. This is neither an
aggregate concurrent-worker memory bound nor a production capacity benchmark.
The worker-only measurement excludes parent parsing, IPC and cleanup. Admission
of all 15 final calls is separately checked by the parent; no unconditional
wall-clock claim follows from these measurements.

## Evidence and exact scope

| Requirement | Status and artifact | Implementation connection |
|---|---|---|
| F-RL-PROCESS-LIFECYCLE | `model_checked`: 57 states, 143 edges, depth 8, rank 11 | Locked parent, exchange and cleanup bodies; compiled fault tests |
| F-RL-PROCESS-ADMISSION | `smt_verified`: satisfiable premise, UNSAT unsafe admission | Exact integer guard and versioned finite reply domain |
| F-RL-PROCESS-ISOLATION | `exhaustively_checked`: four dispatch cases, one self launch | Whole-source/import lock, no production source root/reference, default absence |
| F-RL-PROCESS-CONFORMANCE | `property_tested` | 226 compiled guard cases, 12 networks, ten simple attempts, seven invalid inputs, held-open input, three defaults, eight process faults |

The model has one child, one request, no retries, three saturated request-time
buckets and two abstract poll steps per window. Bounded progress relies on named
OS/scheduler assumptions; it is not a proof that a general-purpose OS meets a hard
deadline. Z3 covers unbounded integer timestamps and admitted elapsed [0,20 ms),
not the physical clock. There is no neural-network arithmetic or compiler theorem.

Faults include sleeping and CPU-bound hangs, ignored TERM, late/invalid output,
EOF, startup hang and explicit cleanup failure. The test harness verifies that the
fault worker PID no longer exists after cleanup. The source-based IO refinement
relation is supported by checked control bodies and compiled tests, not a complete
machine-checked semantics of Haskell IO. No proof placeholders are introduced.

[Proof ledger](../../formal/research/proof-ledger.json),
[source contract](../../formal/research/inference-process-source.json),
[checker/tests](../../scripts/formal/inference_process.py),
[Haskell implementation](../../haskell/research/InferenceProcessV2.hs) and
[runbook](../../formal/research/README.md#offline-inference-process-v2) provide both
traceability directions. Formal specifications and RL-OFFLINE-001 now name the
process boundary and A-INFERENCE-PROCESS. That risk remains HIGH/OPEN.

## Acceptance and remaining work

Three of 38 broader obligations remain closed for their explicit affected offline
scope (11,24,31), with the new isolation certificate added to their sufficiency
sets. Twenty-eight remain partial and seven open. The new process evidence does
not silently narrow their original criteria. Obligations 21/36 still require
composition into a separately registered successor runner and verification of
inherited server draining/shutdown. Real availability admission and trusted
artifact handoff also remain necessary before any policy loader uses this boundary.

No market dataset, protected holdout, funding experiment or OPE campaign was run.
There are no new OOS returns, cost results, drawdown/tail statistics or champion
comparisons. The prior economic and reproducible-delivery acceptance gates remain
open. No new model/policy is integrated, no live flag or fleet setting changes,
no live exploration occurs, and the work stays unmerged and undeployed.

At source freeze `bb647b03067fe83aebbe9c420fc58354bca8cbff`, both
`bash scripts/verify.sh formal` and `bash scripts/verify.sh full` exited 0.
The latter includes Haskell build/format/lint/smoke/tests, 241 web tests and
185 automation tests. The formal gate includes 153 integrity tests, 45 SMT
obligations and all scoped models/conformance checks. The explicit
`python scripts/formal/verify.py --require-complete` command exited 1 with
`research acceptance blocked by open obligations or research evidence gates`.
This is the expected non-acceptance result, not an ignored test failure.

The [verification receipt](inference-process-v2-evidence-2026-10-03.json) records
source hashes, commands, log hashes, tool versions and the observed CI snapshot.
Linux CI reproduced the process proof/fault suite; its full job status is linked
from [draft PR #286](https://github.com/diegueins680/trader/pull/286), stacked on
#284. A passing scoped gate cannot make this draft ready while critical broader
obligations remain unresolved. The next production audit targets are
`runServeShutdown` in Main.hs and the deadline/worker-registry operations in
`Trader.App.GracefulShutdown`; they are not covered by the new child-process proof.

## Benchmark reproduction

Run this from the repository root with the pinned proof Python/NumPy environment,
`VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. It uses only
synthetic values and temporary binaries. Timing values are observations, not golden
expected hashes. For the failed text variant, replace the source assignment with
`source = subprocess.check_output(['git', 'show', 'd13b07a4:' + p.SOURCE], text=True)`.
The intermediate bulk variant can be reconstructed from the final source by
replacing its explicit payload-length branch with
`BS.hPutStrLn input (BS.pack (show (request :: Request)))`, followed by the same
flush/read operations. All input seeds and iterations stay identical.

```python
import json,subprocess,time,statistics,platform,hashlib,sys,tempfile,re
from pathlib import Path
sys.path.insert(0,'scripts/formal')
import inference_process as p
import numpy as np
source=(p.ROOT/p.SOURCE).read_text()
rows=[]
with tempfile.TemporaryDirectory(prefix='trader-process-benchmark-') as tmp:
 exe=p.compile_source(source,Path(tmp)/'build');size=exe.stat().st_size
 for seed in (11,23,47):
  rng=np.random.default_rng(seed);request=repr(((rng.normal(size=12)*.1).tolist(),(rng.normal(size=259)*.1).tolist()))+'\n'
  for trial in range(5):
   a=time.perf_counter_ns()
   child=subprocess.Popen([str(exe),'--offline-inference-v2','--worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
   assert child.stdout.readline()=='IPV2 ready\n'
   b=time.perf_counter_ns();child.stdin.write(request);child.stdin.flush();reply=child.stdout.readline().strip();c=time.perf_counter_ns();child.wait(timeout=3);d=time.perf_counter_ns();child.stdin.close();child.stdout.close()
   e=time.perf_counter_ns();answer=p.invoke(exe,['--offline-inference-v2'],request);f=time.perf_counter_ns()
   rows.append(dict(seed=seed,trial=trial,requestCharacters=len(request),startupNS=b-a,workerRequestReplyNS=c-b,workerReapNS=d-c,supervisorTotalNS=f-e,reply=reply,result=answer))
 time_result=subprocess.run(['/usr/bin/time','-l',str(exe),'--offline-inference-v2'],input=request,text=True,capture_output=True,timeout=3,check=True) if sys.platform == 'darwin' else None
 rss=re.search(r'(\d+)\s+maximum resident set size',time_result.stderr) if time_result else None
receipt={'schemaVersion':1,'source':p.SOURCE,'sourceSha256':hashlib.sha256(source.encode()).hexdigest(),'platform':platform.platform(),'python':platform.python_version(),'numpy':np.__version__,'binaryBytes':size,'maximumResidentSetBytesMacOS':int(rss.group(1)) if rss else None,'rows':rows,'accepted':sum(x['result'].startswith('(QuarterTarget') for x in rows),'scope':'15 synthetic calls only; nonisolated host. Worker timing excludes parent IPC, validation and cleanup; supervisor total includes startup. No hard real-time or production capacity guarantee.'}
receipt['summaryMilliseconds']={key:{'median':statistics.median(x[key]/1e6 for x in rows),'max':max(x[key]/1e6 for x in rows)} for key in ('startupNS','workerRequestReplyNS','workerReapNS','supervisorTotalNS')}
Path('/tmp/trader-inference-benchmark.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='rows'},indent=2))
```

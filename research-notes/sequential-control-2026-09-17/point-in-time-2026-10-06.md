# Witnessed point-in-time entry: engineering result

Base main75571dd13991134952bd625ae6da5611ba9185ad. Preregistration commits
d19b229e and2e790067 preceded implementation. This is an unfrozen research entry,
not a new financial experiment or a reinterpretation of frozen screen evidence.

## Change and interpretation

`point_in_time_v3.train_point_in_time_v3` takes immutable `Record` values, explicit
symbol/close/decision tuples, a processing delay, and the existing PPO training
parameters. Both public entries default disabled and reject unknown versions.
The [contract](../../formal/research/point-in-time-contract.md) specifies exact
UTC microseconds and each record's release, first-seen, collection and revision
witnesses. `admit_v3` chooses the unique highest revision available at that bar's
own decision. It never changes an older bar when a later correction arrives.
Unknown headers, missing slots, duplicate greatest visible revisions and invalid
selected values reject the whole batch; missing funding is never zero-filled.
Unavailable payloads are not inspected. Invalid future *headers* still reject:
metadata must be intelligible before availability can be decided.

This conservative frozen-vintage interpretation has explicit limitations. It does
not rebuild historical features using the latest revisions known at each later
decision. It can refuse late public bars. The caller specifies intended decision
times; no actual fill or collection-clock accuracy follows from admission. Release
may be absent, using mandatory first-seen and collection witnesses; a positive
revision must have a revision-release timestamp. Correct-looking forged witnesses
cannot be detected here. No timestamp was invented for the frozen CSV archive.

The accepted bytes are immutable and training receives independent arrays. A v3
result retains the exact grid and chosen witnesses with the v2 immutable result.
No persistence or v4 artifact-loader consumer accepts the v3 envelope implicitly.
No Python production service, dependency, configuration, live flag or deployment
change is introduced. Ordinary v1/v2 interfaces preserve their semantics.

## Formal and implementation evidence

F-RL-PIT-TIME: seven independent satisfiable-premise/UNSAT pairs establish integer
availability ordering, signed-63-bit published bounds and maximum-revision/tie/
unavailable-record transfer. Python intermediate integers do not overflow;
values whose availability exceeds a bounded decision cannot be selected.

F-RL-PIT-FLOW:126 reachable states,470 directed transitions, maximum shortest
path6, two slots and three ordered revision ranks, explored to a fixed point.
Repeated scans have no retry-depth cutoff. Only completely admitted slots reach
training/publication; disabled and failed calls publish absence. This is a finite
control abstraction, not a compiler or scheduler/liveness proof.

F-RL-PIT-BOUNDARY: complete ten-definition source review, actual guard/selection/
publication extraction, two default gates and one dominated training call. The
research effect/import roster grows from12 to13 modules. All11 existing scoped
closures require this new boundary certificate in the same verification run.
Original38 obligation titles, scope and criteria are unchanged.

F-RL-PIT-CONFORMANCE:64 seeded differential record cases, five unavailable-payload
cases and one actual registered synthetic PPO training step (seed31,121 constant
bars,one symbol). Additional integrity tests cover malformed timestamps, overflow,
non-finite values, missingness, symbol isolation, immutable private handoff,
training failures, source/model mutations and omission of a required certificate.
Property tests and the synthetic learner execution are engineering evidence,
not formal proof or profitability evidence. An initial test setup missed a local
`sys` import; it failed, was corrected, and is not counted as a passing run.

A-PIT-WITNESS records trusted caller timing/revision truth and Python/NumPy
semantics. Source-to-model correspondence is reviewed and tested, not universal
language refinement. No universal numeric/reward/accounting theorem is claimed.
No implementation counterexample was found in this scoped change; deliberate
bypass/availability mutants are verifier regressions, not historical data findings.

## Decision and outstanding work

Obligation4 becomes **partially_verified**, not closed. Current totals:
**11 scoped closures,24 partial,3 open**;27 remain unresolved. No candidate passes.
Timestamp authenticity, frozen-data gaps and inherited ingestion remain blockers.
Numeric obligations10, durable recovery37 and progress/isolation38 remain open.
The v3 entry must remain offline until separate empirical and operational gates
are satisfied; the present change provides no admission to shadow/paper/live use.

Financial evidence is unchanged:108 fits,19,440 replays,19,548 registry rows are
contaminated development; all108 OPE batches invalid;1,227 final returns sealed;
prospective embargo2027-01-20T13:00Z. No market data or final holdout was read.
No new OOS returns, costs, drawdown, tail risk, OPE, inference benchmark or economic
performance is claimed. Recommendation: no adoption; continue offline research.

## Reproduction and verification

Use the existing pinned toolchain and wrappers:

```sh
bash scripts/verify.sh formal
bash scripts/verify.sh full
```

No network is required after dependencies are installed. The existing proof
receipt, ledger and source locks include this entry. Verification run evidence is
recorded below after actual completion; an unrun check is not a pass.

Local targeted verification:20 tests passed in5.822s (PointInTimeTests and
IntegrityTests). The specification coverage check passed:40 specs,353 named
features,483 clauses,599 implementation files,256 evidence links,34 risks.
No proof placeholders found. Full pinned wrapper results remain pending.

Admission-only maximum-size benchmark on the non-isolated local macOS host,
CPython3.13.3/NumPy2.3.5:262,144 records,8 symbols,4,096 bars,4 revisions per
price/funding slot;65,536 selected witnesses. Three runs took2.222s, 2.401s, 2.480s.
Peak process RSS (including inputs and interpreter, not an allocation delta) was
121,991,168 bytes; selected price/funding buffers total524,288 bytes. This is not
a training/inference SLA or deployment-resource benchmark. Reproduce from repo root:

```python
import json,resource,sys,time
sys.path.insert(0,'scripts/research')
import point_in_time_v3 as p
symbols=tuple('S'+str(i) for i in range(8));closes=tuple(1000*i for i in range(4096));decisions=tuple(c+100 for c in closes)
records=tuple(p.Record(s,k,c,r,c+10+r,c+20+r,c+30+r,c+10+r if r else None,100. if k=='price' else 0.) for s in symbols for k in ('price','funding') for c in closes for r in range(4))
measure=[]
for _ in range(3):
 start=time.perf_counter();result=p.admit_v3(records,symbols,closes,decisions,5,enabled=True);measure.append(time.perf_counter()-start)
 assert result is not None and len(result.witnesses)==65536
 assert all(w.revision==3 for w in result.witnesses)
assert len(records)==p.MAX_RECORDS
print(json.dumps({'scope':'synthetic maximum admission only; no training/inference/market','python':sys.version.split()[0],'records':len(records),'symbols':8,'bars':4096,'selectedWitnesses':len(result.witnesses),'seconds':measure,'processPeakRssNativeUnits':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'priceFundingBytes':sum(map(len,result.prices+result.funding))},indent=2))
```

`ru_maxrss` is bytes on macOS and KiB on Linux; convert before comparing platforms.

First pinned reproduction run37400457978 failed before either wrapper: the
shield-consumer composition still required12 modules after the reviewed roster
expanded to13. The explicit count is corrected to13, and a full composed-checker
regression now accepts13 and rejects a12-module receipt. This was a fail-closed
coverage mismatch, not a trading/data counterexample. The failed run is retained
as failed evidence; no result from it is counted as a passing wrapper.

Second pinned reproduction run37401000171 failed before either wrapper because
the archive-preservation certificate retained the two pre-extension promotion
source hashes. Both reviewed dependency hashes are refreshed. An audit of all
current formal source-registry path hashes found no other stale entries. Both
failed runs remain failures; source checks were not disabled or relaxed.

Complete local `scripts/formal/verify.py --record` passed in181.139s using the
pinned Python/NumPy/Z3/GHC toolchain, reproducing76 SMT requirement groups and
all composed models. Overall completion remains false with27 unresolved
obligations. The canonical formal/full wrappers still require pinned CI success.

On0569dcb0, pinned proof reproduction passed, but the formal integrity suite
failed one of243 tests: ChampionArchiveTests supplied the old12-module count
while expecting the later production-root rejection. Its fixture is corrected
to13 so it again tests the intended missing-root gate. The expected rejection
is preserved; the full integrity suite is rerun before another CI attempt.

After the fixture correction, the **entire243-test integrity suite passed locally
in173.660s**. No test was skipped or weakened. This supplements the complete
local76-group verifier pass; pinned formal/full wrappers remain required.

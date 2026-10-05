# PPO artifact byte boundary engineering follow-up

The new pure `ppo-artifact-v4` codec connects completed PPO-v2 training state to
v3 Haskell request decoding. All three public functions default disabled. It has
no file, network, optimizer restoration, candidate activation or order interface.
Frozen v1 artifacts and the historical financial screen retain their semantics.
The preregistration was committed as `633d078c` before implementation.

The codec stores complete immutable actor/critic snapshots, normalization buffers,
configuration, symbol scope and loss values. Little-endian binary64 buffers use
hex bytes; loss values use canonical float hex. An immutable raw byte snapshot is
bounded at 65,536 bytes, checked against independently supplied expected SHA256
and exact provenance, parsed without duplicate keys, checked structurally and
numerically, and required to match its canonical re-encoding. Only the explicit
version/contract tuple, development dataset role and research-only/disabled
metadata are supported. Restored results pass through the unchanged v3 encoder;
Haskell independently checks the resulting request. This does not create a
Haskell hash/provenance admission layer.

## Evidence and limitations

- F-RL-ARTIFACT-V4-GUARD: four SAT-premise/UNSAT-violation pairs for digest
  equality, metadata conjunction, byte bounds and completed-step arithmetic.
  SHA256/JSON implementations and source-to-model correspondence are trusted.
- F-RL-ARTIFACT-V4-FLOW: exhaustive finite gate model for one request, ten atomic
  gates and no retries. Publication requires all gates; failure is terminal.
  This is model checking under returning-primitive assumptions, not a universal
  Python runtime, concurrency, filesystem or cryptographic proof.
- F-RL-ARTIFACT-V4-BOUNDARY: full reviewed source/helper inventory and three
  default-disabled entries. Existing five scoped closures now include this new
  boundary; their criteria were not narrowed.
- F-RL-ARTIFACT-V4-CONFORMANCE: all nine synthetic configurations, seeds 11/23/47
  across horizons 1/3/6 at 17 steps. Restored state, re-encoded artifact bytes and
  original/restored request bytes must match; each request passes compiled GHC
  bit decoding. Registered malformed artifacts, seven expected-reference
  mismatches, invalid observations, non-Boolean enablement and 128 generated
  finite bit-pattern cases are tested. Source/model bypass mutants are deliberate
  regressions, not evidence of a historical defect.

Provenance fields are caller assertions, not signatures or proof of training
truth. Synthetic test references are deliberately fictitious identifiers.
Artifact freshness, external path selection, crash-safe file transactions,
normalizer application, causal observation construction, Haskell-side artifact
admission and production activation are outside this change. No broad obligation
closes: five scoped closed, 26 partial and seven open remain. In particular,
29/30 retain their original closure criteria and unresolved trust/loader gaps.

The [benchmark manifest](ppo-artifact-v4-engineering-manifest.json) records one
synthetic seed11/horizon1/17-step result, 100 repetitions per operation and a
24,823-byte artifact. Median encode/decode/request times were 2.3621/2.9732/4.0236
ms; observed request maximum was 37.7603 ms. This measures the Python byte codec,
excluding file/network I/O, Haskell startup/inference/cleanup and end-to-end
latency. It is not a deadline guarantee. Reproduce with pinned dependencies:

```sh
PYTHONPATH=scripts/formal:scripts/research python3 -c 'import json; from artifact_v4 import benchmark; print(json.dumps(benchmark(), indent=2))'
bash scripts/verify.sh formal
bash scripts/verify.sh full
```

The [engineering registry](ppo-artifact-v4-engineering-registry.json) counts ten
distinct synthetic training configurations: the nine registered combinations plus
one initial single-symbol smoke fit (seed11/horizon1/17 steps, exponential ALPHA
prices). That extra fit was outside the nine-fit preregistration and is recorded
as a deviation, not independent financial evidence. Repeated verification runs
and the repeated benchmark configuration are not new configurations; their failed
checks are retained below. No result selected a financial policy.

## Research disposition

No new academic efficacy claim, market-data access, financial fit, baseline,
cost assumption, economic metric, statistical inference, OPE estimate or holdout
result is introduced. The existing 108 financial fits, 19,440 replays and 19,548
registry rows remain frozen. All 108 existing OPE batches remain invalid;
independent matched-champion confirmation is missing. All 1,227 final returns
remain sealed. No candidate is adopted; continue offline research.

The new code is independently useful for exact serialization, corruption rejection
and reproducible request composition. It does not justify selecting a policy or
promoting an artifact. Live authorization settings, champion, fleet, exposure,
risk limits, ownership and production learning remain unchanged. No orders,
authenticated exchange experiment, live exploration or deployment occurred.

## Local verification failures retained

The first new targeted invocation completed conformance but failed a separate
checker test because its local `json` import was missing; that import was fixed.
An initial AST-text comparison was also corrected to compare parsed syntax trees.
These are verification-code corrections, not financial experiment changes.
A later targeted invocation hit the unchanged three-second compiled Haskell probe
timeout. Local `verify.py --record` also hit that timeout in existing bridge
conformance and did not generate an accepted receipt.

`bash scripts/verify.sh full` locally failed after 192 tests in 240.962 seconds:
`PPOArtifactV4Tests.test_trained_artifact_conformance` exceeded the three-second
probe timeout, and `PPOProcessBridgeTests.test_source_bound_smt_model_and_actual_trained_process`
failed its existing per-policy successful-inference gate. Host load averages
were observed above 40; scheduling pressure is an environmental limitation,
not a proved explanation for every timeout. No deadline, assertion, seed,
acceptance criterion or mandatory successful-policy check was weakened.

The first ordinary PR CI run passed all 192 tests in 47.952 seconds and then
failed solely because the committed receipt had not yet been updated. That
failure is retained; the receipt is imported only from actual isolated pinned
reproduction with matching source hashes, followed by final-head CI.

The deliberate model bypass regression has this reproducible trace, with state
`(next gate, passed-bit-mask, failed)`:
`(0,0,false) -> (1,1,false) -> (2,3,false) -> (3,7,false) -> (10,7,false)`.
The mutated digest gate jumps to publication without the remaining checks; the
required final mask is 1023, so the checker rejects it. Reproduce with
`artifact_v4.check_model(mutant=True)`. This is an intentionally faulty model
fixture, not a counterexample to the delivered codec or a historical incident.


## Receipt portability correction

The first isolated pinned run [37263342240](https://github.com/diegueins680/trader/actions/runs/37263342240)
passed both `formal` and `full` on source commit
`6388b55e816e1f4d44ca66e12b66d9f06da6c9ed`. Its original receipt SHA256 was
`ddb85dc2238b1663aa7f3d95e5c13a8cd0904915503f30c10861e2cf4cb98d69`.
Inspection found a reporting portability defect: it included measured artifact
lengths in the exact-equality proof receipt. Linux lengths for seed-major order
11/23/47 × horizon1/3/6 were
`[24824,24822,24822,24822,24822,24821,24809,24821,24821]`, while the pinned macOS
benchmark of the same seed11/horizon1 fixture measured 24823 bytes. Native
training arithmetic is not asserted to be bitwise identical across these runtime
backends; canonical hexadecimal loss strings can have different lengths.

The corrected certificate retains nine independent successful size-bound checks
and the 65536-byte maximum, plus every state/byte/Haskell parity assertion. Actual
measurements remain in this report and benchmark manifest. No codec behavior,
format, policy, threshold, seed, model or input changed. This corrects receipt
content instead of silently claiming cross-backend training determinism. The
first full result is historical evidence; verification must rerun on the corrected
proof/reporting source before merge.

# PPO training-to-process bridge: engineering result

The default-disabled PPO snapshot encoder and pure Haskell decoder now connect
actual v2 training results to the existing supervised research inference process.
This is a transient request boundary, not an authenticated model artifact or a
production integration. No financial experiment, market-data read, champion
change or final-holdout evaluation occurred. The preregistration and specification
were committed in `0a4cd2a3` before implementation.

The Python entry accepts the exact training/snapshot versions, actor width three,
completed optimizer-step metadata, four immutable parameter buffers and one
width-12 float64 observation. Finite values within [-1000,1000] become unsigned
binary64 words in an ASCII request. Haskell validates Integer ranges before
Word64 conversion, all versions/dimensions/step bounds and finite numeric bounds.
The existing supervisor revalidates the decoded request, launches only its fixed
worker with an empty environment, and applies its unchanged 20 ms final guard
after cleanup. The existing v2 mode and child protocol retain their semantics.
The new explicit mode is `--offline-snapshot-v3`; the bit probe cannot launch work.

A unique Haskell score maximum is required. Exact ties abstain. This preserves
the existing Haskell rule, which intentionally differs from the frozen Python
first-argmax rule. Clear-margin cross-language tests do not prove parity near
ties, the accuracy of BLAS/tanh, or universal finite neural computation.

| Evidence class | Scoped result |
|---|---|
| SMT-verified | Twenty-five satisfiable-premise/UNSAT-violation checks: lossless checked Integer conversion modulo 2^64; exact version/shape/step guard; all completed budgets 1..4096 map to steps 4..64 divisible by 4; binary64 finite/bounded guard; 20 decimal-fold bounds and list-count descent. |
| Model-checked | 83 states, 189 transitions, maximum shortest depth 10, initial rank 13. One child/request; both decoder and revalidation outcomes; three saturated time buckets; two polls per cleanup window. Accepted proposals require valid decoding, revalidation, cleanup and unexpired final admission. |
| Exhaustively checked source boundary | Complete encoder AST, two mandatory definitions, three mandatory helpers, complete pure decoder and supervised Main source. Missing inventories or unknown changes refuse verification. This is a reviewed source abstraction, not a verified Python/Haskell interpreter. |
| Conformance/property tests | Nine actual fits (seeds 11/23/47, horizons 1/3/6, steps 17), 27 clear-margin observations, exact-bit decoder/internal Show-Read checks, 16 special-bit frames, 20 invalid frames, 128 generated encoder cases, disabled/version/type/failure cases, source/model mutations and three actual process faults. Every policy must produce at least one accepted matching action; all other observations may only return the matching action or safe absence. |

Requirements F-RL-BRIDGE-V3-CODEC/FLOW/BOUNDARY/CONFORMANCE map to the
[canonical contract](../../formal/research/ppo-process-bridge-v3-contract.md),
[source registry](../../formal/research/ppo-process-bridge-source.json),
[ledger](../../formal/research/proof-ledger.json),
[checks](../../scripts/formal/ppo_process_bridge.py), implementation, regression
tests and canonical CI command. A-BRIDGE-V3 names trusted pinned runtime,
Integer/Word64 cast, parser, finite-Double Show/Read, stable-input and process
semantics. No parser/compiler, library, floating-point neural equivalence or OS
scheduler theorem is claimed. Scalar guard proofs do not authenticate provenance.

The first unoptimized conformance run failed the per-policy successful-inference
gate: a policy's three requests all safely timed out. A pinned optimized GHC
`-O2` build passed the targeted four-test suite in 24.131 seconds, without changing
the deadline. A subsequent loaded-host benchmark exposed an operational limit:
**all 30 requests abstained in each of the -O0 and -O2 builds**. The host reported
load averages 115.31/73.75/40.82 on 16 logical CPUs near the run. A separate temporary
diagnostic observed both timeouts before replies and valid replies rejected
because cleanup took the total beyond 20 ms. Neither build is operationally qualified
on this evidence. The benchmark preserves failures; it does not justify raising
the deadline, weakening conformance, deployment or adoption.

The [benchmark manifest](ppo-process-bridge-engineering-manifest.json) records
wall-clock timings, frame/executable sizes, host context and reproduction commands.
For the measured single synthetic configuration, training took 288.43ms; encoding
averaged 0.611 ms; requests were 5844..5850bytes. Median end-to-end wall time was
148.57 ms (-O0) and 174.31 ms (-O2), including startup and cleanup. The guarded 20 ms
window begins after worker readiness, so these wall times are not equivalent to
accepted inference latency. There were no accepted requests in this benchmark;
no successful-inference latency or throughput claim follows. Generated weights,
executables and large arrays are not committed.

The existing five scoped closures require the new source-boundary certificate as
well; no closure criteria are narrowed. **Five closed, 26 partial and seven open**
remain. Persistent artifact hashes/provenance, actual observation availability,
revision witnesses, replay accounting, Q/CQL and OPE composition, production
ownership, reconciliation and durable recovery remain outstanding. Previously
frozen 108 fits / 19,440 replays / 19,548 registry rows and invalid OPE results are untouched;
the 1,227 final returns remain sealed. No new drawdown, tail-risk, cost-stress,
statistical-significance or economic-comparison result is reported.

Reproduce without downloaded market data or exchange credentials:

```sh
bash scripts/verify.sh formal
bash scripts/verify.sh haskell
bash scripts/verify.sh automation
bash scripts/verify.sh full
"${TRADER_FORMAL_PYTHON:-python3}" scripts/formal/ppo_process_bridge.py --benchmark
```

Final wrapper and CI outcomes are recorded in the PR; a failed command must not
be reported as passing. Recommend continuing offline engineering/research. No
candidate passes promotion; no live exploration or deployment is authorized.

Parser refinement: the final v3 decoder uses a small bounded decimal grammar
instead of the generic tuple/list Read instance. The wire format is unchanged;
only ASCII whitespace outside tokens and unsigned decimal fields of at most
20 digits are accepted. List traversal counts down fixed widths 12/259 and
rejects extra/missing values or trailing material. Twenty decimal-fold bounds
and a decreasing-list-count lemma supplement the four admission lemmas (25
SAT-premise/UNSAT-violation pairs total). These lemmas and actual codec tests do
not prove the whole parser or internal child Show/Read implementation. The first
bounded-parser test also safely timed out under host load; that failure remains
recorded, and no deadline or successful-policy gate was relaxed.


## Executed verification

[Isolated pinned-toolchain run 37259696383](https://github.com/diegueins680/trader/actions/runs/37259696383)
checked source commit `cc7f64b8ff56454e6d3f70bf4f17e106d69d8b35`.
`verify.py --record` reproduced the receipt in 51.004 seconds; the downloaded
receipt was imported byte-for-byte after checking every source hash against the
local files. Its SHA256 is
`6304a30b2fc39c964eee2201fc4d994bf5b3dddddfd24eaca1852e5b71dfd3d5`.
The temporary reproduction workflow is removed from the delivered tree.

- `bash scripts/verify.sh formal`: PASS; 189 tests in 32.840 seconds, followed by
  full scoped certificate reproduction/comparison in 49.944 seconds.
- `bash scripts/verify.sh full`: PASS; 189 formal tests in 32.443 seconds,
  certificate reproduction/comparison in 49.051 seconds, Haskell build/format/
  HLint 3.8/smoke/tests, 241 web tests and 185 automation tests. The full command
  ran from 03:34:40 to 03:49:37 UTC on 2026-10-05. The Haskell lint phase accounts
  for most of this time; no test, assertion or timing gate was disabled.
- Local `bash scripts/verify.sh haskell`: PASS, including the complete test suite.
- Local `bash scripts/verify.sh formal`: FAIL under extreme concurrent host load;
  189 tests ran in 294.042 seconds, with one error: the compiled
  `--snapshot-contract-v3` probe exceeded its three-second subprocess timeout.
  A later local `verify.py --record` also refused a policy with no accepted action
  within the strict 20 ms guard. These failures are not relabeled as passes.
- Initial ordinary GitHub CI also caught `Use isDigit` in the handwritten parser.
  The fix uses the pinned ASCII predicate and rejects Arabic/fullwidth lookalikes;
  the corrected Haskell check passed. Receipt mismatches on pre-recording commits
  remained failures until an independently reproduced receipt was available.

The complete proof ledger and source locks remain authoritative. Passing these
scoped checks does not establish operational timing on the loaded local machine,
close the remaining 33 broad obligations, or justify trading-policy adoption.
The final PR head also requires its ordinary CI checks before the authorized merge.

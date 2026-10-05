# Trader Firm — Risk Register

A-ROUND-DOWN / CE-ROUND-001: downward epsilon rounding could increase quantity; the exact pure core and final binary64 guard repair this defect. Source-bound SMT and compiled properties do not certify wire rendering, upward rounding, minimum-size increases or all venue adapters. GHC primitives and sufficient resources remain assumptions. RL-OFFLINE-001 remains HIGH/OPEN; five scoped closures, 27 partial, six open. See [contract](../formal/research/quantity-rounding-contract.md).


PPO artifact v4 adds a bounded pure byte codec and integrity/provenance comparison
under A-ARTIFACT-V4. Caller trust anchors, hash/JSON primitives and source-to-model
correspondence are explicit assumptions. Exact byte round trips do not establish
training truth, freshness, causal normalization application or Haskell-side
artifact admission. No path selection, activation or order capability is added.
RL-OFFLINE-001 remains HIGH/OPEN; five scoped closures, 26 partial and seven open
remain. See [the contract](../formal/research/ppo-artifact-v4-contract.md).


PPO snapshot bridge v3: trained-policy requests now pass through checked integer
bit decoding and the existing supervised Haskell inference guard. A-BRIDGE-V3
names the parser, bit-cast, Show/Read, runtime and process assumptions. The first
unoptimized build safely timed out on some policies; `-O2` passed the targeted
per-policy conformance test without changing the deadline, but a later loaded-host
benchmark had 30/30 absences in both builds. Operational timing is not qualified. Request metadata is not authenticated provenance and supplied
observations have no new availability witness. No persistent artifact or order
capability is added. RL-OFFLINE-001 remains HIGH/OPEN; no broad obligation closes.
See the [contract](../formal/research/ppo-process-bridge-v3-contract.md).


PPO successor v2: actual offline training now uses checked GAE and private immutable
actor/critic snapshots under A-PPO-SUCCESSOR. Numeric and partial-update failure
publishes no TrainingResult. Runtime/helper semantics and training-prefix selection
remain assumptions; this does not certify accounting error, funding provenance,
artifact/process inference, OPE, production integration or economic superiority.
Five scoped closures remain; 26 partial and seven open remain. RL-OFFLINE-001 is
HIGH/OPEN. See the [contract](../formal/research/ppo-successor-v2-contract.md).

See [shutdown contract](../formal/research/shutdown-deadline-contract.md) and CE-SHUTDOWN-001/002/003. The repair changes server cleanup code but is not deployed by this research task.

Inference process v2 follow-up: A-INFERENCE-PROCESS assumes bounded OS creation,
pipe, signal and scheduler service, a trusted self executable and exclusive child
reaping. The new research-only executable can kill noncooperating workers, rejects
expired replies after cleanup and reports cleanup failure explicitly. It has no
production component, order interface or artifact loader. It is not a hostile-code
sandbox; parent process death/repeated cancellation and inherited server shutdown
remain unverified. CE-RL-017 remains valid for frozen v1; RL-OFFLINE-001 stays HIGH/OPEN.


Closure audit (2026-10-03): [criteria and scope](../formal/research/obligation-closure-audit.md) replace the impossible completion guard (CE-RL-021). Default-disabled, capability separation and no production learning are closed only for delivered offline boundaries under explicit language/runtime assumptions (A-CAPABILITY-BUILD). Actual deployment images and inherited live authority remain unverified; Dockerfile.optimized still names GHC 8.10.4 rather than canonical 9.4.8. The remaining 35 obligations, numerical counterexamples and economic rejection keep `RL-OFFLINE-001` HIGH/OPEN. Earlier 38-open counts below are historical.

Snapshot v2 follow-up: A-SNAPSHOT-V2 trusts pinned CPython/GIL reference/lock semantics and NumPy scan/serialization, with ordinary immutable snapshot discipline. New update/forward consumers use coherent snapshots; interrupted acknowledgement and orphan-lock recovery remain limitations. CE-RL-023 records backend-dependent raw-byte-view numerical parity and its tested native-copy correction. CE-RL-016 in frozen v1 and broader production synchronization remain unresolved; RL-OFFLINE-001 stays HIGH/OPEN.


Reward accounting evidence (2026-10-03): [contract](../formal/research/reward-accounting-contract.md) separates call-relative reward from compounded economic return. The reporter correctly records the preserved synthetic loss. Exact-real SMT and tolerance tests do not discharge floating-point/production refinement; `RL-OFFLINE-001` remains HIGH/OPEN.

Replay cutoff evidence (2026-10-01): [contract](../formal/research/replay-cutoff-contract.md) extends abstract single-call ordering through an integer cutoff lemma and class-seven graph. Observation/reward and Haskell runtime refinement remain open. A checker mutation is preserved and corrected; `RL-OFFLINE-001` remains HIGH/OPEN.

Replay ordering evidence (2026-10-01): [contract](../formal/research/replay-order-contract.md) checks a source-bound single-call abstraction and synthetic helper traces. Terminal liquidation is distinct from a delayed policy target; early rejected calls can retain simulated state. Runtime, cash arithmetic and production lifecycle remain open; `RL-OFFLINE-001` remains HIGH/OPEN.

Funding boundary evidence (2026-10-01): [contract](../formal/research/funding-boundary-contract.md) records CE-RL-019 product/sum overflow from finite synthetic inputs. Scoped endpoint proofs do not prove provider availability; replay rejects the encountered invalid transition. Frozen loader unchanged; `RL-OFFLINE-001` remains HIGH/OPEN.

Exact ESS diagnostic (2026-09-30): [contract](../formal/research/ess-v2-contract.md) introduces a disabled, disconnected rational kernel. Its supplied-weight arithmetic avoids CE-RL-018 locally; the frozen helper retains the witness. Upstream lost weights, statistical support, full runtime refinement and promotion remain unresolved. `RL-OFFLINE-001` remains HIGH/OPEN.

OPE numerical evidence (2026-09-30): [contract](../formal/research/ope-algebra-contract.md) records CE-RL-018, where squared positive weights underflow and ESS reports zero. This prescribed helper fixture is outside the current deterministic six-step weight domain. Exact-real identities do not establish binary64 accuracy or statistical reliability; `RL-OFFLINE-001` stays HIGH/OPEN.

Inference timing evidence (2026-09-30): [contract](../formal/research/inference-boundary-contract.md) and CE-RL-017 preserve the known non-preemptive call limitation. Measured time excludes later output validation/selection; predicate, clock and cancellation failures remain assumptions or open scope. `RL-OFFLINE-001` stays HIGH/OPEN.

Optimizer publication evidence (2026-09-30): [contract](../formal/research/optimizer-publication-contract.md) and CE-RL-016 expose partial publication under an injected interruption. Prepublication failure tests and a single-writer model do not establish concurrent atomicity. Rounded gradient-norm bounds remain assumptions; `RL-OFFLINE-001` stays HIGH/OPEN and the learner is unchanged.

Value-objective audit (2026-09-30): [contract](../formal/research/value-objective-contract.md) records CE-RL-014/015: finite value inputs can produce a NaN loss at zero CQL weight or lose the conservative loss under a common shift. Finite gradients can pass optimizer admission despite invalid loss. The learner is unchanged and `RL-OFFLINE-001` remains HIGH/OPEN.


PPO objective audit (2026-09-28): [contract](../formal/research/ppo-objective-contract.md) separates exact-real clipping algebra from binary64 safety. CE-RL-012 returns finite loss with NaN gradient; optimizer refusal is tested, the frozen learner remains unchanged and `RL-OFFLINE-001` stays HIGH/OPEN. CE-RL-013 shows the intended absence of a universal clipping multiplier bound; it is not an external action-shield failure.

Numerical proof-query reliability (2026-09-28): [query-isolation contract](../formal/research/query-isolation-contract.md) separates non-vacuity and universal queries without changing arithmetic, domains or the 10-second solver limit. Unknown/canceled checks remain blocking. Timing reliability and trusted solver/compiler semantics remain limitations; no mission obligation or candidate is promoted.

Isolated target-kernel mitigation (2026-09-28): [gae-targets-v2](../research-notes/sequential-control-2026-09-17/target-v2-followup-2026-09-28.md) is disabled by default and not called by training or production. It corrects terminal reward reconstruction or rejects the two numerical fixtures in that kernel only. The frozen learner retains CE-RL-010/011; `RL-OFFLINE-001` stays HIGH/OPEN.


Terminal-target numerical evidence (2026-09-28):
[CE-RL-010/011](../formal/research/terminal-counterexamples.json) are current-source
research-learner counterexamples, not mutations. Finite inputs can lose terminal
reward through cancellation or yield infinity/NaN through residual/carry arithmetic.
No historical-market impact is established. Exact-real and finite-product proofs
do not repair these open blockers; `RL-OFFLINE-001` remains HIGH/OPEN.

Training-transition admission evidence (2026-09-28 continuation):
[source-derived predicates](../formal/research/transition-admission-contract.md)
establish necessary terminal-flatness and successor-admission conditions under
trusted scalar/vector helper semantics. Complete cash accounting, genuine risk
classification and simulator/runtime refinement remain open. `RL-OFFLINE-001`
remains HIGH/OPEN.

Artifact-admission evidence (2026-09-28 continuation): the
[source-linked gate model](../formal/research/artifact-admission-contract.md) and
metadata SMT checks assume trusted helpers and externally supplied expected
provenance. They do not establish authenticity, full runtime refinement or a
production artifact path. `RL-OFFLINE-001` remains HIGH/OPEN.

Training-prefix evidence (2026-09-28 continuation):
[source-derived fit and episode checks](../formal/research/training-prefix-contract.md)
certify index bounds conditional on admitted arrays and trusted primitive/constructor
semantics. Whole-panel validation is not online admission causality; complete
Replay refinement remains open. `RL-OFFLINE-001` remains HIGH/OPEN.

Source-linked causality evidence (2026-09-28 continuation): the
[feature read-footprint check](../formal/research/causal-footprint-contract.md)
links SMT index bounds to the actual Python AST. Base-array, runtime and primitive
semantics remain assumptions; publication timing, caller symbol selection and
full replay-state causality remain open. `RL-OFFLINE-001` stays HIGH/OPEN.

Gap-risk evidence (2026-09-28): `RL-OFFLINE-001` remains HIGH/OPEN.
[CE-RL-002/003](../formal/research/gap-counterexamples.json) reproduce capital-floor
breaches after shielded targets in the offline replay. Conditional real-arithmetic
[risk lemmas](../formal/research/gap-risk-contract.md) require bounded price moves
and cash debits; these are unestablished environmental assumptions, not live risk
guarantees. No production limit or risk lifecycle status changes.

`formal/risk-register.json` is the canonical machine-readable source for risk
IDs, severities, and lifecycle statuses. This table and the typed Haskell
projection in `app/Trader/Formal/RiskRegister.hs` must contain exactly the same
entries in canonical ID order; automation rejects drift and duplicate IDs.

Severity describes impact (`LOW`, `MEDIUM`, `HIGH`, or `CRITICAL`). Status
describes lifecycle (`OPEN`, `MITIGATED`, or `CLOSED`). A mitigation does not
change severity and a fixed risk is `CLOSED`, never encoded as a severity.

| ID | Risk | Severity | Owner | Status | Next Action |
|---|---|---|---|---|---|
| AUTOLOOP-DOWN-003 | Autoloop process was not alive at the prior operational review | CRITICAL | trader-firm-cto | CLOSED | Launchd ownership, bounded-cycle completion, a current heartbeat, clean merged-main sync, and permission-safe PID status are witnessed |
| AUTOLOOP-RESET-2026-05-30 | Autoloop cycle counter reset and broke continuity assumptions | CRITICAL | trader-firm-cto | CLOSED | Merged runtime restart advanced from issued cycle 2761 to atomic reservation 2762 with a valid schema-1 sequence |
| AUTOLOOP-SINGLETON-001 | Multiple autoloop instances may race on the same repository | HIGH | trader-firm-cto | CLOSED | Merged runtime held a private schema-1 owner while a concurrent direct launch failed `EALREADY` without owner or status-identity mutation |
| AUTOLOOP-STALL-001 | Autoloop stall detection depended on manual observation | CRITICAL | trader-firm-cto | CLOSED | Heartbeat telemetry and a bounded stale-heartbeat alert are implemented |
| BINARY-HANG-001 | The trader binary could hang while draining after a termination signal | MEDIUM | trader-firm-cto | MITIGATED | Close after a serve-mode PostgreSQL subprocess termination witness |
| CIO-DEAFNESS-001 | The CIO reporting lane missed recorded deadlines | CRITICAL | trader-firm-ceo | CLOSED | Current CIO report retains the missed deadlines, renders NO-GO, preserves the champion and sealed holdout, and inventories every unresolved risk |
| EXECUTION-DATASET-001 | Backtest dataset generation is not fully reproducible | MEDIUM | trader-firm-data | CLOSED | Fixed-window acquisition records source and CSV SHA-256 values plus explicit no-randomness provenance; expected-manifest and offline verification fail closed on drift |
| EXECUTION-MISSING-001 | The execution reporting lane missed recorded trade-log deadlines | CRITICAL | trader-firm-execution | CLOSED | Current execution report records the missed deadlines, delivered controls, evidence boundary, and unresolved limitations |
| EXECUTION-RESTART-001 | Persisted bot exposure could create a phantom position during restart | CRITICAL | trader-firm-risk | CLOSED | Startup exposure is venue-authoritative; snapshot recovery admits only identity-matched closed-trade memory and deterministic scenario coverage proves exposure fields are ignored |
| EXPECTANCY-INVALID-001 | Missing or non-finite expectancy could bypass a configured minimum | CRITICAL | trader-firm-risk | CLOSED | `specRiskHalt` rejects malformed expectancy and bounded verification covers it |
| FEATURE-MISSINGNESS-001 | Optional predictor features can encode unavailable evidence as the same numeric zero as an observed value | HIGH | trader-firm-research | OPEN | Keep every v2 boundary isolated; begin complete market-context acquisition no earlier than the registered 2027-01-21 boundary, run the offline bundle verifier on every collector output, replay persisted receipts against their exact frozen archives, require separate data admission, build remaining timestamp-preserving OHLCV/external/Coinbase artifacts and source policies, then version fitted market-context/model artifacts and production builders before promotion eligibility |
| GITHUB-502-001 | Transient GitHub API failures can interrupt automation | MEDIUM | trader-firm-cto | CLOSED | Production GitHub reads use a four-attempt 502-only policy; deterministic recovery/exhaustion regressions and a current supervised cycle witness validate the bound |
| KALMAN-NUMSTAB-001 | Kalman numerical instability could produce zero trades or hangs | MEDIUM | trader-firm-cto | CLOSED | Current long-run malformed-input regressions and a bounded Kalman-only CLI witness complete with finite output; flat default behavior is attributable to explicit admission gates |
| LEVERAGE-INVALID-001 | Malformed leverage configuration could bypass position-size protection | CRITICAL | trader-firm-risk | CLOSED | `specRiskHalt` rejects malformed leverage and live futures leverage is capped |
| LEVERAGE-SANITY-001 | Corrupted or absurd venue leverage evidence could bypass size limits | CRITICAL | trader-firm-risk | CLOSED | Venue evidence is capped and configuration validation rejects malformed values |
| LOSS-STREAK-LIMIT-INVALID-001 | A negative loss-streak limit could silently disable protection | CRITICAL | trader-firm-risk | CLOSED | Negative limits fail closed while zero remains the documented disabled boundary |
| MARKET-DATA-TIMESTAMP-OVERFLOW-001 | Timestamp overflow could make stale or discontinuous evidence appear valid | CRITICAL | trader-firm-data | CLOSED | Checked arithmetic fails closed across market-data time validation |
| MAX-POSITION-GUARDRAIL-001 | Malformed maximum-position configuration could silently disable every trade | HIGH | trader-firm-risk | CLOSED | Checked simulation rejects non-positive or non-finite configuration |
| PREDICTOR-IDENTITY-001 | Legacy TCN, PatchTST, and Transformer identifiers can overstate the fidelity of lightweight proxy implementations | HIGH | trader-firm-research | MITIGATED | Preserve legacy semantics, expose accurate versioned implementation identities, and require a new model ID for any faithful architecture |
| RESEARCH-RATE-LIMIT-001 | Shared-IP Binance throttling can interrupt prospective derivatives collection and create irrecoverable acquisition gaps | HIGH | trader-firm-research | MITIGATED | Use conservative local budgets, stop all later requests on throttling, and rotate the request leader by UTC hour; monitor gaps and move collection to stable persistent egress |
| RESEARCH-RECEIPT-001 | A metadata-only derivatives receipt could diverge from its frozen external archive or imply unauthorized outcome access | HIGH | trader-firm-research | CLOSED | The schema-1 verifier binds the exact status and complete archive inventory while enforcing acquisition-only authority |
| RISK-LIMIT-001 | Daily, weekly, and drawdown limits were not enforced in the live loop | HIGH | trader-firm-risk | CLOSED | Runtime invariant checks and guardrail regressions are implemented |
| RISK-LIMIT-NON-FINITE-001 | Non-finite risk limits could silently disable halt checks | CRITICAL | trader-firm-risk | CLOSED | `specRiskHalt` rejects non-finite limits and bounded verification covers it |
| RISK-METRIC-INVALID-001 | Malformed loss or drawdown evidence could bypass live halt checks | CRITICAL | trader-firm-risk | CLOSED | `specRiskHalt` validates risk evidence before threshold comparisons |
| RL-OFFLINE-001 | Simulator policies can exploit costs, coverage, episode boundaries or contaminated history | HIGH | trader-firm-research | OPEN | Keep sequential-control v1 offline and non-authorizing; require explicit boolean gates, finite real proposal/inference inputs and typed replay and training bounds/budgets/modes; validate immutable normalization snapshots and policy parameters before persistence, and reject incomplete learning transitions while retaining accounted risk losses; reject duplicate keys and non-finite numbers while parsing verified evidence snapshots, preserve their admitted hashes, pin registration/source provenance to the exact Git-validated snapshots and reconcile registry/result summaries with versioned completed/truncated episode coverage; require stopped and consistent completion state, contiguous finite bar ledgers and per-bar return/equity reconciliation for economic reports while retaining failed paths; abort archives on terminal-ledger or return-path publication failure; pin export paths and prohibit destinations within the source archive; validate resource metadata domains, timing agreement and explicit RSS units; reject invalid baseline observations, rule names, fitted parameters and non-finite intermediate scores; admit complete short-OPE windows, integer budgets and aligned symbol coverage before sampling; validate policy observations/outputs before selection or transition and reject non-finite neural hidden arithmetic before activation; reject invalid optimizer controls and overflowing update arithmetic, publishing parameters and moments only after whole-update validation; reject masked OPE inputs before coercion and malformed or overflowing OPE evidence without weight clipping; require fresh data, matched champion, credible OPE, complete risk-safe paths and separate human review before any integration |
| SCHEMA-001 | Live trade-log schema could drift from its declared contract | CRITICAL | trader-firm-cio | CLOSED | The schema contract and executable validation are tracked |
| SHUTDOWN-DEADLINE-001 | Shutdown timing and completion may diverge from the promised lifecycle bound | HIGH | trader-firm-cto | OPEN | Monotonic budgets, worker closure/completion, async reservation/publication, sealed completion and backtest cancellation and shared STM drain/pool ordering repair eighteen counterexamples (including a reserve-gap schedule adapter and three stale-ingress schedules); named clock/worker/async/seal/backtest/STM runtime assumptions, pre-drain owners, other HTTP/bot/order admission, durable recovery and full IO refinement remain open |
| THRESHOLD-FACTOR-001 | `thresholdFactor` may not be wired into simulation configuration | MEDIUM | trader-firm-research | CLOSED | CLI validation, optimizer serialization, both simulation constructors, causal simulator application, and an enabled-versus-disabled admission regression are verified |
| TRADE-LOG-GAP-001 | Trade-log records lacked required exit and halt evidence | HIGH | trader-firm-cio | CLOSED | The schema includes `exit_reason` and the trade-log implementation is tracked |
| TRADE-LOG-GAP-002 | Trade logs lack a native snapshot of derived risk-state metrics | MEDIUM | trader-firm-cio | CLOSED | Schema 1.2 live event rows carry null-safe pre-execution risk snapshots while backtest rows retain explicit v1.1 semantics |
| TRAILING-STOP-001 | A trailing-stop exit may re-enter on the same bar | MEDIUM | trader-firm-execution | CLOSED | Intrabar protective exits impose a one-event minimum cooldown, and a regression proves zero configured cooldown cannot reuse the trailing-stop exit index |
| VOL-TARGET-001 | A stale report claimed the volatility-confidence stateful-close regression broke the Haskell suite | CRITICAL | trader-firm-cto | CLOSED | The fix predates the imported report; helper, live/backtest parity, and canonical Haskell verification witnesses pass |
| VOL-TARGET-INVALID-001 | Malformed volatility-target configuration could bypass scaling limits | CRITICAL | trader-firm-risk | CLOSED | `specRiskHalt` rejects malformed targets and bounded verification covers it |
| ZERO-VIABLE-SIGNAL-001 | No strategy signal had met the recorded long-sample viability threshold | CRITICAL | trader-firm-research | OPEN | Preserve the champion and continue only preregistered prospective protocols; require all economic, statistical, cost, delay, and drawdown gates before promotion |

## Update rule

Change `formal/risk-register.json` first, then update both projections in the
same commit. IDs are permanent. Reopening a risk changes its status rather than
creating a duplicate row; a materially different risk receives a new ID.

Last reconciled: 2026-09-07.

## Research proof coverage — 2026-09-20

`RL-OFFLINE-001` remains HIGH/OPEN. The [machine-readable proof ledger](../formal/research/proof-ledger.json)
binds narrow SMT lemmas, a two-caller abstract model and Haskell conformance tests
to exact sources. Its six named environmental/tool assumptions are not hidden
axioms. All 38 whole-system obligations retain implementation gaps. In particular,
finite proposal bounds do not prove loss bounds, source availability, exchange
execution or production lifecycle correctness. No verified candidate or promotion
is implied; [CE-RL-001](../formal/research/counterexamples.json) documents retained
non-authorizing proposals after disable. Existing canonical IDs/statuses and the
typed Haskell risk projection remain unchanged.

The [worker registry contract](../formal/research/worker-registry-contract.md) scopes CE-WORKER-001/002/003 and the source-bound model/SMT/conformance repair. Completion covers tracked callbacks and their finalizers, not descendants or exchange/resource reconciliation. Cancellation timeout remains an explicit failure.

A-ASYNC-ADMISSION and [its contract](../formal/research/async-job-admission-contract.md) scope CE-ASYNC-001/002: failed preparation leaked a queue slot and failed publication could leave an untracked callback. The repaired in-memory boundary does not reconcile persisted running records or close the HTTP drain gate.

### RL-OFFLINE-001 — data composition follow-up (2026-10-04)

The offline screen now has source-composed evidence for causal normalization
and local symbol isolation: 28 checked scale-use sites, six source-derived key
SMT checks, immutable snapshots, and actual loader/fit/replay/OPE conformance.
Obligations 3 and 5 close under A-DATA-COMPOSITION (trusted pinned Python, pandas,
NumPy, cryptographic identity and source grammar; no hostile objects or concurrent
mutation). The shared training transform/policy is explicitly global. Historical
publication/revision witnesses, online admission causality, full split isolation,
numeric/accounting refinement and inherited production correctness remain open.
Risk remains HIGH/OPEN; current broad count is 5 closed, 26 partial, 7 open.

CE-DATA-COMPOSITION-001 records and fixes a verifier coverage omission: the first
source-composition checker did not require its full skeleton roster. The corrected
checker requires every reviewed block and rejects omitted coverage. No historical
market leakage or trading behavior change is inferred from this injected mutant.


Upward rounding follow-up (2026-10-05): [the contract](../formal/research/upward-rounding-contract.md)
adds F-ROUND-UP-INTEGER/FINITE/FLOW/CONFORMANCE and A-ROUND-UP. Exact Integer/Real
ceiling lemmas and binary64 guard checks are distinct from the eight-state local
dispatch model and compiled actual-caller properties. CE-ROUND-003 preserves
unit-grid under-rounding; CE-ROUND-004 preserves invalid-price market fallback.
Invalid maker prices now return an unsent result. Other configured fallback
reasons remain unchanged. Runtime/compiler primitives and source-to-model
mapping remain assumptions; no whole-Main, final-wire or complete order-cap
proof. Five scoped closures, 27 partial, six open; RL-OFFLINE-001 stays HIGH/OPEN.


The [order-number contract](../formal/research/order-number-contract.md) adds
F-ORDER-NUMBER-FINITE/SELECTION/FLOW/CONFORMANCE under A-ORDER-NUMBER. Five
binary64/selection SMT queries and a 40-state local guard model supplement
compiled actual current/legacy prefixes for all five Binance constructors.
CE-ORDER-NUM-001/002 preserve non-finite guard bypasses. Runtime and source-mapping
assumptions remain explicit. No full IO, caller retry, venue acceptance, wire-grid,
cap or authorization proof. Counts remain 5 scoped closures, 27 partial, 6 open;
RL-OFFLINE-001 remains HIGH/OPEN and no research candidate is promoted.


The [wire-admission extension](../formal/research/order-wire-contract.md) adds
F-ORDER-WIRE-POSITIVE and A-ORDER-WIRE to the existing numeric boundary. The
formatter is unchanged and shared by validation and wire generation; exact
Fixed E12 parsing rejects zero-wire values before credentials. Seven conditional
SMT queries, an 80-state local model, 5728 compiled rows across three prefix
versions and 11456 wire-byte comparisons are checked by the formal wrapper.
Parsing/formatting remain explicit pinned-library assumptions. No wire cap/grid,
whole-caller or complete IO theorem; 5 scoped closures, 27 partial, 6 open remain.
CE-ORDER-WIRE-001 is preserved; RL-OFFLINE-001 remains HIGH/OPEN.

The wire-aware maker-price preflight also rejects zero-wire prices before constructor exceptions can enter configured market fallback (CE-ORDER-WIRE-002). Other maker fallback reasons and the fallback flag are unchanged; this is a source-bound local guard with compiled conformance, not a whole-caller proof.


The [sizing admission contract](../formal/research/sizing-input-contract.md)
adds F-SIZING-INPUT/PUBLISH/RETRY/FLOW/CONFORMANCE under A-SIZING-INPUT.
Fourteen conditional SMT queries cover binary64 predicates, publication bounds
and non-retryable errors. A 56-state/32-transition/depth-three local model and
6214 compiled input rows connect metadata/price admission to actual Main
normalizers and the minimum retry classifier. Negative raw quantities also
reject before clamping; zero semantics remain. CE-SIZING-001–006 preserve six
synthetic failures; no historical incident is inferred. Parsing/freshness,
missing optional filters, direct minimum consumers, exact notional arithmetic,
wire caps/grid and complete caller/lifecycle refinement remain unresolved.
Five scoped closures, 27 partial, six open; RL-OFFLINE-001 remains HIGH/OPEN.

Closed-trade recovery now has [numeric/index admission](../formal/research/closed-trade-recovery-contract.md)
under A-RECOVERY-NUMERIC: seven guard/Integer SMT queries, a 22-state local
record-eligibility/publication model, 341 bounded histories, 2675 numeric cases
and 1031 compiled index histories. CE-RECOVERY-001–004 preserve synthetic
failures; actual Aeson integration is tested separately. No parser/compiler,
authenticity, freshness, durable crash recovery, resource-bound or ownership
certificate is inferred. Obligations 6/10/29/37 receive evidence without status
changes: five scoped closures, 27 partial, six open. RL-OFFLINE-001 remains open.

Inventory readiness now has [strict snapshot admission](../formal/research/inventory-readiness-contract.md)
under A-INVENTORY-READINESS. CE-READINESS-001–004 retain flat-owner, unsupported
hedge, wrong-side start-acknowledgment and interrupted-rescan failures. Four SMT
predicate queries, a 25-state/33-transition two-cycle publication model and 6468
actual-Haskell predicate cases cover this boundary. Complete/truthful venue data
and ordinary runtime semantics remain assumptions. Persistent-owner uniqueness,
concurrent snapshot consistency, continuous freshness and HTTP/drain linearization
are not proved. Obligation16 advances from open to partial; totals are five scoped
closures, 28 partial, five open. No closure criterion is weakened; RL-OFFLINE-001
remains open. This health repair does not modify adoption or production ownership.

Promotion boundary follow-up (2026-10-05): [A-PROMOTION-BOUNDARY](../formal/research/promotion-boundary-contract.md)
records ordinary runtime/numeric objects, trusted fixed code and primitive effects,
registered scalar identifiers, fresh exclusive directories and a stable filesystem
namespace. Source inventory equality is drift detection, not whole-language proof.
The actual field predicates/native key sets are SMT checked; the finite lifecycle
and existing build/type/process certificates compose the affected boundary.
Obligations23/26 now have conditional scoped closure, with their original criteria
and scope unchanged: **7 scoped closures,26 partial,5 open**. No inherited live
control-plane, ownership, deployed-image, crash-atomicity, hash-authenticity or OS
sandbox theorem is claimed. RL-OFFLINE-001 stays open. No financial evidence changes.

Artifact admission follow-up (2026-10-05): [A-ARTIFACT-COMPOSITION](../formal/research/artifact-composition-contract.md)
records immutable identity/byte semantics and the trusted JSON/hash/array/compiler
primitives. The old v1 expected-provenance mutation race is reproduced and corrected;
version/hash predicates and finite concurrent replacement behavior are checked.
Source correspondence is reviewed and conformance-tested, not a whole-interpreter
proof. Expected identity authenticity, historical data availability, crash recovery,
ownership and inherited production correctness remain separate unresolved claims.
Scoped29/30 close without changing their criteria: **9 scoped closures,24 partial,
5 open**. RL-OFFLINE-001 remains open; no new economic or production authorization.

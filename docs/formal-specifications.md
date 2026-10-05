# Canonical Formal Specifications

Downward rounding now has F-ROUND-DOWN-INTEGER (exact quotient/reconstruction SMT), F-ROUND-DOWN-FINITE (binary64 guard SMT) and F-ROUND-DOWN-CONFORMANCE (compiled properties). [The contract](../formal/research/quantity-rounding-contract.md) traces the pure core, Binance adapter, tests, CI and CE-ROUND-001. Source binding plus differential tests is not whole-compiler refinement. Obligation 9 is partial; final wire/order-cap correctness remains open.


The [inference process v2 contract](../formal/research/inference-process-v2-contract.md)
adds F-RL-PROCESS-LIFECYCLE (conditional finite model), F-RL-PROCESS-ADMISSION
(integer SMT), F-RL-PROCESS-ISOLATION (reviewed finite source boundary) and
F-RL-PROCESS-CONFORMANCE (compiled property/fault tests). A-INFERENCE-PROCESS
names OS, compiler, self-binary and PID assumptions. Source locks and tests do not
constitute full IO refinement. No broader closure is inferred from these components.


Current closure status (2026-10-03): [all 38 audited obligations](../formal/research/obligation-closure-audit.md) now have explicit criteria. Obligations 11, 24 and 31 are `exhaustively_checked` for the delivered offline boundary; 35 remain open/partial. The [capability contract](../formal/research/capability-isolation-contract.md) combines complete parsed dependency closure, enumerated inference effects and SMT artifact-key incompatibility, under explicit compiler/primitive/packaging assumptions; it does not certify inherited live authority or deployed images. Earlier all-38 statements below describe historical audit snapshots. Formal completion is computed separately from empirical research acceptance.

The [snapshot v2 contract](../formal/research/optimizer-snapshot-v2-contract.md) adds a source-bound two-writer model, integer/IEEE guard checks, source-effect enumeration and real implementation conformance. Immutable single-reference publication repairs the new optimizer path; inherited atomicity/race obligations remain partial. No hard deadline, durable transaction, recovery or frozen-v1 repair is implied.


The [reward accounting contract](../formal/research/reward-accounting-contract.md) adds exact-real row, wealth-fold and reward identities plus a refuted additive-reward interpretation. Synthetic conformance is not a binary64 error proof. Existing lifecycle/order models retain their scoped assumptions.

The [replay cutoff contract](../formal/research/replay-cutoff-contract.md) adds F-RL-REPLAY-CUTOFF (integer SMT) and F-RL-REPLAY-QUOTIENT (finite ordering model). Class seven represents longer calls only for ordering; time-to-end observations and rewards are not equivalent under the projection.

The [replay event-order contract](../formal/research/replay-order-contract.md) clarifies A-SEQUENTIAL-RESEARCH-E1: policy targets fill later than their decision; mandatory liquidation reconciles at the terminal endpoint. The source-bound finite model and due-time SMT do not prove numeric accounting or production lifecycle.

The [funding boundary audit](../formal/research/funding-boundary-contract.md) adds conditional registered-grid and left-endpoint SMT obligations plus a refuted finite-output claim. CE-RL-019 is synthetic negative evidence, not a historical incident.

The [2026-09-28 gap-risk extension](../formal/research/gap-risk-contract.md)
distinguishes target bounds, post-cost exposure and pathwise loss limits.
`F-RL-GAP-BOUND`, `F-RL-DRAWDOWN-COMPOSE` and `F-RL-POSTCOST-EXPOSURE`
are conditional exact-real SMT lemmas. `F-RL-UNCONDITIONAL-FLOOR` is explicitly
refuted by two replayable witnesses; `F-RL-GAP-CONFORMANCE` checks a finite
180-case accounting grid. The proof ledger retains whole-system blockers and
the risk register names the unestablished bounded-jump/debit assumption.

The canonical, machine-readable specification set is [`formal/specifications.json`](../formal/specifications.json). It covers the production Haskell, web, automation, research, deployment, and CI surfaces. `FORMAL_METHODS.md` and `docs/formal-specs-extracted.md` remain deeper explanations of selected trading-critical contracts; if prose conflicts with the registry or executable implementation, the conflict is a verification failure to resolve, not an alternate specification.

## Semantics

Each feature family is modeled as a partial transition:

`F : Input x State -> Output x State'`

- `requires` defines admissible input and state.
- `ensures` defines the postcondition for an admitted transition.
- `invariants` must hold for every reachable transition.
- `failures` defines the conservative result outside the admissible domain.
- `uses` binds the feature to global invariants such as finite arithmetic, fail-closed exposure, point-in-time causality, accounting conservation, tenant isolation, bounded resources, and verified promotion.
- `dependsOn` forms an acyclic refinement graph checked by automation.

The clauses are formal contracts, but evidence levels differ. Bounded enumeration proves only the disclosed finite model; regression/integration evidence is a refinement witness; a static build proves type/build consistency only. The repository does not claim unbounded theorem-prover verification of network services, React orchestration, or deployments.

## Specification index

| Domain | IDs | Covered feature families |
|---|---|---|
| Haskell interfaces/data | `H-INTERFACE` through `H-EXTERNAL` | CLI/API/auth/runtime, refined domains, market integrity, venues, external/PIT evidence |
| Predictors | `H-FEATURES` through `H-KALMAN` | feature schema, tabular/probabilistic/sequence predictors, LSTM, Kalman/online statistics |
| Decisions/trading | `H-TA` through `H-RISK` | TA, signal gates, governors, trading state machine, execution reconciliation, risk halts |
| Evaluation/lifecycle | `H-METRICS` through `H-FORMAL` | metrics/ROI/sensitivity, optimizer, top combos, persistence, telemetry/calibration, executable reference models |
| Haskell programs | `H-EXECUTABLES` | all six tracked entrypoints and their Cabal build membership |
| Web | `W-FORM` through `W-BOOTSTRAP` | form and request domains, transport/security, trading orchestration, truthful presentation, runtime proxy/container |
| Automation/research | `A-AUTOLOOP` through `A-VERIFICATION` | branch/recovery state machine, PIT research, market-prediction registrations, calibration/risk scripts, operations, canonical verification |
| Deployment/CI | `D-FLY-AWS-RENDER`, `D-HETZNER`, `C-CI` | exact-revision promotion, live-trading safety, secrets, role isolation, CI gates |

The registry names every inventoried feature inside coherent feature-family specifications. Its coverage roots enumerate implementation files dynamically, so a new production file fails verification until it is assigned to a contract; the verifier prints the current counts on each run.

`A-AUTOLOOP` treats a permission-denied zero-signal PID probe as proof that the process exists, not as evidence that it is dead. The runner atomically creates a token-bound owner record before shared mutation, quarantines only proven-stale ownership, and releases only its own token. It atomically reserves each bounded cycle identifier before the cycle status/log boundary from the maximum durable metrics, status, incomplete-cycle, and sequence evidence. The merged-runtime restart advanced from legacy-issued cycle 2761 to schema-1 reservation 2762, and a direct concurrent launch failed `EALREADY` without changing owner or status identity; these operational witnesses close the adjacent reset and singleton risks. GitHub repository, review, commit, workflow-run, failed-workflow log, and check-suite reads use a tested 502-only policy with four total attempts and fixed 2, 4, and 8 second backoffs; non-502 errors fail on their first attempt and a persistent 502 propagates after the fourth.

`H-PREDICTORS-SEQUENCE` preserves the historical `tcn`, `patch_tst`, and `transformer` configuration semantics while giving their lightweight proxy implementations explicit versioned identities. `H-EXTERNAL` and `H-FEATURES` additionally define the opt-in `feature_availability_v2` contract: event time and availability time remain distinct, observed zero differs from unavailable evidence through a parallel mask, required missing evidence yields no row, and a deterministic ordered signature binds model inputs to their schema. The v2 external-family bundle preserves those masks and timestamps across duplicate aggregation and revision-safe alignment; the legacy bundle is a one-timestamp dense projection. The opt-in `external_family_model_features_v2` adapter accepts only exact contiguous grids and shapes, preserves the legacy 19-family level/delta order plus masks, and retains causal timestamp witnesses. The isolated Haskell `external_feature_panel_v2` decoder fixes the Python panel's 40-column order and retains values plus fractional coverage, but that panel does not contain selected source timestamp witnesses and cannot be silently projected into the adapter. The separate `binance_derivatives_first_seen_v2` bar decoder validates complete canonical family groups, causal timestamps, family-specific freshness, explicit missingness, and stale neutralization; it accepts pandas' exact zero-fraction timestamp serialization without a precision-losing `Double` conversion. The opt-in `binance_derivatives_model_features_v2` adapter accepts only exact single-symbol contiguous grids, maps usable decoded evidence to the legacy five derivatives formulas plus explicit masks, and retains the causal timestamp witnesses. The opt-in `coinbase_cross_exchange_model_features_v2` contract similarly reproduces the legacy five same-asset basis, z-score, and return-spread formulas only on complete exact-bar evidence, appends explicit masks, retains causal witnesses, and never forward-fills a missing Coinbase bucket. Its required Binance close failures and structural grid/scope failures reject the bundle; unusable optional Coinbase cells remain unavailable. The separate `complete_ohlcv_feature_inputs_v2` source boundary requires exact contiguous grids, explicit first-seen and decision times, complete finite positive OHLC, non-negative volume, and valid candle geometry before projecting only those five fields into the unchanged legacy formulas. It rejects missing core fields rather than invoking synthetic fallbacks and strips unrelated optional channels. `H-MARKET-DATA` additionally defines `point_in_time_liquidity_universe_v2`: every selection comes from one coherent quote-scoped snapshot, binds event/first-seen/decision times and explicit contemporaneous eligibility, applies a bounded event age, selects revisions only after their availability, resolves volume ties by canonical symbol, and normalizes weights without summing raw magnitudes. Missing, stale, incomplete, mixed-scope, or ambiguous snapshot evidence is unavailable. The additive `binance_usdm_market_context_panel_v2` decoder now binds each prospective derived population and optional exact-bar peer vector to a unique frozen source-manifest digest, validates declared counts and ordinals, preserves rolling-ticker and first-seen clocks, and projects only through the canonical v2 selector. A syntactically valid digest is not proof of manifest bytes or complete venue responses, so external manifest verification remains mandatory. The separate `point_in_time_market_context_factor_v2` adapter consumes one selection at every bar, excludes the target before taking the declared peer count, renormalizes weights after exclusion, requires exact-bar completed peer returns, and exposes the raw market return with causal witnesses and an availability mask. Missing or unusable chosen peers make the optional factor unavailable; structural scope, grid, shape, symbol, bar, or decision failures reject the row. Only the peer-basket raw weighted-return formula, without optional same-asset Coinbase augmentation, has complete-input parity; fold-fitted legacy OLS context features require a separate versioned artifact boundary. None of these v2 boundaries is imported by legacy source, feature, artifact, predictor, bot, or execution paths. Current historical OHLCV and Coinbase candle paths do not prove original per-bar first-seen/publication times, and the legacy universe CSV has no separate availability or snapshot-completeness witness, so they cannot be silently projected into the new contracts. `A-RESEARCH` binds every materialized panel manifest to its cache, bar grid, panel bytes, ordered columns, coverage semantics, recomputed populations, and deterministic reconstruction; digest replacement cannot make changed feature rows admissible. Its public Binance collector separately records missing publication metadata as first-seen availability in `binance_derivatives_first_seen_v2` ledgers, preserves later revisions, emits causal observed/fresh masks without relabeling legacy cache cells, and preserves canonical v2 field order across overlap merges. Bounded provider publication lag may leave an explicit trailing unavailable bucket only while the latest finite observation remains inside the unchanged family freshness limit. Conservative local request budgets incorporate the provider's observed shared-IP weight; HTTP 418/429 or Binance `-1003` opens a run circuit, records sanitized failure evidence, and skips the remaining universe without additional calls. Collector status schema 3 binds each accepted bar file and all four raw ledgers by path, rows, hash, schema, source-license record, and code commit; its verifier independently recomputes coverage, validates lag/count arithmetic, and causally reconstructs versioned cells, while schema-2 or throttled statuses remain non-admissible historical evidence. The schema-1 receipt verifier additionally binds the committed acquisition-only metadata to the exact frozen archive, rejects unsafe or extra paths and any outcome/model/order/live authority, and delegates semantic reconstruction to the collector verifier. The first verified final-main receipt commits only the status and 50 artifact identities plus explicit zero outcome/model/order/live use; all downloaded bytes stay outside Git, and the receipt grants no evaluation authority. Production source adapters and predictors remain on the legacy path until source-specific availability, artifact versioning, and prospective validation are complete. `A-MARKET-PREDICTION-RESEARCH` binds preregistered future-data boundaries, complete experiment budgets, model/artifact provenance, unavailable-input abstention, disabled-by-default challenger isolation, and the prohibition on automatic promotion or live authorization. A namesake neural successor is a new semantic model version; it cannot inherit a legacy identifier.

The `A-RESEARCH` rate-limit refinement preserves canonical universe membership
while deriving a separate `utc_epoch_hour_rotation_v1` request permutation.
Schema-3 status records that permutation and the verifier binds it to the UTC
run start; legacy statuses without either additive field retain fixed-order
meaning. A partial field pair, duplicate or missing member, unknown policy, or
time-incoherent order fails closed. Rotation neither increases request volume
nor makes a throttled or incomplete run admissible.

`A-RESEARCH` also defines an offline source verifier for prospective
`binance_usdm_market_context_panel_v2` evidence. The verifier has no network or
credential interface. It binds exact public endpoint parameters, raw byte
hashes and sizes, request clocks, a provider server-time bracket, the complete
exchange-info eligibility population, rolling-ticker evidence, and two exact
completed peer bars. It independently derives close-to-close returns and the
panel bytes, then emits a separate receipt whose outcome, model, order, and
live-authorization fields are all false. Verification cannot substitute for a
collector, admit a dataset into an experiment, open a holdout, promote a model,
or authorize trading.

Collector-produced evidence additionally passes through the offline bundle
verifier. It accepts only the exact `complete_unverified` status with a
published manifest, clean tracked provenance, and all authority fields false;
rejects incomplete, recovery-pending, changed, malformed, or non-finite
evidence; and checks status/manifest clocks, inventory, and population. The
recorded Git object must be a commit whose collector, source-verifier, and
source-license blobs match the collection evidence. The executing bundle and
source verifiers must also be byte-identical to their versions at that commit;
version drift therefore fails before receipt emission. Only then does the
wrapper invoke the independent source verifier and bind its receipt and panel
hashes in a deterministic bundle receipt. The receipt still sets research admission,
experiment, holdout, model, promotion, deployment, order, and live authority
to false.

A saved bundle receipt has a separate offline, read-only replay boundary. The
receipt verifier requires the exact bundle-verifier bytes stored at the
collection commit, strict type-sensitive receipt semantics, and an archive
containing only the status, manifest, and registered raw files. It hashes every
archive file before and after delegated bundle/source reconstruction, compares
the complete recomputed receipt and derived-panel digest, and fails closed on
version drift, tampering, extra files, unsafe paths, or in-run mutation. Its
success summary remains integrity-only and sets research admission, experiment,
holdout, model, promotion, deployment, order, and live authority to false.

The prospective collector that supplies this boundary is a separate
`A-RESEARCH` component. It accepts an explicit aligned bar and a new output
directory, derives its commit from Git only when every provenance file is
tracked and unchanged, binds the loaded provenance bytes to that captured
commit, revalidates the exact binding at both publication boundaries, uses only
fixed public GET endpoints, refuses redirects, and enforces response-size,
causal-window, deadline, local request-weight, and
observed shared-IP weight limits. It requests every contemporaneously eligible
member, bounds each member's ticker event to the ticker request/response window,
and writes the source manifest only after complete coverage. Blocking transport
is bounded by the overall monotonic deadline even if response bytes continue to
trickle without an idle timeout. Its success state is `complete_unverified`;
any transport, provider-throttle, clock,
provenance, response, population, peer-grid, or output failure leaves no source
manifest and grants no downstream authority. File and directory publication is
durably flushed and deadline-checked before successful status. If removal of a
manifest after a final status failure is required, the collector first durably
replaces any possibly visible success status with an indeterminate cleanup
pending sentinel. Only a confirmed removal may advance status to unpublished
partial failure; interruption retains the pending state and failed removal is
recorded as an indeterminate cleanup failure when possible.

The small public backtest-data path is separately fixed-window and hash-bound.
It requires an explicit end time, accepts only exact contiguous completed bars,
records public source/license metadata plus explicit no-randomness provenance,
and binds the generator, normalized source rows, and deterministic CSV bytes by
SHA-256 in a schema-1 manifest. Expected-manifest drift is rejected before any
replacement, and offline verification fails on changed code, provenance,
request identity, metadata, or data bytes. Legacy unmanifested CSV fixtures
remain historical Git evidence and are not retroactively promoted to this
contract.

`H-EXECUTION` also binds restart recovery to a narrow closed-trade-memory
contract. A status snapshot must match symbol, interval, market, and method;
only its bounded `trades` history is decoded. Persisted `positions` and
`openTrade` evidence is intentionally excluded from that recovery type because
`Main.initBotState` derives startup exposure from the venue when trading is
enabled and otherwise starts flat.

The same execution contract binds schema-1.2 live trade-event observability to
the canonical pre-execution halt boundary. Every applied OPEN/CLOSE row retains
the complete v1.1 interface and adds equal `riskState`/`risk_state` snapshots;
drawdown, daily/weekly loss, expectancy availability, their equity references,
and distinct market-event/processing timestamps are preserved. Non-finite or
out-of-domain values become `null` with explicit finite/valid flags, and missing
expectancy remains absent rather than becoming a directional zero. Backtest trade rows deliberately stay
at v1.1 because retroactively approximating the simulator's exact intrabar
daily/weekly decision state would violate the evidence contract.

The current execution-owner report closes `EXECUTION-MISSING-001` as a
reporting obligation while preserving the missed historical deadlines and the
remaining exchange/network, accounting, and end-to-end IO limitations. That
administrative disposition changes no `H-EXECUTION` guarantee and grants no
production-readiness or live-order authority.

The current CIO accountability report separately closes `CIO-DEAFNESS-001`
with an explicit NO-GO while retaining both missed historical deadlines. It
preserves the current champion, the sealed historical holdout, the prospective
evaluation boundary, all three substantive open risks, and all mitigated-risk
conditions; the report itself cannot authorize promotion or an order.

## Verification

Run:

```sh
npm run test:formal
bash scripts/verify.sh automation
bash scripts/verify.sh full
```

The formal registry verifier rejects:

- an implementation file with no specification;
- duplicate spec, global-invariant, or clause IDs;
- missing clauses, implementation scopes, or evidence;
- unknown global invariants or dependencies;
- dependency cycles;
- stale/missing evidence files or markers;
- a safety-critical spec with no executable witness.

Haskell CI additionally forces every Boolean and state-count field in the executable `Formal.*` reports. This prevents lazy, unreferenced obligations from appearing green without evaluation.

## Change rule

Every production feature change must update the matching registry clauses/evidence when behavior changes. A new implementation file must either refine an existing feature family or add a new specification. User-visible behavior changes still require `README.md` and `CHANGELOG.md` updates.

Open conformance gaps and the exact repairs made during the repository-wide audit are recorded in [`formal-verification-audit-2026-07-12.md`](formal-verification-audit-2026-07-12.md).

## Offline sequential-policy boundary (2026-09-17)

`A-SEQUENTIAL-RESEARCH` binds the registered offline environment, causal prefixes,
finite bounded proposals, inventory/reward/terminal accounting, training isolation,
artifact provenance, default-disabled behavior and no promotion/order authority.
`Trader.Research.PolicyProposalV1` is isolated from executables: its constructor
is private, all deterministic evidence guards precede admission, and accepted
proposals still have `orderAuthorized == False`. The test suite exhaustively
checks 2,688 combinations; Python fixtures cover replay/accounting/causality and
three learning paradigms. This is executable contract evidence, not a proof that
markets cannot gap through risk limits or that observations identify an MDP.

`RL-OFFLINE-001` remains HIGH/OPEN because simulator fidelity, joint state–action
support, live behavior propensities, complete statistical comparison and fresh
holdout evidence are missing. Every candidate is rejected. Research does not
alter existing production risk gates, ownership, caps, champion semantics or
order permissions. See the [decision memo](../research-notes/sequential-control-2026-09-17/final-decision-memo.md)
and [environment contract](../research-notes/sequential-control-2026-09-17/environment-contract.md).

The sequential-policy provenance boundary additionally rejects malformed required
metadata even when a caller supplies its matching digest and matching expected
object. Digest syntax and exact integer/family/horizon domains are checked before
save or load; finite JSON applies to additional metadata. The matching-hash
negative regression and 108-artifact compatibility receipt establish this narrow
repair. They do not authenticate an untrusted caller or satisfy economic gates.

The OPE boundary requires nonempty aligned trajectories, finite numeric inputs,
valid integer logged actions, probability/discount domains and zero terminal
bootstrap. Floating-point overflow in weights, moments, estimates or bootstrap
means invalidates the batch rather than emitting NaN/Infinity or clipping weights.
Zero target support remains ESS zero with absent WIS. An enumerated two-action,
two-decision fixture independently checks IS, PDIS, WIS and DR; finite estimates
still grant no promotion authority. `RL-OFFLINE-001` remains HIGH/OPEN.

Sequential CSV admission binds verification and parsing to the same immutable
byte snapshots. Both hashes must match before either CSV is parsed. Run and
policy metadata retain those proven hashes instead of reopening mutable source
paths for provenance. Replacement-race and complete synthetic-run fixtures check
these obligations; no financial experiment or protected holdout is reopened.

Compact report export applies the same obligation to the archive index and every
JSON/event input it parses. The complete indexed inventory is checked before
output creation; non-report files, including the large return CSV, retain bounded
streaming verification. Parsed inputs and exported index provenance refer to the
admitted snapshots even if paths are subsequently replaced. This is evidence
identity, not a lock on the source archive or a new promotion gate pass. Synthetic
replacement regressions and byte-identical reproduction of seven original reports
support `A-SEQUENTIAL-RESEARCH-R5/E4`; `RL-OFFLINE-001` remains HIGH/OPEN.

`A-SEQUENTIAL-RESEARCH-R6/E5` adds cross-file reconciliation before output creation.
Unique roster IDs, event lifecycle, fit/replay metadata and outcomes, verified
successful-fit artifact identities, OPE fit coverage, counts and descriptive group
metrics must agree. Counts are exact integers; finite aggregate comparisons allow
only `1e-12` absolute/relative rounding tolerance. Failed/stopped paths remain in
the aggregates. Deterministic contradictory-archive fixtures and a hand-calculated
mixed-outcome example exercise this boundary. It does not prove the supplied
roster matches an external registration, reconstruct per-bar P&L or make OPE
reliable. Original successful v1 fits without explicit status are supported only
with terminal completion and a matching verified artifact digest.

Every fit and normally evaluated replay must have its producer-defined start
event; only explicit failed-fit cascades may omit it. Evaluated RL rows must also
provide finite nonnegative latency p99 and an OOD observation rate in `[0,1]`
before any output directory is created. Validator and exporter share the same RL
family classification, preventing report-only field requirements from diverging.

Terminal and replay failures require nonblank reasons other than the reserved
`complete` label; completion has no failure reason. Both manifest and summary
must retain the v1 contaminated-development disposition, false promotion/holdout
flags and absent promotion statistics. Contradictory disposition fields fail
before output creation; these checks confer no production authorization.

Peak-memory metadata must be finite and nonnegative. Every compact JSON/CSV report
is rendered before output creation, so field/serialization errors leave no partial
report directory. This does not promise atomic publication across filesystem
failures during the subsequent writes.

The runner includes the exception class in training, OPE and replay failure
records, so empty or reserved exception messages cannot produce blank reasons
or completion labels. Synthetic producer-to-export replay tests retain all
failed paths while keeping promotion disabled.

Training-row reasons obey the same status semantics and must match terminal-event
reasons. Conflicting older failure records are rejected for investigation; no
archive is rewritten to manufacture consistency.

OPE fit coverage requires a nonempty valid v1 payload: failed/invalid outcomes or
conditional estimates with finite metrics, support counts and interval bounds,
and explicit unreliability. A matching fit ID alone is insufficient. These
structural checks do not reproduce estimates or establish behavior-policy support.

Prepared reports are encoded as UTF-8 and written as bytes, preserving explicit
LF terminators without platform text-mode newline translation.

The Python research shield requires exact boolean `True` for enabled, valid and
ownership evidence; timing and actions must be finite real scalars, excluding
booleans, complex values and arrays. Replay checks these gates before observation
reads or processing a pending fill. Rejection ends the path with no simulated
transition or invented liquidation. Inference admits only finite real unmasked
vectors with the declared dimensions and returns an absent proposal on a model
exception. The elapsed-time check remains post-call and does not preempt a hang.

`A-SEQUENTIAL-RESEARCH-R7` requires integer replay indices/horizons and collection
seeds/budgets, excluding booleans; real unmasked one-dimensional market arrays;
matching collection symbol histories; valid behavior-probability representation;
and finite real execution coefficients. Invalid admission fails before sampling.
Only metadata and bounds are inspected at admission. Future missing values do not
invalidate the current observation; feature and transition values are checked
when causally consumed. This does not prove external data provenance or valid
real-exchange fills.

`A-SEQUENTIAL-RESEARCH-R8` prevents incomplete replay outcomes from entering a
training batch. The collector checks time progress, finite reward/equity,
nonterminal successor validity and terminal inventory/pending-state accounting.
An invalid transition aborts the call without returning a partial batch. Fully
accounted risk-triggered terminal losses remain admissible learning samples;
they remain failures for economic promotion. Executable witnesses cover missing
funding, partial decision intervals, insolvency, malformed successors, normal
terminal padding and update isolation across PPO, Double DQN and CQL seeds.

`A-SEQUENTIAL-RESEARCH-R9` validates normalization parameters and snapshots them
onto immutable float64 backing bytes. It rejects incomplete supplied prefixes,
malformed feature/support vectors, nonpositive deviations, inverted support bounds
and non-finite normalized outputs. Replay maps normalization errors to absent
observations before fills. Four executable witnesses cover constructor admission,
array aliasing/write flags, integer-wraparound and overflow, and prefix coverage.
These checks do not prove that supplied parameters came from an authentic
training dataset or that coordinate-wise support implies joint policy coverage.

`A-SEQUENTIAL-RESEARCH-R10` shares the v1 policy parameter contract between save
and load. Writers validate detached numeric snapshots before opening the target;
readers reject nonnumeric JSON leaves before the same shape/value checks. Four
executable witnesses cover invalid save admission, 36-case artifact byte and
inference compatibility, exclusive writes, and runner-to-export failure accounting.
Invalid parameters cannot produce a success digest or a completed-fit artifact.
This does not guarantee atomic filesystem writes or validate statistical merit.

`A-SEQUENTIAL-RESEARCH-R11` validates training controls before model/RNG
initialization: integer horizons/seeds/positive budgets, explicit boolean Q mode
and valid risk coefficients. Integer normalization precedes seed offsets and batch
arithmetic. Four executable witnesses cover rejected initialization, mode/risk
admission, fixed-width overflow boundaries and exact valid-budget behavior. This
does not enforce an upper compute limit or supply new market evidence.


`A-SEQUENTIAL-RESEARCH-R12` records terminal episodes when they occur and adds
optional `episodeAccountingV2` coverage counts. Completion includes accounted risk
stops; budget truncation retains a nonterminal successor and has no invented
episode return. Four executable witnesses cover terminal timing, trainer
aggregation, export rejection of inconsistent/unsupported metadata and exact legacy
report bytes. Count reconciliation establishes internal consistency, not financial
significance or complete coverage inside an aborted training call.


`A-SEQUENTIAL-RESEARCH-R13` separates terminal-ledger and return-path publication
from model/replay exception handling. Three executable witnesses inject terminal
write/flush faults and early/partial CSV faults, and verify one terminal per trial
on a successful runner-to-export path. Publication errors propagate before final
summary/index creation; partial files remain incomplete evidence. Existing
computational-failure accounting is retained. This is not an atomic-write, durable
commit or crash-recovery protocol.


`A-SEQUENTIAL-RESEARCH-R14` admits evidence JSON before reconciliation and rendering.
The index, seven report snapshots and every JSONL event share rejection of duplicate
keys and non-finite numbers, including overflowing exponents and omitted fields.
Three witnesses cover all input routes, nested/escaped collisions, discarded loss
values and valid-number/text compatibility. Existing exact legacy-report and
snapshot-replacement tests remain applicable. This does not prove evidence truth,
require canonical JSON encoding or add an input-size limit.


`A-SEQUENTIAL-RESEARCH-R15` isolates exported reports from their source archive.
Source and destination aliases resolve once before evidence reads; same/descendant
destinations are rejected and later retargeting of the original destination alias
cannot redirect publication. Three witnesses cover direct/nested/aliased paths,
alias retargeting, exact external report compatibility and no-overwrite behavior.
Resolved directory ancestors must remain stable; this is not inode locking or
atomic publication.


`A-SEQUENTIAL-RESEARCH-R16` admits resource metadata before report publication.
Supplied fit/terminal durations are finite nonnegative numbers (not booleans), and
agree exactly when both occur. Artifact byte counts are positive integers; RSS
units must be explicitly `bytes` or `kib` before evidence reads. Three integration
witnesses cover malformed/contradictory fields, API units, and valid measurements.
Legacy missing fields and exact report bytes remain compatible. These are domain
and consistency checks, not proof of measurement accuracy or completeness.


`A-SEQUENTIAL-RESEARCH-R17` gives offline baselines explicit input admission.
All twelve action rules and five forecasting rules reject invalid representations,
shapes, non-finite observations and unknown names with absence before computation
or RNG use. Four witnesses cover the input matrix, name admission, replay shield
precedence and exact valid-output compatibility. Fitted-parameter validity,
timeouts and simulator adequacy remain separate concerns.


`A-SEQUENTIAL-RESEARCH-R18` preserves OPE missingness at admission by rejecting
masked arrays before conversion for all six trajectory/value inputs. Three
witnesses cover partial/full/all-false masks, unchanged ordinary array-like
estimates, and runner/export propagation of explicit failure without discarding
the completed fit. There is no imputation or trajectory exclusion. This does not
make OPE reliable or resolve behavior-support and simulator limitations.


`A-SEQUENTIAL-RESEARCH-R19` checks fitted baseline parameters and arithmetic before
forecast/action selection. Four witnesses cover malformed coefficients/scalars,
overflow before clipping/argmax, replay rejection of absent results, and independent
fixed rules plus finite-logit compatibility. The existing exact valid-baseline
fixture remains applicable. No provenance, calibration, execution-time or economic
claim follows from numerical validity.


`A-SEQUENTIAL-RESEARCH-R20` validates short-OPE policy inputs and outputs on both
logged and direct paths before action selection/transition. Five witnesses cover
malformed output vectors, observation rejection before inference, a preserved
valid action trace, explicit forward exceptions and retained failed-episode accounting. Existing failed-OPE archive
and export tests remain applicable. Finite policy values are not calibrated Q
estimates; neither inference deadlines nor OPE reliability follow from this check.


`A-SEQUENTIAL-RESEARCH-R21` admits complete short-OPE requests before RNG creation.
Four witnesses cover invalid controls, full-universe series coverage, native/NumPy
integer compatibility with unused values left unread, and the smallest valid
six-decision windows at horizons 1/3/6. Types and lengths are checked up front;
market values remain checked when replay consumes them. Budget registration,
data provenance and estimator reliability remain separate requirements.


`A-SEQUENTIAL-RESEARCH-R22` admits stopped replay states for economic reporting.
Successful paths require the final bar, zero inventory, no pending action, full bar
coverage and matching positive finite equity. Four witnesses cover unfinished
paths, corrupted completion state, preservation of explicit failed paths and exact
completed-report bytes. Failed unliquidated paths remain failures; these checks
do not estimate their recoverable value or validate every ledger field.


`A-SEQUENTIAL-RESEARCH-R23` checks contiguous stopped-path ledgers and finite real
inputs before economic metrics. Each bar reconciles equity, P&L, funding and costs
at relative/absolute tolerance 1e-10 and net returns at 1e-12. Four witnesses cover
corrupted returns/ledgers, failed losses beyond initial equity and exact nonzero
report parity. These checks do not reconstruct fills, certify market data or
establish statistical significance. `RL-OFFLINE-001` remains HIGH/OPEN.


`A-SEQUENTIAL-RESEARCH-R24` checks neural forward arithmetic before activation and
before returning policy/value outputs. Four witnesses cover actor/critic vector
and batch failures, absent inference with no fill, OPE rejection before transition,
and finite forward golden parity including valid saturation. Gradient/optimizer
state, parameter provenance and preemptive timeouts remain outside this guard.
`RL-OFFLINE-001` remains HIGH/OPEN.


`A-SEQUENTIAL-RESEARCH-R25` requires explicit failures for invalid optimizer
controls, overflowing clipping norms and invalid Adam arithmetic. Four witnesses
cover actor/critic cold/warm norm overflow, control admission before gradients,
late failures without partial mutation, and stored pre-repair numerical parity.
Parameters, moments and the counter are published only after complete validation.
This preserves pre-existing state on failures before publication under non-mutating helpers and a single writer, including pre-existing corruption;
it is not state repair, a complete optimizer representation contract or a
convergence claim. `RL-OFFLINE-001` remains HIGH/OPEN.


`A-SEQUENTIAL-RESEARCH-R26` pins registration parsing and source/provenance hashes
to bytes checked against one Git commit before dataset access. Four witnesses
cover replacement after validation/data admission/during training, validation of
captured bytes without reopening paths, pre-data rejection, and one-read identity
on valid runs. Later path contents cannot relabel a saved policy. This is file
identity, not interpreter/dependency attestation or independent research evidence.
`RL-OFFLINE-001` remains HIGH/OPEN.

## Scoped machine checks for offline research

`A-FORMAL-RESEARCH` and the `F-RL-*` clauses add a separate proof-status ledger and
[canonical contract](../formal/research/contract.md). `bash scripts/verify.sh formal`
checks SMT formulas, the finite abstract protocol, source-bound receipts and
compiled Haskell conformance; `full` invokes it. The existing registry validator
continues to check coverage and references, not to prove its prose clauses.

See the [runbook and limitations](../formal/research/README.md). All 38 broader
mission obligations remain open/partial and prevent candidate acceptance. The
abstract two-caller protocol does not establish production concurrency refinement.
The exact-real accounting identity does not establish binary64 ledger correctness.
The retained-proposal counterexample refutes revocation by pure mode disabling;
the unchanged private Haskell proposal type still has no order authorization.

The [source-linked causal footprint contract](../formal/research/causal-footprint-contract.md)
extends the scoped research gate with actual-AST read-bound extraction and SMT
checks. This reduces a manually translated index-model gap; it does not prove
Python/NumPy, caller normalization provenance or historical publication timing.
The whole-system acceptance obligations remain open or partially verified.

The [training-prefix contract](../formal/research/training-prefix-contract.md)
adds source-derived normalization and episode index bounds plus registered
fold/horizon checks. Its admission precondition is explicit: the runner validates
the entire historical panel before fitting, so training-value isolation is not
claimed to prove online admission causality or full Replay transition refinement.

The [artifact-admission contract](../formal/research/artifact-admission-contract.md)
adds source-linked atomic admission gates and actual compatibility/digest
predicate checks. The finite 27-state model is conditional on trusted terminating
primitives; SMT uses abstract string equality, exact-false identity and action/
provenance equality atoms. Neither result proves artifact authenticity, helper
internals, neural inference or production-loader refinement. CE-RL-007/008 are
deliberate bypass mutants, not current source defects. Whole-mission obligations
29/30 gain partial evidence; all 38 obligations remain open/partial.

The [training-transition contract](../formal/research/transition-admission-contract.md)
adds source-derived integer/binary64 admission checks and collector ordering
recognition. Terminal inventory must be zero and pending action absent; nonterminal
successors must pass the existing representation, shape and finiteness predicates.
CE-RL-009 is a deliberate guard-removal mutant. Trusted helper semantics, valid
risk classification and complete replay/cash accounting refinement remain open;
whole-mission obligations 18/19 gain partial evidence without becoming complete.

The [terminal-numerics contract](../formal/research/terminal-numerics-contract.md)
checks source-derived exact-real terminal algebra and conditional binary64 zero
products. Current-source CE-RL-010/011 refute exact reward reconstruction and
unconditional finite targets from finite inputs. These refuted claims are
explicitly recorded, not silently treated as passing safety properties. Their
reproduction passes CI as negative evidence; it does not resolve the blockers or
authorize a policy. Batch normalization, complete learner refinement and economic
impact remain unproved.

## Isolated GAE target v2 kernel

See [contract](../formal/research/target-v2-contract.md) and [report](../research-notes/sequential-control-2026-09-17/target-v2-followup-2026-09-28.md). Three scoped SMT requirements cover terminal reward-bit preservation, finite/disabled admission and exact-real recurrence. A separate 1..256-row publication model checks atomic output and a progress rank. Source skeletons, Python primitives and scalarization remain trusted; no normalized-advantage, whole-learner, market or production refinement is claimed. The kernel is default-disabled and disconnected from the frozen runner. Old numerical refutations remain valid.

### Numerical proof-query isolation (2026-09-28)

The [query-isolation contract](../formal/research/query-isolation-contract.md)
refines `F-RL-INTEGRITY` for the terminal-numerics and target-v2 checkers. Independent
solver instances check SAT(P AND W) and UNSAT(P AND NOT C); witness constraints never
enter the latter query. Timeout, random seed, arithmetic domains and claims remain
unchanged. Unknown, invalid premises, counterexamples and solver exceptions block
success. Exact assertion capture, the complete finite result table and actual Z3
regressions are conformance tests, not a universal proof of the checker. The 22 SMT
requirements, bounded models and 38 open/partial mission obligations are unchanged.

### PPO objective and numeric scope (2026-09-28)

The [PPO objective contract](../formal/research/ppo-objective-contract.md) adds two
source-derived exact-real requirements and two explicit refutations. Ratio, clipped
loss and active multiplier follow the registered branches; conditional simplex
components sum to zero. Literals are represented as exact rationals of their
binary64 values. The proofs do not certify NumPy softmax, floating-point accuracy,
a hard trust region, convergence or economic results. CE-RL-012 preserves finite
loss with NaN gradient; CE-RL-013 preserves the intended lack of a uniform clipping
multiplier cap. Actual-source regressions, source mutants and optimizer-state
preservation tests supplement proofs. All 38 broader obligations remain blockers.

### Value-based target and objective scope (2026-09-30)

The [contract](../formal/research/value-objective-contract.md) adds two scoped SMT
requirements and two refutations. The target slice selects with the online network
and evaluates with the target network; the CQL gradient has bounded conservative
components under an assumed exact-real simplex. No machine-checked logarithm,
softmax derivative or CQL policy lower-bound theorem is claimed. CE-RL-014/015
reproduce helper loss failures without proving full-training reachability. Actual
NumPy source tests, 13,122 target-grid cases, 324 gradient cases and 27 ordinary
finite-difference comparisons supplement the proofs. All 38 broader obligations
remain open/partial; no financial trial, champion or live behavior changes.

### Optimizer publication and interruption scope (2026-09-30)

The [contract](../formal/research/optimizer-publication-contract.md) resolves the
broad failure wording in A-SEQUENTIAL-RESEARCH-R25 against actual sequential
attribute stores. F-RL-OPTIMIZER-PUBLISH checks a single-call model with 32 staging
cuts and four stores, under non-mutating primitives and uninterrupted final stores.
F-RL-GRADIENT-CLIP checks exact-real clipping conditional on norm >= abs(grad).
That premise is not a proved property of the rounded NumPy norm.
F-RL-OPTIMIZER-ATOMIC is refuted by CE-RL-016 in an observer/interruption extension
and a deliberate CPython opcode-instrumented regression. This is not an unassisted
thread-race or historical-occurrence claim. Broader atomicity and race freedom
remain open; the correction does not relax any acceptance gate or change training.

## Inference admission and liveness audit

The [contract](../formal/research/inference-boundary-contract.md) binds the unchanged inference predicates, action constants and one-call admission model. Two conditional SMT results cover measured-time guards and first-maximum action selection. CE-RL-017 preserves the pending-call lasso; no timeout or cancellation transition exists. Output validation/selection follows the final clock read. Ordinary Exception fallback is distinct from BaseException, clock or predicate failures. No full runtime/neural refinement, end-to-end latency guarantee or production authority is established. Existing formal/full wrappers enforce this scope.

## OPE algebra and underflow audit

The [contract](../formal/research/ope-algebra-contract.md) fixes two-trajectory exact-real ESS/WIS
bounds and scale invariance, plus DR telescoping for horizons 1..6 conditional on
unit weights, q_t=v_t and zero terminal bootstrap. Eight independent SAT-premise
and UNSAT-violation checks cover three requirements. The [prescribed fixture](../formal/research/ope-counterexamples.json)
refutes positive-weight ESS positivity in the public binary64 helper under ignored
underflow; raised underflow produces an explicit error. All 64 deterministic
six-step target patterns and 201 support-count cases check the current restricted
weight domain separately. These are engineering cases, not market OPE evidence.
Eight new integrity tests run in formal/full; no new tool or frozen-estimator change.
Runtime refinement, unbiasedness, confidence coverage and behavior support remain open.

## Isolated exact ESS diagnostic

The [contract](../formal/research/ess-v2-contract.md) specifies native immutable admitted weights,
exact Fraction conversion/accumulation, zero-mass output and bounded positive ESS.
Source-derived SMT base/step/bound lemmas and a 1..256-row publication model
run in formal/full. The model has 66,309 reachable states, 99,718 transitions,
maximum shortest depth 515 and strict rank bound 517 under terminating primitives.
Eight new tests cover extreme weights, invalid admission and exact-reference
conformance. `ess-rational-v2` defaults disabled and is not imported by existing
research consumers. Dynamic import exclusion and full Python/Fraction refinement
are not proved. CE-RL-018 remains in the frozen helper; no statistical or policy
claim is repaired by this independent arithmetic diagnostic.


### Inherited shutdown deadline repair (2026-10-04)

`F-SHUTDOWN-BUDGET` checks integer deadline bounds with Z3;
`F-SHUTDOWN-STAGES` checks the six-stage finite lifecycle;
`F-SHUTDOWN-CONFORMANCE` records compiled model/implementation comparisons and
runtime failure tests. The [contract](../formal/research/shutdown-deadline-contract.md)
resolves UTC-clock and completion-wording inconsistencies without changing live
configuration. Exact source bodies and hashes bind the proof to
`runServeShutdown` and `Trader.App.GracefulShutdown`; pinned tools run through
`bash scripts/verify.sh formal` and `full`. Source locking plus tests is not a
full IO refinement proof. A-SHUTDOWN-CLOCK and SHUTDOWN-DEADLINE-001 preserve
scheduler/logging assumptions and unresolved worker/route races. Broad obligations
10/21/36 remain unresolved; closure criteria have not been weakened.

### Supervised worker registry follow-up (2026-10-04)

F-WORKER-REGISTRY-LIFECYCLE checks 14,095 states and 55,904 transitions (two workers, two callers, two timeout ticks each). F-WORKER-REGISTRY-INVARIANTS proves seven reviewed Boolean atomic-step predicates with Z3. F-WORKER-REGISTRY-CONFORMANCE runs five compiled tests, 32 fixed-seed scheduling cases, and three preserved old/new regression comparisons. The [contract](../formal/research/worker-registry-contract.md) names A-WORKER-REGISTRY and the model-to-IO refinement gap. Source hashes, traceability and receipts are checked by the existing formal/full gate. SHUTDOWN-DEADLINE-001 remains HIGH/OPEN; no entire broader obligation is closed by this worker-only repair.

### Async job admission follow-up (2026-10-04)

F-ASYNC-ADMISSION-NUMERIC checks six bounded-Int SMT predicates; F-ASYNC-ADMISSION-LIFECYCLE checks two callers at capacities one/two (369 states, 846 transitions, depth 14, rank 20); F-ASYNC-ADMISSION-CONFORMANCE runs 292 numeric cases, seven runtime tests and 32 generated concurrency cases. The [contract](../formal/research/async-job-admission-contract.md) scopes A-ASYNC-ADMISSION, the baseline control-slice adapter and the unresolved full IO refinement, durable persistence and HTTP admission gaps. No broader obligation is closed by these scoped certificates.


Async pool seal follow-up: [contract](../formal/research/async-shutdown-seal-contract.md),
F-ASYNC-SEAL-INVARIANTS (SMT), F-ASYNC-SEAL-LIFECYCLE (finite model) and
F-ASYNC-SEAL-CONFORMANCE (compiled tests) cover local admission closure and
reservation-finalization acknowledgement. Two callers/two sealers at capacities
1/2: 2,932 states, 8,622 edges, maximum shortest depth 17, decreasing rank 206.
Delayed notifications are explicit. A-ASYNC-SEAL names runtime/progress assumptions.
CE-ASYNC-SEAL-001/002 preserve snapshot/result-cell false-completion witnesses.
Source binding plus conformance does not establish full Haskell IO refinement;
whole-server draining and durable ownership remain unresolved. The broader
obligation count remains 3 closed, 28 partial, 7 open.


Backtest gate follow-up: [contract](../formal/research/backtest-gate-contract.md)
adds F-BACKTEST-GATE-NUMERIC, F-BACKTEST-GATE-LIFECYCLE and
F-BACKTEST-GATE-CONFORMANCE. Five integer SMT duration claims, 1,656 finite states /
2,984 edges (two callers, capacities 1/2, immediate/waiting pairs; depth at most 8),
264 compiled numeric cases and seven runtime tests bind the extracted implementation.
AG safety and EF quiescence are checked; retry cycles are retained, so AF termination
and queue fairness are not claimed. A-BACKTEST-GATE names runtime/interruptibility
assumptions. CE-BACKTEST-001–005 preserve timeout, cancellation, overflow and a
separately labeled reserve-gap schedule. Source binding is not full IO refinement.
The broader 38-obligation statuses and economic acceptance gates are unchanged.

## Shared drain and pool admission (2026-10-04)

F-DRAIN-POOL-ORDER checks five SMT algebraic properties.
F-DRAIN-POOL-LIFECYCLE checks 5,376 states / 17,728 transitions across eight
configurations (two callers, two pools, two drainers, capacities 1/2).
F-DRAIN-POOL-CONFORMANCE connects the model to compiled helpers and source-bound
Main constructors: four runtime tests, 32 generated concurrent cases, three preserved
stale-ingress regressions. A-DRAIN-POOL trusts pinned STM primitive semantics.
Pre-drain reservations retain ownership; callback start/quiescence and bot/order
authorization are not certified. The 38-obligation ledger remains 3 closed /
28 partial / 7 open. See `formal/research/drain-pool-contract.md`.

### Source-composed normalization and symbol isolation (2026-10-04)

[Data composition](../formal/research/data-composition-contract.md) closes
obligations 3 and 5 against their existing criteria. F-RL-DATA-COMPOSITION checks
28 actual Scale/scale source references, immutable bytes-backed fields and
loader/caller composition. F-RL-DATA-KEYS discharges six arbitrary-string SMT
queries extracted from panel filters and paired market arguments. Shared training
transforms and policy parameters are explicitly global; local price/funding
observations remain symbol-scoped. Full loader/fit/replay/OPE conformance is an
automation test, not an additional proof class. The Python/pandas/NumPy primitive
contract remains in the assumptions ledger. Current count: 5 closed, 26 partial,
7 open; no mission completion or new economic evidence.


The [PPO successor contract](../formal/research/ppo-successor-v2-contract.md) adds
F-RL-PPO-V2-FLOW/BOUNDS/FINITE/BOUNDARY/CONFORMANCE under A-PPO-SUCCESSOR. The
finite control model, source-derived integer/IEEE guard SMT, enumerated source
boundary and actual synthetic tests retain distinct proof statuses. The successor
composes real training with existing checked kernels but does not close any
additional broad obligation or establish economic acceptance.


The [PPO snapshot/process bridge](../formal/research/ppo-process-bridge-v3-contract.md)
adds F-RL-BRIDGE-V3-CODEC/FLOW/BOUNDARY/CONFORMANCE under A-BRIDGE-V3. Twenty-five SMT
checks cover checked Integer conversion, metadata shape/step admission, completed
PPO step budgets, binary64 finite/bounded selection, decimal-fold bounds and list-count descent. The composed process
model has 83 states and 189 transitions (one request/child; three time buckets;
two cleanup poll steps per window); its maximum shortest depth is 10.
Actual training-to-Haskell tests supplement these conditional abstractions.
A transient exact-bit request does not authenticate provenance, prove observation
causality or establish neural numerical parity near ties. Existing v2 interfaces,
source-level capability separation and disabled defaults remain checked. Counts
remain five scoped closures, 26 partial and seven open; no candidate promotion.

The [PPO artifact v4 contract](../formal/research/ppo-artifact-v4-contract.md)
adds F-RL-ARTIFACT-V4-GUARD/FLOW/BOUNDARY/CONFORMANCE under A-ARTIFACT-V4.
Four conditional SMT checks, a ten-gate finite lifecycle and complete reviewed
source inventory supplement exact trained-state and compiled bit conformance.
Byte identity is distinct from provenance authenticity. Broad 29/30 remain partial.


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

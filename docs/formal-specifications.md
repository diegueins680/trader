# Canonical Formal Specifications

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
source-license blobs match the collection evidence. Only then does the wrapper
invoke the independent source verifier and bind its receipt and panel hashes in
a deterministic bundle receipt. The receipt still sets research admission,
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

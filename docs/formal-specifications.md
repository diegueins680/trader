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
| Automation/research | `A-AUTOLOOP` through `A-VERIFICATION` | branch/recovery state machine, PIT research, calibration/risk scripts, operations, canonical verification |
| Deployment/CI | `D-FLY-AWS-RENDER`, `D-HETZNER`, `C-CI` | exact-revision promotion, live-trading safety, secrets, role isolation, CI gates |

The registry names every inventoried feature inside coherent feature-family specifications. Its coverage roots enumerate implementation files dynamically, so a new production file fails verification until it is assigned to a contract; the verifier prints the current counts on each run.

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

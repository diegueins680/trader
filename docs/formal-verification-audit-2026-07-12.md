# Formal Verification and Correctness Audit — 2026-07-12

## Result

The prior extracted set contained 18 trading-centric specifications and did not cover most backend modules, web behavior, automation, research, deployment, or CI. The new canonical registry covers every enumerated production implementation file and defines shared transition semantics and global invariants. Coverage is machine checked by `scripts/verify-formal-specs.mjs`.

This is bounded/executable formal verification, not a claim of unbounded mathematical proof. The Haskell reference models enumerate disclosed finite domains; tests and builds are refinement witnesses for other code; external systems remain subject to their stated assumptions.

## Correctness defects repaired

1. `Trader.Types.Safe` exported constructors and derived arithmetic instances that violated its own finite/non-negative guarantees. Constructors are now opaque and construction goes through invariant-preserving functions.
2. Three `Trader.Formal.Risk` obligations checked `halt => breach` while claiming `breach => halt`, making non-halts vacuously pass. The bounded checks now verify both breach and non-breach outcomes.
3. Most `Formal.Optimization` fields, one risk field, and new execution fields were never forced by the test suite. CI now evaluates every report Boolean and positive state count.
4. ROI spec and implementation called the same scoring function. The reference formula is now independent.
5. API-route and CORS test suites existed but were not run. They are now wired into `TestMain`.
6. Multivariate LSTM prediction accepted the wrong channel count and silently truncated/reinterpreted inputs. Model parameter shape and query dimension now validate exactly and fail closed.
7. Market-data `Int64` close/freshness arithmetic could wrap and classify overflowed/future bars as fresh/closed. Time arithmetic is checked and yields `MARKET_DATA_TIMESTAMP_OVERFLOW`.
8. Risk metrics accepted non-finite/negative loss and drawdown evidence, and a negative loss-streak limit disabled protection. They now produce explicit invalid-risk halts.
9. Threshold calibration accepted non-finite/negative method parameters and could emit non-finite output from finite-but-overflowing samples. Method and derived outputs now fail closed.
10. Gate telemetry ignored its configured recent-event bound. Recording now honors and preserves a normalized bound.
11. Applied fill fractions could exceed intended exposure, including when requested-base evidence was absent. The implementation and independent reference model now cap/all-or-nothing reconcile against intended exposure.
12. Online statistics and performance/sensitivity calculations could retain or emit NaN/Infinity. Their external boundaries and sensitivity grids now reject or conservatively normalize malformed evidence.
13. `SensitivityAnalysis`, `Formal.RiskRegister`, and `LstmBench` were outside normal Cabal verification. They are now built/exercised by declared components/tests.
14. Hetzner unconditionally supplied `--binance-live` and ignored documented `TRADER_BOT_TRADE=false`. Effective CLI live mode and backend bot starts now enforce the deployment kill switches.
15. Hetzner referenced a nonexistent webhook bridge. A bounded, redacting implementation and health check are now tracked.
16. Fresh web forms started live and armed without operator acknowledgement. Fresh sessions now default to paper/unarmed mode.
17. Autoloop could delete the only unmerged recovery/checkpoint refs and promote non-Haskell branches without their canonical checks. Recovery reachability and diff-targeted promotion gates are now enforced.
18. Research preprocessing standardized from full history, cache refresh could erase prior PIT values with nulls, and edge calibration mixed realized outcomes with decision evidence. Normalization is past-only, cache merge preserves known PIT values, and outcome series is separate.
19. The frontend proxy used a configurable URL with hardcoded Fly Host/SNI. URL, Host, and TLS server identity now derive from the same origin.
20. Top-combo rows were defaulted from malformed payloads, duplicate ranks produced colliding identities, and future timestamps could win freshness selection. Rows are now schema-sanitized before display/auto-apply, identities are deterministic and unique, and future-dated evidence is rejected.
21. Credential tuples were joined with a separator, so distinct credentials containing `:` could derive the same tenant. Backend, browser, and AWS now trim the same ASCII boundary whitespace, preserve legacy keys for separator-free tuples including Unicode credentials, and use a domain-separated, UTF-8 byte-length-framed `platform:v2` identity only for separator-bearing tuples; collision and Unicode-boundary vectors agree across all three runtimes.
22. Successful bot lifecycle responses relied on compile-time TypeScript shapes. Bot start/status/stop now decode the safety envelope before UI mutation, require non-empty aligned core running series, bind applicable response tenant/symbol identities to the request, and reject malformed snapshot or multi-bot evidence without retrying a successful mutation.
23. Risk IDs and lifecycle state drifted between duplicate Markdown rows and a partial Haskell representation. `formal/risk-register.json` is now canonical, with automation enforcing unique ordered IDs and exact severity/status parity in both projections.
24. The research edge campaign was covered only by a broad script glob, leaving its specific nested-validation, formal DSR, CSCV/PBO, derivatives-ablation, and one-shot final-holdout guarantees implicit. `A-RESEARCH` now names those obligations and links direct regression evidence.

## Remaining conformance gaps

The registry specifies these behaviors, but current evidence is incomplete and must not be mistaken for proof:

- `GateTelemetry` is a correct standalone accumulator but is not wired through every production gate; total-bar and candidate counters do not yet have an end-to-end source.
- Bot start/status/stop responses now have a fail-closed safety-envelope decoder. Other TypeScript endpoint response types remain compile-time only and need endpoint-specific runtime decoders.
- Existing separator-bearing tenants need an explicit one-time state-key migration; ambiguous legacy aliases cannot be dual-read safely.
- Web orchestration, SSE EOF/reconnect behavior, and multi-chunk state-sync partial failure lack a full state-machine test harness.
- Client runtime API tokens are public credentials by construction and must never be treated as server secrets.
- Venue signing/filter/pagination behavior, S3/state-sync atomicity, outbox delivery, migrations, and deployment rollback have static/integration witnesses rather than model-checked proofs.
- Optimizer no-lookahead contracts and non-edge research preprocessing paths need generated prefix/walk-forward properties across every preprocessing path.
- Hetzner exact remote projection, watchdog process identity, radio SSRF restrictions, AWS temporary-secret cleanup, and transactional state sync need additional implementation hardening.

The strongest next step is a small pure core library with QuickCheck/Hedgehog properties and SBV/Z3 proofs for time arithmetic, gates, execution, and risk transitions, followed by a state-machine refinement test shared by live and backtest paths.

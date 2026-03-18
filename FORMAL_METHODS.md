# Formal Methods

In a few words: maximize daily ROI without paying for fragility, churn, or inactivity.

## Formalized requirements

The repo already states the optimizer contract in prose. This document turns that into a small executable specification for the ROI objective used by `--tune-objective roi`.

Clauses:

1. Prefer higher annualized return as the repo's daily-ROI proxy.
2. Penalize drawdown and tail loss.
3. Penalize turnover.
4. Reward positive expectancy.
5. Reward faster payback.
6. Penalize low activity and idle capital.

Threshold-sweep tie-break contract:

1. Prefer higher final equity.
2. If equity ties, prefer lower turnover.
3. If turnover ties, prefer more round trips.
4. If still tied, prefer non-inverted hysteresis (`closeThreshold <= openThreshold`).
5. If still tied, prefer the lexicographically larger `(openThreshold, closeThreshold)` pair.

## Where it lives

- Spec + implementation mirror: `haskell/app/Trader/Formal/Optimization.hs`
- Production optimizer usage: `haskell/app/Trader/Optimization.hs`
- Verification harness: `haskell/test/TestMain.hs`

## What is proved

`verifyFormalOptimization` performs bounded exhaustive model checking over:

- `139,968` ROI states
- `11,664` ordered tie-break candidate pairs

The verifier checks:

1. The executable spec and the production ROI implementation return the same score on every modeled state.
2. ROI score is monotone in the intended direction:
   - non-decreasing in return and expectancy
   - non-increasing in drawdown, tail loss, turnover, and slower payback
3. Activity and exposure penalties are ordered as intended.
4. The threshold tie-break implementation matches the lexicographic spec on every modeled pair.

## Formal UI trade-sizing model

The web UI now treats manual `/trade` sizing as a finite-state contract instead of a loose collection of field-level hints.

Clauses:

1. A manual trade is only ready when there is an effective sizing mode.
2. Effective sizing follows a fixed precedence: `orderQuantity` > `orderQuote` > `orderQuoteFraction`.
3. Invalid quote-fraction ranges (`< 0` or `> 1`) block trading only when no higher-precedence valid size is present.
4. Multiple valid sizing inputs are allowed but must be reported as a conflict so the effective mode is explicit.
5. `maxOrderQuote` is cap-only metadata: it never becomes an active sizing mode, never creates trade readiness by itself, and only changes `effectiveLabel` / `hint` when `orderQuoteFraction` is the effective mode.
6. `effectiveLabel` is exact and total for the modeled state space: `Quantity <qty>`, `Quote <quote>`, `Fraction <pct>[ cap <quote>]`, or `No sizing selected`.
7. The blocking-focus target is `orderQuoteFraction` only for fraction-only invalid states; every other state keeps the safe default target `orderQuote`.
8. `statusLabel` is derived from the same finite state: blocked states say `Sizing required`, and ready states name the effective sizing mode.
9. `hint` is exact and state-derived: blocked states surface the blocking error verbatim, conflict states use the precedence warning, and non-conflict ready states mirror `effectiveLabel` as `Effective sizing: ...`.

The verifier in `haskell/web/test/utils.test.mjs` performs bounded exhaustive enumeration over the modeled sizing state space (`32` states), plus paired cap/no-cap comparisons over every `(orderQuantity, orderQuote, orderQuoteFraction)` combination:

- `orderQuantity \u2208 {0, 1}`
- `orderQuote \u2208 {0, 1}`
- `orderQuoteFraction \u2208 {-0.25, 0, 0.5, 1.25}`
- `maxOrderQuote \u2208 {0, 25}`

For every state, it checks:

1. The active sizing modes match the executable spec.
2. The effective sizing mode matches the documented precedence.
3. Trade readiness, blocking reason, and `blockingTargetId` match the modeled state.
4. `effectiveLabel` and `hint` match the modeled string contract for every state.
5. `statusLabel` matches the modeled readiness/effective-mode summary.
6. Conflict severity (`ok` / `warn` / `bad`) matches the modeled state.
7. The quote-cap metadata stays inert outside effective quote-fraction sizing and only changes `effectiveLabel` / `hint` when quote-fraction sizing is actually effective, including restored cap-only form states.

Proof sketch:

- `summarizeOrderSizing` computes precedence before any copy fields, so `effectiveLabel`, `statusLabel`, and `hint` are total functions of the same finite readiness/conflict state.
- The only `maxOrderQuote` interpolation sits inside the `effective === \"orderQuoteFraction\"` label branch and only when `maxOrderQuote > 0`, so cap-only metadata cannot create readiness or alter non-fraction copy.
- `hint` is derived from exactly three branches: the blocking error, the precedence warning for conflicts, or `Effective sizing: ${effectiveLabel}.`; once the modeled state and `effectiveLabel` are fixed, the hint string is fixed too.
- `haskell/web/test/utils.test.mjs` now exhaustively enumerates all 32 modeled states and asserts the exact `effectiveLabel` / `hint` outputs, with paired cap/no-cap comparisons to prove the cap-only inertness contract.

## Formal API base normalization contract

`normalizeApiBaseUrlInput` in `haskell/web/src/app/utils.ts` is treated as a conservative normalizer for the manual API-base field in the web UI.

Clauses:

1. Empty or whitespace-only input normalizes to the empty string.
2. Explicit same-origin path targets that already start with `/` pass through unchanged after trimming, so entries such as `/api` stay local.
3. Bare relative targets whose first segment does not look like a host authority normalize to same-origin paths by gaining a leading slash while preserving any trailing path/query/fragment suffix, so inputs such as `api`, `api/v1`, and `api?tenant=paper#mode=bot` become `/api`, `/api/v1`, and `/api?tenant=paper#mode=bot` instead of cross-origin URLs.
4. Scheme-less loopback hosts (`localhost`, `127.0.0.1`, `0.0.0.0`, bare/bracketed `::1`) infer `http://`, preserving any port and trailing path/query/fragment suffix, so `::1?tenant=paper#mode=bot`, `::1/api`, and `[::1]/api?tenant=paper#mode=bot` stay loopback URLs instead of being rewritten as same-origin paths.
5. Bare or bracketed non-loopback IPv6 literals normalize as direct hosts rather than same-origin paths: bare literals are bracketized, bracketed literals are preserved, and any trailing path/query/fragment suffix is preserved.
6. Scheme-less non-loopback host-like authorities infer a scheme conservatively: default to `https://`, but use `http://` when an explicit non-`443` port is present; preserve the authority and any trailing path/query/fragment suffix. For bare non-loopback IPv6 literals whose trailing `:...` segment is ambiguous, preserve direct-host intent by treating the full authority as the literal instead of guessing a port.
7. Explicit URLs that already contain a scheme (`http://`, `https://`, or any other `://` form) pass through unchanged after trimming.

Proof sketch:

- `normalizeApiBaseUrlInput` trims once and returns `""` for blank input, making empty edits idempotent.
- Before host inference, it now splits unschemed input at the first `/`, `?`, or `#`, so authority detection, loopback checks, IPv6 bracket normalization, and port inference only inspect the authority while the full trailing suffix is appended back verbatim.
- The leading-slash fast path returns `/api`-style values verbatim, so explicit same-origin targets cannot be reinterpreted as hosts.
- After trimming, the first-segment authority check only treats values containing `localhost`, a dot, or a colon as host-like; every other bare relative target falls through to `/${v}`, which preserves same-origin intent and any trailing query/fragment suffix for inputs such as `api`, `api/v1`, and `api?tenant=paper#mode=bot`.
- Host inference only applies after those same-origin branches; the `isLocal` predicate forces loopback authorities onto `http://`, while non-loopback host-like authorities use `https://` by default and switch to `http://` only for explicit non-`443` ports.
- Because the host-like check treats any first segment containing `:` as an authority candidate, bare and bracketed non-loopback IPv6 literals stay on the direct-host path instead of falling through to same-origin `/${v}` rewriting.
- The IPv6 bracket normalization keeps bare/bracketed loopback authorities stable across `::1`, `::1?tenant=paper#mode=bot`, `::1/api`, `::1:PORT`, `[::1]?tenant=paper#mode=bot`, `[::1]/api`, and `[::1]:PORT`, and it bracketizes bare non-loopback IPv6 literals so `2001:db8::1?tenant=paper#mode=bot` and `2001:db8::1/api` remain direct host targets.
- `portFromAuthority` only extracts IPv6 ports from bracketed authorities or the special `::1:PORT` loopback shorthand, so ambiguous bare non-loopback inputs such as `2001:db8::1:8443` preserve direct-host intent by treating the full authority as the literal, while bracketed `[2001:db8::1]:8443?tenant=paper#mode=bot` still triggers the explicit non-`443` `http://` inference.
- The `includes("://")` fast path returns any already-schemed target verbatim, which preserves explicit URLs across HTTP(S) and other URL schemes.
- `haskell/web/test/utils.test.mjs` mirrors this contract with regression rows for blank input, `/api` passthrough, bare relative same-origin rewrites, suffix preservation across `/api`, `api`, loopback hosts, bare/bracketed `::1`, non-loopback host inference (`example.com`, `example.com:8443`, bare/bracketed `2001:db8::1`, ambiguous bare `2001:db8::1:8443`, and bracketed `[2001:db8::1]:8443`), and explicit-URL preservation.

## Formal numeric input fallback contract

`numFromInput` in `haskell/web/src/app/utils.ts` is treated as a conservative parser for shared web numeric fields and restored numeric form fields.

Clauses:

1. Empty or whitespace-only input keeps the supplied fallback unchanged.
2. Signed zero-prefixed single-comma forms remain decimal-comma inputs even when the suffix has exactly three digits, so entries such as `0,125`, `-0,125`, and `+0,125` parse as decimals instead of preserving fallback.
3. A single comma with a signed non-zero 1-3 digit prefix and an exactly 3-digit suffix (for example `1,234`, `12,345`, `-123,456`, `+1,234`) is ambiguous between decimal-comma and thousands grouping, so the fallback is preserved.
4. Other finite single-comma forms parse as decimals, including shorter suffixes such as `1,23` and long-prefix forms such as `1234,567`, `-1234,567`, and `+1234,567`.
5. Explicit multi-group thousands forms such as `1,234,567` remain parseable.
6. After normalization, only finite numeric results are accepted; non-finite results keep the fallback.

Proof sketch:

- `numFromInput` trims first and returns `fallback` on the empty string, so blank edits cannot overwrite a stored numeric value.
- In the two-part comma branch, the signed-zero check runs before the ambiguity guard, so `0,125`, `-0,125`, and `+0,125` normalize to decimal forms even when the suffix length is three digits.
- The ambiguity guard only fires for signed non-zero 1-3 digit prefixes paired with a 3-digit suffix, so `1,234`, `12,345`, `-123,456`, and `+1,234` keep the prior value instead of guessing between decimal-comma and thousands-grouping intent.
- Once the input falls past those branches, the remaining two-part comma forms normalize to decimal-comma values; this is why shorter suffixes and long-prefix inputs such as `1,23`, `1234,567`, `-1234,567`, and `+1234,567` remain parseable.
- The explicit multi-group branch keeps standard thousands-group forms such as `1,234,567` parseable.
- The intended boundary can be described as a bounded sign/prefix/suffix matrix over prefixes `1`, `12`, `123`, `-1`, `-12`, `-123`, `+1`, `+12`, `+123`, `0`, `-0`, `+0`, `1234`, `-1234`, `+1234` and suffixes `2`, `23`, `234`, `2345`: only the signed non-zero 1-3 digit prefix plus 3-digit suffix rows preserve fallback, while every other finite single-comma row parses as a decimal.

## Formal persisted form restore safety

`normalizeFormState` in `haskell/web/src/app/formState.ts` treats restored live-trading settings as a safety-preserving normalization step rather than a blind local-storage replay.

Clauses:

1. `botAdoptExistingPosition` always restores to `true`, even if older saved state persisted `false`.
2. Non-Binance platforms always normalize `market` to `spot`.
3. `binanceLive` can only stay enabled on live-order platforms (`binance` and `coinbase`); every other platform restores it to `false`.
4. `binanceTestnet` can only stay enabled for Binance `spot` / `futures` restores; it always restores to `false` for non-Binance platforms and for any persisted Binance `margin` state, even when that margin state later falls back to `spot`.
5. Binance `margin` restore state is only accepted when live trading is already enabled; otherwise normalization falls back to `market = spot` instead of implicitly enabling live mode.
6. `tradeArmed` is only preserved for live-order platforms; non-trading platforms restore it to `false`.
7. Boolean-like saved strings are normalized by trimming whitespace and ASCII-case-folding before the accepted `true` / `false` / `1` / `0` check runs, so values such as `\" TRUE \"` and `\"False\"` follow the same safety gates as canonical booleans.
8. `binanceSymbol` must restore to a symbol that the active platform sanitizer accepts unchanged; platform-compatible aliases should canonicalize, and any unsanitizable restored symbol must fall back to `PLATFORM_DEFAULT_SYMBOL[platform]`.

Proof sketch:

- `normalizePlatform` first bounds the platform domain.
- `normalizeBool` first accepts native booleans, then trims and lowercases string inputs before checking the accepted `true` / `false` / `1` / `0` domain, so values such as `\" TRUE \"` and `\"False\"` enter the same restore branches as canonical booleans instead of falling back to defaults.
- `restoredBinanceMarket` captures the bounded persisted Binance market before any safety fallback runs.
- `market` is initialized as `spot` for every non-Binance platform, so stale `margin` and `futures` values cannot leak outside Binance.
- `liveOrdersSupported` gates `binanceLiveCandidate`, forcing `binanceLive = false` on non-trading platforms while still allowing supported Coinbase live restores.
- The Binance margin branch rewrites `market` back to `spot` when the restored state is `margin` without live mode, preserving the current safe fallback instead of upgrading to live orders.
- `binanceTestnet` is gated by the original bounded Binance market, so a stale margin-only testnet toggle cannot survive either a non-Binance restore or a margin-to-spot safety fallback.
- `tradeArmed` is gated separately to Binance and Coinbase, so a non-trading platform cannot restore an armed trade toggle.
- `normalizeSymbol` now returns `sanitizeSymbolForPlatform(platform, raw) ?? PLATFORM_DEFAULT_SYMBOL[platform]`, so restore-time symbol hydration is total for the active platform: compatible aliases canonicalize through the sanitizer, and unsanitizable persisted symbols fall back to the current platform default instead of surviving as invalid UI state.
- The returned object overwrites any persisted value with `botAdoptExistingPosition: true`, so reloads keep orphaned-position adoption enabled by construction.
- `haskell/web/test/utils.test.mjs` continues to mirror the boolean/live-trading restore invariants with representative restored states plus an exhaustive 4 x 3 x 2 x 2 platform/market/live/testnet matrix, including mixed-case/whitespace regression rows for `binanceLive`, `binanceTestnet`, `tradeArmed`, and representative boolean toggles.
- `haskell/web/test/formSymbolBehavior.test.mjs` adds cross-platform restore regressions proving the symbol branch: Binance-style persisted symbols fall back to Coinbase/Poloniex defaults when they cannot be sanitized for those platforms, while stale delimited Binance/Kraken restores canonicalize to platform-valid symbols.

## Formal autoloop safety contract

The GitHub autoloop runner treats model output as untrusted input and checks a small executable contract before any commit is created.

Clauses:

1. Every proposed path must be relative, non-empty, and traversal-free.
2. A patch plan may not repeat the same path twice.
3. Every bounded cycle must name one local UI review file under `haskell/web/src/` and one correctness/formal review file under `test/`, `haskell/test/`, `haskell/web/test/`, or `FORMAL_METHODS.md`; both files must already be part of the inspected file set.
4. A patch plan may only modify files that were explicitly requested for inspection by the idea-selection phase.
5. The patch plan must report both review outcomes: `uiReviewSummary` for the UI/UX pass and `correctnessSummary` for the invariant/property/test or proof-sketch pass.
6. Verification commands must come from a fixed allowlist.
7. Git staging is restricted to the planned files that actually changed, so generated byproducts outside the plan are excluded from bot commits.
8. The repo-local forever runner may write only under the gitignored `.tmp/autoloop/` state directory, must honor a stop file or `SIGINT`/`SIGTERM`, and must wait between cycles instead of spinning.

The verifier in `test/autoloop.test.mjs` checks the JSON/path normalization and review-target clauses, `scripts/autoloop.mjs` enforces the subset/staging/phase clauses at runtime, and `scripts/autoloop-forever.mjs` enforces the local stop/sleep/state-directory clauses for the persistent supervisor.

## What is not proved

This does not prove that:

- the predictors forecast correctly
- the trading strategy is profitable on real markets
- exchange integrations, network I/O, or persistence layers are free of bugs
- floating-point arithmetic is globally free of all numerical issues

It proves that the repo's stated ROI contract, as encoded in the optimizer, matches the implementation over the modeled state space.

## Running the verifier

```bash
cd haskell
cabal test
```

## Formal close-timing model (combo-aware)

For each position with open time `ta` and realized close time `tc`, we define an optimization window:

- `tm \u2208 [ta, ta + 2*(tc-ta)]`

`optimalCloseObservation` first filters the PnL path to finite candidates whose timestamps stay inside that window. If `tc <= ta` or no candidate survives, no observation is emitted. Otherwise `tm` is selected as the timestamp that maximizes path PnL inside the filtered window.

Close-timing selection invariant:

1. `tm` must maximize path PnL over all in-window observations.
2. If multiple in-window timestamps share the maximum path PnL, the model selects the earliest such timestamp.
3. The earliest-max rule must be invariant to input path order, so downstream stats and risk-budget targets do not depend on how equal-PnL samples arrive.

We then normalize by realized duration:

- `r = (tm-ta)/(tc-ta)`, so `r \u2208 [0,2]`

Before fitting per-combo stats, `buildCloseTimingStats` drops any invalid stored observation:

- realized duration must be positive (`tc > ta`)
- optimal-close time must stay inside the modeled window, so normalized `r` remains in `[0,2]`

For stats emitted by `buildCloseTimingStats`, we estimate robust distribution statistics over `r`:

- median (`Q50`) as center
- MAD (`median |r-Q50|`) as robust dispersion
- interquartile band (`Q25`, `Q75`) as a policy interval
- `boundedPercentile` clamps every percentile back into `[0,2]`, and `orderQuartiles` sorts the fitted quartiles so `0 <= Q25 <= Q50 <= Q75 <= 2`

A risk-budgeted close policy is encoded as a convex blend target:

- `beta = clamp(0, 1, riskBudget)` when the supplied budget is finite; otherwise `beta = 0`
- `closeTimingDecision` re-clamps `Q50`, promotes `Q75` to at least `Q50`, and computes `target = clampRatio ((1-beta)*Q50 + beta*Q75)`
- therefore `target` stays finite and inside `[Q50, Q75] \u2286 [0,2]`

A live position is marked close-ready when its age ratio meets or exceeds `target`.

Close-readiness boundary invariant:

1. `ageRatio < target` implies hold.
2. `ageRatio >= target` implies close-ready, including the exact boundary `ageRatio == target`.

Bounded-arithmetic invariant:

1. Window membership is evaluated in mathematical integer space before comparing timestamps, so a mathematically valid `tm` is not dropped when `ta + 2*(tc-ta)` exceeds `maxBound :: Int`.
2. Boundary windows such as `ta = minBound :: Int`, `tc = 0`, `tm = maxBound :: Int` remain admissible whenever they satisfy the mathematical window.
3. Observation validity and normalized `r` use mathematical integer deltas for `tm-ta` and `tc-ta`, so full-span `Int` observations such as `ta = minBound :: Int`, `tc = maxBound :: Int`, `tm = maxBound :: Int` retain `r = 1`.
4. Live age ratios use the same overflow-free delta arithmetic, so full-span `Int` close-readiness remains finite and depends on the modeled timestamps instead of machine-width wraparound.

The regression checks in `haskell/test/TestMain.hs` cover representative window selection and the boundary risk-budget decisions (`beta = 0` and `beta = 1`). This proof sketch now also makes the invalid-sample dropping (`tc <= ta`, `tm < ta`, and `tm > ta + 2*(tc-ta)`) and non-finite-budget contract (`NaN`, `+Infinity`, and `-Infinity` normalize to the same median-target policy as `beta = 0`) explicit, matching `observationRatioParts`, `validObservation`, `decisionTargetBand`, `normalizeRiskBudget`, and `clampRatio` in `haskell/app/Trader/Formal/CloseTiming.hs`. The local `isFinite` guard in `optimalCloseObservation` also excludes non-finite PnL candidates before `tm` selection, while `normalizeRiskBudget` plus `clampRatio` keep the downstream `ctdTargetRatio` finite and bounded inside `[0,2]`, so this formatter-only repair does not change runtime behavior.

### Implementation pointers

- Model + policy: `haskell/app/Trader/Formal/CloseTiming.hs`
- Unit tests: `haskell/test/TestMain.hs`
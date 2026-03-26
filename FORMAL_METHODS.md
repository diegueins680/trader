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

## Formal sampled-chart downsampling contract

The sampled backtest-chart path treats `downsampleIndices`, `downsampleArray`, `downsampleOptionalArray`, and `remapIndexToSample` in `haskell/web/src/app/utils.ts` as a shared projection/remap contract for every downsampled chart series.

Clauses:

1. `downsampleIndices(total, maxPoints)` is total: it truncates finite inputs, treats non-finite `total` as `0`, treats non-finite `maxPoints` as the lossless identity budget `total`, clamps the effective sample budget to at least `1`, returns `[]` for `total <= 0`, and otherwise returns at least one in-bounds raw index.
2. The sampled index list is strictly increasing, starts at raw index `0` whenever data is present, and preserves the last raw endpoint `total - 1` whenever the effective budget can show at least two points.
3. The sampled index count never exceeds `min(total, effectiveBudget)` and grows monotonically as `maxPoints` increases, saturating at the raw series length once the budget is large enough.
4. `downsampleArray` and `downsampleOptionalArray` are order-preserving projections over that sampled index list: each sampled element equals the raw element at the corresponding sampled index, and optional `null`/`undefined` inputs stay absent.
5. `remapIndexToSample(indices, idx)` is total: exact sampled hits map back to their exact sampled slot, and every other raw index maps to one of the nearest visible sampled points.
6. Nearest-point remapping is deterministic and left-biased on ties, and the returned sampled slot is monotone non-decreasing as the raw index increases.

The verifier in `haskell/web/test/utils.test.mjs` now checks this contract with:

- a bounded sample-budget matrix (`17,028` states) over `total ∈ {0..257}` and `maxPoints ∈ {0..65}`
- explicit fractional/non-finite normalization rows for `downsampleIndices`
- aligned projection checks for required and optional arrays over that same matrix
- a bounded remap matrix (`2,188,098` raw-index cases) over every `rawIdx ∈ {0..total-1}` for `total ∈ {1..257}` and `maxPoints ∈ {0..65}`

For every state, it checks:

1. Sampled indices stay in bounds, within budget, and strictly increasing.
2. First/last visible endpoints are preserved according to the budget rules.
3. Sample count grows monotonically with `maxPoints` and saturates once the budget reaches the raw length.
4. Fractional inputs truncate before sampling, non-finite totals fall back to `[]`, and non-finite budgets fall back to the lossless identity projection.
5. Required and optional sampled arrays remain pointwise aligned with the sampled indices, while nullish optional sources remain absent.
6. Exact-hit remaps return the exact sampled slot.
7. Non-hit remaps choose the documented nearest sampled point, use deterministic left-biased ties, and never move backward as raw indices increase.

Proof sketch:

- `downsampleIndices` now normalizes the raw series length before any array allocation: non-finite totals collapse to `0`, and non-finite budgets reuse the finite raw length as the lossless identity budget, so malformed numeric inputs cannot produce `NaN`/`Infinity` lengths or drop the leading endpoint.
- `downsampleIndices` emits candidates in increasing `i` order across the closed range `[0, n - 1]` and skips duplicates with the `last` guard, so every emitted raw index is in bounds and the output is strictly increasing.
- The `n <= max` identity branch proves saturation, while the `max === 1` branch explains why the one-point budget preserves only the first visible endpoint.
- For `max >= 2`, the first loop iteration emits `0` and the final iteration reaches `n - 1` (with a final endpoint append as a fallback), which yields the two-endpoint preservation rule whenever two visible points are available.
- `downsampleArray` is just `indices.map((idx) => arr[idx])`, and `downsampleOptionalArray` either returns the nullish source as absent or delegates to that same projection, so alignment follows directly from the sampled index list.
- `remapIndexToSample` binary-searches the insertion interval and compares absolute distances to the two surrounding sampled points; the `<=` comparison makes equal-distance ties resolve to the left neighbor deterministically.
- Because the sampled indices are strictly increasing, the midpoint boundaries between adjacent sampled points are ordered, so the nearest-sample slot selected by `remapIndexToSample` cannot decrease as raw indices increase.
- `haskell/web/src/components/BacktestChart.tsx` consumes this contract through `indexFor` and `remapIndexToSample` for sampled hover/marker rendering, so proving the helper contract bounds the sampled chart alignment behavior without changing the chart logic in this cycle.

## Formal orphaned-position classification contract

`buildOrphanedPositions` in `haskell/web/src/app/utils.ts` is treated as a precedence-ordered classifier for the orphaned-operations panel.

Clauses:

1. Finite zero/dust positions are not orphan candidates at all: they are treated as flat and suppressed before any reason classification.
2. Market scoping is symbol-local: exact target-market bots and unknown-market snapshots remain in scope for the requested panel market, while same-symbol bots from a different explicit market only contribute fallback `market mismatch` evidence.
3. Adoption is reconcile-first: trade-enabled running bots with a matching side are adopted, and trade-enabled running or starting bots with no known internal side are also treated as adopted/reconciling when the position side is known, so flat or initializing bots do not surface false orphan warnings.
4. Once a position is not adopted, the displayed reason string follows a total precedence order: `market mismatch` -> `bot stopped` -> `trading disabled` -> `position side unknown` -> `bot side unknown` -> `side mismatch (bot ...)` -> `no bot`.
5. Other-market evidence is only allowed to win when there is no in-scope status for the symbol; stopped or active unknown-market snapshots must outrank same-symbol other-market bots.
6. The representative `status` attached to an orphan stays tied to the in-scope status set: prefer a running status, otherwise an in-scope active status, otherwise the first in-scope stopped snapshot; pure `market mismatch` / `no bot` cases keep `status = null`.

The verifier in `haskell/web/test/utils.test.mjs` now checks this contract with a bounded orphan-state matrix (`180` states):

- `marketScope \u2208 {none, other-only, target-only, unknown-only, unknown+other}`
- `lifecycle \u2208 {stopped, starting, running}`
- `tradeEnabled \u2208 {false, true}`
- `positionSideKnown \u2208 {false, true}`
- `botSide \u2208 {match, mismatch, unknown}`

`positionSideKnown = false` is modeled with a non-finite amount rather than a zero amount so the matrix still exercises the true `position side unknown` branch after the flat-position contract removes zero/dust rows from orphan classification.

For every state, it checks:

1. Adopted/reconciling states return no orphan row.
2. Non-adopted states return exactly one orphan row with the precedence-ordered reason.
3. Unknown-market stopped snapshots keep beating same-symbol other-market evidence.
4. The representative `status` remains the in-scope status for scoped states and `null` for pure `market mismatch` / `no bot` states.
5. Separate zero/dust regressions prove that stale explicit sides do not manufacture orphan rows for effectively flat positions.

Proof sketch:

- `buildOrphanedPositions` partitions bot evidence into an in-scope `statusesBySymbol` map and an `otherMarketSymbols` set. Because the market filter only excludes explicit non-target markets, unknown-market snapshots stay in scope instead of being downgraded into `market mismatch` evidence.
- `buildOrphanedPositions` now exits early for finite zero/dust amounts, so stale exchange-side metadata cannot manufacture orphan rows for effectively closed positions.
- The adoption check runs before the reason chain and treats running/starting bots with no known internal side as adopted whenever the position side is known, preserving the current flat/startup suppression behavior.
- The remaining `if` / `else if` chain is a fixed ordered precedence model: market scope first, then lifecycle, trade-enabled state, position-side knowledge, bot-side knowledge, and finally side mismatch or no bot.
- `haskell/web/test/utils.test.mjs` mirrors the contract with the bounded 5 x 3 x 2 x 2 x 3 enumeration plus targeted regressions for hedge-side mismatches, flat-running adoption, starting adoption, stopped unknown-market precedence, and zero/dust stale-side suppression.

## Formal flat-position side contract

`positionSideFromAmount` in `haskell/web/src/app/utils.ts`, `positionSideInfo` / `inferBinancePositionOpenTime` in `haskell/web/src/app/appHelpers.ts`, and the Open positions/orphaned-operations consumers in `haskell/web/src/App.tsx` now share one amount-first flat-position contract.

Clauses:

1. Finite position amounts with `abs(amount) <= 1e-12` are flat, regardless of stale `positionSide` metadata.
2. Directional side inference only starts once the amount is outside that flat epsilon.
3. For directional amounts, explicit hedge-side metadata (`LONG` / `SHORT`) still outranks raw sign inference.
4. Flat positions must not render as orphan candidates and must not appear in the Open positions panel.
5. Open-time inference is only defined for directional positions, so flat amounts return `null`.

The verifier in `haskell/web/test/utils.test.mjs` and `haskell/web/test/appHelpers.test.mjs` checks representative zero/dust and directional states:

- `positionSideFromAmount(0)` and `positionSideFromAmount(±1e-13)` collapse to `null`
- `positionSideInfo(0, "LONG")` and `positionSideInfo(1e-13, "SHORT")` collapse to `{ dir: 0, label: "FLAT", key: "FLAT" }`
- `buildOrphanedPositions` drops zero/dust positions even when stale explicit sides are present
- `positionSideInfo(2, "SHORT")` preserves the directional hedge side

Proof sketch:

- `isEffectivelyFlatPositionAmount` is now the shared gate for flatness, so side inference, orphan classification, and open-time inference no longer disagree on whether a zero/dust amount is directional.
- `positionSideInfo` checks flatness before consulting `positionSide`, which prevents stale hedge-side metadata from manufacturing a `LONG` / `SHORT` label for an effectively closed position.
- `buildOrphanedPositions` exits early on flat amounts, so the orphan panel cannot surface phantom rows for positions that the rest of the UI already treats as flat.
- `App.tsx` filters the raw `/binance/positions` list through `positionSideInfo(...).dir !== 0`, making the Open positions panel a direct consumer of the same flatness contract instead of duplicating ad hoc zero checks.

## Formal request-issue ordering contract

`buildRequestIssueDetails` in `haskell/web/src/app/utils.ts` is treated as a straight-line request-validation contract for the web UI fetch/action gating.

Clauses:

1. Issue rows are emitted in a fixed priority order: `rateLimit -> apiStatus -> symbol -> interval -> lookback -> apiLimits`.
2. The symbol branch is mutually exclusive: `missingSymbol` emits `Symbol is required.` and suppresses `symbolError`; otherwise a truthy `symbolError` emits the symbol row.
3. Falsy optional inputs never emit rows, and `apiBlockedReason` by itself is inert when `apiStatusIssue` is absent.
4. `disabledMessage` is only attached to the API-status row, where it equals `apiBlockedReason ?? apiStatusIssue`.
5. The first actionable `targetId` is the first populated target in that same row order, so untargetable rows such as `rateLimit` do not steal focus from later actionable issues.

The verifier in `haskell/web/test/utils.test.mjs` now checks this contract with:

- a bounded issue-presence matrix (`192` states):
  - `rateLimit ∈ {absent, present}`
  - `apiStatus ∈ {absent, issue, blocked}`
  - `symbol ∈ {absent, missing, error, both}`
  - `interval ∈ {absent, present}`
  - `lookback ∈ {absent, present}`
  - `apiLimits ∈ {absent, present}`
- a sparse-target matrix (`32` states) over present/absent target IDs for the actionable rows
- a falsy-input inertness matrix (`243` states) over `rateLimitReason`, `apiStatusIssue`, `symbolError`, `lookbackError`, and `apiLimitsReason` in {`undefined`, `null`, `""`} with `apiBlockedReason` set but no other enabled issue

For every state, it checks:

1. The emitted row list matches the documented order and per-row message contract.
2. Missing-symbol rows beat symbol-error rows.
3. Falsy issue inputs create no rows.
4. `apiBlockedReason` only affects the API-status row `disabledMessage`.
5. The first actionable `targetId` matches the first populated target in precedence order.

Proof sketch:

- `buildRequestIssueDetails` is straight-line code that appends to a single `issues` array, so emitted order is exactly the source branch order.
- The symbol branch uses `if (missingSymbol) ... else if (symbolError) ...`, which makes the missing-symbol row dominate symbol-error output by construction.
- `disabledMessage` only appears inside the API-status push branch; no other branch assigns it.
- Every other row is gated by a direct truthiness check, so falsy optional inputs are inert, and `apiBlockedReason` cannot create a row without the enclosing `apiStatusIssue` branch.
- Because only the actionable rows carry `targetId`, and they are appended in precedence order, the UI first actionable target is simply the first emitted row with a populated `targetId`.

## Formal latest-signal badge contract

`latestSignalTone` and `actionBadgeClass` in `haskell/web/src/app/utils.ts` are treated as the shared classifier from latest-signal action strings to UI tones.

Clauses:

1. Classification depends only on the trimmed first token of the action string; any explanatory suffix is inert.
2. `LONG` maps to the shared `bullish` tone.
3. `SHORT` and `FLAT` map to the shared `bearish` tone.
4. Every other head token maps to the shared `neutral` tone.
5. `actionBadgeClass` is a total projection of that shared tone: `bullish -> badgeLong`, `bearish -> badgeFlat`, `neutral -> badgeHold`.
6. All latest-signal UI surfaces must consume the shared tone instead of duplicating token parsing, so `FLAT` cannot render bearish in one panel and neutral in another.

The verifier in `haskell/web/test/utils.test.mjs` checks representative head-token cases with and without explanatory suffixes.

Proof sketch:

- The backend latest-signal action strings are emitted as a semantic head token (`LONG`, `SHORT`, `FLAT`, `HOLD`) optionally followed by explanatory text, so splitting on the first token preserves the executable signal while ignoring annotation text.
- `latestSignalTone` is now the only head-token classifier, so every consumer starts from the same semantic tone before mapping into surface-specific CSS classes.
- `actionBadgeClass` becomes a pure tone-to-badge projection, which keeps the header/latest-signal badges aligned with Live visuals and prevents `SHORT` or `FLAT` actions from drifting into a neutral tone on one surface while staying bearish on another.

## Formal API base normalization contract

`normalizeApiBaseUrlInput` in `haskell/web/src/app/utils.ts` is treated as a conservative normalizer for the manual API-base field in the web UI.

Clauses:

1. Empty or whitespace-only input normalizes to the empty string.
2. Explicit same-origin path targets that start with exactly one `/` pass through unchanged after trimming, so entries such as `/api` stay local.
3. Inputs that begin with two or more leading slashes must not leak through unchanged. After stripping that unsafe slash run, host-like authorities (for example `//example.com/api`, `///example.com/api`, or `//localhost:8080/api`) normalize to explicit direct-host URLs by inferring the same conservative scheme that the scheme-less host branch would choose.
4. Bare relative targets whose first segment does not look like a host authority normalize to same-origin paths by gaining a leading slash while preserving any trailing path/query/fragment suffix, so inputs such as `api`, `api/v1`, `api:v1`, `api?tenant=paper#mode=bot`, `//api`, and `///api` become `/api`, `/api/v1`, `/api:v1`, `/api?tenant=paper#mode=bot`, `/api`, and `/api` instead of cross-origin URLs.
5. Scheme-less loopback hosts (`localhost`, `127.0.0.1`, `0.0.0.0`, bare/bracketed `::1`) infer `http://`, preserving any port and trailing path/query/fragment suffix, so `::1?tenant=paper#mode=bot`, `::1/api`, and `[::1]/api?tenant=paper#mode=bot` stay loopback URLs instead of being rewritten as same-origin paths.
6. Bare or bracketed non-loopback IPv6 literals normalize as direct hosts rather than same-origin paths: bare literals are bracketized, bracketed literals are preserved, and any trailing path/query/fragment suffix is preserved.
7. Scheme-less non-loopback host-like authorities infer a scheme conservatively: default to `https://`, but use `http://` when an explicit non-`443` port is present; preserve the authority and any trailing path/query/fragment suffix. For bare non-loopback IPv6 literals whose trailing `:...` segment is ambiguous, preserve direct-host intent by treating the full authority as the literal instead of guessing a port.
8. When the normalizer synthesizes an explicit `http(s)://` target from an unschemed input, the result must itself be a parseable URL; malformed pseudo-authorities fall back to same-origin `/${source}` normalization instead of emitting invalid direct-host URLs.
9. Explicit URLs that already contain a scheme (`http://`, `https://`, or any other `://` form) pass through unchanged after trimming, except that a stray leading protocol-relative prefix is collapsed first (`//https://example.com` -> `https://example.com`).

Proof sketch:

- `normalizeApiBaseUrlInput` trims once and returns `""` for blank input, making empty edits idempotent.
- The single-slash fast path now only accepts exactly one leading slash. Any longer slash run is stripped before the ordinary authority parser runs, so values such as `//api`, `///api`, `///example.com/api`, and `////localhost:8080/api` cannot leak through as scheme-relative browser URLs.
- Before host inference, it now splits unschemed input at the first `/`, `?`, or `#`, so authority detection, loopback checks, IPv6 bracket normalization, and port inference only inspect the authority while the full trailing suffix is appended back verbatim.
- The leading-slash fast path still returns exactly-one-slash `/api`-style values verbatim, so explicit same-origin targets cannot be reinterpreted as hosts, while multi-slash prefixes are forced back through the conservative host-vs-path decision.
- After trimming, the first-segment authority check still routes obvious host candidates through the direct-host path, but the synthesized `http(s)://...` candidate is now validated before it can escape; malformed pseudo-authorities such as `api:v1`, `tenant:demo/path?mode=paper#bot`, and `example.com:tenant/api` therefore fall back to same-origin `/${source}` normalization instead of producing invalid cross-origin URLs.
- Host inference only applies after those same-origin branches; the `isLocal` predicate forces loopback authorities onto `http://`, while non-loopback host-like authorities use `https://` by default and switch to `http://` only for explicit non-`443` ports.
- Because the host-like check treats any first segment containing `:` as an authority candidate, bare and bracketed non-loopback IPv6 literals stay on the direct-host path instead of falling through to same-origin `/${v}` rewriting.
- The IPv6 bracket normalization keeps bare/bracketed loopback authorities stable across `::1`, `::1?tenant=paper#mode=bot`, `::1/api`, `::1:PORT`, `[::1]?tenant=paper#mode=bot`, `[::1]/api`, and `[::1]:PORT`, and it bracketizes bare non-loopback IPv6 literals so `2001:db8::1?tenant=paper#mode=bot` and `2001:db8::1/api` remain direct host targets.
- `portFromAuthority` only extracts IPv6 ports from bracketed authorities or the special `::1:PORT` loopback shorthand, so ambiguous bare non-loopback inputs such as `2001:db8::1:8443` preserve direct-host intent by treating the full authority as the literal, while bracketed `[2001:db8::1]:8443?tenant=paper#mode=bot` still triggers the explicit non-`443` `http://` inference.
- The `includes("://")` fast path returns any already-schemed target verbatim after that protocol-relative collapse, which preserves explicit URLs across HTTP(S) and other URL schemes.
- `haskell/web/test/utils.test.mjs` mirrors this contract with regression rows for blank input, exactly-one-slash `/api` passthrough, multi-slash authority/path cases (`//api`, `///api`, `///example.com/api`, `////localhost:8080/api`), bare relative same-origin rewrites (including colon-bearing non-authorities), suffix preservation across `/api`, `api`, loopback hosts, bare/bracketed `::1`, non-loopback host inference (`example.com`, `example.com:8443`, `api:8443`, bare/bracketed `2001:db8::1`, ambiguous bare `2001:db8::1:8443`, and bracketed `[2001:db8::1]:8443`), and explicit-URL preservation.

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

## Formal exact-integer input contract

The web integer parsers and integer-only restore path treat exactness as part of validity rather than accepting any finite `Number(...)` result.

Clauses:

1. `parseMaybeInt`, `parseOptionalInt`, and the raw-millisecond branch of `parseTimeInputMs` only accept integers that are exactly representable as JavaScript safe integers.
2. `parseDurationSeconds` only accepts safe integer magnitudes and only returns a duration when the post-unit product is still a safe integer.
3. Integer-only restore fields normalized through `parseFiniteInteger` / `normalizeWholeNumber` only accept safe integers; fractional, non-finite, and unsafe integer-like persisted values fall back to the documented safe defaults before any clamp is applied.
4. Safe but out-of-range integer restores still clamp through the existing request-aligned bounds, so the change only removes rounded-invalid states, not legitimate bounded restores.

Proof sketch:

- Each integer parser now guards its numeric result with `Number.isSafeInteger`, so strings such as `9007199254740993` cannot round to `9007199254740992` and survive as a different value than the user typed.
- `parseDurationSeconds` checks exactness twice: once after parsing the integer magnitude and again after multiplying by the unit seconds, so overflowed duration products cannot slip through as rounded whole numbers.
- `normalizeWholeNumber` only consumes `parseFiniteInteger` outputs, and `parseFiniteInteger` now accepts only safe integers, so restored bar/count fields cannot enter the clamp path from a rounded-invalid precursor state.
- `haskell/web/test/binanceTradeIpMap.test.mjs` and `haskell/web/test/utils.test.mjs` pin the boundary with `Number.MAX_SAFE_INTEGER + 1`, overflowed duration products, and restored integer-only fields, proving those states are rejected rather than rounded.

## Formal API integer-query contract

The web API client treats integer-valued query parameters as exact values rather than "anything finite that can be truncated later."

Clauses:

1. `/bot/status` `tail`, `/ops` numeric filters (`limit`, `since`, `fromMs`, `toMs`), and `/ops/performance` limits (`commitLimit`, `comboLimit`) are admissible only when the caller supplies a JavaScript safe integer.
2. Fractional, non-finite, and unsafe integer-like values are omitted before query construction, so the client never truncates or rounds them into a different request.
3. Accepted safe integers serialize exactly as their decimal value; `/bot/status` keeps its existing positive-only rule by emitting `tail` only when that exact safe integer is greater than `0`.

Proof sketch:

- `normalizeExactIntegerQueryParam` is now the single gate for these query-valued integers, and it only returns values that satisfy `Number.isSafeInteger`.
- `botStatus`, `ops`, and `opsPerformance` only append query entries when that gate succeeds, so values like `12.5` and `9007199254740993` cannot cross the boundary as `12` or `9007199254740992`.
- Because accepted values are forwarded unchanged after the safe-integer check, exact integers keep their prior wire format and only the rounded-invalid states disappear.
- `haskell/web/test/apiFallback.test.mjs` pins both sides of the contract with representative safe integers and with fractional/unsafe cases that previously truncated into different URLs.

## Formal deploy-time timeout contract

The web deploy-config timeout loader now treats `timeoutsMs.*` as exact whole-millisecond inputs instead of "any finite number that can be rounded later."

Clauses:

1. A configured timeout is admissible only when `Number(...)` yields a JavaScript safe integer millisecond count.
2. Fractional, non-finite, and unsafe integer-like timeout values are rejected before range clamping, so normalization never changes an invalid input into a different valid timeout.
3. Safe integer timeout values still apply the existing bounds: values below `1000` ms are ignored, and values above one day clamp to `86,400,000` ms.

Proof sketch:

- `readNumber` remains the only numeric parser, so the candidate timeout is still determined by a single `Number(...)` conversion for string inputs and by the raw numeric value for numeric inputs.
- `normalizeTimeoutMs` now checks `Number.isSafeInteger(n0)` before any bound logic. Therefore every accepted timeout round-trips exactly through JavaScript integer arithmetic; fractional values like `1000.4` and unsafe magnitudes like `9007199254740993` stop at the validation boundary instead of being rounded or saturated.
- The existing lower/upper bound checks run only after that exactness gate, so previously valid safe integers preserve the same bounded postcondition.

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
9. `botPollSeconds` is an integer-only live-bot poll restore field: finite whole-number values, including legacy numeric strings, normalize through the same `0..3600` bounds as the UI/start path, `0` preserves auto mode, and fractional or non-finite persisted values fall back to the safe default instead of reopening a fractional cadence that later gets truncated.
10. `minHoldBars`, `maxHoldBars`, `cooldownBars`, `maxOrderErrors`, `botOnlineEpochs`, `botTrainBars`, and `botMaxPoints` are integer-only restore fields: finite whole-number values, including legacy numeric strings, survive with the same clamp floors/ceilings the web request builders use, while fractional or non-finite persisted values fall back to their safe defaults.
11. `bars`, `epochs`, `hiddenSize`, and `patience` are integer-backed request restore fields: `bars` restores as a non-negative whole number so the later platform/API cap remains the only dynamic bound, while `epochs`, `hiddenSize`, and `patience` survive within the same request-builder bounds (`0..5000`, `1..512`, and `0..1000` respectively); fractional, non-finite, and unsafe integer-like persisted values fall back to safe defaults instead of reopening the UI with values that later `Math.trunc` or clamp logic would silently change.
12. `trendLookback`, `volLookback`, `rebalanceBars`, and `routerLookback` are integer-only Strategy bar-count restore fields: finite whole-number values survive within request-compatible bounds (`0..1_000_000` for `trendLookback`, `2..1_000_000` for `routerLookback`, and non-negative restore bounds for `volLookback` / `rebalanceBars`), while fractional or non-finite persisted values fall back to their safe defaults instead of reopening the UI with decimals that later request builders truncate.
13. `minRoundTrips`, `walkForwardFolds`, and `walkForwardEmbargoBars` are integer-only tuning restore fields: finite whole-number values survive within the same request-builder floors/ceilings (`0..1_000_000`, `1..1000`, and non-negative respectively), while fractional or non-finite persisted values fall back to their safe defaults.
14. The saved-form whole-number restore invariant is request-aligned: every field normalized with `normalizeWholeNumber` restores as either the same clamped whole number the UI would emit in a request or the safe default, never as a finite fraction that later serializes differently.

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
- `normalizeRestoredNumericFields` still guarantees a finite baseline for every numeric key, but the returned object overwrites `bars`, `epochs`, `hiddenSize`, `patience`, `botPollSeconds`, the integer-only execution counters, the Strategy bar-count controls, and the tuning counters with `normalizeWholeNumber(...)`, so reload-time state cannot retain fractional request fields, poll cadences, counters, or lookbacks from legacy storage.
- `normalizeWholeNumber` only accepts `parseFiniteInteger` results, so fractional strings/numbers and non-finite inputs fall back to the default for `bars`, `epochs`, `hiddenSize`, `patience`, `botPollSeconds`, `minHoldBars`, `maxHoldBars`, `cooldownBars`, `maxOrderErrors`, `trendLookback`, `volLookback`, `rebalanceBars`, `routerLookback`, `minRoundTrips`, `walkForwardFolds`, `walkForwardEmbargoBars`, `botOnlineEpochs`, `botTrainBars`, and `botMaxPoints` instead of surviving until later `Math.trunc` calls.
- Those integer-only overrides mirror the existing request-building semantics: `bars` stays a non-negative whole number until platform/API limits apply later, `epochs` clamps to `0..5000`, `hiddenSize` clamps to `1..512`, `patience` clamps to `0..1000`, `botPollSeconds` clamps to `0..3600` while preserving `0` as auto mode, `minHoldBars`, `maxHoldBars`, `cooldownBars`, and `maxOrderErrors` clamp to the backtest/trade request bounds, `trendLookback` clamps to `0..1_000_000`, `volLookback` and `rebalanceBars` stay non-negative whole numbers, `routerLookback` clamps to `2..1_000_000`, `minRoundTrips` clamps to `0..1_000_000`, `walkForwardFolds` clamps to `1..1000`, `walkForwardEmbargoBars` stays a non-negative whole number, `botOnlineEpochs` clamps to `0..50`, `botTrainBars` keeps the `>= 10` floor, and `botMaxPoints` clamps to `100..100000`.
- The returned object overwrites any persisted value with `botAdoptExistingPosition: true`, so reloads keep orphaned-position adoption enabled by construction.
- `haskell/web/test/utils.test.mjs` continues to mirror the boolean/live-trading restore invariants with representative restored states plus an exhaustive 4 x 3 x 2 x 2 platform/market/live/testnet matrix and mixed-case/whitespace regression rows for `binanceLive`, `binanceTestnet`, `tradeArmed`, and representative boolean toggles.
- `haskell/web/test/formSymbolBehavior.test.mjs` and `haskell/web/test/utils.test.mjs` now pin representative legacy saved-state regressions for the request-aligned whole-number restore contract: accepted integer strings clamp exactly like emitted requests for request/bar-count/live-bot fields, while representative fractional strings/numbers and unsafe integer-like values fall back to defaults; the former file also keeps the existing cross-platform symbol restore regressions.

## Formal combo-apply trade-toggle contract

`applyComboToForm` in `haskell/web/src/app/appHelpers.ts` is treated as a platform-bounded projection for the manual-trade toggles that survive optimizer-combo application.

Clauses:

1. Combo application first resolves the target platform from canonicalized `params.platform`, falling back to canonicalized non-`csv` `source` metadata when `params.platform` is absent.
2. Combo application never invents a live-order or trade-arming toggle: on supported target platforms (`binance`, `coinbase`) it preserves `binanceLive` and `tradeArmed` exactly from the prior form state.
3. Unsupported target platforms (`kraken`, `poloniex`) clear both toggles to `false`.
4. `binanceTestnet` remains Binance-only metadata: non-Binance combo application clears it even when live-order toggles are preserved for Coinbase.
5. The toggle result depends only on the resolved target platform and prior toggles, not on unrelated combo fields such as symbol, thresholds, or sizing.

Proof sketch:

- `applyComboToForm` determines `nextPlatform` once from the canonicalized combo payload (`params.platform` first, then non-`csv` `source`) before building the returned form.
- The `liveOrdersSupported` guard is exactly `nextPlatform === "binance" || nextPlatform === "coinbase"`, so `binanceLive` and `tradeArmed` are copied from `prev` if and only if the resolved target platform supports manual live trading; every other target gets `false`.
- `binanceTestnet` still uses the stricter `nextPlatform === "binance"` guard, so Coinbase keeps live/manual-trade readiness without inheriting Binance-only testnet state.
- `App.tsx` consumes these same toggles to gate `/trade` readiness for Coinbase and Binance, so resolving aliases/source fallbacks before toggle projection removes the prior contradiction where applying an imported Coinbase-prefixed combo could silently downgrade the form into a Binance-like or read-only state.
- `haskell/web/test/formSymbolBehavior.test.mjs` now checks the full supported/unsupported platform matrix plus alias/source-fallback regressions, proving the projection preserves toggles exactly on Binance/Coinbase and clears them on Kraken/Poloniex.

## Formal combo-market classification contract

`comboMarketValue` in `haskell/web/src/app/comboMarket.ts` is treated as the single classifier for optimizer combo filtering and display labels.

Clauses:

1. Classification follows a fixed precedence order: `params.platform` -> non-`csv` `source` -> `csv` -> `unknown`.
2. Before that precedence is applied, `params.platform` and `source` are canonicalized with the same supported exchange alias rules used by web restore/combo apply, so values such as `coinbase-advanced`, `poloniex-v2`, and `binanceusdm` project into the finite UI platform domain.
3. An explicit `params.platform` always wins, even when `source` is present and disagrees.
4. `csv` is only returned when no explicit/canonicalized platform is present.
5. Missing platform/source metadata classify as `unknown`.
6. `comboMarketLabel` is exact for this classifier codomain: platform values use `PLATFORM_LABELS`, `csv` maps to `CSV`, and `unknown` maps to `Unknown`.

Proof sketch:

- `comboMarketValue` first canonicalizes the raw platform/source strings into the finite `{binance, coinbase, kraken, poloniex, csv}` helper codomain, so alias-bearing payloads cannot leak raw strings like `coinbase-advanced` into filters or labels.
- The explicit-platform check still runs before the source fallback, so `params.platform` keeps precedence after canonicalization.
- The `csv` branch sits after those platform projections, so `csv` can only win when no explicit/canonicalized exchange platform is present.
- `haskell/web/src/App.tsx` filters combos via `comboMarketValue`, and `haskell/web/src/components/TopCombosChart.tsx` now derives its title label via `comboMarketLabel(comboMarketValue(combo))`, so filter semantics and displayed market/source copy share the same classifier.
- `haskell/web/test/formSymbolBehavior.test.mjs` bundles `comboMarket.ts` in-memory and asserts representative regression rows for explicit-platform override, exchange-alias canonicalization, non-CSV source fallback, CSV-only fallback, and unknown fallback.

## Formal API base normalization contract

The web UI treats configured API bases and inferred direct-host fallbacks as a small normalization contract, so the same input always maps to the same fetch target.

Clauses:

1. Relative proxy inputs remain relative: `/api` stays `/api`, bare path inputs like `api` normalize to `/api`, and colon-bearing non-authorities like `api:v1` normalize to `/api:v1` instead of malformed cross-origin URLs.
2. Explicit absolute URLs are preserved verbatim.
3. Loopback hosts (`localhost`, `127.0.0.1`, `0.0.0.0`, `::1`, `[::1]`, with optional ports/paths) always normalize to `http://...`, with deterministic IPv6 bracketization when needed.
4. Non-local host-like inputs default to `https://...`, except explicit non-`443` ports default to `http://...`; if the synthesized direct-host target is not itself a parseable URL, normalization falls back to the same-origin `/${source}` path instead of emitting an invalid absolute URL.
5. Split Fly direct-host inference only rewrites `.fly.dev` hostnames when the first label matches a valid `*-web-hs` split-app name; unrelated hostnames remain untouched.
6. Local-host detection for the UI start-help path accepts exactly the supported loopback hostname set.

The verifier in `haskell/web/test/utils.test.mjs` checks a representative invariant matrix over:

- relative proxy paths and explicit absolute URLs
- loopback IPv4/IPv6 authorities
- non-local host-like inputs with and without explicit ports
- valid and invalid Fly direct-host inference inputs
- accepted and rejected local hostname samples

For every case, it checks:

1. Relative proxy inputs do not become cross-origin absolute URLs, including colon-bearing non-authorities.
2. Loopback normalization stays on HTTP and produces deterministic IPv6 authority formatting.
3. Non-local host normalization selects the documented default scheme.
4. Direct Fly inference fires only for supported split-app hostnames.
5. Local hostname detection accepts only the supported loopback set.

## Formal autoloop safety contract

The GitHub autoloop runner treats model output as untrusted input and checks a small executable contract before any commit is created.

Clauses:

1. Every proposed path must be relative, non-empty, traversal-free, and canonical under POSIX slash/dot-segment normalization.
2. A patch plan may not repeat the same canonical path twice.
3. Every bounded cycle must name one backend Haskell algorithm review file under `haskell/app/` and one formal-methods review file under `FORMAL_METHODS.md`, `haskell/app/Trader/Formal/`, `test/`, or `haskell/test/`; both files must already be part of the inspected file set.
4. A patch plan may only modify files that were explicitly requested for inspection by the idea-selection phase.
5. The patch plan must report both review outcomes: `algorithmReviewSummary` for the backend algorithm review pass and `formalMethodsSummary` for the invariant/property/test or proof-sketch pass.
6. Verification commands must come from a fixed allowlist.
7. Any bounded iteration that reaches the push/CI path must preserve the ordered lifecycle `choose-change` -> `algorithm-review` -> `formal-methods-review` -> `plan-patch` -> `apply-patch` -> `verify` -> `commit-push` -> `ci-wait`; the externally required subset is `choose-change` -> `algorithm-review` -> `formal-methods-review` -> `verify` -> `commit-push` -> `ci-wait`.
8. Git staging is restricted to the planned files that actually changed, so generated byproducts outside the plan are excluded from bot commits.
9. The repo-local forever runner may write only under the gitignored `.tmp/autoloop/` state directory, must honor a stop file or `SIGINT`/`SIGTERM`, and must wait between cycles instead of spinning.

The verifier in `test/autoloop.test.mjs` checks the JSON/path normalization, review-target, and phase-sequencing clauses; `scripts/autoloop.mjs` enforces the subset/staging/phase clauses at runtime, and `scripts/autoloop-forever.mjs` enforces the local stop/sleep/state-directory clauses for the persistent supervisor.

Proof sketch:

- Inside `main`, each lifecycle phase is emitted by a single `updateStatus({ phase: ... })` call in straight-line control flow within the bounded iteration loop.
- `sanitizeRelativePath` now normalizes every candidate with `path.posix.normalize` after rejecting absolute/NUL/traversal forms, so aliases such as `haskell/app/Trader/Trading.hs`, `./haskell/app/Trader/Trading.hs`, and `haskell/app/Trader/./Trading.hs` collapse to the same canonical relative path before any allowlist or duplicate-path comparison runs.
- `choose-change` is followed immediately by `algorithm-review` and `formal-methods-review` before any patch planning begins, so the review phases cannot be reordered ahead of selection or behind verification on a patch-producing iteration.
- `plan-patch` and `apply-patch` sit between `formal-methods-review` and `verify`, and the only branches after `verify` either complete early or flow directly into `commit-push` and then `ci-wait`, so verification cannot occur after push and CI wait cannot precede commit.
- `test/autoloop.test.mjs` statically extracts the ordered `phase:` literals from `scripts/autoloop.mjs`, asserts both the required lifecycle subset and the `formal-methods-review` -> `plan-patch` -> `apply-patch` -> `verify` bridge ordering, and now pins the canonical-path invariant by proving that dot-segment aliases collapse before duplicate-path checks.

## What is not proved

This does not prove that:

- the predictors forecast correctly
- the trading strategy is profitable on real markets
- exchange integrations, network I/O, or persistence layers are free of bugs
- floating-point arithmetic is globally free of all numerical issues

It proves that the repo's stated optimizer and selected web-UI contracts match the implementations over the modeled state spaces.

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

The regression checks in `haskell/test/TestMain.hs` now cover representative window selection, equal-max earliest-timestamp order invariance, non-finite PnL candidate rejection (including the `Nothing` result when no finite candidate survives), and the boundary risk-budget decisions (`beta = 0` and `beta = 1`). This proof sketch now also makes the invalid-sample dropping (`tc <= ta`, `tm < ta`, and `tm > ta + 2*(tc-ta)`) and non-finite-budget contract (`NaN`, `+Infinity`, and `-Infinity` normalize to the same median-target policy as `beta = 0`) explicit, matching `chooseBetterClose`, `isCloseTimingCandidate`, `observationRatioParts`, `validObservation`, `decisionTargetBand`, `normalizeRiskBudget`, and `clampRatio` in `haskell/app/Trader/Formal/CloseTiming.hs`. The local `isFinite` guard in `optimalCloseObservation` excludes non-finite PnL candidates before `tm` selection, while `normalizeRiskBudget` plus `clampRatio` keep the downstream `ctdTargetRatio` finite and bounded inside `[0,2]`, so this regression/doc update does not change runtime behavior.

### Implementation pointers

- Model + policy: `haskell/app/Trader/Formal/CloseTiming.hs`
- Unit tests: `haskell/test/TestMain.hs`

## Formal restored-form normalization contract

`normalizeFormState` in `haskell/web/src/app/formState.ts` is treated as a total projection from persisted browser state into the finite `FormState` domain consumed by the UI, request builder, and auto-refresh scheduler.

Clauses:

1. Restored enum fields are closed over their declared domains: `method` must be one of the supported method IDs, and `normalization` must be one of `none|minmax|standard|log`.
2. Restored `method` and `normalization` inputs are trimmed before membership checks; unsupported values fall back to the documented defaults instead of leaking arbitrary strings into the typed UI state.
3. Restored `platform`, `market`, `interval`, `positioning`, `intrabarFill`, `tuneObjective`, and `normalization` values are canonicalized with the same whitespace/casing/alias rules used by the backend parsers; supported exchange aliases such as `coinbase-advanced`, `poloniex-v2`, and `binanceusdm` therefore restore as `coinbase`, `poloniex`, and `binance`, unsupported values still fall back to the documented safe defaults, and Binance `1M` month intervals remain distinct from `1m` minute intervals.
4. Restored numeric strings are conservative after trimming: blank or whitespace-only strings are treated as absent and therefore fall back to the documented defaults instead of silently normalizing to `0` or a clamped boundary value.
5. Restored values for `fee`, `stopLoss`, `takeProfit`, `trailingStop`, `maxDrawdown`, `maxDailyLoss`, `backtestRatio`, and `autoRefreshSec` must already lie inside the same bounded domains later assumed by downstream consumers.
6. For those bounded fields, restore is a fixed point: once a value has been normalized, applying `normalizeFormState` again does not change it.

The verifier in `haskell/web/test/utils.test.mjs` checks this contract with:

- exhaustive membership preservation over the full supported method set and normalization set, with trimmed-string acceptance
- invalid enum regressions proving fallback to the default method/normalization
- blank-string numeric regressions proving trimmed absence falls back to the default numeric state instead of reopening at `0`/boundary values
- representative restore regressions for backend-compatible platform/market/interval/positioning/intrabar-fill/tune-objective aliases and casing, including `1M` month preservation
- representative bounded restore matrices for the downstream-clamped numeric fields, including numeric-string hydration and fixed-point re-normalization

Proof sketch:

- `normalizeFormState` now routes `method` and `normalization` through explicit finite-set membership checks, so the returned `FormState` cannot contain out-of-domain enum strings even when persisted storage is stale or manually edited.
- The restore helpers for `platform`, `market`, `interval`, `positioning`, `intrabarFill`, and `tuneObjective` now mirror the backend parsers' finite canonicalization rules closely enough to preserve supported stale/local-storage spellings without widening the accepted UI state space; the platform helper explicitly projects supported exchange aliases such as `coinbase-advanced`, `poloniex-v2`, and `binanceusdm` into the finite UI platform domain, while the dedicated interval normalizer preserves uppercase `M` so Binance month intervals cannot collapse into minute intervals.
- The numeric restore helpers now trim string inputs before `Number` conversion and reject the empty post-trim case, so malformed persisted blanks stay absent instead of being reinterpreted by JavaScript as numeric zero.
- The restore path now uses the same clamp intervals already enforced later by `haskell/web/src/App.tsx` when building API requests (`fee`, stop/drawdown ratios, `backtestRatio`) and scheduling auto-refresh (`autoRefreshSec`), so restored UI state is aligned with downstream behavior instead of reopening with values that would later serialize or execute differently.
- Because each bounded field is normalized by a pure clamp onto a closed interval, the normalized value is already in the image of the clamp; reapplying the same normalization leaves it unchanged, which yields the restore fixed-point property checked by the test suite.

## Formal local-datetime filter contract

`formatDatetimeLocal` and `parseDatetimeLocal` in `haskell/web/src/app/appHelpers.ts` are treated as the total formatter/parser pair for the Live bot timeline range inputs.

Clauses:

1. `formatDatetimeLocal` is total over numeric input: non-finite values and finite values outside the ECMAScript `Date` domain produce the empty string rather than malformed `NaN-...` fragments.
2. `parseDatetimeLocal` accepts only exact local calendar timestamps in `YYYY-MM-DDTHH:mm[:ss[.sss]]` form (with a single space accepted in place of `T` for compatibility).
3. Parsing is conservative: any input whose parsed local year/month/day/hour/minute/second/millisecond fields differ from the source fields is rejected.
4. Therefore impossible local timestamps such as `2024-02-31T12:34` are rejected instead of being normalized into a neighboring date by `Date.parse`.
5. For accepted minute-aligned inputs in the modeled state space, `formatDatetimeLocal(parseDatetimeLocal(x)) = canonical(x)`.

The verifier in `haskell/web/test/appHelpers.test.mjs` checks this contract with:

- a bounded month-end matrix over `year ∈ {2023, 2024}`, `month ∈ {1..12}`, and `day ∈ {28, 29, 30, 31}` at `12:34`
- targeted regressions for impossible local timestamps that `Date.parse` would otherwise normalize
- out-of-range finite formatter inputs (`±1e20`)

Proof sketch:

- `parseDatetimeLocal` first matches the input against a finite local-datetime grammar, so date-only strings, timezone-bearing strings, and malformed separators are excluded before parsing.
- After `Date.parse`, it re-reads the local calendar fields from the resulting `Date` and compares them against the source tuple; any normalization by the JS date parser therefore becomes an observable mismatch and is rejected.
- `formatDatetimeLocal` now checks `d.getTime()` after constructing the `Date`, so finite numbers outside the representable `Date` range collapse to the same empty-string behavior already used for non-finite inputs.
- The bounded month-end matrix proves the intended accept/reject split around leap-year and 30-day/31-day boundaries, while the round-trip assertions establish the parser/formatter fixed-point property on the accepted states.

## Formal ISO UTC export formatter contract

`formatIsoUtc` in `haskell/web/src/app/utils.ts` is treated as the total UTC timestamp formatter for copy/export surfaces that include raw millisecond fields.

Clauses:

1. `formatIsoUtc` is total over `number | null | undefined`: nullish, non-finite, and out-of-range finite inputs produce `""`.
2. In-range finite timestamps produce exactly `new Date(ms).toISOString()`.
3. Therefore CSV/clipboard export paths that depend on `formatIsoUtc` cannot throw `RangeError: Invalid time value` solely because a payload contains an out-of-range finite timestamp.

The verifier in `haskell/web/test/utils.test.mjs` checks representative nullish, non-finite, out-of-range, and epoch-zero cases.

Proof sketch:

- `formatIsoUtc` first rejects non-number and non-finite inputs, so the only remaining candidates are finite numbers.
- It then constructs `Date(ms)` and checks `d.getTime()` before calling `toISOString()`. ECMAScript invalid-date objects report `NaN` from `getTime()`, so out-of-range finite inputs stop at the empty-string branch instead of reaching the partial `toISOString()` call.
- For every surviving in-range value, the function returns `d.toISOString()` directly, so the formatter agrees exactly with the built-in UTC ISO rendering on its defined domain.

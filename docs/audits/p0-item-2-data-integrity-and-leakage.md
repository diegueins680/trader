# P0 audit item 2 — Data integrity and leakage audit

Date: 2026-04-19

Update note, 2026-06-21:
- The Binance-specific findings in the original audit have been superseded in part: the current kline path retains close time, discards still-open candles, rejects non-finite numeric payloads, rejects negative volume and invalid OHLC shape, validates strictly increasing open times, and the live bot path holds on stale or gapped market data before processing a new decision.
- The production stale-data policy is now stricter than the original audit recommendation: `Trader.MarketDataIntegrity.marketDataFreshness` marks data stale when `ageMs > intervalMs`, measured from the last processed candle close time. The older `2 x interval` threshold below is retained only as historical audit context.

Update note, 2026-06-22:
- Live exchange price-loading paths now route through one enforced post-load/pre-strategy market-series QA gate. Binance, Coinbase, Kraken, and Poloniex inputs reject non-finite OHLCV, invalid OHLC relationships, negative volume, duplicate/non-increasing timestamps, missing-bar continuity, still-open candles, and stale last-closed bars before strategy execution.
- CSV/offline input is structurally validated but intentionally does not enforce interval continuity or wall-clock freshness by default, because CSV can be historical or time-agnostic input rather than a live venue feed.

Scope reviewed:
- `haskell/app/Trader/App/Csv.hs`
- `haskell/app/Trader/Binance.hs`
- `haskell/app/Trader/Coinbase.hs`
- `haskell/app/Trader/Kraken.hs`
- `haskell/app/Trader/Poloniex.hs`
- `haskell/app/Trader/Predictors/Features.hs`

Adjacent entry-point evidence reviewed for trading impact:
- `haskell/app/Main.hs`
- `haskell/app/Trader/App/Args.hs`
- `haskell/app/Trader/Symbol.hs`

## Executive summary

**Current verdict: PASS for live exchange price-loading and trade-boundary safety; CSV/offline caveats remain.**

What is already good:
- CSV and exchange decoders reject non-finite numeric payloads in the fields they parse.
- CLI/args symbol normalization is present for Binance, Coinbase, and Poloniex.
- `Main.validateLoadedPriceSeries` applies a shared post-load QA gate before exchange-backed strategy execution.
- Binance, Coinbase, Kraken, and Poloniex live loaders discard still-open candles, require strict interval continuity, and reject stale market data.
- Coinbase, Kraken, and Poloniex now preserve source open and volume fields instead of forcing live exchange paths through synthetic OHLCV fallbacks.
- `Predictors/Features.hs` itself is **structurally no-lookahead** for supervised labels and feature indexing, assuming the input bars are already closed and correctly ordered.

Remaining caveats:
- CSV can still load without timestamps and does not enforce interval continuity or wall-clock freshness by default; that is an offline/backtest compatibility choice, not a live exchange freshness guarantee.
- `Predictors/Features.hs` still supports synthetic fallback when callers omit OHLCV fields. Live exchange loaders now provide source OHLCV, so this is an offline/legacy caller caveat rather than the live exchange path.
- Cache TTLs remain transport caching only; freshness is enforced from last closed bar time.

## Bottom line

For live exchange price loads, trading is now blocked whenever a series fails timestamp normalization, strict monotonicity, duplicate rejection, missing-bar continuity, closed-bar completeness, stale-data freshness, non-finite OHLCV checks, negative-volume checks, or basic OHLC invariants. CSV remains a softer offline input surface.

---

## Current-state findings by topic

### 1) CSV parser integrity (`Trader.App.Csv`)

**What passes now**
- Header resolution is robust and user-friendly, with exact/case-normalized suggestions (`Csv.hs:70-125`).
- Numeric parsing rejects `NaN`/`Infinity` (`Csv.hs:127-154`).
- Timestamps are parsed via `parseTimestampMs`, which supports integer epochs and ISO timestamps, with second→millisecond normalization in `Args.hs` (`Csv.hs:211-230`, `Args.hs:362-441`).
- The post-load `PriceSeries` validator rejects length mismatches, duplicate/non-increasing timestamps when time is present, invalid OHLC relationships when OHLC columns are present, and negative volume when a volume column is present.

**What remains intentionally soft**
- Missing-bar continuity is not enforced for CSV by default.
- If no time column is present, the CSV still loads; that is fine for some offline use, but it means there is no way to verify ordering, gaps, or staleness (`Csv.hs:48-68`, `Csv.hs:156-168`).
- Wall-clock staleness is not enforced for CSV.

**Risk**
- A CSV with no timestamp, or with timestamp gaps, can still be accepted for offline workflows.
- Time gaps can distort return, ATR, breakout, and volume features unless the caller uses an exchange loader or adds explicit CSV continuity policy.

### 2) Binance parser integrity (`Trader.Binance`)

**What passes now**
- Parsed kline fields reject non-finite numeric strings (`Binance.hs:549-586`).
- The public trading entry path normalizes Binance symbols before use (`Args.hs:1039-1089`, `Symbol.hs:136-148`).
- Kline parsing retains the close-time field when present.
- `fetchKlinesRaw` and `fetchKlinesBetweenRaw` pass responses through `normalizeClosedKlines`, which sorts by open time, rejects non-finite OHLCV payloads, rejects negative volume and invalid OHLC relationships, rejects duplicate/non-increasing open times, and filters out candles whose close time is not yet in the past.
- The live market-data helpers mark stale data when the last processed candle close is more than one interval old and detect non-contiguous follow-on candles as `MARKET_DATA_GAP`.
- Closed Binance series now require exact interval continuity before strategy execution.
- Post-load validation applies the same freshness and market-series checks to the `PriceSeries` passed into strategy code.

**Residual caveats**
- Duplicate open times are rejected rather than deduplicated.
- Cache TTL (`5s` fresh, `60s` stale) is transport caching only, not feed freshness validation (`Binance.hs:326-330`).

**Risk**
- Residual risk is mostly operational: upstream gaps now block exchange-backed strategy execution instead of being smoothed over.

### 3) Coinbase parser integrity (`Trader.Coinbase`)

**What passes now**
- Parsed candle fields reject non-finite numbers (`Coinbase.hs:356-378`, `Coinbase.hs:406-430`).
- Timestamp values are normalized to seconds if the upstream payload is in milliseconds (`Coinbase.hs:420-424`).
- Source open/high/low/close/volume values are parsed and preserved.
- Closed-candle normalization sorts by timestamp, rejects duplicates/non-increasing times, rejects invalid OHLC and negative volume, discards still-open buckets, and requires exact interval continuity.
- Post-load validation enforces last-closed-bar freshness before strategy execution.

**Residual caveats**
- Module-local symbol normalization is only `trim + uppercase`; delimiter sanitation is effectively delegated to args validation (`Coinbase.hs:263-299`, `Args.hs:1039-1089`).
- Cache TTL (`30s` fresh, `300s` stale) is not feed freshness validation (`Coinbase.hs:147-151`).

**Risk**
- Symbol canonicalization still relies on the args layer for full delimiter sanitation.

### 4) Kraken parser integrity (`Trader.Kraken`)

**What passes now**
- Parsed candle fields reject non-finite numbers (`Kraken.hs:117-176`).
- The response parser rejects explicit Kraken API errors (`Kraken.hs:88-96`).
- Source open/high/low/close/volume values are parsed and preserved.
- Closed-candle normalization sorts by timestamp, rejects duplicates/non-increasing times, rejects invalid OHLC and negative volume, discards still-open buckets, and requires exact interval continuity.
- Post-load validation enforces last-closed-bar freshness before strategy execution.

**Residual caveats**
- No module-local symbol canonicalization beyond caller-provided input (`Kraken.hs:59-76`).
- Cache TTL (`30s` fresh, `300s` stale) is not feed freshness validation (`Kraken.hs:50-54`).

**Risk**
- Symbol canonicalization is still mostly caller-owned, but candle integrity is enforced before strategy execution.

### 5) Poloniex parser integrity (`Trader.Poloniex`)

**What passes now**
- Parsed candle fields reject non-finite numbers (`Poloniex.hs:139-216`).
- Timestamps are normalized from ms→s when needed (`Poloniex.hs:206-210`).
- Source open/high/low/close/volume values are parsed and preserved.
- Closed-candle normalization sorts by timestamp, rejects duplicates/non-increasing times, rejects invalid OHLC and negative volume, discards still-open buckets, and requires exact interval continuity.
- Post-load validation enforces last-closed-bar freshness before strategy execution.
- There is at least some symbol normalization/candidate logic (`BASE_QUOTE` and reversed candidate retry) (`Poloniex.hs:65-91`, `Poloniex.hs:218-227`).

**Residual caveats**
- Cache TTL (`30s` fresh, `300s` stale) is not feed freshness validation (`Poloniex.hs:56-60`).

**Risk**
- Residual risk is mostly platform-symbol drift or upstream outages that now fail closed at the series gate.

### 6) Symbol normalization across paths

**What passes now**
- CLI argument normalization and validation for Binance/Coinbase/Poloniex is solid (`Args.hs:1039-1089`).
- `Trader.Symbol` can canonicalize platform-specific delimiters and salvage common Binance variants (`Symbol.hs:113-228`).

**Residual caveats**
- Module-local fetch functions are inconsistent:
  - Binance: uppercases symbol, but relies on caller for canonical structure.
  - Coinbase: uppercases only; does not locally coerce `/`→`-`.
  - Kraken: takes raw pair string.
  - Poloniex: stronger candidate handling than others.
- This is acceptable if every call site is guaranteed to pass through args normalization, but that guarantee is weaker than a module-level invariant.

**Risk**
- Internal or future call sites can bypass the current arg-layer guardrails and hit wrong endpoints or silently fail.

### 7) No-lookahead feature construction (`Trader.Predictors.Features`)

**What passes now**
- `featuresAtWithInputsWithMarket` uses data at or before `t` only (`Features.hs:107-157`).
- The label is `forwardReturnAt closes t = p[t+1] / p[t] - 1`, which is correct for supervised next-bar prediction (`Features.hs:78-87`).
- Dataset generation uses `t in [lookback-1 .. n-2]`, so targets are aligned and do not read beyond `t+1` (`Features.hs:174-188`).
- Predictor training preserves time order: calibration is taken from the tail, not shuffled into training (`Predictors.hs:91-115`).

**Residual caveats**
- `barAt` silently fabricates open/high/low/volume when those vectors are missing or invalid (`Features.hs:228-283`).
  - missing open → previous close
  - missing high/low → coerced around open/close
  - missing/invalid volume → `1`
- `klineFeatures` then sanitizes non-finite derived values to `0` (`Features.hs:327-431`, `Features.hs:470-474`).
- This is dimensionally convenient, but it can hide real data defects instead of surfacing them.

**Risk**
- **No direct lookahead bug found in feature indexing.**
- However, there is an **operational leakage/integrity risk** if the latest input bar is still open/incomplete. In that case, the feature code remains mathematically causal but is acting on a bar that production may later revise.

---

## Explicit data-QA checklist with pass/fail conditions

The following checklist should be applied to every series used for **live signal generation or trade placement**. For offline analysis/backtests, items marked “soft-fail allowed offline” may degrade to warnings if explicitly intended.

| Check | Pass condition | Fail condition | Recommended action |
|---|---|---|---|
| Symbol canonicalization | Symbol is canonical for platform before HTTP request (`BTCUSDT`, `BTC-USD`, `BTC_USDT`) | Raw delimiter mismatch, empty symbol, or unsanitized internal variant | **Hard-fail** live + offline |
| Timestamp parse/unit normalization | Every bar has a parseable timestamp, normalized to a single unit (ms recommended at loader boundary) | Any missing/unparseable timestamp where time-aware behavior is needed | **Hard-fail** live; soft-fail allowed only for explicitly time-agnostic offline CSV use |
| Strict monotonicity | `t[i] < t[i+1]` for all kept bars | Duplicate or non-increasing timestamp remains after normalization | **Hard-fail** live; dedupe-then-warn acceptable offline only if explicitly chosen |
| Missing-bar continuity | `t[i+1] - t[i] == intervalMs` for all consecutive closed bars after normalization | Any gap or compression of bars | **Hard-fail** live; soft-fail offline only if gap-aware resampling is explicitly enabled |
| Closed-bar completeness | Latest usable bar is confirmed closed; current/open bucket is discarded | Loader uses a possibly in-progress current candle | **Hard-fail** live |
| Stale-data freshness | Last **closed** bar is newer than the stale threshold | Last closed bar exceeds threshold | **Hard-fail** live |
| Finite OHLCV | All required numeric inputs are finite | Any `NaN`/`Infinity` in required fields | **Hard-fail** live + offline |
| OHLC invariants | `high >= max(open, close, low)` and `low <= min(open, close, high)` | Any impossible OHLC relationship | **Hard-fail** live + offline |
| Volume invariants | Volume is finite and `>= 0`; if required by enabled features, it must be present | Negative or non-finite volume; silently fabricated volume in trading path | **Hard-fail** live when volume-sensitive features are enabled; soft-fail offline otherwise |
| Feature causality | Features at `t` use only data `<= t`; target is at `t+1` | Any feature reads beyond `t` | **Hard-fail** live + offline |
| Feature missingness handling | Missing OHLCV is surfaced explicitly or blocked in live trading | Synthetic OHLCV fallback hides malformed upstream data in trading path | **Hard-fail** live; soft-fail offline only if documented |
| Trade-entry QA gate | Trade path refuses order placement on any failed data-QA check | Trade path only checks row count/lookback | **Hard-fail** live |

---

## Current pass/fail assessment against the checklist

### CSV (`Csv.hs`)
- Symbol canonicalization: **N/A**
- Timestamp parse/unit normalization: **PARTIAL PASS**
- Strict monotonicity: **PASS when timestamps are present at post-load validation**
- Missing-bar continuity: **SOFT / not enforced by default for offline CSV**
- Closed-bar completeness: **N/A / not guaranteed for CSV**
- Stale-data freshness: **SOFT / not enforced by default for offline CSV**
- Finite OHLCV: **PASS** for parsed columns
- OHLC invariants: **PASS when OHLC columns are present**
- Volume invariants: **PASS when volume column is present**

### Binance (`Binance.hs`)
- Symbol canonicalization: **PARTIAL PASS** (args layer yes; module layer partial)
- Timestamp parse/unit normalization: **PASS** at loader boundary via `normalizeEpochMs`
- Strict monotonicity: **PASS**
- Missing-bar continuity: **PASS**
- Closed-bar completeness: **PASS**
- Stale-data freshness: **PASS**
- Finite OHLCV: **PASS**
- OHLC invariants: **PASS**
- Volume invariants: **PASS**

### Coinbase (`Coinbase.hs`)
- Symbol canonicalization: **PARTIAL PASS**
- Timestamp parse/unit normalization: **PASS**
- Strict monotonicity: **PASS**
- Missing-bar continuity: **PASS**
- Closed-bar completeness: **PASS**
- Stale-data freshness: **PASS**
- Finite OHLCV: **PASS** for parsed fields
- OHLC invariants: **PASS**
- Volume invariants: **PASS**

### Kraken (`Kraken.hs`)
- Symbol canonicalization: **PARTIAL**
- Timestamp parse/unit normalization: **PASS** at loader boundary
- Strict monotonicity: **PASS**
- Missing-bar continuity: **PASS**
- Closed-bar completeness: **PASS**
- Stale-data freshness: **PASS**
- Finite OHLCV: **PASS** for parsed fields
- OHLC invariants: **PASS**
- Volume invariants: **PASS**

### Poloniex (`Poloniex.hs`)
- Symbol canonicalization: **PASS/PARTIAL**
- Timestamp parse/unit normalization: **PASS**
- Strict monotonicity: **PASS**
- Missing-bar continuity: **PASS**
- Closed-bar completeness: **PASS**
- Stale-data freshness: **PASS**
- Finite OHLCV: **PASS** for parsed fields
- OHLC invariants: **PASS**
- Volume invariants: **PASS**

### Features (`Predictors/Features.hs`)
- Feature causality / no-lookahead indexing: **PASS**
- Feature missingness handling for live exchange safety: **PASS through venue loaders providing source OHLCV**
- Feature missingness handling for CSV/offline callers: **SOFT / caller-policy dependent**

---

## Hard-fail vs soft-fail recommendations

### Hard-fail (must block live trading)
1. Any non-finite OHLCV value in a required field.
2. Any impossible OHLC relationship.
3. Any unparseable or non-normalized timestamp in a live series.
4. Any duplicate or non-increasing timestamp after normalization.
5. Any missing bar / time gap in a live series unless there is an explicit resampling policy.
6. Any possibly open/incomplete latest candle in a live path.
7. Any stale latest closed candle beyond threshold.
8. Any symbol that is not canonical for the target venue.
9. Any trading-path feature request that relies on silently synthesized OHLCV/volume instead of validated source data.

### Soft-fail (warning/log/metrics; trading may continue only in non-live or explicitly degraded modes)
1. CSV with no timestamp column for strictly offline experiments.
2. Missing volume when volume-derived features are disabled.
3. Exchange response delivered with fewer bars than requested, **if** the remaining kept bars still pass continuity/freshness rules and the strategy has enough history.
4. Offline backtests where the operator explicitly opts into gap-tolerant resampling and the resampling policy is recorded.

---

## Leakage risks

### A. No direct feature lookahead found in `Features.hs`
The indexing logic is correct:
- features at `t` use bars `<= t`
- target uses `t+1`
- dataset generation stops at `n-2`

This is a **PASS**, not the problem.

### B. Previously high risk: incomplete/open-bar ingestion
If exchange loaders include the currently forming candle, the model is not technically reading the future, but it is reading a bar that is not final and may later be revised. That creates a training/serving mismatch and can make the live system behave as if it had a “closed” bar when it does not.

Current exchange loaders now filter still-open candles before strategy input, and post-load freshness uses the last closed candle close time. The remaining caveat is CSV/offline input, where closed-bar status is not knowable unless the dataset carries enough timestamp policy for the caller to enforce it.

### C. Silent synthetic OHLCV fallback in feature construction
`barAt` and `klineFeatures` make invalid/missing inputs look valid enough for downstream models (`Features.hs:228-283`, `Features.hs:327-431`).

That is acceptable for some offline robustness work, but not as a hidden default in a live trading path. In trading mode, malformed upstream bars should be surfaced and blocked before features are built.

### D. Post-load trade gate
Exchange-backed signal/trade entry paths now validate the loaded `PriceSeries` before strategy execution, so row-count/lookback checks are no longer the only barrier for malformed venue data. CSV/offline input keeps structural validation but does not add live freshness semantics.

---

## Stale-data threshold assumptions used for this audit

The codebase currently has **cache TTLs**, but those TTLs are not the same thing as bar freshness:
- Binance candles cache: `5s` fresh / `60s` stale (`Binance.hs:326-330`)
- Coinbase candles cache: `30s` fresh / `300s` stale (`Coinbase.hs:147-151`)
- Kraken candles cache: `30s` fresh / `300s` stale (`Kraken.hs:50-54`)
- Poloniex candles cache: `30s` fresh / `300s` stale (`Poloniex.hs:56-60`)

**Audit assumption for live trading recommendation:**
- First, discard any bar that is not known to be fully closed.
- Production now defines stale as: **the last closed bar's close time is more than `1 x interval` behind wall clock** (`ageMs > intervalMs`).

Examples under the production policy:
- `1m` strategy: hard-fail if the last closed bar close time is more than 1 minute old.
- `5m` strategy: hard-fail if the last closed bar close time is more than 5 minutes old.
- `1h` strategy: hard-fail if the last closed bar close time is more than 1 hour old.

This is stricter than the original audit's `2 x interval` recommendation and is the current code-backed policy.

---

## Recommended P0 implementation shape (without changing strategy logic)

The implemented shape is a single **validated market-series gate** after each exchange loader and before signal/trade execution:
1. canonicalize symbol
2. normalize timestamps to ms
3. sort and reject duplicates/non-increasing timestamps
4. require strict monotonicity
5. require exact interval continuity for kept bars
6. drop current/open candle; require at least one closed latest candle
7. enforce stale threshold on last closed bar
8. enforce finite OHLCV and OHLC invariants
9. preserve source OHLCV in live exchange paths
10. only then call `computeTradeOnlySignal` / `computeTradeFromSeries`

The bot polling path continues to apply stale/gap checks before processing new market data. CSV input keeps the structural subset of the gate without continuity/freshness enforcement by default.

---

## Acceptance-oriented closeout

### Acceptance status against requested outcome
- Explicit QA checklist: **DONE**
- Pass/fail conditions: **DONE**
- Hard-fail vs soft-fail recommendations: **DONE**
- Leakage risks: **DONE**
- Stale-data threshold assumptions: **DONE**
- “Malformed or stale data blocks trading where appropriate”: **DONE FOR LIVE EXCHANGE PRICE-LOADING PATHS** after the shared post-load/pre-strategy QA gate; **SOFT FOR CSV/OFFLINE INPUT** by design.

### P0 disposition
**Pass for live exchange price-loading and trade-boundary safety. CSV/offline input remains intentionally softer and should not be used as a live exchange freshness guarantee.**

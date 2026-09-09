# Alpha-research data-acquisition spec

**Date:** 2026-06-10
**Why this exists:** every signal investigation so far (Kalman, method-mix, optimizer warm-start, exogenous funding/basis) looked promising in-sample and dissolved under honest out-of-sample testing. The binding constraint turned out not to be the model but the **data**: the Binance `/futures/data` stats endpoints return only ~30 days, so every OOS test ran on ~9–60 day windows where sharpe estimates are statistically meaningless (the harness's bootstrap CIs straddle zero — see `scripts/research/`). You cannot establish or refute an edge on that. This spec lists what to acquire so the harness has **years and thousands of trades**, not days.

**Evaluation bar (enforced by `scripts/research/harness.py`):** a signal is only deployable if, on this data, it shows OOS sharpe whose bootstrap CI **excludes 0**, a deflated `P(SR>0) ≥ 0.95` after the multiple-testing haircut, consistency **across symbols**, and stability **across up/down regimes**. Target ≥ 1500 OOS observations minimum (years of daily, or months of hourly).

---

## Tier 0 — free, start accumulating immediately (no spend)

| Dataset | Source | Fields | Granularity | History available | Notes |
|---|---|---|---|---|---|
| Perp OHLCV | Binance `/fapi/v1/klines` | O/H/L/C/V | 1m–1d | **full (years)** via pagination | already used; deep history is free |
| Funding rate | Binance `/fapi/v1/fundingRate` | fundingRate, fundingTime | 8h | **full (years)** free | best free exogenous series; the funding IC was the most consistent signal |
| Open interest | Binance `/futures/data/openInterestHist` | sumOpenInterest | 5m–1d | **~30 days only** | **must accumulate** (run `datafeed.update_cache` on cron) or buy archival |
| Basis | Binance `/futures/data/basis` | basisRate | 5m–1d | **~30 days only** | same — accumulate or buy |
| Taker buy/sell | Binance `/futures/data/takerlongshortRatio` | buySellRatio | 5m–1d | **~30 days only** | same |
| Long/short ratios | Binance `/futures/data/global...Ratio`, `top...Ratio` | account/position ratios | 5m–1d | ~30 days | accumulate |

**Action:** schedule `python3 scripts/research/datafeed.py <symbols>` (hourly/daily) so the 30-day-limited series build a permanent history in `data/research/`. The collector requests a safely bounded retained OI/basis/taker window in fixed chunks ending at the last closed bar, leaves missing intervals null, records any trailing unavailable buckets, and merges the point-in-time evidence into the cache; it cannot recover observations that expired before collection began. A refresh is healthy during provider publication lag only while its latest finite observation remains inside the fixed family freshness bound. Conservative local request budgets and Binance's observed shared-IP weight header pace calls. HTTP 418/429 or JSON `-1003` aborts the remaining run without further symbol requests, records sanitized throttle evidence, and cannot yield an admissible artifact receipt. The canonical universe remains fixed, but `utc_epoch_hour_rotation_v1` rotates its request leader by UTC epoch hour so recurring partial runs do not always starve the same tail; status records the exact permutation and its verifier binds it to the run start. Rotation does not make a partial run complete or recover missing first-seen evidence. Cost: $0; you just need to start now — every day not collecting is lost.

The missingness-aware candidate has a distinct, later prospective boundary. Its `collect_market_context.py` command captures one explicit completed bar, contemporaneous exchange eligibility, the all-symbol rolling ticker, and exactly two klines for every eligible USDⓈ-M perpetual into a new immutable-by-convention bundle. The collector uses only public GET requests, records exact first-seen clocks and shared-IP weight, publishes `complete_unverified`, and never invokes its independent verifier as admission authority. The offline `verify_market_context_bundle.py` wrapper is the required integrity check for collector output: it rejects every incomplete or indeterminate collector state, verifies the historical Git-bound provenance and exact status/manifest relationship, and then invokes `market_context_source.py` to reconstruct the raw bundle and panel. A persisted receipt must later pass `verify_market_context_receipt.py` against an exact frozen archive using the bundle-verifier version stored at the collection commit. Do not begin this candidate's registered dataset before 2027-01-21. A verified or replay-verified future bundle still requires a separate experiment-manifest update and admission review before model use and cannot open the final holdout or authorize trading.

## Tier 1 — paid archival (highest value; buys the history Tier 0 can't backfill)

| Dataset | Vendors | Why it matters | Rough cost |
|---|---|---|---|
| Historical OI / basis / funding / liquidations (years) | **Tardis.dev**, Amberdata, Kaiko, CoinAPI | backfills the 30-day gap immediately → can test funding/basis edge over multiple regimes *today* instead of waiting months | ~$100–500/mo or per-dataset |
| L2 order book + trades (tick) | **Tardis.dev** (best coverage), Kaiko | order-flow imbalance / microstructure — the highest-frequency edge, and raises trade count for statistical power | $$ (storage-heavy) |
| Liquidation feed | Tardis, Coinglass API | cascade/reversal signal | $ |
| On-chain | **Glassnode**, CryptoQuant, Nansen | exchange net-flows, stablecoin supply, whale/miner moves — daily-cadence, fits daily bars | ~$30–800/mo by tier |

**Recommended first buy:** Tardis.dev historical funding + OI + basis + liquidations for the top ~10 perps, 2+ years. It's the cheapest way to turn the already-promising-but-unprovable funding/basis result into a verdict.

## Tier 2 — macro / options (mostly free; regime context)

| Dataset | Source | Use |
|---|---|---|
| DXY, US 2y/10y, real rates, net Fed liquidity (reserves−RRP−TGA) | **FRED** (free) | crypto is liquidity/risk-on driven; explains regime the per-coin model can't see |
| Equities / VIX | Yahoo, Stooq (free) | risk-on/off |
| Implied vol / DVOL term structure | Deribit (free API), Laevitas | regime detection; scales the Kalman Q/R adaptively |

---

## Storage & point-in-time discipline

- **Legacy schema:** one CSV per `(symbol, interval)` in `data/research/` — `openTime, open, high, low, close, volume, funding, oi, basis, taker, …`. Those columns retain their historical event-time alignment for compatibility. Drop-in archival data must follow the documented schema/alignment and cannot claim v2 availability without release evidence.
- **Derivatives v2 schema:** the public collector stores raw `binance_derivatives_first_seen_v2` ledgers under `data/research/.observations/` because Binance history responses do not provide a separate publication/revision time. Each row records event time, fetch-completion availability time, an observed flag, and a finite dense value; a known missing grid observation is a zero-valued unavailable tombstone. Additive per-family bar columns retain dense value, observed mask, fresh mask, selected event time, and selected availability time in stable funding/OI/basis/taker order across cache merges. Existing legacy rows are not relabeled; they remain ineligible wherever `feature_availability_v2` is required. A complete collector status schema 3 binds the bar file and all four ledgers under `binance_derivatives_collection_artifacts_v3`; `verify-artifacts` checks the hashes and reconstructs versioned rows from those ledgers. The isolated Haskell v2 decoder validates row semantics. A separate opt-in `binance_derivatives_model_features_v2` adapter preserves the legacy funding-level/delta, open-interest-relative-delta, basis-level, and centered-taker formulas while appending explicit masks and retaining causal timestamp witnesses. Artifact verification remains a prerequisite, and neither decoder nor adapter is imported by legacy predictor or trading paths.
- **Market-context source v1:** each new bundle contains exact public response bytes, a `complete_unverified` collection status, and a `binance_usdm_market_context_source_manifest_v1` published only after full eligible-population coverage. The offline bundle verifier binds that status to the manifest, historical Git provenance, source/license declaration, independent raw reconstruction, derived v2 panel, and a separate non-authorizing receipt. The read-only receipt verifier replays that receipt against an exact external archive, with before/after content hashes and the historical bundle-verifier version. No real bundle exists as of 2026-09-08; synthetic fixtures only exercise the contract.
- **External v2 boundary:** `ExternalFeatureInputsV2` retains selected event time, availability time, value, and availability for every aligned family. The opt-in `external_family_model_features_v2` adapter preserves the legacy 19-family level/delta order and appends explicit masks without entering production. The current 40-column `external_feature_panel_v2` artifact retains value and fractional coverage but not the selected source timestamp witnesses, so it cannot be silently projected into the model adapter. A later panel/artifact version must preserve those timestamps and pass `verify-panel`; assigning the decision time as a synthetic event or release time is prohibited.
- **Point-in-time is non-negotiable.** Every exogenous value must be lagged to when it was actually available (funding settles on a schedule; on-chain/macro publish with delay). Leakage here fabricates fake in-sample edge — which is exactly the trap this whole effort kept falling into.
- **Survivorship:** include delisted/changed perps when buying archival, or cross-sectional tests are biased.

## How this plugs into the harness

`scripts/research/`:
- `datafeed.py` — incremental cache + point-in-time alignment + `load_panel`.
- `harness.py` — cost-aware walk-forward, **block-bootstrap sharpe CI**, **deflated `P(SR>0)`** (multiple-testing haircut), regime split, cross-sectional book.
- `run_example.py` — runnable demo; today it correctly **flags small-sample unreliability** on the 30-day window.

Once Tier-0 accumulation (or a Tier-1 buy) gives ≥ ~1500 clean OOS observations, re-run the harness. If funding/basis (or order-flow) clears the evaluation bar across symbols and regimes, *then* wire it into the trading system (the inert foundation is already merged: commit `e513c8d5`). If it doesn't clear the bar at that sample size, the edge isn't there and the strategy needs a different basis — but you'll know honestly, which is the entire point.

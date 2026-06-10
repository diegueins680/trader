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

**Action:** schedule `python3 scripts/research/datafeed.py <symbols>` (hourly/daily) so the 30-day-limited series build a permanent history in `data/research/`. Cost: $0; you just need to start now — every day not collecting is lost.

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

- **Schema:** one CSV per `(symbol, interval)` in `data/research/` — `openTime, open, high, low, close, volume, funding, oi, basis, taker, …`. Exogenous columns are stored **already point-in-time aligned** to bar close (`datafeed.align_pit`). Drop-in archival data must follow the same schema/alignment.
- **Point-in-time is non-negotiable.** Every exogenous value must be lagged to when it was actually available (funding settles on a schedule; on-chain/macro publish with delay). Leakage here fabricates fake in-sample edge — which is exactly the trap this whole effort kept falling into.
- **Survivorship:** include delisted/changed perps when buying archival, or cross-sectional tests are biased.

## How this plugs into the harness

`scripts/research/`:
- `datafeed.py` — incremental cache + point-in-time alignment + `load_panel`.
- `harness.py` — cost-aware walk-forward, **block-bootstrap sharpe CI**, **deflated `P(SR>0)`** (multiple-testing haircut), regime split, cross-sectional book.
- `run_example.py` — runnable demo; today it correctly **flags small-sample unreliability** on the 30-day window.

Once Tier-0 accumulation (or a Tier-1 buy) gives ≥ ~1500 clean OOS observations, re-run the harness. If funding/basis (or order-flow) clears the evaluation bar across symbols and regimes, *then* wire it into the trading system (the inert foundation is already merged: commit `e513c8d5`). If it doesn't clear the bar at that sample size, the edge isn't there and the strategy needs a different basis — but you'll know honestly, which is the entire point.

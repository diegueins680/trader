# Cost and execution-assumption audit

## Current production/backtest model

The Haskell backtester charges costs as equity fractions on position changes and records fee, slippage, spread, and funding attribution. Configurable terms include:

- per-side fee;
- base slippage plus volatility-sensitive slippage;
- size-dependent slippage/impact coefficient and exponent;
- total bid–ask spread plus a volatility increment, with half charged per crossing;
- signed or unsigned funding and whether funding applies on the opening interval;
- turnover from entry, exit, reversal, partial exit, and resize;
- bar-delay controls in research paths;
- venue quantity/notional filters and live book-walk checks in the execution path.

The canonical venue floors in `Trader.CostCalibration` are 5 bp taker fee per side, 0.5 bp slippage per side, and 1 bp full spread, implying a 12 bp round trip. The minimum forecast edge floor applies a 1.5 multiplier, or 18 bp. Realized per-side slippage can replace the configured prior only after eight valid fills; it uses a bounded 64-fill median, shrinkage of 16 observations, a 25% prior floor, a 1% cap, and rejection beyond 5% as malformed evidence.

These are safety floors, not claims about every symbol or regime. A candidate comparison must freeze the exact contemporaneous fee tier, spread/depth evidence, funding schedule, and latency before development evaluation.

## Existing research-campaign assumptions

The historical funding/reversal campaigns use 10 bp per unit turnover and 20 bp for the doubled-cost stress. Funding is charged from every endpoint-returned settlement in the exact `(left close, right close]` interval using signed holdings and a causal mark. The more recent reversal campaigns propagate units/equity through rebalances, charge cash entry and terminal liquidation, test an additional delay, and stop at the first solvency or drawdown breach.

The historical funding result shows why gross forecasts are insufficient: the nested path averaged approximately -2.09 bp gross per eight-hour bar and 3.02 bp modeled cost. Simply inverting the sign would still have produced about -0.93 bp net before any extra impact or missed-fill allowance. At 2x cost its outer-OOS Sharpe was -2.059 and total return -88.86%; one extra bar produced Sharpe -1.418 and total return -79.32%.

## What is not established

The current bar backtest does not establish:

- queue position or maker-fill priority;
- partial and missed fills from historical order placement;
- nonlinear impact calibrated by order size relative to book/volume;
- adverse selection after a fill;
- sub-bar decision, network, exchange, and acknowledgement latency;
- maintenance-margin and intrabar liquidation paths;
- venue outage, withdrawal, custody, settlement, or counterparty loss;
- cross-venue capital fragmentation;
- infrastructure cost allocation to a strategy.

Live book walking and market-risk admission improve current-order safety, but they do not retroactively make historical bars executable. Consequently, OFI and other microstructure candidates remain blocked until event replay and fill assumptions exist.

## Frozen requirements for future candidates

Every candidate registration fixes or requires a pre-development freeze for:

| Component | Baseline | Required stresses |
|---|---|---|
| Fees | Causal venue tier; taker unless maker fill is proven | 1.5x and 2x all-in costs |
| Spread | Causal observed or conservative symbol/interval floor | doubled spread and extreme 25 bp/turnover scenario |
| Slippage | Shrunk live estimate or conservative prior | 1.5x, 2x, and adverse-impact sensitivity |
| Funding | Every signed settlement with causal mark | adverse funding where direction permits |
| Turnover | Absolute drift-aware target change; entry and liquidation included | report by symbol/fold/regime |
| Latency | Next-bar open for bar candidates; 250 ms for OFI only with event replay | one extra bar; 1000 ms for OFI |
| Quantity/notional | Historical/current venue filters frozen point in time where available | rejected/minimum-size orders count as missed fills, not free trades |
| Partial/missed fills | Taker fill only where replay supports available depth | OFI extreme case uses 50% fill probability and penalizes misses |
| Impact | Adverse square-root sensitivity in addition to visible book | zero-impact result cannot authorize promotion |
| Infrastructure | Report if material relative to net PnL | cannot be omitted from a capacity claim |

No leverage is optimized. Research exposure is normalized and constrained to 0.25 gross unless a lower existing champion exposure is required. Long and short costs and funding are reported separately.

## Acceptance interpretation

A candidate fails if it improves RMSE but loses money net, needs a favorable maker-fill assumption, fails at 1.5x/2x costs or one extra bar, depends on one anomalous fill/trade, or increases drawdown/tail risk. Cost parameters cannot be loosened after a development or holdout result is seen.

# Costs, simulator fidelity, ablation and robustness

The registered baseline charges 10 basis points per unit notional turnover:
5 bp taker fee, 0.5 bp half-spread and 4.5 bp slippage. These are assumptions,
not reconstructed historical account fees or observed spreads. Both entry and
terminal liquidation pay costs. Signed funding uses every admitted settlement
and its mark; mean funding and fee contributions for every seed/scenario are in
[all-seed-results.csv](all-seed-results.csv). Maker execution is not claimed.

Delayed next-close fills prevent same-close exposure gains. Nevertheless this
is historical replay with assumed fills, not exchange execution validation.
No real policy execution exists, so simulator-to-actual-fill performance gap is
**unmeasured**, not zero. Training and later historical folds share the simulator;
chronological reuse alone cannot remove simulator bias.

| Scenario | RL failed paths / 1,080 |
|---|---:|
| base | 686 |
| cost1_5x | 688 |
| cost2x | 697 |
| delay1bar | 693 |
| extreme25bp | 700 |
| funding2x | 700 |
| impact10bp | 688 |
| missed10pct | 687 |
| partial50pct | 454 |

All horizons, algorithms, seeds and folds remain in these counts. Stress paths
are correlated sensitivities, not independent trials. Lower failure under 50%
fills reflects less realized exposure, not proof of a better execution method.
No scenario qualifies a policy for promotion. At baseline, failures comprise
387 drawdown, 280 turnover, 14 exposure and 5 capital-floor breaches. Hard gates
must not be weakened to rescue these candidates.

The preregistered inventory-penalty ablation retrains CQL with coefficient zero.
At horizons 1 and 3, seed-level descriptive returns and failure counts match
within reported precision. Horizon 6 changes a few decisions and returns without
resolving rejection. The small variance coefficient has no demonstrated robust
incremental value. This is the only trained component/reward ablation; no claim
is made about unrun feature-removal, distributional-critic, normalization,
architecture, support-range or risk-shield-removal ablations. Safety controls are
never removed from an executable/order path.

Engineering fixtures exercise exact fees/funding/liquidation, future perturbation,
non-finite observations/actions, disabled mode, ownership mismatch, delayed,
partial/missed fills, bounded episodes, train-prefix isolation, deterministic
replay, corrupt/incompatible/hash-mismatched artifacts, terminal bootstrap,
three training seeds, and known OPE identities/no support. Missing timestamps,
revised bar vintages, true queue dynamics, downtime, unseen exchanges, delistings,
1h/4h adjacency, liquidity regimes and prospective distribution shift have no
admitted empirical evidence. They remain failed or untested gates, not passes.

## Reward-hacking and simulator exploitation audit

| Mechanism | Evidence / disposition |
|---|---|
| Hide terminal inventory losses | Valid terminal fixture debits full costs and closes units; invalid paths remain failed |
| Earn next price change before fill | Exact delayed-fill fixture rejects this accounting pattern |
| Churn for gross rewards | Fees are in equity and independent ledger; turnover breaches reject policies |
| Hide unrealized losses | Mark-to-market equity each bar, checked against gross/funding/cost ledger |
| Bootstrap through terminal | Dedicated value-target regression forbids terminal bootstrap |
| Backdate funding | Interval settlements charged to old units before endpoint fills |
| Learn from future observations | Prefix perturbation and physically sliced replay buffers |
| Exploit missing/invalid prices | Fail stopped replay; never fill at manufactured zero price |
| Exploit episode horizon | Remaining horizon is observed; end-effect generalization unproven |
| Exploit risk gaps | Gap breach is retained failure, not clipped P&L or guaranteed stop |
| Benefit only from one seed | All three seeds and every failure reported; no seed selection |
| Excessive inactivity / tail bets | Cash and deterministic comparators retained; action/ES fields retained externally |
| Exploit realistic queue/impact mismatch | Not ruled out; no L2/own-order calibration, blocks adoption |

These checks establish bounded accounting properties, not absence of all reward
hacking. Worse than 15% drawdowns occur because endpoint detection cannot prevent
market gaps. That finding is evidence against promotion, not a simulator license
to claim compliance with the drawdown limit.

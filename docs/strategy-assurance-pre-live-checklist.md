# Before an automated crypto strategy trades real capital

Use this one-page evidence checklist before moving an automated strategy from research or paper trading to capital at risk. A checked box means the named evidence exists and has been reviewed—not that the strategy is profitable, safe in every market, certified, or compliant with every applicable rule.

This is an engineering self-check, not investment, legal, tax, accounting, or regulatory advice and not a return or loss-prevention guarantee.

## Identify the decision

- [ ] The strategy name, version, repository, and exact reviewed commit are recorded.
- [ ] The exchange, account, deployment, symbols, intervals, and settlement asset are unambiguous.
- [ ] One owner can state the decision: operate, operate with restrictions, remain paper-only, or stop pending remediation.
- [ ] Maximum capital at risk and the people authorized to accept residual risk are named.

## Reconcile the economics

- [ ] Exchange income history—not reconstructed fills alone—reconciles realized P&L, funding, commissions, rebates, and adjustments for the review period.
- [ ] Transfers, bonuses, deposits, and withdrawals are disclosed separately rather than treated as trading revenue.
- [ ] Infrastructure cost, market-data cost, borrowing, slippage, spread, and realistic maker/taker fees are included where applicable.
- [ ] Any truncated export, missing asset, unknown income type, or accounting disagreement remains visible as a limitation.

## Challenge the strategy evidence

- [ ] Training, validation, walk-forward, and untouched holdout periods are point-in-time and separated.
- [ ] The full search history or trial count is retained; the selected result is not presented as if it were the only test.
- [ ] Activity, turnover, drawdown, risk-adjusted results, parameter sensitivity, and regime dependence are reported with costs.
- [ ] Backtest order assumptions match the production venue closely enough to support the operating decision.

## Prove execution and risk controls

- [ ] Partial fills, rejected orders, duplicate requests, stale prices, rate limits, timeouts, and restart behavior have explicit handling.
- [ ] Position, leverage, order-size, daily-loss, drawdown, loss-streak, and portfolio exposure limits fail closed.
- [ ] Stop-loss, take-profit, liquidation-distance, reduce-only, and close-ownership behavior is tested against the actual account mode.
- [ ] A stale or incomplete signal cannot silently create new exposure.

## Prove the deployment boundary

- [ ] Exactly one production executor owns each live position; standby systems are demonstrably read-only.
- [ ] Credentials are least-privilege, scoped, rotated, and excluded from source, logs, tickets, reports, and browser storage.
- [ ] Withdrawal permission, seed phrases, private keys, and unrestricted cloud credentials are never required.
- [ ] Readiness, persistence, time synchronization, observability, rollback, and post-restart exchange reconciliation are tested.

## Make operations accountable

- [ ] Alerts identify who responds, within what time, and what evidence determines pause, reduce-only, restart, or shutdown.
- [ ] Incidents, manual overrides, accepted risks, and remediations have durable owners and timestamps.
- [ ] The production revision and configuration passed the repository's canonical verification before release.
- [ ] A rollback or capital-reduction path can be executed without changing strategy logic under pressure.

## Minimum go-live boundary

Do not treat the deployment as ready when its identity is ambiguous, exchange economics do not reconcile, hard exposure and loss limits are absent, stale data can open risk, restart ownership is unknown, secrets exceed least privilege, or rollback has not been exercised. Missing evidence is a finding, not permission to infer the safest answer.

When a decision depends on independent review, the standard Strategy Assurance engagement covers one strategy, one exchange account, and one production deployment for USD 2,500 paid before kickoff. It delivers an evidence-backed operate, restrict, paper-only, or stop-pending-remediation memo within five business days after complete inputs are available. It does not provide custody, discretionary trading, signals, certification, or a performance guarantee.

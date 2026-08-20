# Strategy Assurance Proposal and Statement of Work

Replace every bracketed field before sending this document. Commercial and legal terms should be reviewed for the provider's jurisdiction and business structure.

## Engagement

**Client:** [CLIENT LEGAL NAME]
**Provider:** [PROVIDER LEGAL NAME]
**Proposal date:** [YYYY-MM-DD]
**Valid through:** [YYYY-MM-DD]
**Client decision owner:** [NAME AND TITLE]

## Objective

Provide an evidence-backed technical and economic assessment of one algorithmic crypto-trading deployment so the client can decide whether to operate it as-is, operate with restrictions, continue paper trading, or stop pending remediation.

This engagement is engineering assurance. It is not investment, legal, tax, regulatory, or accounting advice; an audit opinion; custody; discretionary trading; or a guarantee of profitability or future performance.

## Fixed scope

- **Strategy:** [STRATEGY NAME AND VERSION]
- **Repository and commit:** [REPOSITORY] at [COMMIT]
- **Exchange account:** [VENUE AND NON-SECRET ACCOUNT LABEL]
- **Deployment:** [DEPLOYMENT NAME / REGION]
- **Revenue period:** [START] through [END]
- **Settlement asset:** [ASSET]
- **Infrastructure cost supplied by client:** [AMOUNT AND ASSET]

The standard review covers one strategy, one exchange account, and one production deployment. Anything else requires written change approval before work begins.

## Deliverables

1. Exchange-reconciled revenue report with machine-readable JSON plus daily and symbol CSV exports.
2. Strategy-evidence assessment covering walk-forward design, costs, activity, selection bias, holdout discipline, and backtest-to-production gaps.
3. Execution and risk-control assessment covering orders, fees/funding, exposure, loss/drawdown halts, stale data, and restart ownership.
4. Deployment assessment covering credentials, executor boundaries, persistence, readiness, observability, rollback, and verification evidence.
5. Red/amber/green findings ledger with severity, evidence, impact, owner, recommended action, and disposition.
6. Decision memo with an operate, restrict, paper-only, or stop-pending-remediation conclusion.
7. One 60-minute readout and one written clarification round requested within seven calendar days after delivery.

## Schedule

Delivery is targeted within five business days after all required inputs and access are complete. Missing, contradictory, or inaccessible evidence pauses the delivery clock and will be recorded in writing.

## Fees and payment

**Fixed fee:** USD 2,500, due before kickoff.

The fee excludes remediation implementation, incident response, new strategy or deployment reviews, travel, third-party charges, and continuous monitoring. After delivery, the client may elect monthly monitoring at USD 399 per reviewed deployment under a separate written order.

## Client responsibilities

The client will:

- provide the exact reviewed commit, deployment configuration, strategy evidence, exchange evidence, cost inputs, and incident history;
- provide customer-run exports or temporary least-privilege read-only access without withdrawal or trading permission;
- identify an owner able to answer architecture and operational questions;
- verify the accuracy and completeness of supplied business/accounting inputs; and
- revoke temporary access after delivery.

## Access and confidentiality

Credentials must not be placed in this proposal, tickets, chat transcripts, or final reports. Temporary access must have an agreed purpose and expiry. Each party will protect the other party's confidential information using reasonable safeguards and use it only for this engagement, subject to the parties' governing agreement.

## Assumptions and limitations

- Findings describe the reviewed evidence, commit, environment, and period only.
- Missing evidence is reported as a limitation or finding, not silently inferred.
- Exchange income history is the realized accounting authority; transfers, bonuses, and unclassified income are excluded from net revenue and separately disclosed.
- The client remains responsible for trading, capital allocation, risk acceptance, legal compliance, and accounting treatment.
- Any liability, warranty, indemnity, intellectual-property, confidentiality, termination, or dispute terms not stated here are governed by [MASTER AGREEMENT OR ATTACHED TERMS].

## Acceptance

By signing, the parties approve this scope, fee, schedule, access boundary, and the governing agreement identified above.

| Client | Provider |
| --- | --- |
| Name: [NAME] | Name: [NAME] |
| Title: [TITLE] | Title: [TITLE] |
| Signature: ____________________ | Signature: ____________________ |
| Date: [DATE] | Date: [DATE] |

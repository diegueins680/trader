# Strategy Assurance Decision Memo

## Document control

| Field | Value |
| --- | --- |
| Client | [CLIENT] |
| Strategy | [STRATEGY] |
| Deployment | [DEPLOYMENT] |
| Reviewed commit | [COMMIT] |
| Exchange account label | [NON-SECRET LABEL] |
| Review period | [START] through [END] |
| Report date | [DATE] |
| Reviewer | [NAME] |

## Decision

**Conclusion:** [OPERATE AS-IS / OPERATE WITH RESTRICTIONS / PAPER-TRADE ONLY / STOP PENDING REMEDIATION]

[Two or three sentences explaining the decision, the strongest evidence, and the most important limitation.]

## Executive findings

| ID | Severity | Area | Finding | Required action | Owner | Due |
| --- | --- | --- | --- | --- | --- | --- |
| SA-001 | [RED/AMBER/GREEN] | [AREA] | [FINDING] | [ACTION] | [OWNER] | [DATE] |

## Revenue reconciliation

Attach the dashboard-generated Markdown snapshot and reference its matching evidence JSON, daily CSV, and symbol CSV here.

| Measure | Amount | Evidence |
| --- | ---: | --- |
| Realized P&L | [AMOUNT] | [FILE / HASH] |
| Funding | [AMOUNT] | [FILE / HASH] |
| Signed commission | [AMOUNT] | [FILE / HASH] |
| Rebates / other operating | [AMOUNT] | [FILE / HASH] |
| Exchange net | [AMOUNT] | [FILE / HASH] |
| Current unrealized P&L | [AMOUNT] | [FILE / HASH] |
| Infrastructure cost | [AMOUNT] | [CLIENT INPUT] |
| Net revenue | [AMOUNT] | [FILE / HASH] |
| Excluded non-operating | [AMOUNT] | [FILE / HASH] |

**Completeness:** [COMPLETE / REVIEW REQUIRED]
**Limitations:** [TRUNCATION, UNCLASSIFIED TYPES, MISSING PERIODS, OR NONE]

## Strategy evidence

Document the walk-forward design, activity, modeled costs, selection process, holdout state, uncertainty, and any gap between research assumptions and production behavior. Link every material statement to a command, artifact, source location, or supplied record.

## Execution and risk controls

Cover maker/taker behavior, order fallbacks, fees/funding, sizing and exposure bounds, loss/drawdown halts, stale data, position ownership, recovery, and fail-closed behavior.

## Deployment controls

Cover credential scope, sole-executor boundaries, persistence, readiness, monitoring, backups, rollback, incident history, and canonical verification results.

## Detailed findings

For each finding include:

- **ID / severity / area**
- **Condition:** what was observed
- **Evidence:** reproducible reference
- **Impact:** plausible operational or financial consequence
- **Recommendation:** concrete corrective action
- **Owner / target date**
- **Disposition:** open, accepted, remediated, or not applicable
- **Verification:** how closure will be demonstrated

## Restrictions and remediation order

1. [IMMEDIATE CONTROL OR STOP CONDITION]
2. [REQUIRED BEFORE LIVE OPERATION]
3. [FOLLOW-UP IMPROVEMENT]

## Evidence inventory

| Artifact | Version/hash | Purpose | Limitation |
| --- | --- | --- | --- |
| [ARTIFACT] | [HASH] | [PURPOSE] | [LIMITATION] |

## Limitations and reliance

State missing evidence, customer-supplied assumptions, inaccessible systems, time bounds, testnet/live distinctions, and matters requiring legal, tax, regulatory, or accounting professionals.

## Client disposition

| Finding | Decision | Owner | Date |
| --- | --- | --- | --- |
| [ID] | [ACCEPT / REMEDIATE / REJECT AS NOT APPLICABLE] | [OWNER] | [DATE] |

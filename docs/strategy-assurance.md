# Strategy Assurance

Strategy Assurance is a fixed-scope technical and economic review for teams operating an algorithmic crypto-trading system. It converts Trader's existing safety, deployment, execution, and exchange-accounting evidence into a report a founder or engineering lead can act on.

This is engineering assurance, not investment advice, a profitability guarantee, custody, discretionary trading, or a promise that a strategy will perform in the future.

## Standard review

**Price:** USD 2,500 paid before kickoff
**Scope:** one strategy, one exchange account, and one production deployment
**Turnaround:** five business days after complete inputs are available

The customer receives:

1. **Exchange-reconciled economics.** Realized P&L, funding, signed commissions, rebates, operating adjustments, excluded transfers, infrastructure cost, net revenue, and maker/taker execution for an agreed period.
2. **Strategy evidence review.** Walk-forward design, activity, cost assumptions, selection bias, holdout discipline, risk-adjusted results, and gaps between backtest assumptions and executable behavior.
3. **Execution and risk review.** Order path, fee/funding exposure, position limits, drawdown and loss halts, stale-data behavior, restart ownership, and fail-closed controls.
4. **Deployment review.** Credential boundaries, sole-executor configuration, persistence, readiness, observability, rollback, and CI verification evidence.
5. **Decision memo.** A concise red/amber/green finding register, prioritized remediation plan, and explicit conclusion: operate as-is, operate with restrictions, paper-trade only, or stop pending remediation.
6. **Readout.** A 60-minute findings session and one written clarification round within seven days.

Deliverables are a Markdown/PDF report, a machine-readable findings ledger, and the revenue export used by the report. Reproducible commands and source evidence accompany each finding where access permits.

Delivery artifacts are ready to copy and tailor:

- [Proposal and statement of work](strategy-assurance-proposal-template.md)
- [Decision memo template](strategy-assurance-report-template.md)
- [Machine-readable findings template](strategy-assurance-findings-template.json)
- [Shareable pre-live evidence checklist](strategy-assurance-pre-live-checklist.md)

For a real prospect, generate all pre-sale artifacts from one validated scope instead of editing four copies by hand:

```bash
npm run assurance:kit -- \
  --client "Client legal name" \
  --provider "Provider legal name" \
  --decision-owner "Name, title" \
  --strategy "Strategy name/version" \
  --deployment "Deployment name/region" \
  --repository "Repository identifier" \
  --commit "Reviewed commit"
```

The local-only generator creates `proposal.md`, `evidence-request.md`, `outreach.md`, `payment-request.md`, `payment-request.json`, and `engagement.json` under a client-and-date-specific `.tmp/strategy-assurance/` directory. The pro-forma payment request is not a tax invoice or receipt; approved payment instructions are supplied separately through a secure channel, never committed to the package. The generator validates dates and commercial bounds, rejects multiline field injection, contains no credential input, performs no network request, and will not replace an existing package without explicit `--force`. Use `npm run assurance:kit -- --help` for optional period, price, validity, account-label, and output arguments.

Import the generated engagement record into the local pipeline and advance it when—not before—the corresponding business event occurs:

```bash
npm run assurance:pipeline -- import \
  --engagement .tmp/strategy-assurance/CLIENT-DATE/engagement.json

npm run assurance:pipeline -- advance \
  --id ENGAGEMENT_ID \
  --status accepted \
  --at 2026-08-14

npm run assurance:pipeline -- advance \
  --id ENGAGEMENT_ID \
  --status paid \
  --amount 2500 \
  --at 2026-08-15

npm run assurance:pipeline -- summary
```

Allowed transitions are proposal → accepted or lost; accepted → paid or cancelled; paid → in-delivery or refunded; in-delivery → delivered or refunded; delivered → monitoring or closed; and monitoring → closed. A transition to `paid` or `refunded` requires `--amount` with the actual USD cash movement. A transition to `delivered` requires `--hours` with actual delivery effort. The registry uses dated lifecycle events and atomic replacement, rejects backward or skipped transitions, prevents refunds above recorded cash, makes identical imports idempotent, and performs no outreach or billing action. The versioned summary exposes exact conversion ratios, expired proposals, next actions, open and booked contract value, gross/refunded/net cash, delivered net cash, delivery hours, realized review revenue per hour, and contracted monitoring MRR/ARR. Run `npm run assurance:pipeline -- --help` for JSON output and alternate registry paths.

After delivery, generate the separate monitoring order from the exact pipeline scope:

```bash
npm run assurance:renewal -- \
  --id ENGAGEMENT_ID \
  --offer-date 2026-08-15 \
  --start 2026-09-01 \
  --months 3
```

The generator requires the engagement to be at `delivered`, uses its contracted USD 399 monthly price unless the original engagement was generated with another monitoring price, and computes the exact initial-cycle contract value. It produces `monitoring-order.md` plus `monitoring-order.json`, makes no external action, does not change pipeline state, and refuses implicit overwrites. Review the order and governing terms before sending; advance the engagement to `monitoring` only after actual acceptance.

After reconciling Binance revenue in the web dashboard, fill the optional client, strategy, deployment, and commit fields. The dashboard can then download a client-ready Markdown snapshot, versioned evidence JSON, daily CSV, and symbol CSV. Report identity stays browser-local and is not sent to the API.

## Optional recurring assurance

After a standard review, monthly monitoring is USD 399 per deployment. It includes one monthly revenue reconciliation, drift and control-state review, a change summary, and a 30-minute readout. It excludes implementation work, incident response, new strategy reviews, and continuous on-call coverage.

A focused remediation sprint can be quoted separately only after the customer accepts the review findings. Keeping diagnosis and implementation commercially distinct makes the original review useful even if another team performs the fixes.

## Inputs and access boundary

Before the clock starts, the customer provides:

- the strategy repository and exact reviewed commit;
- deployment manifests and non-secret runtime configuration;
- backtest/optimizer evidence and the intended production universe;
- exchange exports or a read-only Binance API key restricted to the required account-data endpoints;
- infrastructure cost for the review period; and
- the production owner who can answer architecture and incident-history questions.

Never request withdrawal permission, trading permission, seed phrases, private wallet keys, or unrestricted cloud credentials. Prefer customer-run exports. If temporary access is necessary, use least privilege, agree on an expiry, avoid copying credentials into tickets or reports, and have the customer revoke access after delivery.

## Acceptance criteria

The review is complete when:

- the reviewed commit, environment, strategy, account, asset, and period are identified;
- exchange revenue is reconciled or the exact missing evidence is documented;
- every material finding has severity, evidence, impact, owner, and recommended action;
- limitations and unresolved assumptions are explicit;
- the decision memo and findings ledger are delivered; and
- the readout has been held or offered with at least two reasonable scheduling options.

The review does not certify regulatory compliance. Legal, tax, accounting, and jurisdiction-specific questions must go to qualified professionals.

## Delivery playbook

### Qualification

A suitable customer already runs or is preparing to run an automated strategy, can identify the code and deployment under review, and values loss prevention or operational confidence enough to act on findings. Decline engagements whose primary request is a return guarantee, strategy signals, custody, or operating the customer's account.

Ask these questions before quoting:

1. What strategy, venue, account, and deployment are in scope?
2. Is it live, paper, or pre-production, and what capital can it affect?
3. What decision must this review unblock?
4. Which backtest, exchange, incident, and deployment evidence exists today?
5. Who owns remediation and signs off on accepted risk?

### Kickoff and review

1. Record the scope, reviewed commit, evidence inventory, access method, and exclusions.
2. Run the canonical repository verification appropriate to the reviewed revision.
3. Reconcile revenue first; disagreements between reconstructed performance and the exchange ledger become findings, not manual adjustments.
4. Trace each production risk control from configuration through implementation, verification, telemetry, and operator response.
5. Rank findings by plausible financial impact and urgency. Do not bury missing evidence inside a low-severity appendix.
6. Issue the report, findings ledger, and readout. Track accepted, remediated, and explicitly retained risks separately.

### Commercial funnel

Start with people already operating small systematic strategies: trading-tool founders, quantitative developers, crypto funds with lean engineering teams, and agencies maintaining exchange automation. A short message is enough:

> I offer a fixed five-day engineering review of one automated trading deployment. It reconciles exchange P&L and fees, tests strategy evidence and risk controls, and ends with a prioritized operate/restrict/stop memo. The standard review is $2,500. If that decision would be useful, I can send the exact evidence checklist.

Do not claim that the review improves returns. Sell a faster, evidence-backed operational decision and reduced risk of silent accounting, execution, or deployment failures.

Use the dated [prospecting brief](strategy-assurance-prospecting-brief.md) and [machine-readable queue](strategy-assurance-prospect-queue.json) to prioritize organization-level acquisition routes. Revalidate every official source before outreach: a public community or partner page demonstrates audience fit, not interest in the offer. Open-source communities require moderator permission, Hummingbot explicitly prohibits unsolicited direct messages, and no community member list should be scraped or treated as a sales list.

Prepare individually reviewed drafts for one to three exact queue organizations without making an external action:

```bash
npm run assurance:outreach -- \
  --provider "Provider legal name" \
  --sender "Name, title" \
  --prospect "OctoBot" \
  --prospect "Hummingbot Botcamp" \
  --prospect "QuantConnect Integration Partners"
```

The local-only command snapshots each official route and its dated evidence, tailors an initial permission or partner-evaluation note, computes the earliest follow-up after five business days, permits only one follow-up, hashes the exact drafts, and copies the pre-live checklist into the campaign package. It refuses more than three prospects, duplicate or unknown organizations, and watchlist entries unless `--include-watchlist` is explicitly supplied. Its records remain `prepared` with null send and response fields: it cannot send, submit forms, join communities, claim affiliations, collect member data, or advance the commercial pipeline.

Import a reviewed campaign into the separate local acquisition registry, then record a transition only after the corresponding event actually occurs:

```bash
npm run assurance:acquisition -- import \
  --campaign .tmp/strategy-assurance/outreach/CAMPAIGN/campaign.json

npm run assurance:acquisition -- advance \
  --id LEAD_ID \
  --status contacted \
  --channel "Official organization contact form" \
  --evidence "Receipt or sent-record reference" \
  --at 2026-08-20

npm run assurance:acquisition -- summary --as-of 2026-08-20
```

The acquisition lifecycle is prepared → contacted → follow-up-sent → responded → qualified → proposed. A prepared or responded lead may instead become disqualified, and an unanswered lead may become closed-no-response only five business days after its single recorded follow-up. Every transition requires a short evidence reference; contact events also require the channel. The manual `advance proposed` route requires the generated `engagement.json`, verifies the provider and proposal date, and records its digest and stable ID without importing it into the commercial pipeline. Campaign import verifies the record, message, follow-up, and checklist hashes, is idempotent for identical evidence, and refuses a second campaign for an organization already in the registry.

Use `npm run assurance:acquisition -- summary` for exact prepared-to-contacted, contacted-to-responded, responded-to-qualified, qualified-to-proposed, contacted-to-proposed, and follow-up-to-response ratios, plus current stages, performance by queue source kind, eligible follow-ups/closures, and dated next actions. The default registry is `.tmp/strategy-assurance/acquisition.json`; it contains evidence references, not credentials or pasted message bodies, and makes no external action.

For the first week, validate the top four sources, select at most three distinct routes, make at most two reviewed organization-level asks plus one formal partner or moderator-permission request, and prepare proof only for respondents. Record actual outcomes in the acquisition registry. Wait at least five business days after the real contact date before one follow-up and another five business days before closing it as unanswered. The repository does not send messages or submit partner forms.

Once a prospect confirms that the decision is useful, generate their commercial package and review the proposal for applicable legal terms. After the real proposal event, commit one recoverable handoff instead of separately updating the two registries:

```bash
npm run assurance:handoff -- commit \
  --lead LEAD_ID \
  --engagement .tmp/strategy-assurance/CLIENT-DATE/engagement.json \
  --evidence "Reviewed proposal sent-record reference" \
  --at 2026-08-21

npm run assurance:handoff -- reconcile --as-of 2026-08-21
```

The command computes and validates both next states before any write. It saves the commercial pipeline first and the acquisition link second; if that second atomic write is interrupted, the identical command safely completes it on rerun. The reconciliation report exposes qualified leads awaiting a proposal, proposed leads missing from the pipeline, inconsistent identity or digest evidence, linked quoted value and cash, and pipeline records with no acquisition link. Unlinked pipeline records are reported for provenance review because direct and referral engagements can be legitimate. The handoff does not generate, send, sign, invoice, request payment, charge, or perform any external action.

Send the evidence request only after scope and payment expectations are understood. Keep `engagement.json` with the delivery record so the reviewed scope, quoted economics, and later monitoring conversion share one stable identity.

Track weekly:

- prepared, contacted, responded, qualified, and proposed leads by source kind;
- qualified conversations;
- evidence checklists sent;
- proposals issued and accepted;
- days from complete inputs to delivery;
- findings accepted and remediated;
- standard reviews converted to monthly monitoring; and
- revenue and delivery hours per engagement.

Use `npm run assurance:acquisition -- summary` for pre-proposal source conversion and next actions, `npm run assurance:handoff -- reconcile` for qualified-to-linked leakage and linked value, and `npm run assurance:pipeline -- summary` as the baseline for proposal, payment, delivery, cash, unit-economics, expiry, and contracted monitoring metrics. Qualitative findings still require the explicit delivery artifacts; do not infer them from pipeline status.

At two standard reviews per month, the offer produces USD 5,000 in one-time monthly revenue before add-ons. Monitor actual delivery hours and conversion before changing the price or scope.

## Internal kickoff checklist

- [ ] Signed scope and payment received
- [ ] Reviewed commit and deployment identified
- [ ] Secrets/access method approved and expiry recorded
- [ ] Exchange period, asset, and infrastructure cost agreed
- [ ] Evidence inventory complete
- [ ] Revenue ledger exported and truncation warnings resolved
- [ ] Canonical verification captured
- [ ] Findings peer-checked
- [ ] Decision memo and machine-readable ledger delivered
- [ ] Readout completed or offered
- [ ] Temporary access revoked

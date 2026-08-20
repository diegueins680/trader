# Strategy Assurance prospecting brief

**Research date:** 2026-08-17
**Purpose:** identify organization-level routes to operators of automated crypto strategies
**External actions performed:** none

This brief turns the Strategy Assurance offer into a permission-based acquisition queue. Every organization below publicly serves algorithmic-trading users or operates adjacent infrastructure, but public fit evidence is not evidence of need, budget, approval, or interest. Validate the route immediately before any outreach, use only an official organization channel, and record a real response before changing an engagement's pipeline status.

The machine-readable companion is [strategy-assurance-prospect-queue.json](strategy-assurance-prospect-queue.json).
The shareable proof asset is [the pre-live evidence checklist](strategy-assurance-pre-live-checklist.md). Prepare bounded, local-only drafts with `npm run assurance:outreach -- --help`; generation does not contact any organization.

## Ranking method

The score is `audience fit + buyer access + commercial alignment + current activity - conflict penalty`, using the following bounds:

- audience fit: 0–3;
- organization-level access: 0–2;
- commercial alignment: 0–3;
- current public activity: 0–2; and
- overlap or channel-conflict penalty: 0–3.

Scores prioritize the next research action; they do not estimate conversion probability.

## Prioritized queue

| Rank | Organization and route | Score | Public fit evidence | Hypothesis and first ask |
| ---: | --- | ---: | --- | --- |
| 1 | **OctoBot** — official [product/contact page](https://www.octobot.cloud/trading-bot/) | 10 | OctoBot says users can create, backtest, and deploy automated strategies across more than 12 exchanges; the same page exposes community and sponsored-development routes. | A fixed independent readiness review could complement its built-in testing for advanced users or sponsored strategies. Ask whether the team would evaluate one co-branded pilot or refer one suitable operator; do not imply endorsement. |
| 2 | **Hummingbot Botcamp** — official [course portal](https://courses.botcamp.xyz/) | 9 | Botcamp describes professional training and certification for market makers and algorithmic traders, with participants building and deploying Hummingbot strategies. | Offer a capstone or alumni pre-live assurance clinic. First validate that a current cohort or alumni program is active, then ask the organization whether an independent operate/restrict/stop review would be useful. |
| 3 | **QuantConnect Integration Partners** — official [partner directory and application](https://www.quantconnect.com/integration-partners) | 9 | QuantConnect maintains a formal directory for independent consultants and firms and invites applications to serve its community. | Apply as an independent strategy-assurance provider, leading with deployment evidence and exchange reconciliation rather than strategy development. Do not claim partner status until accepted. |
| 4 | **Hummingbot Foundation** — official [community page](https://hummingbot.org/community/) | 8 | The Foundation describes a global community of developers and traders building algorithmic strategies; Discord is its primary community hub, and its page explicitly warns against unsolicited direct messages. | Ask a moderator in the appropriate public channel whether a no-pitch pre-live checklist or clinic is acceptable. Never direct-message members or scrape the community. |
| 5 | **Jesse** — official [documentation/community entry](https://docs.jesse.trade/) | 8 | Jesse documents self-hosted backtesting, live trading, exchange connectivity, risk helpers, and an official community route. | Ask maintainers for permission to offer an educational deployment-readiness session that complements Jesse's native metrics with exchange-reconciled economics and operational controls. |
| 6 | **HaasOnline** — official [white-label partner page](https://haasonline.com/partners/white-label) | 7 | HaasOnline markets trading automation to exchanges, platforms, wealth managers, and crypto projects through a formal enterprise partnership route. | Explore an independent customer-deployment acceptance review as a partner add-on. Expect an objection that its platform already provides enterprise reliability; focus only on the customer's strategy, configuration, and operating evidence. |
| 7 | **Freqtrade** — official [documentation and support route](https://docs.freqtrade.io/en/latest/) | 7 | Freqtrade is an open-source crypto bot with dry-run, live-trade, backtesting, optimization, and an official Discord support community. Its documentation stresses user responsibility and safe dry-run use. | Ask maintainers for permission before sharing a neutral pre-live evidence checklist. Do not turn support channels into a sales list or request exchange keys. |
| 8 | **Superalgos** — official [community resources](https://superalgos.org/community-resources.shtml) | 6 | Superalgos provides open-source strategy design, testing, deployment, and community collaboration; its documentation says the project is governed by contributors rather than a legal entity. | Treat this only as a community referral channel. Ask moderators whether an educational clinic is welcome; contract with an actual operator, never with an assumed central organization. |
| 9 | **Enflux** — official [market-making product page](https://www.enflux.io/market-making) | 5 | Enflux publicly describes algorithmic 24/7 market making, live execution telemetry, risk controls, and audit-grade traceability. | Keep on the watchlist: its own assurance and analytics overlap materially. Approach only if it explicitly wants independent second-line validation or a narrow external review; otherwise avoid competitive noise. |

## Qualification rule

Move an organization from `research-ready` to a real conversation only when all of these are true:

1. The official entry route is still current and permits the proposed kind of contact.
2. The audience operates or is preparing to operate an identifiable automated strategy.
3. The first ask is for permission, a pilot, or a partner evaluation—not access to member data.
4. The offer can stay within one strategy, one exchange account, and one deployment.
5. The counterparty is not seeking signals, custody, account operation, or a return guarantee.

Community membership, a partner application, or silence is not a qualified conversation. A positive response still is not an accepted proposal, and an accepted proposal still is not cash.

## First-week cadence

This cadence is a human checklist. Nothing in the repository sends messages, submits forms, joins communities, or changes external state.

### Day 1 — validate and tailor

- Re-open the official source for ranks 1–4 and confirm the route, rules, and recent activity.
- Select at most three organizations: one commercial platform, one education or partner route, and one open community.
- Tailor one permission-based first ask to the organization's published model. Do not collect individual member identities.

### Day 2 — two organization-level asks

- Send at most two individually reviewed messages through official organization routes.
- Lead with the decision the review supports, its fixed scope, and the evidence boundary.
- Record the date, route, and exact message locally; leave pipeline status untouched.

### Day 3 — one structured channel action

- Submit at most one formal partner application, or ask one community moderator publicly for permission to share an educational checklist.
- Never claim an affiliation before written acceptance. Respect Hummingbot's no-DM rule and equivalent community rules.

### Day 4 — prepare proof, not volume

- Prepare a redacted one-page sample finding or pre-live checklist for any organization that responds.
- Do not expose customer, exchange-account, credential, or live-strategy data.
- If there is no response, wait; do not add more channels merely to increase activity counts.

### Day 5 — qualify and measure

- Classify each outcome as no response, permission denied, permission granted, discovery requested, or not a fit.
- Generate a prospect-specific commercial kit only after a real decision owner confirms the review could be useful.
- Schedule one follow-up no sooner than five business days after the initial ask, then close the outreach after one unanswered follow-up.

Track source-to-conversation and conversation-to-proposal conversion by route. After 20 permission-based asks, stop any route with zero qualified conversations and concentrate on the route with the strongest paid-review conversion—not the largest audience.

## Guardrails

- No mass messaging, member scraping, purchased lists, unsolicited community DMs, or automated outreach.
- No promises of returns, loss avoidance, certification, regulatory compliance, or platform endorsement.
- No affiliate link inside the independent review unless it is disclosed and accepted in writing; independence is more valuable than a small referral commission.
- No secret, API key, wallet, or unrestricted infrastructure access during qualification.
- Re-check jurisdiction, privacy, marketing, sanctions, tax, and contract requirements with qualified professionals before commercial outreach.

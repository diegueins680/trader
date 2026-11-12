# Adoption memo: `exposure_rl_control_v1`

Date 2026-11-11. Registration `research-notes/registrations/exposure-rl-control-v1.json`. Registered run `dev-20261111` from commit `3ba6c042`: 104 planned jobs, 104 completed, 0 failed, about 15.5 CPU-hours.

## Decision

**No RL candidate passes the registered development gates. Do not integrate any of them; keep the current champion. The simpler comparators are at least as good on the evidence available.** The confirmation window (2026-05-01 → 2026-07-16) was not acquired or inspected and remains untouched.

A negative result was an acceptable outcome of this campaign. The implementation is complete and reproducible; the evidence does not support adoption.

## What was run

- **Decision problem:** bounded target exposure for one perpetual per account, 6h bars, fills at the next open, 8 bp per side, actual funding settlements, a projecting integer shield.
- **RL learners, all trained:** PPO (on-policy simulator), TD3 (off-policy simulator), IQL (offline, on a simulator-generated dataset). One extension, variance-penalised PPO, and one ablation, PPO without funding and BTC context.
- **Comparators:** cash, buy-and-hold, an unfitted regime rule on the same observations, a ridge forecast, behaviour cloning, a myopic contextual bandit, and the champion method.
- **Design:** four expanding walk-forward folds (2022-07 → 2025-03), five seeds per learner, fixed hyperparameters, checkpoints chosen on inner validation only.
- **Data:** 42,087 six-hour bars, 7 symbols, public Binance history. All of it previously exposed, so this is development evidence.

## Results (fleet aggregate, four test folds pooled, base cost)

| Policy | Ann. mean | Sharpe | Max drawdown | Sharpe at 2× cost | Sharpe, 1-bar delay |
|---|---|---|---|---|---|
| IQL | 41.4% | 1.02 | 38.0% | 0.55 | 0.97 |
| PPO without context (ablation) | 27.6% | 0.83 | 32.2% | 0.41 | 0.64 |
| TD3 | 23.9% | 0.62 | 32.1% | −0.27 | 0.36 |
| PPO | 6.5% | 0.18 | 42.1% | −0.29 | 0.03 |
| PPO, variance penalty | 4.4% | 0.61 | 9.8% | −0.05 | 0.40 |
| Buy-and-hold | 49.5% | 0.70 | 61.3% | 0.68 | 0.70 |
| Ridge forecast | 26.5% | 0.50 | 38.2% | 0.13 | 0.33 |
| Behaviour cloning | 13.4% | 0.59 | 25.1% | 0.28 | 0.37 |
| Champion method, own ledger | 4.9% | 0.25 | 26.7% | n/a | n/a |
| Regime rule | 3.8% | 0.08 | 46.5% | −0.20 | −0.32 |
| Contextual bandit | −31.7% | −0.90 | 73.1% | −2.76 | −0.68 |

The fleet aggregate sums six accounts each capped at 16.66%, so annual figures are on a fully allocated basis. Every seed, fold, symbol and stress is in [`results/comparison-report.md`](results/comparison-report.md) and [`results/all-seed-results.csv`](results/all-seed-results.csv).

### Primary statistic

Annualised return difference, candidate minus comparator, with a 95% stationary-bootstrap interval.

| Candidate | vs cash | vs buy-and-hold | vs ridge forecast | vs champion (own ledger) |
|---|---|---|---|---|
| IQL | +41.4% [−3.7%, +90.6%] | −8.1% [−76.8%, +60.0%] | +14.9% [−23.9%, +53.1%] | +36.5% [−5.6%, +79.0%] |
| TD3 | +23.9% [−18.9%, +68.5%] | −25.6% [−85.3%, +35.8%] | −2.7% [−49.8%, +40.7%] | +19.0% [−18.9%, +56.3%] |
| PPO | +6.5% [−32.8%, +48.2%] | −43.0% [−114.6%, +22.1%] | −20.0% [−59.3%, +14.7%] | +1.6% [−34.9%, +39.0%] |

**No interval excludes zero against cash, buy-and-hold, the ridge forecast or the champion on its own ledger.** Every candidate trails buy-and-hold on the point estimate.

### Gates

| Gate | IQL | TD3 | PPO |
|---|---|---|---|
| Lower bound > 0 against every comparator | fail | fail | fail |
| Point estimate ≥ 2% against every comparator | fail | fail | fail |
| Still positive at 2× cost and at one-bar delay | fail | fail | fail |
| Drawdown non-inferior | fail | fail | fail |
| Deflated Sharpe probability ≥ 0.95 | 0.04 | 0.01 | 0.00 |
| PBO ≤ 0.20 | 0.44 | 0.44 | 0.44 |
| At least 4 of 5 seeds above the best comparator | 1 of 5 | 0 of 5 | 0 of 5 |

### Stability

- **Seeds.** IQL is consistent (Sharpe 0.79–1.31). TD3 ranges from −0.44 to +0.80 and PPO from −0.17 to +0.58. Seeds are algorithmic replications, not independent market histories.
- **Regimes.** Most of every learner's return comes from fold 3 (2023-11 → 2024-07), when buy-and-hold made 135% annualised. In fold 1 (2022-07 → 2023-02) PPO lost 41% annualised and no learner made money.
- **Costs.** PPO and TD3 turn negative at 2× cost. All three are negative or near zero at 25 bp with a one-bar delay.
- **Benchmark.** On BTC, no learner made money over the pooled folds; buy-and-hold made 27.5%.

## What the comparisons say

- **RL versus the bandit and behaviour cloning.** IQL beats behaviour cloning by 28% a year (interval [−2.7%, +58.4%]) and the myopic bandit by 73% (interval excludes zero). The bandit ignores what it already holds and turns over 148× a year, so sequential credit clearly helps against a myopic rule. It does not help enough to beat holding the assets.
- **Input ablation.** PPO without funding and BTC context did better than PPO with them (27.6% versus 6.5%). The added inputs did not help PPO here, so no gain can be credited to them.
- **Reward shaping.** The variance penalty cut drawdown to 9.8% by holding about 1.6% average exposure. It is a risk-reduction result at a fifth of the exposure, not added return.
- **Simpler is equivalent or better.** Buy-and-hold has the highest return, and a ridge regression is level with TD3.

## Champion comparison: a correction made after the run

The registered comparator `champion_method` replays the champion rule's end-of-bar positions on the common ledger. It shows −45.6% a year, and that figure is **not** the champion's performance. The rule takes much of its profit from intrabar take-profit exits (3.5% target), which a ledger without intrabar fills cannot represent, so the replay drops those gains.

After seeing this I added `champion_native`: the CLI's own per-bar equity returns for the same rule and windows. That ledger fills at the decision close, fills stops and targets at their exact price, and charges no funding, so it flatters the champion. It gives +4.9% a year, Sharpe 0.25. Both rows are reported, and the gates now require beating both. This is a post-registration change to a comparator, made in the direction that is harder on RL. It does not alter any RL result, and the exported positions are byte-identical between the two exports.

Neither row is the six pinned production combos, whose parameters are not available locally. The registered unavailable-baseline gate therefore still blocks any superiority claim over production.

## Offline policy evaluation

- **Real logs:** none were available to this task, so there is no estimate for real logged behaviour.
- **FQE on simulated data, checked against simulator truth:** not credible. For IQL the mean error is −2.38 against a truth of 3.54, with rank correlation 0.18 across start states.
- **Importance sampling:** effective sample size is 12% of episodes at a horizon of one decision and 0.15% at 256. It is unusable beyond a few steps, and deterministic targets have no support at all.
- **Consequence:** OPE cannot support any claim here. Historical replay remains model-based counterfactual evidence under the simulator's assumptions.

## Simulator and reward audit

- An early pilot showed an out-of-sample Sharpe of 2.57 for PPO. Auditing it found no leak: features are bit-identical when later bars are altered, the window was a bull market in which buy-and-hold scored 2.01, and with bar order shuffled PPO fell below buy-and-hold. The registered run confirms that the pilot window was unrepresentative.
- Exploit probes pass: cash earns exactly zero, churning pays its costs, a missed fill creates no position, and the binary64 ledger matches an exact-rational replay to 1e-11.
- A pilot exposed a lockout in my first shield design (an all-time drawdown peak left an account reduce-only 61% of the time). The registered design uses a trailing 120-bar peak.

## Shield behaviour

- No policy produced an intent outside the admitted set. Admitted targets never exceeded the cap.
- Projection was rare for the learners (0.2%–0.5% of decisions, all of it the reduce-only gate). 14%–33% of decisions were abstentions under the minimum trade size.
- Peak exposure after fills reached 20.7%–22.3% against a 16.66% cap. That is price drift within a bar and cost deduction, a state effect the next snapshot reports, not an admitted action.
- With the shield disabled in simulation (diagnostic only), PPO's result was almost unchanged: 7.3% versus 6.5%. The shield bounded actions; it did not create or remove the economics.

## Formal assurance

See [`formal/research/exposure-rl/README.md`](../../formal/research/exposure-rl/README.md). In short: 20 SMT claims, one bounded model, Haskell/Python conformance on 124,176 cases, a forgery check and a source audit are machine-checked. Two of twelve obligations are discharged in scope (authorization, noninterference); ten are partially verified, each with its open items listed. Proofs establish action-admission invariants, not profitability or realised risk.

## Integration status

- **Research implementation:** complete.
- **Disabled-challenger integration:** not eligible, and not built. No Haskell inference path, artifact loader or UI change was added, because no candidate earned one. The Haskell shield is research-only and linked into no executable.
- **Live authorization:** outside this task. Nothing in it placed, changed or cancelled an order, or changed fleet membership, leverage, risk limits or trading flags.

## Limits of this evidence

- All data was previously exposed; nothing here is confirmation.
- Three years of 6h bars across six correlated symbols is one short history dominated by one bull phase.
- One fixed configuration per learner, small NumPy networks, CPU only. A stronger learner could do better; that would be a new registered trial.
- The universe is today's fleet, chosen with hindsight.
- A uniform 8 bp cost and next-open fills are assumptions, not measurements.

## If this is continued

Each item is a new registered trial, not a rerun.

1. Export matched per-bar decisions for the six pinned production combos so the real champion can be compared.
2. Ask a narrower question where a paired test has power, such as sizing or exit timing on top of the champion's entries.
3. Wait for data that has not been exposed, and keep the 2026-05 → 2026-07 window sealed until a candidate earns it.

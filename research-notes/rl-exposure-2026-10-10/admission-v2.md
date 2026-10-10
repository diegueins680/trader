# Admission-continuation successor (2026-10-10)

This adaptive development experiment follows the [v1 rejection](report.md).
The [registration](../registrations/rl-admission-continuation-v2.json) was committed
at `d2cf5957` before implementation tests or successor market outcomes. The
verified-main base remains `354c853176b7ee2ac6570951f7f6f2d0aed6909d`.
The original execution and evidence archive remain immutable.

## Decision and accounting contract

The problem remains sequential target exposure with inventory, costs and a
pending slot. Only target admission changes. The raw policy proposal remains the
PPO likelihood action and Q-learning action label. Observation support may first
project it to zero. At the next completed close, the exact immutable accounting
kernel previews the due target against marked equity and funding. A
`turnover_limit` rejection causes a second calculation with zero target **from
the original session**. The rejected preview is never committed, charged or
used as the next state. The accepted calculation is the only economic reward.

The new target is zero, not necessarily the filled position: 50% fill stress
retains half the previous units. Costs apply to actual turnover. Delay stress
retains one pending target, revalidated at its due close. Other economic
risk termination, liquidation, insolvency and arithmetic exhaustion retain v1
semantics and still fail the trial. No failure is silently reset in evaluation.
Terminal horizon liquidation remains separate from target execution.

A pre-fill admissible position has absolute exposure at most 0.35. A zero target
with fill fraction in (0,1] requires turnover at most 0.35, below the 0.50 cap.
With registered linear costs at most 50bp, this action also preserves the
0.35 endpoint exposure bound and positive equity relative to positive pre-fill
equity. It does not guarantee the capital floor, drawdown, gap losses or live
recovery. These are exact-real lemmas with bounded Fraction conformance, not a
theorem about every Python/Haskell execution or every market disturbance.

`actionChanged` reports support-induced proposal changes or a turnover-induced
due-target change. `unsupported`, `legacyReasonFlag`, `schedulingDeferred` and
`newProposalIgnored` remain separate. Unsupported cash proposals therefore do
not count as interventions. Delay counts do not pretend that the current
proposal was filled. Traces add units, prices, funding per unit, fill fraction
and cost multiplier so an independent auditor can reconstruct gross P&L,
funding, costs and nonterminal fills. Terminal paths may include both a trade
and risk liquidation; the auditor does not claim to reconstruct that sequence
from one aggregate row.

## Registered comparison

Same two exposed windows, BTC/ETH/SOL, five seeds, 512 transitions or updates,
three RL learners (PPO, simulator Double DQN, frozen-dataset discrete CQL),
uniform-behavior cloning, cash, constant-long, momentum and reversal controls.
CQL alpha-zero remains the registered diagnostic. All policies train and
evaluate through v2 admission. New simulated behavior buffers are generated
under v2; old transition rewards are incompatible. The seven cost, delay,
partial-fill and funding cases remain unchanged. No search or best-seed choice.

The matched champion, authentic historical first-seen timestamps, real logged
support and untouched confirmation remain unavailable. Sealed holdouts remain
protected. Differences from v1 cannot be credited to algorithmic alpha because
task termination changed. This continuation tests whether corrected admission
permits a trustworthy full-horizon comparison.

## Reproduction

Use Python3.13.3/NumPy2.3.5/Z3 4.15.4.0. From the isolated worktree, use the
already authorized development exports (hashes in the registration) and a new
external output directory:

```sh
python scripts/research/run_rl_admission_v2.py \
  --run-registered-development-v2 \
  --panel /Users/diegosaa/GitHub/trader/.tmp/research/historical-funding-campaign-v1/registered-development-panel.csv \
  --settlements /Users/diegosaa/GitHub/trader/.tmp/research/historical-funding-campaign-v1/registered-development-settlements.csv \
  --output /private/tmp/trader-rl-admission-v2-evidence
python test/rl_composition_test.py
bash scripts/verify.sh formal
bash scripts/verify.sh full
```

The campaign rejects uncommitted experiment sources and input-hash mismatches.
Execute with an empty inherited environment except PATH and single-thread BLAS
settings. No credentials or order-capable clients are needed. CPU inference
remains Python research inference, with no Haskell production loader, numerical
parity certificate or operational latency guarantee.

## Assurance scope

`F-RL-ADMISSION-V2` adds two nonvacuous SMT checks and a preserved synthetic
counterexample: v1 stops after three transitions; v2 completes 32 with 13
turnover projections, each compared to a fresh original-state ledger call.
Tests cover economic gaps, all seven stresses, state continuity, terminal
liquidation and corrupt primitive traces. An instrumented test now exercises
the actual DDQN training loop's online-selection/target-evaluation, terminal
mask and 100-update target synchronization. This is stronger than a stand-alone
formula test, but still not full update-by-update independent-library parity.
Existing PPO/CQL gradient and three-learner known-solution tests remain.

Canonical checks must reproduce the original scoped lifecycle/Haskell oracle
and new admission evidence. The 18 broader open obligations remain open.
This adds no production authorization, shared-budget admission or challenger
promotion path. The scheduled review-feedback autoloop still prevents safe
publication; retain local commits/patch unless that blocker is resolved.

## Execution status

Registered; implementation validation in progress. Append historical outcomes
with exact commit and archive hashes after the frozen campaign finishes.

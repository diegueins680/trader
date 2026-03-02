# Trader ROI Playbook (Codex + CI)

Purpose: increase engineering output per hour while keeping quality stable or better.

## 1) Target Outcomes

Use these as weekly targets:

- `first_pass_acceptance >= 70%` (changes merged without rework-heavy follow-up).
- `rework_rate <= 25%` (tasks requiring substantial second pass).
- `escaped_defects <= 1` (bugs found after merge).
- `median_cycle_time <= 1 day` for small/medium tasks.
- `quality_gate_pass_rate >= 90%` (`cabal build`, `cabal test`, web checks pass on first run).

## 2) High-ROI Task Lanes

Prioritize these lanes in this order:

1. Regression fixes with reproducible steps.
2. Tests for fragile code paths (trading logic, API parsing, UI fallbacks).
3. Small refactors that reduce repeated manual work.
4. Docs and runbook updates tied to changed behavior.

Defer low-ROI work:

- Large speculative rewrites with no measurable target.
- Broad style-only churn across many files.
- Multi-domain changes without a narrow acceptance contract.

## 3) Intake Contract (Before Starting Any Task)

Every task should define:

- `problem`: one sentence.
- `repro`: exact command(s) or UI steps.
- `constraints`: what must not change.
- `done_when`: objective checks.
- `risk_class`: `low|medium|high`.

If any field is missing, define it first. This prevents expensive back-and-forth.

## 4) Repo Command Packs (Copy/Paste)

### A) Fast quality gate (default)

```bash
cd haskell
cabal build
cabal test
cd web
npm ci
npm run typecheck
npm test
npm run build
```

### B) Haskell-only fix/test cycle

```bash
cd haskell
cabal build
cabal test --test-show-details=direct
```

### C) Local API + UI loop

```bash
./haskell/scripts/start_api_bg.sh
./haskell/scripts/start_ui_bg.sh
```

### D) Optimizer ROI runs (quality defaults)

```bash
COUNT=5 TRIALS=300 BARS=1000 QUALITY=1 \
  ./haskell/scripts/run_optimize_equity_top5.sh
```

Artifacts are written under `.tmp/opt-eq-top5-*/`.

## 5) Execution Guardrails (Cost Control)

Use these hard limits:

1. `5-turn rule`: if no concrete progress after 5 agent turns, tighten scope.
2. `15-minute rule`: if not reproducible in 15 minutes, stop and write a narrower repro.
3. `2-failed-runs rule`: after two failed full runs, switch to the smallest failing target.
4. `diff budget`: prefer <= 5 files for routine tasks; split bigger changes.
5. `verification budget`: always run at least one objective check before declaring done.

## 6) Definition of Done

A task is complete only if all apply:

- Repro no longer fails (or expected output now matches behavior).
- Relevant tests/checks pass.
- README + CHANGELOG updated for user-facing behavior changes.
- Risks and known follow-ups are recorded in the task note.

## 7) Weekly Scorecard Template

Create `tmp/roi-scorecard-YYYY-WW.md` each week.

Auto-generate a baseline scorecard:

```bash
./haskell/scripts/generate_roi_scorecard.sh
```

Generate a specific week:

```bash
./haskell/scripts/generate_roi_scorecard.sh --week 2026-09
```

```md
# ROI Scorecard - YYYY-WW

## Delivery
- Tasks completed:
- Median cycle time (hours):
- First-pass acceptance (%):
- Rework rate (%):

## Quality
- Escaped defects (count):
- Haskell gate first-pass rate (%):
- Web gate first-pass rate (%):

## Efficiency
- Mean agent turns per task:
- Mean human intervention points per task:
- Estimated hours saved:
  - Formula: (baseline_hours_per_task - actual_hours_per_task) * tasks_completed

## Top ROI lanes this week
- Lane:
- Why it paid off:

## Low ROI patterns this week
- Pattern:
- Change for next week:
```

## 8) Baseline + Improvement Cadence

1. Week 1: record baseline only (no target pressure).
2. Week 2-3: enforce guardrails and high-ROI lanes.
3. Week 4: compare against baseline and adjust thresholds.

Keep only the policies that improve both speed and quality metrics.

## 9) Ready-to-Use Task Prompt Template

Use this when starting Codex tasks:

```text
Task: <one-sentence problem>
Repro: <commands/steps>
Constraints: <what cannot change>
Done when: <objective checks>
Risk: <low|medium|high>
Run checks: <exact commands>
```

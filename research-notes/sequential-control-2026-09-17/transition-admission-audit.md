# Training transition admission audit — 2026-09-17

Baseline: merged main `9566caaa729e8cb03b690d7e9da87b86113b78d1`.
Scope: synthetic engineering fixtures in the existing offline collector. No
market data, protected holdout, new financial trial or trained artifact is used.

## Reproduced defect

With 121 constant prices of 100 and missing funding at index 25, collection
starts at index 24. The old collector returned reward `[0.0]`, terminal `[True]`
and no episode summary for a one-sample budget, despite no valid market
transition taking place. Invalid funding reached after part of a multi-bar
decision similarly produced an incomplete terminal training target. A nonterminal
missing successor was padded with twelve zeros. Insolvency with remaining
inventory could also enter the buffer without terminal accounting.

Four new regression methods brought the suite to 50 tests and initially
reproduced 27 failures. The failures included all three registered seeds
(11, 23, 47) for PPO, Double DQN and CQL: mocked update hooks detected attempts
to train from invalid funding samples. These are tiny synthetic contract tests,
not new financial experiments or an algorithm-performance comparison.

## Repair

After each simulated step and before appending its sample, collection checks:

- Finite reward, positive finite equity and forward progress within the decision
  interval and episode boundary.
- For a continuing episode, a complete decision interval, no failure and a real,
  unmasked finite 12-component successor.
- For a terminal, absent successor, zero inventory, no pending target and either
  the normal endpoint or an accounted capital-floor, drawdown, exposure or
  turnover stop.

An incomplete step raises an explicit error and the call returns no partial
batch. It is not skipped, resampled, imputed or converted to cash. A failing
first collection cannot reach the learner's update hook; a failure in a later
collection does not undo earlier valid updates, but propagates out of training
instead of returning a policy. The existing runner handles training exceptions.

Fully observed risk losses remain admissible training targets and retain their
failure status for promotion analysis. A deterministic funding-loss fixture
retains a -16.05 reward, including entry and terminal costs, with zero inventory
at the drawdown stop. A normal 96-step cash episode still pads only the terminal
successor; continuing observations retain their real state.

## Verification and parity

All 50 Python methods pass after repair, including later-invalid-step batch
abortion, partial-step failures, malformed successors, insolvency, valid risk
losses, normal terminal padding and all-seed learner update isolation.

The 108 scenarios in the [replay recipe](proposal-types-audit.md#reproduction)
remain byte-identical to baseline, SHA-256
`22276e1b07f503e97c2bb57315aac1ef3079ed1c25dcbb0a07f6cb7e10f3325a`.
The 18 seeded collection scenarios in the [collection recipe](episode-admission-audit.md#reproduction)
also retain their exact bytes, SHA-256
`81f811bae0802aba61cfba6674be0f90afb65ad828158cda281492a0ba9d1926`.
These are same-host engineering comparisons, not cross-platform guarantees or
new historical-market evidence. Generated comparison files stay outside Git.

Run from the repository root:

```bash
python3 test/sequential_screen_test.py
bash scripts/verify.sh automation
bash scripts/verify.sh full
```

The PR records the final source revision, actual full-wrapper result and hash,
and final-head GitHub CI. `A-SEQUENTIAL-RESEARCH-R8` links the four executable
witnesses. Haskell and Markdown risk mitigations remain synchronized;
`RL-OFFLINE-001` remains HIGH/OPEN.

## Limits and decision

Existing observation, action, reward, execution and artifact v1 semantics do not
change for valid samples. Historical artifacts and reported results are not
rewritten. This audit does not establish that every original training transition
was valid; it adds a guard to future collection. Training episode-summary timing
is unchanged, including summaries appended only when collection starts another
episode. This repair does not claim complete per-episode summary coverage.

There is no change to a production caller, exchange interface, deployment,
authorization flag, risk limit, champion or protected holdout. No new dependency,
configuration, promotion authority or live exploration is introduced. It does not
validate simulator realism, OPE support or statistical superiority. No candidate
passed; preserve the champion and continue offline research.

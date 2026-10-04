# Terminal learning-target numerical audit — 2026-09-28

Subsequent [target-v2 continuation](target-v2-followup-2026-09-28.md) introduces a
separate disabled kernel. The frozen learner and refutations below remain unchanged.

**No adoption. Two current-source numerical blockers remain unresolved.** This
continuation distinguishes terminal masking from accurate and finite learning
targets. It changes verification infrastructure and documentation only. The frozen
learner, policy artifacts, registrations, financial results, production champion
and live settings are unchanged. No market archive or protected holdout is opened.

Latest main is still `dbd45e26`; the isolated branch remains
`research/sequential-review-2026-09-28`, draft #284 stacked on #281. The
[contract](../../formal/research/terminal-numerics-contract.md) was committed at
`409b8a89` before implementation of verification artifacts.

## Paper-to-code scope

[Schulman et al., GAE](https://arxiv.org/html/1506.02438v6), rechecked 2026-09-28,
combines temporal-difference residuals with exponentially weighted advantage
estimation. Its mathematical estimator analysis and simulated control experiments
do not certify this repository's finite-precision implementation or financial
performance. [PPO](https://arxiv.org/abs/1707.06347) motivates the policy-update
family already screened here; it does not supply a numerical certificate for the
repository's custom target code. The focused [paper update](terminal-numerics-paper-update-2026-09-28.csv)
records the scope without adding a candidate or changing prior paper dispositions.

The existing code computes a masked residual, backward raw advantage and target
`adv + v`, then separately normalizes advantages over the whole batch. In exact
arithmetic, a terminal gives raw advantage `r-v` and target `r`. In binary64,
subtraction, addition and intermediate overflow must be considered separately.
The current terminal regression using critic value 100 is a useful finite fixture;
it does not establish this identity for every finite critic value.

## Current-source counterexamples

These are **not deliberate source mutations**. They reproduce in the unchanged
`advantages` function using a small untrained constant critic with finite weights
and outputs, float64 observations/rewards and Boolean terminal flags.

| ID | Finite inputs | Current target output | Finding |
| --- | --- | --- | --- |
| CE-RL-010 | Critic 1e16, terminal reward 1, discount .99 | 0 | `(r-v)+v` cancels the reward despite finite intermediates. |
| CE-RL-011 | Critic -2^1023, two terminal rewards [1,2^1023], discount .99 | [NaN,+infinity] | The later residual overflows; zero times its infinite carry contaminates the earlier terminal. |

[Hex-encoded fixtures](../../formal/research/terminal-counterexamples.json) preserve
exact binary64 inputs. Prescribed SMT SAT witnesses and actual NumPy regressions
reproduce both outcomes. The second case yields NaN normalized advantages too.
The current optimizer rejects gradients derived from those NaNs without advancing
its counter or changing parameters. That downstream rejection does not make the
upstream function finite or correct. The finite cancellation case is not detected
by a non-finite guard.

No evidence establishes that either fixture occurred in the historical runs. They
are numerical-domain counterexamples, not new failures counted as market trials,
proof of economic harm, or grounds to rewrite prior returns. Their existence
blocks claims of unrestricted numerical correctness and faithful terminal targets.

## Positive certificates and assumptions

`F-RL-TERMINAL-REAL` translates actual source arithmetic into exact reals and checks
terminal raw advantage `r-v` and target `r`. `F-RL-TERMINAL-FP-MASK` checks both actual
terminal bootstrap products under separate IEEE binary64 RNE operations, discount
in [0,1], finite next value **and finite later carry**. Products compare numerically
equal to zero; signed-zero bit identity is not claimed. Three SAT-premise and UNSAT
violation queries support these two scoped requirements.

`F-RL-TERMINAL-EXACT` and `F-RL-TERMINAL-FINITE` are explicitly **refuted**. Passing
counterexample reproduction in CI is negative evidence, not a passing claim of
exactness or finiteness. The scoped positive SMT count rises from 17 to **19**.
Existing lifecycle/artifact models and Haskell/replay conformance are unchanged.

`A-TERMINAL-NUMERICS` names correctly shaped float64 data, Boolean done masks,
stable bindings, no concurrent mutation, trusted AST recognition/scalarization and
NumPy/Python operation correspondence. The full function skeleton fixes loop order,
mask syntax, reconstruction and subsequent batch normalization. Unknown operations
and changed control flow fail certificate admission. This is not a verified
interpreter/compiler or full learner refinement.

Finite network output does not imply finite GAE carry, as CE-RL-011 demonstrates.
Normalized advantages intentionally pool the training batch: altering a later
terminal reward changes an earlier normalized advantage even when its raw target
is unchanged. No normalized-advantage noninterference claim is made. This allowed
within-training coupling is not new evidence of validation/holdout leakage.

## Regression evidence and disposition

Six new tests cover both current-source witnesses, source mutations removing either
mask, incorrect arithmetic/order, unknown syntax, vacuous premises, tampered
receipts, downstream optimizer rejection and batch-normalization scope. Twenty-seven
well-conditioned terminal cases cross critic values -8/0/8, rewards -2/0/2 and
discounts 0/.99/1; exact recovery holds on that finite grid. The grid is regression
evidence, not an extension of the real-number proof to all binary64 inputs.

No training-target repair is silently applied to frozen algorithms. A correction
must have an explicit implementation/version identity and a reviewed registration
before new financial trials; all adaptive successor trials must be counted. Current
PPO/Double-DQN/CQL results and rejected artifacts retain their recorded meaning.
The generic RL family is not rejected by these two implementation counterexamples.

The proof ledger and canonical clauses link the source, assumptions, counterexamples,
tests, pinned tools and CI. Numeric mission obligations gain explicit negative
evidence; **all 38 broader obligations remain open/partial**. `RL-OFFLINE-001`
remains HIGH/OPEN. No candidate becomes eligible for shadow, paper or live use.

## Verification receipt

Source freeze: `77b3d656`; report revision: `4599d248`; specification precedes both
at `409b8a89`. Subsequent receipt changes are documentation only. Commands from the
isolated worktree:

```bash
export PATH=/private/tmp/node-v20.19.0-darwin-x64/bin:$PATH
export TRADER_FORMAL_PYTHON=/private/tmp/trader-proof-20260928/bin/python
"$TRADER_FORMAL_PYTHON" scripts/formal/test_integrity.py TerminalNumericsTests
bash scripts/verify.sh formal
bash scripts/verify.sh full
"$TRADER_FORMAL_PYTHON" scripts/formal/verify.py --require-complete
```

Pinned tools checked locally: Python 3.13.3, NumPy 2.3.5, z3-solver 4.15.4.0
(solver 4.15.4), GHC 9.4.8, Cabal 3.12.1.0, fourmolu 0.15.0.0, hlint 3.8 and
Node 20.19.0.

- Targeted numerical suite: **exit 0**, six tests, 13.270 seconds.
- Formal wrapper: **exit 0**, 41 integrity tests (17.848 seconds), 19 scoped SMT
  requirements and two new expected current-source SAT counterexamples. The
  scoped verifier reported 14.210 seconds.
- Existing lifecycle model: two callers, 75 reachable states, 349 transitions,
  maximum shortest depth 6. Separate artifact model: 13 gates, 27 states,
  40 transitions, maximum depth 13. These are not a composed production proof.
- Conformance: 16,384 bounded Haskell cases plus 4,096 generated cases,
  180 exact-rational replay traces. New numerical cases are the two witnesses
  and the 27-case well-conditioned terminal grid, not financial experiments.
- Numerical domains: unbounded exact-real terminal algebra; all finite binary64
  next values/carries and finite discounts in [0,1] for the product certificate;
  separate RNE operations. SAT witnesses use the exact hex inputs in the fixture.
- Full wrapper: **exit 0**. Formal, Haskell build/format/lint/smoke/tests, web
  typecheck/241 tests/build and 185 automation tests passed, none skipped.
  The scoped verifier reported 10.778 seconds. No retry, disabled check,
  altered timeout or weakened gate was needed.
- Remote [CI run 36441350297](https://github.com/diegueins680/trader/actions/runs/36441350297)
  at `4599d248`: formal, Haskell, web and automation passed. Docker build and
  deployment were skipped.
- Acceptance diagnostic: expected **exit 1**, `ValueError: research acceptance
  blocked by open obligations`; all 38 broader obligations remain open/partial.
  Both new current-source numerical blockers remain unresolved.

Logs remain outside Git. SHA-256 receipts, prefix
`/private/tmp/trader-terminal-`, suffix `-20260928.log`:

| Log | SHA-256 |
| --- | --- |
| formal | `af184fc0250ecfc824666b66cebd08f920de5fd1a1e7393ee25fdd8b3dcad559` |
| acceptance | `34b0503d0c8cc33163cca6434683bb147d63fa27ab031fee43326273fa3d7e56` |
| full | `e9d930ebb39ee4fdc93a41e0d5b5197182249ac4388e5d02ab30035625c73821` |

No live authorization, order, authenticated trading experiment, live exploration,
holdout access, merge, deployment or champion change occurred. No proof placeholder
was introduced; unresolved proof and empirical obligations remain explicit. The
passing scoped checks reproduce refutations rather than declaring those claims true.

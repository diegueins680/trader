# Terminal learning-target numerical audit — 2026-09-28

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

The verification implementation is frozen at `77b3d656`, following specification
`409b8a89`. The targeted six-test suite passed, including expected counterexamples.
`bash scripts/verify.sh formal` returned **exit 0**: 41 integrity regressions,
19 scoped SMT requirements, existing state models and conformance checks. The full
command and final log hashes remain pending. No full-verification or mission-
completion claim is made yet.

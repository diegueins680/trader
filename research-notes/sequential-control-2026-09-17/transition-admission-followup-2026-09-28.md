# Training-transition admission continuation — 2026-09-28

**No adoption. The broader mission remains incomplete.** This continuation checks
the existing training-transition admission boundary without changing replay,
training algorithms, data, policies, artifacts, registrations, current champion or
live configuration. It uses synthetic engineering fixtures only. No financial
trial, protected holdout or independent out-of-sample period is accessed.

Latest remote main remains `dbd45e26`; work continues on the isolated branch
`research/sequential-review-2026-09-28` in draft #284, stacked on #281. Existing
[transition admission rules](transition-admission-audit.md) remain authoritative;
this continuation fills their proof/traceability gap, not a simulator behavior gap.

## Specification, source and SMT

The [contract](../../formal/research/transition-admission-contract.md) was committed
at `bcc816e0` before implementation. `F-RL-TRANSITION-ADMISSION` derives the base,
terminal and nonterminal validity expressions from the actual helper source.
An audited source skeleton binds its branch and raising behavior. The entire
collector source is recognized to preserve step → admission → append ordering;
unknown source/control-flow changes require review rather than silently passing.

Three SAT-premise/UNSAT-violation checks establish necessary admission conditions:

- Reward is finite, equity is finite and positive, the failure tag is permitted,
  and time advances within the supplied horizon and episode endpoint.
- An admitted terminal has absent successor, zero units, no pending action, and
  either a recognized risk-failure tag or final-bar endpoint.
- An admitted nonterminal has native false done flag, no failure, exactly one
  supplied horizon of progress before the endpoint, and a successor passing the
  existing real-vector representation, shape and finiteness predicates.

Indices are unbounded mathematical integers. Reward, equity and units use IEEE
binary64 predicates, including NaN, infinities and signed zero. The theorem does
not silently apply real-number equality to floating-point execution. Non-binary64
numeric types are outside this SMT domain. Array validity and absence predicates
are abstraction atoms under trusted primitive semantics. The scoped SMT count is
now **17**; the existing 75-state lifecycle and 27-state artifact models are unchanged.

## Counterexample and actual implementation tests

[CE-RL-009](../../formal/research/transition-counterexamples.json) deliberately
removes the zero-inventory terminal guard. At left=30, t=33, stop=34, horizon=3,
equity=1 and units=0.25, the original helper rejects the terminal and the compiled
mutant admits it. SMT rejects the mutated source. This is a regression mutant,
not an existing loader/replay defect or evidence of financial performance.

Six new regression tests cover source drift, missing reward/pending/finite-vector
checks, partial-horizon admission, vacuous premises and append-order bypasses.
Thirty invalid helper fixtures include non-finite scalars, invalid done types,
wrong indices, incomplete failures, missing/malformed successors, outstanding
inventory and pending actions. Valid terminal signed zeros and all four recognized
risk-failure categories are also exercised.

Thirty-six actual replay cases cross horizons 1/3/6, targets -0.25/0/0.25, costs
1x/2x and full/half fills on a small flat-price fixture. Every transition passes
the actual admission helper, and each episode terminates with zero units and no
pending action. An injected unliquidated terminal makes the real collector raise
before it can publish training output. These are deterministic regression tests,
not proofs of all traces or market realism. No trained artifact is retained.

## Assumptions and remaining blockers

`A-TRANSITION-ADMISSION` names ordinary stable Replay fields, integer indices,
binary64 scalars, primitive identity/comparison semantics, ordinary arrays and
stable helper bindings; no concurrent writes, hostile subclasses or monkeypatches.
Python exception/control flow, `_finite_real`, vector representation/shape/finite
checks, the AST parser and restricted translator are trusted. The abstraction maps
successful actual helper calls to satisfying assignments of the extracted predicate;
a machine-checked interpreter/compiler refinement is not provided.

These necessary admission conditions do not establish complete liquidation fees,
fill prices, funding, cash accounting, truthful failure classification, future
market safety, Bellman/GAE target correctness, finite optimizer updates or
preemptive timeouts. An earlier recognized risk failure remains failed economic
evidence; its accounted training terminal is not a successful evaluation path.

Canonical clauses, proof ledger, source locks, tests, CI and risk documentation
map this requirement bidirectionally. Mission obligations 6/18/19 gain partial
evidence, but **all 38 broader obligations remain open/partial**. `RL-OFFLINE-001`
remains HIGH/OPEN. The tested RL policies remain rejected; contaminated development
results, invalid OPE, missing matched champion comparison and sealed holdouts are
unchanged. Neither production authorization nor candidate promotion follows.

## Verification receipt

Source freeze: `2fcd591d`; report revision: `62a45b95`. Later receipt changes
are documentation only. Commands from the isolated worktree:

```bash
export PATH=/private/tmp/node-v20.19.0-darwin-x64/bin:$PATH
export TRADER_FORMAL_PYTHON=/private/tmp/trader-proof-20260928/bin/python
"$TRADER_FORMAL_PYTHON" scripts/formal/test_integrity.py TransitionAdmissionTests
bash scripts/verify.sh formal
bash scripts/verify.sh full
"$TRADER_FORMAL_PYTHON" scripts/formal/verify.py --require-complete
```

Pinned tools: Python 3.13.3, NumPy 2.3.5, z3-solver 4.15.4.0 (solver 4.15.4),
GHC 9.4.8, Cabal 3.12.1.0, fourmolu 0.15.0.0, hlint 3.8 and Node 20.19.0.

- Targeted transition suite: **exit 0**, six tests, 1.822 seconds.
- Formal wrapper: **exit 0**, 35 tests, 17 scoped SMT requirements, both existing
  state models, 20,480 Haskell conformance cases and 180 rational accounting
  traces. Standalone verifier: 5.429 seconds. No universal refinement claim.
- Full wrapper: **exit 0**. Formal, Haskell build/format/lint/smoke/tests, web
  typecheck/241 tests/build and 185 automation tests passed, none skipped.
  Automation took 57.034 seconds; the scoped verifier reported 8.099 seconds.
  No retry, disabled check, altered timeout or weakened gate was needed.
- Remote [CI run 36437313290](https://github.com/diegueins680/trader/actions/runs/36437313290)
  at `62a45b95`: formal, Haskell, web and automation passed. Docker build and
  deployment were skipped.
- Acceptance diagnostic: expected **exit 1**, `ValueError: research acceptance
  blocked by open obligations`; all 38 broader obligations remain open/partial.

Logs remain outside Git. SHA-256 receipts, prefix
`/private/tmp/trader-transition-`, suffix `-20260928.log`:

| Log | SHA-256 |
| --- | --- |
| formal | `3d69055d075dce1050a97c6ed32e09fee190cf5740db7c254496c553dbcd688d` |
| acceptance | `b64ace3bda36b2d7b3c17ccdfe7a0995aaac133bfd529318d80762fa7382ece4` |
| full | `a1196e5143512a1c6d35f2b0b410abf47013e72ca066d20ff6474a5e6bb476e4` |

No live authorization, order, authenticated trading experiment, live exploration,
holdout access, merge, deployment or champion change occurred. No proof placeholder
was introduced; the unresolved proof and empirical obligations remain explicit.

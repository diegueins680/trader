# Verification receipt — 2026-09-21 UTC

The scoped verification and repository checks passed. **Research acceptance is
not achieved.** No candidate is adopted, no protected holdout is opened, and the
38 broader implementation obligations remain open or partially verified.

## Tested revisions and commands

The final local full run tested documentation branch revision
`b6cc4e562d90a4958fee1dc3bff5deca76105df1`, containing tooling revision
`1817742d7def10b5b0d88a92a03a9f5403768a67`. This receipt is a subsequent
documentation-only addition; it does not claim a full run on its own future hash.

From the repository root, after installing the pinned dependencies:

```sh
TRADER_FORMAL_PYTHON=/private/tmp/trader-formal-tools/bin/python bash scripts/verify.sh formal
TRADER_FORMAL_PYTHON=/private/tmp/trader-formal-tools/bin/python bash scripts/verify.sh full
```

Both returned exit 0. The standalone formal run tested tooling revision
`1817742d`; the full run repeated formal verification on `b6cc4e56` and completed:

- Formal registry: 40 specifications, 335 clauses, 346 implementation files,
  223 evidence links and 33 canonical risks.
- Six proof-ledger/model integrity regression tests passed.
- Nine negated SMT obligations were UNSAT. The two-caller abstract protocol
  reached 75 states and 349 transitions, maximum shortest-path depth 6;
  conditional drain bound is two completion events.
- Compiled Haskell conformance passed 16,384 boundary combinations and 4,096
  seeded generated cases. All 240 admitted outputs preserved target bits and
  denied order authority; private-constructor rejection and default disable passed.
- Haskell build, formatting, lint, smoke and the Cabal test suite passed.
  Cabal reports one aggregate test case; this is not a count of internal assertions.
- Web typecheck, 241 tests and production build passed.
- Deployment configuration validation, registry validation and 185 automation
  tests passed. Validation did not deploy anything.

The formal section took 20.679 seconds in this local full run. This is an observed
runtime, not a production inference measurement or a portable performance bound.
Local GHC was 9.4.8, Cabal 3.12.1.0, Python 3.13.3 and solver 4.15.4
(distribution 4.15.4.0). Local Node was 24.8.0; CI used the pinned 20.19.0.

External local logs, excluded from Git:

| Log | SHA-256 |
|---|---|
| `/private/tmp/trader-formal-full-retry.log` | `92afaaabab39cfc9b5e2642f45f9bd664ce6eaf845da2c7530b458c2ca597815` |
| `/private/tmp/trader-formal-final-gate.log` | `889fc675def2ed27f32f084b2c3752a67d242fd10d105b41874be38b6c1b30c8` |

## Independent CI and resolved failures

All four jobs (`formal`, `haskell`, `web`, `automation`) passed in both:

- [Tooling revision 1817742d, run 35562489766](https://github.com/diegueins680/trader/actions/runs/35562489766).
- [Documentation revision b6cc4e56, run 35562516294](https://github.com/diegueins680/trader/actions/runs/35562516294).

Docker build and Fly deployment jobs were skipped. Linux CI independently checked
the pinned formal results. These links identify tested revisions, not a promise
that a later documentation commit has already finished CI.

The initial local full/automation runs failed three localhost fixtures with
`listen EPERM` under the sandbox. Authorized outside-sandbox reruns passed all
185 tests and the final full wrapper. Dependency installation initially failed
with sandbox DNS/registry access errors; authorized retries succeeded. Neither
failure is reported as an initial pass.

Initial GitHub run 35561824198 failed before jobs because YAML interpreted the
colon in a plain `run` scalar containing `--only-binary=:all:`. Revision 1817742d
uses a block scalar; YAML parsing and both subsequent CI runs passed. No assertion
or verification step was disabled to resolve these failures.

## Acceptance and evidence limitations

The separate `scripts/formal/verify.py --require-complete` diagnostic returned
exit 1 with `ValueError: research acceptance blocked by open obligations`, as
required by the ledger. This is a real acceptance blocker, not a waived test.
The scoped gate reports `missionComplete: false` and 38 unresolved obligations.

SMT claims concern the specified numeric and algebraic models; the lifecycle is
an abstract bounded-caller protocol. Conformance tests do not establish universal
source refinement, real server concurrency, market causality or future returns.
No neural-policy region or probabilistic market-risk model is verified.

[Archive revalidation](formal-recheck-receipt-2026-09-20.json) independently
reproduced all seven compact reports byte-for-byte without retraining or replaying
financial paths. The [decision memo](formal-followup-2026-09-20.md) records the
negative inherited experiments, unavailable comparisons and untouched holdouts.
The September 20 literature supplement remains dated to its source checks. A
limited September 21 recency search returned no additional primary source admitted
to the review; irrelevant secondary results were excluded. This is not exhaustive
literature coverage through the execution date.

Both PRs remain drafts. Live flags, fleet, champion, risk limits, production
ownership and deployment identity are unchanged. No order, authenticated exchange
experiment, live exploration, policy promotion, merge or deployment occurred.

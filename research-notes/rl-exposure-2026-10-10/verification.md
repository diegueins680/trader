# Verification and publication record

Execution date: 2026-10-10. Worktree `/private/tmp/trader-rl-20261010`, branch
`research/rl-exposure-v1-20261010`, based on verified main
`354c853176b7ee2ac6570951f7f6f2d0aed6909d`. Existing tracked modifications in the
original checkout were preserved. Research used an empty inherited environment
(`env -i`) with only PATH and CPU-thread settings; no trading credentials were read.

## Checks

- `python test/rl_composition_test.py`: **PASS**, 13 tests in 3.614s.
- `python scripts/formal/rl_composition_check.py`: **PASS**; reviewed source
  identities, seven nonvacuous SMT lemmas, 43-state/107-edge finite model and eight
  selected Python/Haskell differential transitions. Receipt in
  [composition-check.json](results/composition-check.json).
- Registered experiment: **PASS as an execution/inventory check**, 50/50 fits,
  1,218/1,218 replay attempts recorded. **314 failed economic paths**, all
  `turnover_limit`; no integration pass is implied.
- Independent archive audit: 1,218 paths and 63,309 rows checked after complete
  index/hash verification; zero violations in the declared trace checks.
- `bash scripts/verify.sh haskell`: **PASS**; build, fourmolu, hlint, smoke and
  Cabal test suite. GHC 9.4.8, Cabal 3.12.1.0, fourmolu 0.15.0.0, hlint 3.8.
- `bash scripts/verify.sh web`: **PASS**; typecheck, 241 tests and build, first
  targeted run using Node 20.19.6. Exact Node 20.19.0 was subsequently installed
  in the temporary tool directory for final canonical verification.

- `python scripts/formal/verify.py --record`, attempt 7: **PASS**, all scoped
  certificates reproduced in 282.574s; updated `formal/research/results.json`.
  The receipt explicitly reports 20 scoped closed and 18 unresolved mission
  obligations, with `economic_evidence` and `reproducible_delivery` blocked.

- `bash scripts/verify.sh full`: **PASS**, exit 0, on the final code. Node
  20.19.0 and `TRADER_FORMAL_PYTHON=/private/tmp/trader-rl-tools/bin/python`;
  execution outside the filesystem sandbox allowed localhost test listeners.
  This included all 341 formal regression tests, all 13 new RL tests, independent
  reproduction of the reviewed 90-entry SMT/certificate receipt, Haskell
  build/format/lint/smoke/tests, web typecheck/241 tests/build, deployment-config
  validation, and all 185 automation tests. No tests were skipped or cancelled
  in the web or automation summaries. Intentional negative parser fixtures emit
  rejection messages; the formal regression suite completed with `OK`.

The final full run supersedes the earlier environmental/timing failures as a
verification result, without erasing those attempts. No timeout, test, solver
requirement or production guard was relaxed. A passing scoped verification command
does not override the receipt's explicit `missionComplete: false`.

Verification logs are retained separately from the immutable experiment archive at
`/Users/diegosaa/GitHub/trader/.tmp/research/rl-causal-composition-delivery-20261010`.
The [verification receipt](results/verification-receipt.json) records their hashes.

## Preserved failed attempts

- Early synthetic learner tests exposed the distinction between the mutable
  optimizer interface and immutable inference snapshots. Checked snapshot decoding
  was implemented and tested before the frozen market experiment. The initial
  synthetic replay also exposed repeated burn-in fitting; the pre-market
  construction-cache amendment is recorded in `contract.md`.
- Initial formal recording refused changed source identities until the new
  implementation and registry changes were reviewed and explicitly pinned.
- `python scripts/formal/verify.py --record`, attempt 3: failed the unchanged
  Haskell PPO process-bridge positive inference assertion while compilation and
  research ran concurrently. Its 20ms admission deadline was not relaxed.
- Attempt 4: failed an unchanged accounting-reconciliation SMT obligation through
  `ppo_successor.certify` (`SAT/UNKNOWN/vacuous proof`). The old error does not
  distinguish those reasons. A separate diagnostic using the same formulas,
  random seed and 10s timeout returned SAT premises and UNSAT counterexamples;
  the gross query took 9.686s, close to that timeout. No failed attempt is counted
  as a proof, and no proof timeout was increased.
- Attempt 5, after the concurrent build finished: passed the earlier failure
  points, then correctly rejected the new learner as an unregistered kernel
  consumer. The audit now admits only six exact kernel imports in the learner
  and runner with reviewed complete source hashes; mutation tests reject wildcard imports, unregistered
  files and changed consumer bytes. The Replay receiver roster was not widened.
- Attempt 6 was terminated after a targeted test exposed an incomplete importer
  registration, before any certificate was published. The six actual imports
  were then enumerated and tested together; no broad allowlist bypass was added.
  A subsequent mutation test expected the import error before the source-hash
  error; the assertion now accepts either rejection while still requiring failure.
- First automation run: 180/185 tests passed. Three loopback-server fixtures
  failed with sandbox `listen EPERM 127.0.0.1`; two scheduler/campaign fixtures
  exceeded their 30s/60s bounds during concurrent heavy compilation. A retry with
  `NODE_OPTIONS=--test-concurrency=1` was rejected by Node before tests; that
  unsupported invocation is not a verification result.

The source/economic implementation was not tuned in response to market outcomes.
Post-run changes remove a constant safety-reporting field, clarify buffer
provenance, add trace-audit regressions and publish evidence. The original archive
and experiment SHA remain unchanged. No failure, seed or trial was erased.

## Why no remote publication

Direct branch CI itself is gated safely: `.github/workflows/ci.yml` only pushes
deployments from main/master, and `deploy-hetzner.yml` rejects other head branches.
However, the scheduled `.github/workflows/autoloop.yml` uses the repository's
default branch and a dedicated push token. In `scripts/autoloop.mjs`:

- `AI_REVIEW_POLL_ENABLED` defaults true (line 67).
- `listRecentPullRequestsForReview` requests `state=all` and does not exclude draft
  or unmerged PRs (line 824).
- AI review comments on allowed paths, including README, CHANGELOG and tests,
  can select a repair on the default branch (lines 260–274).
- `LOOP_BRANCH = BASE_BRANCH` (line 37), and `pushBranch` pushes directly to it
  (line 1758), which can trigger deployment CI.

Thus a draft PR is not a verified isolation boundary against later automated
review feedback. This task does not modify that production-connected automation.
Local commits and a prepared draft-PR description are retained; no branch push,
PR, merge, deployment or schedule change is performed. Publication can resume
only after an independently established isolation path. This is a concrete
workflow blocker, not a request to authorize live execution.

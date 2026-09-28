# Verification receipt — 2026-09-28

This receipt distinguishes scoped checking from whole-mission acceptance. The
implementation revision is `95f9b48d`, on latest main `dbd45e26` plus the unchanged
PR #281 dependency `1817742d`. The final full run tested `44177344`; the subsequent receipt-only commit changes no executable source.

## Completed scoped checks

`bash scripts/verify.sh formal` returned **exit 0** using
`TRADER_FORMAL_PYTHON=/private/tmp/trader-proof-20260928/bin/python` and Node
20.19.0 on PATH. Python 3.13.3, NumPy 2.3.5, Z3 4.15.4 and GHC 9.4.8 were checked.
Ten integrity/regression tests passed, followed by 12 UNSAT obligations, two SAT
refutations, 75-state/349-transition lifecycle exploration, 20,480 compiled
Haskell conformance cases and 180 two-bar replay cases. The standalone scoped verifier took 7.418 seconds; the same verifier inside the
full run took 12.425 seconds. Timing is diagnostic, not a performance guarantee.

`scripts/formal/verify.py --require-complete` returned the expected **exit 1**:
`ValueError: research acceptance blocked by open obligations`. All 38 mandatory
whole-system obligations remain open or partial. No candidate is eligible.

The frozen external index and all seven compact reports were verified and
reproduced byte-for-byte; see [evidence receipt](gap-risk-evidence-receipt-2026-09-28.json).
No financial training, new market trial or holdout access occurred.

## Environment and full verification

The first offline npm install failed with `ENOTCACHED` for `csstype`. The authorized
locked online install succeeded, reporting zero npm audit vulnerabilities. Z3's
first sandbox install failed DNS resolution; the authorized hash-locked install
succeeded. The exact Node 20.19.0 archive was checked against official SHA256
`a8554af97d6491fdbdabe63d3a1cfb9571228d25a3ad9aed2df856facb131b20`.

The initial targeted `bash scripts/verify.sh automation` returned **exit 1**:
182/185 passed; three local HTTP fixtures failed with
`listen EPERM: operation not permitted 127.0.0.1`. The authorized outside-sandbox
rerun returned **exit 0**, 185/185 passed, none skipped, 50.846 seconds. No fixture
was disabled or modified.

The first `bash scripts/verify.sh full` returned **exit 1**, with the same three
sandbox socket failures. Haskell build/format/lint/smoke/tests and web checks passed.
That run used Node 20.19.6 and began before the final proof-source freeze; it is not
the final verification evidence.

The final authorized outside-sandbox run at `44177344` returned **exit 0**:

```sh
PATH=/private/tmp/node-v20.19.0-darwin-x64/bin:$PATH \
TRADER_FORMAL_PYTHON=/private/tmp/trader-proof-20260928/bin/python \
bash scripts/verify.sh full
```

The standalone formal invocation uses the same environment with
`bash scripts/verify.sh formal`. The final full run passed the formal registry,
ten integrity tests, all scoped proof/conformance checks, Haskell build,
Fourmolu/HLint/smoke/test suite, web typecheck/241 tests/build and automation
185 tests (47.622 seconds). No tests were skipped. Pinned Cabal is 3.12.1.0,
Fourmolu 0.15.0.0 and HLint 3.8; other versions are listed above.

GitHub Actions [run 36376101549](https://github.com/diegueins680/trader/actions/runs/36376101549)
also passed formal, Haskell, web and automation at `44177344`. Docker image build
and Fly deployment were skipped. This records that tested revision; it does not
claim a later documentation-only revision has already passed remote CI.
The branch is published in [draft PR #284](https://github.com/diegueins680/trader/pull/284),
stacked on #281; neither is merged by this task.

Local logs are retained outside Git. Their hashes identify the inspected bytes;
reproduction requires rerunning commands, since timing/path text is host-dependent.

| Log under `/private/tmp/` | Exit | SHA256 |
|---|---:|---|
| `trader-formal-final-20260928.log` | 0 | `748b6e734149a13ffafe66894ffc34e462a5716ac51869ded83bdae97204ea73` |
| `trader-full-final-20260928.log` | 0 | `aef8c9f43e8233bb1dc0edc23485f8f94b64e3df4bfe6ac497077bb4e88b8b19` |
| `trader-automation-final-20260928.log` | 0 | `7c4018cccf504eab571572afce60ebd38f3351ce06101a7a9a608abd2f0b7fa1` |
| `trader-full-20260928.log` | 1 | `f88a980acbacf2646c173d2f3c77c197a2ff1348c89937d1e9eec64fc9125a36` |
| `trader-acceptance-20260928.log` | 1, expected | `16cddd4fc62bde4c5306b628d2f3a29aa0cbead60cbe69b422f387089ae8ba66` |

## Acceptance boundary

The draft is not declared ready for candidate integration. No universal compiler
refinement, production concurrency proof, probabilistic market-risk certificate,
neural-region certificate or future profit/loss guarantee is claimed. All sources
and scoped proof statuses are linked in the machine-readable ledger. Proof source
placeholder checks run in the formal gate; open obligations remain explicit.

# Verification receipt — 2026-09-28

This receipt distinguishes scoped checking from whole-mission acceptance. The
implementation revision is `95f9b48d`, on latest main `dbd45e26` plus the unchanged
PR #281 dependency `1817742d`. Documentation may have a later commit.

## Completed scoped checks

`bash scripts/verify.sh formal` returned **exit 0** using
`TRADER_FORMAL_PYTHON=/private/tmp/trader-proof-20260928/bin/python` and Node
20.19.0 on PATH. Python 3.13.3, NumPy 2.3.5, Z3 4.15.4 and GHC 9.4.8 were checked.
Ten integrity/regression tests passed, followed by 12 UNSAT obligations, two SAT
refutations, 75-state/349-transition lifecycle exploration, 20,480 compiled
Haskell conformance cases and 180 two-bar replay cases. The last scoped verifier
runtime was 6.111 seconds before the final explicit NumPy receipt check; timing
is diagnostic, not a performance guarantee.

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
`listen EPERM: operation not permitted 127.0.0.1`. An authorized outside-sandbox
rerun is in progress. No fixture was disabled or modified.

The first `bash scripts/verify.sh full` invocation is still running; no full pass
is claimed in this interim receipt. Its Haskell compilation and format checks
passed; lint/tests/web/automation completion is not yet asserted. It started with
local Node 20.19.6 and before the final proof-source freeze. A final pinned-tool
run and completed log hashes will be recorded before final delivery.

## Acceptance boundary

The draft is not declared ready for candidate integration. No universal compiler
refinement, production concurrency proof, probabilistic market-risk certificate,
neural-region certificate or future profit/loss guarantee is claimed. All sources
and scoped proof statuses are linked in the machine-readable ledger. Proof source
placeholder checks run in the formal gate; open obligations remain explicit.

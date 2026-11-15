# Data composition and obligation closure — 2026-10-04

Two existing closure criteria are now connected to the actual offline runner:
causal normalization (3) and symbol isolation (5). No algorithm, archived result,
production setting or dataset has changed. No candidate is promoted.

The prior lemmas assumed that the caller supplied the right arrays and preserved
an immutable transform. The added checker follows the actual registered loader,
training-prefix fit, every scale reference, paired collector/replay/OPE input and
snapshot publication. It checks 28 finite source-use sites. Six key identities
are SMT-verified for arbitrary symbol strings, with satisfiable premises. Existing
unbounded integer prefix proofs supply the feature-read bounds. The source
contracts explicitly identify pooled training transforms and learned parameters
as shared inputs; they are not a claim of cross-symbol statistical independence.

The abstraction and library assumptions are in
[the contract](../../formal/research/data-composition-contract.md).
The full-source locks prevent helper substitution; source hashes alone are not
presented as semantic proofs. Checked source grammar, Python, pandas, NumPy,
cryptographic byte identity and ordinary immutable-input semantics remain trusted.
There is no universal compiler/interpreter or deployed-image refinement claim.

Deterministic conformance executes the real CSV loader, extracted actual fold
setup, scale fitting, collection, historical replay and short OPE constructors on
two distinct synthetic symbols. It verifies exact pairing, training-only fit
under future corruption, immutable fields, foreign-symbol perturbation and
hash-rejection of replaced files. Eight deliberate leakage/escape/mutability
mutants must be rejected before the final source-hash check; an additional
foreign-symbol mutation produces a satisfiable SMT violation. These are injected
regression mutants, not newly discovered historical market failures.

Verification setup initially exposed a missing pandas dependency in the minimal
formal environment. The loader conformance test was placed in the existing
research automation suite, which already installs pinned pandas 2.3.3; no new
formal or production dependency was added. An initial proof-record attempt
correctly rejected an unrefreshed audit-document source hash. Neither failure
was bypassed. The placeholder scanner also flagged an ordinary English word in a comment; the comment was rephrased without changing the scanner or any proof. Successful exact-source proof reproduction took 100.916 seconds. Exact wrapper commands and final results are recorded in the PR.

Current status: five affected-scope closures (3, 5, 11, 24, 31), 26 partial and
seven open. The criteria were not weakened. In particular, no closure is claimed
for revision timing: the historical records lack original availability witnesses.
Validation feature warmup may overlap training history, and the historical
purge/embargo labels do not establish two additive gaps. Full split isolation,
complete state causality, numeric repairs composed into a successor, server
ownership/readiness/shutdown, persistence recovery and promotion isolation remain
work. The protected holdout stays sealed; prior 108 fits/19,440 replays remain
contaminated development with invalid OPE and no matched-champion confirmation.

Recommendation remains no adoption. This verification change is independently
useful because it replaces two assumption-only composition gaps with replayable,
source-bound evidence and rejects future dataflow drift in CI.

Review counterexample CE-DATA-COMPOSITION-001: the first checker revision at
12bec119 accepted a registry with its reviewed skeleton roster deleted. The
canonical roster itself was complete. The repaired checker now requires the
entire eleven-block/four-file roster and rejects partial omissions. The fixture
preserves the observed pre-fix acceptance, and deterministic regressions cover
missing schema/shared-input declarations as well. An additional SMT regression
rejects paired keys that agree with each other but name a foreign symbol.
All final checks are rerun against the hardened checker; earlier successful
checks do not substitute for verification of the final head.

The superseded pre-hardening full verification was intentionally stopped during
its own confirmed hlint process after its formal phase passed. It is not reported
as a completed full check. No unrelated process was stopped.

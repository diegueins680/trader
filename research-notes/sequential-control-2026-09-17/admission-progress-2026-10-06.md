# Backtest FIFO progress — 2026-10-06

The inherited waiting backtest loop could starve while other jobs repeatedly acquired
its vacancies. Thread scheduling fairness did not imply fairness of the once-per-second
admission attempts. The repaired gate registers a private FIFO ticket once, prevents
immediate callers from barging, and atomically trades the head ticket for one slot.
Cancellation removes its own ticket; drain wakes waiters and admits no new owner.
The callback timeout still starts only after acquisition. No live limit or policy
behavior is changed. This is a repair to the existing gate, not a learned challenger.

Preregistration: `b8852d55`, [contract](../../formal/research/admission-progress-contract.md),
[registration](../registrations/admission-progress-engineering.json); base main
`fb00a66707d2e645e15fb347833a661e044a7a77`. All38 original claims, scopes and closure
criteria remain unchanged. Obligation38 may move open to partial, not closed.
Current counts12 scoped closures /25 partial /1 open;26 remain unresolved.

## Evidence and limits

- SMT: eight SAT-premise/UNSAT-violation queries over exact Integer tickets and
  predecessor-rank summaries plus arbitrary bounded Int counts. Seq and STM
  operations are explicit primitive assumptions, not library theorems.
- Model: three callers, capacities1/2,838 reachable states,3356 transitions,
  maximum shortest depth11; retry, stutter, cancellation, owner release and drain.
  A weak-fair SCC check excludes an infinite unresolved target waiter. Cancellation
  and draining resolve a waiter without implying successful execution. In continued
  open service without target cancellation, termination assumptions imply admission.
- Existing ownership/timeout and drain models now overapproximate queue-erased
  contention explicitly: wait/reject can occur even with free capacity. Their EF
  termination checks are not promoted to fair AF claims.
- Preserved CE-BACKTEST-STARVATION contains a feasible weak-fair legacy lasso.
  The actual old/new Haskell fixture observes `barged=True`/`barged=False`.
  The old finite witness must run its contender within the original one-second
  sleep; scheduler delay can fail that fixture. It is not an infinite runtime trace.
- Five new compiled tests cover FIFO, no barging, head/middle cancellation, drain,
  separate gates and32 fixed-seed failure/success schedules. The existing ownership,
  callback-cancellation and timeout suites also remain required. Tests supplement
  the source-mapped model; no compiler/OS refinement or universal scheduler theorem.
- SCC implementation is compared against independent transitive closure for all512
  directed three-vertex graphs. Mutation tests restore barging, leak a cancelled
  ticket, and distinguish intermittent from continuously enabled weak fairness.

Named A-ADMISSION-PROGRESS requires ordinary pinned GHC/base/STM/containers behavior,
private cells, finite memory, eventual callback/cleanup completion and weak service
of continuously enabled transitions. The model does not assume fairness for a
momentarily free slot. FIFO registration order is commit order, not HTTP arrival
order. No queue-length or wall-clock SLA, denial-of-service resistance, fair success
of immediate callers, durable restart or cross-instance fairness is established.
Cancellation filtering is linear in queue length; memory is linear in registered
waiters. Additional queue bounds require a separately specified overload policy.

The primary runtime references are [STM2.5.1.0](https://hackage-content.haskell.org/package/stm-2.5.1.0/docs/Control-Concurrent-STM.html)
and [base4.17.2.1 exceptions](https://hackage-content.haskell.org/package/base-4.17.2.1/docs/Control-Exception.html),
read2026-10-06. Applying their primitives to this protocol is our specification
and verification work, not a library assertion of whole-server fairness.

## Reproduction and delivery

Use pinned tools from `.tool-versions` and `formal/research/toolchain.json`.
Run `bash scripts/verify.sh formal`, then `bash scripts/verify.sh full`.
Source bindings, proof ledger, traceability, risk register, mutation fixtures,
compiled conformance and receipt equality are CI gates. No new dependency.

Initial20 targeted tests passed in17.272s after refreshing the reviewed source lock;
the preceding run correctly failed on a stale test-file digest. The expanded23-test targeted suite passed in30.825s. The first full-verifier attempt
rejected an action label matching the forbidden proof-placeholder word; the label
was renamed to `acquire` without changing the placeholder gate. No incomplete
theorem was present. Canonical verification and final-head results will be recorded
after execution, not assumed passed.

No financial experiment, market-data read, model-training campaign, holdout evaluation, OPE
rerun, financial metric or inference benchmark occurred. Frozen108 fits and19,440
replays remain contaminated development; all108 OPE batches remain invalid. The
1,227-return final holdout remains sealed and prospective embargo remains
2027-01-20T13:00Z. No candidate passed. Continue offline research; no adoption.
Existing synthetic training/conformance fixtures are re-executed by the formal gate;
they are engineering tests, not financial trials.
No order, authenticated trading call, live exploration, champion/fleet/risk-limit
change, deployment, or live-authorization change is authorized by these results.

The next local full-verifier attempt failed closed at `admission/cleanup control drift`: lint cleanup had changed the reviewed cancellation expression to `second (Seq.filter (/= ticket))`, while one source-shape assertion still expected the equivalent lambda. The assertion was updated to the actual reviewed expression; no runtime or safety guard was relaxed. This attempt is not a pass.

After the source assertion fix, all23 targeted tests passed in23.733s. Superseded ordinary CI37419042071 and reproduction37419042098 were canceled; neither is a passing result.

Pinned reproduction37419355225/job112125007813 failed with `bot worker publication: support source drift: haskell/trader.cabal`. Its formal/full steps did not run. The nested support registry still pinned the pre-test-registration Cabal/TestMain files; those two reviewed hashes were updated. The worker implementation and predicates are unchanged. A recursive audit found no other stale current-source digest; no source-drift gate was relaxed. Ordinary stale-receipt CI37419355321 was canceled, not passed.

Reproduction37419828641 passed receipt generation and the canonical formal wrapper. Review then found stale numeric bounds in the old backtest/drain ledger descriptions after their conservative models expanded. These are corrected to1792/3456 and5376/17872 states/edges, with a new mandatory receipt-to-ledger bounds check and mutation regression. Its full wrapper was superseded; passing final verification must use the corrected ledger/check. No runtime change accompanies this correction.


Final pinned reproduction [37420428707](https://github.com/diegueins680/trader/actions/runs/37420428707),
job112128332119, tested source `4b1ff2954a5cb67481bd0581f4f2a09d0531829b`:
receipt generation80s; `bash scripts/verify.sh formal`163s; `bash scripts/verify.sh full`539s,
all exit0. Formal/full integrity suites each passed282 tests (85.276s/83.591s).
Haskell build/format/lint/smoke/tests passed, including the new admission suite;
web241/241 and automation185/185 passed. All80 SMT groups returned UNSAT violations.
Receipt SHA256 `f30812898b89f47fffbea4ba171a37ffccc6417a8d1131176d5cb0ab5ad44049`
was imported verbatim after checking every current source digest. Only the new
admission section, the two reviewed model extensions, SMT roster and source hashes
changed; capability graph/effects stayed identical apart from the BacktestGate
source digest. The old12/26 closure summary and both blocked research gates stayed
identical. The temporary reproduction workflow is removed. Ordinary stale-receipt
CI37419828653/37420428765 were canceled, not passed. Final-head CI and merge audits
are recorded in the PR; broader mission and formal closure remain incomplete.

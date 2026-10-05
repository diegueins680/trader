# PPO successor composition — engineering result

This follow-up connects actual offline PPO collection/training to the checked
GAE and immutable optimizer implementations. It adds no financial evidence and
closes no additional broad obligation. Five scoped closures remain; 26 partial
and seven open obligations, plus economic and delivery gates, remain blocked.
The engineering preregistration was committed at `c3145835` before implementation.

The original mechanism sources remain [PPO](https://arxiv.org/abs/1707.06347) and
[GAE](https://arxiv.org/abs/1506.02438), checked at execution time. These papers
study general control benchmarks; neither supports a profitability claim for this
cryptocurrency simulator. This small NumPy implementation remains a mechanism
prototype, not a replication of their benchmark scores or complete architectures.

The new `scripts/research/ppo_successor_v2.py` exposes one versioned,
default-disabled training entry. It copies bounded, already-selected training
prefixes into immutable buffers, rejects invalid prices/funding, fits a shared
training scale and uses the actual replay collector. Every actor/critic forward
and update uses optimizer_snapshot_v2; targets use batch_v2. Four updates per
rollout preserve the original schedule. Terminal targets intentionally use the
v2 correction. Every epoch loss is returned, rather than only the last loss per
rollout. Non-finite loss/gradient/normalization and any rejected update cause an
absent final result. Local actor progress before a critic failure cannot escape
as a completed result. No persistence or partial-policy checkpoint is exposed.

The [contract](../../formal/research/ppo-successor-v2-contract.md),
[source registry](../../formal/research/ppo-successor-source.json),
[proof ledger](../../formal/research/proof-ledger.json) and
[reproduction receipt](../../formal/research/results.json) provide traceability.
No proof status is inferred from the number of related certificates:

| Evidence | Scope and outcome |
|---|---|
| Model checking | 95 states, 110 transitions across one/two-batch models; four epochs per batch; maximum shortest depths 20/35. Only completed paired updates publish. Every modeled transition lowers a finite rank. |
| SMT | Four integer budget/partition/seed claims and one binary64 finite-loss guard; satisfiable premises and UNSAT violations. All admitted 1..4096-step configurations are covered by the arithmetic claims. |
| Source boundary | Nine mandatory complete definitions, four helper files, one disabled/version-guarded entry and one final constructor. Unknown source or missing coverage refuses verification. |
| Actual conformance | 27 combinations of seeds 11/23/47, horizons 1/3/6 and steps 1/17/257, each repeated; eight generated synthetic fits and 512 generated partition checks with generator seed 20261004. Immutable results repeat exactly on the pinned runtime. |
| Failure tests | Actor-only local update, each training-stage rejection, non-finite objectives, unsupported inputs, invalid funding and disabled/version/configuration cases all produce no result. Seven source mutations and a partial-publication model mutation reject. The actual GAE/normalization and objective boundary also replays preserved CE-RL-010/011/012: terminal reward is retained and overflow/NaN paths reject. |

The AST checker binds a reviewed helper/stage abstraction to exact source; it is
not a Python interpreter proof. Trusted helper/runtime semantics are explicit in
A-PPO-SUCCESSOR, A-SNAPSHOT-V2 and A-TARGET-V2. The model contains one private
training call and one/two batches; integer budget lemmas do not turn that finite
model into a universal library or process theorem. Tests supplement this model
and do not establish all-runtime determinism. Concurrent mutation during input
copy, malicious reflection, substituted packages and arbitrary callbacks are
outside the admitted domain. BLAS numerical accuracy and silent underflow are not
proved. A raised failure is handled; a stalled native call is not bounded here.

The same frozen simulator retains unresolved accounting, tail-risk, funding
publication and execution-fidelity limitations. Data availability and historical
prefix selection are not established by accepting finite arrays. This successor
has no dataset loader, holdout access, artifact decoder/writer, OPE estimator,
process-inference adapter, exchange interface or production caller. Exact ESS and
Haskell process cancellation are still to be composed with a future evaluation
and artifact boundary. Q/CQL training repairs remain separate work.

Frozen v1 code, model identifiers, all 108 financial fits, 19,440 historical
replays, rejected-policy/OPE conclusions and sealed holdout remain unchanged.
No new out-of-sample, cost-stress, drawdown, expected-shortfall or statistical
superiority result is reported. The champion and live configuration are unchanged.
Recommend continuing offline engineering/research; no candidate is eligible for
integration or promotion.

Reproduction uses the existing pinned toolchain and requires no downloaded data:

```sh
bash scripts/verify.sh formal
bash scripts/verify.sh automation
bash scripts/verify.sh full
```

The aggregate commands must pass before merge. Their final execution results and
CI evidence are recorded in the pull request. `--require-complete` intentionally
continues to reject the broader mission. The small engineering manifest records
all successful-fit configurations; failure cases remain explicit test definitions.

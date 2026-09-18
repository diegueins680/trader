# Rejected policy cards and lifecycle

All 108 parameter artifacts remain outside Git. Their individual SHA-256 values,
algorithm, horizon, fold and seed are recorded in
[experiment-manifest.json](experiment-manifest.json) and
[multi-seed-training.json](multi-seed-training.json). No policy is retained as a
production, shadow or paper challenger. These cards describe research artifacts
only; there is no integrated candidate model card to imply approval.

| Family / semantic identity | Academic origin | Interpretation | Disposition |
|---|---|---|---|
| PPO / offline_policy_v1 | Schulman et al., 2017 | Categorical clipped actor–critic, separate value model, GAE, 16 tanh units | Reject tested configurations |
| Double DQN / offline_policy_v1 | van Hasselt et al., AAAI 2016 | Separate online argmax and target evaluation; simulated epsilon-greedy replay | Reject tested configurations |
| CQL / offline_policy_v1 | Kumar et al., NeurIPS 2020 | Finite-action pessimism plus Double-Q squared loss; fixed uniform simulated logs | Reject tested configurations |
| CQL no inventory penalty / offline_policy_v1 | Registered ablation | Same offline objective; variance penalty removed from reward | Reject; no robust incremental evidence |

All use `sequential_replay_v1`, `close_inventory_v1`, three absolute exposure
targets and the versioned reward/execution contracts. Source code, registration,
bar and funding hashes, scaler, seed, split and cadence accompany each artifact.
Runtime/library versions and dates are bound by the shared experiment manifest.
No hidden production model ID, pickle, ONNX service, GPU dependency or model
autoload path exists. The strict JSON loader verifies hash, exact fields,
versions, typed required provenance, expected provenance equality, finite numeric
shapes and bounded size; it never
activates a model. Loading remains offline. Post-run hardening rejects numeric strings/booleans, bounds the artifact read,
and records blocked replays after training exceptions. Training and inference
results remain bound to the original frozen source commit.

Intended use: reproduce this failed development screen. Prohibited uses: live
orders, live exploration, deployment, automatic checkpoint selection/promotion,
production self-modification, changing risk limits or treating these artifacts
as evidence of economic efficacy. Invalid/missing/unsupported observations or
inference failure produce absence or a documented neutral proposal, never a
new direction inferred from a missing value. The 20 ms post-call timing check
is not a preemptive production timeout.

The full rejected-policy register is [experiment-registry.csv](experiment-registry.csv),
not a best-seed table. All fits completed, but risk and timing failures remain
in replay results. Training return does not establish trading value. Current
champion comparisons, independent holdout, credible OPE, real state–action
support and production latency tests are absent; they block every policy.

## Future lifecycle conditions, not activation instructions

No candidate qualifies for shadow or paper evaluation. If a separate future
registration passes all offline gates, retain Haskell authorization and risk
control, use independently logged challenger metrics and a fixed artifact hash,
and begin historical replay before shadow and then paper. Each stage needs its
own reviewed admission decision; no direct backtest-to-live transition exists.

That future evaluation must separately monitor feature/observation drift,
residual and prediction bias, calibration decay, regime performance, latency,
failures, costs and safety interventions. These are requirements, not implemented
production monitors in this branch. Rollback here is to stop the explicitly
invoked offline process; default-disabled proposal/inference behavior preserves
the champion. Any future real-capital trial needs a separate human decision.

A follow-up regression reproduced 14 malformed-provenance admissions despite
matching hashes and matching caller metadata. Save and load now reject missing
required fields, malformed digest strings, Boolean/float/negative integer fields,
unsupported horizons/families and non-finite JSON. All 108 original artifacts
remain compatible; [provenance-validation.json](provenance-validation.json) binds
the validator bytes and archive index. This validates metadata structure and
expected identity, not independent authenticity of caller-supplied evidence.
No market observations or policy performance were evaluated in this check.

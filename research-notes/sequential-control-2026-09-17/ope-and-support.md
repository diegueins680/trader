# Off-policy evaluation and support

The implementation contains ordinary trajectory IS, per-decision IS, weighted
IS and sequential doubly robust estimators. Deterministic fixture identities and
zero-support cases pass. Known behavior propensities are exactly 1/3; no estimated
production logging policy is substituted. Importance weights are not clipped.
Conditional 1,000-resample intervals concern randomized simulated behavior,
not generalization across future markets. The policy network is only a DR
control variate: PPO logits are not calibrated Q values.

The campaign attempted 108 batches, each with 200 six-decision episodes.
**Every batch is invalid**: 69–139 episodes per batch failed in behavior or direct
policy replay. The preregistration forbids dropping those failures. Accordingly
no valid empirical ESS, policy-value interval, estimator-agreement statistic or
Q-value calibration is reported. These values are unavailable, not zero and not
passes. [ope-report.json](ope-report.json) retains every failed count.

Fitted Q evaluation, weighted doubly robust, validated model-based estimators
and uncertainty-aware estimator combinations are reviewed but not implemented
here. They cannot repair unknown live logging support or invalid trajectories
merely by reporting smaller variance. OPE remains an unsatisfied acceptance gate.

Uniform random replay covers all three action labels, and per-fit counts are in
[multi-seed-training.json](multi-seed-training.json). That does not establish
joint state–action coverage: real fills, orders rejected by exchanges, liquidity
and inventory distributions are absent. Market-range OOD rates range from 0 to
1.00167% over reported RL replays; a low coordinate-wise OOD rate is not a joint
support guarantee. Unsupported valid observations request neutral exposure;
invalid observations terminate. No policy has demonstrated reliable live
counterfactual support. Behavior cloning imitates the modal uniform action and
is deliberately weak; its result cannot establish superiority over the champion.

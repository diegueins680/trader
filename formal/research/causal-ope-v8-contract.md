# Causal OPE v8 engineering contract

Scope recorded before implementation, 2026-11-13; base
d81af76160ce32249b1a20eb5e7800ab4940e033.

## Scope and interpretation

A separate, default-disabled adapter that feeds the recorded transitions and
behavior probabilities of the two frozen-free successor learners
(causal-value-v6, causal-ppo-v7) to the exact estimator ope-rational-v2. Its
import closure contains no frozen v1 module. It does not train, sample, read a
file or select a policy.

Frozen `sequential_evaluation`, every refuted requirement and preserved
counterexample, ope-rational-v2, ess-rational-v2 and both learners are
unchanged. Acceptance is engineering evidence, not a financial experiment, and
the published estimate carries `reliable = False`.

## Inputs

`evaluate_causal_ope_v8(results, models, horizon, enabled=True)` requires the
exact version `causal-ope-v8`, a tuple of 1..256 learner results, an equally
long tuple of `Model` values and an exact integer horizon H in 1..32 (the
estimator's own limits).

Each result must be exactly a `causal_value_v6.TrainingResult` or a
`causal_ppo_v7.TrainingResult` carrying its own version, an integer seed, and
equally many transitions and behavior probabilities. Results are trusted values
produced by the learners; a forged result is outside the claim.

Each run contributes its first H transitions as one episode. A run with fewer
than H transitions, including one that terminated earlier, rejects the whole
evaluation: no run is dropped, padded or truncated to a shorter horizon, so the
adapter cannot introduce survivorship selection. A caller who removes failing
runs before calling reintroduces it; that is a caller obligation. A run whose
last evaluated row is terminal is admitted. Because H <= 32 is below both
learners' rollout length of 256, every evaluated step was collected with the
run's initial snapshot.

Two results with the same (version, seed) reject: a learner's generator stream
is a function of that pair, so the second is not a second sample. Independence
across distinct seeds, and of the supplied outcomes, is not established.

A `Model` holds, for one run, H target-policy probabilities of the recorded
actions (binary64 in [0, 1]), H action values and H + 1 state values (finite,
magnitude at most 2^99, the last exactly zero). They are caller-supplied. That
each is a function of information available before the corresponding action is
a caller obligation; the adapter cannot check it.

## Behavior probabilities

For a policy-gradient run the recorded probability is the binary64 softmax
value of the drawn action. For a value-learner run it is the exact rational
11/12 or 1/24, converted once to binary64 because the estimator takes binary64
inputs; any other value rejects. Every evaluated behavior probability must be
at least 2^-10. The learners themselves admit probabilities down to 2^-100;
below the evaluation floor the sampling-law gap is not small relative to the
probability, so such a run rejects the evaluation.

## Estimate

Rewards are the exact transition rewards converted under the learners' rule
(relative error at most 2^-52, nonzero values normal) and guarded at 2^99. The
discount is 0.99 per transition. ope-rational-v2 computes, in exact rationals
of the binary64 inputs, the discounted return, trajectory weight, ordinary,
per-decision, weighted and doubly robust estimates, effective sample size,
largest weight and nonzero-weight count. The effective sample size is the estimator's exact field;
ess-rational-v2 is not called because its binary64 weight input would add a
rounding step.

The estimator rejects any intermediate rational above 8192 bits. Each binary64
behavior probability contributes its numerator, up to 53 bits, to the common
denominator of the trajectory weights, and twice that to the sum of squared
weights. Capacity is therefore small: roughly runs x H x 106 bits for
policy-gradient runs. In the fixtures all six runs are evaluated together at
H = 12, four at H = 32, and six at H = 32 reject. The rejection is
all-or-nothing and never silent, but this adapter is not usable for a
statistically meaningful number of long policy-gradient episodes. Raising the
limit or recording dyadic probabilities needs a separately versioned successor.

The estimand is the H-step discounted return from the fixed initial prefix
under the supplied target probabilities, with zero value beyond H. Nothing is
published if any stage fails.

## Sampling-law gap

A recorded probability is not exactly the probability with which the action
was drawn. Under the standard rounding model (A-FP-ROUNDING, u = 2^-53) and a
uniform draw k/2^53:

* The three recorded softmax probabilities sum to one within 2u/(1-u). This
  uses one correctly rounded sum and three correctly rounded quotients and
  needs no accuracy of the exponential.
* The learner draws action 0 if the draw is below p0, action 1 if below the
  rounded p0 + p1, and action 2 otherwise. The probability of each action then
  differs from its recorded probability by at most 2^-50.
* For a recorded probability of at least 2^-10 that is a relative difference
  of at most 2^-40, and the drawing probability is positive.
* Over at most 32 decisions the trajectory weight computed from recorded
  probabilities equals the weight under the drawing law times a factor within
  2^-34 of one.

For the value learner the drawing law is exactly 11/12 and 1/24 under an ideal
generator; only the binary64 conversion rounds, by at most 2^-53 relatively,
which is inside the same bounds. The result publishes the floor, the law gap
2^-50 and the weight gap 2^-34.

Without the floor the relative gap is unbounded: a recorded probability of
2^-100 is drawn with probability 2^-53.

## Obligations / methods

F-RL-OPE-V8-SOURCE: complete reviewed AST lock; activation first; exact import
list and a frozen-free transitive import closure; only the two learner result
types are admitted; a short run rejects; every run index is the horizon prefix;
behavior floor and exact value-learner law; guarded rewards and values;
distinct streams; the estimate is checked before the single publication; the
module neither trains nor samples. The learners' sampling statements and the
estimator limits the lemmas rely on are re-read from their reviewed sources.

F-RL-OPE-V8-ARITH (SMT, quantifier-free real arithmetic, plus exact rational
evaluation): the four statements above, a satisfiable control showing the
floor is necessary, and the exact value of (1 + 2^-40)^32 - 1.

F-RL-OPE-V8-FLOW: finite model of activation, per-run conversion, stream
distinctness, estimation and publication. Publication happens only after every
requested run was converted and the estimate was checked; any failure is
terminal and publishes nothing.

F-RL-OPE-V8-CONFORMANCE: three seeds per learner and 40 synthetic steps; all
six runs together at horizons 1, 7 and 12, each run alone at horizon 32, and
the registered capacity boundary. Every published field equals an independent exact-rational
evaluation. A target equal to the recorded behavior gives unit weights and
full effective sample size. For every evaluated policy-gradient step the exact
drawing probabilities, counted on the 2^-53 grid, lie within the certified
gaps of the recorded ones and reproduce the recorded action. Rewriting every
outcome after the horizon leaves the evaluation unchanged. Short, forged,
reused, out-of-domain and oversized inputs publish nothing. Tests are not
proofs.

A-CAUSAL-OPE-V8 trusts pinned CPython 3.13.3 float, Fraction and dataclass
semantics, correctly rounded binary64 addition, division and `math.fsum`, a
NumPy 2.3.5 generator whose `random()` is uniform on multiples of 2^-53 and
whose `integers(3)` is uniform, honest learner results and helper modules, the
reviewed AST translation, Z3 and sufficient resources. The rounding and
generator models are standard and are not machine-proved for the CPython or
NumPy build. No hostile reflection or monkeypatching is addressed. No
unbiasedness, coverage, independence, variance, data-support or economic claim
follows, and numeric admission is not empirical support.

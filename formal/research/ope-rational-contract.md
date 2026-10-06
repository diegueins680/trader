# Exact finite-horizon OPE v2 contract

Engineering registration: `research-notes/registrations/ope-rational-engineering.json`.
This successor is disabled by default. It consumes explicit synthetic or separately
approved logged trajectories; it does not collect data, reopen the frozen screen,
select a policy, publish an artifact, or grant an order capability. The existing
screen and its invalid OPE evidence retain their semantics.

For N in 1..256 equal-length episodes, T in 1..32 decisions, accept immutable
base tuples containing base floats r,b,p,q and T+1 values v, base integer actions
in {0,1,2}, finite gamma in [0,1], 0<b<=1, 0<=p<=1, terminal v[T]=0.
Convert each binary64 input to its exact rational value *before any arithmetic*.
No clipping, floating ratio, cumulative product, discount, mean or ESS is allowed.
An absent or invalid datum rejects the entire batch. Authentic propensity,
trajectory completeness, causal logging, support and valid Q/V meanings remain
caller assumptions; numeric admission does not establish those facts.

w[-1]=1, d[0]=1, w[t]=w[t-1]*p[t]/b[t], d[t+1]=d[t]*gamma.
G=sum d[t]*r[t]. I=w[T-1]*G. P=sum w[t]*d[t]*r[t].
D=v[0]+sum w[t]*d[t]*(r[t]+gamma*v[t+1]-q[t]).
Return mean(I), mean(P), mean(D), sum(I)/sum(w) when sum(w)>0 (else absent),
ESS=sum(w)^2/sum(w^2) when sum(w^2)>0 (else zero), maximum w and nonzero
trajectory count. Preserve per-episode (G,w,I,P,D) for reproducibility. All numeric
results are exact rationals in an immutable versioned envelope with reliable=False.
No bootstrap, confidence interval, significance or statistical acceptance is
inferred. These are descriptive estimators, not evidence of reliable OPE.

Each rational intermediate must have numerator and denominator bit lengths <=8192.
A checked arithmetic primitive rejects any larger result; products of admitted
operands need at most16384 bits and addition numerators at most16385 before
reduction (sign excluded). Conversion needs <=1075 bits. Allocation/arithmetic
failure rejects the batch; no partial result is observable. This is a resource
bound on arithmetic objects, not a wall-clock or process-memory guarantee.

Requirements: F-RL-OPE-V2-ARITH (source-extracted rational recurrences/denominators
and size lemmas, SMT); F-RL-OPE-V2-BOUNDARY (complete source/interface/effect review);
F-RL-OPE-V2-FLOW (finite batch admission, accumulation, rejection and publication
model); F-RL-OPE-V2-CONFORMANCE (independent exact oracle, numeric extremes,
mutants, deterministic replay, all-zero support and disabled paths).

Abstraction: each Fraction maps to numerator/denominator over rationals; private
lists map to completed episode prefixes; public envelope maps to complete batch.
Trusted CPython Fraction/integer/tuple/dataclass semantics and sufficient resources
are explicit assumptions. No whole-Python refinement or future-profitability proof.
The model abstracts episode payloads; implementation correspondence is source
extraction plus differential testing, not a full compiler theorem. Obligation10
stays open: frozen replay, Q/CQL, funding and learning numeric paths remain uncovered.

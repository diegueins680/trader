# PPO snapshot to supervised Haskell inference v3

Specified before implementation against f0903713fee9f6f60664cd1ca29e3594c0ba7694.
This engineering bridge composes trained PPO successor snapshots with the existing
Haskell process supervisor. No financial trial, artifact persistence, deployment,
production integration or additional broad obligation closure is authorized.

`encode_request_v3` is a pure, default-disabled Python entry. Only the exact
`ppo-successor-v2` TrainingResult representation, the supported optimizer version,
actor width 3, supported configuration, matching positive completed optimizer step
and four immutable parameter buffers of lengths 192/16/48/3 are accepted. It takes
one already constructed float64 observation of width 12. Values must be finite
and bounded in absolute value by 1000. There is no dynamic executable, callback,
file, network, promotion, artifact writer or order interface.

The ASCII envelope is the Haskell-readable tuple
`("PPO-SNAPSHOT-V3", step, observationWords, parameterWords)` plus newline.
Each word is the unsigned integer representation of a little-endian IEEE754
binary64 bit pattern; Python emits integer text, avoiding cross-language decimal
float conversion at this boundary. Field lengths are exactly 12 and 259; step is
4..64 and divisible by 4. The frame must be shorter than 32768 bytes. Metadata is
compatibility data, not authenticated evidence that training occurred.

A small pure Haskell decoder uses a bounded decimal grammar and Integer before checked conversion to Word64.
Reject negative words, words >=2^64, wrong tag/shape/step, NaN, infinity and values
outside [-1000,1000]. Cast valid Word64 bits to Double. It returns the same bounded
Request consumed by the current worker, never an order or capability. A separate
versioned decoder probe supports exact-bit conformance tests without starting a
worker. No existing v2 flag or payload acquires different semantics.

The new explicit `--offline-snapshot-v3` mode selects this decoder inside the
existing 20 ms supervised exchange. Factor only the pure decoder parameter through
exchange/supervise/offline; preserve child identity, empty environment, startup
budget, response protocol, cleanup and final admission guard. The child continues
to receive the existing Haskell Show/Read Double request. Revalidate decoded input
before sending. Parsing, reply and cleanup remain inside the admission budget.
Unknown/default modes still reject before input/spawn. Existing v2 fixtures must
all pass; new-mode timeout/cleanup tests must use actual processes, not mocks.

Requirements:
- F-RL-BRIDGE-V3-CODEC: source-bound SMT for Integer word range/conversion,
  step/shape/version admission and binary64 finite/bounded selection. Prove
  source configuration step arithmetic maps completed PPO budgets to 4..64.
- F-RL-BRIDGE-V3-FLOW: source-bound finite decoder/validation/send/cleanup model,
  composed with the existing process lifecycle. A failed decode, validation,
  expired deadline or failed cleanup cannot publish a proposal. Check every
  abstract gate outcome; no arbitrary two-case bound where full enumeration fits.
- F-RL-BRIDGE-V3-BOUNDARY: complete reviewed Python/Haskell source and helper
  inventories; default-disabled encoder, explicit Haskell mode, no file/network/
  authorization effect. Extend existing closed boundaries to include this entry.
- F-RL-BRIDGE-V3-CONFORMANCE: actual trained PPO seeds 11/23/47 across horizons
  1/3/6; exact-bit decoder round trips, invalid inputs, source mutations, supervised
  evaluation and timeout/cleanup fixtures. Compare clear-margin NumPy decisions
  with Haskell observations while allowing safe timeout absence. Ties abstain.

A-BRIDGE-V3: pinned CPython/NumPy/GHC/base semantics, IEEE754 binary64 layout,
Integer-to-Word64 after a checked range, GHC bit casts, trusted stable base inputs,
Haskell finite-Double Show/Read round trip, and existing process/OS assumptions.
No hostile reflection, substituted packages, external reaper or arbitrary runtime
callbacks. Decimal internal child transport and numerical agreement are tested;
no verified parser/compiler/BLAS/tanh theorem is claimed. Haskell evaluates the
model and has final proposal authority; cross-language near-tie numerical parity
is unproved. Unique maximum differs intentionally from the frozen Python argmax
rule; this new version uses the existing Haskell abstention rule.

This is a transient inference request, not a persisted policy artifact or proof
of training provenance. Availability/revision witnesses, accounting, OPE/ESS,
persistent artifact hashes, production lifecycle/ownership and neural robustness
remain unresolved. Training and inference timings are engineering measurements,
not future performance or hard OS timing guarantees. No historical data/holdout.


Build interpretation, recorded after the first conformance attempt: GHC `-O0`
safely timed out for all three observations of one trained configuration and
therefore failed the per-policy successful-inference test. The new bridge is
verified with pinned GHC9.4.8/base4.17.2.1 `-O2`; the existing v2 suite retains
`-O0`. The deadline and failure expectations are unchanged. At least one of the
three supervised observations must succeed for each of the nine fitted policies;
every returned proposal must agree on these clear-margin fixtures. Safe timeout
absence is separately counted in the engineering benchmark. No financial protocol
or registered seed/horizon/step budget was changed.

A subsequent loaded-host benchmark observed 30/30 safe absences for both builds.
Optimization is not a demonstrated real-time guarantee. Preserve this evidence;
operational qualification remains blocked and conformance gates stay unchanged.

Parser refinement: the final v3 decoder uses a small bounded decimal grammar
instead of the generic tuple/list Read instance. The wire format is unchanged;
only ASCII whitespace outside tokens and unsigned decimal fields of at most
20 digits are accepted. List traversal counts down fixed widths 12/259 and
rejects extra/missing values or trailing material. Twenty decimal-fold bounds
and a decreasing-list-count lemma supplement the four admission lemmas (25
SAT-premise/UNSAT-violation pairs total). These lemmas and actual codec tests do
not prove the whole parser or internal child Show/Read implementation. The first
bounded-parser test also safely timed out under host load; that failure remains
recorded, and no deadline or successful-policy gate was relaxed.

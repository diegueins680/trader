# PPO artifact byte boundary v4

Preregistered against a8f3dcd166ec853037cf168eb2bf1993fd99b43c before implementation.
This pure offline codec connects completed PPO-v2 immutable results to the v3
Haskell inference request. It performs no file selection, I/O, training, promotion
or order authorization. Public encode/decode/request functions default disabled.
The frozen v1 artifact and financial screen retain their existing semantics.

The versioned JSON object has exactly schema, contracts, enabled, promotion,
provenance and result fields. Schema is ppo-artifact-v4; enabled is false;
promotion is research-only. Contracts identify PPO-v2, optimizer-v2, replay-v1,
close-inventory-v1, signed-quarter-v1, reward-v1 and snapshot-v3. Unknown versions,
fields, duplicate keys and noncanonical encodings reject. Raw bytes are bounded
at 65536; the same immutable bytes are hashed and decoded, without reopening a
path. SHA256 is integrity against a caller-supplied expected digest, not identity
or proof that the described training occurred.

Expected provenance is an exact seven-field dictionary: codeCommit (40 lowercase
hex), registrationSha256/dataSha256/fundingSha256/splitSha256/proofSha256
(64 lowercase hex) and datasetRole, exactly development. Neither validation nor
holdout claims are accepted by this engineering version. Caller-provided hashes
are trusted reference values; provenance truth and authenticity remain assumptions.

Result preserves version, seed, horizon, steps, sorted unique symbol names,
normalizer bytes, actor/critic snapshot bytes and loss values without decimal
float roundoff. Snapshots have exactly the supported version, output width 3/1,
completed step 4*ceil(steps/256), four p/m/v buffers with exact shapes and finite
binary64 values, with nonnegative v. Normalizer is four six-value finite buffers;
losses have completed-step length and finite native floats. All nested arrays are
immutable tuples/bytes after decoding. No optimizer restoration API is introduced.
The current v3 encoder independently bounds actor parameters and observations.

F-RL-ARTIFACT-V4-GUARD: source-bound SMT establishes that accepted digest/metadata
comparisons exclude mismatches and that completed steps and byte bounds agree
with the supported training configuration. Primitive hash, JSON and byte semantics
are trusted; no collision resistance, cryptographic authenticity or parser theorem.
F-RL-ARTIFACT-V4-FLOW: finite state model covers enablement, byte/type/size,
digest, JSON, provenance, structural/numeric/canonical checks, restoration and
v3-request rejection/publication. A failed gate cannot publish; no retry or authority.
F-RL-ARTIFACT-V4-BOUNDARY: entire module and all helper sources are inventoried;
three explicit default-disabled public entries; no effects or production imports.
F-RL-ARTIFACT-V4-CONFORMANCE: actual PPO fits at seeds 11/23/47, horizons 1/3/6,
17 steps; byte-exact restore/re-encode and pre/post Haskell decoder parity; corrupted,
incompatible, forged-metadata, oversized, malformed and non-finite artifacts reject.
Generated valid bit patterns test signed zero/subnormal preservation. Deliberate
source/model mutants must be rejected. The existing supervised-process suite still
runs unchanged. No new historical experiment or sealed holdout access occurs.

A-ARTIFACT-V4: pinned Python/NumPy/GHC/runtime primitives, stable ordinary inputs,
immutable bytes, truthful externally supplied expected metadata and trusted
hashlib SHA256/JSON/dataclasses/hex conversion; no monkeypatching or hostile
subclasses. Source inspection and a finite abstraction are not whole-interpreter
refinement. External file retrieval/atomic replacement, freshness, signed manifests,
normalizer application, Haskell-side hash/provenance admission, production loader
integration and full obligations 29/30 remain unresolved. This is not a candidate
activation capability. All existing five scoped closures must include this boundary.

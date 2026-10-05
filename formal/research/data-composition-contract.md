# Normalization and symbol composition, version 1

Preregistered at `32740f68` before implementation. The closure criteria for
mission obligations 3 and 5 are unchanged. This contract covers the delivered
hash-pinned offline screen, including its loader, training, baseline, OPE and
replay callers. It introduces no production integration or financial trial.

## Concrete composition

The input loader reads each file once and checks its registered digest before
parsing those same bytes. The registered symbol filter creates each local price
and funding array. The price and funding dictionaries preserve the same key.
Training comprehensions preserve keys and slice at the registered training stop.
All training calls receive those prefixes. All replay constructors receive a
price/funding pair for one symbol; no other local market series is an argument.
The OPE and collection symbol selectors use the same key for both arrays.

Normalization is an explicitly **shared training transform**: the registration
permits pooling all symbols' training prefixes. Learned policy and baseline
parameters are also shared training inputs. This is distinct from local market
observations; no claim of independence from other symbols' training history is
made. There are no global live-market features in this screen.

Every Scale creation in the runner is its single `Scale.fit(train.values())`
site. The existing source-derived prefix theorem establishes the exact read
bounds. Each parameter is copied to a NumPy view backed by immutable `bytes`;
the frozen dataclass stores those snapshots. The finite source-use checker
checks **every** Scale/scale reference, each passthrough call and each alias
assignment, including the two `self.scale` owners. Only reads, serialization,
checked forwarding and initial owner binding are admitted. A changed call,
extra use, alternate fit, escape or write must fail the certificate.

## Verification and refinement

F-RL-DATA-COMPOSITION exhaustively checks the finite source-use/dispatch graph
and the reviewed loader/immutable-constructor skeleton. Source identities are
not themselves proofs: the checker additionally extracts the actual selection,
output keys, paired call arguments and array-snapshot operations. F-RL-DATA-KEYS
uses those extracted key expressions to prove arbitrary-symbol equality with
SMT. The existing F-RL-FIT-PREFIX and F-RL-FEATURE-FOOTPRINT certificates supply
the universal index part. Runtime fixtures execute the actual loader and fit
setup, then exercise actual collect/replay/OPE construction with distinct symbols.
They check immutable parameters and perturb evaluation data. These fixtures are
conformance tests, not a proof of the Python interpreter or numerical libraries.

The abstraction maps a concrete source use to a finite forwarding/read edge,
each admitted dictionary key to a mathematical string, each prefix extent to
an integer, and a Scale field to an immutable bytes-backed value. Structural
admission rejects unsupported source rather than extrapolating to arbitrary
Python. The reviewed primitive contract is that numerical functions consume
only their explicit inputs and do not introspect callers or mutate bytes.

A-DATA-COMPOSITION names trusted Python AST/control-flow/argument binding,
ordinary pandas equality filtering and numeric conversion, NumPy immutable
bytes views and read-only reductions, cryptographic digest identity, standard
library behavior, and the checker. Inputs are ordinary registered files/base
arrays, with stable library bindings and no runtime monkeypatching, hostile
subclasses, arbitrary injected callbacks or concurrent writers. These are
library/runtime assumptions, not assumptions that normalization is causal or
that the chosen keys agree. Those latter facts are checked from the source.
The pinned complete source inventory prevents unreviewed helper substitutions.
This is source-level composition under a reviewed library contract, not a
machine-verified compiler or deployed-image theorem.

## Remaining gaps

The historical panel has no original publication/revision witnesses. Whole-panel
admission can reject on future invalid data. Validation features legitimately
read their trailing warmup, which can overlap training history; the two six-bar
purge/embargo labels do not allocate two separate gaps. This work therefore
**does not close obligations 1, 2, 4 or 17**. It does not close numeric safety,
accounting, promotion, shutdown, ownership, recovery or statistical acceptance.
The frozen v1 counterexamples and rejected policy evidence remain unchanged.

## Checker counterexample

CE-DATA-COMPOSITION-001 preserves an admitted coverage mutant at 12bec119:
removing the reviewed skeleton roster still returned a certificate. The fixed
checker requires all eleven skeletons and all four source hashes, supported
schema and explicit shared inputs before any semantic check. Tests reject an
empty or partially omitted roster. This was a verifier admission defect, not
observed cross-symbol market leakage. Key claims additionally require each
paired key to equal the declared symbol; equal foreign keys alone are insufficient.

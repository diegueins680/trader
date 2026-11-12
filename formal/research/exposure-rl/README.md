# Exposure-control RL path: scoped formal assurance

Machine-readable ledger: [`assurance-ledger.json`](assurance-ledger.json). Reviewed receipt: [`results.json`](results.json). Both are checked by

```bash
bash scripts/verify.sh formal
```

which runs `scripts/formal/exposure_rl/test_exposure_formal.py` (checker mutation tests) and `scripts/formal/exposure_rl/verify.py` after the existing research proof steps. To run only this path:

```bash
python3 scripts/formal/exposure_rl/verify.py            # compare with the receipt
python3 scripts/formal/exposure_rl/verify.py --record   # rewrite the receipt after a reviewed change
```

It needs the pinned toolchain in `formal/research/toolchain.json` (Z3 4.15.4, GHC 9.4.8). A solver timeout, UNKNOWN, a vacuous premise, a failed conformance case, an unreproduced ledger status or a receipt mismatch is a failure.

## What is checked, and how

| Kind | Claims | Tool | What it covers |
|---|---|---|---|
| SMT proof | 7 shield, 4 reservation | Z3 | The integer projection for every valid snapshot and every integer proposal. The Z3 term is derived from the syntax tree of `scripts/research/exposure_rl/shield.py`. |
| SMT proof | 6 ledger, 3 partition | Z3 | The declared ledger over the reals and the fold index arithmetic over the integers. Specification level. |
| Model checking | 1 | Breadth-first exploration | A bounded lifecycle model: 2 proposers, budget 2, one retry, 648 states. A non-atomic mutant must yield the budget counterexample. |
| Exhaustive bounded check | 3 | GHC-compiled driver, compile check, source audit | Haskell and Python agree on 120,080 shield cases and 4,096 reservation cases; a forged intent does not compile; the path's sources reach no order capability. |
| Property tests | 9 | `test/exposure_rl_test.py` | Feature causality, partitions, the binary64 ledger against exact rationals, action validation, termination, reward-exploit probes, artifacts, determinism, learner mechanics. Not proofs. |
| Open | 2 | none | Neural-network bounds; refinement to an order-capable production path. |

## Obligations

| ID | Obligation | Status | Why not fully discharged |
|---|---|---|---|
| FM-01 | Causality | partially verified | Property-tested only. |
| FM-02 | Partition integrity | partially verified | Index model proved; implementation linked by a property test. |
| FM-03 | Accounting | partially verified | Declared ledger proved; binary64 code linked by exact-rational tests. |
| FM-04 | Action validation | partially verified | Binary64 decoding checked on a bounded grid and a seeded sample. |
| FM-05 | Risk admission | partially verified | Projection proved; pending orders, precision, minimum notional and margin are not modelled. |
| FM-06 | Authorization | discharged in scope | — |
| FM-07 | Lifecycle | partially verified | No refinement proof from the model to code. |
| FM-08 | Failure handling | partially verified | Inference timeouts are modelled, not implemented. |
| FM-09 | Atomicity and freshness | partially verified | No concurrent store implements the reservation. |
| FM-10 | Compatibility | partially verified | Property-tested only. |
| FM-11 | Noninterference | discharged in scope | — |
| FM-12 | Recovery | partially verified | Real recovery actions, crashes and persistence are not modelled. |

"Discharged in scope" means every required certificate is machine-checked in the same run and no open item is listed. The scope is the research path's listed sources. `verify.py` computes each status from the certificates it actually reproduced and rejects a ledger that claims more.

## Three kinds of claim, kept apart

- **Action-admission invariants** (proved): what the shield will admit given the snapshot it is handed.
- **State invariants conditional on the environment** (not proved): exposure after a fill can exceed the admitted target through cost deduction and price drift. The next snapshot reports it, and above the cap plus the minimum trade the shield provably forces a reduction.
- **Financial-risk outcomes** (empirical only): drawdown, tail loss and solvency under real gaps, outages or exchange failures. Nothing here bounds them.

## Limits

- The proofs are about the Python source's integer logic. The NumPy batch form and the Haskell module are tied to it by tests and conformance.
- The Haskell shield is research-only: `haskell/research/ExposureShieldV1.hs` is compiled by this target and is not a component of any executable.
- A snapshot that misreports the account voids every admission invariant.
- This sub-ledger is separate from `formal/research/proof-ledger.json`; that ledger's 38 mission obligations and their statuses are unchanged.

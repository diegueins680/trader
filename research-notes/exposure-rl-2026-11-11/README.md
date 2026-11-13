# Exposure-control RL campaign (`exposure_rl_control_v1`), 2026-11-11

**Outcome: no RL candidate passed its registered gates. Nothing is integrated; the champion is unchanged; the confirmation window was never opened.**

| Read | For |
|---|---|
| [adoption-memo.md](adoption-memo.md) | The decision, results, gates and limits |
| [policy-cards.md](policy-cards.md) | One card per policy |
| [results/comparison-report.md](results/comparison-report.md) | Every seed, fold, symbol and stress |
| [repository-baseline-and-decision-problem.md](repository-baseline-and-decision-problem.md) | What the repository already had, and why this decision problem |
| [environment-contract.md](environment-contract.md) | Observation, action, shield, timing, ledger |
| [literature-review.md](literature-review.md), [paper-to-implementation.md](paper-to-implementation.md), [paper-matrix.csv](paper-matrix.csv), [search-log.md](search-log.md) | The research review and its sources |
| [../registrations/exposure-rl-control-v1.json](../registrations/exposure-rl-control-v1.json) | The registration |
| [experiment-manifest.json](experiment-manifest.json) | Data hashes, licences, cost model, trial accounting and holdout ledger |
| [../../formal/research/exposure-rl/README.md](../../formal/research/exposure-rl/README.md) | Formal claims and what remains open |

`results/results.json` is the machine-readable summary, `results/all-seed-results.csv` has one row per policy, fold, seed and stress, and `results/experiment-registry.csv` lists all 104 jobs.

## Reproduce

Python 3.13 with `scripts/research/sequential-requirements.txt` (NumPy 2.3.5, pandas 2.3.3). CPU only; about 1.5 hours on 12 workers. Data and artifacts go to the ignored `.tmp/research/exposure-rl-v1/`.

```bash
python3 scripts/research/exposure_rl/acquire.py --partition development --output .tmp/research/exposure-rl-v1/data-dev
(cd haskell && cabal build trader-hs)
python3 scripts/research/exposure_rl/run_campaign.py champion --binary "$(cd haskell && cabal list-bin trader-hs)"
python3 scripts/research/exposure_rl/run_campaign.py development --run my-run --workers 12
python3 scripts/research/exposure_rl/report.py --run my-run --output /tmp/exposure-report
python3 test/exposure_rl_test.py
bash scripts/verify.sh formal
```

The runner refuses to start unless the campaign sources are committed and the snapshot matches the registered hashes. A fresh download is expected to match those hashes only if the provider has not revised its history. Training is seeded; results repeat on the same machine and library build, and bitwise equality across hardware is not claimed.

Everything here is offline research on public data with simulated fills. It reads no credential and has no order-capable client.

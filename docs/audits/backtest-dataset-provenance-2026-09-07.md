# Backtest dataset provenance audit — 2026-09-07

## Decision

Close `EXECUTION-DATASET-001` at the current generator boundary. Replace the
moving “latest 1,000 bars” fetch with a fixed-window, completed-bar-only,
SHA-256-bound acquisition and offline verification contract. Record explicitly
that this source path uses no randomness; keep every randomized research
calculation on an explicit seed.

This is research-data infrastructure. It changes no checked-in dataset, return,
forecast, metric, model, champion, holdout, promotion, deployment, credential,
order path, or live authorization.

## Finding before the change

`scripts/fetch-data-pipeline.sh` described itself as repeatable but queried the
Binance spot kline endpoint with only symbol, `4h`, and `limit=1000`. The
endpoint therefore returned a different moving window as time advanced. The
script immediately replaced each tracked CSV and printed an MD5 checksum, but
recorded no fixed end time, provider/license boundary, source-row hash,
generator identity, or batch manifest. An interrupted multi-symbol refresh
could also leave a mixture of old and new files without a manifest capable of
rejecting the state.

The resulting ambiguity was observed historically: the May 2026 baseline memo
listed data drift as a falsifiable explanation for a Sharpe collapse, and the
BTC file was later explicitly reverted to a ratified Git revision. The current
tracked 1,000-row files are immutable Git evidence with these byte hashes, but
they predate the new manifest and are not retroactively claimed as generated
by it:

| File | SHA-256 | Provenance disposition |
|---|---|---|
| `data/BTCUSDT-4h-1000.csv` | `d14b68b5c47ad875cef05ae4cac188ae028d854420fe90ade91aeef87557dcc9` | Historical Git fixture; no schema-1 source manifest |
| `data/ETHUSDT-4h-1000.csv` | `d2ab3f715853e35ec1dcfa74819214830be61b2d0c9bd67b65930dd96c1f6552` | Historical Git fixture; no schema-1 source manifest |
| `data/SOLUSDT-4h-1000.csv` | `947c2bf646203cef86dfedfdf6ce41eaadad64cf9ca93dc59cb407c188a1fd84` | Historical Git fixture; no schema-1 source manifest |

The `data/stress-*.csv` files are likewise historical fixtures without a
reconstructable generator/seed manifest. They may reproduce old code paths by
Git byte identity, but they are not admissible as fresh confirmatory market
evidence and are not relabeled as such here.

Current research randomness is explicit rather than ambient: the NumPy random
generators in `scripts/research/harness.py` receive a seed, campaign runners
persist the bootstrap seed, and synthetic regression fixtures use a fixed seed.
The repaired market-data generator itself has no random operation, so its
manifest records `{"used": false, "seed": null}` instead of inventing a seed.

## Implemented contract

The existing shell path remains compatible as the operator entry point and now
delegates to `scripts/fetch-backtest-data.py`, which uses only Python's standard
library. A fetch requires `END_TIME_MS`; omission fails before any request. The
generator:

1. sends the public read-only Binance request with explicit symbol, interval,
   row limit, and fixed end time; the production URL is restricted to the
   official endpoint and URL credentials are rejected;
2. bounds response size and request timeout;
3. requires the exact row count, 12-field schema, positive finite OHLC,
   non-negative finite volumes, coherent OHLC, exact timestamp contiguity, and
   a close time no later than the declared end;
4. emits deterministic UTF-8/LF CSV bytes;
5. records the request, symbol, interval, limit, time bounds, source-row
   SHA-256, CSV SHA-256, public source/license boundary, generator SHA-256, and
   explicit no-randomness status in a sorted schema-1 JSON manifest;
6. stages and atomically replaces each file, writing the manifest last, so an
   interrupted batch cannot agree with the prior manifest by accident; and
7. prints the source-row, CSV, and manifest SHA-256 values in command output.

`EXPECTED_MANIFEST` turns a retrieval into an exact reproduction attempt. Any
provider revision, request drift, code-hash change, or dataset change fails
before output replacement. `VERIFY_MANIFEST` performs a network-free check of
the generator identity, public source/license contract, request/filename
identity, CSV bytes, schema, rows, timestamps, source-row hash, and all recorded
digests.

The manifest proves the acquired bytes and request contract. It does not prove
that Binance will retain history forever, that a historical API response was
never revised, or that a dataset supports a profitable strategy. A first
acquisition must be stored outside Git when large; its manifest, lawful source
and license record, and expected hashes can be committed or otherwise frozen
according to the experiment registration.

## Usage

Choose a UTC millisecond cutoff at or after the close of the final desired
candle, then run from the repository root:

```sh
END_TIME_MS=1767225599999 bash scripts/fetch-data-pipeline.sh
```

The defaults request BTCUSDT, ETHUSDT, and SOLUSDT, 1,000 `4h` spot bars, and
write `data/backtest-data-manifest-v1.json`. For a previously frozen manifest:

```sh
END_TIME_MS=1767225599999 \
EXPECTED_MANIFEST=/path/to/backtest-data-manifest-v1.json \
DATA_DIR=/path/to/staged-data \
bash scripts/fetch-data-pipeline.sh

VERIFY_MANIFEST=/path/to/staged-data/backtest-data-manifest-v1.json \
bash scripts/fetch-data-pipeline.sh
```

Alternative `SYMBOLS`, `KLINE_INTERVAL`, `KLINE_LIMIT`, and `DATA_DIR` values
are explicit inputs. `BINANCE_KLINES_URL` exists for the loopback regression
fixture; production use is restricted to the official Binance endpoint and
non-loopback HTTP, alternate HTTPS hosts, or URL credentials are rejected.

## Verification evidence

`test/backtest-data-pipeline.test.mjs` uses a loopback Binance fixture and
proves:

- request query parameters contain the fixed end time;
- an identical response reproduces byte-identical CSV and manifest files;
- stdout contains the source-row and CSV SHA-256 values plus the no-randomness
  declaration;
- offline verification succeeds on the intact manifest;
- omission of `END_TIME_MS` rejects the moving-window request;
- provider-row drift against an expected manifest changes no output;
- a bar closing after the cutoff is rejected;
- a non-finite market value and a credential-bearing endpoint are rejected
  without echoing the credential; and
- a changed CSV fails offline hash verification.

The test makes no external request, downloads no market data, opens no holdout,
and produces no performance metric.

## Remaining limitations

- The three tracked May 2026 market CSVs and old stress fixtures remain legacy
  evidence without schema-1 acquisition manifests. They cannot be upgraded by
  assertion; only their current Git bytes are reproducible.
- Binance is a mutable external source. A changed response fails an expected
  reproduction instead of being silently accepted; the tool cannot force the
  provider to return old bytes.
- One manifest covers one bounded endpoint page per symbol. Larger historical
  campaigns continue to use the separately tested paginated snapshot tools.
- Atomic file replacement plus manifest-last publication makes partial state
  detectable, not transactionally impossible across several files.
- The generator validates source integrity and causality, not universe
  construction, label alignment, model leakage, or economic fitness.

These limitations do not leave the original moving-window generator defect
open. They prevent overclaiming legacy provenance and route larger or
randomized research through the repository's stricter campaign manifests.

# Binance derivatives bar decoder v2 — infrastructure result

Date: 2026-09-04.

## Result

`Trader.Predictors.DerivativesPanelSchema` is a pure, opt-in Haskell decoder
for the additive `binance_derivatives_first_seen_v2` fields in research bar
CSVs. It binds availability semantics to `feature_availability_v2` and decodes
funding, open interest, basis, and taker flow without changing any legacy
column. A family whose five v2 cells are all blank is legacy-absent; an
explicit unavailable cell has false observed/fresh masks, zero dense value,
and no timestamps; an observed zero retains true masks and causal witnesses.
An observed-but-stale cell retains its timestamps and observed mask but is
unusable and must have zero dense value.

The decoder requires each complete five-field family group exactly once and
in canonical relative order while permitting unrelated legacy columns. It
rejects duplicate headers, partial cells, non-finite values, non-binary masks,
future availability, `eventTime > availabilityTime`, timestamp arithmetic
overflow, incorrect family-specific freshness, non-zero stale/unavailable
values, empty panels, and non-increasing bar grids. Funding uses the collector's
nine-hour limit; the other families use two bar intervals. The supplied symbol
must be canonical ASCII alphanumeric text. File hashes and symbol-to-file scope
remain caller/manifest responsibilities rather than properties inferred from
CSV bytes.

During cross-runtime testing, pandas `combine_first` was found to alphabetize
columns on overlap refreshes. The offline cache merge now explicitly restores
the fresh frame's canonical order and appends only legacy-exclusive fields, so
repeated collection cannot drift the versioned schema.

## Compatibility and promotion boundary

The module is compiled and tested but has no import from `Features`,
`ExogenousFetch`, a model loader, a predictor, a bot, or an execution module.
It does not reinterpret an old cache row, alter a saved model identifier, open
a registered holdout, fit a candidate, or authorize trading. A later candidate
must separately bind the cache to a verified manifest and artifact, preserve
these masks through a versioned feature builder, preregister prospective data,
and pass the unchanged promotion gates. `FEATURE-MISSINGNESS-001` therefore
remains open.

## Verification

The shared fixture
`haskell/test/fixtures/binance_derivatives_first_seen_v2.csv` covers observed
zero, explicit unavailable evidence, a legacy-absent family, and an observed
stale witness. Haskell regression tests exercise decoding and failure cases;
the Python collector validates the same fixture and its output order; an
isolation regression checks that production feature construction and fetch
paths do not import the decoder.

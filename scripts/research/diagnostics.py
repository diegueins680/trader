"""Formal multiple-testing diagnostics for aligned strategy returns.

The functions in this module operate on a *complete* return matrix: rows are
ordered observations and columns are every trial considered in a search.  They
fail rather than silently intersect dates or discard missing values because
either operation changes the multiple-testing experiment being measured.

``deflated_sharpe_ratio`` implements the Bailey and Lopez de Prado Deflated
Sharpe Ratio (DSR).  ``cscv_pbo`` implements combinatorially symmetric
cross-validation (CSCV) and reports the Probability of Backtest Overfitting
(PBO).  These are formal diagnostics; the inexpensive Sidak adjustment in
``harness.py`` is retained only as an explicitly named proxy.

Dependencies: numpy, pandas (stdlib otherwise).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb, e, isfinite, log, sqrt
from statistics import NormalDist
from typing import Hashable

import numpy as np
import pandas as pd

EULER_MASCHERONI = 0.5772156649015329
_NORMAL = NormalDist()


@dataclass(frozen=True)
class CompoundedReturnMatrix:
    """A full-block return matrix and its auditable source boundaries."""

    matrix: pd.DataFrame
    blocks: pd.DataFrame
    original_observations: int
    retained_observations: int
    block_size: int
    dropped_leading_observations: int
    source_first_key: Hashable
    source_last_key: Hashable
    retained_first_key: Hashable
    dropped_leading_keys: tuple[Hashable, ...]

    def to_dict(self) -> dict[str, object]:
        boundaries = []
        for record in self.blocks.to_dict("records"):
            boundaries.append(
                {
                    key: _json_safe_alignment_key(value)
                    for key, value in record.items()
                }
            )
        return {
            "method": "non_overlapping_compounded_returns",
            "originalObservations": self.original_observations,
            "retainedObservations": self.retained_observations,
            "compoundedObservations": len(self.matrix),
            "blockSize": self.block_size,
            "droppedLeadingObservations": self.dropped_leading_observations,
            "sourceFirstKey": _json_safe_alignment_key(self.source_first_key),
            "sourceLastKey": _json_safe_alignment_key(self.source_last_key),
            "retainedFirstKey": _json_safe_alignment_key(self.retained_first_key),
            "droppedLeadingKeys": [
                _json_safe_alignment_key(value)
                for value in self.dropped_leading_keys
            ],
            "blockBoundaries": boundaries,
        }


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """Inputs and output of a Deflated Sharpe Ratio calculation.

    Sharpe values used in the test statistic are per-observation.  Annualized
    values are presentation-only; substituting annualized Sharpe into the DSR
    finite-sample formula would be incorrect.
    """

    selected_trial: Hashable
    probability: float
    test_statistic: float
    observations: int
    trials: int
    independent_trials: float
    selected_sharpe: float
    benchmark_sharpe: float
    selected_sharpe_annualized: float
    benchmark_sharpe_annualized: float
    trial_sharpe_mean: float
    trial_sharpe_std: float
    skewness: float
    kurtosis: float

    def to_dict(self) -> dict[str, object]:
        return {
            "method": "deflated_sharpe_ratio",
            "selectedTrial": self.selected_trial,
            "probability": self.probability,
            "testStatistic": self.test_statistic,
            "observations": self.observations,
            "trials": self.trials,
            "independentTrials": self.independent_trials,
            "selectedSharpePerPeriod": self.selected_sharpe,
            "benchmarkSharpePerPeriod": self.benchmark_sharpe,
            "selectedSharpeAnnualized": self.selected_sharpe_annualized,
            "benchmarkSharpeAnnualized": self.benchmark_sharpe_annualized,
            "trialSharpeMeanPerPeriod": self.trial_sharpe_mean,
            "trialSharpeStdPerPeriod": self.trial_sharpe_std,
            "selectedReturnSkewness": self.skewness,
            "selectedReturnPearsonKurtosis": self.kurtosis,
        }


@dataclass(frozen=True)
class CSCVPBOResult:
    """Result of a complete CSCV enumeration.

    ``relative_ranks`` rank the in-sample winner's out-of-sample Sharpe from
    worst to best.  PBO is the fraction of corresponding logits at or below
    zero.
    """

    probability: float
    observations: int
    trials: int
    slices: int
    slice_size: int
    splits: int
    logits: tuple[float, ...]
    relative_ranks: tuple[float, ...]
    selected_trials: tuple[Hashable, ...]
    selected_in_sample_sharpes: tuple[float, ...]
    selected_out_of_sample_sharpes: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "method": "cscv_probability_of_backtest_overfitting",
            "probability": self.probability,
            "observations": self.observations,
            "trials": self.trials,
            "slices": self.slices,
            "sliceSize": self.slice_size,
            "splits": self.splits,
            "logits": list(self.logits),
            "relativeRanks": list(self.relative_ranks),
            "selectedTrials": list(self.selected_trials),
            "selectedInSampleSharpes": list(self.selected_in_sample_sharpes),
            "selectedOutOfSampleSharpes": list(self.selected_out_of_sample_sharpes),
        }


def validate_return_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Validate and copy a complete observation-by-trial return matrix.

    The index is the alignment key (normally a timestamp).  Every trial must
    have one finite return at every key.  Callers must build the matrix from
    the full trial ledger, not only from surviving or reportable trials.
    """

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    if returns.shape[0] < 4:
        raise ValueError("return matrix needs at least four observations")
    if returns.shape[1] < 1:
        raise ValueError("return matrix needs at least one trial column")
    if not returns.index.is_unique:
        raise ValueError("return matrix index contains duplicate alignment keys")
    if not returns.index.is_monotonic_increasing:
        raise ValueError("return matrix index must be ordered chronologically")
    if not returns.columns.is_unique:
        raise ValueError("return matrix contains duplicate trial identifiers")

    try:
        values = returns.to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("all trial returns must be numeric") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError("return matrix must be complete and finite; no rows were dropped")

    return pd.DataFrame(values, index=returns.index.copy(), columns=returns.columns.copy())


def _json_safe_alignment_key(value: object) -> object:
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_safe_alignment_key(value.item())
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return value


def compound_return_matrix(
    returns: pd.DataFrame,
    block_size: int,
) -> CompoundedReturnMatrix:
    """Compound a complete matrix into non-overlapping full blocks.

    Only the leading ``len(returns) % block_size`` observations are removed, so
    the newest observation is always retained. Each output row is labelled by
    its block's ending alignment key. ``blocks`` records the inclusive source
    keys and original positional bounds (with ``stopPosition`` exclusive).
    """

    matrix = validate_return_matrix(returns)
    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)):
        raise TypeError("block_size must be a positive integer")
    block_size = int(block_size)
    if block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if block_size > len(matrix):
        raise ValueError("return matrix does not contain one complete block")

    dropped = len(matrix) % block_size
    retained = matrix.iloc[dropped:]
    block_count = len(retained) // block_size
    values = retained.to_numpy(dtype=float).reshape(
        block_count, block_size, matrix.shape[1]
    )
    with np.errstate(over="ignore", invalid="ignore"):
        compounded_values = np.prod(1.0 + values, axis=1) - 1.0
    if not np.all(np.isfinite(compounded_values)):
        raise ValueError("compounded returns are non-finite")

    ending_offsets = np.arange(block_size - 1, len(retained), block_size)
    compounded_index = pd.Index(
        retained.index.take(ending_offsets),
        name=matrix.index.name,
    )
    compounded = pd.DataFrame(
        compounded_values,
        index=compounded_index,
        columns=matrix.columns.copy(),
    )

    boundaries = []
    for block in range(block_count):
        start_position = dropped + block * block_size
        stop_position = start_position + block_size
        boundaries.append(
            {
                "block": block,
                "startPosition": start_position,
                "stopPosition": stop_position,
                "startKey": matrix.index[start_position],
                "endKey": matrix.index[stop_position - 1],
                "observations": block_size,
            }
        )

    return CompoundedReturnMatrix(
        matrix=compounded,
        blocks=pd.DataFrame(boundaries),
        original_observations=len(matrix),
        retained_observations=len(retained),
        block_size=block_size,
        dropped_leading_observations=dropped,
        source_first_key=matrix.index[0],
        source_last_key=matrix.index[-1],
        retained_first_key=retained.index[0],
        dropped_leading_keys=tuple(matrix.index[:dropped]),
    )


def return_matrix_from_long(
    returns: pd.DataFrame,
    *,
    timestamp_column: str = "timestamp",
    trial_column: str = "trial_id",
    return_column: str = "net_return",
) -> pd.DataFrame:
    """Pivot a long trial ledger without masking duplicates or missing trials."""

    required = {timestamp_column, trial_column, return_column}
    missing_columns = required.difference(returns.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"long return ledger is missing columns: {missing}")
    keys = returns[[timestamp_column, trial_column]]
    if keys.isna().any().any():
        raise ValueError("timestamp and trial identifiers cannot be missing")
    if keys.duplicated().any():
        raise ValueError("long return ledger has duplicate timestamp/trial rows")

    ordered_keys = pd.Index(returns[timestamp_column].drop_duplicates())
    if not ordered_keys.is_monotonic_increasing:
        raise ValueError("long return ledger timestamps must be ordered chronologically")

    matrix = returns.pivot(
        index=timestamp_column,
        columns=trial_column,
        values=return_column,
    )
    matrix = matrix.reindex(index=ordered_keys)
    matrix.columns.name = None
    return validate_return_matrix(matrix)


def _column_sharpes(values: np.ndarray) -> np.ndarray:
    means = np.mean(values, axis=0)
    stds = np.std(values, axis=0, ddof=1)
    sharpes = np.full_like(means, np.nan, dtype=float)
    variable = stds > 0.0
    sharpes[variable] = means[variable] / stds[variable]
    return sharpes


def _unbiased_skewness_and_kurtosis(values: np.ndarray) -> tuple[float, float]:
    """Bias-corrected sample skewness and Pearson kurtosis."""

    n = len(values)
    centered = values - np.mean(values)
    m2 = float(np.mean(centered**2))
    if m2 == 0.0:
        raise ValueError("selected trial has zero return variance")
    m3 = float(np.mean(centered**3))
    m4 = float(np.mean(centered**4))

    raw_skew = m3 / (m2**1.5)
    skew = sqrt(n * (n - 1)) / (n - 2) * raw_skew

    raw_excess = m4 / (m2 * m2) - 3.0
    excess = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * raw_excess + 6.0)
    return float(skew), float(excess + 3.0)


def expected_maximum_sharpe(
    trial_sharpes: np.ndarray,
    *,
    independent_trials: float | None = None,
    null_sharpe: float = 0.0,
) -> tuple[float, float]:
    """Return the DSR benchmark Sharpe and cross-trial Sharpe dispersion.

    The Gaussian order-statistic approximation assumes ``independent_trials``
    independent trials under a null with mean ``null_sharpe``.  Dependence must
    not be hidden: callers may supply a separately justified effective trial
    count, while the conservative default treats every matrix column as an
    independent trial.
    """

    estimates = np.asarray(trial_sharpes, dtype=float)
    if estimates.ndim != 1 or len(estimates) == 0:
        raise ValueError("trial_sharpes must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(estimates)):
        raise ValueError("every trial needs non-constant finite returns for DSR")
    if not isfinite(null_sharpe):
        raise ValueError("null_sharpe must be finite")

    count = float(len(estimates) if independent_trials is None else independent_trials)
    if not isfinite(count) or count < 1.0 or count > len(estimates):
        raise ValueError("independent_trials must be between one and the trial count")
    dispersion = float(np.std(estimates, ddof=1)) if len(estimates) > 1 else 0.0
    if count == 1.0 or dispersion == 0.0:
        return float(null_sharpe), dispersion

    first_quantile = _NORMAL.inv_cdf(1.0 - 1.0 / count)
    second_quantile = _NORMAL.inv_cdf(1.0 - 1.0 / (count * e))
    expected_max = null_sharpe + dispersion * (
        (1.0 - EULER_MASCHERONI) * first_quantile
        + EULER_MASCHERONI * second_quantile
    )
    return float(expected_max), dispersion


def deflated_sharpe_ratio(
    returns: pd.DataFrame,
    *,
    selected_trial: Hashable | None = None,
    periods_per_year: float = 1.0,
    independent_trials: float | None = None,
    null_sharpe_per_period: float = 0.0,
) -> DeflatedSharpeResult:
    """Compute the formal Deflated Sharpe Ratio from all attempted trials.

    If ``selected_trial`` is omitted, the full-sample Sharpe winner is tested.
    ``independent_trials`` defaults to the number of matrix columns.  Reducing
    it is valid only when an effective independent-trial count has been
    estimated outside this function.
    """

    matrix = validate_return_matrix(returns)
    if not isfinite(periods_per_year) or periods_per_year <= 0.0:
        raise ValueError("periods_per_year must be positive and finite")

    values = matrix.to_numpy(dtype=float)
    trial_sharpes = _column_sharpes(values)
    if not np.all(np.isfinite(trial_sharpes)):
        raise ValueError("every trial needs non-constant finite returns for DSR")

    if selected_trial is None:
        selected_position = int(np.argmax(trial_sharpes))
        selected_trial = matrix.columns[selected_position]
    else:
        matching = np.flatnonzero(matrix.columns == selected_trial)
        if len(matching) != 1:
            raise KeyError(f"unknown selected trial: {selected_trial!r}")
        selected_position = int(matching[0])

    count = float(matrix.shape[1] if independent_trials is None else independent_trials)
    benchmark, dispersion = expected_maximum_sharpe(
        trial_sharpes,
        independent_trials=count,
        null_sharpe=null_sharpe_per_period,
    )
    selected_sharpe = float(trial_sharpes[selected_position])
    skewness, kurtosis = _unbiased_skewness_and_kurtosis(values[:, selected_position])
    denominator_sq = (
        1.0
        - skewness * selected_sharpe
        + ((kurtosis - 1.0) / 4.0) * selected_sharpe**2
    )
    if not isfinite(denominator_sq) or denominator_sq <= 0.0:
        raise ValueError("DSR sampling-variance estimate is not positive")
    statistic = (selected_sharpe - benchmark) * sqrt(matrix.shape[0] - 1) / sqrt(
        denominator_sq
    )
    probability = _NORMAL.cdf(statistic)
    annualizer = sqrt(periods_per_year)

    return DeflatedSharpeResult(
        selected_trial=selected_trial,
        probability=float(probability),
        test_statistic=float(statistic),
        observations=matrix.shape[0],
        trials=matrix.shape[1],
        independent_trials=count,
        selected_sharpe=selected_sharpe,
        benchmark_sharpe=benchmark,
        selected_sharpe_annualized=selected_sharpe * annualizer,
        benchmark_sharpe_annualized=benchmark * annualizer,
        trial_sharpe_mean=float(np.mean(trial_sharpes)),
        trial_sharpe_std=dispersion,
        skewness=skewness,
        kurtosis=kurtosis,
    )


def _average_rank_from_worst(scores: np.ndarray, selected: int) -> float:
    selected_score = scores[selected]
    below = int(np.count_nonzero(scores < selected_score))
    tied = int(np.count_nonzero(scores == selected_score))
    return below + (tied + 1.0) / 2.0


def cscv_pbo(
    returns: pd.DataFrame,
    *,
    n_slices: int = 16,
    max_splits: int = 100_000,
) -> CSCVPBOResult:
    """Compute CSCV Probability of Backtest Overfitting using Sharpe ranking.

    Observations are divided into equal contiguous slices.  Every combination
    of half the slices is an in-sample set and its exact complement is the
    out-of-sample set. No tail observations are discarded, so the observation
    count must be divisible by ``n_slices``. In-sample ties use stable column
    order; out-of-sample ties receive their average rank.
    """

    matrix = validate_return_matrix(returns)
    if isinstance(n_slices, bool) or not isinstance(n_slices, int):
        raise TypeError("n_slices must be an even integer")
    if n_slices < 2 or n_slices % 2 != 0:
        raise ValueError("n_slices must be an even integer of at least two")
    if n_slices > matrix.shape[0]:
        raise ValueError("n_slices cannot exceed the observation count")
    if matrix.shape[0] % n_slices != 0:
        raise ValueError("CSCV requires equal slices; observations must divide n_slices")
    if matrix.shape[1] < 2:
        raise ValueError("CSCV PBO requires at least two trials")
    if isinstance(max_splits, bool) or not isinstance(max_splits, int) or max_splits < 1:
        raise ValueError("max_splits must be a positive integer")

    split_count = comb(n_slices, n_slices // 2)
    if split_count > max_splits:
        raise ValueError(
            f"CSCV would enumerate {split_count} splits, above max_splits={max_splits}"
        )

    values = matrix.to_numpy(dtype=float)
    slice_size = matrix.shape[0] // n_slices
    slices = np.arange(matrix.shape[0]).reshape(n_slices, slice_size)
    all_slice_ids = set(range(n_slices))
    logits: list[float] = []
    ranks: list[float] = []
    selected_trials: list[Hashable] = []
    selected_in_sample_sharpes: list[float] = []
    selected_out_of_sample_sharpes: list[float] = []

    for in_slice_ids in combinations(range(n_slices), n_slices // 2):
        out_slice_ids = sorted(all_slice_ids.difference(in_slice_ids))
        in_rows = slices[list(in_slice_ids)].reshape(-1)
        out_rows = slices[out_slice_ids].reshape(-1)
        in_scores = _column_sharpes(values[in_rows])
        out_scores = _column_sharpes(values[out_rows])
        if not np.all(np.isfinite(in_scores)) or not np.all(np.isfinite(out_scores)):
            raise ValueError(
                "every trial needs non-constant returns in every CSCV half-sample"
            )
        selected = int(np.argmax(in_scores))
        rank = _average_rank_from_worst(out_scores, selected)
        relative_rank = rank / (matrix.shape[1] + 1.0)
        logit = log(relative_rank / (1.0 - relative_rank))

        ranks.append(float(relative_rank))
        logits.append(float(logit))
        selected_trials.append(matrix.columns[selected])
        selected_in_sample_sharpes.append(float(in_scores[selected]))
        selected_out_of_sample_sharpes.append(float(out_scores[selected]))

    probability = float(np.mean(np.asarray(logits) <= 0.0))
    return CSCVPBOResult(
        probability=probability,
        observations=matrix.shape[0],
        trials=matrix.shape[1],
        slices=n_slices,
        slice_size=slice_size,
        splits=split_count,
        logits=tuple(logits),
        relative_ranks=tuple(ranks),
        selected_trials=tuple(selected_trials),
        selected_in_sample_sharpes=tuple(selected_in_sample_sharpes),
        selected_out_of_sample_sharpes=tuple(selected_out_of_sample_sharpes),
    )

"""Disabled exact OPE composition over successor-learner runs; no collection, statistical acceptance or production caller."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
import causal_ppo_v7 as ppo
import causal_value_v6 as value
from causal_transition_v4 import Transition
from ope_rational_v2 import VERSION as ESTIMATOR, Episode, Estimate, estimate_v2

VERSION = "causal-ope-v8"
FLOOR = 2.0 ** -10
GAMMA = 0.99
HORIZON = 32
RUNS = 256
LAW_GAP = F(1, 2 ** 50)
WEIGHT_GAP = F(1, 2 ** 34)
LEARNERS = {value.TrainingResult: value.VERSION, ppo.TrainingResult: ppo.VERSION}
VALUE_LAW = (value.GREEDY, value.OTHER)
__all__ = ["VERSION", "Model", "Evaluation", "evaluate_causal_ope_v8"]


@dataclass(frozen=True, slots=True)
class Model:
    target: tuple
    q: tuple
    v: tuple


@dataclass(frozen=True, slots=True)
class Evaluation:
    version: str
    estimator_version: str
    gamma: float
    horizon: int
    sources: tuple[tuple[str, int], ...]
    behavior_floor: float
    law_gap: F
    weight_gap: F
    estimate: Estimate
    reliable: bool = False


def _behavior(result: object, horizon: int) -> tuple:
    recorded = result.behavior_probabilities[:horizon]
    if type(result) is value.TrainingResult:
        # The epsilon-greedy law is exactly 11/12 or 1/24; only the conversion rounds.
        if not all(type(p) is F and p in VALUE_LAW for p in recorded):
            raise ValueError("unrecognized value-learner behavior probability")
        recorded = tuple(float(p) for p in recorded)
    if not all(type(p) is float and FLOOR <= p <= 1.0 for p in recorded):
        raise ValueError("behavior probability below the evaluation floor")
    return recorded


def _episode(result: object, model: object, horizon: int) -> Episode:
    if (type(result) not in LEARNERS or type(result.version) is not str or
        result.version != LEARNERS[type(result)] or type(result.seed) is not int or
        type(result.transitions) is not tuple or type(result.behavior_probabilities) is not tuple or
        len(result.transitions) != len(result.behavior_probabilities)):
        raise ValueError("unrecognized learner result")
    # No run is dropped or padded: a run shorter than the horizon rejects the whole evaluation.
    if len(result.transitions) < horizon:
        raise ValueError("run shorter than the horizon")
    rows = result.transitions[:horizon]
    if (not all(type(row) is Transition and type(row.action) is int and row.action in (0, 1, 2) for row in rows) or
        any(row.done is not False for row in rows[:-1])):
        raise ValueError("invalid transition")
    if (type(model) is not Model or type(model.target) is not tuple or type(model.q) is not tuple or
        type(model.v) is not tuple or (len(model.target), len(model.q), len(model.v)) != (horizon, horizon, horizon + 1) or
        not all(type(p) is float and 0.0 <= p <= 1.0 for p in model.target)):
        raise ValueError("invalid target model")
    q = tuple(ppo._bounded(x) for x in model.q)
    v = tuple(ppo._bounded(x) for x in model.v)
    if v[-1] != 0.0:
        raise ValueError("nonzero value beyond the horizon")
    rewards = tuple(ppo._bounded(ppo._number(row.reward)) for row in rows)
    return Episode(rewards, tuple(row.action for row in rows), _behavior(result, horizon), model.target, q, v)


def evaluate_causal_ope_v8(results: object, models: object, horizon: object, *,
                           enabled: object = False, version: object = VERSION) -> Evaluation | None:
    if (enabled is not True or type(version) is not str or version != VERSION or
        type(results) is not tuple or type(models) is not tuple or
        not 1 <= len(results) <= RUNS or len(models) != len(results) or
        type(horizon) is not int or not 1 <= horizon <= HORIZON):
        return None
    try:
        episodes = tuple(_episode(result, model, horizon) for result, model in zip(results, models))
        sources = tuple((result.version, result.seed) for result in results)
        # A learner's generator stream is a function of (version, seed): reuse is not a second sample.
        if len(set(sources)) != len(sources):
            raise ValueError("reused generator stream")
        estimate = estimate_v2(episodes, GAMMA, enabled=True)
        if estimate is None or len(estimate.episodes) != len(results):
            raise ValueError("rejected estimate")
        return Evaluation(VERSION, ESTIMATOR, GAMMA, horizon, sources, FLOOR, LAW_GAP, WEIGHT_GAP, estimate)
    except (ArithmeticError, ValueError, TypeError, MemoryError):
        return None

"""Disabled exact OPE arithmetic; no data collection or statistical acceptance."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite

VERSION = "ope-rational-v2"
MAX_EPISODES = 256
MAX_DECISIONS = 32
MAX_BITS = 8192


@dataclass(frozen=True, slots=True)
class Episode:
    rewards: tuple
    actions: tuple
    behavior: tuple
    target: tuple
    q: tuple
    v: tuple


@dataclass(frozen=True, slots=True)
class EpisodeEstimate:
    discounted_return: Fraction
    weight: Fraction
    ordinary_is: Fraction
    per_decision_is: Fraction
    doubly_robust: Fraction


@dataclass(frozen=True, slots=True)
class Estimate:
    version: str
    ordinary_is: Fraction
    per_decision_is: Fraction
    weighted_is: Fraction | None
    doubly_robust: Fraction
    effective_sample_size: Fraction
    max_weight: Fraction
    nonzero_trajectories: int
    horizon: int
    episodes: tuple[EpisodeEstimate, ...]
    reliable: bool = False


def _bounded(value: Fraction) -> Fraction:
    if value.numerator.bit_length() > MAX_BITS or value.denominator.bit_length() > MAX_BITS:
        raise ArithmeticError("OPE rational size limit")
    return value


def _add(a: Fraction, b: Fraction) -> Fraction:
    return _bounded(a + b)


def _mul(a: Fraction, b: Fraction) -> Fraction:
    return _bounded(a * b)


def _div(a: Fraction, b: Fraction) -> Fraction:
    return _bounded(a / b)


def _episode_valid(episode: object, horizon: int) -> bool:
    if type(episode) is not Episode:
        return False
    fields = (episode.rewards, episode.behavior, episode.target, episode.q, episode.v)
    if any(type(xs) is not tuple for xs in fields) or type(episode.actions) is not tuple:
        return False
    if any(len(xs) != horizon for xs in fields[:-1]) or len(episode.v) != horizon + 1:
        return False
    if len(episode.actions) != horizon or any(type(a) is not int or a not in (0, 1, 2) for a in episode.actions):
        return False
    if not all(type(x) is float and isfinite(x) for xs in fields for x in xs):
        return False
    return (all(0 < b <= 1 for b in episode.behavior) and
            all(0 <= p <= 1 for p in episode.target) and episode.v[-1] == 0)


def _episode(episode: Episode, gamma: Fraction) -> EpisodeEstimate:
    weight = Fraction(1)
    discount = Fraction(1)
    total = Fraction(0)
    pdis = Fraction(0)
    dr = Fraction.from_float(episode.v[0])
    for t in range(len(episode.rewards)):
        reward, behavior, target, q, next_v = (
            Fraction.from_float(xs[t]) for xs in
            (episode.rewards, episode.behavior, episode.target, episode.q, episode.v[1:]))
        weight = _mul(weight, _div(target, behavior))
        total = _add(total, _mul(discount, reward))
        pdis = _add(pdis, _mul(_mul(weight, discount), reward))
        residual = _add(_add(reward, _mul(gamma, next_v)), -q)
        dr = _add(dr, _mul(_mul(weight, discount), residual))
        discount = _mul(discount, gamma)
    return EpisodeEstimate(total, weight, _mul(weight, total), pdis, dr)


def estimate_v2(episodes: object, gamma: object, *, enabled: object = False,
                version: object = VERSION) -> Estimate | None:
    if enabled is not True or type(version) is not str or version != VERSION:
        return None
    if type(episodes) is not tuple or not 1 <= len(episodes) <= MAX_EPISODES:
        return None
    if type(gamma) is not float or not isfinite(gamma) or not 0 <= gamma <= 1:
        return None
    if type(episodes[0]) is not Episode or type(episodes[0].rewards) is not tuple:
        return None
    horizon = len(episodes[0].rewards)
    if not 1 <= horizon <= MAX_DECISIONS:
        return None
    if not all(_episode_valid(episode, horizon) for episode in episodes):
        return None
    try:
        discount = Fraction.from_float(gamma)
        estimates = []
        weights = squares = ordinary = per_decision = robust = Fraction(0)
        for episode in episodes:
            value = _episode(episode, discount)
            estimates.append(value)
            weights = _add(weights, value.weight)
            squares = _add(squares, _mul(value.weight, value.weight))
            ordinary = _add(ordinary, value.ordinary_is)
            per_decision = _add(per_decision, value.per_decision_is)
            robust = _add(robust, value.doubly_robust)
        count = Fraction(len(estimates))
        wis = _div(ordinary, weights) if weights > 0 else None
        ess = _div(_mul(weights, weights), squares) if squares > 0 else Fraction(0)
        return Estimate(VERSION, _div(ordinary, count), _div(per_decision, count), wis,
                        _div(robust, count), ess, max(v.weight for v in estimates),
                        sum(v.weight > 0 for v in estimates), horizon, tuple(estimates))
    except (ArithmeticError, ValueError, TypeError, MemoryError):
        return None

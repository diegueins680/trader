"""Disabled, isolated raw GAE targets. No normalization or training integration."""
from __future__ import annotations

from math import isfinite

VERSION = "gae-targets-v2"
MAX_ROWS = 256


def step_v2(reward: object, value: object, next_value: object,
            later: object, done: object, gamma: object, *,
            enabled: object = False, version: object = VERSION) -> tuple[float, float] | None:
    if enabled is not True or type(version) is not str or version != VERSION:
        return None
    if type(done) is not bool or not all(type(x) is float and isfinite(x)
                                       for x in (reward, value, next_value, later, gamma)):
        return None
    if not 0.0 <= gamma <= 1.0:
        return None
    if done:
        raw = reward - value
        target = reward
    else:
        bootstrap = gamma * next_value
        delta = (reward + bootstrap) - value
        trace = (gamma * 0.95) * later
        raw = delta + trace
        target = raw + value
    if not all(isfinite(x) for x in (raw, target)):
        return None
    return raw, target


def batch_v2(rows: object, gamma: object, *, enabled: object = False,
             version: object = VERSION) -> tuple[tuple[float, float], ...] | None:
    if enabled is not True or type(version) is not str or version != VERSION:
        return None
    if type(rows) is not tuple or not 1 <= len(rows) <= MAX_ROWS:
        return None
    staged = []
    later = 0.0
    for row in reversed(rows):
        if type(row) is not tuple or len(row) != 4:
            return None
        reward, value, next_value, done = row
        pair = step_v2(reward, value, next_value, later, done, gamma,
                       enabled=True, version=version)
        if pair is None:
            return None
        staged.append(pair)
        later = pair[0]
    return tuple(reversed(staged))

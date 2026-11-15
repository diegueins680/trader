"""Disabled, isolated exact ESS diagnostic; no estimator or trading integration."""
from __future__ import annotations

from fractions import Fraction
from math import isfinite

VERSION = "ess-rational-v2"
MAX_ROWS = 256


def effective_sample_size_v2(weights: object, *, enabled: object = False,
                             version: object = VERSION) -> Fraction | None:
    if enabled is not True or type(version) is not str or version != VERSION:
        return None
    if type(weights) is not tuple or not 1 <= len(weights) <= MAX_ROWS:
        return None
    if not all(type(value) is float and isfinite(value) and value >= 0 for value in weights):
        return None
    total = Fraction(0)
    squares = Fraction(0)
    for value in weights:
        weight = Fraction.from_float(value)
        total = total + weight
        squares = squares + weight * weight
    return Fraction(0) if squares == 0 else total * total / squares

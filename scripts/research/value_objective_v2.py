"""Disabled finite, shift-stable conservative Double-DQN objective. No learner, data or order interface."""
from __future__ import annotations

from dataclasses import dataclass
import math

VERSION = "value-objective-v2"
ACTIONS = 3
MAX_ROWS = 256
BOUND = 2.0 ** 100
TINY = 2.0 ** -1022
__all__ = ["VERSION", "Objective", "objective_v2"]


@dataclass(frozen=True, slots=True)
class Objective:
    version: str
    loss: float
    grad: tuple
    underflow: int
    rows: int


def _value(x: object) -> bool:
    return type(x) is float and math.isfinite(x) and abs(x) <= BOUND


def _admissible(q: object, actions: object, targets: object, alpha: object) -> bool:
    if (type(q) is not tuple or type(actions) is not tuple or type(targets) is not tuple or
        not 1 <= len(q) <= MAX_ROWS or len(actions) != len(q) or len(targets) != len(q)):
        return False
    if type(alpha) is not float or not 0.0 <= alpha <= 1.0:
        return False
    for row, action, target in zip(q, actions, targets):
        if type(row) is not tuple or len(row) != ACTIONS or not all(_value(x) for x in row):
            return False
        if type(action) is not int or not 0 <= action < ACTIONS or not _value(target):
            return False
    return True


def _penalty(row: tuple, action: int) -> tuple:
    # Differences only: never form m + log(...) and subtract a large selected value.
    m = max(row)
    terms = [math.exp(x - m) for x in row]
    underflow = sum(1 for t in terms if t < TINY)
    total = math.fsum(terms)
    gap = (m - row[action]) + math.log(total)
    return gap, tuple(t / total for t in terms), underflow


def _compute(q: tuple, actions: tuple, targets: tuple, alpha: float) -> Objective | None:
    n = len(q)
    residuals = [row[a] - y for row, a, y in zip(q, actions, targets)]
    loss = math.fsum(e * e for e in residuals) / (2 * n)
    grad = [[(e / n if j == a else 0.0) for j in range(ACTIONS)] for e, a in zip(residuals, actions)]
    underflow = 0
    if alpha > 0.0:
        gaps = []
        for i, (row, a) in enumerate(zip(q, actions)):
            gap, probs, lost = _penalty(row, a)
            gaps.append(gap)
            underflow += lost
            for j in range(ACTIONS):
                grad[i][j] += alpha * (probs[j] - (1.0 if j == a else 0.0)) / n
        loss += alpha * math.fsum(gaps) / n
    # x + 0.0 maps -0.0 to +0.0 under RNE and is the identity otherwise (CE-RL-024).
    loss += 0.0
    published = tuple(tuple(g + 0.0 for g in r) for r in grad)
    if not math.isfinite(loss) or not all(math.isfinite(g) for r in published for g in r):
        return None
    return Objective(VERSION, loss, published, underflow, n)


def objective_v2(q: object, actions: object, targets: object, alpha: object, *,
                 enabled: object = False, version: object = VERSION) -> Objective | None:
    if enabled is not True or type(version) is not str or version != VERSION:
        return None
    try:
        if not _admissible(q, actions, targets, alpha):
            return None
        return _compute(q, actions, targets, alpha)
    except (ArithmeticError, ValueError, TypeError, MemoryError):
        return None

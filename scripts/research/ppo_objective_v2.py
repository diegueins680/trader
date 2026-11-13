"""Disabled finite clipped-PPO objective. No learner, data or order interface."""
from __future__ import annotations

from dataclasses import dataclass
import math

VERSION = "ppo-objective-v2"
ACTIONS = 3
MAX_ROWS = 256
BOUND = 2.0 ** 100
MIN_PROB = 2.0 ** -100
TINY = 2.0 ** -1022
LOW = 0.8
HIGH = 1.2
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


def _admissible(logits: object, actions: object, old_prob: object, advantage: object) -> bool:
    if (type(logits) is not tuple or type(actions) is not tuple or type(old_prob) is not tuple or
        type(advantage) is not tuple or not 1 <= len(logits) <= MAX_ROWS or
        len(actions) != len(logits) or len(old_prob) != len(logits) or len(advantage) != len(logits)):
        return False
    for row, action, old, adv in zip(logits, actions, old_prob, advantage):
        if type(row) is not tuple or len(row) != ACTIONS or not all(_value(x) for x in row):
            return False
        if type(action) is not int or not 0 <= action < ACTIONS or not _value(adv):
            return False
        if type(old) is not float or not MIN_PROB <= old <= 1.0:
            return False
    return True


def _lost(a: float, b: float, product: float) -> int:
    return 1 if a != 0.0 and b != 0.0 and abs(product) < TINY else 0


def _row(row: tuple, action: int, old: float, adv: float, n: int) -> tuple:
    # Differences only: the softmax never forms a large exponent.
    m = max(row)
    terms = [math.exp(x - m) for x in row]
    underflow = sum(1 for t in terms if t < TINY)
    total = math.fsum(terms)
    probs = [t / total for t in terms]
    ratio = probs[action] / old
    clipped = min(max(ratio, LOW), HIGH)
    plain = ratio * adv
    bounded = clipped * adv
    underflow += _lost(ratio, adv, plain) + _lost(clipped, adv, bounded)
    active = (adv >= 0.0 and ratio <= HIGH) or (adv < 0.0 and ratio >= LOW)
    # Branch selection, not multiplication by a flag: an inactive row is exactly zero.
    coefficient = plain / n if active else 0.0
    if active and plain != 0.0 and abs(coefficient) < TINY:
        underflow += 1
    grad = []
    for j, p in enumerate(probs):
        delta = p - (1.0 if j == action else 0.0)
        g = delta * coefficient
        underflow += _lost(delta, coefficient, g)
        grad.append(g)
    return min(plain, bounded), grad, underflow


def _compute(logits: tuple, actions: tuple, old_prob: tuple, advantage: tuple) -> Objective | None:
    n = len(logits)
    terms, grads, underflow = [], [], 0
    for row, action, old, adv in zip(logits, actions, old_prob, advantage):
        term, grad, lost = _row(row, action, old, adv, n)
        terms.append(term)
        grads.append(grad)
        underflow += lost
    loss = -math.fsum(terms) / n
    # x + 0.0 maps -0.0 to +0.0 under RNE and is the identity otherwise.
    loss += 0.0
    published = tuple(tuple(g + 0.0 for g in r) for r in grads)
    if not math.isfinite(loss) or not all(math.isfinite(g) for r in published for g in r):
        return None
    return Objective(VERSION, loss, published, underflow, n)


def objective_v2(logits: object, actions: object, old_prob: object, advantage: object, *,
                 enabled: object = False, version: object = VERSION) -> Objective | None:
    if enabled is not True or type(version) is not str or version != VERSION:
        return None
    try:
        if not _admissible(logits, actions, old_prob, advantage):
            return None
        return _compute(logits, actions, old_prob, advantage)
    except (ArithmeticError, ValueError, TypeError, MemoryError):
        return None

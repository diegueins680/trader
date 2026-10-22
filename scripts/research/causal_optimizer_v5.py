"""Disabled, functional nine-input optimizer. Snapshots carry no order authority."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from causal_replay_v3 import OBSERVATION

VERSION = "causal-optimizer-v5"
STEP_CAP = 2**31 - 1
__all__ = ["VERSION", "Snapshot", "create_v5", "forward_v5", "update_v5"]


@dataclass(frozen=True, slots=True)
class Snapshot:
    version: str
    observation_version: str
    outputs: int
    step: int
    p: tuple[bytes, ...]
    m: tuple[bytes, ...]
    v: tuple[bytes, ...]


def _shapes(outputs: int) -> tuple:
    return ((9, 16), (16,), (16, outputs), (outputs,))


def _admit(snapshot: object) -> bool:
    if (type(snapshot) is not Snapshot or type(snapshot.version) is not str or
        snapshot.version != VERSION or type(snapshot.observation_version) is not str or
        snapshot.observation_version != OBSERVATION or type(snapshot.outputs) is not int or
        snapshot.outputs not in (1, 3) or type(snapshot.step) is not int or
        not 0 <= snapshot.step <= STEP_CAP):
        return False
    for index, group in enumerate((snapshot.p, snapshot.m, snapshot.v)):
        if type(group) is not tuple or len(group) != 4:
            return False
        for raw, shape in zip(group, _shapes(snapshot.outputs)):
            if type(raw) is not bytes or len(raw) != math.prod(shape) * 8:
                return False
            values = np.frombuffer(raw, dtype="<f8")
            if not np.isfinite(values).all() or (index == 2 and np.any(values < 0)):
                return False
    return True


def _arrays(group: tuple, outputs: int) -> tuple:
    return tuple(np.frombuffer(raw, dtype="<f8").reshape(shape).copy()
                 for raw, shape in zip(group, _shapes(outputs)))


def _pack(outputs: int, step: int, p: tuple, m: tuple, v: tuple) -> Snapshot | None:
    groups = tuple(tuple(value.astype("<f8").tobytes() for value in group) for group in (p, m, v))
    snapshot = Snapshot(VERSION, OBSERVATION, outputs, step, *groups)
    return snapshot if _admit(snapshot) else None


def _batch(value: object, width: int) -> np.ndarray:
    if (type(value) is not np.ndarray or value.dtype != np.dtype("float64") or
        value.ndim != 2 or value.shape[1] != width or not 1 <= len(value) <= 256):
        raise ValueError("invalid numeric batch")
    copied = value.copy()
    if not np.isfinite(copied).all():
        raise ValueError("non-finite numeric batch")
    return copied


def create_v5(seed: object, outputs: object = 3, *, enabled: object = False,
              version: object = VERSION) -> Snapshot | None:
    if (enabled is not True or type(version) is not str or version != VERSION or
        type(seed) is not int or not 0 <= seed < 2**32 or
        type(outputs) is not int or outputs not in (1, 3)):
        return None
    try:
        with np.errstate(all="raise"):
            rng = np.random.default_rng(seed)
            p = (rng.normal(0, 1 / np.sqrt(9), (9, 16)), np.zeros(16),
                 rng.normal(0, 0.01, (16, outputs)), np.zeros(outputs))
            zeros = tuple(np.zeros_like(value) for value in p)
            return _pack(outputs, 0, p, zeros, zeros)
    except (ArithmeticError, ValueError, TypeError, MemoryError):
        return None


def _evaluate(p: tuple, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w1, b1, w2, b2 = p
    preactivation = x @ w1 + b1
    if not np.isfinite(preactivation).all():
        raise ValueError("non-finite hidden state")
    hidden = np.tanh(preactivation)
    output = hidden @ w2 + b2
    if not np.isfinite(output).all():
        raise ValueError("non-finite output")
    return hidden, output


def forward_v5(snapshot: object, x: object, *, enabled: object = False,
               version: object = VERSION) -> np.ndarray | None:
    if enabled is not True or type(version) is not str or version != VERSION:
        return None
    try:
        with np.errstate(all="raise"):
            if not _admit(snapshot):
                return None
            return _evaluate(_arrays(snapshot.p, snapshot.outputs), _batch(x, 9))[1]
    except (ArithmeticError, ValueError, TypeError, MemoryError):
        return None


def update_v5(snapshot: object, x: object, dz: object, lr: object = 0.0003, *,
              enabled: object = False, version: object = VERSION) -> Snapshot | None:
    if (enabled is not True or type(version) is not str or version != VERSION or
        type(lr) is not float or not math.isfinite(lr) or not 0 < lr <= 1):
        return None
    try:
        with np.errstate(all="raise"):
            if not _admit(snapshot) or snapshot.step >= STEP_CAP:
                return None
            inputs = _batch(x, 9)
            residuals = _batch(dz, snapshot.outputs)
            if len(inputs) != len(residuals):
                return None
            p = _arrays(snapshot.p, snapshot.outputs)
            m0 = _arrays(snapshot.m, snapshot.outputs)
            v0 = _arrays(snapshot.v, snapshot.outputs)
            hidden, _ = _evaluate(p, inputs)
            dh = (residuals @ p[2].T) * (1 - hidden**2)
            grads = (inputs.T @ dh, dh.sum(0), hidden.T @ residuals, residuals.sum(0))
            if not all(np.isfinite(g).all() for g in grads):
                return None
            norm = np.sqrt(sum(np.sum(g**2) for g in grads))
            if not np.isfinite(norm):
                return None
            step = snapshot.step + 1
            candidates, moments, variances = [], [], []
            for old, m, v, grad in zip(p, m0, v0, grads):
                g = grad / max(1.0, float(norm))
                moment = 0.9 * m + 0.1 * g
                variance = 0.999 * v + 0.001 * g**2
                delta = lr * (moment / (1 - 0.9**step)) / (np.sqrt(variance / (1 - 0.999**step)) + 1e-8)
                candidates.append(old - delta)
                moments.append(moment)
                variances.append(variance)
            return _pack(snapshot.outputs, step, tuple(candidates), tuple(moments), tuple(variances))
    except (ArithmeticError, ValueError, TypeError, MemoryError):
        return None

"""Default-disabled, bounded offline PPO training with immutable publication.

Engineering successor only. No persistence, production caller, order capability,
held-out evaluation or inference deadline claim. Frozen v1 stays unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from gae_targets_v2 import batch_v2
from optimizer_snapshot_v2 import Snapshot, create_v2, forward_v2, update_v2
from sequential_env import Scale, collect
from sequential_learning import ppo_gradient, softmax

VERSION = "ppo-successor-v2"
__all__ = ["VERSION", "TrainingResult", "train_ppo_v2"]


@dataclass(frozen=True, slots=True)
class TrainingResult:
    version: str
    seed: int
    horizon: int
    steps: int
    symbols: tuple[str, ...]
    scale: tuple[bytes, ...]
    actor: Snapshot
    critic: Snapshot
    losses: tuple[float, ...]


def _configuration(horizon: object, seed: object, steps: object) -> bool:
    return (type(horizon) is int and horizon in (1, 3, 6) and
            type(seed) is int and 0 <= seed <= 2**32 - 10000 and
            type(steps) is int and 1 <= steps <= 4096)


def _array(value: object, shape: tuple[int, ...]) -> bool:
    return (type(value) is np.ndarray and value.dtype == np.dtype("float64") and
            value.shape == shape and bool(np.isfinite(value).all()))


def _prefixes(prices: object, funding: object) -> tuple[dict, dict]:
    if (type(prices) is not dict or type(funding) is not dict or
        not 1 <= len(prices) <= 8 or set(prices) != set(funding) or
        any(type(s) is not str or not 1 <= len(s) <= 32 for s in prices)):
        raise ValueError("unsupported training symbol scope")
    copied = [{}, {}]
    for symbol in sorted(prices):
        p, f = prices[symbol], funding[symbol]
        if type(p) is not np.ndarray or p.ndim != 1 or not 121 <= len(p) <= 4096:
            raise ValueError("unsupported training prefix")
        if not _array(p, p.shape) or not _array(f, p.shape) or np.any(p <= 0):
            raise ValueError("invalid price or funding prefix")
        for target, value in zip(copied, (p, f)):
            target[symbol] = np.frombuffer(value.tobytes(), dtype="float64")
    return copied[0], copied[1]


def _forward(net: object, value: np.ndarray) -> np.ndarray:
    result = forward_v2(net, value, enabled=True)
    if result is None:
        raise ValueError("rejected network evaluation")
    return result


def _rollout(data: dict, count: int) -> None:
    for key, shape in (("s", (count, 12)), ("next", (count, 12)),
                       ("r", (count,)), ("prob", (count,))):
        if not _array(data[key], shape):
            raise ValueError("invalid rollout numeric field")
    if (type(data["a"]) is not np.ndarray or data["a"].dtype.kind not in "iu" or
        data["a"].shape != (count,) or np.any(data["a"] < 0) or np.any(data["a"] > 2) or
        type(data["done"]) is not np.ndarray or data["done"].dtype != np.dtype("bool") or
        data["done"].shape != (count,) or np.any(data["prob"] <= 0) or np.any(data["prob"] > 1)):
        raise ValueError("invalid rollout action, terminal or probability field")


def _targets(data: dict, critic: object, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    values = _forward(critic, data["s"]).ravel()
    following = _forward(critic, data["next"]).ravel()
    rows = tuple((float(r), float(v), float(n), bool(d))
                 for r, v, n, d in zip(data["r"], values, following, data["done"]))
    pairs = batch_v2(rows, gamma, enabled=True)
    if pairs is None:
        raise ValueError("rejected GAE targets")
    advantage, targets = np.asarray(pairs, dtype="float64").T
    advantage = (advantage - advantage.mean()) / max(advantage.std(), 1e-8)
    if not _array(advantage, values.shape) or not _array(targets, values.shape):
        raise ValueError("non-finite normalized GAE")
    return advantage, targets


def _update(actor: object, critic: object, data: dict, adv: np.ndarray,
            targets: np.ndarray, count: int) -> float:
    loss, gradient = ppo_gradient(_forward(actor, data["s"]), data["a"], data["prob"], adv)
    if type(loss) is not float or not np.isfinite(loss) or not _array(gradient, (count, 3)):
        raise ValueError("non-finite PPO objective")
    if update_v2(actor, data["s"], gradient, 0.0003, enabled=True) is None:
        raise ValueError("rejected actor update")
    residual = (_forward(critic, data["s"]).ravel() - targets) / count
    if not _array(residual, (count,)):
        raise ValueError("non-finite critic residual")
    if update_v2(critic, data["s"], residual[:, None], 0.0003, enabled=True) is None:
        raise ValueError("rejected critic update")
    return loss


def train_ppo_v2(prices: object, funding: object, horizon: object, seed: object,
                 steps: object = 4096, *, enabled: object = False,
                 version: object = VERSION) -> TrainingResult | None:
    if enabled is not True or type(version) is not str or version != VERSION:
        return None
    if not _configuration(horizon, seed, steps):
        return None
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            p, f = _prefixes(prices, funding)
            scale = Scale.fit(list(p.values()))
            actor = create_v2(seed, 3, enabled=True)
            critic = create_v2(seed + 1, 1, enabled=True)
            if actor is None or critic is None:
                return None

            def behavior(observation: np.ndarray) -> np.ndarray:
                return softmax(_forward(actor, observation[None, :]))[0]

            losses = []
            for batch in range((steps + 255) // 256):
                count = min(256, steps - batch * 256)
                data = collect(p, f, scale, horizon, seed + 1000 + batch, count, policy=behavior)
                _rollout(data, count)
                adv, targets = _targets(data, critic, float(0.99**horizon))
                for _ in range(4):
                    loss = _update(actor, critic, data, adv, targets, count)
                    losses.append(loss)
            frozen_scale = tuple(value.astype("<f8").tobytes() for value in
                                 (scale.mean, scale.std, scale.low, scale.high))
            return TrainingResult(VERSION, seed, horizon, steps, tuple(sorted(p)),
                                  frozen_scale, actor.snapshot(), critic.snapshot(), tuple(losses))
    except (ArithmeticError, ValueError, TypeError, MemoryError, KeyError):
        return None

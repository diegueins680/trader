"""Small original NumPy implementations of PPO, Double DQN and discrete CQL.

Mechanism prototypes, not reproductions of published benchmark scores. No
production imports, policy promotion, network or exchange authorization.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
from sequential_env import ACTIONS, ENVIRONMENT, FEATURE_COUNT, OBSERVATION, Execution, _finite_real, _integer, collect


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class Network:
    def __init__(self, seed: int, outputs: int = 3) -> None:
        rng = np.random.default_rng(seed)
        self.p = {"w1": rng.normal(0, 1 / np.sqrt(FEATURE_COUNT), (FEATURE_COUNT, 16)),
                  "b1": np.zeros(16), "w2": rng.normal(0, 0.01, (16, outputs)),
                  "b2": np.zeros(outputs)}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.steps = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                hidden = x @ self.p["w1"] + self.p["b1"]
                if not np.isfinite(hidden).all():
                    raise ValueError("non-finite network hidden state")
                out = np.tanh(hidden) @ self.p["w2"] + self.p["b2"]
                if not np.isfinite(out).all():
                    raise ValueError("non-finite network output")
                return out
        except FloatingPointError as exc:
            raise ValueError("non-finite network arithmetic") from exc

    def gradients(self, x: np.ndarray, dz: np.ndarray) -> dict:
        h = np.tanh(x @ self.p["w1"] + self.p["b1"])
        dh = (dz @ self.p["w2"].T) * (1 - h**2)
        return {"w1": x.T @ dh, "b1": dh.sum(0), "w2": h.T @ dz, "b2": dz.sum(0)}

    def update(self, x: np.ndarray, dz: np.ndarray, lr: float) -> None:
        if not _finite_real(lr) or lr <= 0 or not _integer(self.steps) or self.steps < 0:
            raise ValueError("invalid optimizer learning rate or step counter")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                grads = self.gradients(x, dz)
                if not all(np.isfinite(g).all() for g in grads.values()):
                    raise ValueError("non-finite gradient")
                norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
                if not np.isfinite(norm):
                    raise ValueError("non-finite gradient norm")
                steps = int(self.steps) + 1
                params, moments, variances = {}, {}, {}
                for k, grad in grads.items():
                    g = grad / max(1.0, norm)
                    moments[k] = 0.9 * self.m[k] + 0.1 * g
                    variances[k] = 0.999 * self.v[k] + 0.001 * g**2
                    m = moments[k] / (1 - 0.9**steps)
                    v = variances[k] / (1 - 0.999**steps)
                    params[k] = self.p[k] - lr * m / (np.sqrt(v) + 1e-8)
                if not all(np.isfinite(v).all() for state in (params, moments, variances)
                           for v in state.values()):
                    raise ValueError("non-finite optimizer state")
        except FloatingPointError as exc:
            raise ValueError("non-finite optimizer arithmetic") from exc
        # Publish only after every parameter and moment update succeeds.
        self.p, self.m, self.v, self.steps = params, moments, variances, steps

    def copy_from(self, other: Network) -> None:
        self.p = {k: v.copy() for k, v in other.p.items()}


def ppo_gradient(logits: np.ndarray, actions: np.ndarray, old_prob: np.ndarray,
                 advantage: np.ndarray) -> tuple[float, np.ndarray]:
    probs = softmax(logits)
    idx = np.arange(len(actions))
    ratio = probs[idx, actions] / old_prob
    clipped = np.clip(ratio, 0.8, 1.2)
    loss = -np.minimum(ratio * advantage, clipped * advantage).mean()
    active = ((advantage >= 0) & (ratio <= 1.2)) | ((advantage < 0) & (ratio >= 0.8))
    gradient = probs.copy()
    gradient[idx, actions] -= 1
    gradient *= (active * advantage * ratio / len(actions))[:, None]
    return float(loss), gradient


def advantages(data: dict, value: Network, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    v = value.forward(data["s"]).ravel()
    nxt = value.forward(data["next"]).ravel()
    delta = data["r"] + gamma * (~data["done"]) * nxt - v
    adv = np.empty(len(v))
    carry = 0.0
    for i in reversed(range(len(v))):
        carry = delta[i] + gamma * 0.95 * (not data["done"][i]) * carry
        adv[i] = carry
    targets = adv + v
    adv = (adv - adv.mean()) / max(adv.std(), 1e-8)
    return adv, targets


def _training_indices(horizon: int, seed: int, steps: int) -> tuple[int, int, int]:
    if (not all(_integer(v) for v in (horizon, seed, steps)) or
        horizon not in (1, 3, 6) or seed < 0 or steps <= 0):
        raise ValueError("invalid training horizon, seed or step budget")
    # Convert before seed offsets and batch arithmetic to avoid fixed-width wrapping.
    return int(horizon), int(seed), int(steps)


def _episode_totals(rollouts: list[dict]) -> dict:
    return {key: sum(row[key] for row in rollouts)
            for key in ("collections", "started", "completed", "truncated", "decisions")}


def train_ppo(prices, funding, scale, horizon: int, seed: int, steps: int = 4096):
    horizon, seed, steps = _training_indices(horizon, seed, steps)
    net, value = Network(seed), Network(seed + 1, 1)
    episodes, losses, rollouts = [], [], []
    for batch in range((steps + 255) // 256):
        count = min(256, steps - batch * 256)
        data = collect(prices, funding, scale, horizon, seed + 1000 + batch, count,
                       policy=lambda s: softmax(net.forward(s)))
        episodes.extend(data["episodes"])
        rollouts.append(data["episodeAccountingV2"])
        adv, targets = advantages(data, value, 0.99**horizon)
        for _ in range(4):
            loss, grad = ppo_gradient(net.forward(data["s"]), data["a"], data["prob"], adv)
            net.update(data["s"], grad, 0.0003)
            residual = value.forward(data["s"]).ravel() - targets
            value.update(data["s"], (residual / count)[:, None], 0.0003)
        losses.append(loss)
    return net, {"steps": steps, "updates": net.steps, "episodes": episodes, "losses": losses,
                 "episodeAccountingV2": _episode_totals(rollouts)}


def bellman_gradient(q: np.ndarray, actions: np.ndarray, targets: np.ndarray,
                     alpha: float) -> tuple[float, np.ndarray]:
    n, idx = len(actions), np.arange(len(actions))
    residual = q[idx, actions] - targets
    grad = np.zeros_like(q)
    grad[idx, actions] = residual / n
    m = q.max(1)
    logsumexp = m + np.log(np.exp(q - m[:, None]).sum(1))
    penalty = softmax(q)
    penalty[idx, actions] -= 1
    grad += alpha * penalty / n
    loss = 0.5 * np.mean(residual**2) + alpha * np.mean(logsumexp - q[idx, actions])
    return float(loss), grad


def train_q(prices, funding, scale, horizon: int, seed: int,
            *, offline: bool, risk_penalty: float = 0.01, steps: int = 4096):
    horizon, seed, steps = _training_indices(horizon, seed, steps)
    if type(offline) is not bool:
        raise ValueError("offline training mode must be boolean")
    cfg = Execution(risk_penalty=risk_penalty)
    net, target = Network(seed), Network(seed)
    rng = np.random.default_rng(seed + 2)
    episodes, losses, rollouts = [], [], []
    buffer = None
    if offline:
        buffer = collect(prices, funding, scale, horizon, seed + 2000, steps, execution=cfg)
        episodes.extend(buffer["episodes"])
        rollouts.append(buffer["episodeAccountingV2"])
    for batch in range((steps + 255) // 256):
        count = min(256, steps - batch * 256)
        if not offline:
            def explore(s):
                probs = np.full(3, 0.2 / 3)
                probs[int(np.argmax(net.forward(s)))] += 0.8
                return probs
            new = collect(prices, funding, scale, horizon, seed + 3000 + batch, count,
                          policy=explore, execution=cfg)
            episodes.extend(new["episodes"])
            rollouts.append(new["episodeAccountingV2"])
            buffer = new if buffer is None else {k: np.concatenate([buffer[k], new[k]])
                                                for k in ("s", "a", "r", "next", "done", "prob")}
        for _ in range(count):
            idx = rng.integers(len(buffer["a"]), size=64)
            x, nxt = buffer["s"][idx], buffer["next"][idx]
            greedy = np.argmax(net.forward(nxt), axis=1)
            targets = buffer["r"][idx] + 0.99**horizon * (~buffer["done"][idx]) * target.forward(nxt)[np.arange(64), greedy]
            loss, grad = bellman_gradient(net.forward(x), buffer["a"][idx], targets, 0.1 if offline else 0.0)
            net.update(x, grad, 0.001)
            if net.steps % 100 == 0:
                target.copy_from(net)
        losses.append(loss)
    support = np.bincount(buffer["a"], minlength=3).tolist()
    return net, {"steps": steps, "updates": net.steps, "episodes": episodes,
                 "episodeAccountingV2": _episode_totals(rollouts),
                 "losses": losses, "behaviorActionCounts": support,
                 "bufferTransitions": len(buffer["a"]), "behavior": "uniform_simulated" if offline else "epsilon_greedy_simulated"}


def _finite_real_vector(value, width: int) -> bool:
    return (isinstance(value, np.ndarray) and not np.ma.isMaskedArray(value) and value.shape == (width,) and
            value.dtype.kind in "iuf" and bool(np.isfinite(value).all()))


def infer(net: Network, observation: np.ndarray | None, *, enabled: bool = False) -> tuple[float | None, float]:
    start = time.perf_counter_ns()
    if enabled is not True or not _finite_real_vector(observation, FEATURE_COUNT):
        return None, (time.perf_counter_ns() - start) / 1e6
    try:
        out = net.forward(observation)
    except Exception:
        # An inference failure is an absent proposal, never a directional default.
        out = None
    elapsed = (time.perf_counter_ns() - start) / 1e6
    if not _finite_real_vector(out, 3) or not 0 <= elapsed <= 20:
        return None, elapsed
    return float(ACTIONS[int(np.argmax(out))]), elapsed


def validate_provenance(provenance: dict) -> None:
    """A matching digest proves bytes, not that their provenance is meaningful."""
    required = {"codeCommit", "registrationSha256", "dataSha256", "seed", "horizon", "algorithm", "fold"}
    if not isinstance(provenance, dict) or not required <= provenance.keys():
        raise ValueError("missing provenance")
    hashes = {"codeCommit": 40, "registrationSha256": 64, "dataSha256": 64}
    if "fundingSha256" in provenance:
        hashes["fundingSha256"] = 64
    for key, length in hashes.items():
        value = provenance[key]
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value) is None:
            raise ValueError("invalid provenance hash")
    for key in ("seed", "horizon", "fold"):
        if type(provenance[key]) is not int or provenance[key] < 0:
            raise ValueError("invalid provenance integer")
    if (provenance["horizon"] not in (1, 3, 6) or
        provenance["algorithm"] not in ("ppo", "double_dqn", "cql", "cql_no_inventory_penalty")):
        raise ValueError("unsupported policy provenance")
    try:
        json.dumps(provenance, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid provenance JSON") from exc


def _parameter_snapshots(parameters: dict) -> dict:
    """One v1 parameter contract for policy writers and readers; critics are distinct."""
    shapes = {"w1": (FEATURE_COUNT, 16), "b1": (16,), "w2": (16, 3), "b2": (3,)}
    if not isinstance(parameters, dict) or set(parameters) != set(shapes):
        raise ValueError("parameter fields")
    snapshots = {}
    for name, shape in shapes.items():
        value = parameters[name]
        if (not isinstance(value, np.ndarray) or np.ma.isMaskedArray(value) or
            value.dtype.kind not in "iuf" or value.shape != shape):
            raise ValueError("invalid parameter shape or type")
        snapshot = value.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            portable = np.asarray(snapshot, dtype=float)
        if not np.isfinite(portable).all():
            raise ValueError("non-finite parameters")
        # Preserve integer JSON values; floating parameters use portable float64.
        snapshots[name] = portable if snapshot.dtype.kind == "f" else snapshot
    return snapshots


def save_policy(path: Path, net: Network, provenance: dict) -> str:
    validate_provenance(provenance)
    parameters = _parameter_snapshots(net.p)
    value = {"schema": "offline_policy_v1", "environment": ENVIRONMENT,
             "observation": OBSERVATION, "actions": ACTIONS.tolist(),
             "promotion": "rejected_research_only", "enabled": False,
             "provenance": provenance, "parameters": {k: v.tolist() for k, v in parameters.items()}}
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    if len(raw) > 65536:
        raise ValueError("artifact too large")
    with path.open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def load_policy(path: Path, expected_sha256: str, expected_provenance: dict) -> Network:
    validate_provenance(expected_provenance)
    with path.open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError("artifact too large")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("artifact hash mismatch")
    def unique(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                raise ValueError("duplicate artifact field")
            d[k] = v
        return d
    a = json.loads(raw, object_pairs_hook=unique, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite artifact")))
    if not isinstance(a, dict) or set(a) != {"schema", "environment", "observation", "actions", "promotion", "enabled", "provenance", "parameters"}:
        raise ValueError("artifact fields")
    if (not isinstance(a["actions"], list) or
        any(type(v) not in (int, float) for v in a["actions"]) or
        not isinstance(a["provenance"], dict) or not isinstance(a["parameters"], dict)):
        raise ValueError("artifact types")
    validate_provenance(a["provenance"])
    if (a["schema"] != "offline_policy_v1" or a["environment"] != ENVIRONMENT or
        a["observation"] != OBSERVATION or a["actions"] != ACTIONS.tolist() or
        a["promotion"] != "rejected_research_only" or a["enabled"] is not False or
        json.dumps(a["provenance"], sort_keys=True) != json.dumps(expected_provenance, sort_keys=True)):
        raise ValueError("incompatible artifact")
    def numeric(value):
        return all(numeric(v) for v in value) if isinstance(value, list) else type(value) in (int, float)
    parameters = {}
    for name, value in a["parameters"].items():
        if not numeric(value):
            raise ValueError("non-numeric parameter")
        try:
            parameters[name] = np.asarray(value, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("invalid numeric parameters") from exc
    snapshots = _parameter_snapshots(parameters)
    net = Network(0)
    net.p = snapshots
    return net

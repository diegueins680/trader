"""Small original NumPy implementations of PPO, Double DQN and discrete CQL.

Mechanism prototypes, not reproductions of published benchmark scores. No
production imports, policy promotion, network or exchange authorization.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import numpy as np
from sequential_env import ACTIONS, ENVIRONMENT, FEATURE_COUNT, OBSERVATION, Execution, collect


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
        return np.tanh(x @ self.p["w1"] + self.p["b1"]) @ self.p["w2"] + self.p["b2"]

    def gradients(self, x: np.ndarray, dz: np.ndarray) -> dict:
        h = np.tanh(x @ self.p["w1"] + self.p["b1"])
        dh = (dz @ self.p["w2"].T) * (1 - h**2)
        return {"w1": x.T @ dh, "b1": dh.sum(0), "w2": h.T @ dz, "b2": dz.sum(0)}

    def update(self, x: np.ndarray, dz: np.ndarray, lr: float) -> None:
        grads = self.gradients(x, dz)
        if not all(np.isfinite(g).all() for g in grads.values()):
            raise ValueError("non-finite gradient")
        norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
        self.steps += 1
        for k, grad in grads.items():
            g = grad / max(1.0, norm)
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * g**2
            m = self.m[k] / (1 - 0.9**self.steps)
            v = self.v[k] / (1 - 0.999**self.steps)
            self.p[k] -= lr * m / (np.sqrt(v) + 1e-8)
        if not all(np.isfinite(v).all() for v in self.p.values()):
            raise ValueError("non-finite model")

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


def train_ppo(prices, funding, scale, horizon: int, seed: int, steps: int = 4096):
    net, value = Network(seed), Network(seed + 1, 1)
    episodes, losses = [], []
    for batch in range((steps + 255) // 256):
        count = min(256, steps - batch * 256)
        data = collect(prices, funding, scale, horizon, seed + 1000 + batch, count,
                       policy=lambda s: softmax(net.forward(s)))
        episodes.extend(data["episodes"])
        adv, targets = advantages(data, value, 0.99**horizon)
        for _ in range(4):
            loss, grad = ppo_gradient(net.forward(data["s"]), data["a"], data["prob"], adv)
            net.update(data["s"], grad, 0.0003)
            residual = value.forward(data["s"]).ravel() - targets
            value.update(data["s"], (residual / count)[:, None], 0.0003)
        losses.append(loss)
    return net, {"steps": steps, "updates": net.steps, "episodes": episodes, "losses": losses}


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
    net, target = Network(seed), Network(seed)
    rng = np.random.default_rng(seed + 2)
    cfg = Execution(risk_penalty=risk_penalty)
    episodes, losses = [], []
    buffer = None
    if offline:
        buffer = collect(prices, funding, scale, horizon, seed + 2000, steps, execution=cfg)
        episodes.extend(buffer["episodes"])
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
                 "losses": losses, "behaviorActionCounts": support,
                 "bufferTransitions": len(buffer["a"]), "behavior": "uniform_simulated" if offline else "epsilon_greedy_simulated"}


def infer(net: Network, observation: np.ndarray | None, *, enabled: bool = False) -> tuple[float | None, float]:
    start = time.perf_counter_ns()
    if not enabled or observation is None or observation.shape != (FEATURE_COUNT,) or not np.isfinite(observation).all():
        return None, (time.perf_counter_ns() - start) / 1e6
    out = net.forward(observation)
    elapsed = (time.perf_counter_ns() - start) / 1e6
    if out.shape != (3,) or not np.isfinite(out).all() or elapsed > 20:
        return None, elapsed
    return float(ACTIONS[int(np.argmax(out))]), elapsed


def save_policy(path: Path, net: Network, provenance: dict) -> str:
    required = {"codeCommit", "registrationSha256", "dataSha256", "seed", "horizon", "algorithm", "fold"}
    if not required <= provenance.keys():
        raise ValueError("missing provenance")
    value = {"schema": "offline_policy_v1", "environment": ENVIRONMENT,
             "observation": OBSERVATION, "actions": ACTIONS.tolist(),
             "promotion": "rejected_research_only", "enabled": False,
             "provenance": provenance, "parameters": {k: v.tolist() for k, v in net.p.items()}}
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    if len(raw) > 65536:
        raise ValueError("artifact too large")
    with path.open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def load_policy(path: Path, expected_sha256: str, expected_provenance: dict) -> Network:
    if path.stat().st_size > 65536:
        raise ValueError("artifact too large")
    raw = path.read_bytes()
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
    if set(a) != {"schema", "environment", "observation", "actions", "promotion", "enabled", "provenance", "parameters"}:
        raise ValueError("artifact fields")
    if (a["schema"] != "offline_policy_v1" or a["environment"] != ENVIRONMENT or
        a["observation"] != OBSERVATION or a["actions"] != ACTIONS.tolist() or
        a["promotion"] != "rejected_research_only" or a["enabled"] is not False or
        json.dumps(a["provenance"], sort_keys=True) != json.dumps(expected_provenance, sort_keys=True)):
        raise ValueError("incompatible artifact")
    net = Network(0)
    if set(a["parameters"]) != set(net.p):
        raise ValueError("parameter fields")
    for k, template in net.p.items():
        arr = np.asarray(a["parameters"][k], dtype=float)
        if arr.shape != template.shape or not np.isfinite(arr).all():
            raise ValueError("invalid parameters")
        net.p[k] = arr
    return net

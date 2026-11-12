"""Minimal dense networks with hand-written reverse-mode gradients (NumPy only).

Gradients are verified against central finite differences in
``test/exposure_rl_test.py``. There is no autograd dependency.
"""

from __future__ import annotations

import numpy as np


class MLP:
    """Fully connected tanh network with a linear (or tanh) output layer."""

    def __init__(self, sizes: list[int], rng: np.random.Generator, out_tanh: bool = False, out_scale: float = 1.0):
        self.sizes = list(sizes)
        self.out_tanh = out_tanh
        self.params: list[np.ndarray] = []
        for index, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            scale = np.sqrt(1.0 / fan_in) * (out_scale if index == len(sizes) - 2 else 1.0)
            self.params.append(rng.normal(0.0, scale, size=(fan_in, fan_out)))
            self.params.append(np.zeros(fan_out))
        self._cache: list[np.ndarray] = []

    def forward(self, inputs: np.ndarray, params: list[np.ndarray] | None = None, cache: bool = True) -> np.ndarray:
        params = self.params if params is None else params
        activations = [np.asarray(inputs, dtype=np.float64)]
        layers = len(params) // 2
        for layer in range(layers):
            z = activations[-1] @ params[2 * layer] + params[2 * layer + 1]
            if layer < layers - 1 or self.out_tanh:
                z = np.tanh(z)
            activations.append(z)
        if cache:
            self._cache = activations
        return activations[-1]

    def backward(self, grad_output: np.ndarray, params: list[np.ndarray] | None = None):
        """Return (parameter gradients, gradient w.r.t. inputs) for the cached forward pass."""
        params = self.params if params is None else params
        activations = self._cache
        layers = len(params) // 2
        grads: list[np.ndarray] = [None] * len(params)  # type: ignore[list-item]
        delta = np.asarray(grad_output, dtype=np.float64)
        for layer in reversed(range(layers)):
            if layer < layers - 1 or self.out_tanh:
                delta = delta * (1.0 - activations[layer + 1] ** 2)
            grads[2 * layer] = activations[layer].T @ delta
            grads[2 * layer + 1] = delta.sum(axis=0)
            delta = delta @ params[2 * layer].T
        return grads, delta

    def copy_params(self) -> list[np.ndarray]:
        return [p.copy() for p in self.params]


class Adam:
    def __init__(self, params: list[np.ndarray], lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, max_norm: float | None = None):
        self.params, self.lr, self.beta1, self.beta2, self.eps, self.max_norm = params, lr, beta1, beta2, eps, max_norm
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.steps = 0

    def step(self, grads: list[np.ndarray]) -> float:
        """Apply one descent step; returns the pre-clip global gradient norm."""
        norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))
        if not np.isfinite(norm):
            raise FloatingPointError("non-finite gradient")
        scale = 1.0 if self.max_norm is None or norm <= self.max_norm else self.max_norm / norm
        self.steps += 1
        correction1 = 1.0 - self.beta1**self.steps
        correction2 = 1.0 - self.beta2**self.steps
        for param, grad, m, v in zip(self.params, grads, self.m, self.v):
            g = grad * scale
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * g * g
            param -= self.lr * (m / correction1) / (np.sqrt(v / correction2) + self.eps)
        return norm


def polyak(target: list[np.ndarray], source: list[np.ndarray], tau: float) -> None:
    for t, s in zip(target, source):
        t *= 1.0 - tau
        t += tau * s


def numerical_gradient(function, params: list[np.ndarray], eps: float = 1e-6) -> list[np.ndarray]:
    """Central finite-difference gradient of scalar ``function()`` w.r.t. params."""
    grads = []
    for param in params:
        grad = np.zeros_like(param)
        flat, flat_grad = param.reshape(-1), grad.reshape(-1)
        for index in range(flat.size):
            original = flat[index]
            flat[index] = original + eps
            upper = function()
            flat[index] = original - eps
            lower = function()
            flat[index] = original
            flat_grad[index] = (upper - lower) / (2.0 * eps)
        grads.append(grad)
    return grads

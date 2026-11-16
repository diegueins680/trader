"""Synthetic engineering fixtures; no financial trials or historical inputs."""
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
import math
from pathlib import Path
import random
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research"))
import ope_rational_v2 as old
import ope_rational_v3 as ope

TINY, TOP = 5e-324, math.nextafter(math.inf, 0.0)


def batch(count, horizon, seed=7, module=ope):
    rng = random.Random(seed)
    draw = lambda f: tuple(f() for _ in range(horizon))
    return tuple(module.Episode(draw(lambda: rng.gauss(0.0, 1.0)), draw(lambda: rng.randrange(3)),
                                draw(lambda: rng.uniform(2.0 ** -10, 1.0)), draw(rng.random),
                                draw(lambda: rng.gauss(0.0, 1.0)), draw(lambda: rng.gauss(0.0, 1.0)) + (0.0,))
                 for _ in range(count))


def fields(estimate):
    return (estimate.ordinary_is, estimate.per_decision_is, estimate.weighted_is, estimate.doubly_robust,
            estimate.effective_sample_size, estimate.max_weight, estimate.nonzero_trajectories, estimate.horizon,
            tuple((e.discounted_return, e.weight, e.ordinary_is, e.per_decision_is, e.doubly_robust) for e in estimate.episodes))


class ActivationTests(unittest.TestCase):
    def test_disabled_by_default_and_exact_version_required(self):
        episodes = batch(2, 3)
        self.assertIsNone(ope.estimate_v3(episodes, 0.99))
        self.assertIsNone(ope.estimate_v3(episodes, 0.99, enabled=1))
        self.assertIsNone(ope.estimate_v3(episodes, 0.99, enabled=True, version="ope-rational-v2"))
        got = ope.estimate_v3(episodes, 0.99, enabled=True)
        self.assertEqual((got.version, got.reliable, got.horizon, len(got.episodes)), ("ope-rational-v3", False, 3, 2))
        with self.assertRaises(FrozenInstanceError):
            got.reliable = True

    def test_module_loads_only_the_standard_library(self):
        code = ("import sys; before = set(sys.modules); sys.path.insert(0, 'scripts/research'); import ope_rational_v3; "
                "print(sorted(m for m in set(sys.modules) - before if m not in sys.stdlib_module_names "
                "and m.split('.')[0] not in sys.stdlib_module_names))")
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[1], timeout=60)
        self.assertEqual((done.returncode, done.stdout.strip()), (0, "['ope_rational_v3']"), done.stderr)


class ArithmeticTests(unittest.TestCase):
    def test_equals_the_predecessor_where_the_predecessor_publishes(self):
        for count, horizon in ((1, 1), (3, 5), (8, 8)):
            before = old.estimate_v2(batch(count, horizon, module=old), 0.99, enabled=True)
            after = ope.estimate_v3(batch(count, horizon), 0.99, enabled=True)
            self.assertIsNotNone(before)
            self.assertEqual(fields(before), fields(after))

    def test_exact_weights_and_weighted_estimate(self):
        episodes = batch(4, 6)
        got = ope.estimate_v3(episodes, 0.5, enabled=True)
        weights = [math.prod((F(p) / F(b) for p, b in zip(e.target, e.behavior)), start=F(1)) for e in episodes]
        returns = [sum(F(1, 2 ** t) * F(r) for t, r in enumerate(e.rewards)) for e in episodes]
        self.assertEqual([e.weight for e in got.episodes], weights)
        self.assertEqual(got.weighted_is, sum(w * g for w, g in zip(weights, returns)) / sum(weights))
        self.assertEqual(got.effective_sample_size, sum(weights) ** 2 / sum(w * w for w in weights))

    def test_admits_what_the_predecessor_rejects(self):
        self.assertIsNone(old.estimate_v2(batch(48, 32, module=old), 0.99, enabled=True))
        got = ope.estimate_v3(batch(48, 32), 0.99, enabled=True)
        self.assertEqual(len(got.episodes), 48)
        self.assertTrue(1 <= got.effective_sample_size <= 48)
        self.assertGreater(got.effective_sample_size.denominator.bit_length(), 8192)

    def test_extreme_binary64_values_publish(self):
        extreme = tuple(ope.Episode((sign * TOP,) * 32, (0,) * 32, (TINY,) * 32, (1.0,) * 32, (-sign * TOP,) * 32,
                                    (sign * TOP,) * 32 + (0.0,)) for sign in (1.0, -1.0, 1.0))
        got = ope.estimate_v3(extreme, 1.0, enabled=True)
        self.assertEqual(got.max_weight, F(2) ** (1074 * 32))
        self.assertEqual(got.effective_sample_size, 3)

    def test_no_support_is_reported_not_divided(self):
        zero = tuple(ope.Episode(e.rewards, e.actions, e.behavior, (0.0,) * 4, e.q, e.v) for e in batch(3, 4))
        got = ope.estimate_v3(zero, 0.99, enabled=True)
        self.assertEqual((got.weighted_is, got.effective_sample_size, got.nonzero_trajectories), (None, 0, 0))


class RejectionTests(unittest.TestCase):
    def test_invalid_batches_reject(self):
        episodes = batch(2, 3)
        first = episodes[0]
        bad = [
            (batch(257, 1), 0.99), (batch(1, 33), 0.99), ((), 0.99), (list(episodes), 0.99), (episodes, 1), (episodes, 1.5),
            (episodes, math.nan), ((first, None), 0.99), ((first, batch(1, 2)[0]), 0.99),
            ((first, old.Episode(first.rewards, first.actions, first.behavior, first.target, first.q, first.v)), 0.99),
            ((ope.Episode(first.rewards, first.actions, (0.0,) + first.behavior[1:], first.target, first.q, first.v),), 0.99),
            ((ope.Episode(first.rewards, first.actions, first.behavior, (1.5,) + first.target[1:], first.q, first.v),), 0.99),
            ((ope.Episode((math.inf,) + first.rewards[1:], first.actions, first.behavior, first.target, first.q, first.v),), 0.99),
            ((ope.Episode(first.rewards, (3,) + first.actions[1:], first.behavior, first.target, first.q, first.v),), 0.99),
            ((ope.Episode(first.rewards, first.actions, first.behavior, first.target, first.q, first.v[:-1] + (0.5,)),), 0.99),
        ]
        for episodes_, gamma in bad:
            with self.subTest(gamma=gamma, count=len(episodes_)):
                self.assertIsNone(ope.estimate_v3(episodes_, gamma, enabled=True))

    def test_size_limit_is_still_evaluated(self):
        episodes = batch(8, 32)
        self.assertIsNotNone(ope.estimate_v3(episodes, 0.99, enabled=True))
        with patch.object(ope, "MAX_BITS", 8192):
            self.assertIsNone(ope.estimate_v3(episodes, 0.99, enabled=True))

    def test_resource_failure_publishes_nothing(self):
        with patch.object(ope, "_mul", side_effect=MemoryError):
            self.assertIsNone(ope.estimate_v3(batch(2, 3), 0.99, enabled=True))


if __name__ == "__main__":
    unittest.main()

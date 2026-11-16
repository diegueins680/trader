"""Synthetic engineering fixtures; no financial trials or historical inputs."""
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research"))
import causal_ope_v8 as ope
import causal_optimizer_v5 as optimizer
import causal_ppo_v7 as ppo
import causal_replay_v3 as replay
import causal_value_v6 as value

GRID = 2 ** 53


def initial():
    return tuple(replay.Bar("SYNTH", i * 10, i * 10 + 1, F(100)) for i in range(26))


def outcomes(module, count, *, shift=0, start=0):
    return tuple(module.Outcome(replay.Bar("SYNTH", i * 10, i * 10 + 1,
                                           F(100) + F(i % 7 - 3, 8) + (shift if i - 26 >= start else 0)), (), fill=F(1, 2))
                 for i in range(26, 26 + count))


def run(module, seed=11, count=40, **kwargs):
    train = value.train_causal_value_v6 if module is value else ppo.train_causal_ppo_v7
    return train(initial(), outcomes(module, count, **kwargs), seed, enabled=True)


def model(index, horizon):
    return ope.Model(tuple(((7 * index + 3 * t) % 9 + 1) / 16 for t in range(horizon)),
                     tuple(((index + t) % 5 - 2) / 8 for t in range(horizon)),
                     tuple(((index + 2 * t) % 7 - 3) / 16 for t in range(horizon)) + (0.0,))


def evaluate(runs, models, horizon, **kwargs):
    return ope.evaluate_causal_ope_v8(runs, models, horizon, enabled=True, **kwargs)


RUNS = tuple(run(module, seed) for module in (value, ppo) for seed in (11, 23))


class ImportClosureTests(unittest.TestCase):
    def test_adapter_loads_no_frozen_module(self):
        frozen = {"sequential_learning", "sequential_env", "sequential_evaluation", "sequential_registry",
                  "run_sequential_screen", "causal_ppo_v5", "ppo_successor_v2"}
        code = ("import sys; sys.path.insert(0, 'scripts/research'); import causal_ope_v8; "
                "print(sorted(set(sys.modules) & %r))" % frozen)
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[1], timeout=60)
        self.assertEqual((done.returncode, done.stdout.strip()), (0, "[]"), done.stderr)


class CausalOpeTests(unittest.TestCase):
    def test_default_disabled_and_version_gated(self):
        models = tuple(model(i, 7) for i in range(4))
        self.assertIsNone(ope.evaluate_causal_ope_v8(RUNS, models, 7))
        self.assertIsNone(ope.evaluate_causal_ope_v8(RUNS, models, 7, enabled=1))
        self.assertIsNone(evaluate(RUNS, models, 7, version="ope-rational-v2"))
        self.assertIsNotNone(evaluate(RUNS, models, 7))

    def test_estimate_matches_hand_computed_exact_rationals(self):
        horizon = 7
        models = tuple(model(i, horizon) for i in range(4))
        before = repr((RUNS, models))
        got = evaluate(RUNS, models, horizon)
        self.assertEqual(got, evaluate(RUNS, models, horizon))
        self.assertEqual(repr((RUNS, models)), before)
        gamma = F(0.99)
        rows = []
        for result, m in zip(RUNS, models):
            weight, plain, stepwise, robust = F(1), F(0), F(0), F(m.v[0])
            for t in range(horizon):
                reward = F(float(result.transitions[t].reward))
                weight *= F(m.target[t]) / F(float(result.behavior_probabilities[t]))
                plain += gamma ** t * reward
                stepwise += weight * gamma ** t * reward
                robust += weight * gamma ** t * (reward + gamma * F(m.v[t + 1]) - F(m.q[t]))
            rows.append((plain, weight, stepwise, robust))
        estimate = got.estimate
        mass = sum(r[1] for r in rows)
        self.assertEqual(estimate.ordinary_is, sum(r[1] * r[0] for r in rows) / 4)
        self.assertEqual(estimate.per_decision_is, sum(r[2] for r in rows) / 4)
        self.assertEqual(estimate.doubly_robust, sum(r[3] for r in rows) / 4)
        self.assertEqual(estimate.weighted_is, sum(r[1] * r[0] for r in rows) / mass)
        self.assertEqual(estimate.effective_sample_size, mass ** 2 / sum(r[1] ** 2 for r in rows))
        self.assertEqual([e.weight for e in estimate.episodes], [r[1] for r in rows])
        self.assertEqual((got.version, got.estimator_version, got.gamma, got.horizon, got.reliable, estimate.reliable),
                         ("causal-ope-v8", "ope-rational-v2", 0.99, horizon, False, False))
        self.assertEqual(got.sources, (("causal-value-v6", 11), ("causal-value-v6", 23),
                                       ("causal-ppo-v7", 11), ("causal-ppo-v7", 23)))
        self.assertEqual((got.behavior_floor, got.law_gap, got.weight_gap), (2.0 ** -10, F(1, 2 ** 50), F(1, 2 ** 34)))
        with self.assertRaises(FrozenInstanceError):
            got.reliable = True

    def test_on_policy_target_gives_unit_weights_and_full_sample_size(self):
        horizon = 12
        models = tuple(replace(model(i, horizon), target=tuple(float(p) for p in r.behavior_probabilities[:horizon]))
                       for i, r in enumerate(RUNS))
        estimate = evaluate(RUNS, models, horizon).estimate
        self.assertTrue(all(e.weight == 1 for e in estimate.episodes))
        self.assertEqual((estimate.effective_sample_size, estimate.max_weight, estimate.nonzero_trajectories), (4, 1, 4))
        self.assertEqual(estimate.ordinary_is, estimate.per_decision_is)
        self.assertEqual(estimate.ordinary_is, estimate.weighted_is)

    def test_zero_target_probability_removes_the_trajectory_without_dividing_by_zero(self):
        models = tuple(replace(model(i, 3), target=(0.0, 0.5, 0.5)) for i in range(4))
        estimate = evaluate(RUNS, models, 3).estimate
        self.assertEqual((estimate.nonzero_trajectories, estimate.weighted_is, estimate.effective_sample_size), (0, None, 0))

    def test_short_run_rejects_everything_and_is_never_dropped_or_padded(self):
        short = run(ppo, 47, count=5)
        models = tuple(model(i, 7) for i in range(5))
        self.assertIsNone(evaluate(RUNS + (short,), models, 7))
        self.assertIsNotNone(evaluate(RUNS + (short,), tuple(model(i, 5) for i in range(5)), 5))
        future = outcomes(ppo, 3)
        ended = ppo.train_causal_ppo_v7(initial(), (future[0], ppo.Outcome(future[1].bar, (), terminal=True, fill=F(1, 2)),
                                                    future[2]), 47, enabled=True)
        self.assertEqual((ended.steps, ended.transitions[-1].done), (2, True))
        self.assertIsNone(evaluate((ended,), (model(0, 3),), 3))
        self.assertIsNotNone(evaluate((ended,), (model(0, 2),), 2))

    def test_evaluation_reads_only_the_horizon_prefix(self):
        for module in (value, ppo):
            for horizon in (1, 7, 32):
                with self.subTest(module=module.__name__, horizon=horizon):
                    base, other = run(module), run(module, shift=3, start=horizon)
                    self.assertNotEqual(base.transitions[horizon], other.transitions[horizon])
                    m = (model(1, horizon),)
                    self.assertEqual(evaluate((base,), m, horizon), evaluate((other,), m, horizon))
                    self.assertIsNotNone(evaluate((base,), m, horizon))

    def test_behavior_floor_and_exact_value_law(self):
        horizon = 7
        models = tuple(model(i, horizon) for i in range(4))
        forge = lambda result, p: replace(result, behavior_probabilities=(p,) + result.behavior_probabilities[1:])
        policy, learned = RUNS[2], RUNS[0]
        for bad in (forge(policy, 2.0 ** -11), forge(policy, 0.0), forge(policy, 1.5), forge(policy, float("nan")),
                    forge(policy, F(1, 2)), forge(learned, F(1, 3)), forge(learned, 11 / 12), forge(learned, F(1, 8))):
            with self.subTest(probability=bad.behavior_probabilities[0]):
                runs = tuple(bad if r.version == bad.version and r.seed == bad.seed else r for r in RUNS)
                self.assertIsNone(evaluate(runs, models, horizon))
        self.assertIsNotNone(evaluate((RUNS[0], RUNS[1], forge(policy, 2.0 ** -10), RUNS[3]), models, horizon))
        self.assertTrue(all(p in (F(11, 12), F(1, 24)) for p in learned.behavior_probabilities))

    def test_recorded_probability_is_within_the_certified_gap_of_the_drawing_law(self):
        for seed in (11, 23, 47):
            result = run(ppo, seed)
            actor = optimizer.create_v5(seed, 3, enabled=True)
            rng = np.random.default_rng([seed, 7])
            for row, recorded in zip(result.transitions[:32], result.behavior_probabilities):
                x = np.asarray([[float(v) for v in row.observation.values]])
                probs = ppo._policy(tuple(optimizer.forward_v5(actor, x, enabled=True)[0].tolist()))
                k = F(float(rng.random())) * GRID
                self.assertEqual(k.denominator, 1)
                low = math.ceil(F(probs[0]) * GRID)
                cut = min(math.ceil(F(probs[0] + probs[1]) * GRID), GRID)
                law = (F(low, GRID), F(cut - low, GRID), F(GRID - cut, GRID))
                self.assertEqual(row.action, 0 if k < low else 1 if k < cut else 2)
                self.assertEqual(recorded, probs[row.action])
                for exact, p in zip(law, probs):
                    self.assertLessEqual(abs(exact - F(p)), ope.LAW_GAP)
                self.assertLessEqual(abs(law[row.action] - F(recorded)), F(recorded) / 2 ** 40)

    def test_gap_constants_are_consistent(self):
        step = ope.LAW_GAP / F(ope.FLOOR)
        self.assertEqual(step, F(1, 2 ** 40))
        self.assertLessEqual((1 + step) ** ope.HORIZON - 1, ope.WEIGHT_GAP)
        self.assertGreater((1 + step) ** ope.HORIZON - 1, ope.WEIGHT_GAP / 2)
        for exact in ope.VALUE_LAW:
            self.assertLessEqual(abs(F(float(exact)) - exact), exact / 2 ** 53)
            self.assertGreaterEqual(float(exact), ope.FLOOR)

    def test_invalid_inputs_publish_nothing(self):
        horizon = 7
        models = tuple(model(i, horizon) for i in range(4))
        swap = lambda items, index, item: items[:index] + (item,) + items[index + 1:]
        m = models[1]
        cases = [
            (list(RUNS), models, horizon), (RUNS, list(models), horizon), (RUNS, models[:-1], horizon), ((), (), horizon),
            (RUNS, models, 0), (RUNS, models, 33), (RUNS, models, True), (RUNS, models, 7.0),
            (RUNS + (RUNS[0],), models + (models[0],), horizon),
            (swap(RUNS, 0, None), models, horizon), (swap(RUNS, 0, replace(RUNS[0], version="causal-ppo-v7")), models, horizon),
            (swap(RUNS, 0, replace(RUNS[0], seed=True)), models, horizon),
            (swap(RUNS, 0, replace(RUNS[0], transitions=list(RUNS[0].transitions))), models, horizon),
            (swap(RUNS, 0, replace(RUNS[0], behavior_probabilities=RUNS[0].behavior_probabilities[:-1])), models, horizon),
            (swap(RUNS, 0, replace(RUNS[0], transitions=swap(RUNS[0].transitions, 2, replace(RUNS[0].transitions[2], action=3)))),
             models, horizon),
            (swap(RUNS, 0, replace(RUNS[0], transitions=swap(RUNS[0].transitions, 2, replace(RUNS[0].transitions[2], done=True)))),
             models, horizon),
            (swap(RUNS, 0, replace(RUNS[0], transitions=swap(RUNS[0].transitions, 2,
                                                             replace(RUNS[0].transitions[2], reward=F(2) ** 100)))), models, horizon),
            (RUNS, swap(models, 1, None), horizon), (RUNS, swap(models, 1, replace(m, target=list(m.target))), horizon),
            (RUNS, swap(models, 1, replace(m, target=swap(m.target, 0, 1.5))), horizon),
            (RUNS, swap(models, 1, replace(m, target=swap(m.target, 0, -0.5))), horizon),
            (RUNS, swap(models, 1, replace(m, target=swap(m.target, 0, float("nan")))), horizon),
            (RUNS, swap(models, 1, replace(m, target=swap(m.target, 0, 1))), horizon),
            (RUNS, swap(models, 1, replace(m, q=swap(m.q, 0, float("inf")))), horizon),
            (RUNS, swap(models, 1, replace(m, q=swap(m.q, 0, 2.0 ** 100))), horizon),
            (RUNS, swap(models, 1, replace(m, q=m.q[:-1])), horizon),
            (RUNS, swap(models, 1, replace(m, v=swap(m.v, horizon, 0.5))), horizon),
            (RUNS, swap(models, 1, replace(m, v=swap(m.v, 0, float("nan")))), horizon),
            (RUNS, swap(models, 1, replace(m, v=m.v[:-1])), horizon),
        ]
        for index, arguments in enumerate(cases):
            with self.subTest(case=index):
                self.assertIsNone(evaluate(*arguments))

    def test_estimator_rejection_and_size_limit_publish_nothing(self):
        models = tuple(model(i, 7) for i in range(4))
        with patch.object(ope, "estimate_v2", return_value=None):
            self.assertIsNone(evaluate(RUNS, models, 7))
        with patch.object(ope, "estimate_v2", side_effect=MemoryError):
            self.assertIsNone(evaluate(RUNS, models, 7))
        tiny = (replace(models[0], target=(5e-324,) * 7),) + models[1:]
        self.assertIsNone(evaluate(RUNS, tiny, 7))

    def test_capacity_is_small_and_detected(self):
        runs = tuple(run(ppo, seed) for seed in (11, 23, 47))
        models = tuple(model(i, 32) for i in range(3))
        self.assertIsNotNone(evaluate(runs[:2], models[:2], 32))
        self.assertIsNone(evaluate(runs, models, 32))
        self.assertIsNotNone(evaluate(runs, tuple(model(i, 16) for i in range(3)), 16))


if __name__ == "__main__":
    unittest.main()

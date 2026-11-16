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
import causal_optimizer_v5 as optimizer
import causal_ppo_v7 as learner
import causal_replay_v3 as replay
import gae_targets_v2 as gae
import ppo_objective_v2 as objective


def initial():
    return tuple(replay.Bar("SYNTH", i * 10, i * 10 + 1, F(100)) for i in range(26))


def outcomes(count, *, multiplier=F(1), drift=0):
    return tuple(learner.Outcome(replay.Bar("SYNTH", i * 10, i * 10 + 1, F(100) + drift * (i % 7 - 3)), (),
                                 fill=F(1, 2), multiplier=multiplier) for i in range(26, 26 + count))


def train(count=17, seed=11, **kwargs):
    return learner.train_causal_ppo_v7(initial(), outcomes(count, drift=F(1, 8)), seed, enabled=True, **kwargs)


def softmax(row):
    m = max(row)
    terms = [math.exp(x - m) for x in row]
    total = math.fsum(terms)
    return tuple(t / total for t in terms)


class ImportClosureTests(unittest.TestCase):
    def test_learner_loads_no_frozen_module(self):
        frozen = {"sequential_learning", "sequential_env", "sequential_evaluation", "sequential_registry",
                  "run_sequential_screen", "causal_ppo_v5", "ppo_successor_v2"}
        code = ("import sys; sys.path.insert(0, 'scripts/research'); import causal_ppo_v7; "
                "print(sorted(set(sys.modules) & %r))" % frozen)
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[1], timeout=60)
        self.assertEqual((done.returncode, done.stdout.strip()), (0, "[]"), done.stderr)


class CausalPpoTrainingTests(unittest.TestCase):
    def test_repeatability_exact_replay_and_published_shape(self):
        for seed in (11, 23, 47):
            for count in (1, 17):
                with self.subTest(seed=seed, count=count):
                    bars, future = initial(), outcomes(count, drift=F(1, 8))
                    before = repr((bars, future))
                    result = learner.train_causal_ppo_v7(bars, future, seed, enabled=True)
                    self.assertIsNotNone(result)
                    self.assertEqual(result, learner.train_causal_ppo_v7(bars, future, seed, enabled=True))
                    self.assertEqual((result.steps, result.unused_outcomes, result.stop_reason), (count, 0, "input_exhausted"))
                    self.assertEqual((result.actor.step, result.critic.step, result.actor.outputs, result.critic.outputs),
                                     (4, 4, 3, 1))
                    self.assertEqual((len(result.losses), len(result.targets), len(result.advantages)), (4, count, count))
                    self.assertEqual((result.version, result.objective_version, result.target_version, result.optimizer_version),
                                     ("causal-ppo-v7", objective.VERSION, gae.VERSION, optimizer.VERSION))
                    self.assertTrue(all(type(x) is float and math.isfinite(x) for x in result.losses + result.targets))
                    self.assertTrue(all(2.0 ** -100 <= p <= 1.0 for p in result.behavior_probabilities))
                    session = replay.start_v3(bars, enabled=True)
                    for row, outcome in zip(result.transitions, future):
                        expected, receipt = replay.step_v3(session, outcome.bar, outcome.events,
                            (F(-1, 4), F(0), F(1, 4))[row.action], terminal=outcome.terminal,
                            fill=outcome.fill, multiplier=outcome.multiplier, impact=outcome.impact, enabled=True)
                        self.assertEqual(row.receipt, receipt)
                        self.assertEqual(row.observation, replay.observe_v3(session, enabled=True))
                        session = expected
                    self.assertEqual(result.session, session)
                    self.assertEqual(repr((bars, future)), before)
                    with self.assertRaises(FrozenInstanceError):
                        result.steps = 0

    def test_first_rollout_sampling_matches_seeded_actor_and_replayed_draws(self):
        for seed in (11, 23, 47):
            with self.subTest(seed=seed):
                result = train(17, seed)
                actor = optimizer.create_v5(seed, 3, enabled=True)
                rng = np.random.default_rng([seed, 7])
                for row, probability in zip(result.transitions, result.behavior_probabilities):
                    x = np.asarray([float(v) for v in row.observation.values])[None, :]
                    probs = softmax(optimizer.forward_v5(actor, x, enabled=True)[0].tolist())
                    u = float(rng.random())
                    self.assertEqual(row.action, 0 if u < probs[0] else 1 if u < probs[0] + probs[1] else 2)
                    self.assertEqual(probability, probs[row.action])

    def test_one_draw_per_step_whatever_the_data(self):
        draws = []

        class Spy:
            def __init__(self, inner):
                self.inner = inner

            def random(self):
                draws.append("u")
                return self.inner.random()

        actual = np.random.default_rng
        with patch.object(learner.np.random, "default_rng",
                          side_effect=lambda seed: Spy(actual(seed)) if type(seed) is list else actual(seed)):
            self.assertIsNotNone(train(17))
        self.assertEqual(draws, ["u"] * 17)

    def test_first_rollout_targets_match_exact_gae_recursion(self):
        gamma = F(learner.GAMMA)
        for seed in (11, 23, 47):
            with self.subTest(seed=seed):
                result = train(17, seed)
                critic = optimizer.create_v5(seed + 1, 1, enabled=True)
                rows = result.transitions
                inputs = np.asarray([[float(v) for v in row.observation.values] for row in rows])
                nexts = np.asarray([[float(v) for v in row.next_observation.values] for row in rows])
                values = [v[0] for v in optimizer.forward_v5(critic, inputs, enabled=True).tolist()]
                following = [v[0] for v in optimizer.forward_v5(critic, nexts, enabled=True).tolist()]
                later, expected = 0.0, []
                for row, value, nxt in reversed(list(zip(rows, values, following))):
                    bootstrap = float(gamma * F(nxt))
                    delta = float(F(float(F(float(row.reward)) + F(bootstrap))) - F(value))
                    trace = float(F(float(gamma * F(0.95))) * F(later))
                    later = float(F(delta) + F(trace))
                    expected.append(float(F(later) + F(value)))
                self.assertEqual([t.hex() for t in result.targets], [t.hex() for t in reversed(expected)])

    def test_normalized_advantages_are_centered_and_bounded_by_sqrt_n(self):
        result = train(257, 47)
        for chunk in (result.advantages[:256], result.advantages[256:]):
            self.assertLess(abs(math.fsum(chunk)), 1e-9 * max(1, len(chunk)))
            self.assertLessEqual(max(abs(a) for a in chunk), math.sqrt(len(chunk)) * (1 + 1e-12))
        self.assertEqual(result.advantages[256], 0.0)
        self.assertGreater(max(abs(a) for a in result.advantages[:256]), 1.0)

    def test_future_rewrite_cannot_change_any_earlier_step(self):
        future = outcomes(40, drift=F(1, 8))
        base = learner.train_causal_ppo_v7(initial(), future, 11, enabled=True)
        for k in (0, 1, 16, 39):
            with self.subTest(prefix=k):
                changed = future[:k] + tuple(replace(o, bar=replace(o.bar, price=o.bar.price + 3)) for o in future[k:])
                other = learner.train_causal_ppo_v7(initial(), changed, 11, enabled=True)
                self.assertIsNotNone(other)
                self.assertEqual(base.transitions[:k], other.transitions[:k])
                self.assertEqual(base.behavior_probabilities[:k + 1], other.behavior_probabilities[:k + 1])
                self.assertEqual(base.transitions[k].action, other.transitions[k].action)
                self.assertNotEqual(base.transitions[k].receipt, other.transitions[k].receipt)

    def test_multiple_rollouts_keep_boundary_nonterminal_and_update_both_networks(self):
        seen = []
        actual = learner._targets

        def spy(rows, inputs, rewards, critic):
            seen.append((len(rows), critic.step))
            return actual(rows, inputs, rewards, critic)

        with patch.object(learner, "_targets", side_effect=spy):
            result = learner.train_causal_ppo_v7(initial(), outcomes(257, multiplier=F(0)), 23, enabled=True)
        self.assertEqual(seen, [(256, 0), (1, 4)])
        self.assertEqual((result.steps, result.actor.step, result.critic.step, len(result.losses)), (257, 8, 8, 8))
        self.assertFalse(result.transitions[255].done)
        self.assertEqual(result.transitions[255].next_observation, result.transitions[256].observation)

    def test_terminal_rows_stop_without_padding_and_target_is_the_reward(self):
        future = outcomes(3)
        future = (future[0], replace(future[1], bar=replace(future[1].bar, price=F(1000))), None)
        # The jump exhausts equity only for a seed whose first sampled action is short.
        seed = next(s for s in range(64) if train(1, s).transitions[0].action == 0)
        result = learner.train_causal_ppo_v7(initial(), future, seed, enabled=True)
        self.assertIsNotNone(result)
        self.assertEqual((result.requested_steps, result.steps, result.unused_outcomes), (3, 2, 1))
        row = result.transitions[-1]
        self.assertTrue(row.done)
        self.assertIsNone(row.next_observation)
        self.assertEqual(result.stop_reason, "equity_exhausted")
        self.assertEqual(result.targets[-1], float(row.reward))
        closed = learner.train_causal_ppo_v7(initial(), (outcomes(2)[0], replace(outcomes(2)[1], terminal=True)),
                                            11, enabled=True)
        self.assertEqual((closed.stop_reason, closed.session.state.units), ("terminal", 0))
        self.assertEqual(closed.targets[-1], float(closed.transitions[-1].reward))

    def test_probability_floor_and_magnitude_guards_reject_before_use(self):
        self.assertEqual(len(learner._policy((0.0, 0.0, 0.0))), 3)
        self.assertAlmostEqual(sum(learner._policy((1.0, -2.0, 0.5))), 1.0, places=15)
        self.assertEqual(min(learner._policy((0.0, -68.0, 0.0))), math.exp(-68.0) / math.fsum([1.0, math.exp(-68.0), 1.0]))
        for row in ((0.0, -70.0, 0.0), (0.0, -800.0, 0.0), (2.0 ** 99, 0.0, 0.0)):
            with self.subTest(row=row), self.assertRaises(ValueError):
                learner._policy(row)
        for bad in (2.0 ** 99 * (1 + 2.0 ** -52), float("inf"), float("nan"), 1, True, np.float64(1.0)):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                learner._bounded(bad)
        with patch.object(learner, "forward_v5", return_value=np.full((1, 3), 2.0 ** 100)):
            self.assertIsNone(train())
        skewed = np.asarray([[0.0, -80.0, 0.0]])
        with patch.object(learner, "forward_v5", return_value=skewed):
            self.assertIsNone(train(1))

    def test_normalization_counts_square_underflow(self):
        rows = train(2).transitions
        critic = optimizer.create_v5(12, 1, enabled=True)
        inputs = np.asarray([[float(v) for v in row.observation.values] for row in rows])
        pairs = ((2.0 ** -600, 0.0), (-(2.0 ** -600), 0.0))
        with patch.object(learner, "batch_v2", return_value=pairs):
            targets, advantages, underflow = learner._targets(rows, inputs, (0.0, 0.0), critic)
        self.assertEqual((underflow, targets), (2, (0.0, 0.0)))
        self.assertEqual(advantages, (2.0 ** -600 / 1e-08, -(2.0 ** -600) / 1e-08))

    def test_unrepresentable_nonzero_reward_rejects(self):
        # A flat first action has an exactly zero reward, so use a seed that trades.
        seed = next(s for s in range(64) if train(1, s).transitions[0].action != 1)
        flat = next(s for s in range(64) if train(1, s).transitions[0].action == 1)
        tiny = outcomes(1, multiplier=F(1, 2**1100))
        self.assertIsNone(learner.train_causal_ppo_v7(initial(), tiny, seed, enabled=True))
        self.assertEqual(learner.train_causal_ppo_v7(initial(), tiny, flat, enabled=True).transitions[0].reward, 0)

    def test_stage_failures_publish_nothing_and_never_split_the_pair(self):
        for name in ("create_v5", "forward_v5", "transition_v4", "batch_v2", "objective_v2", "update_v5"):
            with self.subTest(stage=name), patch.object(learner, name, return_value=None):
                self.assertIsNone(train())
        for name in ("_number", "_bounded", "_vector", "_policy", "_targets"):
            with self.subTest(stage=name), patch.object(learner, name, side_effect=ValueError("synthetic")):
                self.assertIsNone(train())
        actual = learner.update_v5
        calls = []

        def critic_fails_once(snapshot, *args, **kwargs):
            calls.append((snapshot.outputs, snapshot.step))
            return None if len(calls) == 4 else actual(snapshot, *args, **kwargs)

        with patch.object(learner, "update_v5", side_effect=critic_fails_once):
            self.assertIsNone(train())
        self.assertEqual(calls, [(3, 0), (1, 0), (3, 1), (1, 1)])

    def test_training_rejects_configuration_and_consumed_invalid_outcomes(self):
        bars, future = initial(), outcomes(1)
        self.assertIsNone(learner.train_causal_ppo_v7(bars, future, 11))
        for kwargs in ({"enabled": 1}, {"enabled": "true"}, {"enabled": True, "version": "causal-ppo-v5"}):
            self.assertIsNone(learner.train_causal_ppo_v7(bars, future, 11, **kwargs))
        for seed in (True, -1, 2**32 - 1, 11.):
            self.assertIsNone(learner.train_causal_ppo_v7(bars, future, seed, enabled=True))
        for bad in (None, [], (), future * 4097, (None,), (replace(future[0], terminal=1),),
                    (replace(future[0], events=[]),), (replace(future[0], fill=F(0)),),
                    (replace(future[0], bar=replace(future[0].bar, available=0)),)):
            self.assertIsNone(learner.train_causal_ppo_v7(bars, bad, 11, enabled=True))
        self.assertIsNone(learner.train_causal_ppo_v7(bars[:25], future, 11, enabled=True))

    def test_first_loss_is_the_objective_at_the_seeded_actor(self):
        result = train(17, 47)
        actor = optimizer.create_v5(47, 3, enabled=True)
        inputs = np.asarray([[float(v) for v in row.observation.values] for row in result.transitions])
        logits = tuple(tuple(r) for r in optimizer.forward_v5(actor, inputs, enabled=True).tolist())
        actions = tuple(row.action for row in result.transitions)
        expected = objective.objective_v2(logits, actions, result.behavior_probabilities, result.advantages, enabled=True)
        self.assertEqual(result.losses[0], expected.loss)
        self.assertLess(result.losses[-1], result.losses[0])


if __name__ == "__main__":
    unittest.main()

"""Synthetic engineering fixtures; no financial trials or historical inputs."""
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research"))
import causal_optimizer_v5 as optimizer
import causal_ppo_v5 as learner
import causal_replay_v3 as replay
from sequential_learning import softmax


def initial():
    return tuple(replay.Bar("SYNTH", i * 10, i * 10 + 1, F(100)) for i in range(26))


def outcomes(count, *, multiplier=F(1)):
    return tuple(learner.Outcome(replay.Bar("SYNTH", i * 10, i * 10 + 1, F(100)), (),
                                 fill=F(1, 2), multiplier=multiplier) for i in range(26, 26 + count))


class NumericBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.observation = replay.observe_v3(replay.start_v3(initial(), enabled=True), enabled=True)

    def test_conversion_error_bound_and_extremes(self):
        for value in (F(0), F(1, 3), F(-5, 7), F(2**1023), F(1, 2**1022)):
            with self.subTest(value=value):
                observation = replace(self.observation, values=(value,) * 9)
                encoded = learner.encode_v5(observation, enabled=True)
                self.assertIsNotNone(encoded)
                self.assertEqual(len(encoded), 9)
                for number in encoded:
                    self.assertIs(type(number), float)
                    self.assertLessEqual(abs(F.from_float(number) - value), abs(value) * F(1, 2**52))
        for value in (F(1, 2**1023), F(1, 2**8191), F(2**1024), F(2**8192), 1., True):
            with self.subTest(value=value):
                self.assertIsNone(learner.encode_v5(replace(self.observation, values=(value,) * 9), enabled=True))

    def test_encoding_requires_exact_schema_header_and_opt_in(self):
        self.assertIsNone(learner.encode_v5(self.observation))
        for kwargs in ({"enabled": 1}, {"enabled": "true"}, {"enabled": True, "version": replay.VERSION}):
            self.assertIsNone(learner.encode_v5(self.observation, **kwargs))
        for bad in (None, replace(self.observation, values=(F(0),) * 12),
                    replace(self.observation, version="sequential-observation-v1"),
                    replace(self.observation, decision=True), replace(self.observation, decision=2**63),
                    replace(self.observation, supported=1), replace(self.observation, values=[F(0)] * 9)):
            self.assertIsNone(learner.encode_v5(bad, enabled=True))


class FunctionalOptimizerTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = optimizer.create_v5(11, 1, enabled=True)
        self.x = np.linspace(-.5, .5, 18).reshape(2, 9)

    def test_gradient_direction_matches_finite_differences_and_decreases_loss(self):
        snapshot = self.snapshot
        target = np.asarray([[.1], [-.2]])

        def loss(candidate):
            predictions = optimizer.forward_v5(candidate, self.x, enabled=True)
            return float(np.sum((predictions - target)**2) / 2)

        predictions = optimizer.forward_v5(snapshot, self.x, enabled=True)
        candidate = optimizer.update_v5(snapshot, self.x, predictions - target, enabled=True)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.step, 1)
        self.assertEqual(snapshot.step, 0)
        self.assertLess(loss(candidate), loss(snapshot))
        for group in range(4):
            raw = np.frombuffer(snapshot.p[group], dtype="<f8").copy()
            next_raw = np.frombuffer(candidate.p[group], dtype="<f8")
            # Test the strongest coordinate in every layer/bias group.
            index = int(np.argmax(abs(next_raw - raw)))
            probes = []
            for epsilon in (-1e-6, 1e-6):
                changed = raw.copy()
                changed[index] += epsilon
                buffers = list(snapshot.p)
                buffers[group] = changed.tobytes()
                probes.append(loss(replace(snapshot, p=tuple(buffers))))
            derivative = (probes[1] - probes[0]) / 2e-6
            self.assertLess(derivative * float(next_raw[index] - raw[index]), 0)
        with self.assertRaises(FrozenInstanceError):
            snapshot.step = 1

    def test_failures_never_change_old_snapshot_or_caller_arrays(self):
        before = repr(self.snapshot)
        x = self.x.copy()
        dz = np.ones((2, 1))
        for bad_lr in (True, 1, 0., -1., float("nan"), float("inf")):
            self.assertIsNone(optimizer.update_v5(self.snapshot, self.x, dz, bad_lr, enabled=True))
        for bad in (None, self.x.tolist(), self.x.astype("float32"), np.zeros((0, 9)),
                    np.zeros((257, 9)), np.zeros((2, 12)), np.full((2, 9), np.inf)):
            self.assertIsNone(optimizer.forward_v5(self.snapshot, bad, enabled=True))
            self.assertIsNone(optimizer.update_v5(self.snapshot, bad, dz, enabled=True))
        self.assertIsNone(optimizer.update_v5(self.snapshot, x, np.ones((1, 1)), enabled=True))
        self.assertIsNone(optimizer.update_v5(self.snapshot, x, np.full((2, 1), 1e308), enabled=True))
        self.assertIsNone(optimizer.update_v5(replace(self.snapshot, step=optimizer.STEP_CAP), x, dz, enabled=True))
        output = optimizer.forward_v5(self.snapshot, self.x, enabled=True)
        output[:] = 99
        self.assertFalse(np.any(optimizer.forward_v5(self.snapshot, self.x, enabled=True) == 99))
        self.assertEqual(repr(self.snapshot), before)
        np.testing.assert_array_equal(self.x, x)
        np.testing.assert_array_equal(dz, np.ones((2, 1)))

    def test_snapshot_schema_buffer_and_numeric_corruption_reject(self):
        nan = (np.full(9 * 16, np.nan).tobytes(),) + self.snapshot.p[1:]
        negative = (np.full(9 * 16, -1.).tobytes(),) + self.snapshot.v[1:]
        for bad in (None, replace(self.snapshot, version="optimizer-snapshot-v2"),
                    replace(self.snapshot, observation_version="frozen-twelve-input"),
                    replace(self.snapshot, outputs=True), replace(self.snapshot, step=True),
                    replace(self.snapshot, step=-1), replace(self.snapshot, p=()),
                    replace(self.snapshot, p=(b"",) + self.snapshot.p[1:]),
                    replace(self.snapshot, p=nan), replace(self.snapshot, v=negative)):
            self.assertIsNone(optimizer.forward_v5(bad, self.x, enabled=True))
            self.assertIsNone(optimizer.update_v5(bad, self.x, np.ones((2, 1)), enabled=True))

    def test_optimizer_entries_require_strict_configuration_and_opt_in(self):
        self.assertIsNone(optimizer.create_v5(11))
        self.assertIsNone(optimizer.forward_v5(self.snapshot, self.x))
        self.assertIsNone(optimizer.update_v5(self.snapshot, self.x, np.zeros((2, 1))))
        for kwargs in ({"enabled": 1}, {"enabled": True, "version": "old"}):
            self.assertIsNone(optimizer.create_v5(11, **kwargs))
            self.assertIsNone(optimizer.forward_v5(self.snapshot, self.x, **kwargs))
            self.assertIsNone(optimizer.update_v5(self.snapshot, self.x, np.zeros((2, 1)), **kwargs))
        for seed in (True, -1, 2**32, 11.):
            self.assertIsNone(optimizer.create_v5(seed, enabled=True))
        for outputs in (True, 0, 2, 12):
            self.assertIsNone(optimizer.create_v5(11, outputs, enabled=True))


class CausalTrainingTests(unittest.TestCase):
    def train(self, count=17, seed=11, **kwargs):
        return learner.train_causal_ppo_v5(initial(), outcomes(count), seed, enabled=True, **kwargs)

    def test_actual_training_repeatability_and_exact_replay_conformance(self):
        for seed in (11, 23, 47):
            for count in (1, 17):
                with self.subTest(seed=seed, count=count):
                    bars, future = initial(), outcomes(count)
                    before = repr((bars, future))
                    result = learner.train_causal_ppo_v5(bars, future, seed, enabled=True)
                    self.assertIsNotNone(result)
                    self.assertEqual(result, learner.train_causal_ppo_v5(bars, future, seed, enabled=True))
                    self.assertEqual((result.steps, result.unused_outcomes), (count, 0))
                    self.assertEqual((result.actor.step, result.critic.step), (4, 4))
                    self.assertEqual(len(result.losses), 4)
                    self.assertEqual(result.stop_reason, "input_exhausted")
                    self.assertEqual(result.actor.observation_version, replay.OBSERVATION)
                    session = replay.start_v3(bars, enabled=True)
                    for row, outcome in zip(result.transitions, future):
                        expected, receipt = replay.step_v3(session, outcome.bar, outcome.events,
                            (F(-1, 4), F(0), F(1, 4))[row.action], terminal=outcome.terminal,
                            fill=outcome.fill, multiplier=outcome.multiplier, impact=outcome.impact, enabled=True)
                        self.assertEqual(row.receipt, receipt)
                        self.assertEqual(row.observation, replay.observe_v3(session, enabled=True))
                        self.assertEqual(row.reward, receipt.after.equity - receipt.before.equity)
                        self.assertEqual(expected.scale, session.scale)
                        session = expected
                    self.assertEqual(result.session, session)
                    self.assertEqual(repr((bars, future)), before)

    def test_multiple_batches_keep_boundary_nonterminal_and_publish_all_updates(self):
        future = outcomes(257, multiplier=F(0))
        calls = []
        actual = learner.batch_v2

        def spy(rows, gamma, **kwargs):
            calls.append(rows)
            return actual(rows, gamma, **kwargs)

        with patch.object(learner, "batch_v2", side_effect=spy):
            result = learner.train_causal_ppo_v5(initial(), future, 23, enabled=True)
        self.assertIsNotNone(result)
        self.assertEqual([len(rows) for rows in calls], [256, 1])
        self.assertEqual(result.steps, 257)
        self.assertEqual(result.actor.step, 8)
        self.assertEqual(result.critic.step, 8)
        self.assertEqual(len(result.losses), 8)
        self.assertFalse(result.transitions[255].done)
        self.assertEqual(result.transitions[255].next_observation, result.transitions[256].observation)
        self.assertFalse(calls[0][-1][3])
        self.assertFalse(result.session.state.terminal)

    def test_future_rewrite_cannot_change_earlier_observations_actions_or_probabilities(self):
        future = outcomes(17)
        changed = future[:-1] + (replace(future[-1], bar=replace(future[-1].bar, price=F(101))),)
        a = learner.train_causal_ppo_v5(initial(), future, 11, enabled=True)
        b = learner.train_causal_ppo_v5(initial(), changed, 11, enabled=True)
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertEqual(a.transitions[:-1], b.transitions[:-1])
        self.assertEqual(a.transitions[-1].observation, b.transitions[-1].observation)
        self.assertEqual(a.transitions[-1].action, b.transitions[-1].action)
        self.assertEqual(a.behavior_probabilities, b.behavior_probabilities)
        self.assertEqual(a.session.scale, b.session.scale)
        self.assertNotEqual(a.transitions[-1].reward, b.transitions[-1].reward)

    def test_neutralized_target_retains_probability_of_requested_action(self):
        bars = initial()[:-1] + (replace(initial()[-1], price=F(101)),)
        calls = []
        actual = learner.ppo_gradient

        def spy(logits, actions, probabilities, advantages):
            calls.append((actions.copy(), probabilities.copy()))
            return actual(logits, actions, probabilities, advantages)

        with patch.object(learner, "ppo_gradient", side_effect=spy):
            result = learner.train_causal_ppo_v5(bars, outcomes(1), 11, enabled=True)
        self.assertIsNotNone(result)
        row = result.transitions[0]
        self.assertFalse(row.observation.supported)
        self.assertNotEqual(row.action, 1)
        self.assertEqual(row.replay_target, 0)
        actor = optimizer.create_v5(11, enabled=True)
        encoded = np.asarray(learner.encode_v5(row.observation, enabled=True))[None, :]
        probabilities = softmax(optimizer.forward_v5(actor, encoded, enabled=True))[0]
        self.assertEqual(result.behavior_probabilities[0], float(probabilities[row.action]))
        self.assertNotEqual(result.behavior_probabilities[0], float(probabilities[1]))
        self.assertEqual(len(calls), 4)
        for actions, old_probabilities in calls:
            self.assertEqual(actions.tolist(), [row.action])
            self.assertEqual(old_probabilities.tolist(), list(result.behavior_probabilities))

    def test_unrepresentable_nonzero_reward_rejects_instead_of_rounding_to_zero(self):
        future = outcomes(1, multiplier=F(1, 2**1100))
        session = replay.start_v3(initial(), enabled=True)
        _, receipt = replay.step_v3(session, future[0].bar, (), F(-1, 4),
                                    fill=future[0].fill, multiplier=future[0].multiplier, enabled=True)
        self.assertNotEqual(receipt.after.equity - receipt.before.equity, 0)
        self.assertIsNone(learner.train_causal_ppo_v5(initial(), future, 11, enabled=True))

    def test_terminal_losses_stop_without_padding_and_have_no_bootstrap(self):
        future = outcomes(3)
        future = (future[0], replace(future[1], bar=replace(future[1].bar, price=F(1000))), None)
        calls = []
        actual = learner.batch_v2

        def spy(rows, gamma, **kwargs):
            pairs = actual(rows, gamma, **kwargs)
            calls.append((rows, pairs))
            return pairs

        with patch.object(learner, "batch_v2", side_effect=spy):
            result = learner.train_causal_ppo_v5(initial(), future, 11, enabled=True)
        self.assertIsNotNone(result)
        self.assertEqual((result.requested_steps, result.steps, result.unused_outcomes), (3, 2, 1))
        row = result.transitions[-1]
        self.assertTrue(row.done)
        self.assertIsNone(row.next_observation)
        self.assertLessEqual(row.receipt.after.equity, 0)
        self.assertLess(row.reward, 0)
        self.assertEqual(result.stop_reason, "equity_exhausted")
        self.assertEqual(calls[0][0][-1][2:], (0., True))
        self.assertEqual(calls[0][1][-1][1], float(row.reward))

    def test_explicit_terminal_liquidation_and_input_exhaustion_are_distinct(self):
        future = outcomes(2)
        future = (future[0], replace(future[1], terminal=True))
        closed = learner.train_causal_ppo_v5(initial(), future, 11, enabled=True)
        open_result = self.train(count=1)
        self.assertIsNotNone(closed)
        self.assertIsNotNone(open_result)
        self.assertEqual(closed.stop_reason, "terminal")
        self.assertTrue(closed.transitions[-1].done)
        self.assertEqual(closed.session.state.units, 0)
        self.assertEqual(closed.transitions[-1].receipt.liquidation, "flat")
        self.assertFalse(open_result.transitions[-1].done)
        self.assertNotEqual(open_result.session.state.units, 0)

    def test_stage_failures_discard_every_candidate_and_later_rollout_progress(self):
        for name in ("create_v5", "forward_v5", "encode_v5", "transition_v4", "batch_v2", "update_v5"):
            with self.subTest(stage=name), patch.object(learner, name, return_value=None):
                self.assertIsNone(self.train())
        with patch.object(learner, "softmax", return_value=np.full((1, 3), np.nan)):
            self.assertIsNone(self.train())
        with patch.object(learner, "ppo_gradient", return_value=(float("nan"), np.zeros((17, 3)))):
            self.assertIsNone(self.train())
        with patch.object(learner, "_number", side_effect=ValueError("synthetic numeric failure")):
            self.assertIsNone(self.train())
        actual = learner.update_v5
        calls = []

        def fail_after_one_rollout(*args, **kwargs):
            calls.append(args[0])
            return None if len(calls) == 9 else actual(*args, **kwargs)

        with patch.object(learner, "update_v5", side_effect=fail_after_one_rollout):
            result = learner.train_causal_ppo_v5(initial(), outcomes(257, multiplier=F(0)), 23, enabled=True)
        self.assertIsNone(result)
        self.assertGreaterEqual(len(calls), 9)
        self.assertEqual(calls[0].step, 0)
        self.assertEqual(calls[8].step, 4)

    def test_training_rejects_configuration_and_consumed_invalid_outcomes(self):
        bars, future = initial(), outcomes(1)
        self.assertIsNone(learner.train_causal_ppo_v5(bars, future, 11))
        for kwargs in ({"enabled": 1}, {"enabled": True, "version": "ppo-successor-v2"}):
            self.assertIsNone(learner.train_causal_ppo_v5(bars, future, 11, **kwargs))
        for seed in (True, -1, 2**32 - 1, 11.):
            self.assertIsNone(learner.train_causal_ppo_v5(bars, future, seed, enabled=True))
        for bad in (None, [], (), future * 4097, (None,), (replace(future[0], terminal=1),),
                    (replace(future[0], events=[]),), (replace(future[0], fill=F(0)),),
                    (replace(future[0], bar=replace(future[0].bar, available=0)),)):
            self.assertIsNone(learner.train_causal_ppo_v5(bars, bad, 11, enabled=True))
        self.assertIsNone(learner.train_causal_ppo_v5(bars[:25], future, 11, enabled=True))


if __name__ == "__main__":
    unittest.main()

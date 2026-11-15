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
import causal_replay_v3 as replay
import causal_value_v6 as learner
import value_objective_v2 as objective


def initial():
    return tuple(replay.Bar("SYNTH", i * 10, i * 10 + 1, F(100)) for i in range(26))


def outcomes(count, *, multiplier=F(1), drift=0):
    return tuple(learner.Outcome(replay.Bar("SYNTH", i * 10, i * 10 + 1, F(100) + drift * (i % 7 - 3)), (),
                                 fill=F(1, 2), multiplier=multiplier) for i in range(26, 26 + count))


def train(count=17, seed=11, **kwargs):
    return learner.train_causal_value_v6(initial(), outcomes(count), seed, enabled=True, **kwargs)


def exact_target(reward, selected):
    """Two correctly rounded binary64 operations, evaluated with exact rationals."""
    product = float(F(learner.GAMMA) * F(selected))
    return float(F(reward) + F(product))


def following(snapshot, rows):
    batch = np.asarray([[float(v) for v in row.next_observation.values] for row in rows])
    return optimizer.forward_v5(snapshot, batch, enabled=True).tolist()


class ImportClosureTests(unittest.TestCase):
    def test_learner_loads_no_frozen_module(self):
        frozen = {"sequential_learning", "sequential_env", "sequential_evaluation", "sequential_registry",
                  "run_sequential_screen", "causal_ppo_v5", "ppo_successor_v2", "gae_targets_v2"}
        import subprocess
        code = ("import sys; sys.path.insert(0, 'scripts/research'); import causal_value_v6; "
                "print(sorted(set(sys.modules) & %r))" % frozen)
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[1], timeout=60)
        self.assertEqual((done.returncode, done.stdout.strip()), (0, "[]"), done.stderr)


class CausalValueTrainingTests(unittest.TestCase):
    def test_repeatability_exact_replay_and_exact_behavior_probabilities(self):
        for seed in (11, 23, 47):
            for count in (1, 17):
                with self.subTest(seed=seed, count=count):
                    bars, future = initial(), outcomes(count, drift=F(1, 8))
                    before = repr((bars, future))
                    result = learner.train_causal_value_v6(bars, future, seed, enabled=True)
                    self.assertIsNotNone(result)
                    self.assertEqual(result, learner.train_causal_value_v6(bars, future, seed, enabled=True))
                    self.assertEqual((result.steps, result.unused_outcomes), (count, 0))
                    self.assertEqual((result.online.step, result.online.outputs), (4, 3))
                    self.assertIs(result.target, result.online)
                    self.assertEqual((len(result.losses), len(result.targets)), (4, count))
                    self.assertEqual(result.stop_reason, "input_exhausted")
                    self.assertEqual((result.version, result.objective_version, result.optimizer_version),
                                     ("causal-value-v6", objective.VERSION, optimizer.VERSION))
                    self.assertTrue(all(type(x) is float and np.isfinite(x) and x >= 0 for x in result.losses))
                    self.assertTrue(all(type(y) is float and abs(y) <= 2.0 ** 100 for y in result.targets))
                    self.assertIs(type(result.underflow), int)
                    session = replay.start_v3(bars, enabled=True)
                    for row, outcome in zip(result.transitions, future):
                        expected, receipt = replay.step_v3(session, outcome.bar, outcome.events,
                            (F(-1, 4), F(0), F(1, 4))[row.action], terminal=outcome.terminal,
                            fill=outcome.fill, multiplier=outcome.multiplier, impact=outcome.impact, enabled=True)
                        self.assertEqual(row.receipt, receipt)
                        self.assertEqual(row.observation, replay.observe_v3(session, enabled=True))
                        self.assertEqual(row.reward, receipt.after.equity - receipt.before.equity)
                        session = expected
                    self.assertEqual(result.session, session)
                    self.assertEqual(repr((bars, future)), before)
                    with self.assertRaises(FrozenInstanceError):
                        result.steps = 0

    def test_behavior_probability_is_the_exact_rule_for_the_requested_action(self):
        result = train(257, 23)
        self.assertIsNotNone(result)
        self.assertEqual((learner.GREEDY, learner.OTHER), (F(11, 12), F(1, 24)))
        self.assertEqual(learner.GREEDY + 2 * learner.OTHER, 1)
        self.assertEqual(set(result.behavior_probabilities), {F(11, 12), F(1, 24)})
        # First rollout: the collecting snapshot is the seeded initial one.
        snapshot = optimizer.create_v5(23, 3, enabled=True)
        for row, probability in list(zip(result.transitions, result.behavior_probabilities))[:256]:
            x = np.asarray([float(v) for v in row.observation.values])[None, :]
            q = optimizer.forward_v5(snapshot, x, enabled=True)[0].tolist()
            self.assertEqual(probability, F(11, 12) if row.action == q.index(max(q)) else F(1, 24))

    def test_random_stream_position_is_independent_of_data(self):
        draws = []

        class Spy:
            def __init__(self, inner):
                self.inner = inner

            def random(self):
                draws.append("u")
                return self.inner.random()

            def integers(self, n):
                draws.append("i")
                return self.inner.integers(n)

        actual = np.random.default_rng
        # The optimizer seeds with an int; only the learner's own stream is observed.
        with patch.object(learner.np.random, "default_rng",
                          side_effect=lambda seed: Spy(actual(seed)) if type(seed) is list else actual(seed)):
            self.assertIsNotNone(train(17))
        self.assertEqual(draws, ["u", "i"] * 17)

    def test_first_rollout_targets_match_exact_double_dqn_oracle(self):
        for seed in (11, 23, 47):
            with self.subTest(seed=seed):
                result = learner.train_causal_value_v6(initial(), outcomes(17, drift=F(1, 8)), seed, enabled=True)
                self.assertIsNotNone(result)
                snapshot = optimizer.create_v5(seed, 3, enabled=True)
                # Network outputs depend on batch shape at the last-bit level, so the oracle
                # evaluates the same stacked batch and checks the two scalar operations exactly.
                values = following(snapshot, result.transitions)
                for row, q, target in zip(result.transitions, values, result.targets):
                    self.assertFalse(row.done)
                    # Online and target snapshots coincide in the first rollout.
                    self.assertEqual(target.hex(), exact_target(float(row.reward), max(q)).hex())

    def test_double_dqn_selects_with_online_and_evaluates_with_target(self):
        rows = train(2).transitions
        online = optimizer.create_v5(101, 3, enabled=True)
        target = optimizer.create_v5(202, 3, enabled=True)
        rewards = tuple(float(row.reward) for row in rows)
        targets, underflow = learner._targets(rows, rewards, online, target)
        self.assertEqual(underflow, 0)
        different = 0
        for reward, got, o, t in zip(rewards, targets, following(online, rows), following(target, rows)):
            self.assertEqual(got.hex(), exact_target(reward, t[o.index(max(o))]).hex())
            different += t[o.index(max(o))] != max(t)
        self.assertGreater(different, 0, "fixture must distinguish Double-DQN from max over the target network")

    def test_future_rewrite_cannot_change_any_earlier_step(self):
        future = outcomes(40, drift=F(1, 8))
        base = learner.train_causal_value_v6(initial(), future, 11, enabled=True)
        self.assertIsNotNone(base)
        for k in (0, 1, 16, 39):
            with self.subTest(prefix=k):
                changed = future[:k] + tuple(replace(o, bar=replace(o.bar, price=o.bar.price + 3))
                                              for o in future[k:])
                other = learner.train_causal_value_v6(initial(), changed, 11, enabled=True)
                self.assertIsNotNone(other)
                self.assertEqual(base.transitions[:k], other.transitions[:k])
                self.assertEqual(base.behavior_probabilities[:k + 1], other.behavior_probabilities[:k + 1])
                self.assertEqual(base.transitions[k].observation, other.transitions[k].observation)
                self.assertEqual(base.transitions[k].action, other.transitions[k].action)
                self.assertNotEqual(base.transitions[k].receipt, other.transitions[k].receipt)

    def test_multiple_rollouts_keep_boundary_nonterminal_and_replace_target_after_updates(self):
        seen = []
        actual = learner._targets

        def spy(rows, rewards, online, target):
            seen.append((len(rows), online, target))
            return actual(rows, rewards, online, target)

        with patch.object(learner, "_targets", side_effect=spy):
            result = learner.train_causal_value_v6(initial(), outcomes(257, multiplier=F(0)), 23, enabled=True)
        self.assertIsNotNone(result)
        self.assertEqual([n for n, _, _ in seen], [256, 1])
        self.assertTrue(all(online is target for _, online, target in seen))
        self.assertEqual([online.step for _, online, _ in seen], [0, 4])
        self.assertEqual((result.steps, result.online.step, len(result.losses), len(result.targets)), (257, 8, 8, 257))
        self.assertFalse(result.transitions[255].done)
        self.assertEqual(result.transitions[255].next_observation, result.transitions[256].observation)
        self.assertFalse(result.session.state.terminal)

    def test_terminal_rows_stop_without_padding_and_take_no_bootstrap(self):
        future = outcomes(3)
        future = (future[0], replace(future[1], bar=replace(future[1].bar, price=F(1000))), None)
        result = learner.train_causal_value_v6(initial(), future, 11, enabled=True)
        self.assertIsNotNone(result)
        self.assertEqual((result.requested_steps, result.steps, result.unused_outcomes), (3, 2, 1))
        row = result.transitions[-1]
        self.assertTrue(row.done)
        self.assertIsNone(row.next_observation)
        self.assertLess(row.reward, 0)
        self.assertEqual(result.stop_reason, "equity_exhausted")
        self.assertEqual(result.targets[-1], float(row.reward))
        closed = learner.train_causal_value_v6(initial(), (outcomes(2)[0], replace(outcomes(2)[1], terminal=True)),
                                              11, enabled=True)
        self.assertEqual((closed.stop_reason, closed.session.state.units), ("terminal", 0))
        self.assertEqual(closed.targets[-1], float(closed.transitions[-1].reward))

    def test_neutralized_target_keeps_requested_action_and_its_probability(self):
        bars = initial()[:-1] + (replace(initial()[-1], price=F(101)),)
        for seed in range(64):
            result = learner.train_causal_value_v6(bars, outcomes(1), seed, enabled=True)
            self.assertIsNotNone(result)
            row = result.transitions[0]
            self.assertFalse(row.observation.supported)
            self.assertEqual(row.replay_target, 0)
            if row.action != 1:
                break
        self.assertNotEqual(row.action, 1)
        self.assertIn(result.behavior_probabilities[0], (F(11, 12), F(1, 24)))

    def test_magnitude_guards_reject_before_use_and_count_underflow(self):
        self.assertEqual(learner._bounded(2.0 ** 99), 2.0 ** 99)
        for bad in (2.0 ** 99 * (1 + 2.0 ** -52), float("inf"), float("nan"), 1, True, F(1), np.float64(1.0)):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                learner._bounded(bad)
        huge = np.full((17, 3), 2.0 ** 100)
        with patch.object(learner, "forward_v5", return_value=huge):
            self.assertIsNone(train())
        # Worst admitted corner stays inside the objective domain.
        worst = exact_target(2.0 ** 99, 2.0 ** 99)
        self.assertLessEqual(worst, 2.0 ** 100)
        self.assertEqual(objective.objective_v2(((0.0, 0.0, 0.0),), (0,), (worst,), 0.1, enabled=True).rows, 1)
        rows = train(1).transitions
        snapshot = optimizer.create_v5(11, 3, enabled=True)
        tiny = ((2.0 ** -1040, 0.0, -1.0),)
        with patch.object(learner, "_values", return_value=tiny):
            targets, underflow = learner._targets(rows, (0.0,), snapshot, snapshot)
        self.assertEqual((underflow, targets[0]), (1, 0.99 * 2.0 ** -1040))
        with patch.object(learner, "_values", return_value=((0.0, -1.0, -2.0),)):
            self.assertEqual(learner._targets(rows, (0.5,), snapshot, snapshot), ((0.5,), 0))

    def test_unrepresentable_nonzero_reward_rejects_instead_of_rounding_to_zero(self):
        self.assertIsNone(learner.train_causal_value_v6(initial(), outcomes(1, multiplier=F(1, 2**1100)), 11,
                                                       enabled=True))

    def test_stage_failures_publish_nothing(self):
        for name in ("create_v5", "forward_v5", "transition_v4", "objective_v2", "update_v5"):
            with self.subTest(stage=name), patch.object(learner, name, return_value=None):
                self.assertIsNone(train())
        for name in ("_number", "_bounded", "_vector", "_targets"):
            with self.subTest(stage=name), patch.object(learner, name, side_effect=ValueError("synthetic")):
                self.assertIsNone(train())
        actual = learner.update_v5
        calls = []

        def fail_in_second_rollout(*args, **kwargs):
            calls.append(args[0])
            return None if len(calls) == 5 else actual(*args, **kwargs)

        with patch.object(learner, "update_v5", side_effect=fail_in_second_rollout):
            self.assertIsNone(learner.train_causal_value_v6(initial(), outcomes(257, multiplier=F(0)), 23,
                                                           enabled=True))
        self.assertEqual([s.step for s in calls], [0, 1, 2, 3, 4])

    def test_training_rejects_configuration_and_consumed_invalid_outcomes(self):
        bars, future = initial(), outcomes(1)
        self.assertIsNone(learner.train_causal_value_v6(bars, future, 11))
        for kwargs in ({"enabled": 1}, {"enabled": "true"}, {"enabled": True, "version": "causal-ppo-v5"}):
            self.assertIsNone(learner.train_causal_value_v6(bars, future, 11, **kwargs))
        for seed in (True, -1, 2**32, 11.):
            self.assertIsNone(learner.train_causal_value_v6(bars, future, seed, enabled=True))
        for bad in (None, [], (), future * 4097, (None,), (replace(future[0], terminal=1),),
                    (replace(future[0], events=[]),), (replace(future[0], fill=F(0)),),
                    (replace(future[0], bar=replace(future[0].bar, available=0)),)):
            self.assertIsNone(learner.train_causal_value_v6(bars, bad, 11, enabled=True))
        self.assertIsNone(learner.train_causal_value_v6(bars[:25], future, 11, enabled=True))

    def test_updates_follow_the_objective_gradient(self):
        result = train(17, 47)
        snapshot = optimizer.create_v5(47, 3, enabled=True)
        inputs = np.asarray([[float(v) for v in row.observation.values] for row in result.transitions])
        actions = tuple(row.action for row in result.transitions)

        def loss(candidate):
            q = tuple(tuple(r) for r in optimizer.forward_v5(candidate, inputs, enabled=True).tolist())
            return objective.objective_v2(q, actions, result.targets, learner.ALPHA, enabled=True).loss

        self.assertEqual(result.losses[0], loss(snapshot))
        self.assertLess(result.losses[-1], result.losses[0])


if __name__ == "__main__":
    unittest.main()

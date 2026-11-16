"""Synthetic engineering conformance; no policy fitting or historical data."""
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research"))
import causal_replay_v3 as replay
import causal_transition_v4 as adapter
import replay_accounting_v2 as accounting


class CausalTransitionTests(unittest.TestCase):
    def setUp(self):
        bars = tuple(replay.Bar("SYNTH", i * 10, i * 10 + 1, F(100)) for i in range(26))
        self.session = replay.start_v3(bars, enabled=True)
        self.assertIsNotNone(self.session)

    def bar(self, session=None, price=F(100)):
        previous = (session or self.session).history[-1]
        return replay.Bar("SYNTH", previous.close + 10, previous.available + 10, price)

    def step(self, session=None, action=2, **kwargs):
        session = session or self.session
        return adapter.transition_v4(session, self.bar(session), (), action, enabled=True, **kwargs)

    def test_exact_receipts_rewards_and_nine_input_schema(self):
        for action, target in enumerate(adapter.TARGETS):
            with self.subTest(action=action):
                before = self.session
                next_bar = self.bar()
                following, row = self.step(action=action)
                direct, receipt = replay.step_v3(before, next_bar, (), target, enabled=True)
                self.assertEqual(following, direct)
                self.assertEqual(row.receipt, receipt)
                self.assertEqual(row.observation, replay.observe_v3(before, enabled=True))
                self.assertEqual(row.next_observation, replay.observe_v3(following, enabled=True))
                self.assertEqual(len(row.observation.values), 9)
                self.assertEqual(row.version, adapter.VERSION)
                self.assertEqual(row.reward_version, adapter.REWARD)
                self.assertEqual(row.observation.version, replay.OBSERVATION)
                self.assertEqual(row.action, action)
                self.assertEqual(row.replay_target, target)
                self.assertIs(type(row.reward), F)
                self.assertEqual(row.reward, receipt.after.equity - receipt.before.equity)
                self.assertFalse(row.done)
                self.assertEqual(before.state.tick, 0)

    def test_support_neutralization_preserves_requested_action(self):
        # Old flat-price observation is supported; its next observation is not.
        changed = self.bar(price=F(101))
        session, first = adapter.transition_v4(self.session, changed, (), 2, enabled=True)
        self.assertEqual(first.replay_target, F(1, 4))
        self.assertFalse(first.next_observation.supported)
        after, row = adapter.transition_v4(session, self.bar(session, F(101)), (), 0,
                                          fill=F(1, 2), enabled=True)
        self.assertEqual(row.action, 0)
        self.assertEqual(row.replay_target, F(0))
        self.assertFalse(row.observation.supported)
        # A neutral target under a partial fill need not execute a flat position.
        self.assertEqual(after.state.units, session.state.units / 2)
        self.assertNotEqual(after.state.units, 0)

    def test_funding_partial_fill_and_liquidation_reconcile(self):
        opened, opening = self.step(fill=F(1, 2), multiplier=F(2), impact=F(1, 100))
        events = ((F(101), F(1, 1000)), (F(102), F(-1, 2000)))
        bar = self.bar(opened, F(102))
        following, row = adapter.transition_v4(opened, bar, events, 0, terminal=True,
                                              multiplier=F(2), impact=F(1, 100), enabled=True)
        direct, receipt = replay.step_v3(opened, bar, events, F(-1, 4), terminal=True,
                                         multiplier=F(2), impact=F(1, 100), enabled=True)
        self.assertEqual((following, row.receipt), (direct, receipt))
        debit = sum((receipt.costs.fee, receipt.costs.spread,
                     receipt.costs.slippage, receipt.costs.impact), F(0))
        self.assertEqual(row.reward, receipt.gross + receipt.funding - debit)
        self.assertLess(opening.reward, 0)
        self.assertNotEqual(receipt.funding, 0)
        self.assertGreater(debit, 0)
        self.assertTrue(row.done)
        self.assertIsNone(row.next_observation)
        self.assertEqual(receipt.liquidation, "flat")
        self.assertEqual(following.state.units, 0)
        self.assertIsNone(self.step(following))

    def test_risk_and_insolvent_terminal_samples_are_retained(self):
        opened, _ = self.step(action=0)
        for price in (F(190), F(1000)):
            with self.subTest(price=price):
                _, row = adapter.transition_v4(opened, self.bar(opened, price), (), 2, enabled=True)
                self.assertTrue(row.done)
                self.assertIsNone(row.next_observation)
                self.assertIsNotNone(row.receipt.reason)
                self.assertLess(row.reward, 0)
                self.assertEqual(row.reward, row.receipt.after.equity - opened.state.equity)
                if row.receipt.after.equity <= 0:
                    self.assertEqual(row.receipt.liquidation, "failed")

    def test_tick_boundary_is_terminal_without_caller_flag(self):
        # A valid consumed prefix immediately before the accounting tick cap.
        bars = tuple(replay.Bar("SYNTH", i * 10, i * 10 + 1, F(100)) for i in range(26 + 4095))
        session = replace(self.session, history=bars,
                          state=replace(self.session.state, tick=4095))
        self.assertIsNotNone(replay.observe_v3(session, enabled=True))
        following, row = self.step(session)
        self.assertEqual(following.state.tick, 4096)
        self.assertTrue(row.done)
        self.assertIsNone(row.next_observation)
        self.assertEqual(row.reward, 0)

    def test_future_rewrites_preserve_past_samples_and_scale(self):
        def run(last_price):
            session = self.session
            rows = []
            for price in (F(100), F(100), last_price):
                session, row = adapter.transition_v4(session, self.bar(session, price), (), 2, enabled=True)
                rows.append(row)
                self.assertEqual(session.scale, self.session.scale)
            return tuple(rows)
        normal, rewritten = run(F(101)), run(F(103))
        self.assertEqual(normal[:2], rewritten[:2])
        self.assertEqual(normal[2].observation, rewritten[2].observation)
        self.assertNotEqual(normal[2].reward, rewritten[2].reward)
        with self.assertRaises(FrozenInstanceError):
            normal[0].action = 0
        with self.assertRaises(FrozenInstanceError):
            normal[0].receipt.after.equity = F(99)

    def test_disabled_and_invalid_actions_never_admit_a_sample(self):
        args = (self.session, self.bar(), (), 2)
        self.assertIsNone(adapter.transition_v4(*args))
        for gate in (False, None, 1, "true"):
            self.assertIsNone(adapter.transition_v4(*args, enabled=gate))
        for version in (None, 4, replay.VERSION, "causal-transition-v3"):
            self.assertIsNone(adapter.transition_v4(*args, enabled=True, version=version))
        for action in (None, True, False, -1, 3, 1., F(1), "1", (1,)):
            self.assertIsNone(self.step(action=action))
        self.assertEqual(self.session.state.tick, 0)

    def test_invalid_session_bar_events_and_execution_reject_immutably(self):
        snapshot = repr(self.session)
        for session in (None, replace(self.session, version="bad"),
                        replace(self.session, scale=()),
                        replace(self.session, state=replace(self.session.state, terminal=True))):
            self.assertIsNone(adapter.transition_v4(session, self.bar(), (), 2, enabled=True))
        for bar in (None, replace(self.bar(), symbol="OTHER"),
                    replace(self.bar(), close=self.session.history[-1].available),
                    replace(self.bar(), available=0), replace(self.bar(), price=F(0)),
                    replace(self.bar(), price=F(2**accounting.MAX_BITS))):
            self.assertIsNone(adapter.transition_v4(self.session, bar, (), 2, enabled=True))
        for events in (None, [], ((F(0), F(1)),), ((F(100), 0.),),
                       ((F(100), F(0)),) * (accounting.MAX_EVENTS + 1)):
            self.assertIsNone(adapter.transition_v4(self.session, self.bar(), events, 2, enabled=True))
        for kwargs in ({"terminal": 1}, {"terminal": None}, {"fill": F(0)},
                       {"fill": F(2)}, {"multiplier": F(-1)}, {"impact": F(-1)}):
            self.assertIsNone(self.step(**kwargs))
        self.assertEqual(repr(self.session), snapshot)

    def test_numeric_failure_in_next_observation_publishes_nothing(self):
        # A valid return can overflow normalization against the tiny burn-in
        # width; the adapter must not expose a nonterminal partial row.
        tiny = F(1, 2**4095)
        bars = tuple(replace(b, price=tiny) for b in self.session.history)
        session = replay.start_v3(bars, enabled=True)
        bar = self.bar(session, F(2**4096))
        self.assertIsNotNone(replay.step_v3(session, bar, (), F(0), enabled=True))
        self.assertIsNone(adapter.transition_v4(session, bar, (), 1, enabled=True))
        self.assertEqual(session.state.tick, 0)


if __name__ == "__main__":
    unittest.main()

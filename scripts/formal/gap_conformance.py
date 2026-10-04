"""Bounded Replay/refutation conformance, not a universal refinement proof."""
from fractions import Fraction
import itertools
from math import isclose
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))
from sequential_env import Replay, Scale, Execution, _admit_training_transition
from gap_risk import exact_two_bar


def run_trace(target, ratio, multiplier=1, funding=0):
    prices = np.full(65, 100.)
    prices[32:] = 100 * float(ratio)
    settlements = np.zeros(65)
    settlements[32] = float(funding)
    scale = Scale.fit([prices[:31]])
    replay = Replay(prices, settlements, 30, 33, 1, scale,
                    Execution(cost_multiplier=float(multiplier)), enabled=True)
    initial = replay.observation().copy()
    replay.step(float(target))
    transition = replay.step(0.)
    return replay, initial, transition


def check_replay(document):
    if np.__version__ != '2.3.5':
        raise RuntimeError('replay conformance requires NumPy 2.3.5')
    witnesses = {}
    for entry in document['entries']:
        replay, _, transition = run_trace(Fraction(entry['target']), Fraction(entry['priceRatio']))
        expected = Fraction(entry['endingEquity'])
        if not (replay.done and replay.failure == entry['expectedFailure'] and
                replay.units == 0 and replay.pending is None and len(replay.rows) == 2 and
                isclose(replay.equity, float(expected), rel_tol=0, abs_tol=2e-15)):
            raise RuntimeError('Replay counterexample mismatch: ' + entry['id'])
        # Accounted risk losses remain valid training samples, never successes.
        _admit_training_transition(replay, 31, *transition)
        witnesses[entry['id']] = {'failure': replay.failure, 'flat': True, 'rows': 2}

    count = 0
    baseline_observation = None
    for target, ratio, multiplier, funding in itertools.product(
            (Fraction(-1, 4), Fraction(0), Fraction(1, 4)),
            (Fraction(1, 8), Fraction(1, 2), Fraction(1), Fraction(2), Fraction(4)),
            (Fraction(1), Fraction(3, 2), Fraction(2), Fraction(5, 2)),
            (Fraction(-1, 2), Fraction(0), Fraction(1, 2))):
        replay, observation, _ = run_trace(target, ratio, multiplier, funding)
        expected = exact_two_bar(target, ratio, multiplier, funding)
        if not (replay.done and replay.units == 0 and replay.pending is None and
                isclose(replay.equity, float(expected), rel_tol=0, abs_tol=2e-15)):
            raise RuntimeError(f'exact-accounting mismatch: {target, ratio, multiplier, funding}')
        if baseline_observation is None:
            baseline_observation = observation
        if not np.array_equal(observation, baseline_observation):
            raise RuntimeError('future market/funding data altered the current observation')
        cash = sum(row[k] for row in replay.rows for k in ('fee', 'spread', 'slippage', 'impact'))
        ledger_equity = 1 + sum(row['gross'] + row['funding'] for row in replay.rows) - cash
        if not isclose(replay.equity, ledger_equity, rel_tol=0, abs_tol=2e-15):
            raise RuntimeError('per-trace ledger mismatch')
        count += 1
    for invalid_price in (np.nan, np.inf):
        prices = np.full(65, 100.)
        replay = Replay(prices, np.zeros(65), 30, 33, 1,
                        Scale.fit([prices[:31]]), enabled=True)
        replay.step(.25)
        prices[32] = invalid_price
        successor, reward, done = replay.step(0.)
        if not (done and replay.failure == 'invalid_market_transition' and replay.units != 0):
            raise RuntimeError('unavailable market value manufactured liquidation')
        try:
            _admit_training_transition(replay, 31, successor, reward, done)
        except ValueError:
            continue
        raise RuntimeError('unaccounted failure admitted to learning')
    return {'counterexamples': witnesses, 'boundedAccountingTraces': count,
            'causalObservationComparisons': count - 1, 'absoluteTolerance': '2e-15',
            'unaccountedFailuresRejected': 2,
            'universalRefinement': False}

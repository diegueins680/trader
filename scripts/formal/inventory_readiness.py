"""Readiness snapshot evidence; not persistent ownership or continuous freshness."""
from collections import deque
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import tempfile

import z3 as z

ROOT = Path(__file__).resolve().parents[2]
MAIN = 'haskell/app/Main.hs'
REGISTRY = 'formal/research/inventory-readiness-source.json'


def require(ok, message):
    if not ok:
        raise ValueError('inventory readiness: ' + message)


def extract(source=None):
    registry = json.loads((ROOT / REGISTRY).read_text())
    source = (ROOT / MAIN).read_text() if source is None else source
    require(hashlib.sha256(source.encode()).hexdigest() == registry['sha256'], 'source drift')
    require(set(registry['fragments']) == {'predicate', 'scan', 'cycleAdmission', 'postStart', 'httpHealth', 'httpReady'}, 'fragment roster drift')
    for name, fragment in registry['fragments'].items():
        require(source.count(fragment) == 1, 'reviewed fragment drift: ' + name)
    references = [line.strip() for line in source.splitlines()
                  if 'recoveryReadyRef' in line or 'botRecoveryReadyRef' in line]
    require(references == registry['references'], 'readiness reference drift')
    legacy = json.loads((ROOT / 'formal/research/inventory-readiness-legacy.json').read_text())
    require(legacy['adoptionPredicate'] in source, 'adoption behavior changed')
    require(legacy['runtimeType'] in source, 'runtime evidence shape changed')
    return registry


def predicate(amount, side, symbol, info):
    import math
    if not math.isfinite(amount):
        return False
    if amount == 0:
        return True
    if not symbol or side not in (-1, 1) or info is None:
        return False
    running, starting, trading, local = info
    return running and not starting and trading and local == side


def prove():
    a = z.FP('ir_amount', z.Float64())
    side, local = z.Ints('ir_side ir_local')
    present, running, starting, trading, named = z.Bools('ir_present ir_running ir_starting ir_trading ir_named')
    finite = z.And(z.Not(z.fpIsNaN(a)), z.Not(z.fpIsInf(a)))
    zero = z.fpIsZero(a)
    owner = z.And(present, running, z.Not(starting), trading, local == side)
    accepted = z.And(finite, z.Or(zero, z.And(named, z.Or(side == -1, side == 1), owner)))
    claims = [z.Implies(accepted, finite),
              z.Implies(z.And(accepted, z.Not(zero)), z.And(named, owner, z.Or(side == -1, side == 1))),
              z.Implies(z.And(finite, zero), accepted),
              z.Not(z.And(local == -1, local == 1))]
    for claim in claims:
        solver = z.Solver(); solver.set(timeout=10000, random_seed=0)
        require(solver.check() == z.sat, 'empty premise unexpectedly inconsistent')
        solver.add(z.Not(claim))
        require(solver.check() == z.unsat, 'predicate violation or unknown')
    # Non-vacuity: both zero/no-owner and nonzero/matching-owner acceptance exist.
    for condition in (z.And(accepted, zero, z.Not(present)), z.And(accepted, z.Not(zero))):
        solver = z.Solver(); solver.set(timeout=10000, random_seed=0); solver.add(condition)
        require(solver.check() == z.sat, 'acceptance became vacuous')
    return {'F-INVENTORY-READINESS-PREDICATE': 'unsat'}


# (completed cycles, pc, ready, certified snapshot, scan outcome).
# pc: begin, scan, publish, starts, end; outcomes: unset, error, blocked, good.
# good means every inventory row certified AND no pending adoption-start worker.
INITIAL = (0, 'begin', True, True, 'unset')


def successors(state):
    cycle, pc, ready, certified, outcome = state
    if pc == 'begin':
        return [('clear', (cycle, 'scan', False, False, 'unset'))]
    if pc == 'scan':
        return [('scan-' + value, (cycle, 'publish', False, False, value))
                for value in ('error', 'blocked', 'good')] + [('interrupt', (cycle, 'end', False, False, 'error'))]
    if pc == 'publish':
        good = outcome == 'good'
        return [('publish', (cycle, 'starts', good, good, outcome)),
                ('interrupt', (cycle, 'end', False, False, outcome))]
    if pc == 'starts':
        return [('start-ack', (cycle, 'end', ready, certified, outcome))]
    if pc == 'end' and cycle < 1:
        return [('next-cycle', (cycle + 1, 'begin', ready, certified, 'unset'))]
    return []


def model(next_states=successors):
    queue = deque([(INITIAL, 0)]); seen = {INITIAL}; edges = depth = recovered = 0
    while queue:
        state, distance = queue.popleft()
        _, pc, ready, certified, outcome = state
        require(not ready or certified, 'uncertified snapshot published')
        require(pc not in ('scan', 'publish') or not ready, 'scan retained readiness')
        for label, target in next_states(state):
            require(label != 'start-ack' or target[2] == ready, 'start acknowledgment promoted readiness')
            require(label != 'publish' or target[2] == (outcome == 'good'), 'publication ignores scan')
            require(label != 'interrupt' or not target[2], 'interrupted scan ready')
            recovered += label == 'publish' and not ready and target[2]
            edges += 1
            if target not in seen:
                seen.add(target); queue.append((target, distance + 1)); depth = max(depth, distance + 1)
    require(recovered > 0, 'valid recovery unreachable')
    return {'states': len(seen), 'transitions': edges, 'maxShortestDepth': depth,
            'cycles': 2, 'scanOutcomes': 3, 'restoredReadinessTransitions': recovered,
            'scope': 'captured inventory publication; not continuous runtime/exchange consistency'}


def conformance(registry):
    values = [('0', 0.0), ('(-0.0)', -0.0), ('1', 1.0), ('(-1)', -1.0),
              ('1e300', 1e300), ('(-1e300)', -1e300), ('5e-324', 5e-324),
              ('(-5e-324)', -5e-324), ('(0/0)', float('nan')),
              ('(1/0)', float('inf')), ('(-1/0)', -float('inf'))]
    sides = [None, -2, -1, 0, 1, 2]
    infos = [None] + [(*bits, side) for bits in itertools.product((False, True), repeat=3) for side in sides]
    def hs_bool(value):
        return 'True' if value else 'False'
    def hs_side(value):
        return 'Nothing' if value is None else '(Just (' + str(value) + '))'
    def hs_info(value):
        return 'Nothing' if value is None else '(Just (RuntimeAdoptionInfo ' + ' '.join(map(hs_bool, value[:3])) + ' ' + hs_side(value[3]) + '))'
    rows = []; expected = []
    for (raw, amount), side, named, info in itertools.product(values, sides, (False, True), infos):
        rows.append('inventoryRowReconciled ' + raw + ' ' + hs_side(side) + (' "BTCUSDT" ' if named else ' "" ') + hs_info(info))
        expected.append(predicate(amount, side, named, info))
    legacy = json.loads((ROOT / 'formal/research/inventory-readiness-legacy.json').read_text())
    program = 'module Main where\n' + legacy['runtimeType'] + '\n' + registry['fragments']['predicate']
    program += ('\nmain :: IO ()\nmain = mapM_ print [inventoryRowReconciled amount side symbol info'
                + ' | amount <- [' + ','.join(raw for raw, _ in values) + ']'
                + ', side <- [' + ','.join(map(hs_side, sides)) + ']'
                + ', symbol <- [\"\",\"BTCUSDT\"]'
                + ', info <- [' + ','.join(map(hs_info, infos)) + ']]\n')
    with tempfile.TemporaryDirectory(prefix='trader-readiness-') as d:
        path = Path(d); (path / 'Main.hs').write_text(program)
        subprocess.run(['ghc', '-v0', '-O0', '-outputdir', d, str(path/'Main.hs'), '-o', str(path/'test')],
                       check=True, capture_output=True, text=True, timeout=90)
        actual = subprocess.check_output([str(path/'test')], text=True, timeout=30).splitlines()
        repeated = subprocess.check_output([str(path/'test')], text=True, timeout=30).splitlines()
        require(actual == repeated == list(map(hs_bool, expected)), 'compiled predicate mismatch')
    # Legacy planner truth table is retained verbatim above; these witnesses use
    # its exact semantics, not a claim that the old live server was executed.
    flat_owner = (True, False, True, None)
    wrong_owner = (True, False, True, -1)
    old_flat = flat_owner[2] and (flat_owner[3] == 1 or (flat_owner[3] is None and (flat_owner[0] or flat_owner[1])))
    hedge_sides = [-1, 1]
    eligible_orphans = [hedge_sides] if not (-1 in hedge_sides and 1 in hedge_sides) else []
    old_hedge = len(eligible_orphans) == 0
    old_start = wrong_owner[2] and wrong_owner[0]
    require(old_flat and old_hedge and old_start, 'legacy witness lost')
    require(not predicate(1, 1, True, flat_owner), 'flat witness admitted')
    for info in infos:
        require(not (predicate(1, 1, True, info) and predicate(-1, -1, True, info)), 'hedge admitted')
    require(not predicate(1, 1, True, wrong_owner), 'wrong-side witness admitted')
    require(successors(INITIAL)[0][1][2] is False, 'prior readiness retained')
    return {'rows': len(rows), 'repeatRuns': 2, 'accepted': sum(expected),
            'counterexamples': [x['id'] for x in legacy['witnesses']],
            'scope': 'compiled actual pure predicate; source-bound call/scan composition; no live exchange experiment'}


def check_readiness():
    registry = extract()
    return {'smt': prove(), 'model': model(), 'conformance': conformance(registry),
            'referenceCount': len(registry['references']), 'sourceHash': registry['sha256']}

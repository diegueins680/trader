"""Integer cutoff for the pinned ordering model; not full simulator refinement."""
import ast
from collections import deque
from dataclasses import replace
import hashlib
from itertools import product
import json
from pathlib import Path

import z3 as z
import replay_order as order
from target_v2 import certify
from transition_admission import term

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = 'research-notes/registrations/replay-cutoff-audit-engineering.json'
REGISTRATION_SHA256 = '1acde9fbee6c18321669b4435e98240b5fabfb2620249a1c917b53426a423ddc'


def registration():
    raw = (ROOT/REGISTRATION).read_bytes()
    if hashlib.sha256(raw).hexdigest() != REGISTRATION_SHA256:
        raise ValueError('replay cutoff: registration drift')
    return json.loads(raw)


def extract(source, expected_hash):
    if hashlib.sha256(source.encode()).hexdigest() != expected_hash:
        raise ValueError('replay cutoff: ordering source drift')
    module = ast.parse(source)
    functions = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'successors']
    if len(functions) != 1:
        raise ValueError('replay cutoff: successor missing/repeated')
    fn = functions[0]
    reads = [n for n in ast.walk(fn) if isinstance(n, ast.Attribute) and n.attr == 'limit']
    guards = [n.test for n in ast.walk(fn) if isinstance(n, ast.If) and 's.limit' in ast.unparse(n.test)]
    if len(reads) != 1 or len(guards) != 1 or ast.unparse(guards[0]) != 's.failed or s.t == s.limit':
        raise ValueError('replay cutoff: limit dependency changed')
    if ast.unparse(fn.body[0]) != 'go = lambda phase, **kw: replace(s, phase=phase, **kw)':
        raise ValueError('replay cutoff: state copy changed')
    if any(isinstance(n, ast.keyword) and n.arg == 'limit' for n in ast.walk(fn)):
        raise ValueError('replay cutoff: limit mutation')
    return guards[0]


def cutoff_certificate(guard, cutoff=7):
    remaining, horizon, elapsed = z.Ints('cutoff_remaining cutoff_horizon cutoff_elapsed')
    failed = z.Bool('cutoff_failed')
    capped = z.If(remaining <= cutoff, remaining, cutoff)
    end = z.If(remaining <= horizon, remaining, horizon)
    capped_end = z.If(capped <= horizon, capped, horizon)
    original_guard = term(guard, {'s.failed':failed, 's.t':elapsed, 's.limit':remaining})
    capped_guard = term(guard, {'s.failed':failed, 's.t':elapsed, 's.limit':capped})
    premise = z.And(remaining >= 1, z.Or(horizon == 1,horizon == 3,horizon == 6),
                    elapsed >= 0, elapsed <= end)
    conclusion = z.And(end == capped_end, elapsed <= end, end <= capped, capped <= remaining,
                       (elapsed == remaining) == (elapsed == capped), original_guard == capped_guard)
    certify('F-RL-REPLAY-CUTOFF',premise,conclusion,
            [remaining == 7,horizon == 6,elapsed == 6,z.Not(failed)])


def project(state):
    return replace(state,limit=min(state.limit,7))


def reachable_seven(reg):
    cfg = reg['model']
    initial = {order.State('guard',7,h,delay,-1 if pending is None else pending)
               for h in cfg['horizons'] for delay in cfg['extraDelays'] for pending in cfg['initialPendingDue']}
    seen = set(initial); queue = deque(sorted(initial,key=repr))
    while queue:
        state = queue.popleft()
        for _,nxt in order.successors(state):
            if nxt not in seen:
                seen.add(nxt); queue.append(nxt)
    return seen


def check_lifts(reg, transition=order.successors):
    states = reachable_seven(reg); comparisons = 0
    for state in sorted(states,key=repr):
        for remaining in reg['liftRemainingBars']:
            lifted = replace(state,limit=remaining)
            expected = set(transition(project(lifted)))
            actual = {(event,project(nxt)) for event,nxt in transition(lifted)}
            if actual != expected:
                raise RuntimeError('replay cutoff: lifted successor mismatch')
            comparisons += 1
    return {'reachableClassSevenStates':len(states),'successorComparisons':comparisons,
            'remainingSamples':reg['liftRemainingBars'],
            'scope':'exhaustive reachable class-seven states at five registered lifts; not universal runtime refinement'}


def bad_cutoff_witness(reg):
    w = reg['prescribedBadAbstraction']
    original = order.State('terminal_check',w['remaining'],w['horizon'],0,-1,
                           t=w['elapsed'],failed=w['failed'])
    bad = replace(original,limit=min(original.limit,w['cutoff']))
    actual = [event for event,_ in order.successors(original)]
    capped = [event for event,_ in order.successors(bad)]
    if actual != ['continue'] or capped != ['terminal']:
        raise RuntimeError('replay cutoff: prescribed bad-abstraction witness drift')
    return {'kind':'prescribed abstraction mutant, not a current simulator defect',
            'originalEvents':actual,'sixCapEvents':capped,**w}


def match_actual(outcome, horizon, delay, pending=-1):
    initial = order.State('guard',7,horizon,delay,pending)
    ends = order.matching_states(initial,outcome['trace'])
    if not any(s.t == outcome['t'] and s.rows == outcome['rows'] and s.pending == outcome['pending'] and
               (outcome['failure'] is not None or not s.failed) and
               (outcome['failure'] is not None or s.terminal == outcome['done']) for s in ends):
        raise RuntimeError('replay cutoff: actual trace not admitted by class seven')


def check_conformance(reg):
    c = reg['conformance']; count = 0
    for args in product(c['targets'],c['horizons'],c['extraDelays'],c['remainingBars'],
                        c['initialUnits'],c['nextBarReturns']):
        outcome = order.conformance_case(*args)
        match_actual(outcome,args[1],args[2]); count += 1
    scenarios = {}
    for name in c['scenarios']:
        outcome = order.conformance_case(horizon=6,remaining=7,scenario=name)
        match_actual(outcome,6,0,2 if name == 'inherited_pending' else -1)
        scenarios[name] = outcome
    ordinary = scenarios['ordinary']
    if ordinary['done'] or ordinary['t'] != 6 or ordinary['rows'] != 6 or 'liquidate' in ordinary['trace']:
        raise RuntimeError('replay cutoff: nonterminal six-bar regression')
    return {'gridTraces':count,'scenarios':scenarios,
            'scope':'synthetic actual-helper projected traces; observations/rewards are not quotiented'}


def check_replay_cutoff():
    reg = registration()
    guard = extract((ROOT/reg['orderingModel']).read_text(),reg['orderingModelSha256'])
    cutoff_certificate(guard,reg['cutoff'])
    # Recheck the original runtime source binding before invoking its trace helper.
    old = order.registration()
    order.extract((ROOT/order.SOURCE).read_text(),old['sourceFunctionASTSha256'])
    return {'smt':{'F-RL-REPLAY-CUTOFF':'unsat'},'model':order.check_model(reg),
            'lifts':check_lifts(reg),'badAbstraction':bad_cutoff_witness(reg),
            'conformance':check_conformance(reg),
            'scope':'ordering quotient only; integer cutoff lemma unbounded in R; finite graph and sampled runtime traces'}

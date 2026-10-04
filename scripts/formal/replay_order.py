"""Source-bound bounded replay ordering; no numeric or production refinement."""
import ast
from collections import deque
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import z3 as z
from target_v2 import certify
from transition_admission import term

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'scripts/research/sequential_env.py'
REGISTRATION = 'research-notes/registrations/replay-order-audit-engineering.json'
REGISTRATION_SHA256 = '9a240dfe56ce8c85c61a0820ac97b47c0ff48f6410b82f274a2098c25a45f616'


def registration():
    raw = (ROOT/REGISTRATION).read_bytes()
    if hashlib.sha256(raw).hexdigest() != REGISTRATION_SHA256:
        raise ValueError('replay order: registration drift')
    return json.loads(raw)


def extract(source, hashes):
    module = ast.parse(source)
    found = {}
    for key, expected in hashes.items():
        cls, name = key.split('.')
        classes = [n for n in module.body if isinstance(n, ast.ClassDef) and n.name == cls]
        methods = [n for c in classes for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name]
        if len(methods) != 1 or hashlib.sha256(ast.dump(methods[0], include_attributes=False).encode()).hexdigest() != expected:
            raise ValueError('replay order: source AST drift ' + key)
        found[key] = methods[0]
    step = found['Replay.step']
    pending = [n.value.elts[0] for n in ast.walk(step) if isinstance(n, ast.Assign) and
               isinstance(n.value, ast.Tuple) and len(n.targets) == 1 and ast.unparse(n.targets[0]) == 'self.pending']
    guards = [n.test for n in ast.walk(step) if isinstance(n, ast.If) and
              'self.pending[0]' in ast.unparse(n.test)]
    if len(pending) != 1 or len(guards) != 1:
        raise ValueError('replay order: due expression or fill guard missing/repeated')
    return pending[0], guards[0]


def due_certificate(due_node, guard_node):
    created, delay, now = z.Ints('order_created order_delay order_now')
    clear, present = z.Bools('order_risk_clear order_pending_present')
    due = term(due_node, {'self.t':created,'self.execution.extra_delay':delay})
    guard = term(guard_node, {'self.failure is None':clear, 'self.pending is not None':present,
                             'self.pending[0]':due, 'self.t':now})
    certify('F-RL-REPLAY-DUE', z.And(z.Or(delay == 0,delay == 1),guard),
            z.And(due == created+1+delay, due > created, now >= due, now > created, clear, present),
            [created == 24, delay == 1, now == 26, clear, present])


@dataclass(frozen=True)
class State:
    phase: str
    limit: int
    end: int
    delay: int
    pending: int
    t: int = 0
    rows: int = 0
    fills: int = 0
    gate: bool = False
    failed: bool = False
    solvent: bool = True
    terminal: bool = False
    observed: bool = False
    bar_valid: bool = False
    risk_checked: bool = False
    terminal_handled: bool = False


def successors(s):
    go = lambda phase, **kw: replace(s,phase=phase,**kw)
    if s.phase == 'guard':
        return [('reject_gate',go('return')), ('pass_gate',go('observation',gate=True))]
    if s.phase == 'observation':
        return [('reject_observation',go('return')), ('observe',go('schedule',observed=True))]
    if s.phase == 'schedule':
        return [('schedule',go('bar',pending=1+s.delay if s.pending < 0 else s.pending))]
    if s.phase == 'bar':
        return [('reject_market_or_feature',go('return')), ('validate_bar',go('mark',bar_valid=True))]
    if s.phase == 'mark':
        return [('mark_old_and_advance',go('risk',t=s.t+1,bar_valid=False,risk_checked=False))]
    if s.phase in ('risk','postfill_risk'):
        next_phase = 'fill_check' if s.phase == 'risk' else 'terminal_check'
        return [('risk_clear',go(next_phase,failed=False,solvent=True,risk_checked=True)),
                ('risk_solvent_failure',go('terminal_check',failed=True,solvent=True,risk_checked=True)),
                ('risk_insolvent_failure',go('terminal_check',failed=True,solvent=False,risk_checked=True))]
    if s.phase == 'fill_check':
        if s.pending >= 0 and s.pending <= s.t:
            return [('target_fill',go('postfill_risk',pending=-1,fills=s.fills+1,risk_checked=False))]
        return [('not_due',go('terminal_check'))]
    if s.phase == 'terminal_check':
        if s.failed or s.t == s.limit:
            return [('terminal',go('cancel',terminal=True))]
        return [('continue',go('row'))]
    if s.phase == 'cancel':
        return [('cancel_pending',go('liquidate_check',pending=-1))]
    if s.phase == 'liquidate_check':
        event = 'liquidate' if s.solvent else 'skip_insolvent_liquidation'
        return [(event,go('row',terminal_handled=True))]
    if s.phase == 'row':
        return [('append_row',go('after_row',rows=s.rows+1))]
    if s.phase == 'after_row':
        return [('finish',go('return'))] if s.terminal or s.t == s.end else [('next_bar',go('bar'))]
    if s.phase == 'return':
        return [('stutter',s)]
    raise ValueError('replay order: unknown model phase')


def check_edge(s, event, nxt):
    if not (0 <= nxt.rows <= nxt.t <= nxt.rows+1 and nxt.t <= nxt.end <= nxt.limit):
        raise RuntimeError('replay order: time/row bound')
    if not 0 <= nxt.fills <= 1:
        raise RuntimeError('replay order: repeated target fill')
    if event in ('mark_old_and_advance','target_fill','liquidate','append_row') and not s.gate:
        raise RuntimeError('replay order: effect before gate')
    if event == 'mark_old_and_advance' and not (s.phase == 'mark' and s.observed and s.bar_valid and s.t == s.rows and nxt.t == s.t+1):
        raise RuntimeError('replay order: invalid mark order')
    if event == 'target_fill' and not (s.phase == 'fill_check' and s.risk_checked and not s.failed and
                                       0 <= s.pending <= s.t and s.t == s.rows+1):
        raise RuntimeError('replay order: early or unchecked target fill')
    if event == 'liquidate' and not (s.phase == 'liquidate_check' and s.terminal and s.solvent and s.pending == -1):
        raise RuntimeError('replay order: invalid terminal liquidation')
    if event == 'append_row' and not (s.phase == 'row' and s.risk_checked and s.t == s.rows+1 and
                                      nxt.rows == nxt.t and (not s.terminal or (s.pending == -1 and s.terminal_handled))):
        raise RuntimeError('replay order: incomplete or repeated row')
    if s.terminal and event == 'next_bar':
        raise RuntimeError('replay order: bar after terminal')
    if s.phase == 'return' and nxt != s:
        raise RuntimeError('replay order: effect after return')


def check_model(reg, transition=successors):
    cfg = reg['model']
    initial = {State('guard',limit,min(limit,h),delay,-1 if pending is None else pending)
               for limit in cfg['remainingBars'] for h in cfg['horizons']
               for delay in cfg['extraDelays'] for pending in cfg['initialPendingDue']}
    queue = deque(sorted(initial,key=repr)); depths = {s:0 for s in initial}; graph = {}; edges = 0
    while queue:
        state = queue.popleft()
        outgoing = transition(state)
        if not outgoing:
            raise RuntimeError('replay order: deadlock')
        graph[state] = []
        for event,nxt in outgoing:
            check_edge(state,event,nxt); edges += 1
            if nxt != state:
                graph[state].append(nxt)
            elif state.phase != 'return':
                raise RuntimeError('replay order: nonterminal stutter')
            if nxt not in depths:
                depths[nxt] = depths[state]+1; queue.append(nxt)
    # Complete graph is acyclic except final stutters. Longest path gives a
    # strict finite progress rank; helper termination is an external assumption.
    ranks = {}; active = set()
    def rank(s):
        if s in active: raise RuntimeError('replay order: nonterminal cycle')
        if s in ranks: return ranks[s]
        active.add(s)
        ranks[s] = 0 if not graph[s] else 1+max(rank(n) for n in graph[s])
        active.remove(s)
        return ranks[s]
    bound = max(rank(s) for s in initial)
    return {'states':len(depths),'transitions':edges,'distinctInitialStates':len(initial),
            'maxShortestDepth':max(depths.values()),'progressBound':bound,
            'search':'complete reachable fixed point','terminalStuttering':True,
            'scope':'one abstract replay call; final reward check or explicit early return',
            'runtimeRefinement':False,'productionLifecycleProof':False}


def check_replay_order():
    reg = registration()
    nodes = extract((ROOT/SOURCE).read_text(),reg['sourceFunctionASTSha256'])
    due_certificate(*nodes)
    return {'smt':{'F-RL-REPLAY-DUE':'unsat'},'model':check_model(reg),
            'conformance':check_conformance(reg)}


VISIBLE = {'mark_old_and_advance','target_fill','liquidate','append_row'}


def matching_states(initial, events):
    """Projected finite-trace conformance; hidden model steps are epsilon edges."""
    def closure(states):
        seen = set(states); queue = deque(states)
        while queue:
            s = queue.popleft()
            for event,nxt in successors(s):
                if event not in VISIBLE and nxt not in seen:
                    seen.add(nxt); queue.append(nxt)
        return seen
    states = closure([initial])
    for observed in events:
        states = closure({nxt for s in states for event,nxt in successors(s) if event == observed})
    return {s for s in states if s.phase == 'return'}


def conformance_case(target=.25, horizon=1, delay=0, remaining=2, units=0., change=0., scenario='ordinary'):
    import sys
    import numpy as np
    sys.path.insert(0,str(ROOT/'scripts/research'))
    from sequential_env import Replay, Scale, Execution
    start = 24
    prices = np.full(40,100.); funding = np.zeros(40)
    for i in range(1,remaining+1):
        prices[start+i] = 100.*(1+change)**i
        funding[start+i] = .02
    scale = Scale.fit([prices[:start+1]])
    config = Execution(extra_delay=delay, fill_fraction=.5 if scenario == 'partial_fill' else 1.,
                       miss_every=1 if scenario == 'missed_fill' else 0)
    trace = []; marked = set(); marks = []; snapshot = [1.,units]
    class Traced(Replay):
        def _risk(self):
            if self.t not in marked:
                marked.add(self.t); trace.append('mark_old_and_advance')
                marks.append((self.t,float(self.equity),float(self.units),*snapshot))
            return super()._risk()
        def _trade(self, target, terminal=False):
            trace.append('liquidate' if terminal else 'target_fill')
            return super()._trade(target,terminal)
    env = Traced(prices,funding,start,start+remaining+1,horizon,scale,config,enabled=True)
    env.units = units
    inherited = scenario == 'inherited_pending'
    if inherited: env.pending = (start+2,-.25)
    if scenario == 'invalid_market': prices[start+1] = np.nan
    if scenario in ('solvent_risk','insolvency'):
        # Inventory is already present. The next funding debit alone triggers the stop.
        env.units = .0025; snapshot[1] = env.units
        funding[start+1] = 100. if scenario == 'solvent_risk' else 800.
    class Rows(list):
        def append(self,row):
            trace.append('append_row'); super().append(row)
            snapshot[:] = [float(env.equity),float(env.units)]
    env.rows = Rows()
    env.step(target,valid=scenario != 'invalid_gate')
    for tick,equity,old_units,old_equity,expected_units in marks:
        expected = old_equity + expected_units*(prices[tick]-prices[tick-1]) - expected_units*funding[tick]
        if old_units != expected_units or not np.isclose(equity,expected,atol=1e-13,rtol=1e-13):
            raise RuntimeError('replay order: old-inventory mark conformance')
    initial = State('guard',remaining,min(remaining,horizon),delay,2 if inherited else -1)
    ends = matching_states(initial,trace)
    pending = -1 if env.pending is None else env.pending[0]-start
    if not any(s.t == env.t-start and s.rows == len(env.rows) and s.pending == pending and
               (env.failure is not None or s.terminal == env.done) for s in ends):
        raise RuntimeError('replay order: actual trace not admitted by model')
    return {'trace':trace,'t':env.t-start,'rows':len(env.rows),'pending':pending,
            'failure':env.failure,'done':env.done,'units':float(env.units),
            'equity':float(env.equity),'costs':sum(sum(row[k] for k in ('fee','spread','slippage','impact')) for row in env.rows)}


def check_conformance(reg):
    from itertools import product
    c = reg['conformance']; count = 0
    for args in product(c['targets'],c['horizons'],c['extraDelays'],c['remainingBars'],
                        c['initialUnits'],c['nextBarReturns']):
        conformance_case(*args); count += 1
    outcomes = {s:conformance_case(scenario=s) for s in c['scenarios']}
    terminal = conformance_case(remaining=1)
    if terminal['trace'] != ['mark_old_and_advance','target_fill','liquidate','append_row'] or not terminal['costs'] > 0 or terminal['units'] != 0.:
        raise RuntimeError('replay order: terminal round-trip regression')
    return {'gridTraces':count,'scenarios':outcomes,'terminalRoundTrip':terminal,
            'scope':'instrumented actual helpers/row publication; no universal refinement'}

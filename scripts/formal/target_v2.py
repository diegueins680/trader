"""Source-linked, scoped v2 target arithmetic and batch-publication checks."""
import ast
import copy
from pathlib import Path
import z3 as z
from causal_footprint import structure
from terminal_numerics import finite, prove, scalar

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'scripts/research/gae_targets_v2.py'
TEMPLATE = r'''
"""Disabled, isolated raw GAE targets. No normalization or training integration."""
from __future__ import annotations

from math import isfinite

VERSION = "gae-targets-v2"
MAX_ROWS = 256


def step_v2(reward: object, value: object, next_value: object,
            later: object, done: object, gamma: object, *,
            enabled: object = False, version: object = VERSION) -> tuple[float, float] | None:
    if enabled is not True or type(version) is not str or version != VERSION:
        return None
    if type(done) is not bool or not all(type(x) is float and isfinite(x)
                                       for x in (reward, value, next_value, later, gamma)):
        return None
    if not 0.0 <= gamma <= 1.0:
        return None
    if done:
        raw = reward - value
        target = reward
    else:
        bootstrap = gamma * next_value
        delta = (reward + bootstrap) - value
        trace = (gamma * 0.95) * later
        raw = delta + trace
        target = raw + value
    if not all(isfinite(x) for x in (raw, target)):
        return None
    return raw, target


def batch_v2(rows: object, gamma: object, *, enabled: object = False,
             version: object = VERSION) -> tuple[tuple[float, float], ...] | None:
    if enabled is not True or type(version) is not str or version != VERSION:
        return None
    if type(rows) is not tuple or not 1 <= len(rows) <= MAX_ROWS:
        return None
    staged = []
    later = 0.0
    for row in reversed(rows):
        if type(row) is not tuple or len(row) != 4:
            return None
        reward, value, next_value, done = row
        pair = step_v2(reward, value, next_value, later, done, gamma,
                       enabled=True, version=version)
        if pair is None:
            return None
        staged.append(pair)
        later = pair[0]
    return tuple(reversed(staged))
'''


def extract(source):
    module = ast.parse(source)
    expected = ast.parse(TEMPLATE)
    expressions = {}
    for tree, save in ((module, True), (expected, False)):
        try:
            step = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'step_v2')
            branch = step.body[3]
            assignments = [('terminal_' + n.targets[0].id, n) for n in branch.body]
            assignments += [('continuing_' + n.targets[0].id, n) for n in branch.orelse]
            for name, assignment in assignments:
                if save:
                    expressions[name] = copy.deepcopy(assignment.value)
                assignment.value = ast.Name(id=name.upper(), ctx=ast.Load())
        except (AttributeError, IndexError, StopIteration):
            raise ValueError('target v2: source control-flow drift')
    if structure(module) != structure(expected):
        raise ValueError('target v2: source skeleton drift')
    return expressions


def outputs(expressions, values, floating):
    values = dict(values)
    terminal_raw = scalar(expressions['terminal_raw'], values, floating)
    values['raw'] = terminal_raw
    terminal_target = scalar(expressions['terminal_target'], values, floating)
    for name in ('bootstrap', 'delta', 'trace', 'raw', 'target'):
        values[name] = scalar(expressions['continuing_' + name], values, floating)
    return (terminal_raw, terminal_target), (values['raw'], values['target'])


def check_publication():
    # Bound read from the source-locked template. Primitives/loop refinement are
    # assumptions: this is a separate control-flow abstraction, not Python proof.
    maximum = next(n.value.value for n in ast.parse(TEMPLATE).body
                   if isinstance(n, ast.Assign) and n.targets[0].id == 'MAX_ROWS')
    initial = ('entry', 0, 0)
    absent = ('absent', 0, 0)
    seen, pending = {initial: 0}, [initial]
    transitions = 0
    while pending:
        state = pending.pop(0)
        phase, size, count = state
        if phase == 'entry':
            successors = [absent] + [('staging', n, 0) for n in range(1, maximum+1)]
        elif phase == 'staging':
            if count < size:
                successors = [absent, ('staging', size, count+1)]
            else:
                successors = [('published', size, count)]
        else:
            successors = [state]  # Terminal stuttering; no further publication.
        def rank(item):
            phase, size, count = item
            return maximum+2 if phase == 'entry' else size-count+1 if phase == 'staging' else 0
        for successor in successors:
            transitions += 1
            if phase in ('entry','staging') and not rank(successor) < rank(state):
                raise RuntimeError('target v2: nondecreasing progress rank')
            kind, n, k = successor
            if kind == 'published' and not (phase == 'staging' and count == size == n == k or successor == state):
                raise RuntimeError('target v2: partial publication')
            if kind == 'staging' and not (0 <= k <= n <= maximum):
                raise RuntimeError('target v2: invalid staging state')
            if successor not in seen:
                seen[successor] = seen[state]+1
                pending.append(successor)
    return {'requirement':'F-RL-TARGET-V2-PUBLISH', 'result':'pass',
            'maximumRows':maximum, 'states':len(seen), 'transitions':transitions,
            'maxShortestDepth':max(seen.values()), 'terminalStuttering':True,
            'progressBound':maximum+2,'strictProgressRankChecked':True, 'liveAuthorizationTransitions':0,
            'sourceSkeletonChecked':True, 'universalRuntimeRefinement':False}



def certify(name, premise, conclusion, witness):
    solver = z.Solver()
    solver.set(timeout=10000, random_seed=0)
    solver.add(premise, *witness)
    result = solver.check()
    if result != z.sat:
        detail = solver.reason_unknown() if result == z.unknown else str(result)
        raise RuntimeError(name + ': premise witness failed: ' + detail)
    # The fresh universal query never receives the non-vacuity witness.
    solver = z.Solver()
    solver.set(timeout=10000, random_seed=0)
    solver.add(premise, z.Not(conclusion))
    result = solver.check()
    if result != z.unsat:
        detail = str(solver.model()) if result == z.sat else solver.reason_unknown()
        raise RuntimeError(name + ': violating/unknown obligation: ' + detail)


def check_targets(source=None):
    expressions = extract((ROOT/SOURCE).read_text() if source is None else source)
    reward,value,nxt,later,gamma = z.Reals('v2_r v2_v v2_n v2_l v2_g')
    atoms = dict(zip(('reward','value','next_value','later','gamma'),(reward,value,nxt,later,gamma)))
    terminal, continuing = outputs(expressions,atoms,False)
    reference = reward+gamma*nxt-value+gamma*z.RealVal('0.95')*later
    prove('F-RL-TARGET-V2-REAL',z.And(gamma>=0,gamma<=1),
          z.And(terminal[0]==reward-value,terminal[1]==reward,
                continuing[0]==reference,continuing[1]==reference+value))
    r,v,n,c,g = z.FPs('v2_rf v2_vf v2_nf v2_cf v2_gf',z.Float64())
    atoms = dict(zip(atoms,(r,v,n,c,g)))
    terminal, continuing = outputs(expressions,atoms,True)
    done,enabled,version_ok = z.Bools('v2_done v2_enabled v2_version_ok')
    raw = z.If(done,terminal[0],continuing[0])
    target = z.If(done,terminal[1],continuing[1])
    zero,one = z.FPVal(0,z.Float64()),z.FPVal(1,z.Float64())
    # The audited complete source skeleton binds these native-type, range and
    # publication guards. Native input types and trusted primitive semantics are
    # explicitly assumed; this is not a general Python interpreter.
    admitted = z.And(enabled,version_ok,*[finite(x) for x in (r,v,n,c,g)],
                     z.fpGEQ(g,zero),z.fpLEQ(g,one))
    published = z.And(admitted,finite(raw),finite(target))
    witness = [enabled,version_ok,done,*[x == zero for x in (r,v,n,c,g)]]
    certify('F-RL-TARGET-V2-TERMINAL',z.And(published,done),
            z.fpToIEEEBV(target)==z.fpToIEEEBV(r),witness)
    certify('F-RL-TARGET-V2-FINITE',published,
            z.And(finite(raw),finite(target),enabled,version_ok),witness)
    prove('F-RL-TARGET-V2-FINITE-disabled',z.Not(enabled),z.Not(published))
    return {'smt':{name:'unsat' for name in ('F-RL-TARGET-V2-REAL','F-RL-TARGET-V2-TERMINAL','F-RL-TARGET-V2-FINITE')},
            'queries':4,'premiseChecks':4,'premiseWitnessRemovedBeforeViolation':True,'rounding':'RNE binary64; separate operations',
            'model':check_publication(),'normalizationVerified':False,
            'wholeLearnerCorrection':False,'universalRuntimeRefinement':False}

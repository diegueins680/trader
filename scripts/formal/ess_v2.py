"""Source-bound exact ESS recurrence and bounded publication abstraction."""
import ast
import copy
from collections import deque
from pathlib import Path
import z3 as z
from causal_footprint import structure
from target_v2 import certify

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'scripts/research/ess_rational_v2.py'
REGISTRATION = 'research-notes/registrations/ess-v2-engineering.json'
TEMPLATE = r'''
"""Disabled, isolated exact ESS diagnostic; no estimator or trading integration."""
from __future__ import annotations

from fractions import Fraction
from math import isfinite

VERSION = "ess-rational-v2"
MAX_ROWS = 256


def effective_sample_size_v2(weights: object, *, enabled: object = False,
                             version: object = VERSION) -> Fraction | None:
    if enabled is not True or type(version) is not str or version != VERSION:
        return None
    if type(weights) is not tuple or not 1 <= len(weights) <= MAX_ROWS:
        return None
    if not all(type(value) is float and isfinite(value) and value >= 0 for value in weights):
        return None
    total = Fraction(0)
    squares = Fraction(0)
    for value in weights:
        weight = Fraction.from_float(value)
        total = total + weight
        squares = squares + weight * weight
    return Fraction(0) if squares == 0 else total * total / squares
'''


def extract(source):
    expressions = {}
    for text, save in ((source, True), (TEMPLATE, False)):
        module = ast.parse(text)
        try:
            fn = next(n for n in module.body if isinstance(n, ast.FunctionDef))
            loop = fn.body[5]
            slots = [('total', loop.body[1]), ('squares', loop.body[2]), ('result', fn.body[6])]
            for name, node in slots:
                if save:
                    expressions[name] = copy.deepcopy(node.value)
                node.value = ast.Name(id=name.upper(), ctx=ast.Load())
        except (StopIteration, AttributeError, IndexError):
            raise ValueError('ESS v2: control-flow drift')
        if save:
            actual = module
        elif structure(actual) != structure(module):
            raise ValueError('ESS v2: module skeleton drift')
    return expressions


def scalar(node, atoms):
    if isinstance(node, ast.Name) and node.id in atoms:
        return atoms[node.id]
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return z.RealVal(node.value)
    if (isinstance(node, ast.Call) and ast.unparse(node.func) == 'Fraction' and
            len(node.args) == 1 and not node.keywords):
        return scalar(node.args[0], atoms)
    if isinstance(node, ast.BinOp):
        a, b = scalar(node.left, atoms), scalar(node.right, atoms)
        if isinstance(node.op, ast.Add): return a+b
        if isinstance(node.op, ast.Mult): return a*b
        if isinstance(node.op, ast.Div): return a/b
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
        return scalar(node.left, atoms) == scalar(node.comparators[0], atoms)
    if isinstance(node, ast.IfExp):
        return z.If(scalar(node.test, atoms), scalar(node.body, atoms), scalar(node.orelse, atoms))
    raise ValueError('ESS v2: unsupported arithmetic ' + ast.unparse(node))


def invariant(count, total, squares):
    return z.And(count >= 0, total >= 0, squares >= 0, squares <= total*total,
                 total*total <= count*squares)


def check_arithmetic(expressions):
    k, s, q, w = z.Reals('ess_v2_k ess_v2_s ess_v2_q ess_v2_w')
    atoms = {'total': s, 'squares': q, 'weight': w}
    next_s = scalar(expressions['total'], atoms)
    # Source assignments are sequential: a changed squares expression sees new total.
    next_q = scalar(expressions['squares'], dict(atoms, total=next_s))
    certify('F-RL-ESS-V2-ACCUMULATE-base', z.BoolVal(True), invariant(0, z.RealVal(0), z.RealVal(0)), [])
    # k is real here, a stronger domain than the nonnegative integer loop index.
    certify('F-RL-ESS-V2-ACCUMULATE-step', z.And(invariant(k, s, q), w >= 0),
            z.And(next_s == s+w, next_q == q+w*w, invariant(k+1, next_s, next_q)),
            [k == 1, s == 1, q == 1, w == 1])
    result = scalar(expressions['result'], atoms)
    certify('F-RL-ESS-V2-BOUNDS', z.And(invariant(k, s, q), k >= 1, k <= 256),
            z.And(result == z.If(q == 0, z.RealVal(0), s*s/q),
                  z.Implies(q == 0, z.And(s == 0, result == 0)),
                  z.Implies(q > 0, z.And(result >= 1, result <= k))),
            [k == 2, s == 2, q == 2])
    return {'F-RL-ESS-V2-ACCUMULATE': 'unsat', 'F-RL-ESS-V2-BOUNDS': 'unsat'}


def transitions(state, maximum):
    phase, size, count = state
    if phase == 'entry':
        yield ('absent', 0, 0)
        yield ('shape', 0, 0)
    elif phase == 'shape':
        yield ('absent', 0, 0)
        for n in range(1, maximum+1): yield ('validate', n, 0)
    elif phase == 'validate':
        if count < size:
            yield ('absent', 0, 0)
            yield ('validate', size, count+1)
        else:
            yield ('accumulate', size, 0)
    elif phase == 'accumulate':
        if count < size:
            yield ('accumulate', size, count+1)
        else:
            yield ('zero', 0, 0)
            yield ('positive', 0, 0)
    else:
        yield state


def rank(state, maximum):
    phase, size, count = state
    if phase == 'entry': return 2*maximum+5
    if phase == 'shape': return 2*maximum+4
    if phase == 'validate': return 2*size-count+2
    if phase == 'accumulate': return size-count+1
    return 0


def check_publication(maximum):
    initial = ('entry', 0, 0)
    queue = deque([initial]); depth = {initial: 0}; edges = 0
    while queue:
        state = queue.popleft()
        phase, size, count = state
        if phase in ('validate', 'accumulate') and not (1 <= size <= maximum and 0 <= count <= size):
            raise RuntimeError('ESS v2: invalid loop bounds')
        for nxt in transitions(state, maximum):
            edges += 1
            if rank(state, maximum) == 0:
                if nxt != state: raise RuntimeError('ESS v2: terminal escaped')
            elif rank(nxt, maximum) >= rank(state, maximum):
                raise RuntimeError('ESS v2: nondecreasing progress')
            if nxt[0] == 'accumulate' and phase != 'accumulate' and (phase != 'validate' or count != size):
                raise RuntimeError('ESS v2: admission bypass')
            if nxt[0] in ('zero', 'positive') and phase not in ('zero', 'positive') and (phase != 'accumulate' or count != size):
                raise RuntimeError('ESS v2: partial publication')
            if nxt not in depth:
                depth[nxt] = depth[state]+1; queue.append(nxt)
    if not {('zero', 0, 0), ('positive', 0, 0), ('absent', 0, 0)} <= depth.keys():
        raise RuntimeError('ESS v2: vacuous terminal domain')
    return {'states': len(depth), 'transitions': edges, 'maxShortestDepth': max(depth.values()),
            'maximumRows': maximum, 'rankBound': rank(initial, maximum), 'search': 'reachable_fixed_point',
            'terminalStuttering': True, 'orderAuthorizationTransitions': 0,
            'persistentState': False, 'runtimeRefinement': False}


def check_isolation():
    consumers = []
    for path in sorted((ROOT/'scripts/research').glob('*.py')):
        if str(path.relative_to(ROOT)) == SOURCE: continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or '', *[alias.name for alias in node.names]]
            else: continue
            if any('ess_rational_v2' in n.split('.') for n in names):
                consumers.append(str(path.relative_to(ROOT)))
    if consumers:
        raise ValueError('ESS v2: unexpected research imports: ' + ', '.join(consumers))
    return {'existingResearchImports': 0, 'dynamicImportExclusionProved': False}


def check_ess(registration, source=None):
    if (registration['version'] != 'ess-rational-v2' or registration['maxRows'] != 256 or
            registration['financialTrialBudget'] != 0 or registration['marketDataAccess'] is not False or
            registration['holdoutAccess'] is not False or registration['originalOpeReruns'] != 0):
        raise ValueError('ESS v2: registration drift')
    expressions = extract((ROOT/SOURCE).read_text() if source is None else source)
    return {'smt': check_arithmetic(expressions), 'queries': 3, 'premiseChecks': 3,
            'model': check_publication(registration['maxRows']), 'isolation': check_isolation(),
            'exactPrimitiveSemanticsAssumed': True, 'statisticalReliabilityVerified': False}

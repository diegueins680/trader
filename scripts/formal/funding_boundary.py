"""Conditional grid/endpoint certificates and preserved funding overflow witnesses."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import z3 as z
from target_v2 import certify

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'scripts/research/run_sequential_screen.py'
REGISTRATION = 'research-notes/registrations/funding-boundary-audit-engineering.json'
REGISTRATION_SHA256 = 'cc16f1ff23ba0012bc3fb1d3f09edcf3a14742d49ebb64223c9c83858e7fc398'


def registration():
    raw = (ROOT / REGISTRATION).read_bytes()
    if hashlib.sha256(raw).hexdigest() != REGISTRATION_SHA256:
        raise ValueError('funding audit: registration drift')
    return json.loads(raw)


def extract(source, expected):
    functions = [n for n in ast.parse(source).body
                 if isinstance(n, ast.FunctionDef) and n.name == 'load_development']
    if len(functions) != 1:
        raise ValueError('funding audit: loader missing or repeated')
    fn = functions[0]
    digest = hashlib.sha256(ast.dump(fn, include_attributes=False).encode()).hexdigest()
    if digest != expected:
        raise ValueError('funding audit: loader AST drift')
    expressions = {n.targets[0].id: n.value for n in ast.walk(fn)
                   if isinstance(n, ast.Assign) and len(n.targets) == 1 and
                   isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('times', 'closes')}
    loop = next(n for n in ast.walk(fn) if isinstance(n, ast.For) and
                isinstance(n.target, ast.Name) and n.target.id == 'event')
    return expressions, loop


def integer_expression(node, atoms):
    if isinstance(node, ast.Name) and node.id in atoms:
        return atoms[node.id]
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return z.IntVal(node.value)
    if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and
            node.value.id == 'spec' and isinstance(node.slice, ast.Constant)):
        return z.IntVal(atoms['spec'][node.slice.value])
    if isinstance(node, ast.BinOp):
        left, right = integer_expression(node.left, atoms), integer_expression(node.right, atoms)
        if isinstance(node.op, ast.Add): return left + right
        if isinstance(node.op, ast.Sub): return left - right
    raise ValueError('funding audit: unsupported integer expression')


def grid_certificate(expressions, spec):
    times = expressions['times']
    if (not isinstance(times, ast.Call) or ast.unparse(times.func) != 'np.arange' or
            len(times.args) != 3 or [(k.arg, ast.unparse(k.value)) for k in times.keywords] != [('dtype', 'np.int64')]):
        raise ValueError('funding audit: range structure drift')
    atoms = {'spec': spec}
    start, stop, step = [integer_expression(x, atoms) for x in times.args]
    i, n = z.Int('funding_grid_i'), spec['rowsPerSymbol']
    opened = start + i * step
    closed = integer_expression(expressions['closes'], dict(atoms, times=opened))
    certify('F-RL-FUNDING-GRID', z.And(i >= 0, i < n),
            z.And(start == spec['startOpenTime'], step > 0,
                  stop == start + (n-1)*step + 1, opened < stop,
                  closed == opened + step - 1, closed >= opened,
                  opened >= 0, closed < 2**53, closed < 2**63,
                  start + (i+1)*step > closed), [i == 0])
    return {'rows': n, 'firstOpen': spec['startOpenTime'], 'lastOpen': spec['endOpenTime'],
            'firstClose': spec['startOpenTime']+spec['intervalMilliseconds']-1,
            'lastClose': spec['endOpenTime']+spec['intervalMilliseconds']-1}


def endpoint_certificate():
    # Trusted searchsorted(left) relation on an exact regular grid. Include
    # both exterior buckets; this is not a proof of the binary-search program.
    first, step, count, event, j, k = z.Ints('fb_first fb_step fb_count fb_event fb_j fb_k')
    def member(index):
        return z.Or(z.And(index == 0, event <= first),
                    z.And(index > 0, index < count,
                          first+(index-1)*step < event, event <= first+index*step),
                    z.And(index == count, event > first+(count-1)*step))
    premise = z.And(step > 0, count >= 1, member(j), member(k))
    claim = z.And(j == k, j >= 0, j <= count,
                  z.Implies(j < count, event <= first+j*step),
                  z.Implies(j == count, event > first+(count-1)*step))
    certify('F-RL-FUNDING-ENDPOINT', premise, claim,
            [first == 9, step == 10, count == 3, event == 19, j == 1, k == 1])


def compiled_loop(loop):
    # Execute the actual source node with ordinary synthetic event records.
    # This bypasses pandas/hash admission deliberately; full-loader tests are separate.
    fn = ast.parse('def apply(closes, es, f):\n    pass\n    return f').body[0]
    fn.body[0] = copy.deepcopy(loop)
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    scope = {'np': np}
    exec(compile(module, SOURCE + ':settlement-loop', 'exec'), scope)
    return scope['apply']


def apply_events(apply, closes, events):
    rows = [SimpleNamespace(fundingTime=t, fundingRate=r, resolvedMarkPrice=m) for t,r,m in events]
    return apply(np.asarray(closes, dtype=np.int64), SimpleNamespace(itertuples=lambda: iter(rows)),
                 np.zeros(len(closes), dtype=float))


def overflow_evidence(apply, reg):
    out = {}
    old = np.geterr().copy()
    for name, case in reg['overflowWitnesses'].items():
        rates = case['rates']; marks = [float.fromhex(x) for x in case['marks']]
        events = [(10+i, r, m) for i,(r,m) in enumerate(zip(rates, marks))]
        if not all(np.isfinite([t,r,m]).all() and m > 0 for t,r,m in events):
            raise ValueError('funding audit: inadmissible prescribed witness')
        # Prescribed IEEE binary64 witness; no search for an alternative.
        fp = z.FPSort(11, 53); result = z.FPVal(0, fp)
        for rate, mark in zip(rates, marks):
            result = z.fpAdd(z.RNE(), result,
                             z.fpMul(z.RNE(), z.FPVal(rate, fp), z.FPVal(mark, fp)))
        solver = z.Solver(); solver.set(timeout=reg['solverTimeoutMilliseconds'], random_seed=0)
        solver.add(z.fpIsInf(result))
        if solver.check() != z.sat:
            raise ValueError('funding audit: prescribed overflow is not SAT')
        outcomes = {}
        for policy in reg['numpyErrorPolicies']:
            with np.errstate(over=policy, invalid=policy):
                try:
                    f = apply_events(apply, [9,19,29], events)
                except FloatingPointError:
                    outcomes[policy] = 'FloatingPointError'
                else:
                    if not np.isposinf(f[1]):
                        raise ValueError('funding audit: overflow witness changed')
                    outcomes[policy] = 'positive_infinity'
        out[name] = {'smt': 'sat', 'outcomes': outcomes}
    if np.geterr() != old:
        raise ValueError('funding audit: leaked numeric error policy')
    return out


def check_funding():
    reg = registration()
    expressions, loop = extract((ROOT/SOURCE).read_text(), reg['sourceFunctionASTSha256'])
    spec = json.loads((ROOT/'research-notes/registrations/sequential-control-screen-v1.json').read_text())['data']
    grid = grid_certificate(expressions, spec)
    endpoint_certificate()
    witnesses = overflow_evidence(compiled_loop(loop), reg)
    expected = {'product': {'smt': 'sat', 'outcomes': {'ignore': 'positive_infinity', 'raise': 'positive_infinity'}},
                'sum': {'smt': 'sat', 'outcomes': {'ignore': 'positive_infinity', 'raise': 'FloatingPointError'}}}
    if witnesses != expected:
        raise ValueError('funding audit: counterexample/error-policy drift')
    return {'smt': {'F-RL-FUNDING-GRID': 'unsat', 'F-RL-FUNDING-ENDPOINT': 'unsat'},
            'grid': grid, 'counterexample': 'CE-RL-019', 'witnesses': witnesses,
            'scope': 'conditional integer grid semantics and synthetic arithmetic; no historical-data access'}

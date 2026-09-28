"""Exact-real versus binary64 terminal targets; explicitly scoped and refutable."""
import ast
import copy
from pathlib import Path
import struct
import z3 as z
from causal_footprint import structure

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'scripts/research/sequential_learning.py'
TEMPLATE = r'''
def advantages(data: dict, value: Network, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    v = value.forward(data['s']).ravel()
    nxt = value.forward(data['next']).ravel()
    delta = DELTA
    adv = np.empty(len(v))
    carry = 0.0
    for i in reversed(range(len(v))):
        carry = RECURRENCE
        adv[i] = carry
    targets = TARGET
    adv = (adv - adv.mean()) / max(adv.std(), 1e-08)
    return (adv, targets)
'''


def extract(source):
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'advantages']
    if len(nodes) != 1:
        raise ValueError('terminal numerics: missing/duplicate advantages')
    function = copy.deepcopy(nodes[0])
    try:
        expressions = [function.body[2].value, function.body[5].body[0].value, function.body[6].value]
        function.body[2].value = ast.Name(id='DELTA', ctx=ast.Load())
        function.body[5].body[0].value = ast.Name(id='RECURRENCE', ctx=ast.Load())
        function.body[6].value = ast.Name(id='TARGET', ctx=ast.Load())
    except (AttributeError, IndexError):
        raise ValueError('terminal numerics: control-flow drift')
    if structure(function) != structure(ast.parse(TEMPLATE).body[0]):
        raise ValueError('terminal numerics: source skeleton drift')
    delta, carry, _ = expressions
    if not (isinstance(delta, ast.BinOp) and isinstance(delta.op, ast.Sub) and
            isinstance(delta.left, ast.BinOp) and isinstance(delta.left.op, ast.Add) and
            isinstance(carry, ast.BinOp) and isinstance(carry.op, ast.Add)):
        raise ValueError('terminal numerics: unsupported arithmetic decomposition')
    return expressions, [delta.left.right, carry.right]


def scalar(node, symbols, floating):
    key = ast.unparse(node)
    if key in symbols:
        return symbols[key]
    number = (lambda v: z.FPVal(v, z.Float64())) if floating else z.RealVal
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return number(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.Invert)):
        done = scalar(node.operand, symbols, floating)
        return z.If(done, number(0), number(1))
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
        a, b = scalar(node.left, symbols, floating), scalar(node.right, symbols, floating)
        if floating:
            operation = {ast.Add: z.fpAdd, ast.Sub: z.fpSub, ast.Mult: z.fpMul}[type(node.op)]
            return operation(z.RNE(), a, b)
        return {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b}[type(node.op)]()
    raise ValueError('terminal numerics: unsupported expression ' + key)


def symbols(reward, value, nxt, carry, gamma, done):
    return {"data['r']": reward, 'v': value, 'nxt': nxt, 'carry': carry,
            'gamma': gamma, "data['done']": done, "data['done'][i]": done}


def target(expressions, values, floating):
    atoms = dict(values)
    atoms['delta[i]'] = scalar(expressions[0], atoms, floating)
    atoms['adv'] = scalar(expressions[1], atoms, floating)
    return atoms['adv'], scalar(expressions[2], atoms, floating)


def prove(name, premise, conclusion):
    solver = z.Solver()
    solver.set(timeout=10000, random_seed=0)
    solver.add(premise)
    if solver.check() != z.sat:
        raise RuntimeError(name + ': unsatisfied/unknown premise')
    solver.add(z.Not(conclusion))
    result = solver.check()
    if result != z.unsat:
        detail = str(solver.model()) if result == z.sat else solver.reason_unknown()
        raise RuntimeError(name + ': violating terminal claim: ' + detail)


def finite(value):
    return z.And(z.Not(z.fpIsNaN(value)), z.Not(z.fpIsInf(value)))


def fixed_fp(hex_value):
    value = float.fromhex(hex_value)
    bits = struct.unpack('>Q', struct.pack('>d', value))[0]
    return z.fpBVToFP(z.BitVecVal(bits, 64), z.Float64())


def counterexamples(expressions, fixtures):
    results = []
    if [e['id'] for e in fixtures['entries']] != ['CE-RL-010', 'CE-RL-011']:
        raise ValueError('terminal numerics: counterexample roster drift')
    for entry in fixtures['entries']:
        gamma, value = fixed_fp(entry['gammaHex']), fixed_fp(entry['criticHex'])
        rewards = [fixed_fp(h) for h in entry['rewardHex']]
        carry = z.FPVal(0, z.Float64())
        targets = []
        for reward in reversed(rewards):
            carry, result = target(expressions, symbols(reward,value,value,carry,gamma,z.BoolVal(True)), True)
            targets.insert(0, result)
        expected = []
        for actual, stated in zip(targets, entry['expectedTargets']):
            expected.append(z.fpIsNaN(actual) if stated == 'nan' else
                            z.And(z.fpIsInf(actual), z.Not(z.fpIsNegative(actual))) if stated == 'inf' else
                            z.fpEQ(actual, fixed_fp(stated)))
        if len(targets) != len(entry['expectedTargets']):
            raise ValueError('terminal numerics: witness dimensions')
        violation = (z.Not(z.fpEQ(targets[0], rewards[0])) if entry['id'] == 'CE-RL-010' else
                     z.And(z.fpIsNaN(targets[0]), z.fpIsInf(targets[-1])))
        solver = z.Solver(); solver.set(timeout=10000, random_seed=0)
        solver.add(*[finite(v) for v in [gamma,value,*rewards]], *expected, violation)
        if solver.check() != z.sat:
            raise RuntimeError('terminal numerics: prescribed witness not SAT: ' + entry['id'])
        results.append({'id':entry['id'], 'result':'sat', 'expectedTargets':entry['expectedTargets']})
    return results


def check_terminal(fixtures, source=None):
    source = (ROOT / SOURCE).read_text() if source is None else source
    expressions, products = extract(source)
    r, v, nxt, carry, gamma = z.Reals('reward value next carry gamma')
    raw, reconstructed = target(expressions, symbols(r,v,nxt,carry,gamma,z.BoolVal(True)), False)
    prove('F-RL-TERMINAL-REAL', z.And(gamma >= 0, gamma <= 1), z.And(raw == r-v, reconstructed == r))
    n, c, g = z.FPs('next_fp carry_fp gamma_fp', z.Float64())
    atoms = symbols(z.FPVal(0,z.Float64()),z.FPVal(0,z.Float64()),n,c,g,z.BoolVal(True))
    zero, one = z.FPVal(0,z.Float64()), z.FPVal(1,z.Float64())
    premise = z.And(finite(n),finite(c),finite(g),z.fpGEQ(g,zero),z.fpLEQ(g,one))
    for product in products:
        prove('F-RL-TERMINAL-FP-MASK', premise, z.fpEQ(scalar(product,atoms,True),zero))
    return {'smt':{'F-RL-TERMINAL-REAL':'unsat','F-RL-TERMINAL-FP-MASK':'unsat'},
            'positiveQueries':3,'premiseChecks':3,'rounding':'RNE binary64; separate operations',
            'counterexamples':counterexamples(expressions,fixtures),
            'normalizedAdvantageNoninterference':False,'universalRuntimeRefinement':False}

"""Scoped source-derived OPE algebra and prescribed binary64 underflow witness."""
import ast
import copy
from pathlib import Path
import z3 as z
from causal_footprint import structure
from target_v2 import certify

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'scripts/research/sequential_evaluation.py'
REGISTRATION = 'research-notes/registrations/ope-algebra-audit-engineering.json'
TEMPLATE = r'''
def ope_estimates(rewards, actions, behavior_prob, target_prob, q, v, gamma: float) -> dict:
    """Finite-horizon sequential IS/PDIS/WIS/DR, no weight clipping.

    q[i,t] is Q_hat(s_t, logged a_t); v includes final zero bootstrap.
    Exact logged propensities are mandatory. ESS is trajectory-weight ESS.
    """
    inputs = (rewards, actions, behavior_prob, target_prob, q, v)
    if any((np.ma.isMaskedArray(x) for x in inputs)):
        raise ValueError('masked OPE input')
    arrays = [np.asarray(x) for x in inputs]
    r, a, b, pi, q, v = arrays
    if r.ndim != 2 or 0 in r.shape or any((x.shape != r.shape for x in (a, b, pi, q))) or (v.shape != (r.shape[0], r.shape[1] + 1)):
        raise ValueError('OPE shape mismatch or empty episodes')
    if any((x.dtype.kind not in 'iuf' for x in arrays)) or not all((np.isfinite(x).all() for x in arrays)):
        raise ValueError('non-numeric or non-finite OPE input')
    if a.dtype.kind not in 'iu' or np.any(a < 0) or np.any(a >= len(ACTIONS)):
        raise ValueError('invalid logged OPE action')
    if isinstance(gamma, (bool, np.bool_)) or not isinstance(gamma, (int, float, np.integer, np.floating)) or (not np.isfinite(gamma)) or (not 0 <= gamma <= 1):
        raise ValueError('invalid OPE discount')
    if np.any(b <= 0) or np.any(b > 1) or np.any(pi < 0) or np.any(pi > 1):
        raise ValueError('invalid OPE probabilities')
    if np.any(v[:, -1] != 0):
        raise ValueError('nonzero terminal OPE bootstrap')
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            return _ope_estimates(r.astype(float), b.astype(float), pi.astype(float), q.astype(float), v.astype(float), float(gamma))
    except FloatingPointError as exc:
        raise ValueError('non-finite OPE arithmetic') from exc

def _ope_estimates(r, b, pi, q, v, gamma):
    weights = np.cumprod(pi / b, axis=1)
    discount = gamma ** np.arange(r.shape[1])
    returns = r @ discount
    w = weights[:, -1]
    is_values = w * returns
    pdis_values = np.sum(weights * r * discount, axis=1)
    dr_values = v[:, 0] + np.sum(weights * (r + gamma * v[:, 1:] - q) * discount, axis=1)
    ess = float(w.sum() ** 2 / (w @ w)) if w @ w > 0 else 0.0
    result = {'ordinaryIS': float(is_values.mean()), 'perDecisionIS': float(pdis_values.mean()), 'weightedIS': float(w @ returns / w.sum()) if w.sum() else None, 'doublyRobust': float(dr_values.mean()), 'effectiveSampleSize': ess, 'maxTrajectoryWeight': float(max(w)), 'nonzeroTrajectories': int(np.count_nonzero(w)), 'episodes': len(w), 'horizonDecisions': r.shape[1], 'weightClipping': 'none', 'reliable': False, 'reason': 'Conditional simulator OPE; no live behavior support. ESS and estimator agreement are additional necessary gates.'}
    rng = np.random.default_rng(20260917)
    draws = rng.integers(len(w), size=(1000, len(w)))
    result['conditionalBootstrap95'] = {k: np.quantile(values[draws].mean(1), [0.025, 0.975]).tolist() for k, values in (('IS', is_values), ('PDIS', pdis_values), ('DR', dr_values))}
    return result
'''


def extract(source):
    """Check both full function skeletons, then translate only named expressions."""
    expressions = {}
    for text, save in ((source, True), (TEMPLATE, False)):
        nodes = {}
        for name in ('ope_estimates', '_ope_estimates'):
            matches = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name]
            if len(matches) != 1:
                raise ValueError('OPE algebra: missing/duplicate function')
            nodes[name] = copy.deepcopy(matches[0])
        try:
            core = nodes['_ope_estimates'].body
            slots = [('dr', core[6], None), ('ess', core[7], None)]
            index = next(i for i, key in enumerate(core[8].value.keys) if key.value == 'weightedIS')
            slots.append(('wis', core[8].value, index))
            for name, parent, index in slots:
                value = parent.value if index is None else parent.values[index]
                if save:
                    expressions[name] = copy.deepcopy(value)
                placeholder = ast.Name(id=name.upper(), ctx=ast.Load())
                if index is None:
                    parent.value = placeholder
                else:
                    parent.values[index] = placeholder
        except (AttributeError, IndexError, StopIteration):
            raise ValueError('OPE algebra: source control-flow drift')
        if save:
            actual = nodes
        elif any(structure(actual[name]) != structure(node) for name, node in nodes.items()):
            raise ValueError('OPE algebra: source skeleton drift')
    # The WIS conditional has a None branch; translate only on positive mass.
    wis = expressions['wis']
    if (not isinstance(wis, ast.IfExp) or ast.unparse(wis.test) != 'w.sum()' or
            not isinstance(wis.orelse, ast.Constant) or wis.orelse.value is not None):
        raise ValueError('OPE algebra: WIS admission drift')
    return expressions


def expression(node, atoms):
    """Restricted exact-real scalar/vector AST interpretation, not NumPy proof."""
    key = ast.unparse(node)
    if key in atoms:
        return atoms[key]
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return z.RealVal(str(node.value))
    if isinstance(node, ast.Call) and ast.unparse(node.func) == 'float' and len(node.args) == 1 and not node.keywords:
        return expression(node.args[0], atoms)
    if isinstance(node, ast.Call) and ast.unparse(node.func) == 'np.sum' and len(node.args) == 1:
        if [(k.arg, ast.unparse(k.value)) for k in node.keywords] != [('axis', '1')]:
            raise ValueError('OPE algebra: sum axis drift')
        values = expression(node.args[0], atoms)
        if not isinstance(values, list):
            raise ValueError('OPE algebra: expected row sum')
        return z.Sum(values)
    if isinstance(node, ast.BinOp):
        left, right = expression(node.left, atoms), expression(node.right, atoms)
        operations = {ast.Add: lambda a, b: a+b, ast.Sub: lambda a, b: a-b,
                      ast.Mult: lambda a, b: a*b, ast.Div: lambda a, b: a/b}
        if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant) and node.right.value == 2:
            op = lambda a, b: a*a
        elif type(node.op) in operations:
            op = operations[type(node.op)]
        else:
            raise ValueError('OPE algebra: unsupported arithmetic')
        if isinstance(left, list) or isinstance(right, list):
            size = len(left) if isinstance(left, list) else len(right)
            left = left if isinstance(left, list) else [left]*size
            right = right if isinstance(right, list) else [right]*size
            if len(left) != len(right):
                raise ValueError('OPE algebra: vector shape drift')
            return [op(a, b) for a, b in zip(left, right)]
        return op(left, right)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Gt):
        return expression(node.left, atoms) > expression(node.comparators[0], atoms)
    if isinstance(node, ast.IfExp):
        return z.If(expression(node.test, atoms), expression(node.body, atoms), expression(node.orelse, atoms))
    raise ValueError('OPE algebra: unsupported expression ' + key)


def moments(weights, returns):
    return {'w.sum()': z.Sum(weights), 'w @ w': z.Sum([w*w for w in weights]),
            'w @ returns': z.Sum([w*r for w, r in zip(weights, returns)])}


def check_real(expressions):
    w0, w1, c, r0, r1, lo, hi = z.Reals('ope_w0 ope_w1 ope_scale ope_r0 ope_r1 ope_lo ope_hi')
    weights, returns = [w0, w1], [r0, r1]
    atoms = moments(weights, returns)
    scaled = moments([c*w for w in weights], returns)
    ess, scaled_ess = [expression(expressions['ess'], a) for a in (atoms, scaled)]
    premise = z.And(w0 >= 0, w1 >= 0, w0+w1 > 0, c > 0)
    certify('F-RL-OPE-ESS-REAL', premise,
            z.And(ess >= 1, ess <= 2, scaled_ess == ess), [w0 == 1, w1 == 1, c == 1])
    wis, scaled_wis = [expression(expressions['wis'].body, a) for a in (atoms, scaled)]
    certify('F-RL-OPE-WIS-REAL', z.And(premise, lo <= r0, r0 <= hi, lo <= r1, r1 <= hi),
            z.And(wis >= lo, wis <= hi, scaled_wis == wis),
            [w0 == 1, w1 == 1, c == 1, lo == 0, hi == 1, r0 == 0, r1 == 1])
    gamma = z.Real('ope_gamma')
    for size in range(1, 7):
        r = [z.Real(f'ope_r_{size}_{i}') for i in range(size)]
        v = [z.Real(f'ope_v_{size}_{i}') for i in range(size)] + [z.RealVal(0)]
        discount = [z.RealVal(1)]
        for _ in range(1, size):
            discount.append(discount[-1]*gamma)
        atoms = {'v[:, 0]': v[0], 'weights': [z.RealVal(1)]*size, 'r': r,
                 'gamma': gamma, 'v[:, 1:]': v[1:], 'q': v[:-1], 'discount': discount}
        dr = expression(expressions['dr'], atoms)
        reference = z.Sum([a*b for a, b in zip(r, discount)])
        certify('F-RL-OPE-DR-TELESCOPE-' + str(size), z.And(gamma >= 0, gamma <= 1),
                dr == reference, [gamma == 1])
    return {name: 'unsat' for name in ('F-RL-OPE-ESS-REAL', 'F-RL-OPE-WIS-REAL', 'F-RL-OPE-DR-TELESCOPE')}


def check_witness(fixtures, registration):
    expected = {'id': 'CE-RL-018', 'episodes': 2, 'horizon': 6, 'behaviorProbability': 1,
                'targetProbabilityHex': '0x1.0000000000000p-100', 'reward': 1, 'q': 0, 'v': 0,
                'gamma': 1, 'expectedFinalWeightHex': '0x1.0000000000000p-600',
                'expectedESS': 0, 'expectedNonzero': 2, 'expectedWIS': 6,
                'underflowModes': ['ignore', 'raise']}
    if fixtures != {'schemaVersion': 1, 'entries': [expected]} or registration['counterexample'] != expected:
        raise ValueError('OPE algebra: prescribed witness drift')
    if (registration['financialTrialBudget'] != 0 or registration['marketDataAccess'] is not False or
            registration['holdoutAccess'] is not False or registration['originalOpeReruns'] != 0):
        raise ValueError('OPE algebra: engineering boundary drift')
    zero, one = z.FPVal(0, z.Float64()), z.FPVal(1, z.Float64())
    pi = z.FPVal(float.fromhex(expected['targetProbabilityHex']), z.Float64())
    weight = one
    for _ in range(expected['horizon']):
        weight = z.fpMul(z.RNE(), weight, z.fpDiv(z.RNE(), pi, one))
    total = z.fpAdd(z.RNE(), weight, weight)
    square = z.fpMul(z.RNE(), weight, weight)
    denominator = z.fpAdd(z.RNE(), square, square)
    numerator = z.fpMul(z.RNE(), total, total)
    ess = z.If(z.fpGT(denominator, zero), z.fpDiv(z.RNE(), numerator, denominator), zero)
    solver = z.Solver(); solver.set(timeout=10000, random_seed=0)
    solver.add(weight == z.FPVal(float.fromhex(expected['expectedFinalWeightHex']), z.Float64()),
               z.fpGT(weight, zero), square == zero, denominator == zero, numerator == zero, ess == zero)
    if solver.check() != z.sat:
        raise RuntimeError('OPE algebra: underflow witness not reproduced')
    return {'id': expected['id'], 'result': 'sat', 'rounding': 'RNE', 'format': 'IEEE binary64',
            'weightHex': expected['expectedFinalWeightHex'], 'effectiveSampleSize': 0,
            'nonzeroTrajectories': 2, 'publicRuntimeCheckedBy': 'OPEAlgebraTests',
            'currentShortOPEReachabilityClaimed': False}


def check_ope(fixtures, registration, source=None):
    expressions = extract((ROOT / SOURCE).read_text() if source is None else source)
    return {'smt': check_real(expressions), 'queries': 8, 'premiseChecks': 8,
            'counterexample': check_witness(fixtures, registration),
            'runtimeRefinement': False, 'binary64AccuracyVerified': False,
            'statisticalReliabilityVerified': False, 'realEpisodes': 2, 'drHorizons': [1, 6]}

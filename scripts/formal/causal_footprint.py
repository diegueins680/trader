"""Fail-closed source dependency analysis with source-derived SMT read bounds.

The restricted AST interpreter and listed primitive semantics are trusted, not a
verified Python compiler. See causal-footprint-contract.md for the exact scope.
"""
import ast
from pathlib import Path
import z3 as z

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'scripts/research/sequential_env.py'
REQUIREMENT = 'F-RL-FEATURE-FOOTPRINT'

# Audited metadata-only helper contracts. Compare AST, including complete bodies;
# comments, whitespace and docstrings do not influence their semantics.
HELPERS = {
    '_integer': '''
def _integer(value) -> bool:
    return type(value) is int or (isinstance(value, np.integer) and value.dtype.kind in "iu")
''',
    '_real_series': '''
def _real_series(value) -> bool:
    return (isinstance(value, np.ndarray) and not np.ma.isMaskedArray(value) and
            value.ndim == 1 and value.dtype.kind in "iuf")
''',
}
ADMISSION = '''
if not _real_series(prices) or not _integer(t) or not 24 <= t < len(prices):
    return None
'''
CONVERSION = 't = int(t)'


def fail(message):
    raise ValueError('causal footprint: ' + message)


def without_doc(function):
    body = function.body
    return body[1:] if (body and isinstance(body[0], ast.Expr) and
                       isinstance(body[0].value, ast.Constant) and
                       isinstance(body[0].value.value, str)) else body


def structure(node):
    return ast.dump(node, include_attributes=False)


def integer(node, t):
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return z.IntVal(node.value)
    if isinstance(node, ast.Name) and node.id == 't':
        return t
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left, right = integer(node.left, t), integer(node.right, t)
        return left + right if isinstance(node.op, ast.Add) else left - right
    fail('unsupported slice arithmetic: ' + ast.unparse(node))


class Dependencies:
    """All admitted locals depend only on public inputs and checked raw slices."""
    def __init__(self):
        self.locals = {'t'}
        self.reads = []

    def expression(self, node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float, bool, type(None)):
            return
        if isinstance(node, ast.Name):
            if node.id not in self.locals:
                fail('unbounded or unknown value input: ' + node.id)
            return
        if isinstance(node, (ast.List, ast.Tuple)):
            for item in node.elts:
                self.expression(item)
            return
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            self.expression(node.left)
            self.expression(node.right)
            return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.USub, ast.UAdd)):
            self.expression(node.operand)
            return
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            for value in node.values:
                self.expression(value)
            return
        if isinstance(node, ast.Compare) and all(isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)) for op in node.ops):
            for value in (node.left, *node.comparators):
                self.expression(value)
            return
        if isinstance(node, ast.IfExp):
            for value in (node.test, node.body, node.orelse):
                self.expression(value)
            return
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id == 'prices':
                part = node.slice
                if not isinstance(part, ast.Slice) or part.lower is None or part.upper is None or part.step is not None:
                    fail('raw prices require a bounded unit-stride slice')
                # Only immutable public t and integer literals enter these terms.
                integer(part.lower, z.Int('t'))
                integer(part.upper, z.Int('t'))
                self.reads.append((part.lower, part.upper))
            else:
                self.expression(node.value)
                if isinstance(node.slice, ast.Slice):
                    for value in (node.slice.lower, node.slice.upper, node.slice.step):
                        if value is not None:
                            self.expression(value)
                else:
                    self.expression(node.slice)
            return
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name in ('np.asarray', 'np.array', 'np.isfinite', 'np.any', 'np.std'):
                if len(node.args) != 1:
                    fail('unsupported primitive arity')
                for keyword in node.keywords:
                    if not (name == 'np.asarray' and keyword.arg == 'dtype' and
                            isinstance(keyword.value, ast.Name) and keyword.value.id == 'float'):
                        fail('unsupported primitive keyword')
                self.expression(node.args[0])
                return
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'all' and not node.args and not node.keywords:
                self.expression(node.func.value)
                return
            fail('unsupported call: ' + name)
        fail('unsupported expression: ' + type(node).__name__)

    def statement(self, node):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in self.locals or name in ('prices', 'np', 'float'):
                fail('reassignment or reserved binding: ' + name)
            self.expression(node.value)
            self.locals.add(name)
            return
        if isinstance(node, ast.Return) and node.value is not None:
            self.expression(node.value)
            return
        if isinstance(node, ast.If):
            self.expression(node.test)
            # The actual feature core uses an early rejection only. No branch
            # assignments, effects or untracked phi bindings are admitted.
            empty_return = ast.parse('return None').body[0]
            if node.orelse or len(node.body) != 1 or structure(node.body[0]) != structure(empty_return):
                fail('unsupported conditional body')
            return
        fail('unsupported statement: ' + type(node).__name__)


def analyze(source):
    module = ast.parse(source)
    functions = {}
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in functions:
                fail('duplicate function: ' + node.name)
            functions[node.name] = node
    for name, template in HELPERS.items():
        actual = functions.get(name)
        expected = ast.parse(template).body[0]
        if actual is None or not isinstance(actual, ast.FunctionDef):
            fail('missing metadata helper')
        actual.body = without_doc(actual)
        if structure(actual) != structure(expected):
            fail('metadata helper contract drift: ' + name)
    function = functions.get('market_features')
    if not isinstance(function, ast.FunctionDef) or function.decorator_list:
        fail('missing or decorated feature function')
    signature = ast.parse('def market_features(prices: np.ndarray, t: int) -> np.ndarray | None: pass').body[0]
    if structure(function.args) != structure(signature.args) or structure(function.returns) != structure(signature.returns):
        fail('feature signature drift')
    body = without_doc(function)
    if (len(body) < 3 or structure(body[0]) != structure(ast.parse(ADMISSION).body[0]) or
            structure(body[1]) != structure(ast.parse(CONVERSION).body[0])):
        fail('admission precondition drift')
    checker = Dependencies()
    for statement in body[2:]:
        checker.statement(statement)
    if not checker.reads or not isinstance(body[-1], ast.Return):
        fail('missing market read or terminal return')
    return checker.reads


def check_source(source=None):
    if source is None:
        source = (ROOT / SOURCE).read_text()
    reads = analyze(source)
    t, n = z.Ints('t n')
    premise = z.And(t >= 24, t < n)
    nonvacuous = z.Solver()
    nonvacuous.set(timeout=10000, random_seed=0)
    nonvacuous.add(premise)
    if nonvacuous.check() != z.sat:
        raise RuntimeError('causal footprint: non-vacuity check failed')
    receipts = []
    for lower, upper in reads:
        lo, hi = integer(lower, t), integer(upper, t)
        theorem = z.And(lo >= 0, lo >= t - 24, lo <= hi, hi <= n, hi <= t + 1)
        solver = z.Solver()
        solver.set(timeout=10000, random_seed=0)
        solver.add(premise, z.Not(theorem))
        result = solver.check()
        if result != z.unsat:
            detail = str(solver.model()) if result == z.sat else solver.reason_unknown()
            raise RuntimeError('causal footprint: unsafe source read: ' + detail)
        receipts.append({'lower': ast.unparse(lower), 'upperExclusive': ast.unparse(upper), 'result': 'unsat'})
    return {'requirement': REQUIREMENT, 'status': 'smt_verified', 'source': SOURCE,
            'function': 'market_features', 'reads': receipts,
            'domain': 'unbounded integers 24 <= t < n; ordinary fixed-shape numeric arrays',
            'metadataHelpers': sorted(HELPERS), 'universalRuntimeRefinement': False}

"""SMT proofs of the exposure shield's action-admission invariants.

The Z3 term for ``project`` is derived from the abstract syntax tree of
``scripts/research/exposure_rl/shield.py`` (not hand-copied), over unbounded
mathematical integers. Each claim is certified as: premise satisfiable (not
vacuous) and premise-and-not-conclusion unsatisfiable. SAT, UNKNOWN or a
timeout is a failure.

Scope: the integer projection and admission logic for every valid snapshot and
every integer proposal. Not in scope: binary64 decoding of the raw output
(covered by exhaustive/seeded conformance), the NumPy batch form (compared with
the scalar form by tests), and anything about realised losses.
"""

from __future__ import annotations

import ast
from pathlib import Path

import z3 as z

ROOT = Path(__file__).resolve().parents[3]
SOURCE = "scripts/research/exposure_rl/shield.py"
TIMEOUT_MS = 10_000
LOGGED_ONLY = {"reasons"}  # intervention bits are logged and never read for control


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("exposure shield proof: " + reason)


def certify(premise, conclusion) -> None:
    for formula, wanted in ((premise, z.sat), (z.And(premise, z.Not(conclusion)), z.unsat)):
        solver = z.Solver()
        solver.set(timeout=TIMEOUT_MS, random_seed=0)
        solver.add(formula)
        require(solver.check() == wanted, "SAT/UNKNOWN/vacuous proof")


def constants(tree: ast.Module) -> dict[str, int]:
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Constant) and type(node.value.value) is int:
            found[node.targets[0].id] = node.value.value
    return found


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(matches) == 1, "missing or duplicate function " + name)
    return matches[0]


def z_abs(value):
    return z.If(value >= 0, value, -value)


def z_min(a, b):
    return z.If(a <= b, a, b)


def same_side(a, b):
    """Linear form of a*b >= 0."""
    return z.Or(z.And(a >= 0, b >= 0), z.And(a <= 0, b <= 0))


def z_clamp(value, low, high):
    return z.If(value < low, low, z.If(value > high, high, value))


class Translator:
    """Translate the restricted straight-line/if subset used by ``project``."""

    def __init__(self, consts: dict[str, int], snapshot: dict):
        self.consts, self.snapshot = consts, snapshot

    def expr(self, node, env):
        if isinstance(node, ast.Constant):
            require(type(node.value) in (int, bool), "unsupported constant")
            return z.BoolVal(node.value) if type(node.value) is bool else z.IntVal(node.value)
        if isinstance(node, ast.Name):
            require(node.id not in LOGGED_ONLY, "logged-only variable read for control")
            if node.id in env:
                return env[node.id]
            require(node.id in self.consts, "unknown name " + node.id)
            return z.IntVal(self.consts[node.id])
        if isinstance(node, ast.Attribute):
            require(isinstance(node.value, ast.Name) and node.value.id == "snapshot" and node.attr in self.snapshot, "unsupported attribute")
            return self.snapshot[node.attr]
        if isinstance(node, ast.UnaryOp):
            value = self.expr(node.operand, env)
            if isinstance(node.op, ast.USub):
                return -value
            require(isinstance(node.op, ast.Not), "unsupported unary operator")
            return z.Not(value)
        if isinstance(node, ast.BinOp):
            left, right = self.expr(node.left, env), self.expr(node.right, env)
            if isinstance(node.op, ast.Add):
                return left + right
            require(isinstance(node.op, ast.Sub), "unsupported binary operator")
            return left - right
        if isinstance(node, ast.BoolOp):
            values = [self.expr(value, env) for value in node.values]
            return z.Or(*values) if isinstance(node.op, ast.Or) else z.And(*values)
        if isinstance(node, ast.Compare):
            require(len(node.ops) == 1, "chained comparison")
            left, right = self.expr(node.left, env), self.expr(node.comparators[0], env)
            table = {ast.Lt: left < right, ast.LtE: left <= right, ast.Gt: left > right, ast.GtE: left >= right, ast.Eq: left == right, ast.NotEq: left != right}
            require(type(node.ops[0]) in table, "unsupported comparison")
            return table[type(node.ops[0])]
        if isinstance(node, ast.Call):
            require(isinstance(node.func, ast.Name) and not node.keywords, "unsupported call")
            args = [self.expr(arg, env) for arg in node.args]
            table = {"abs": (1, z_abs), "min": (2, z_min), "_clamp": (3, z_clamp)}
            require(node.func.id in table and len(args) == table[node.func.id][0], "unsupported call " + node.func.id)
            return table[node.func.id][1](*args)
        require(False, "unsupported expression " + type(node).__name__)

    def block(self, statements, env):
        env = dict(env)
        for statement in statements:
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
                continue  # docstring
            if isinstance(statement, ast.Assign):
                require(len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name), "unsupported assignment")
                name = statement.targets[0].id
                if name not in LOGGED_ONLY:
                    env[name] = self.expr(statement.value, env)
            elif isinstance(statement, ast.AugAssign):
                require(isinstance(statement.target, ast.Name) and statement.target.id in LOGGED_ONLY, "augmented assignment to a control variable")
            elif isinstance(statement, ast.If):
                test = self.expr(statement.test, env)
                yes, no = self.block(statement.body, env), self.block(statement.orelse, env)
                require("__return__" not in yes and "__return__" not in no, "return inside a branch")
                for name in set(yes) | set(no):
                    a, b = yes.get(name, env.get(name)), no.get(name, env.get(name))
                    if a is None or b is None:
                        env.pop(name, None)  # branch-local; any later read fails as unknown
                    else:
                        env[name] = z.If(test, a, b)
            elif isinstance(statement, ast.Return):
                require(isinstance(statement.value, ast.Tuple) and len(statement.value.elts) == 2, "unexpected return shape")
                env["__return__"] = self.expr(statement.value.elts[0], env)
            else:
                require(False, "unsupported statement " + type(statement).__name__)
        return env


def project_term(source: str, snapshot: dict, proposal):
    tree = ast.parse(source)
    node = function(tree, "project")
    require([arg.arg for arg in node.args.args] == ["snapshot", "proposal"], "unexpected project signature")
    env = Translator(constants(tree), snapshot).block(node.body, {"proposal": proposal})
    require("__return__" in env, "project has no top-level return")
    return env["__return__"], constants(tree)


def claims(source: str) -> dict:
    current, drawdown, headroom, proposal = z.Ints("current drawdown headroom proposal")
    fresh = z.Bool("fresh")
    snapshot = {"exposure": current, "drawdown": drawdown, "fresh": fresh, "headroom": headroom}
    target, c = project_term(source, snapshot, proposal)
    cap, min_trade, dd_limit, bound, scale = c["CAP"], c["MIN_TRADE"], c["DD_LIMIT"], c["SNAPSHOT_BOUND"], c["SCALE"]
    require(0 < min_trade < cap < scale <= bound and 0 < dd_limit <= scale, "constants are not ordered as the proofs assume")
    valid = z.And(z_abs(current) <= bound, drawdown >= 0, drawdown <= scale, headroom >= 0, headroom <= scale)
    risk_off = z.Or(z.Not(fresh), drawdown >= dd_limit)
    intent = z_abs(target - current) >= min_trade  # the wrapper forms an intent exactly then
    # Fallback for an invalid proposal: the wrapper proposes the current exposure.
    fallback, _ = project_term(source, snapshot, current)
    again, _ = project_term(source, {**snapshot, "exposure": target}, target)
    low, high = z_min(current, proposal), z.If(current >= proposal, current, proposal)
    return {
        "F-RLX-SHIELD-CAP": (valid, z_abs(target) <= cap),
        "F-RLX-SHIELD-REDUCE": (z.And(valid, risk_off), z.And(z_abs(target) <= z_abs(current), same_side(target, current))),
        "F-RLX-SHIELD-HEADROOM": (valid, z_abs(target) <= z_min(z_abs(current), cap) + headroom),
        "F-RLX-SHIELD-NOROUNDUP": (valid, z.Or(z.And(low <= target, target <= high), z.And(z_abs(target) < z_abs(current), same_side(target, current)))),
        "F-RLX-SHIELD-IDEMPOTENT": (valid, again == target),
        "F-RLX-SHIELD-FALLBACK": (valid, z.And(z_abs(fallback) <= z_abs(current), same_side(fallback, current), z_abs(fallback) <= cap, z.Implies(z_abs(current) <= cap, fallback == current))),
        "F-RLX-SHIELD-OVERCAP": (z.And(valid, z_abs(current) >= cap + min_trade), z.And(intent, z_abs(target) < z_abs(current))),
    }


def reservation_claims() -> dict:
    """Atomic reservation over three symbolic accounts (mirrors ``shield.reserve``)."""
    budget, version, seen, wanted = z.Ints("budget version seen wanted")
    held = z.Ints("h0 h1 h2")
    index = z.Int("index")
    previous = z.If(index == 0, held[0], z.If(index == 1, held[1], held[2]))
    total = z.Sum(held) - previous + wanted
    refused = z.Or(seen != version, wanted < 0, index < 0, index >= 3, z.And(total > budget, wanted > previous))
    after = [z.If(z.And(z.Not(refused), index == i), wanted, held[i]) for i in range(3)]
    new_version = z.If(refused, version, version + 1)
    shape = z.And(*[h >= 0 for h in held], budget >= 0)
    return {
        "F-RLX-RESERVE-BUDGET": (z.And(shape, z.Sum(held) <= budget), z.Sum(after) <= budget),
        "F-RLX-RESERVE-STALE": (z.And(shape, seen != version), z.And(new_version == version, *[after[i] == held[i] for i in range(3)])),
        "F-RLX-RESERVE-NOINCREASE": (z.And(shape, z.Sum(held) > budget), z.Sum(after) <= z.Sum(held)),
        "F-RLX-RESERVE-VERSION": (z.And(shape, z.Not(refused)), new_version == version + 1),
    }


def check(source: str | None = None) -> dict:
    source = (ROOT / SOURCE).read_text(encoding="utf-8") if source is None else source
    proved = {}
    for name, (premise, conclusion) in {**claims(source), **reservation_claims()}.items():
        certify(premise, conclusion)
        proved[name] = "unsat"
    return {"smt": proved, "source": SOURCE, "domain": "all valid snapshots and all integer proposals; unbounded integers"}


if __name__ == "__main__":
    print(check())

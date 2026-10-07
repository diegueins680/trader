"""Obligation 8: per-row binary64 reconciliation of the frozen Replay ledger against its stated formulas."""
import ast
from fractions import Fraction as Q
import json
from pathlib import Path
import sys

import numpy as np
import z3 as z
from ppo_successor import certify
from bounded_values import ENV, H, HIGH, LOW_PRICE, STRESSES, U, _method, _named, dbl, replay_lock

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = 'research-notes/registrations/accounting-reconciliation-engineering.json'
ROUNDINGS = 14   # 10 chain operations (gross+funding, equity+, 3+1 trade, 3+1 liquidation) + 4 cost merges
STATED = {'rollForward': (Q(15), Q(15)), 'funding': (Q(3), Q(4)), 'costTerm': (Q(4), Q(10)), 'impact': (Q(6), Q(10))}
GROSS_E = Q(1, 2**600)
ROW_RECORD = ("self.rows.append({'t': self.t, 'net': self.equity / old_equity - 1, 'equity': self.equity, 'gross': gross, "
              "'funding': funding, 'exposure': exposure, 'drawdown': 1 - self.equity / self.peak, "
              "'rewardPenalty': 100 * self.execution.risk_penalty * exposure ** 2 * x[5] ** 2, **costs})")


def require(ok, reason):
    if not ok:
        raise ValueError('accounting reconciliation: ' + reason)


def bind(sources=None):
    sources = sources or {}
    locked = replay_lock(sources)                 # complete AST lock: any unreviewed edit fails closed
    env = ast.parse(sources.get(ENV, (ROOT / ENV).read_text()))
    replay = _named(env, 'Replay')
    writes = sorted(ast.unparse(s) for s in ast.walk(replay) if isinstance(s, (ast.Assign, ast.AugAssign)) and
                    'self.equity' in {ast.unparse(t) for t in ast.walk(s.targets[0] if isinstance(s, ast.Assign) else s.target)})
    require(writes == sorted(["self.equity -= sum((terms[k] for k in ('fee', 'spread', 'slippage', 'impact')))",
                       'self.equity += gross + funding', 'self.units, self.equity, self.peak = (0.0, 1.0, 1.0)']), 'every equity write is a reviewed ledger operation')
    step = _method(replay, 'step')
    texts = {ast.unparse(s) for s in ast.walk(step) if isinstance(s, (ast.Assign, ast.AugAssign))}
    require({'gross, funding = (self.units * (p1 - p0), -self.units * f * self.execution.funding_multiplier)',
             "costs = dict.fromkeys(('turnover', 'fee', 'spread', 'slippage', 'impact'), 0.0)",
             'costs = self._trade(self.pending[1])', 'liquidation = self._trade(0.0, terminal=True)',
             'costs = {k: costs[k] + liquidation[k] for k in costs}'} <= texts, 'mark-to-market, cost and merge expressions')
    row = next(n for n in ast.walk(step) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'self.rows.append')
    keys = {ast.unparse(k) for k in row.args[0].keys if k is not None}
    values = {ast.unparse(k): ast.unparse(v) for k, v in zip(row.args[0].keys, row.args[0].values) if k is not None}
    require({"'equity'", "'gross'", "'funding'"} <= keys and values["'equity'"] == 'self.equity' and values["'gross'"] == 'gross'
            and values["'funding'"] == 'funding' and any(k is None and ast.unparse(v) == 'costs' for k, v in zip(row.args[0].keys, row.args[0].values)),
            'row records the executed equity, gross, funding and merged costs')
    # Control-flow order of one bar: the modeled ledger sequence must precede the row record, and nothing may
    # debit equity after it. Only these top-level statements may touch equity, costs or the row.
    loop = next(n for n in ast.walk(step) if isinstance(n, ast.While) and ast.unparse(n.test) == 'self.t < end')
    order = [ast.unparse(s).split(chr(10))[0] for s in loop.body]
    def at(text):
        hits = [i for i, line in enumerate(order) if line == text]
        require(len(hits) == 1, 'statement must occur exactly once in the bar loop: ' + text)
        return hits[0]
    mark, reset = at('self.equity += gross + funding'), at("costs = dict.fromkeys(('turnover', 'fee', 'spread', 'slippage', 'impact'), 0.0)")
    trade_if = at('if self.failure is None and self.pending is not None and (self.pending[0] <= self.t):')
    terminal_if, record = at('if terminal:'), next(i for i, line in enumerate(order) if line.startswith('self.rows.append('))
    require(mark < reset < trade_if < terminal_if < record and order[record + 1:] == ['if self.done:'] and record + 2 == len(order),
            'bar order: mark < cost reset < trade < terminal liquidation+merge < row record, then only the break')
    term_body = [ast.unparse(s) for s in loop.body[terminal_if].body]
    require(term_body[1].startswith('if isfinite(self.equity) and self.equity > 0:') and
            ast.unparse(loop.body[terminal_if].body[1].body[0]) == 'liquidation = self._trade(0.0, terminal=True)' and
            ast.unparse(loop.body[terminal_if].body[1].body[1]) == 'costs = {k: costs[k] + liquidation[k] for k in costs}',
            'terminal liquidation precedes the merge within the terminal block')
    require(ast.unparse(loop.body[trade_if].body[0]) == 'costs = self._trade(self.pending[1])', 'trade assigns the bar costs')
    writers = [i for i, s in enumerate(loop.body) for n in ast.walk(s) if isinstance(n, (ast.Assign, ast.AugAssign))
               and 'self.equity' in {ast.unparse(t) for t in ast.walk(n.targets[0] if isinstance(n, ast.Assign) else n.target)}]
    calls = [i for i, s in enumerate(loop.body) for n in ast.walk(s) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'self._trade']
    require(all(i < record for i in writers + calls), 'no equity write or trade call after the row record')
    _only_reviewed_writes(step, 'costs', {"costs = dict.fromkeys(('turnover', 'fee', 'spread', 'slippage', 'impact'), 0.0)",
                                          'costs = self._trade(self.pending[1])',
                                          'costs = {k: costs[k] + liquidation[k] for k in costs}'},
                          allowed_reads={'costs[k]', 'comprehension-iter', 'dict-unpack'})
    appends = [n for n in ast.walk(replay) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'self.rows.append']
    row_writes = [ast.unparse(n) for n in ast.walk(replay) if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)) and
                  any(ast.unparse(x).startswith('self.rows') for x in ast.walk(n) if isinstance(x, (ast.Attribute, ast.Subscript))
                      and isinstance(getattr(x, 'ctx', None), (ast.Store, ast.Del)))]
    require(len(appends) == 1 and row_writes == ['self.rows: list[dict] = []'], 'single row record and no other row mutation')
    # The row literal is pinned: no explicit cost keys and **costs is its single, final unpack.
    require(ast.unparse(appends[0]) == ROW_RECORD, 'reviewed row-record shape')
    # Exactly the two modeled debit call sites exist in the whole Replay class.
    trade_calls = sorted(ast.unparse(n) for n in ast.walk(replay) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'self._trade')
    require(trade_calls == ['self._trade(0.0, terminal=True)', 'self._trade(self.pending[1])'], 'exactly the trade and liquidation call sites')
    require(not [n for n in ast.walk(replay) if isinstance(n, ast.Attribute) and n.attr == '_trade' and
                 not any(isinstance(c, ast.Call) and c.func is n for c in ast.walk(replay))], '_trade is never referenced except by its two calls')
    require(not [n for n in ast.walk(replay) if isinstance(n, ast.Call) and ast.unparse(n.func).startswith('self.rows.')
                 and ast.unparse(n.func) != 'self.rows.append'], 'no other row-list method calls')
    trade = _method(replay, '_trade')
    terms = next(s for s in ast.walk(trade) if isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == 'terms')
    require(ast.unparse(terms.value) == "{'turnover': cash, 'fee': cash * 0.0005 * cfg.cost_multiplier, 'spread': cash * 5e-05 * cfg.cost_multiplier, "
            "'slippage': cash * 0.00045 * cfg.cost_multiplier, 'impact': cash * cfg.impact_bps * 0.0001 * np.sqrt(turnover)}", 'cost formulas')
    _only_reviewed_writes(trade, 'terms', {ast.unparse(terms)}, allowed_reads={'terms[k]', 'return terms'})
    rejected = [ast.unparse(r) for r in ast.walk(trade) if isinstance(r, ast.Return)]
    require("return {'turnover': 0.0, 'fee': 0.0, 'spread': 0.0, 'slippage': 0.0, 'impact': 0.0}" in rejected, 'turnover rejection debits nothing')
    loop = next(n for n in ast.walk(step) if isinstance(n, ast.While))
    first_exit = next(s for s in loop.body if isinstance(s, ast.If) and any(isinstance(b, ast.Break) for b in s.body))
    require(ast.unparse(first_exit.test) == 'not all((_finite_real(v) for v in (p0, p1, f))) or min(p0, p1) <= 0' and
            loop.body.index(first_exit) < next(i for i, s in enumerate(loop.body) if 'self.rows.append' in ast.unparse(s)),
            'invalid market data rejects before any row')
    return {'status': 'exhaustively_checked', 'lockedDefinitions': locked, 'equityWrites': len(writes),
            'scope': 'complete AST lock of the reviewed Replay class and helpers plus structural checks; interpreter semantics assumed'}


def _only_reviewed_writes(scope, name, allowed, allowed_reads):
    """Every occurrence of `name` must be reviewed. Stores: only the reviewed assignments. Loads: only the
    reviewed read contexts (value subscripts, the merge comprehension iterable, the row-record **unpack, the
    debit sum, the return). Any alias, call argument, method call or other exposure fails closed."""
    parents = {c: p for p in ast.walk(scope) for c in ast.iter_child_nodes(p)}
    for node in ast.walk(scope):
        if not (isinstance(node, ast.Name) and node.id == name):
            continue
        parent = parents[node]
        if isinstance(node.ctx, ast.Store):
            stmt = parent
            while not isinstance(stmt, ast.stmt):
                stmt = parents[stmt]
            require(isinstance(stmt, ast.Assign) and stmt.targets == [node] and ast.unparse(stmt) in allowed,
                    f'unreviewed write to {name}: ' + ast.unparse(stmt))
            continue
        require(not isinstance(node.ctx, ast.Del), f'deletion of {name}')
        if isinstance(parent, ast.comprehension) and parent.iter is node:
            context = 'comprehension-iter'
        elif isinstance(parent, ast.Dict) and any(k is None and v is node for k, v in zip(parent.keys, parent.values)):
            context = 'dict-unpack'
        else:
            context = ast.unparse(parent)
        require(context in allowed_reads, f'unreviewed use of {name}: ' + context)
    return True


def _abs(x):
    return z.If(x >= 0, x, -x)


def _query(premise, claim):
    """Positive certificate: premise satisfiable and premise-with-negated-claim unsatisfiable."""
    certify(premise, claim)
    return 1


def _source_expr(name):
    """Return the AST of a reviewed expression from the actual Replay source."""
    replay = _named(ast.parse((ROOT / ENV).read_text()), 'Replay')
    if name in ('gross', 'funding'):
        node = next(s for s in ast.walk(_method(replay, 'step')) if isinstance(s, ast.Assign)
                    and ast.unparse(s.targets[0]) == '(gross, funding)')
        return node.value.elts[0 if name == 'gross' else 1]
    terms = next(s for s in ast.walk(_method(replay, '_trade')) if isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == 'terms')
    return dict(zip((ast.literal_eval(k) for k in terms.value.keys), terms.value.values))[name]


def _factors(node):
    """Left-associated multiplication chain -> list of factor ASTs (fails closed on any other shape)."""
    out = []
    while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        out.insert(0, node.right)
        node = node.left
    out.insert(0, node)
    require(all(isinstance(f, (ast.Name, ast.Attribute, ast.Constant, ast.Call)) for f in out), 'unsupported factor')
    return out


def _direct(name):
    """Final inequality for gross/funding as one query on the AST-translated rounded expression."""
    u, h = z.RealVal(str(U)), z.RealVal(str(H))
    node, cons, count = _source_expr(name), [], [0]
    Uv, p0, p1, ft, m, Ev = z.Reals(f'ar_{name}_U ar_{name}_p0 ar_{name}_p1 ar_{name}_f ar_{name}_m ar_{name}_E')
    names = {'self.units': Uv, 'p0': p0, 'p1': p1, 'f': ft, 'self.execution.funding_multiplier': m}
    def tr(n):
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            c, x = tr(n.operand)
            return -c, -x                                                    # negation is exact
        if isinstance(n, ast.BinOp):
            (ca, xa), (cb, xb) = tr(n.left), tr(n.right)
            op = {ast.Mult: lambda a, b: a * b, ast.Sub: lambda a, b: a - b}[type(n.op)]
            count[0] += 1
            d, e = z.Reals(f'ar_d_{name}{count[0]} ar_e_{name}{count[0]}')
            cons.extend([d >= -u, d <= u, e >= -h, e <= h])
            return op(ca, cb) * (1 + d) + e, op(xa, xb)
        v = names[ast.unparse(n)]
        return v, v
    computed, exact = tr(node)
    require(count[0] == 2, 'reviewed operation count for ' + name)
    if name == 'gross':
        x0 = (dbl(0.35) + 3 * H) / ((1 - U) * (1 - U))
        premise = z.And(*cons, Ev >= z.RealVal('4/5'), Ev <= z.RealVal(str(HIGH)), p0 >= z.RealVal(str(LOW_PRICE)),
                        p1 >= p0 / 2, p1 <= 3 * p0 / 2, _abs(Uv) * p0 <= z.RealVal(str(x0)) * Ev)
        claim = _abs(computed - exact) <= 3 * u * _abs(exact) + z.RealVal(str(GROSS_E)) * Ev + 2 * h
    else:
        premise = z.And(*cons, m >= 0, m <= 2)
        claim = _abs(computed - exact) <= STATED['funding'][0] * u * _abs(exact) + STATED['funding'][1] * h
    return _query(premise, claim)


def _chain(name, bounds):
    """Per-call multiplicative chain from the AST, one certified step-lemma instance per multiplication.
    State (c, a): |computed - exact| <= c*|exact| + a*H."""
    factors = _factors(_source_expr(name))
    require(ast.unparse(factors[0]) == 'cash', 'chain starts from executed cash')
    u, h = z.RealVal(str(U)), z.RealVal(str(H))
    c, a, queries = Q(0), Q(0), 0
    for i, f in enumerate(factors[1:], start=1):
        label = ast.unparse(f)
        rel_in = U if label == 'np.sqrt(turnover)' else Q(0)                 # correctly rounded sqrt: relative only
        B = bounds[label]
        c_new = (1 + c) * (1 + rel_in) * (1 + U) - 1
        a_new = a * B * (1 + rel_in) * (1 + U) + 1
        C, X, F, dF, d, e = z.Reals(f'ar_c_{name}{i} ar_x_{name}{i} ar_f_{name}{i} ar_df_{name}{i} ar_d_{name}{i} ar_e_{name}{i}')
        premise = z.And(X >= 0, F >= 0, F <= z.RealVal(str(B)), d >= -u, d <= u, e >= -h, e <= h,
                        dF >= -z.RealVal(str(rel_in)), dF <= z.RealVal(str(rel_in)),
                        _abs(C - X) <= z.RealVal(str(c)) * X + z.RealVal(str(a)) * h)
        out = C * F * (1 + dF) * (1 + d) + e
        queries += _query(premise, _abs(out - X * F) <= z.RealVal(str(c_new)) * X * F + z.RealVal(str(a_new)) * h)
        c, a = c_new, a_new
    return c, a, len(factors) - 1, queries


def _merge(c, a, name):
    """Certified merge instance rnd(T_t + T_l) against X_t + X_l, bound to the AST merge comprehension."""
    step = _method(_named(ast.parse((ROOT / ENV).read_text()), 'Replay'), 'step')
    require(any(isinstance(s, ast.Assign) and ast.unparse(s) == 'costs = {k: costs[k] + liquidation[k] for k in costs}' for s in ast.walk(step)),
            'reviewed merge expression')
    u, h = z.RealVal(str(U)), z.RealVal(str(H))
    Tt, Tl, Xt, Xl, d, e = z.Reals(f'ar_mt_{name} ar_ml_{name} ar_mxt_{name} ar_mxl_{name} ar_md_{name} ar_me_{name}')
    c_new, a_new = c * (1 + U) + U, 2 * a * (1 + U) + 1
    premise = z.And(Xt >= 0, Xl >= 0, d >= -u, d <= u, e >= -h, e <= h,
                    _abs(Tt - Xt) <= z.RealVal(str(c)) * Xt + z.RealVal(str(a)) * h,
                    _abs(Tl - Xl) <= z.RealVal(str(c)) * Xl + z.RealVal(str(a)) * h)
    _query(premise, _abs((Tt + Tl) * (1 + d) + e - (Xt + Xl)) <= z.RealVal(str(c_new)) * (Xt + Xl) + z.RealVal(str(a_new)) * h)
    return c_new, a_new


def _roll_forward():
    """Two-stage certificate generated from the AST-bound ledger chain.
    Stage A (one query per rounding): operand magnitude |y_i| <= M + sum_{j<i} a_j given |e_j| <= a_j.
    Stage B (one query): with |e_i| <= a_i <= u*(M + sum_{j<i} a_j) + H, the final |E_c - L| <= 15u*M + 15H."""
    replay = _named(ast.parse((ROOT / ENV).read_text()), 'Replay')
    debit = next(s for s in ast.walk(_method(replay, '_trade')) if isinstance(s, ast.AugAssign) and ast.unparse(s.target) == 'self.equity')
    keys = ast.literal_eval(debit.value.args[0].generators[0].iter)
    require(keys == ('fee', 'spread', 'slippage', 'impact') and isinstance(debit.op, ast.Sub), 'reviewed debit sum')
    mark = next(s for s in ast.walk(_method(replay, 'step')) if isinstance(s, ast.AugAssign) and ast.unparse(s.target) == 'self.equity')
    require(isinstance(mark.op, ast.Add) and ast.unparse(mark.value) == 'gross + funding', 'reviewed mark update')
    u, h = z.RealVal(str(U)), z.RealVal(str(H))
    E, g, f = z.Reals('ar_rf_E ar_rf_g ar_rf_f')
    t = z.Reals(' '.join(f'ar_rf_t_{k}' for k in keys)); l = z.Reals(' '.join(f'ar_rf_l_{k}' for k in keys))
    ops = []                                       # (y_i, e_i, a_i) in evaluation order
    def rnd(y, tag):
        e, a = z.Reals(f'ar_rf_e_{tag} ar_rf_a_{tag}')
        ops.append((y, e, a))
        return y + e
    def debit_of(vals, tag):
        acc = vals[0]                              # sum(): 0 + first term is exact
        for i, v in enumerate(vals[1:], start=1):
            acc = rnd(acc + v, f'{tag}{i}')
        return acc
    Ea = rnd(E + rnd(g + f, 'gf'), 'Ea')           # self.equity += gross + funding
    Eb = rnd(Ea - debit_of(t, 'q'), 'Eb')          # trade debit
    Ec = rnd(Eb - debit_of(l, 'r'), 'Ec')          # liquidation debit
    rows = [rnd(t[i] + l[i], f'm{i}') for i in range(len(keys))]   # merge comprehension
    ledger = E + g + f - sum(rows)
    M = E + _abs(g) + _abs(f) + sum(t) + sum(l)                     # operand budget (unmerged)
    M_recorded = E + _abs(g) + _abs(f) + sum(rows)                   # the advertised M uses the recorded merged costs
    base = [E > 0, *[x >= 0 for x in t + l]]
    for i, (y, _, _) in enumerate(ops):           # Stage A
        prior = [z.And(a >= 0, e >= -a, e <= a) for _, e, a in ops[:i]]
        _query(z.And(*base, *prior), _abs(y) <= M + sum((a for _, _, a in ops[:i]), z.RealVal(0)))
    budget = []                                    # Stage B
    for i, (_, e, a) in enumerate(ops):
        budget.append(z.And(a >= 0, e >= -a, e <= a, a <= u * (M + sum((b for _, _, b in ops[:i]), z.RealVal(0))) + h))
    _query(z.And(*base, *budget), _abs(Ec - ledger) <= STATED['rollForward'][0] * u * M_recorded + STATED['rollForward'][1] * h)
    return len(ops), len(ops) + 1


def lemmas():
    queries = _direct('gross') + _direct('funding')
    term_bounds = {'0.0005': dbl(0.0005), '5e-05': dbl(5e-05), '0.00045': dbl(0.00045), 'cfg.cost_multiplier': Q(5, 2),
                   'cfg.impact_bps': Q(10), '0.0001': dbl(0.0001), 'np.sqrt(turnover)': Q(1)}
    summary = {}
    for name in ('fee', 'spread', 'slippage', 'impact'):
        c, a, ops, n = _chain(name, term_bounds)
        c, a = _merge(c, a, name)
        queries += n + 1
        rel, abs_ = STATED['impact' if name == 'impact' else 'costTerm']
        require(c <= rel * U and a <= abs_, 'stated bound below the AST-derived certified constant: ' + name)
        summary[name] = {'multiplications': ops, 'relative': float(c / U), 'absolute': float(a)}
    roundings, roll_queries = _roll_forward()
    require(roundings == ROUNDINGS, 'roll-forward rounding count drifted from the AST')
    return {'termQueries': queries, 'rollForwardRoundings': roundings, 'rollForwardQueries': roll_queries, 'terms': summary}


def _replay():
    sys.path.insert(0, str(ROOT / 'scripts/research'))
    from sequential_env import Execution, Replay, Scale
    return Execution, Replay, Scale


def _sqrt_interval(x, bits=200):
    """Rigorous rational enclosure [lo, hi] of sqrt(x) for a nonnegative rational x."""
    from math import isqrt
    scale = 2 ** bits
    n = isqrt(x.numerator * scale * scale // x.denominator)
    return Q(n, scale), Q(n + 1, scale)


def conformance():
    Execution, Replay, Scale = _replay()
    reg = json.loads((ROOT / REGISTRATION).read_text())
    rng = np.random.default_rng(reg['probeSeed'])
    rows = gross_checks = cost_checks = impact_checks = 0
    c = {'fee': dbl(0.0005), 'spread': dbl(5e-05), 'slippage': dbl(0.00045)}
    (term_rel, term_abs), (imp_rel, imp_abs) = STATED['costTerm'], STATED['impact']
    calls, trade_calls, impact_seen = [], 0, False
    original = Replay._trade
    def spy(self, target, terminal=False):
        # Pass-through instrumentation: record the executed inputs of each actual _trade call.
        equity, index = self.equity, len(self.rows)
        terms = original(self, target, terminal)
        calls.append((id(self), index, equity, terms))
        return terms
    Replay._trade = spy
    try:
        for k in range(reg['probeEpisodes']):
            stress = list(STRESSES)[k % len(STRESSES)]
            cfg = Execution(**STRESSES[stress])
            n = 140
            gaps = np.clip(rng.uniform(-0.5, 0.5, n) * (rng.random(n) < 0.1) + rng.normal(0, 0.01, n), -0.5, 0.5)
            p = 100 * np.cumprod(1 + gaps)
            f = np.r_[0.0, p[:-1] * rng.uniform(-0.03, 0.03, n - 1)]
            horizon = 1 if k % 2 == 0 else (3, 6)[k % 4 // 2]
            env = Replay(p, f, 60, 130, horizon, Scale.fit([p[:50]]), cfg, enabled=True)
            trade_calls += len(calls)
            calls.clear()   # object ids can be reused across episodes; scope the log to this episode
            previous = Q(env.equity)
            while not env.done:
                units, t0, seen, before = env.units, env.t, len(env.rows), previous
                env.step(float(rng.choice([-0.25, 0.0, 0.25])))
                for index in range(seen, len(env.rows)):
                    row = env.rows[index]
                    g, fu = Q(row['gross']), Q(row['funding'])
                    costs = [Q(row[key]) for key in ('fee', 'spread', 'slippage', 'impact')]
                    M = previous + abs(g) + abs(fu) + sum(costs)
                    ledger = previous + g + fu - sum(costs)
                    require(abs(Q(row['equity']) - ledger) <= STATED['rollForward'][0] * U * M + STATED['rollForward'][1] * H,
                            'roll-forward outside the stated bound')
                    executed = [(Q(e), terms) for owner, i, e, terms in calls if owner == id(env) and i == index]
                    for key, lit in c.items():
                        exact = sum((Q(t['turnover']) * lit * Q(cfg.cost_multiplier) for _, t in executed), Q(0))
                        require(abs(Q(row[key]) - exact) <= term_rel * U * exact + term_abs * H, 'cost term outside the stated bound: ' + key)
                        cost_checks += 1
                    lo = hi = Q(0)
                    for e, t in executed:
                        cash = Q(t['turnover'])
                        if cash:
                            s_lo, s_hi = _sqrt_interval(Q(float(t['turnover']) / float(e)))
                            factor = cash * Q(cfg.impact_bps) * dbl(0.0001)
                            lo, hi = lo + factor * s_lo, hi + factor * s_hi
                    require(Q(row['impact']) >= lo * (1 - imp_rel * U) - imp_abs * H and Q(row['impact']) <= hi * (1 + imp_rel * U) + imp_abs * H,
                            'impact outside the stated bound')
                    impact_checks += 1
                    impact_seen = impact_seen or Q(row['impact']) > 0
                    previous = Q(row['equity']); rows += 1
                if horizon == 1 and len(env.rows) > seen:
                    row = env.rows[seen]
                    exact_g = Q(units) * (Q(p[t0 + 1]) - Q(p[t0]))
                    exact_f = -Q(units) * Q(f[t0 + 1]) * Q(cfg.funding_multiplier)
                    require(abs(Q(row['gross']) - exact_g) <= 3 * U * abs(exact_g) + GROSS_E * before + 2 * H, 'gross outside stated bound')
                    require(abs(Q(row['funding']) - exact_f) <= STATED['funding'][0] * U * abs(exact_f) + STATED['funding'][1] * H,
                            'funding outside stated bound')
                    gross_checks += 1
    finally:
        Replay._trade = original
    trade_calls += len(calls)
    require(rows > 1000 and gross_checks > 100 and impact_seen, 'vacuous probe')
    return {'status': 'property_tested', 'episodes': reg['probeEpisodes'], 'rows': rows, 'grossFundingChecks': gross_checks,
            'costTermChecks': cost_checks, 'impactChecks': impact_checks, 'tradeCalls': trade_calls, 'seed': reg['probeSeed']}


def check_reconciliation():
    reg = json.loads((ROOT / REGISTRATION).read_text())
    require(reg['sourceChanges'] == 0 and reg['holdoutOpened'] is False, 'registration')
    certified = lemmas()
    return {'source': bind(), 'smt': {'F-RL-RECON-ERROR': 'unsat'}, 'certified': certified, 'conformance': conformance()}

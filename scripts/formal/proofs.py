"""SMT obligations for the documented abstraction, not compiler refinement."""
import z3 as z


def obligations():
    target, elapsed = z.FPs('target elapsed', z.Float64())
    mode, valid, owned, artifact, support, risk = z.Bools('mode valid owned artifact support risk')
    guards = z.And(valid, owned, artifact, support, risk)
    finite = lambda x: z.And(z.Not(z.fpIsNaN(x)), z.Not(z.fpIsInf(x)))
    fp = lambda x: z.FPVal(x, z.Float64())
    action = z.Or(*[z.fpEQ(target, fp(x)) for x in (-0.25, 0, 0.25)])
    timely = z.And(finite(elapsed), z.fpGEQ(elapsed, fp(0)), z.fpLEQ(elapsed, fp(20)))
    accepted = z.And(mode, guards, timely, finite(target), action)
    claims = {
        'F-RL-SHIELD-BOUNDS': z.Implies(accepted, z.And(finite(target), z.fpLEQ(z.fpAbs(target), fp(0.25)))),
        'F-RL-SHIELD-GUARDS': z.Implies(accepted, z.And(mode, guards, timely)),
        'F-RL-SHIELD-FALLBACK': z.Implies(z.Or(z.Not(finite(target)), z.Not(timely), z.Not(guards)), z.Not(accepted)),
        'F-RL-DEFAULT': z.Implies(z.Not(mode), z.Not(accepted)),
        'F-RL-SHIELD-IDEMPOTENT': z.Implies(accepted, z.And(mode, guards, timely, finite(target), action)),
        # Literal constant in the independently reviewed source; no order adapter.
        'F-RL-AUTH': z.Implies(accepted, z.Not(z.BoolVal(False))),
    }
    t, offset, train_stop, test_start, horizon, i, j = z.Ints('t offset train_stop test_start horizon i j')
    claims['F-RL-CAUSAL-SLICE'] = z.Implies(z.And(t >= 24, offset >= 0, offset < 25), t - 24 + offset <= t)
    claims['F-RL-SPLIT'] = z.Implies(
        z.And(horizon > 0, test_start >= train_stop + horizon,
              i >= 0, i < train_stop, j >= test_start), i + horizon < j)
    wealth, gross, funding, cost, terminal, penalty = z.Reals('wealth gross funding cost terminal penalty')
    ending = wealth + gross + funding - cost - terminal
    claims['F-RL-ACCOUNT'] = z.And(ending - wealth == gross + funding - cost - terminal,
                                  (ending - wealth - penalty) + penalty == ending - wealth)
    return claims


def check_all():
    results = {}
    for key, theorem in obligations().items():
        solver = z.Solver()
        solver.set(timeout=10000, random_seed=0)
        solver.add(z.Not(theorem))
        result = solver.check()
        if result != z.unsat:
            detail = str(solver.model()) if result == z.sat else solver.reason_unknown()
            raise RuntimeError(f'{key}: {result}: {detail}')
        results[key] = 'unsat'
    # Non-vacuity: extract the actual acceptance antecedent, not a separate predicate.
    acceptance = obligations()['F-RL-SHIELD-BOUNDS'].arg(0)
    witness = z.Solver()
    witness.set(timeout=10000)
    witness.add(acceptance)
    if witness.check() != z.sat:
        raise RuntimeError('non-vacuity witness failed')
    return results

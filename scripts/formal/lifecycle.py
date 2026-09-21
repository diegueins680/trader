"""Exhaustive explicit-state checking of a two-caller abstract protocol.

No production lifecycle/refinement claim. A returned proposal has no capability.
No truncation: explore until the finite reachable-state fixed point.
"""
from collections import deque

INITIAL = (False, False, 0, 0)
# Worker: idle=0, admissible in flight=1, rejected in flight=2,
# non-authorizing proposal returned=3, absent result returned=4.


def successors(s):
    enabled, stopping, *workers = s
    yield 'disable', (False, stopping, *workers)
    yield 'stop', (False, True, *workers)
    if not stopping:
        yield 'enable_offline', (True, False, *workers)
    if stopping and not any(w in (1, 2) for w in workers):
        yield 'reset_quiescent', INITIAL
    for n, w in enumerate(workers):
        if enabled and not stopping and w == 0:
            for value in (1, 2):
                changed = workers.copy()
                changed[n] = value
                yield f'start{n}:{value}', (enabled, stopping, *changed)
        if w in (1, 2):
            changed = workers.copy()
            changed[n] = 3 if w == 1 else 4
            yield f'finish{n}', (enabled, stopping, *changed)
        if w in (3, 4):
            changed = workers.copy()
            changed[n] = 0
            yield f'consume{n}', (enabled, stopping, *changed)


def authority(_state):
    return False


def check_model():
    queue = deque([INITIAL])
    depths = {INITIAL: 0}
    parents = {}
    edges = 0
    stale_trace = None
    while queue:
        state = queue.popleft()
        if authority(state) or (state[1] and state[0]):
            raise RuntimeError(f'unsafe state: {state}')
        transitions = list(successors(state))
        if not transitions:
            raise RuntimeError('deadlock')
        for event, nxt in transitions:
            edges += 1
            if state[1] and event.startswith('start'):
                raise RuntimeError('start while draining')
            if event.startswith('finish'):
                n = int(event[-1]) + 2
                if nxt[n] == 3 and state[n] != 1:
                    raise RuntimeError('unshielded proposal')
            if event == 'disable' and tuple(nxt[2:]) != tuple(state[2:]):
                raise RuntimeError('disable falsely erases caller-owned immutable values')
            if nxt not in depths:
                depths[nxt] = depths[state] + 1
                parents[nxt] = (state, event)
                queue.append(nxt)
        # Conditional liveness: finishing one call strictly decreases this rank.
        if state[1]:
            rank = sum(w in (1, 2) for w in state[2:])
            if rank > 0 and not any(event.startswith('finish') for event, _ in transitions):
                raise RuntimeError('draining has no completion transition')
            for event, nxt in transitions:
                if event.startswith('finish') and sum(w in (1, 2) for w in nxt[2:]) != rank - 1:
                    raise RuntimeError('drain rank failed')
        if stale_trace is None and not state[0] and 3 in state[2:]:
            trace, cursor = [], state
            while cursor != INITIAL:
                cursor, event = parents[cursor]
                trace.append(event)
            stale_trace = list(reversed(trace))
    if stale_trace is None:
        raise RuntimeError('expected stale-proposal counterexample was not found')
    return {'states': len(depths), 'transitions': edges, 'maxShortestDepth': max(depths.values()),
            'callers': 2, 'drainCompletionBound': 2, 'search': 'reachable_fixed_point',
            'counterexampleToRevocation': stale_trace}

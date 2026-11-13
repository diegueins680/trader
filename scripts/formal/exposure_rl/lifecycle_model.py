"""Explicit-state model of the exposure-path decision lifecycle.

Two concurrent proposers share one risk budget. The model is explored breadth
first to a fixed point; every reachable state is checked. It is a bounded
abstraction (two proposers, one decision each with bounded retries, three
exposure levels), not the Haskell or Python implementation.

Checked properties:
  * budget:        reserved exposure never exceeds the shared budget
  * shield-first:  an intent is submitted only after a shield admission and a
                   granted, still-fresh reservation
  * no-live:       no state holds a live-order capability; the model has no
                   transition that creates one
  * no-self-promo: only the operator actor changes the mode; the challenger
                   cannot enable itself, replace the champion or promote
  * disabled:      while the mode is disabled no proposer leaves idle and the
                   champion's decision is never changed (noninterference)
  * fail-closed:   timeout, corrupt artifact and stale state reach the hold
                   fallback, never a submission
  * exactly-once:  retries never submit one decision twice
  * progress:      every reachable state can reach a quiescent state

A mutant with a non-atomic check-then-act reservation must violate ``budget``;
finding that counterexample shows the check is not vacuous.
"""

from __future__ import annotations

from collections import deque

BUDGET = 2
LEVELS = (0, 1, 2)  # abstract |exposure| requests
RETRIES = 1
IDLE, OBSERVED, ADMITTED, RESERVED, SUBMITTED, HOLD = range(6)
MODES = ("disabled", "research_replay")


def initial():
    # (mode, version, held, proposers, champion_changed, live_capability)
    proposer = (IDLE, 0, 0, 0, 0)  # (pc, seen_version, wanted, retries_used, submissions)
    return ("disabled", 0, (0, 0), (proposer, proposer), False, False)


def successors(state, mutant: bool = False):
    mode, version, held, proposers, champion_changed, live = state
    # Operator: the only actor that may change the mode. No transition reaches a live mode.
    for target in MODES:
        if target != mode:
            yield ("operator:set_" + target, (target, version, held, proposers, champion_changed, live))
    for i, (pc, seen, wanted, retries, submissions) in enumerate(proposers):
        def put(new, new_version=version, new_held=held):
            updated = proposers[:i] + (new,) + proposers[i + 1 :]
            return (mode, new_version, new_held, updated, champion_changed, live)

        if mode == "disabled":
            continue  # a disabled challenger takes no step at all
        if pc == IDLE and submissions == 0:
            yield (f"p{i}:observe", put((OBSERVED, version, 0, retries, submissions)))
        elif pc == OBSERVED:
            for level in LEVELS:  # the untrusted proposal is arbitrary; the shield bounds it
                yield (f"p{i}:shield_admit_{level}", put((ADMITTED, seen, level, retries, submissions)))
            for failure in ("timeout", "corrupt_artifact", "invalid_proposal", "stale_state"):
                yield (f"p{i}:{failure}", put((HOLD, seen, 0, retries, submissions)))
        elif pc == ADMITTED:
            total = held[1 - i] + wanted
            if mutant:
                # Non-atomic: budget checked against the proposer's old read, no version check.
                if total <= BUDGET or wanted <= held[i]:
                    yield (f"p{i}:reserve_unsafe", put((RESERVED, seen, wanted, retries, submissions)))
                continue
            if seen != version:
                if retries < RETRIES:
                    yield (f"p{i}:stale_retry", put((OBSERVED, version, 0, retries + 1, submissions)))
                else:
                    yield (f"p{i}:stale_hold", put((HOLD, seen, 0, retries, submissions)))
            elif total > BUDGET and wanted > held[i]:
                yield (f"p{i}:reserve_refused", put((HOLD, seen, 0, retries, submissions)))
            else:
                new_held = tuple(wanted if j == i else h for j, h in enumerate(held))
                yield (f"p{i}:reserve_granted", put((RESERVED, version + 1, wanted, retries, submissions), version + 1, new_held))
        elif pc == RESERVED:
            if mutant:
                new_held = tuple(wanted if j == i else h for j, h in enumerate(held))
                yield (f"p{i}:submit_unsafe", put((SUBMITTED, seen, wanted, retries, submissions + 1), version + 1, new_held))
            else:
                yield (f"p{i}:submit_simulated", put((SUBMITTED, seen, wanted, retries, submissions + 1)))
        # SUBMITTED and HOLD are terminal for this decision.


def quiescent(state) -> bool:
    return all(p[0] in (IDLE, SUBMITTED, HOLD) for p in state[3])


def violations(state, previous=None, event: str = ""):
    mode, version, held, proposers, champion_changed, live = state
    if sum(held) > BUDGET:
        yield "budget"
    if live:
        yield "no-live"
    if champion_changed:
        yield "no-self-promo"
    if any(p[4] > 1 for p in proposers):
        yield "exactly-once"
    if previous is not None:
        if previous[0] != mode and not event.startswith("operator:"):
            yield "no-self-promo"
        if previous[0] == "disabled" and not event.startswith("operator:"):
            yield "disabled"
        for before, after in zip(previous[3], proposers):
            if after[0] == SUBMITTED and before[0] != SUBMITTED and before[0] != RESERVED:
                yield "shield-first"
            if after[0] == SUBMITTED and before[0] == HOLD:
                yield "fail-closed"


def explore(mutant: bool = False):
    start = initial()
    parents = {start: None}
    queue = deque([start])
    transitions = 0
    edges = {}
    while queue:
        state = queue.popleft()
        outgoing = list(successors(state, mutant))
        edges[state] = [nxt for _, nxt in outgoing]
        for event, nxt in outgoing:
            transitions += 1
            bad = sorted(set(violations(nxt, state, event)))
            if nxt not in parents:
                parents[nxt] = (state, event)
                queue.append(nxt)
            if bad:
                trace, cursor = [], nxt
                while parents[cursor] is not None:
                    cursor, label = parents[cursor][0], parents[cursor][1]
                    trace.append(label)
                return {"violation": bad[0], "trace": list(reversed(trace)), "states": len(parents), "transitions": transitions}
    # Progress: from every reachable state some quiescent state is reachable.
    good = {s for s in parents if quiescent(s)}
    changed = True
    while changed:
        changed = False
        for state, targets in edges.items():
            if state not in good and any(t in good for t in targets):
                good.add(state)
                changed = True
    if len(good) != len(parents):
        return {"violation": "progress", "trace": [], "states": len(parents), "transitions": transitions}
    submitted = sum(1 for s in parents if any(p[0] == SUBMITTED for p in s[3]))
    held_out = sum(1 for s in parents if any(p[0] == HOLD for p in s[3]))
    return {"violation": None, "states": len(parents), "transitions": transitions, "statesWithSubmission": submitted, "statesWithHold": held_out}


def check() -> dict:
    result = explore()
    if result["violation"] is not None:
        raise ValueError(f"exposure lifecycle model: {result['violation']} violated: {result['trace']}")
    if not (result["states"] > 0 and result["statesWithSubmission"] > 0 and result["statesWithHold"] > 0):
        raise ValueError("exposure lifecycle model: vacuous exploration")
    mutant = explore(mutant=True)
    if mutant["violation"] != "budget":
        raise ValueError("exposure lifecycle model: non-atomic mutant was not caught")
    return {"model": {k: result[k] for k in ("states", "transitions", "statesWithSubmission", "statesWithHold")}, "mutantCounterexample": {"violation": mutant["violation"], "trace": mutant["trace"]}, "bounds": {"proposers": 2, "budget": BUDGET, "levels": len(LEVELS), "retries": RETRIES}}


if __name__ == "__main__":
    print(check())

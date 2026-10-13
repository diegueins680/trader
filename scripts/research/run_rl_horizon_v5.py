"""Opt-in finite-objective horizon sensitivity on registered development exports."""
import argparse
import json
from pathlib import Path
import rl_admission_v2 as v2
from run_rl_budget_v3 import RULES, COMPARATORS, baseline
from run_rl_composition import ROOT, require, run

REG = ROOT/'research-notes/registrations/rl-matched-horizon-v5.json'


def train(algorithm, factory, seed, steps, frozen, *, enabled=False):
    def forbidden(_):
        raise ValueError('offline training attempted simulator interaction')
    offline = algorithm in ('cql', 'cql_alpha0', 'behavior_cloning')
    require(not offline or frozen is not None, 'offline dataset required')
    return v2.train(algorithm, forbidden if offline else factory, seed, steps,
                    frozen, enabled=enabled)


def configuration(reg):
    require(reg['id'] == 'rl-matched-horizon-v5' and reg['steps'] == 4096 and
            reg['offlineTransitions'] == 512 and
            reg['environment']['trainingEpisodeTransitions'] == 64 and
            reg['environment']['version'] == v2.VERSION and
            reg['baselines'] == [*RULES, 'behavior_cloning'] and
            reg['seeds'] == [11, 23, 47, 71, 101] and
            reg['promotionAllowed'] is False and reg['enabledByDefault'] is False and
            reg['liveAuthorization'] is False, 'horizon registration drift')
    return dict(registration=REG, episode_type=v2.Episode, trainer=train,
        version=v2.VERSION, observation=v2.OBSERVATION, rules=RULES,
        rule_policy=baseline, comparators=COMPARATORS, offline_transitions=512,
        training_transitions=64,
        extra_sources=('rl_admission_v2.py', 'run_rl_budget_v3.py', 'run_rl_horizon_v5.py',
                       'compare_rl_observations_v2.py', 'summarize_rl_composition.py'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-registered-horizon-v5', action='store_true')
    for name in ('panel', 'settlements', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if not args.run_registered_horizon_v5:
        parser.error('disabled: explicit registered horizon opt-in required')
    run(args.panel, args.settlements, args.output,
        **configuration(json.loads(REG.read_bytes())))

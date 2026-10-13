"""Opt-in frozen simulated dataset sensitivity; no online learning interactions."""
import argparse
import json
from pathlib import Path
import rl_admission_v2 as v2
from run_rl_budget_v3 import RULES, COMPARATORS, baseline
from run_rl_composition import ROOT, require, run

REG = ROOT/'research-notes/registrations/rl-offline-data-v4.json'
ALGORITHMS = ('cql', 'cql_alpha0', 'behavior_cloning')


def offline_train(algorithm, factory, seed, steps, frozen, *, enabled=False):
    require(algorithm in ALGORITHMS and frozen is not None, 'offline learner required')
    def forbidden(_):
        raise ValueError('offline training attempted simulator interaction')
    return v2.train(algorithm, forbidden, seed, steps, frozen, enabled=enabled)


def configuration(reg):
    require(reg['id'] == 'rl-offline-data-v4' and reg['steps'] == 4096 and
            reg['offlineTransitions'] == 4096 and reg['parentArchive']['prefixTransitions'] == 512 and
            reg['environment']['version'] == v2.VERSION and
            reg['baselines'] == [*RULES, 'behavior_cloning'] and
            reg['seeds'] == [11, 23, 47, 71, 101] and
            reg['promotionAllowed'] is False and reg['enabledByDefault'] is False and
            reg['liveAuthorization'] is False, 'offline data registration drift')
    return dict(registration=REG, episode_type=v2.Episode, trainer=offline_train,
        version=v2.VERSION, observation=v2.OBSERVATION, rules=RULES,
        rule_policy=baseline, comparators=COMPARATORS, algorithms=ALGORITHMS,
        paired_algorithms=('cql',), offline_transitions=reg['offlineTransitions'],
        behavior_identity_prefix=reg['parentArchive']['prefixTransitions'],
        expected_behavior_hashes=tuple(reg['parentArchive']['behaviorHashes']),
        extra_sources=('rl_admission_v2.py', 'run_rl_budget_v3.py', 'run_rl_offline_data_v4.py',
                       'compare_rl_observations_v2.py', 'summarize_rl_composition.py'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-registered-offline-data-v4', action='store_true')
    for name in ('panel', 'settlements', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if not args.run_registered_offline_data_v4:
        parser.error('disabled: explicit registered offline data opt-in required')
    run(args.panel, args.settlements, args.output,
        **configuration(json.loads(REG.read_bytes())))

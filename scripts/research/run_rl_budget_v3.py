"""Opt-in registered budget sensitivity; unchanged v2 actions and accounting."""
import argparse
import json
from pathlib import Path
import rl_admission_v2 as v2
from compare_rl_observations_v2 import observation_rule
from run_rl_composition import ROOT, require, run

REG = ROOT/'research-notes/registrations/rl-training-budget-v3.json'
RULES = ('cash', 'constant_long', 'normalized_momentum3', 'normalized_reversal3')
COMPARATORS = ('cash', 'normalized_momentum3')


def baseline(name, episode):
    if name == 'cash':
        return 1
    if name == 'constant_long':
        return 2
    return observation_rule(name, episode.observation())


def configuration(reg):
    """Fail before reading market data if the registered experiment drifts."""
    require(reg['id'] == 'rl-training-budget-v3' and reg['steps'] == 4096 and
            reg['offlineTransitions'] == 512 and
            reg['environment']['version'] == v2.VERSION and
            reg['baselines'] == [*RULES, 'behavior_cloning'] and
            reg['seeds'] == [11, 23, 47, 71, 101] and
            reg['promotionAllowed'] is False and reg['enabledByDefault'] is False and
            reg['liveAuthorization'] is False, 'budget registration drift')
    return dict(registration=REG, episode_type=v2.Episode, trainer=v2.train,
        version=v2.VERSION, observation=v2.OBSERVATION, rules=RULES,
        rule_policy=baseline, comparators=COMPARATORS,
        offline_transitions=reg['offlineTransitions'],
        expected_behavior_hashes=tuple(reg['parentArchive']['behaviorHashes']),
        extra_sources=('rl_admission_v2.py', 'run_rl_budget_v3.py',
                       'compare_rl_observations_v2.py', 'summarize_rl_composition.py'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-registered-budget-v3', action='store_true')
    for name in ('panel', 'settlements', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if not args.run_registered_budget_v3:
        parser.error('disabled: explicit registered budget opt-in required')
    run(args.panel, args.settlements, args.output,
        **configuration(json.loads(REG.read_bytes())))

"""Opt-in PPO reward ablation. The economic ledger and shield are unchanged."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import rl_admission_v2 as v2
from run_rl_budget_v3 import RULES, COMPARATORS, baseline
from run_rl_composition import ROOT, require, run

REG = ROOT/'research-notes/registrations/rl-proposal-penalty-v6.json'
REWARD_VERSION = 'equity-minus-proposal-intervention-v6'
PENALTY = .02


class TrainingEpisode:
    """Only the reward returned to PPO changes; keep a separate training trace."""
    def __init__(self, episode, rows, number):
        require(type(episode) is v2.Episode, 'unexpected training environment')
        self.episode, self.rows, self.number = episode, rows, number

    @property
    def done(self):
        return self.episode.done

    @property
    def failure(self):
        return self.episode.failure

    def observation(self):
        return self.episode.observation()

    def step(self, action):
        observation, economic, done = self.episode.step(action)
        row = self.episode.rows[-1]
        require(type(row['actionChanged']) is bool, 'invalid intervention indicator')
        penalty = PENALTY if row['actionChanged'] else 0.
        shaped = economic - penalty
        require(math.isfinite(shaped), 'nonfinite training reward')
        self.rows.append({**row, 'trainingEpisode':self.number,
            'trainingSymbol':self.episode.bars[self.episode.start].symbol,
            'trainingStart':self.episode.start, 'trainingStop':self.episode.stop,
            'trainingReward':shaped, 'trainingPenalty':penalty})
        return observation, shaped, done


def train(algorithm, factory, seed, steps, frozen, *, enabled=False):
    require(enabled is True and algorithm == 'ppo', 'disabled or non-PPO reward ablation')
    rows, episodes = [], 0
    def wrapped(rng):
        nonlocal episodes
        episodes += 1
        return TrainingEpisode(factory(rng), rows, episodes)
    fit = v2.train(algorithm, wrapped, seed, steps, frozen, enabled=True)
    require(len(rows) == steps == fit.metadata['collectedTransitions'], 'training trace incomplete')
    economic = math.fsum(r['reward'] for r in rows)
    shaped = math.fsum(r['trainingReward'] for r in rows)
    changed = sum(r['actionChanged'] for r in rows)
    require(math.isclose(shaped, economic-PENALTY*changed, abs_tol=1e-9, rel_tol=1e-12),
            'training reward reconciliation failed')
    return replace(fit, metadata={**fit.metadata,
        'trainingRewardVersion':REWARD_VERSION,
        'trainingPenaltyPerChangedProposal':PENALTY,
        'trainingRewardAudit':{'transitions':len(rows), 'episodes':episodes,
            'changedProposals':changed, 'economicRewardSum':economic,
            'trainingRewardSum':shaped, 'penaltySum':PENALTY*changed,
            'finalEpisodeTerminal':rows[-1]['terminal']},
        'trainingRewardRows':rows})


def configuration(reg):
    require(reg['id'] == 'rl-proposal-penalty-v6' and reg['steps'] == 4096 and
        reg['offlineTransitions'] == 512 and reg['environment']['trainingEpisodeTransitions'] == 64 and
        reg['environment']['trainingRewardVersion'] == REWARD_VERSION and
        reg['environment']['trainingPenaltyPerChangedProposal'] == PENALTY and
        reg['environment']['version'] == v2.VERSION and
        reg['seeds'] == [11,23,47,71,101] and reg['promotionAllowed'] is False and
        reg['enabledByDefault'] is False and reg['liveAuthorization'] is False, 'penalty registration drift')
    return dict(registration=REG, episode_type=v2.Episode, trainer=train,
        version=v2.VERSION, observation=v2.OBSERVATION, rules=RULES, rule_policy=baseline,
        comparators=COMPARATORS, algorithms=('ppo',), paired_algorithms=('ppo',),
        offline_transitions=512, training_transitions=64,
        expected_behavior_hashes=tuple(reg['parentArchive']['behaviorHashes']),
        extra_sources=('rl_admission_v2.py','run_rl_budget_v3.py','run_rl_penalty_v6.py',
                       'compare_rl_observations_v2.py','summarize_rl_composition.py'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-registered-penalty-v6', action='store_true')
    for name in ('panel','settlements','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if not args.run_registered_penalty_v6:
        parser.error('disabled: explicit registered penalty opt-in required')
    run(args.panel,args.settlements,args.output,**configuration(json.loads(REG.read_bytes())))

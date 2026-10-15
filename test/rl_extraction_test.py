"""Decoder boundary, reference, artifact and simulated-state conformance."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/research'))
import run_rl_extraction_v7 as v7
from optimizer_snapshot_v2 import create_v2


class ExtractionTests(unittest.TestCase):
    def test_half_open_boundaries_and_zero_mass(self):
        for p,u,want in (([.25,.5,.25],0.,0),([.25,.5,.25],.25,1),
                         ([.25,.5,.25],.75,2),([0,1,0],0.,1),
                         ([0,0,1],0.,2),([1,0,0],np.nextafter(1.,0.).item(),0)):
            self.assertEqual(v7.categorical(p,u)[0],want)

    def test_invalid_values_fail_closed(self):
        for p in ([1,0],[1,0,0,0],[[1,0,0]],[.5,.5,.1],[-.1,.6,.5],[float('nan'),0,1],[float('inf'),0,0]):
            with self.assertRaises(ValueError): v7.categorical(p,.3)
        for u in (-.1,1.,float('nan'),float('inf'),True,0):
            with self.assertRaises(ValueError): v7.categorical([.3,.3,.4],u)

    def test_reference_choice_and_repeatability(self):
        p=np.array([.125,.375,.5]); rng=np.random.default_rng(19)
        uniforms=rng.random(4096)
        observed=[v7.categorical(p,float(u))[0] for u in uniforms]
        reference=np.random.default_rng(19).choice(3,size=4096,p=p).tolist()
        self.assertEqual(observed,reference)
        # Deterministic quadrature has known exact category masses, not a flaky statistical test.
        grid=[v7.categorical(p,(i+.5)/800)[0] for i in range(800)]
        self.assertEqual(np.bincount(grid).tolist(),[100,300,400])

    def fixture(self):
        actor=create_v2(7,3,enabled=True).snapshot()
        value=dict(version=v7.VERSION,observation=v7.OBSERVATION,mode='research_only',algorithm='ppo',
                   trainingEpisodeTransitions=64,fold=0,seed=11,updates=128,sourceCommit='commit',
                   registrationSha256='reg',assuranceLedgerSha256='proof',
                   parametersLittleEndianHex=[p.hex() for p in actor.p])
        def archive(value):
            raw=json.dumps(value).encode()
            return {'policies/0-ppo-11.json':raw,'manifest.json':json.dumps(dict(
                campaign='rl-matched-horizon-v5',promotionAllowed=False,holdoutOpened=False,
                commit='commit',registrationSha256='reg',proofLedgerSha256='proof')).encode(),
                'summary.json':json.dumps(dict(fits=[dict(id='0/ppo/11/training',status='complete',artifactSha256=v7.sha(raw))])).encode()}
        return actor,value,archive

    def test_artifact_roundtrip_and_strict_contract(self):
        actor,value,archive=self.fixture()
        fit,_=v7.admit(archive(value),'v5',0,11)
        x=np.arange(12,dtype=float)[None,:]/12
        np.testing.assert_array_equal(v7.forward(actor,x),v7.forward(fit.actor,x))
        for key,bad in (('mode','live'),('seed',True),('trainingEpisodeTransitions',64.),
                        ('observation','other'),('updates',0),('trainingRewardVersion','shaped')):
            with self.assertRaises(ValueError):v7.admit(archive({**value,key:bad}),'v5',0,11)
        for params in (value['parametersLittleEndianHex'][:3],['00']*4,
                       [*value['parametersLittleEndianHex'][:3],np.array([np.nan,0,0],dtype='<f8').tobytes().hex()]):
            with self.assertRaises(ValueError):v7.admit(archive({**value,'parametersLittleEndianHex':params}),'v5',0,11)

    def test_hash_binding_and_complete_roster(self):
        _,value,archive=self.fixture(); data=archive(value)
        data['policies/0-ppo-11.json']=json.dumps({**value,'seed':23}).encode()
        with self.assertRaises(ValueError):v7.admit(data,'v5',0,11)
        reg=v7.decode(v7.REG.read_bytes()); rows=v7.roster(reg)
        self.assertEqual(sum(r['algorithm'].endswith('_sampled') for r in rows),2100)
        self.assertEqual(sum(r['algorithm'].endswith('_greedy') for r in rows),420)
        with self.assertRaises(ValueError):v7.roster({**reg,'actionSeeds':[19,19,59,83,109]})

    def test_seeded_composed_replay_matches_reference_sampler(self):
        from fractions import Fraction as F
        from causal_replay_v3 import Bar
        bars=tuple(Bar('FIXTURE',i*1000,i*1000+1,F(10000+(i%7)*10,100)) for i in range(200))
        funding=tuple(() for _ in bars)
        _,value,archive=self.fixture();fit,_=v7.admit(archive(value),'v5',0,11)
        for stress in ('base','delay1bar','partial50pct'):
            episodes=[v7.Episode(bars,funding,96,161,stress,enabled=True) for _ in range(2)]
            rngs=[np.random.default_rng(19),np.random.default_rng(19)]
            for index,ep in enumerate(episodes):
                while not ep.done:
                    p=v7.softmax(v7.forward(fit.actor,ep.observation()[None,:]))[0]
                    a=v7.categorical(p,float(rngs[index].random()))[0] if index==0 else int(rngs[index].choice(3,p=p))
                    ep.step(a)
                self.assertEqual(len(ep.rows),64)
                self.assertEqual(v7.audit_rows(ep.rows)['violations'],[])
            v7.same(episodes[0].rows,episodes[1].rows,'sampled composed replay')

    def test_proposal_audit_detects_probability_action_mismatch(self):
        row=dict(proposalProbabilities=[.25,.5,.25],uniform=.25,actionIndex=1,
                 proposal=0.,proposalLogProbability=float(np.log(.5)))
        with patch.object(v7,'audit_rows',return_value={'violations':[]}):
            v7.checked_path([row],True)
            for key,bad in (('actionIndex',2),('proposal',.25),('proposalLogProbability',0.)):
                with self.assertRaises(ValueError):v7.checked_path([{**row,key:bad}],True)


if __name__=='__main__':unittest.main()

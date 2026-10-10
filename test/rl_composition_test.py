import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/research'))
import unittest
from fractions import Fraction as F
import numpy as np
import rl_composition as r
from causal_replay_v3 import Bar
from gae_targets_v2 import batch_v2
from sequential_learning import ppo_gradient
from value_objective_v2 import objective_v2


def fixture():
    bars=tuple(Bar('FIXTURE',i*1000,i*1000+1,F(10000+(i%7)*10,100)) for i in range(200))
    return bars,tuple(() for _ in bars)


class CompositionTests(unittest.TestCase):
    def test_disabled_invalid_and_future(self):
        bars,events=fixture()
        with self.assertRaises(ValueError): r.Episode(bars,events,96,129)
        ep=r.Episode(bars,events,96,129,enabled=True)
        other=r.Episode(bars[:97]+tuple(Bar('X',-1,-1,F(-1)) for _ in bars[97:]),events,96,129,enabled=True)
        np.testing.assert_array_equal(ep.observation(),other.observation())
        state=ep.session
        for action in (None,True,3,-1,float('nan'),[0]):
            with self.assertRaises(ValueError): ep.step(action)
            self.assertEqual(ep.session,state)

    def test_accounting_terminal_and_projection(self):
        bars,events=fixture()
        for stress in r.STRESSES:
            ep=r.Episode(bars,events,96,129,stress,enabled=True)
            total=0.
            while not ep.done:
                _,reward,_=ep.step(2); total+=reward
            self.assertAlmostEqual(total,100*float(ep.session.state.equity-1),places=11)
            self.assertEqual(ep.session.state.units,0)
            self.assertIsNone(ep.pending)
            with self.assertRaises(ValueError): ep.step(1)
            self.assertTrue(all(x['decisionTime']<x['fillClose'] for x in ep.rows))

    def test_pending_does_not_free_rebalance(self):
        bars,events=fixture()
        ep=r.Episode(bars,events,96,129,'delay1bar',enabled=True)
        ep.step(2)
        self.assertEqual(ep.session.state.units,0)
        self.assertIsNotNone(ep.pending)
        ep.step(0)
        self.assertIsNone(ep.pending)
        prior=ep.session.state.units
        ep.step(1)
        self.assertEqual(ep.session.state.units,prior)
        self.assertEqual(ep.rows[-1]['turnover'],0)

    def test_collector_rollout_cut_bootstraps_without_reset(self):
        bars,events=fixture(); calls=[]
        def factory(_):
            calls.append(1); return r.Episode(bars,events,96,129,enabled=True)
        collector=r.Collector(factory,11)
        a=collector.collect(lambda _:np.array([0.,1.,0.]),5)
        b=collector.collect(lambda _:np.array([0.,1.,0.]),5)
        self.assertEqual(len(calls),1)
        self.assertFalse(a['done'][-1])
        np.testing.assert_array_equal(a['next'][-1],b['s'][0])
        self.assertEqual(batch_v2(((1.,2.,3.,False),),1.,enabled=True),((2.,4.),))
        self.assertEqual(batch_v2(((1.,2.,999.,True),),1.,enabled=True),((-1.,1.),))

    def test_ppo_gradient_finite_difference(self):
        logits=np.array([[.2,.1,-.3],[.4,-.4,.3]])
        actions=np.array([0,2]); old=np.array([.4,.3]); advantages=np.array([1.,-.7])
        _,gradient=ppo_gradient(logits,actions,old,advantages)
        for i in range(2):
            for j in range(3):
                plus,minus=logits.copy(),logits.copy(); plus[i,j]+=1e-6;minus[i,j]-=1e-6
                derivative=(ppo_gradient(plus,actions,old,advantages)[0]-ppo_gradient(minus,actions,old,advantages)[0])/2e-6
                self.assertAlmostEqual(derivative,gradient[i,j],places=7)

    def test_double_q_golden_and_cql_gradient(self):
        online=np.array([[1.,3.,2.]])
        target=np.array([[9.,4.,6.]])
        self.assertEqual(target[0,np.argmax(online)],4.) # not target max=9
        q=((.2,.1,-.3),); actions=(0,); targets=(.5,)
        obj=objective_v2(q,actions,targets,.1,enabled=True)
        for j in range(3):
            plus=list(q[0]); minus=list(q[0]);plus[j]+=1e-6;minus[j]-=1e-6
            d=(objective_v2((tuple(plus),),actions,targets,.1,enabled=True).loss-objective_v2((tuple(minus),),actions,targets,.1,enabled=True).loss)/2e-6
            self.assertAlmostEqual(d,obj.grad[0][j],places=7)

    def test_toy_three_learners_and_offline_no_collection(self):
        # Known one-state finite MDP, unique optimal long action; this checks
        # the learner mechanisms, not the financial environment's efficacy.
        class Toy:
            def __init__(self): self.done=False; self.n=0; self.failure=None
            def observation(self): return np.ones(12,dtype='float64')
            def step(self,a):
                self.n+=1; self.done=self.n==8
                return (np.zeros(12) if self.done else self.observation()),float(a-1),self.done
        factory=lambda _:Toy()
        frozen=r.Collector(factory,3).collect(lambda _:np.full(3,1/3),512)
        before=r.buffer_hash(frozen)
        def forbidden(_): raise AssertionError('offline learner collected new data')
        for alg in ('ppo','double_dqn','cql'):
            fit=r.train(alg,forbidden if alg=='cql' else factory,11,512,frozen,enabled=True)
            self.assertEqual(r.policy(fit,np.ones(12)),2,alg)
            self.assertEqual(fit.metadata['collectedTransitions'],0 if alg=='cql' else 512)
            self.assertGreater(fit.actor.step,0)
        self.assertEqual(before,r.buffer_hash(frozen))

if __name__=='__main__': unittest.main()

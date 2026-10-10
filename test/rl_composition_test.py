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

    def test_owned_state_and_late_data_fail_before_fill(self):
        from dataclasses import replace
        bars, events = fixture()
        ep = r.Episode(bars, events, 96, 129, enabled=True)
        ep.session = replace(ep.session, state=replace(ep.session.state, units=F(100)))
        with self.assertRaises(ValueError): ep.step(2)
        self.assertEqual(ep.rows, [])
        broken = bars[:97] + (replace(bars[97], available=bars[97].close-1),) + bars[98:]
        ep = r.Episode(broken, events, 96, 129, enabled=True)
        old = ep.session
        with self.assertRaises(ValueError): ep.step(2)
        self.assertIs(ep.session, old)
        self.assertEqual(ep.rows, [])

    def test_frozen_inference_finiteness_and_parity(self):
        from dataclasses import replace
        net = r.create_v2(11, 3, enabled=True)
        x = np.ones((1,12), dtype="float64")
        np.testing.assert_allclose(r.forward(net, x), r.forward(net.snapshot(), x), rtol=1e-12, atol=1e-12)
        with self.assertRaises(ValueError): r.forward(net.snapshot(), np.full((1,12), np.nan))
        bad = replace(net.snapshot(), p=(b"broken",) + net.snapshot().p[1:])
        with self.assertRaises(ValueError): r.forward(bad, x)
        with self.assertRaises(ValueError): r.forward(net.snapshot(), np.ones((1,9)))

    def test_trace_audit_detects_omitted_cost_and_oversized_intent(self):
        from copy import deepcopy
        from summarize_rl_composition import audit_rows
        bars,events=fixture()
        ep=r.Episode(bars,events,96,129,enabled=True)
        while not ep.done: ep.step(2)
        self.assertEqual(audit_rows(ep.rows)['violations'],[])
        corrupt=deepcopy(ep.rows)
        corrupt[0]['equity']+=.01
        corrupt[0]['permitted']=1.
        reasons={v['reason'] for v in audit_rows(corrupt)['violations']}
        self.assertTrue({'ledger_identity','action_domain','equity_continuity'}<=reasons)

    def test_repeatable_evaluation_and_partition_reset(self):
        bars,events=fixture()
        net=r.create_v2(23,3,enabled=True).snapshot()
        first=r.Episode(bars,events,96,129,'delay1bar',enabled=True)
        while not first.done: first.step(int(np.argmax(r.forward(net,first.observation()[None,:]))))
        # An unrelated episode cannot contaminate a fresh account or pending slot.
        unrelated=r.Episode(bars,events,120,153,enabled=True)
        unrelated.step(0)
        second=r.Episode(bars,events,96,129,'delay1bar',enabled=True)
        self.assertIsNone(second.pending)
        self.assertEqual(second.session.state.equity,1)
        while not second.done: second.step(int(np.argmax(r.forward(net,second.observation()[None,:]))))
        self.assertEqual(first.rows,second.rows)

    def test_exporter_rejects_bad_identity_and_nonfinite_json(self):
        import tempfile
        from summarize_rl_composition import export,read
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source';source.mkdir()
            (source/'index.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'index identity'): export(source,root/'output','wrong')
            self.assertFalse((root/'output').exists())
            bad=root/'bad.json';bad.write_text('{"value":1e999}')
            with self.assertRaisesRegex(ValueError,'overflowing'):read(bad)

    def test_registered_numeric_helper_import_cannot_expand_replay_surface(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/formal'))
        from shield_consumers import research_importers
        root=Path(__file__).resolve().parents[1]
        path='scripts/research/rl_composition.py'
        source=(root/path).read_text()
        self.assertEqual(len(research_importers({path:source})),5)
        runner='scripts/research/run_rl_composition.py'
        self.assertEqual(len(research_importers({runner:(root/runner).read_text()})),1)
        with self.assertRaisesRegex(ValueError,'unregistered research consumer'):
            research_importers({'scripts/research/unregistered.py':source})
        mutant=source.replace('from sequential_learning import ppo_gradient, softmax','from sequential_learning import *')
        with self.assertRaisesRegex(ValueError,'unregistered research consumer|source drift'): research_importers({path:mutant})
        with self.assertRaisesRegex(ValueError,'source drift'): research_importers({path:source+'\n# changed source\n'})

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

class AdmissionV2Tests(unittest.TestCase):
    def test_matched_control_uses_only_observation_vector(self):
        from compare_rl_observations_v2 import observation_rule
        for value,expected in ((-1.,0),(0.,1),(1.,2)):
            x=np.zeros(12,dtype='float64');x[1]=value
            self.assertEqual(observation_rule('normalized_momentum3',x),expected)
            self.assertEqual(observation_rule('normalized_reversal3',x),2-expected)
            y=np.full(12,100.,dtype='float64');y[1]=value
            self.assertEqual(observation_rule('normalized_momentum3',y),expected)
        for bad in (np.zeros(9),np.full(12,np.nan),[0.]*12):
            with self.assertRaises(ValueError):observation_rule('normalized_momentum3',bad)
        with self.assertRaises(ValueError):observation_rule('raw_momentum3',np.zeros(12))

    def test_rejected_reversal_continues_without_preview_debit(self):
        import rl_admission_v2 as v2
        bars,events=fixture()
        old=r.Episode(bars,events,96,129,enabled=True)
        new=v2.Episode(bars,events,96,129,enabled=True)
        rejected=0
        while not new.done:
            prior=new.session
            action=0 if new.index%2 else 2
            if not old.done: old.step(action)
            new.step(action)
            if new.rows[-1]['turnoverRejected']:
                reference=r.advance_owned(prior,bars[new.index],(),F(0),terminal=False,
                                          multiplier=F(1),fill=F(1),enabled=True)
                self.assertEqual(new.session,reference[0])
                self.assertFalse(new.done)
                self.assertEqual(new.rows[-1]['finalTarget'],0)
                rejected+=1
        self.assertEqual((len(old.rows),old.failure),(3,'turnover_limit'))
        self.assertEqual((len(new.rows),new.failure,rejected),(32,None,13))
        self.assertAlmostEqual(sum(x['reward'] for x in new.rows),100*float(new.session.state.equity-1),places=11)

    def test_economic_failure_is_not_hidden_or_reset(self):
        import rl_admission_v2 as v2
        from dataclasses import replace
        bars,events=fixture()
        # Establish a supported long position, then a severe adverse gap.
        bars=bars[:98]+(replace(bars[98],price=F(40)),)+bars[99:]
        for cls in (r.Episode,v2.Episode):
            ep=cls(bars,events,96,129,enabled=True)
            ep.step(2)
            self.assertNotEqual(ep.session.state.units,0)
            ep.step(1)
            self.assertTrue(ep.done)
            self.assertIn(ep.failure,('drawdown_limit','capital_floor','endpoint_exposure'))
            self.assertLess(ep.session.state.equity,1)
            with self.assertRaises(ValueError): ep.step(1)

    def test_v2_stresses_and_metrics(self):
        import rl_admission_v2 as v2
        from summarize_rl_composition import audit_rows
        from copy import deepcopy
        bars,events=fixture()
        with self.assertRaises(ValueError): v2.Episode(bars,events,96,129)
        for stress in r.STRESSES:
            ep=v2.Episode(bars,events,96,129,stress,enabled=True)
            while not ep.done: ep.step(0 if ep.index%2 else 2)
            self.assertEqual(len(ep.rows),32)
            self.assertIsNone(ep.failure)
            self.assertEqual(ep.session.state.units,0)
            self.assertEqual(audit_rows(ep.rows)['violations'],[])
            self.assertTrue(all(x['intervention']==(x['supportChangedAction'] or x['turnoverRejected']) for x in ep.rows))
            if stress=='delay1bar':
                self.assertTrue(any(x['newProposalIgnored'] for x in ep.rows))
                self.assertTrue(any(x['schedulingDeferred'] for x in ep.rows))
                self.assertTrue(all(x['unitsBefore']==x['unitsAfter'] and x['turnover']==0
                                    for x in ep.rows if x['schedulingDeferred'] and not x['terminal']))
        cash=v2.Episode(bars,events,96,129,enabled=True)
        while not cash.done: cash.step(1)
        self.assertFalse(any(x['actionChanged'] for x in cash.rows))
        corrupt=deepcopy(cash.rows)
        corrupt[0]['unitsAfter']=1.
        self.assertIn('fill_reconstruction',{v['reason'] for v in audit_rows(corrupt)['violations']})

    def test_budget_registration_and_vector_only_baselines(self):
        import json
        from copy import deepcopy
        from types import SimpleNamespace
        import run_rl_budget_v3 as budget
        reg = json.loads(budget.REG.read_bytes())
        config = budget.configuration(reg)
        self.assertEqual(config['offline_transitions'], 512)
        self.assertEqual(config['comparators'], ('cash', 'normalized_momentum3'))
        for key, value in (('steps', 512), ('offlineTransitions', 4096), ('liveAuthorization', True)):
            corrupt = deepcopy(reg); corrupt[key] = value
            with self.assertRaises(ValueError): budget.configuration(corrupt)
        # No history/session member is available to these controls.
        x = np.zeros(12); x[1] = -.2
        ep = SimpleNamespace(observation=lambda: x)
        self.assertEqual([budget.baseline(name, ep) for name in budget.RULES], [1, 2, 0, 2])
        x[1] = .2
        self.assertEqual(budget.baseline('normalized_momentum3', ep), 2)

    def test_budget_buffer_identity_blocks_training(self):
        import json
        import tempfile
        from unittest.mock import Mock, patch
        import run_rl_budget_v3 as budget
        import run_rl_composition as driver
        config = budget.configuration(json.loads(budget.REG.read_bytes()))
        config['expected_behavior_hashes'] = ('0'*64, '0'*64)
        trainer = Mock(); config['trainer'] = trainer
        bars = tuple(Bar('FIXTURE', i*1000, i*1000+1, F(10000+(i%7)*10,100)) for i in range(1600))
        events = tuple(() for _ in bars)
        real_check_output = driver.subprocess.check_output
        def committed(cmd, **kwargs):
            if cmd[0] != 'git': return real_check_output(cmd, **kwargs)
            if cmd[1] == 'rev-parse': return 'fixture\n'
            return (driver.ROOT/cmd[-1].split(':', 1)[1]).read_bytes()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(driver, 'load', return_value=(dict.fromkeys(('BTCUSDT','ETHUSDT','SOLUSDT'), bars),
                                                        dict.fromkeys(('BTCUSDT','ETHUSDT','SOLUSDT'), events))), \
             patch.object(driver.subprocess, 'check_output', side_effect=committed):
            output = Path(directory)/'evidence'
            with self.assertRaisesRegex(ValueError, 'fixed dataset differs'):
                driver.run(Path('unused-panel'), Path('unused-settlements'), output, **config)
            trainer.assert_not_called()
            self.assertEqual(list((output/'policies').iterdir()), [])

    def test_budget_paired_comparator_excludes_raw_history_rule(self):
        import run_rl_composition as driver
        records = [dict(algorithm=a, fold=f, stress='base', status='complete',
                        increments=[value]*64)
                   for f in (0,1) for a,value in (('ppo',.001),('double_dqn',.001),('cql',.001),
                                                  ('cash',0.),('normalized_momentum3',.002),('momentum3',99.))]
        paired = driver.paired(records, ('cash','normalized_momentum3'))
        self.assertEqual(len(paired), 18)
        for row in paired:
            self.assertAlmostEqual(row['meanPairedNetChange'], .064 if row['baseline']=='cash' else -.064)

    def test_actual_ddqn_loop_targets_masks_and_sync(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        class Net:
            step=0
            def snapshot(self):return SimpleNamespace(step=self.step,outputs=3)
        net=Net(); observed=[]
        data={'s':np.ones((128,12)), 'next':np.ones((128,12)),
              'a':np.arange(128)%3, 'r':np.arange(128,dtype=float),
              'done':np.arange(128)%2==0, 'prob':np.full(128,1/3)}
        def forward(n,x):
            # Online chooses index1; target max is index0. Distinct values
            # detect using max(target), online evaluation or premature sync.
            values=[1.,3.,2.] if n is net else [n.step+9.,n.step+4.,n.step+6.]
            return np.tile(values,(len(x),1))
        def objective(q,actions,targets,alpha,enabled):
            for value in targets:
                # Even rewards are terminal; odd rewards bootstrap index1.
                # Batch index reconstruction below verifies every exact target.
                self.assertTrue(np.isfinite(value))
            observed.append((net.step,tuple(targets)))
            return SimpleNamespace(loss=0.,grad=np.zeros((64,3)))
        def update(*_):net.step+=1
        with patch.object(r,'create_v2',return_value=net), patch.object(r.Collector,'collect',return_value=data), \
             patch.object(r,'forward',side_effect=forward), patch.object(r,'objective_v2',side_effect=objective), \
             patch.object(r,'update',side_effect=update):
            r.train('double_dqn',lambda _:None,11,128,enabled=True)
        rng=np.random.default_rng(13)
        for step,actual in observed:
            idx=rng.integers(128,size=64)
            expected=np.where(data['done'][idx],data['r'][idx],data['r'][idx]+(step//100)*100+4)
            np.testing.assert_array_equal(actual,expected)
        self.assertEqual(len(observed),128)


if __name__=='__main__': unittest.main()

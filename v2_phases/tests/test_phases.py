"""Small software tests use an existing original checkpoint, without training."""
import copy
import json
import random
import sqlite3
import zipfile

import numpy as np
import pytest
import torch
from env.siege_env import SiegeEnv
from env.symmetry import SymmetryWrapper
from experiments.rollout import play_episode
from version2.evaluation import ActingPolicy, episode_seed
from v2_phases import runner as R


@pytest.fixture(scope='module')
def fixture():
    R.E.configure_cpu()
    pool = R.E.load_pool(R.ROOT/'bases/dev-v2.json',required_split='dev')[:1]
    path = 'runs/study-v2/P-0/model.zip'
    target = {'run':'P-0','condition':'P','learner_seed':0,'transitions':262144,
              'path':path,'sha256':R.E.sha256(R.ROOT/path)}
    model = R.E.load_target(target,R.ROOT)
    contract = {'scope':'Software tests only','models':[target],'layouts':[{'seed':pool[0].seed,'motif':pool[0].motif}],
                'new_modes':['DS','SD'],'sample_seeds':[31001,31002],'views':[0],'monitor_panel':[], 'expected_episodes':4}
    baselines = []
    for mode, labels in [('sampled',[31001,31002]),('deterministic',[0])]:
        for label in labels:
            row = R.E.evaluate_one(target,model,pool[0],0,mode,label,set())
            row['mode'] = 'SS' if mode=='sampled' else 'DD'
            baselines.append(row)
    return contract,pool,baselines,model


@pytest.mark.parametrize('mode',['SS','DD','DS','SD'])
def test_phase_flags_include_finish_shopping(mode):
    class Spy:
        def __init__(self): self.flags=[]
        def predict(self,obs,action_masks,deterministic):
            self.flags.append(deterministic)
            return int(np.flatnonzero(action_masks)[0]),None
    spy=Spy();policy=R.PhasePolicy(spy,mode)
    obs={'vector':np.zeros(9)}
    mask=np.zeros(R.C.N_ACTIONS,dtype=bool);mask[R.C.ACTION_FINISH_SHOPPING]=True
    assert policy.act(obs,mask)==R.C.ACTION_FINISH_SHOPPING
    obs['vector'][6]=1;mask[:]=False;mask[R.C.ACTION_WAIT]=True
    assert policy.act(obs,mask)==R.C.ACTION_WAIT
    assert tuple(spy.flags)==R.MODES[mode]


@pytest.mark.parametrize('mode,label,deterministic',[('SS',31001,False),('DD',0,True)])
def test_uniform_modes_match_existing_action_sequences(fixture,mode,label,deterministic):
    _,pool,_,model=fixture;base=pool[0];seed=episode_seed(label,base,3)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    env=SymmetryWrapper(SiegeEnv(motifs=R.C.DEV_MOTIFS),transform=3)
    expected=play_episode(env,ActingPolicy(model.policy,deterministic),options={'spec':base})
    env.close();rng=torch.get_rng_state().clone()
    actual=R.episode(model,base,3,mode,label)
    assert actual['actions']==expected['actions'] and actual['summary']==expected['summary']
    assert torch.equal(torch.get_rng_state(),rng)


def test_mixed_modes_preserve_matching_purchasing_rule(fixture):
    _,pool,_,model=fixture
    episodes={mode:R.episode(model,pool[0],0,mode,31001)['actions'] for mode in R.MODES}
    def purchase(actions):return actions[:actions.index(R.C.ACTION_FINISH_SHOPPING)+1]
    assert purchase(episodes['SS'])==purchase(episodes['SD'])
    assert purchase(episodes['DD'])==purchase(episodes['DS'])


def test_resume_completion_and_package(fixture,tmp_path):
    contract,pool,baselines,_=fixture
    a,b=tmp_path/'a',tmp_path/'b'
    assert R.execute(contract,pool,baselines,a)['status']=='completed'
    assert R.execute(contract,pool,baselines,b,max_new=1)['status']=='paused'
    assert not (b/'complete.json').exists()
    assert R.execute(contract,pool,baselines,b)['status']=='completed'
    assert R.execute(contract,pool,baselines,b)['new_episodes_this_attempt']==0
    receipt=R.read(a/'complete.json')
    assert receipt==R.read(b/'complete.json')
    assert receipt['new_episodes']==4 and receipt['reused_baseline_episodes']==3 and receipt['combined_episodes']==7
    for name in receipt['outputs']:
        assert (a/name).read_bytes()==(b/name).read_bytes()
    with zipfile.ZipFile(R.pack(b)) as archive:
        assert archive.testzip() is None
        assert not any('sqlite3' in name for name in archive.namelist())
    (b/'mixed_episodes.csv').write_text('changed')
    with pytest.raises(ValueError,match='Result hash mismatch'):
        R.pack(b)


def test_pause_request_commits_current_attack(fixture,tmp_path,monkeypatch):
    contract,pool,baselines,_=fixture
    original=R.evaluate_one
    def pause(*args):
        row=original(*args)
        R.atomic_json(tmp_path/'STOP_REQUESTED.json',{'request':'test'})
        return row
    with monkeypatch.context() as patch:
        patch.setattr(R,'evaluate_one',pause)
        result=R.execute(contract,pool,baselines,tmp_path)
    assert result['status']=='paused' and result['completed_episodes']==1
    assert R.execute(contract,pool,baselines,tmp_path)['status']=='completed'
    assert list((tmp_path/'stop_requests').glob('handled-*.json'))


def test_changed_contract_and_corrupt_progress_are_rejected(fixture,tmp_path):
    contract,pool,baselines,_=fixture
    R.execute(contract,pool,baselines,tmp_path,max_new=1)
    changed=copy.deepcopy(contract);changed['sample_seeds']=[77]
    with pytest.raises(ValueError,match='contract changed'):
        R.execute(changed,pool,baselines,tmp_path)
    with sqlite3.connect(tmp_path/'progress.sqlite3') as db:
        db.execute("UPDATE items SET value='{}'")
    with pytest.raises(ValueError,match='checksum mismatch'):
        R.execute(contract,pool,baselines,tmp_path)


def test_baseline_import_keeps_dd_once_and_rejects_duplicates(fixture,tmp_path):
    contract,_,baselines,_=fixture
    rows=copy.deepcopy(baselines)
    for row in rows:row['mode']='sampled' if row['mode']=='SS' else 'deterministic'
    (tmp_path/'episodes.csv').write_bytes(R.csv_bytes(rows))
    result=R.read_baselines(tmp_path,contract)
    assert len(result)==3 and sum(r['mode']=='DD' for r in result)==1
    (tmp_path/'episodes.csv').write_bytes(R.csv_bytes(rows+[rows[0]]))
    with pytest.raises(ValueError,match='duplicate baseline'):
        R.read_baselines(tmp_path,contract)


def test_complete_design_has_44928_unique_new_units():
    expected=R.E.read(R.E.PACKAGE/'expected.json')
    models=[m for m in expected['models'] if m['transitions']==4194304]
    pool=R.E.load_pool(R.ROOT/'bases/dev-v2.json',required_split='dev')
    contract={'models':models,'new_modes':list(R.NEW_MODES),'views':list(range(8)),'sample_seeds':[31001,31002,31003]}
    keys=[R.E.unit_key(*unit) for unit in R.units(contract,pool)]
    assert len(keys)==len(set(keys))==44928


def test_comparison_uses_equal_learner_means():
    rows=[]
    for seed in range(3):
        for mode,score,labels in [('SS',.6,[1,2,3]),('DD',.4,[0]),('DS',.7,[1,2,3]),('SD',.5,[1,2,3])]:
            for monitor in (False,True):
                for label in labels:
                    rows.append({'condition':'P','learner_seed':seed,'transitions':4194304,'mode':mode,
                                 'base_seed':10200+int(monitor),'motif':'closed_ring','view':0,'sample_seed':label,
                                 'monitor_layout':monitor,'destruction':score,'full_clear':False,'town_hall':True,
                                 'zero_destruction':0.,'episode_return':score,'spent':200,'tick':300,'decisions':45})
    models,conditions,comparisons,aggregates=R.summaries(rows)
    assert len(models)==24 and len(conditions)==8 and len(comparisons)==24 and len(aggregates)==8
    for row in aggregates:
        expected={('DS','SS'):10,('DS','DD'):30,('SD','SS'):-10,('SD','DD'):10}[row['mode'],row['reference_mode']]
        assert row['mean_change_destruction_percent_points']==pytest.approx(expected)
        assert row['learner_runs']==3
        assert row['learners_with_higher_destruction']==(3 if expected>0 else 0)
    assert all(r['layouts']==(1 if r['scope']=='outside_monitor_panel' else 2) for r in models)

"""Evaluation tests use an existing original checkpoint; no training occurs."""
import copy
import json
import sqlite3
from pathlib import Path
import pytest
from v2_development import runner as R


@pytest.fixture
def fixture():
    R.configure_cpu()
    pool=R.load_pool(R.ROOT/'bases/dev-v2.json',required_split='dev')[:1]
    path='runs/study-v2/P-0/model.zip'
    model={'run':'P-0','condition':'P','learner_seed':0,'transitions':262144,
           'path':path,'sha256':R.sha256(R.ROOT/path)}
    contract={'scope':'software test only','models':[model],'sample_seeds':[31001,31002,31003],
              'views':[0],'monitor_panel':[],'expected_episodes':5}
    return contract,pool


def test_real_policy_resume_matches_uninterrupted(tmp_path,fixture):
    contract,pool=fixture
    a,b=tmp_path/'a',tmp_path/'b'
    assert R.execute(contract,pool,a)['status']=='completed'
    paused=R.execute(contract,pool,b,max_new=2)
    assert paused['status']=='paused' and paused['completed_episodes']==2
    assert not (b/'complete.json').exists()
    assert R.execute(contract,pool,b)['status']=='completed'
    for name in ('episodes.csv','model_summary.csv','condition_summary.csv','budget_comparison.csv','complete.json'):
        assert (a/name).read_bytes()==(b/name).read_bytes()
    receipt=(b/'complete.json').read_bytes()
    result=R.execute(contract,pool,b)
    assert result['new_episodes_this_attempt']==0
    assert (b/'complete.json').read_bytes()==receipt


def test_changed_contract_is_rejected(tmp_path,fixture):
    contract,pool=fixture
    R.execute(contract,pool,tmp_path,max_new=1)
    changed=copy.deepcopy(contract);changed['sample_seeds']=[99]
    with pytest.raises(ValueError,match='contract changed'):
        R.execute(changed,pool,tmp_path)


def test_pause_finishes_current_episode(tmp_path,fixture,monkeypatch):
    contract,pool=fixture
    original=R.evaluate_one
    def request_pause(*args):
        result=original(*args)
        R.atomic_json(tmp_path/'STOP_REQUESTED.json',{'request':'test'})
        return result
    with monkeypatch.context() as patch:
        patch.setattr(R,'evaluate_one',request_pause)
        result=R.execute(contract,pool,tmp_path)
    assert result['status']=='paused' and result['completed_episodes']==1
    assert R.execute(contract,pool,tmp_path)['status']=='completed'
    assert list((tmp_path/'stop_requests').glob('handled-*.json'))


def test_corrupt_episode_is_not_silently_used(tmp_path,fixture):
    contract,pool=fixture
    R.execute(contract,pool,tmp_path,max_new=1)
    with sqlite3.connect(tmp_path/'progress.sqlite3') as db:
        db.execute("UPDATE items SET value='{}'")
    with pytest.raises(ValueError,match='checksum mismatch'):
        R.execute(contract,pool,tmp_path)


def test_model_counter_and_hash_guards(fixture):
    contract,_=fixture
    model=copy.deepcopy(contract['models'][0])
    model['sha256']='0'*64
    with pytest.raises(ValueError,match='Model changed'):
        R.load_target(model,R.ROOT)
    model=copy.deepcopy(contract['models'][0]);model['transitions']=1
    with pytest.raises(ValueError,match='timestep counter'):
        R.load_target(model,R.ROOT)


def test_full_design_has_unique_expected_units():
    expected=R.read(R.PACKAGE/'expected.json')
    pool=R.load_pool(R.ROOT/'bases/dev-v2.json',required_split='dev')
    contract={'models':expected['models'],'views':R.VIEWS,'sample_seeds':R.SEEDS}
    keys=[R.unit_key(t,b,v,m,s) for t in R.targets(contract) for b,v,m,s in R.units(t,contract,pool)]
    assert len(keys)==len(set(keys))==121056
    assert len(pool)==156
    assert len(R.development_panel(pool))==12


def test_summary_averages_learners_and_separates_monitor_layouts():
    rows=[]
    for seed in range(3):
        for steps,score in ((262144,0.5+seed*0.1),(4194304,0.6+seed*0.1)):
            for monitor,base in ((False,10200),(True,10201)):
                rows.append({'condition':'P','learner_seed':seed,'transitions':steps,'mode':'sampled',
                    'monitor_layout':monitor,'base_seed':base,'motif':'closed_ring',
                    'destruction':score,'full_clear':False,'town_hall':True,'zero_destruction':0,
                    'episode_return':1,'spent':200,'tick':300,'decisions':40})
    models,means,changes=R.summarize(rows)
    assert len(models)==12 and len(means)==4 and len(changes)==6
    for row in changes:
        assert row['change_percentage_points']==pytest.approx(10)
    for row in means:
        assert row['learner_runs']==3
        assert row['episodes']==(6 if row['scope']=='all_dev' else 3)
        assert row['destruction_percent']==pytest.approx(60 if row['transitions']==262144 else 70)

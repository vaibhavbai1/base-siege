"""Small read-only inference tests; no training and no research-result mutation."""
import copy
import hashlib
import json
import random
import sqlite3
import zipfile
import numpy as np
import pytest
import torch

from v2_cohort_evaluation import runner as R
from v2_cohort_evaluation.analysis import summaries, interval
from experiments.storage import atomic_json
from experiments.journal import open_journal
from version2.evaluation import ActingPolicy, episode_seed
from env.siege_env import SiegeEnv
from env.symmetry import SymmetryWrapper
from experiments.rollout import play_episode


@pytest.fixture(scope='module')
def fixture():
    R.E.configure_cpu()
    pool = R.E.load_pool(R.ROOT/'bases/dev-v2.json', required_split='dev')[:1]
    candidates = [('runs/study-v2/P-0/model.zip',262144),
                  ('runs/version2-longer-control/P-0/milestones/step-000004194304.zip',4194304)]
    path, steps = next((p,s) for p,s in candidates if (R.ROOT/p).is_file())
    models = [{'run':f'{c}-{s}','condition':c,'learner_seed':s,'transitions':steps,
               'path':path,'sha256':R.E.sha256(R.ROOT/path)} for s in (0,3) for c in ('P','PA')]
    model = R.E.load_target(models[0],R.ROOT)
    contract = {'scope':'Software fixture only: identities share one existing policy',
                'models':models,'screen_models':models[:2],'new_models':models[2:],
                'layouts':[{'seed':pool[0].seed,'motif':pool[0].motif}], 'modes':['DS','SS','DD'],
                'sample_seeds':[31001,31002],'views':[0],'monitor_panel':[], 'expected_episodes':10,
                'protocol':{'intervals':{**R.read(R.PACKAGE/'protocol.json')['intervals'],'resamples':100}}}
    baseline = [R.evaluate_one(t,model,pool[0],0,mode,label,set()) for t in models[:2]
                for mode in contract['modes'] for label in R.labels(contract,mode)]
    return contract,pool,baseline,model


@pytest.mark.parametrize('mode',['DS','SS','DD'])
def test_phase_flags_include_finish_shopping(mode):
    class Spy:
        def __init__(self): self.flags=[]
        def predict(self,obs,action_masks,deterministic):
            self.flags.append(deterministic)
            return int(np.flatnonzero(action_masks)[0]),None
    spy=Spy(); policy=R.PhasePolicy(spy,mode)
    obs={'vector':np.zeros(9)}; mask=np.zeros(R.F.C.N_ACTIONS,dtype=bool)
    mask[R.F.C.ACTION_FINISH_SHOPPING]=True
    assert policy.act(obs,mask)==R.F.C.ACTION_FINISH_SHOPPING
    obs['vector'][6]=1;mask[:]=False;mask[R.F.C.ACTION_WAIT]=True
    assert policy.act(obs,mask)==R.F.C.ACTION_WAIT
    assert tuple(spy.flags)==R.MODES[mode]


@pytest.mark.parametrize('mode,label,deterministic',[('SS',31001,False),('DD',0,True)])
def test_uniform_modes_match_original_actions_and_rng(fixture,mode,label,deterministic):
    _,pool,_,model=fixture; base=pool[0];seed=episode_seed(label,base,3)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    env=SymmetryWrapper(SiegeEnv(motifs=R.F.C.DEV_MOTIFS),transform=3)
    try:expected=play_episode(env,ActingPolicy(model.policy,deterministic),options={'spec':base})
    finally:env.close()
    rng=torch.get_rng_state().clone()
    actual=R.episode(model,base,3,mode,label)
    assert actual['actions']==expected['actions'] and actual['summary']==expected['summary']
    assert torch.equal(torch.get_rng_state(),rng)


def test_ds_keeps_deterministic_purchase_prefix(fixture):
    _,pool,_,model=fixture
    sequences=[R.episode(model,pool[0],0,m,31001)['actions'] for m in ('DS','DD')]
    def prefix(a):return a[:a.index(R.F.C.ACTION_FINISH_SHOPPING)+1]
    assert prefix(sequences[0])==prefix(sequences[1])


def test_resume_exports_and_pack_match_uninterrupted(fixture,tmp_path):
    contract,pool,baseline,_=fixture;a,b=tmp_path/'a',tmp_path/'b'
    assert R.execute(contract,pool,baseline,a)['status']=='completed'
    assert R.execute(contract,pool,baseline,b,max_new=1)['status']=='paused'
    assert not (b/'complete.json').exists()
    assert R.execute(contract,pool,baseline,b)['status']=='completed'
    assert R.execute(contract,pool,baseline,b)['new_episodes_this_attempt']==0
    receipt=R.read(a/'complete.json')
    assert receipt==R.read(b/'complete.json')
    assert (receipt['new_episodes'],receipt['reused_baseline_episodes'],receipt['combined_episodes'])==(10,10,20)
    for name in receipt['outputs']: assert (a/name).read_bytes()==(b/name).read_bytes()
    with zipfile.ZipFile(R.pack(b)) as z:
        assert z.testzip() is None and not any('sqlite3' in n for n in z.namelist())
    (b/'new_episodes.csv').write_text('changed')
    with pytest.raises(ValueError,match='File changed'):R.pack(b)


def test_pause_commits_current_attack(fixture,tmp_path,monkeypatch):
    contract,pool,baseline,_=fixture;original=R.evaluate_one
    def pause(*args):
        row=original(*args);R.atomic_json(tmp_path/'STOP_REQUESTED.json',{'request':'test'});return row
    with monkeypatch.context() as patch:
        patch.setattr(R,'evaluate_one',pause)
        result=R.execute(contract,pool,baseline,tmp_path)
    assert result['status']=='paused' and result['completed_episodes']==1
    assert R.execute(contract,pool,baseline,tmp_path)['status']=='completed'
    assert list((tmp_path/'stop_requests').glob('handled-*.json'))


def test_changed_contract_and_corrupt_journal_rejected(fixture,tmp_path):
    contract,pool,baseline,_=fixture
    R.execute(contract,pool,baseline,tmp_path,max_new=1)
    changed=copy.deepcopy(contract);changed['sample_seeds']=[77]
    with pytest.raises(ValueError,match='contract changed'):R.execute(changed,pool,baseline,tmp_path)
    with sqlite3.connect(tmp_path/'progress.sqlite3') as db:db.execute("UPDATE items SET value='{}'")
    with pytest.raises(ValueError,match='checksum mismatch'):R.execute(contract,pool,baseline,tmp_path)


def test_concurrent_writer_rejected(fixture,tmp_path):
    contract,pool,baseline,_=fixture
    with open_journal(tmp_path/'progress.sqlite3',contract):
        with pytest.raises(RuntimeError,match='active'):R.execute(contract,pool,baseline,tmp_path)


def test_reused_rows_keep_dd_once_and_reject_duplicates(fixture,tmp_path):
    contract,_,baseline,_=fixture
    (tmp_path/'combined_episodes.csv').write_bytes(R.csv_bytes(baseline))
    actual=R.read_baselines(tmp_path,contract)
    assert len(actual)==10 and sum(r['mode']=='DD' for r in actual)==2
    (tmp_path/'combined_episodes.csv').write_bytes(R.csv_bytes(baseline+[baseline[0]]))
    with pytest.raises(ValueError,match='duplicate'):R.read_baselines(tmp_path,contract)


def test_full_design_and_real_reused_baseline():
    phase=R.read(R.ROOT/'artifacts/version2-phase-evaluation/evaluation_contract.json')
    new=R.read(R.PACKAGE/'replication_reference.json')['models']
    contract={**phase,'screen_models':phase['models'],'new_models':new,'models':phase['models']+new,'modes':['DS','SS','DD']}
    pool=R.E.load_pool(R.ROOT/'bases/dev-v2.json',required_split='dev')
    keys=[R.E.unit_key(*unit) for unit in R.units(contract,pool)]
    assert len(keys)==len(set(keys))==122304
    baseline=R.read_baselines(R.ROOT/'artifacts/version2-phase-evaluation',contract)
    assert len(baseline)==52416 and sum(r['mode']=='DD' for r in baseline)==7488
    assert len(R.expected_keys(contract,contract['models']))==174720
    # Verify the exact phase evidence referenced by this package.
    ref=R.read(R.PACKAGE/'baseline_reference.json')
    R.verify_files(R.ROOT/ref['directory'],ref['files'])
    assert phase['package_hashes']==R.F.package_hashes()


def test_cohort_means_and_paired_intervals():
    rows=[]
    for seed in range(10):
        for c in ('P','PA'):
            for mode in ('DS','SS','DD'):
                score=(.6 if seed<3 else .8)+(.02 if c=='PA' else 0)-({'DS':0,'SS':.03,'DD':.1}[mode])
                # Unequal numbers of attacks must not give a learner more weight.
                for view in range(1+seed%2):
                    for monitor in (True,False):
                        rows.append({'condition':c,'learner_seed':seed,'transitions':4194304,'mode':mode,
                                     'base_seed':10200+int(monitor),'motif':'closed_ring','view':view,
                                     'monitor_layout':monitor,'destruction':score,'full_clear':False,'town_hall':True,
                                     'zero_destruction':0.,'episode_return':score,'spent':200,'tick':300,'decisions':45})
    options=R.read(R.PACKAGE/'protocol.json')['intervals'];result=summaries(rows,options)
    assert len(result['model_summary.csv'])==120
    assert len(result['condition_summary.csv'])==36
    assert len(result['mode_comparisons_by_learner.csv'])==80
    assert len(result['mode_comparisons.csv'])==24
    assert len(result['augmentation_by_seed.csv'])==60
    assert len(result['augmentation_comparisons.csv'])==18
    for r in result['condition_summary.csv']:
        expected={'screening':60,'replication':80,'all':74}[r['cohort']]+(2 if r['condition']=='PA' else 0)-{'DS':0,'SS':3,'DD':10}[r['mode']]
        assert r['destruction_percent']==pytest.approx(expected)
    for r in result['mode_comparisons.csv']:
        expected=3 if r['reference_mode']=='SS' else 10
        assert r['mean_change_destruction_percent_points']==pytest.approx(expected)
        assert r['learner_pairs']=={'screening':3,'replication':7,'all':10}[r['cohort']]
        assert r['ci95_low_points']==('' if r['cohort']=='screening' else pytest.approx(expected))
    for r in result['augmentation_comparisons.csv']:
        assert r['mean_pa_minus_p_destruction_percent_points']==pytest.approx(2)
    np.random.seed(99);before=np.random.get_state()
    assert interval([1,2,3,4,5,6,7],options)==interval([1,2,3,4,5,6,7],options)
    after=np.random.get_state()
    assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]


def test_replication_references_reject_altered_model(tmp_path):
    reference={'registration_path':'registration.json','replication_source_hashes':{},'runs':{},'models':[]}
    shared={k:{} for k in ('source_hashes','version2_source_hashes','replication_source_hashes','config','python','packages')}
    atomic_json(tmp_path/'registration.json',shared);reference['registration_sha256']=R.E.sha256(tmp_path/'registration.json')
    for c in ('P','PA'):
        for s in range(3,10):
            name=f'{c}-{s}';prefix=f'runs/{name}';p=tmp_path/prefix;p.mkdir(parents=True)
            (p/'model.zip').write_bytes(b'software-fixture-only');digest=R.E.sha256(p/'model.zip')
            atomic_json(p/'contract.json',{**shared,'settings':{'condition':c,'seed':s}})
            contract=R.E.sha256(p/'contract.json')
            atomic_json(p/'manifest.json',{**shared,'status':'completed','actual_steps':4194304,'contract_sha256':contract,
                        'registration_sha256':reference['registration_sha256'],'model_sha256':digest})
            atomic_json(p/'milestones/step-000004194304.json',{'sha256':digest,'contract_sha256':contract})
            reference['runs'][name]={'path':prefix,'manifest_sha256':R.E.sha256(p/'manifest.json'),'contract_sha256':contract,
                                    'final_metadata_sha256':R.E.sha256(p/'milestones/step-000004194304.json')}
            reference['models'].append({'run':name,'condition':c,'learner_seed':s,'path':prefix+'/model.zip','sha256':digest})
    assert R.verify_replicas(tmp_path,reference)==shared
    (tmp_path/'runs/P-3/model.zip').write_bytes(b'changed')
    with pytest.raises(ValueError,match='File changed'):R.verify_replicas(tmp_path,reference)

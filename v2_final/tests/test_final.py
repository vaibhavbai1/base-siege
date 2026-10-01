"""Software checks only: development seeds and archived weights, never fresh final seeds."""
import copy
import hashlib
import json
import random
import sqlite3
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from v2_final import runner as R
from v2_final.analysis import summaries, interval
from experiments.storage import atomic_json, run_lock
from experiments.rollout import play_episode
from env.generator import generate_base
from env.siege_env import SiegeEnv
from env.symmetry import SymmetryWrapper
from v2_phases.runner import PhasePolicy


@pytest.fixture(scope='module')
def fixture():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    paths=['runs/study-v2/P-0/model.zip','runs/version2-longer-control/P-0/milestones/step-000004194304.zip']
    path=next((p for p in paths if (R.ROOT/p).is_file()),None)
    if path is None:pytest.skip('Existing checkpoint required for inference fixture; no training is run')
    loaded=R.MaskablePPO.load(R.ROOT/path,device='cpu')
    models=[{'condition':c,'learner_seed':0,'transitions':loaded.num_timesteps,'run':c+'-0','path':path,
             'sha256':R.sha256(R.ROOT/path)} for c in ['P','PA']]
    protocol=copy.deepcopy(R.read(R.PACKAGE/'protocol.json'))
    protocol.update(candidate_seed_interval=[10200,10200],motifs=['closed_ring','ew_gates'],candidate_layouts=2,views=[0,3],sample_seeds=[41001,41002])
    protocol['intervals']['resamples']=100
    contract={'kind':'SOFTWARE FIXTURE: shared old checkpoint, development layouts only','models':models,'protocol':protocol}
    return contract,R.load_target(models[0])


def frozen(folder,contract):
    folder.mkdir(parents=True,exist_ok=True)
    # Fixed timestamp makes exports from resumed and uninterrupted fixtures comparable.
    atomic_json(folder/'freeze.json',{'frozen_utc':'software-fixture','contract':contract})


def test_real_ds_inference_is_repeatable_and_matches_phase_policy(fixture):
    contract,model=fixture;target=contract['models'][0];base=generate_base(10200,'closed_ring')
    actual=R.episode(target,model,base,3,41001)
    rng=torch.get_rng_state().clone()
    seed=R.acting_seed(41001,base,3)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    env=SymmetryWrapper(SiegeEnv(),transform=3)
    try:expected=play_episode(env,PhasePolicy(model.policy,'DS'),options={'spec':base})
    finally:env.close()
    assert actual['actions']==expected['actions'] and actual['summary']==expected['summary']
    assert torch.equal(torch.get_rng_state(),rng)
    assert R.episode(target,model,base,3,41001)['actions']==actual['actions']
    # Purchasing prefix is identical to DD, including finish-shopping.
    env=SymmetryWrapper(SiegeEnv(),transform=3)
    try:dd=play_episode(env,PhasePolicy(model.policy,'DD'),options={'spec':base})
    finally:env.close()
    def purchase(a):return a[:a.index(R.C.ACTION_FINISH_SHOPPING)+1]
    assert purchase(actual['actions'])==purchase(dd['actions'])


def test_real_heuristic_is_ordinary_policy_not_oracle(fixture):
    base=generate_base(10200,'ew_gates');target={'condition':'heuristic','learner_seed':-1,'transitions':0}
    actual=R.episode(target,None,base,3,0)
    env=SymmetryWrapper(SiegeEnv(),transform=3)
    try:expected=play_episode(env,R.HeuristicCommander(),options={'spec':base})
    finally:env.close()
    assert actual['actions']==expected['actions'] and actual['summary']==expected['summary']


def test_resume_idempotence_export_pack_and_tamper(fixture,tmp_path):
    contract,_=fixture;a,b=tmp_path/'a',tmp_path/'b';frozen(a,contract);frozen(b,contract)
    assert R.execute(contract,a)['status']=='completed'
    result=R.execute(contract,b,max_new=1)
    assert result['status']=='paused' and result['new_attacks_this_attempt']==1 and not (b/'complete.json').exists()
    assert R.execute(contract,b)['status']=='completed'
    assert R.execute(contract,b)['new_attacks_this_attempt']==0
    receipt=R.read(a/'complete.json');assert receipt==R.read(b/'complete.json') and receipt['episodes']==20
    for name in receipt['outputs']:assert (a/name).read_bytes()==(b/name).read_bytes()
    with zipfile.ZipFile(R.pack(b)) as z:
        assert z.testzip() is None
        assert not any('sqlite3' in n for n in z.namelist())
        assert json.loads(z.read('version2-final-evaluation/timing_summary.json'))['attempts_missing_end_record']==[]
    (b/'episodes.csv').write_text('tampered')
    with pytest.raises(ValueError,match='File changed'):R.pack(b)


def test_pause_during_attack_commits_once(fixture,tmp_path,monkeypatch):
    contract,_=fixture;frozen(tmp_path,contract);original=R.evaluate_one
    def pausing(*args):
        row=original(*args);atomic_json(tmp_path/'STOP_REQUESTED.json',{});return row
    with monkeypatch.context() as m:
        m.setattr(R,'evaluate_one',pausing)
        result=R.execute(contract,tmp_path)
    assert result['status']=='paused' and result['new_attacks_this_attempt']==1
    assert R.execute(contract,tmp_path)['status']=='completed'


def test_pause_during_pool_and_resume_without_rechecking(fixture,tmp_path,monkeypatch):
    contract,_=fixture;frozen(tmp_path,contract);original=R.validate_base;calls=[]
    def pausing(base,**kwargs):
        calls.append((base.seed,base.motif));answer=original(base,**kwargs);atomic_json(tmp_path/'STOP_REQUESTED.json',{});return answer
    with monkeypatch.context() as m:
        m.setattr(R,'validate_base',pausing)
        result=R.execute(contract,tmp_path)
    assert result['status']=='paused' and result['stage']=='pool_generation' and result['new_pool_candidates_this_attempt']==1
    assert not (tmp_path/'final_pool.json').exists()
    def counting(base,**kwargs):calls.append((base.seed,base.motif));return original(base,**kwargs)
    with monkeypatch.context() as m:
        m.setattr(R,'validate_base',counting);assert R.execute(contract,tmp_path,max_new=1)['status']=='paused'
    assert len(calls)==2 and len(set(calls))==2


def test_freeze_required_changed_contract_and_concurrent_writer(fixture,tmp_path):
    contract,_=fixture
    with pytest.raises(ValueError,match='freeze before'):R.execute(contract,tmp_path)
    frozen(tmp_path,contract)
    changed=copy.deepcopy(contract);changed['protocol']['sample_seeds']=[9]
    with pytest.raises(ValueError,match='contract changed'):R.execute(changed,tmp_path)
    with run_lock(tmp_path,'.pipeline.lock'):
        with pytest.raises(RuntimeError,match='active'):R.execute(contract,tmp_path)


def test_corrupt_journal_rejected_on_resume(fixture,tmp_path):
    contract,_=fixture;frozen(tmp_path,contract);R.execute(contract,tmp_path,max_new=1)
    with sqlite3.connect(tmp_path/'progress.sqlite3') as db:db.execute("UPDATE items SET value='{}'")
    with pytest.raises(ValueError,match='checksum mismatch'):R.execute(contract,tmp_path)


def test_pool_and_inference_exceptions_are_resumable(fixture,tmp_path,monkeypatch):
    contract,_=fixture;frozen(tmp_path,contract)
    def fail(*args,**kwargs):raise RuntimeError('software injected interruption')
    with monkeypatch.context() as m:
        m.setattr(R,'validate_base',fail)
        with pytest.raises(RuntimeError,match='injected'):R.execute(contract,tmp_path)
    assert R.read(tmp_path/'progress.json')['status']=='failed'
    with monkeypatch.context() as m:
        m.setattr(R,'evaluate_one',fail)
        with pytest.raises(RuntimeError,match='injected'):R.execute(contract,tmp_path)
    assert R.execute(contract,tmp_path)['status']=='completed'
    assert R.timing(tmp_path)['attempts_missing_end_record']==[]


def test_aggregation_learner_weights_scopes_and_paired_ci():
    rows=[]
    for motif in ['closed_ring','layered_core']:
        for seed in range(10):
            for c in ['P','PA']:
                for repeat in range(1+seed%2):
                    score=.60+.01*seed+(.02 if c=='PA' else 0)-(.10 if motif=='layered_core' else 0)
                    rows.append(dict(condition=c,learner_seed=seed,mode='DS',transitions=4194304,base_seed=1,motif=motif,
                                     destruction=score,full_clear=False,town_hall=True,zero_destruction=0,
                                     episode_return=score,spent=200,tick=300,decisions=40))
        rows.append(dict(condition='heuristic',learner_seed=-1,mode='heuristic',transitions=0,base_seed=1,motif=motif,
                         destruction=.4,full_clear=False,town_hall=False,zero_destruction=0,episode_return=.3,spent=199,tick=300,decisions=40))
    options=R.read(R.PACKAGE/'protocol.json')['intervals'];out=summaries(rows,options)
    for r in out['comparisons.csv']:
        assert r['learner_observations']==10
        if r['comparison']=='PA_minus_P':
            assert r['mean_destruction_change_points']==pytest.approx(2)
            assert r['ci95_low_points']==pytest.approx(2) and r['ci95_high_points']==pytest.approx(2)
    row=next(r for r in out['condition_summary.csv'] if r['scope']=='all_final' and r['condition']=='P')
    assert row['destruction_percent']==pytest.approx(59.5)  # Equal learners despite unequal attack counts.
    pairs=[r['destruction_change_points'] for r in out['comparisons_by_learner.csv'] if r['scope']=='all_final' and r['comparison']=='P_minus_heuristic']
    expected=np.quantile(np.array(pairs)[np.random.default_rng(53001).integers(0,10,(20000,10))].mean(axis=1),[.025,.975])
    actual=next(r for r in out['comparisons.csv'] if r['scope']=='all_final' and r['comparison']=='P_minus_heuristic')
    assert [actual['ci95_low_points'],actual['ci95_high_points']]==pytest.approx(expected)


def test_design_identity_counts_seeds_and_no_final_generation():
    protocol=R.read(R.PACKAGE/'protocol.json');ref=R.read(R.PACKAGE/'reference.json')
    contract={'protocol':protocol,'models':ref['models']}
    assert len(R.pool_candidates(contract))==800
    assert protocol['candidate_seed_interval']==[11200,11399]
    assert len(ref['models'])==20 and all(m['transitions']==4194304 for m in ref['models'])
    base=generate_base(10200,'closed_ring')
    keys=[R.unit_key(*u) for u in R.units(contract,[base])]
    assert len(keys)==len(set(keys))==488 and sum(k.startswith('heuristic:') for k in keys)==8
    assert R.acting_seed(41001,base,0)==int.from_bytes(hashlib.sha256(b'v2-final-ds-v1:41001:closed_ring:10200:0').digest()[:4],'big')


def test_freshness_check_blocks_prior_overlap(tmp_path):
    (tmp_path/'artifacts'/'old').mkdir(parents=True);protocol=R.read(R.PACKAGE/'protocol.json')
    atomic_json(tmp_path/'artifacts/old/evaluation_contract.json',{'layouts':[{'seed':11255,'motif':'closed_ring'}]})
    with pytest.raises(ValueError,match='earlier metadata'):R.freshness_check(tmp_path,tmp_path/'artifacts/new',protocol)
    atomic_json(tmp_path/'artifacts/old/evaluation_contract.json',{'layouts':[{'seed':11055,'motif':'closed_ring'}]})
    assert len(R.freshness_check(tmp_path,tmp_path/'artifacts/new',protocol))==1
    atomic_json(tmp_path/'artifacts/old/evaluation_contract.json',{'seed_interval':[11200,11399]})
    with pytest.raises(ValueError,match='earlier metadata'):R.freshness_check(tmp_path,tmp_path/'artifacts/new',protocol)


def test_checkpoint_hash_and_package_tamper_rejected(tmp_path):
    path=tmp_path/'model.zip';path.write_bytes(b'known')
    mapping={'model.zip':R.sha256(path)};R.verify_files(tmp_path,mapping)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='File changed'):R.verify_files(tmp_path,mapping)
    path.unlink()
    with pytest.raises(ValueError,match='Missing required'):R.verify_files(tmp_path,mapping)


def test_missing_timing_is_not_reported_as_complete(tmp_path):
    events=[{'event':'started','attempt_id':'killed'},{'event':'started','attempt_id':'done'},
            {'event':'finished','attempt_id':'done','wall_seconds':10}]
    (tmp_path/'attempts.jsonl').write_text('\n'.join(json.dumps(x) for x in events))
    assert R.timing(tmp_path)['attempts_missing_end_record']==['killed']
    assert R.timing(tmp_path)['measured_finished_attempt_seconds']==10


def test_freeze_is_idempotent_and_rejects_changed_plan(fixture,tmp_path,monkeypatch):
    contract,_=fixture
    with monkeypatch.context() as m:
        m.setattr(R,'prepare',lambda **kw:(contract,{'status':'ready'}))
        m.setattr(R,'generate_base',lambda *a,**k:pytest.fail('Freeze must not generate layouts'))
        R.freeze(tmp_path);first=(tmp_path/'freeze.json').read_bytes();R.freeze(tmp_path)
        assert (tmp_path/'freeze.json').read_bytes()==first
    changed=copy.deepcopy(contract);changed['protocol']['sample_seeds']=[7]
    with monkeypatch.context() as m:
        m.setattr(R,'prepare',lambda **kw:(changed,{}))
        with pytest.raises(ValueError,match='Frozen protocol differs'):R.freeze(tmp_path)
    assert (tmp_path/'freeze.json').read_bytes()==first


def test_preflight_runtime_and_learner_identity_checks(tmp_path,monkeypatch):
    reference=copy.deepcopy(R.read(R.PACKAGE/'reference.json'))
    original_read=R.read
    def read_fixture(path):
        path=Path(path)
        if path.name=='manifest.json':
            run=next(v for v in reference['runs'].values() if path.parent.as_posix().endswith(v['path']))
            return {'status':'completed','actual_steps':4194304,'contract_sha256':run['contract_sha256']}
        if path.name=='contract.json':
            name=path.parent.name;condition,seed=name.split('-')
            return {'settings':{'condition':condition,'seed':int(seed)}}
        return original_read(path)
    current=copy.deepcopy(reference['scientific_runtime'])
    with monkeypatch.context() as m:
        m.setattr(R,'read',read_fixture);m.setattr(R,'verify_files',lambda *a:None)
        m.setattr(R,'runtime_manifest',lambda:current)
        contract,report=R.prepare(root=tmp_path,output=tmp_path/'new')
        assert report['models']==20 and len(contract['models'])==20
        current['packages']['torch']='changed'
        with pytest.raises(ValueError,match='Restore recorded torch'):R.prepare(root=tmp_path,output=tmp_path/'new')
        current['packages']['torch']=reference['scientific_runtime']['packages']['torch']
        current['source_hashes']['env/combat.py']='changed'
        with pytest.raises(ValueError,match='Scientific source_hashes changed'):R.prepare(root=tmp_path,output=tmp_path/'new')

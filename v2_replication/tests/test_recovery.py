"""Tiny software tests; these models are not research results."""
from dataclasses import asdict
import json
import numpy as np
import pytest
import torch
from v2_replication.protocol import read_plan, check_stop, ROOT
from version2.protocol import verify_baseline
from v2_replication import runner
from version2.evaluation import development_check
from train.checkpoints import latest_checkpoint, capture_rng, atomic_json
from experiments.provenance import sha256
from experiments.pools import load_pool


def same_tree(a,b):
    if isinstance(a,torch.Tensor):
        assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):
        assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: same_tree(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): same_tree(x,y)
    else:
        assert a==b


def same_runs(a,b):
    left=runner.RecoverablePPO.load(a/'model.zip',device='cpu')
    right=runner.RecoverablePPO.load(b/'model.zip',device='cpu')
    assert left.num_timesteps==right.num_timesteps
    assert left._n_updates==right._n_updates
    same_tree(left.policy.state_dict(),right.policy.state_dict())
    same_tree(left.policy.optimizer.state_dict(),right.policy.optimizer.state_dict())
    assert (a/'episodes.jsonl').read_bytes()==(b/'episodes.jsonl').read_bytes()
    assert json.loads((a/'development.json').read_text())==json.loads((b/'development.json').read_text())


def tiny_run(path,condition='PA',pause=None,dev=False,workers=1):
    settings=runner.V2Settings(condition=condition,placement_steps=32,joint_steps=160,
        n_envs=workers,rollout_steps=16,batch_size=16,epochs=1,device='cpu')
    path.mkdir()
    inputs={'train':None,'dev':None,'registration':None}
    if dev:
        source=json.loads((ROOT/'bases/dev-v2.json').read_text())
        source['bases']=source['bases'][:1]
        atomic_json(path/'inputs/dev.json',source)
        inputs['dev']={'path':'inputs/dev.json','sha256':sha256(path/'inputs/dev.json')}
    contract={'experiment':'version2-seed-replication','settings':asdict(settings),
        'budgets':runner.phase_budgets(settings),'inputs':inputs,'checkpoint_every_rollouts':1,
        'keep_checkpoints':3,'development_interval':96,'sample_seeds':[31001],
        'milestones':[32,96,192],**runner.checked_runtime()}
    atomic_json(path/'contract.json',contract)
    manifest={'contract_sha256':sha256(path/'contract.json'),'actual_steps':0,
        'evidence_scope':'software test only'}
    return runner._execute(settings,path,contract,manifest,pause_after_steps=pause)


@pytest.mark.parametrize('condition,workers',[('P',1),('PA',2)])
def test_resume_and_retention_match_uninterrupted(tmp_path,condition,workers):
    a,b=tmp_path/'continuous',tmp_path/'resumed'
    tiny_run(a,condition,workers=workers)
    result=tiny_run(b,condition,pause=96,workers=workers)
    assert result['status']=='paused' and result['actual_steps']==96
    before=json.loads((b/'manifest.json').read_text())['attempts']
    runner.resume(b,pause_after_steps=96)
    assert json.loads((b/'manifest.json').read_text())['attempts']==before
    assert runner.resume(b)['status']=='completed'
    same_runs(a,b)
    assert len(list((b/'checkpoints').glob('checkpoint-*')))==3
    assert len(list((b/'milestones').glob('*.zip')))==3
    assert runner.resume(b)['attempts']==json.loads((b/'manifest.json').read_text())['attempts']


def test_sampled_evaluation_survives_resume(tmp_path):
    a,b=tmp_path/'continuous',tmp_path/'resumed'
    tiny_run(a,dev=True)
    tiny_run(b,pause=96,dev=True)
    runner.resume(b)
    same_runs(a,b)
    history=json.loads((b/'development.json').read_text())['records']
    assert {r['mode'] for r in history}=={'sampled','deterministic'}
    assert all(r['episodes']==4 for r in history)


def test_evaluation_restores_rng_and_leaves_learner_unchanged(tmp_path):
    path=tmp_path/'run'
    tiny_run(path)
    model=runner.RecoverablePPO.load(path/'model.zip',device='cpu')
    before=capture_rng()
    policy_before={k:v.clone() for k,v in model.policy.state_dict().items()}
    training_before=model.policy.training
    pool=load_pool(ROOT/'bases/dev-v2.json',required_split='dev')[:1]
    history={'next_step':0,'records':[]}
    development_check(model,pool,history,96,[31001])
    same_tree(before,capture_rng())
    same_tree(policy_before,model.policy.state_dict())
    assert model.policy.training==training_before
    assert history['next_step']>model.num_timesteps


def test_corrupt_latest_uses_retained_prior_update(tmp_path):
    a,b=tmp_path/'continuous',tmp_path/'resumed'
    tiny_run(a)
    tiny_run(b,pause=128)
    checkpoint,_,_=latest_checkpoint(b)
    (checkpoint/'state.pt').write_bytes(b'broken')
    assert latest_checkpoint(b)[1]['actual_steps']<128
    runner.resume(b)
    same_runs(a,b)


def test_resume_rejects_changed_v2_code(tmp_path,monkeypatch):
    path=tmp_path/'run'
    tiny_run(path,pause=96)
    original=runner.checked_runtime
    def changed():
        value=original()
        value['version2_source_hashes']={'changed':'abc'}
        return value
    monkeypatch.setattr(runner,'checked_runtime',changed)
    with pytest.raises(ValueError,match='version2_source_hashes changed'):
        runner.resume(path)


def test_frozen_baseline_and_budget_guards(tmp_path):
    assert len(verify_baseline()['source_hashes'])==28
    plan=read_plan()
    check_stop(1048576,plan)
    for wrong in (0,-1,1000000,4194305):
        with pytest.raises(ValueError): check_stop(wrong,plan)
    with pytest.raises(ValueError,match='Missing:'): verify_baseline(tmp_path)


def test_registered_entrypoint_with_four_workers(tmp_path,monkeypatch):
    # Exercise public registration/start/resume with real frozen pools and four
    # processes, but tiny explicitly test-only budgets. Evaluation is separately
    # exercised above, so avoid repeating the full 192-episode panel here.
    plan={**read_plan(),'placement_steps':32,'total_steps':128,'screen_steps':96,
          'n_envs':4,'rollout_steps':8,'batch_size':32,'epochs':1,
          'development_interval':96,'checkpoint_every_rollouts':1,'milestones':[32,96,128],
          'evidence_scope':'software test only'}
    reg=tmp_path/'records'/'registration.json'
    monkeypatch.setattr(runner,'read_plan',lambda:plan)
    monkeypatch.setattr(runner,'registration_path',lambda:reg)
    monkeypatch.setattr(runner,'run_path',lambda condition,seed:tmp_path/f'{condition}-{seed}')
    monkeypatch.setattr(runner,'development_check',lambda *args:None)
    # Temporary software fixture only; production evidence check is separately tested.
    monkeypatch.setattr(runner,'verify_evidence',lambda: {'software_fixture': True})
    runner.register('cpu')
    before=reg.read_bytes()
    runner.register('cpu')
    assert reg.read_bytes()==before
    with pytest.raises(ValueError,match='Condition/seed'):
        runner.start_or_resume('P',99,96)
    result=runner.start_or_resume('P',3,96)
    assert result['status']=='paused' and result['actual_steps']==96
    result=runner.start_or_resume('P',3,128)
    assert result['status']=='completed' and result['save_reload_verified']
    assert runner.start_or_resume('P',3,128)['attempts']==result['attempts']

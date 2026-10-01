"""Application integration tests. Fixtures never train or evaluate final layouts."""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch
from streamlit.testing.v1 import AppTest
from ui import policy_playback as P
from ui import v2_models as V
from ui.evaluation_tables import evaluation_summary
from env import config as C
from env.generator import generate_base
from ui.custom_bases import custom_spec,layout_from_spec

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def existing_policy():
    from sb3_contrib import MaskablePPO
    candidates=[ROOT/'runs/study-v2/P-0/model.zip',ROOT/'runs/version2-longer-control/P-0/milestones/step-000004194304.zip']
    path=next((p for p in candidates if p.is_file()),None)
    if path is None:pytest.skip('Existing trusted checkpoint required; tests do not train')
    torch.set_num_threads(1)
    model=MaskablePPO.load(path,device='cpu');model.policy.set_training_mode(False)
    return model


@pytest.mark.parametrize('motif,view',[('closed_ring',0),('ew_gates',3),('layered_core',7)])
def test_v2_replay_matches_frozen_evaluator_actions_and_summary(existing_policy,motif,view):
    from v2_final.runner import episode,acting_seed
    base=generate_base(10200,motif)
    target={'condition':'P'}
    expected=episode(target,existing_policy,base,view,41001)
    random.seed(456);np.random.seed(789);torch.manual_seed(123)
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    actual=P.record_policy(existing_policy,base_seed=base.seed,motif=base.motif,spec=base,view=view,
                           action_seed=41001,acting_mode='DS',seed_scheme='v2-final-ds-v1')
    assert actual['actions']==expected['actions'] and actual['summary']==expected['summary']
    assert actual['visual_frames'] and actual['acting']['mode']=='DS'
    assert actual['acting']['rng_seed']==acting_seed(41001,base,view)
    after_numpy=np.random.get_state()
    assert random.getstate()==before[0] and np.array_equal(after_numpy[1],before[1][1])
    assert after_numpy[0]==before[1][0] and after_numpy[2:]==before[1][2:]
    assert torch.equal(torch.get_rng_state(),before[2])
    assert b'streamlit:render' in P.html_bytes(actual)


def test_phase_flags_legal_actions_and_custom_identity():
    class Spy:
        def __init__(self):self.flags=[]
        def predict(self,obs,action_masks,deterministic):
            self.flags.append(deterministic);return int(np.flatnonzero(action_masks)[0]),None
    for mode,wanted in [('DS',[True,False]),('SS',[False,False]),('DD',[True,True])]:
        model=Spy();policy=P.PhaseActingPolicy(model,mode)
        for phase,action in [(0,C.ACTION_FINISH_SHOPPING),(1,C.ACTION_WAIT)]:
            obs={'vector':np.zeros(9)};obs['vector'][6]=phase;mask=np.zeros(C.N_ACTIONS,dtype=bool);mask[action]=True
            assert policy.act(obs,mask)==action
        assert model.flags==wanted
    layout=layout_from_spec(generate_base(42,'layered_core'));base=custom_spec(layout)
    changed=custom_spec(dict(layout,walls=[]))
    assert P.action_rng_seed(base,0,41001,'v2-final-ds-v1')!=P.action_rng_seed(changed,0,41001,'v2-final-ds-v1')
    with pytest.raises(PermissionError):P.record_policy(Spy(),base_seed=11200,motif='closed_ring',acting_mode='DS')


def test_real_v2_custom_replay(existing_policy):
    base=custom_spec(layout_from_spec(generate_base(42,'layered_core')))
    replay=P.record_policy(existing_policy,base_seed=0,motif='custom',spec=base,view=5,action_seed=41001,
                           acting_mode='DS',seed_scheme='v2-final-ds-v1')
    geometry={(r['type'],r['row'],r['col']) for r in replay['snapshots'][0]['buildings']}
    assert geometry==set(base.buildings)|{('wall',r,c) for r,c in base.walls}
    assert replay['summary']['finish_reason'] and replay['visual_frames']


def registry_fixture(root):
    registry=copy.deepcopy(V.read_registry());registry['models']=[];paths=[]
    for condition in ['P','PA']:
        path=root/condition/'milestones/step-000004194304.zip';path.parent.mkdir(parents=True);path.write_bytes(b'test fixture, not a model')
        manifest_path=root/condition/'manifest.json'
        manifest={'status':'completed','settings':{'condition':condition,'seed':0,'mode':'full'},'actual_steps':4194304,
                  'model_sha256':V.digest(path),'evidence_scope':'SOFTWARE FIXTURE'}
        manifest_path.write_text(json.dumps(manifest))
        registry['models'].append({'condition':condition,'learner_seed':0,'transitions':4194304,'run':condition+'-0',
                                  'path':str(path.relative_to(ROOT)),'sha256':V.digest(path),
                                  'manifest_path':str(manifest_path.relative_to(ROOT)),'manifest_sha256':V.digest(manifest_path)})
        paths.append(path)
    return registry,paths


def test_registry_uses_approved_milestones_blocks_tamper_and_public_scan(monkeypatch):
    with tempfile.TemporaryDirectory(dir=ROOT) as temp:
        registry,paths=registry_fixture(Path(temp));monkeypatch.setattr(V,'read_registry',lambda:registry)
        catalog,issues=V.application_catalog(ROOT,[],public=False)
        assert not issues and [p for p,_ in catalog]==paths
        assert all(m['application_version']=='V2' and m['actual_steps']==4194304 for _,m in catalog)
        assert V.application_catalog(ROOT,[],public=True)==([],[])
        (ROOT/registry['models'][0]['manifest_path']).write_text('{}')
        catalog,issues=V.application_catalog(ROOT,[],public=False)
        assert len(catalog)==1 and issues


def test_published_v2_export_is_recognised_without_local_runs(tmp_path,monkeypatch):
    registry=V.read_registry();target=registry['models'][0];path=tmp_path/'model.zip';path.write_bytes(b'export fixture')
    manifest={'status':'completed','settings':{'condition':target['condition'],'seed':target['learner_seed'],'mode':'full'},
              'actual_steps':target['transitions'],'source_model_sha256':target['sha256'],'model_sha256':V.digest(path)}
    path.with_name('manifest.json').write_text(json.dumps(manifest))
    catalog,issues=V.application_catalog(tmp_path,[path],public=True)
    assert not issues and len(catalog)==1 and catalog[0][1]['application_version']=='V2'
    manifest['settings']['seed']=123;path.with_name('manifest.json').write_text(json.dumps(manifest))
    assert V.application_catalog(tmp_path,[path],public=True)[1]


def test_v2_controls_default_ds_stale_replay_and_results(monkeypatch):
    import ui.deployment as D
    class TestPolicy:
        def predict(self,obs,action_masks,deterministic):
            legal=np.flatnonzero(action_masks)
            return int(legal[0] if deterministic else np.random.choice(legal)),None
    with tempfile.TemporaryDirectory(dir=ROOT) as temp:
        registry,paths=registry_fixture(Path(temp));monkeypatch.setattr(V,'read_registry',lambda:registry)
        monkeypatch.setattr(D,'load_cpu_policy',lambda *a,**kw:TestPolicy())
        original=D.asset_paths
        monkeypatch.setattr(D,'asset_paths',lambda root,kind,public:[] if kind=='model' else original(root,kind,public=public))
        app=AppTest.from_file(str(ROOT/'ui/app.py')).run(timeout=30)
        def select(label,value):
            next(w for w in app.selectbox if w.label==label).set_value(value).run(timeout=30);assert not app.exception
        select('Mode','Policy replay')
        assert next(w for w in app.selectbox if w.label=='Model version').value=='V2'
        assert next(w for w in app.selectbox if w.label=='Action selection').value.startswith('DS')
        assert next(w for w in app.number_input if w.label=='Action seed').value==41001
        next(b for b in app.button if b.label=='Generate replay').click().run(timeout=30)
        assert not app.exception and app.session_state['replay']['acting']['mode']=='DS'
        assert app.session_state['replay']['policy']['model_version']=='V2'
        select('Action selection','Sampled');assert 'replay' not in app.session_state
        select('Condition','PA')
        next(b for b in app.button if b.label=='Generate replay').click().run(timeout=30)
        assert app.session_state['replay']['acting']['mode']=='SS'
        assert app.session_state['replay']['policy']['name'].startswith('V2 PA PPO')
        assert len(app.get('download_button'))==2
        # V2 episode table has episode_return and no evidence_scope.
        table=Path(temp)/'episodes.csv'
        table.write_text('condition,learner_seed,mode,motif,destruction,full_clear,town_hall,episode_return\nP,0,DS,closed_ring,0.5,False,True,0.4\n')
        monkeypatch.setattr(D,'asset_paths',lambda root,kind,public:[table] if kind=='evaluation' else [] if kind in ('model','manifest') else original(root,kind,public=public))
        select('Mode','Experiment results');assert app.dataframe


def test_result_summary_handles_v2_modes_equal_learners_and_legacy():
    data=pd.DataFrame([dict(condition='P',learner_seed=0,mode='DS',destruction=.2,episode_return=.1)]*3+
                      [dict(condition='P',learner_seed=1,mode='DS',destruction=.8,episode_return=.7),
                       dict(condition='P',learner_seed=0,mode='DD',destruction=.1,episode_return=0)])
    out=evaluation_summary(data)
    assert out.loc[('P','DS'),'destruction']==pytest.approx(.5) and out.loc[('P','DS'),'return']==pytest.approx(.4)
    assert out.loc[('P','DD'),'destruction']==pytest.approx(.1)
    old=pd.DataFrame([{'condition':'heuristic','destruction':.4,'return':.3}]);assert evaluation_summary(old).loc['heuristic','return']==.3


def test_v2_source_guard_detects_movement_change(monkeypatch):
    registry=V.read_registry();m={'model_sha256':'fixture','application_version':'V2','config':registry['config'],'source_hashes':registry['source_hashes']}
    P.validate_compatibility(m)
    m['source_hashes']={**m['source_hashes'],'env/combat.py':'changed'}
    with pytest.raises(ValueError,match='Scientific source'):P.validate_compatibility(m)

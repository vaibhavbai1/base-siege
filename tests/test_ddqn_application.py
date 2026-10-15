"""Application checks use synthetic weights, never project performance evidence."""
import json
from pathlib import Path
import random
import numpy as np
import pytest
import torch
from streamlit.testing.v1 import AppTest
from ddqn_experiment.core import QNetwork
from ddqn_experiment.runner import sources, runtime
from env.generator import generate_base
from env.siege_env import SiegeEnv
from env.symmetry import SymmetryWrapper
from experiments.provenance import sha256
from experiments.rollout import play_episode, load_replay, save_replay
from ui import ddqn_models as D
from ui.policy_playback import record_policy, html_bytes
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def policy():
    torch.set_num_threads(1)
    env=SiegeEnv();net=QNetwork(env.observation_space);env.close()
    with torch.no_grad():
        for p in net.parameters():p.zero_()
    net.eval()
    return D.DDQNPolicy(net)


def test_selected_model_integrity_and_frozen_preference(tmp_path,monkeypatch,policy):
    from ddqn_experiment.evaluation import GreedyPolicy
    frozen=tmp_path/D.CANDIDATES[0];(frozen/'policies').mkdir(parents=True)
    (frozen/'contract.json').write_text(json.dumps({'sources':sources(),'runtime':runtime(),'seed':100,'mode':'pilot'}))
    path=frozen/'policies'/f'step-{D.STEPS:012d}.pt'
    torch.save({'algorithm':'masked-double-dqn','steps':D.STEPS,'online':policy.network.state_dict()},path)
    path.with_suffix('.json').write_text(json.dumps({'sha256':sha256(path),'contract_sha256':sha256(frozen/'contract.json')}))
    monkeypatch.setattr(D,'POLICY_SHA256',sha256(path));monkeypatch.setattr(D,'CONTRACT_SHA256',sha256(frozen/'contract.json'))
    assert D.selected_checkpoint(tmp_path)[0]==path
    loaded=D.load_selected_policy(path)
    env=SiegeEnv();obs,_=env.reset(seed=42);mask=env.action_masks()
    assert loaded.predict(obs,action_masks=mask)[0]==GreedyPolicy(policy.network).act(obs,mask)
    env.close()
    with path.open('ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError,match='differs'):D.load_selected_policy(path)


def test_recording_matches_evaluation_and_preserves_rng(policy,tmp_path):
    from ddqn_experiment.evaluation import GreedyPolicy
    base=generate_base(10200,'closed_ring');env=SymmetryWrapper(SiegeEnv(),transform=5)
    expected=play_episode(env,GreedyPolicy(policy.network),seed=0,options={'spec':base});env.close()
    random.seed(913);np.random.seed(217);torch.manual_seed(329)
    states=(random.getstate(),np.random.get_state(),torch.random.get_rng_state())
    actual=record_policy(policy,base_seed=base.seed,motif=base.motif,spec=base,view=5,deterministic=True)
    assert actual['actions']==expected['actions'] and actual['summary']==expected['summary']
    assert actual['visual_frames'] and b'streamlit:render' in html_bytes(actual)
    assert states[0]==random.getstate() and np.array_equal(states[1][1],np.random.get_state()[1])
    assert torch.equal(states[2],torch.random.get_rng_state())
    save_replay(actual,tmp_path/'replay.json')
    assert load_replay(tmp_path/'replay.json')['actions']==expected['actions']
    with pytest.raises(ValueError,match='greedy'):policy.predict({},action_masks=[],deterministic=False)


def test_custom_base_and_reserved_seed_guard(policy):
    from ui.custom_bases import custom_spec,layout_from_spec
    base=custom_spec(layout_from_spec(generate_base(42,'layered_core')))
    replay=record_policy(policy,base_seed=base.seed,motif='custom',spec=base,view=7,deterministic=True)
    assert replay['summary']['motif']=='custom' and replay['visual_frames']
    with pytest.raises(PermissionError):record_policy(policy,base_seed=11200,motif='layered_core',deterministic=True)


def test_ui_selection_downloads_and_stale_replay(policy,monkeypatch):
    path=ROOT/'runs/ddqn/test-fixture/policies'/f'step-{D.STEPS:012d}.pt'
    details={'model_sha256':'TEST FIXTURE','actual_steps':D.STEPS,'training_contract_sha256':'TEST FIXTURE',
             'settings':{'condition':'DDQN','seed':100,'mode':'pilot'}}
    monkeypatch.setattr(D,'selected_checkpoint',lambda root:(path,details))
    monkeypatch.setattr(D,'load_selected_policy',lambda path:policy)
    app=AppTest.from_file(str(ROOT/'ui/app.py')).run(timeout=30)
    def select(label,value):
        next(w for w in app.selectbox if w.label==label).set_value(value).run(timeout=30)
        assert not app.exception
    select('Mode','Policy replay');select('Policy','Saved Masked DDQN')
    assert not any(w.label=='Action seed' for w in app.number_input)
    next(b for b in app.button if b.label=='Generate replay').click().run(timeout=30)
    assert not app.exception
    replay=app.session_state['replay']
    assert replay['policy']['model_version']=='DDQN' and replay['policy']['algorithm']=='masked-double-dqn'
    assert replay['policy']['training_transitions']==D.STEPS and replay['acting']['mode']=='greedy-legal'
    assert replay['visual_frames'] and len(app.get('download_button'))==2
    select('Agent view',1);assert 'replay' not in app.session_state
    select('Policy','Heuristic')
    next(b for b in app.button if b.label=='Generate replay').click().run(timeout=30)
    assert not app.exception and app.session_state['replay']['policy']['name']=='Heuristic'


def test_missing_checkpoint_disables_generation(monkeypatch):
    monkeypatch.setattr(D,'selected_checkpoint',lambda root:(None,{}))
    app=AppTest.from_file(str(ROOT/'ui/app.py')).run(timeout=30)
    next(w for w in app.selectbox if w.label=='Mode').set_value('Policy replay').run(timeout=30)
    next(w for w in app.selectbox if w.label=='Policy').set_value('Saved Masked DDQN').run(timeout=30)
    assert not app.exception
    assert next(b for b in app.button if b.label=='Generate replay').disabled


def test_public_app_does_not_import_local_ddqn_adapter(monkeypatch):
    import builtins
    original=builtins.__import__
    def guarded(name,*a,**kw):
        if name=='ui.ddqn_models' or name.startswith('ddqn_experiment'):
            raise AssertionError('Public bundle attempted to import a private training dependency')
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',guarded)
    app=AppTest.from_file(str(ROOT/'ui/streamlit_app.py')).run(timeout=30)
    next(w for w in app.selectbox if w.label=='Mode').set_value('Policy replay').run(timeout=30)
    assert not app.exception
    assert 'Saved Masked DDQN' not in next(w for w in app.selectbox if w.label=='Policy').options

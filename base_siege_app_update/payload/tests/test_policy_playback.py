"""Playback contract tests use the real simulator with a lightweight test policy."""
from contextlib import nullcontext
import hashlib
import json
import random
import sys
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from env.generator import generate_base
from env.siege_env import SiegeEnv
from env.symmetry import SymmetryWrapper
from experiments.rollout import play_episode
from ui import policy_playback as P


class TestPolicy:
    __test__ = False
    def __init__(self): self.calls=[]
    def predict(self, obs, action_masks, deterministic):
        legal=np.flatnonzero(action_masks)
        self.calls.append(deterministic)
        weights=1+legal%7
        return int(legal[weights.argmax()] if deterministic else np.random.choice(legal,p=weights/weights.sum())),None


@pytest.fixture
def torch_fixture(monkeypatch):
    # No inference claim: only simulate the Torch context for this test policy.
    fake=SimpleNamespace(are_deterministic_algorithms_enabled=lambda:False,
        is_deterministic_algorithms_warn_only_enabled=lambda:False,
        set_num_threads=lambda n:None,use_deterministic_algorithms=lambda *a,**kw:None,
        random=SimpleNamespace(fork_rng=lambda **kw:nullcontext(),default_generator=SimpleNamespace(manual_seed=lambda s:None)))
    monkeypatch.setitem(sys.modules,'torch',fake)


def test_recording_matches_evaluation_and_restores_rng(torch_fixture):
    base=generate_base(10200,'closed_ring');seed=P.action_rng_seed(base,3,2)
    np.random.seed(seed);random.seed(seed)
    env=SymmetryWrapper(SiegeEnv(motifs=('closed_ring',)),transform=3)
    expected=play_episode(env,P.ActingPolicy(TestPolicy(),False),seed=0,options={'spec':base});env.close()
    random.seed(654);np.random.seed(321)
    saved_random=random.getstate();saved_numpy=np.random.get_state()
    policy=TestPolicy()
    actual=P.record_policy(policy,base_seed=base.seed,motif=base.motif,view=3,action_seed=2,spec=base)
    assert actual['actions']==expected['actions']
    assert actual['summary']==expected['summary']
    assert not any(policy.calls)
    assert random.getstate()==saved_random
    assert np.array_equal(np.random.get_state()[1],saved_numpy[1])
    assert actual['visual_frames'] and actual['snapshots']
    assert b'streamlit:render' in P.html_bytes(actual)


def test_deterministic_forwarding_and_reserved_seed_rejection(torch_fixture):
    model=TestPolicy()
    P.record_policy(model,base_seed=42,motif='closed_ring',deterministic=True)
    assert model.calls and all(model.calls)
    with pytest.raises(PermissionError):P.record_policy(model,base_seed=11000,motif='closed_ring')
    layered=P.record_policy(model,base_seed=42,motif='layered_core')
    assert layered['summary']['motif']=='layered_core'


def test_multiple_sessions_remain_reproducible(torch_fixture):
    def play(seed):return P.record_policy(TestPolicy(),base_seed=42,motif='closed_ring',action_seed=seed)['actions']
    expected=[play(s) for s in [0,1,2,0]]
    with ThreadPoolExecutor(max_workers=2) as executor:actual=list(executor.map(play,[0,1,2,0]))
    assert actual==expected and actual[0]==actual[3]


def test_model_catalog_and_configuration_guard(tmp_path):
    paths=[]
    for mode in ['pilot','full']:
        path=tmp_path/mode/'model.zip';path.parent.mkdir();path.write_bytes(b'fixture')
        path.with_name('manifest.json').write_text(json.dumps({'status':'completed','settings':{'condition':'P','seed':0,'mode':mode}}))
        paths.append(path)
    assert P.model_catalog(paths)[0][1]['settings']['mode']=='full'
    with pytest.raises(ValueError,match='checksum'):P.validate_compatibility({})
    with pytest.raises(ValueError,match='configuration'):P.validate_compatibility({'model_sha256':'fixture','config':{}})


def test_trained_policy_controls_and_stale_replay_clear(tmp_path,monkeypatch,torch_fixture):
    from pathlib import Path
    import ui.deployment as deployment
    root=Path(__file__).resolve().parents[1]
    paths=[]
    # Fixtures live under root so the UI's displayed relative paths are real.
    import tempfile
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        for condition in ['P','PA']:
            path=Path(temporary)/condition/'model.zip';path.parent.mkdir();path.write_bytes(b'fixture')
            manifest={'status':'completed','settings':{'condition':condition,'seed':0,'mode':'full'},
                      'model_sha256':hashlib.sha256(b'fixture').hexdigest(),'actual_steps':262144,'evidence_scope':'TEST FIXTURE'}
            path.with_name('manifest.json').write_text(json.dumps(manifest));paths.append(path)
        original=deployment.asset_paths
        monkeypatch.setattr(deployment,'asset_paths',lambda root,kind,public:paths if kind=='model' else original(root,kind,public=public))
        monkeypatch.setattr(deployment,'load_cpu_policy',lambda *a,**kw:TestPolicy())
        app=AppTest.from_file(str(root/'ui/app.py')).run(timeout=30)
        def select(label,value):
            next(w for w in app.selectbox if w.label==label).set_value(value).run(timeout=30)
            assert not app.exception
        select('Mode','Policy replay')
        assert next(w for w in app.selectbox if w.label=='Action selection').value=='Sampled'
        next(b for b in app.button if b.label=='Generate replay').click().run(timeout=30)
        assert not app.exception and app.session_state['replay']['acting']['mode']=='sampled'
        assert len(app.get('download_button'))==2
        next(w for w in app.number_input if w.label=='Action seed').set_value(1).run(timeout=30)
        assert 'replay' not in app.session_state
        select('Condition','PA');select('Action selection','Deterministic')
        next(b for b in app.button if b.label=='Generate replay').click().run(timeout=30)
        assert not app.exception
        assert app.session_state['replay']['acting']['mode']=='deterministic'
        assert app.session_state['replay']['policy']['name'].startswith('PA PPO')


def test_custom_policy_views_keep_original_geometry_and_rng_identity(torch_fixture):
    from ui.custom_bases import layout_from_spec, custom_spec
    layout = layout_from_spec(generate_base(42, 'layered_core'))
    base = custom_spec(layout)
    expected = set(base.buildings) | {('wall',r,c) for r,c in base.walls}
    for view in range(8):
        replay = P.record_policy(TestPolicy(), base_seed=0, motif='custom', spec=base, view=view)
        initial = replay['snapshots'][0]
        assert {(b['type'],b['row'],b['col']) for b in initial['buildings']} == expected
        assert replay['summary']['motif'] == 'custom' and replay['summary']['finish_reason']
        assert replay['acting']['view'] == view
    changed = dict(layout, walls=[])
    assert P.action_rng_seed(base,0,0) != P.action_rng_seed(custom_spec(changed),0,0)
    with pytest.raises(ValueError):
        P.record_policy(TestPolicy(), base_seed=0, motif='custom')

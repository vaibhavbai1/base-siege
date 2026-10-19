"""Short inference checks on development/example bases; no training or benchmark runs."""
from pathlib import Path
import hashlib
import json
import random
import shutil

import numpy as np
import pytest
import torch
from streamlit.testing.v1 import AppTest
from env.generator import generate_base
from env.siege_env import SiegeEnv
from env.symmetry import SymmetryWrapper
from experiments.rollout import play_episode, save_replay, load_replay
from ui import trpo_models as T
from ui.custom_bases import custom_spec, layout_from_spec
from ui.policy_playback import record_policy, isolated_rng, action_rng_seed, html_bytes
from v2_phases.runner import PhasePolicy

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def policy():
    path, _ = T.selected_checkpoint(ROOT)
    assert path is not None, 'Install the included T-1 checkpoint before running these checks.'
    return T.load_selected_policy(path)


@pytest.mark.parametrize('mode', ['SS', 'DS', 'DD'])
def test_actual_checkpoint_replay_matches_phase_evaluator(policy, mode):
    base = generate_base(10200, 'closed_ring')
    seed = int.from_bytes(hashlib.sha256(b'v3-final-ds-v1:41001:closed_ring:10200:3').digest()[:4], 'big')
    assert action_rng_seed(base, 3, 41001, 'v3-final-ds-v1') == seed
    with isolated_rng(seed):
        env = SymmetryWrapper(SiegeEnv(), transform=3)
        try:
            expected = play_episode(env, PhasePolicy(policy.model.policy, mode), seed=0, options={'spec': base})
        finally:
            env.close()
    random.seed(456); np.random.seed(789); torch.manual_seed(123)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    actual = record_policy(policy, base_seed=base.seed, motif=base.motif, spec=base,
                           view=3, action_seed=41001, acting_mode=mode, seed_scheme='v3-final-ds-v1')
    assert actual['actions'] == expected['actions']
    assert actual['summary'] == expected['summary']
    assert actual['visual_frames']
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])


def test_custom_base_export_and_reserved_seed_guard(policy, tmp_path):
    layout = layout_from_spec(generate_base(42, 'layered_core'), name='TRPO custom check')
    base = custom_spec(layout)
    replay = record_policy(policy, base_seed=0, motif='custom', spec=base, view=7,
                           action_seed=41001, acting_mode='SS', seed_scheme='v3-final-ds-v1')
    initial = {(b['type'], b['row'], b['col']) for b in replay['snapshots'][0]['buildings']}
    assert initial == set(base.buildings) | {('wall', r, c) for r, c in base.walls}
    assert replay['summary']['motif'] == 'custom' and replay['visual_frames']
    save_replay(replay, tmp_path/'replay.json')
    assert load_replay(tmp_path/'replay.json')['actions'] == replay['actions']
    assert b'streamlit:render' in html_bytes(replay)
    with pytest.raises(PermissionError):
        record_policy(policy, base_seed=11200, motif='closed_ring', acting_mode='SS', seed_scheme='v3-final-ds-v1')


def test_tamper_is_rejected_before_load(tmp_path):
    path, _ = T.selected_checkpoint(ROOT)
    for name in T.CHECKSUMS:
        shutil.copy2(path.with_name(name), tmp_path/name)
    T.validate_checkpoint(tmp_path/'model.zip')
    with (tmp_path/'model.zip').open('ab') as f:
        f.write(b'altered')
    with pytest.raises(ValueError, match='checksum'):
        T.load_selected_policy(tmp_path/'model.zip')


def test_real_ui_generated_and_custom_replay(policy):
    app = AppTest.from_file(str(ROOT/'ui/app.py')).run(timeout=30)
    def select(label, value):
        next(w for w in app.selectbox if w.label == label).set_value(value).run(timeout=30)
        assert not app.exception
    def generate():
        next(w for w in app.button if w.label == 'Generate replay').click().run(timeout=30)
        assert not app.exception
        return app.session_state['replay']
    select('Mode', 'Policy replay'); select('Policy', 'Saved Maskable TRPO')
    assert next(w for w in app.selectbox if w.label == 'Action selection').value.startswith('SS')
    replay = generate()
    assert replay['policy']['algorithm'] == 'MaskableTRPO'
    assert replay['policy']['learner_seed'] == 1
    assert replay['policy']['training_transitions'] == T.STEPS
    assert replay['acting']['mode'] == 'SS'
    assert len(app.get('download_button')) == 2
    select('Action selection', 'DS — deterministic purchasing, sampled deployment')
    assert 'replay' not in app.session_state
    assert generate()['acting']['mode'] == 'DS'
    app.session_state['custom_layout'] = layout_from_spec(generate_base(42, 'closed_ring'), 'Test custom')
    select('Play on', 'My custom base')
    assert generate()['policy']['custom_base']['name'] == 'Test custom'
    select('Agent view', 4)
    assert 'replay' not in app.session_state


def test_missing_checkpoint_disables_replay(monkeypatch):
    monkeypatch.setattr(T, 'selected_checkpoint', lambda root: (None, {}))
    app = AppTest.from_file(str(ROOT/'ui/app.py')).run(timeout=30)
    next(w for w in app.selectbox if w.label == 'Mode').set_value('Policy replay').run(timeout=30)
    next(w for w in app.selectbox if w.label == 'Policy').set_value('Saved Maskable TRPO').run(timeout=30)
    assert not app.exception
    assert next(w for w in app.button if w.label == 'Generate replay').disabled


def test_public_entrypoint_stays_independent_of_private_trpo(monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith('trpo') or name == 'ui.trpo_models' or (name == 'ui' and 'trpo_models' in fromlist):
            raise AssertionError('Public entrypoint loaded a local TRPO dependency')
        return original(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, '__import__', guarded)
    app = AppTest.from_file(str(ROOT/'ui/streamlit_app.py')).run(timeout=30)
    assert not app.exception

"""Software checks use an existing original checkpoint; no training occurs."""
import copy
import json
import random
import zipfile

import numpy as np
import pytest
import torch
from env.siege_env import SiegeEnv
from env.symmetry import SymmetryWrapper
from experiments.rollout import play_episode
from version2.evaluation import ActingPolicy
from v2_replays import runner as R
from v2_replays.selection import key, select_cases


@pytest.fixture(scope='module')
def fixture():
    R.E.configure_cpu()
    base = R.E.load_pool(R.ROOT/'bases/dev-v2.json', required_split='dev')[0]
    path = 'runs/study-v2/P-0/model.zip'
    target = {'run': 'P-0', 'condition': 'P', 'learner_seed': 0, 'transitions': 262144,
              'path': path, 'sha256': R.E.sha256(R.ROOT/path)}
    model = R.E.load_target(target, R.ROOT)
    cases = []
    for mode, label in [('sampled', 31001), ('deterministic', 0)]:
        expected = R.E.evaluate_one(target, model, base, 3, mode, label, set())
        cases.append({'id': f'case-{len(cases)+1:03d}', 'roles': ['software test'], 'expected': expected})
    plan = {'scope': 'Software tests only', 'cases': cases,
            'pairs': [{'condition': 'P', 'learner_seed': 0, 'selection_role': 'software test',
                       'sampled_id': cases[0]['id'], 'deterministic_id': cases[1]['id']}]}
    return {'scope': 'Software tests only', 'models': [target]}, plan, [base], model


@pytest.mark.parametrize('case_index', [0, 1])
def test_passive_recording_preserves_actions_metrics_and_rng(fixture, case_index):
    _, plan, pool, model = fixture
    case = plan['cases'][case_index]
    expected = case['expected']
    seed = expected['acting_seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = SymmetryWrapper(SiegeEnv(motifs=R.C.DEV_MOTIFS), transform=3)
    plain = play_episode(env, ActingPolicy(model.policy, expected['mode']=='deterministic'), options={'spec': pool[0]})
    env.close()
    rng = torch.get_rng_state().clone()
    recorded = R.record(case, model, pool[0])
    assert recorded['actions'] == plain['actions']
    assert recorded['summary'] == plain['summary']
    assert torch.equal(torch.get_rng_state(), rng)
    assert {f['tick'] for f in recorded['visual_frames']} == set(range(plain['summary']['tick']+1))
    diagnostic = R.diagnostics(recorded)
    assert sum(diagnostic['purchased_army'][t]*R.C.TROOP_COSTS[t] for t in R.C.TROOP_TYPES) == expected['spent']
    for t in R.C.TROOP_TYPES:
        assert diagnostic['purchased_army'][t] == diagnostic['deployed_army'][t]+diagnostic['reserve_at_end'][t]
    assert all(r['base_action'] == int(env.inverse[r['view_action']]) for r in diagnostic['timeline'])


def test_changed_reference_is_rejected(fixture):
    _, plan, pool, model = fixture
    case = copy.deepcopy(plan['cases'][0])
    case['expected']['destruction'] += .1
    with pytest.raises(ValueError, match='Replay mismatch'):
        R.record(case, model, pool[0])
    case = copy.deepcopy(plan['cases'][0])
    case['expected']['acting_seed'] += 1
    with pytest.raises(ValueError, match='Acting-seed'):
        R.record(case, model, pool[0])


def test_resume_exports_and_pack_are_identical(tmp_path, fixture):
    contract, plan, pool, _ = fixture
    a, b = tmp_path/'a', tmp_path/'b'
    receipt = R.execute(contract, plan, pool, a)
    assert R.execute(contract, plan, pool, b, max_cases=1)['status'] == 'paused'
    assert R.execute(contract, plan, pool, b) == receipt
    assert R.execute(contract, plan, pool, b) == receipt
    for name in receipt['files']:
        assert (a/name).read_bytes() == (b/name).read_bytes()
    packed = R.pack(b)
    assert R.pack(b).read_bytes() == packed.read_bytes()
    with zipfile.ZipFile(packed) as archive:
        assert archive.testzip() is None
        assert 'version2-replay-inspection/index.html' in archive.namelist()
    page = (b/'cases/case-001.html').read_text()
    assert 'streamlit:render' in page and 'manual_engine.js' not in page
    assert '<script src=' not in page and '<link rel="stylesheet"' not in page
    assert 'case-001.details.html' in (b/'index.html').read_text()


def test_corrupt_saved_replay_is_rejected(tmp_path, fixture):
    contract, plan, pool, _ = fixture
    R.execute(contract, plan, pool, tmp_path)
    (tmp_path/'cases/case-001.json').write_text('{}')
    with pytest.raises(ValueError, match='Existing replay output changed'):
        R.execute(contract, plan, pool, tmp_path)
    with pytest.raises(ValueError, match='Output checksum mismatch'):
        R.pack(tmp_path)


def test_selection_is_unique_and_pairs_exact_layouts():
    plan = R.read(R.PACKAGE/'selection_plan.json')
    assert len(plan['cases']) == 48 and len(plan['pairs']) == 24
    assert len({key(c['expected']) for c in plan['cases']}) == 48
    by_id = {c['id']: c for c in plan['cases']}
    for pair in plan['pairs']:
        a, b = by_id[pair['sampled_id']]['expected'], by_id[pair['deterministic_id']]['expected']
        for field in ('condition', 'learner_seed', 'transitions', 'base_seed', 'motif', 'view'):
            assert a[field] == b[field]
        assert a['mode'] == 'sampled' and b['mode'] == 'deterministic'
    assert any(c['expected']['destruction'] == 0 for c in plan['cases'])
    assert all(c['expected']['transitions'] == 4194304 for c in plan['cases'])


def test_result_checksum_failure_stops_preflight(tmp_path, monkeypatch):
    package, results = tmp_path/'package', tmp_path/'results'
    package.mkdir(); results.mkdir()
    (results/'episodes.csv').write_text('modified')
    (package/'reviewed_results.json').write_text(json.dumps({'files': {'episodes.csv': '0'*64}}))
    monkeypatch.setattr(R, 'PACKAGE', package)
    monkeypatch.setattr(R, 'RESULTS', results)
    with pytest.raises(ValueError, match='Reviewed result changed'):
        R.prepare()

"""Bounded checks; synthetic weights/results below are never research evidence."""
import csv
from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest
import torch

from ddqn_experiment.core import QNetwork, Settings
from ddqn_experiment.runner import make_contract
from ddqn_experiment.runner import checked_pool as checked_dev_pool
from env.siege_env import SiegeEnv
from experiments.journal import open_journal
from experiments.provenance import sha256
from experiments.storage import publish_json
from ddqn_final_benchmark import runner as r


@pytest.fixture
def network():
    r.configure(0, 'cpu')
    env = SiegeEnv()
    try:
        net = QNetwork(env.observation_space)
    finally:
        env.close()
    # Constant tied legal values deliberately test the deterministic tie rule.
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
    net.eval()
    return net


def test_reference_archive_and_training_access_guard():
    bases = r.checked_benchmark(r.ROOT)
    assert len(bases) == 235
    with pytest.raises((PermissionError, ValueError)):
        checked_dev_pool(r.ROOT/r.ARCHIVE/'final_pool.json', 'dev')


def test_changed_pool_is_rejected(tmp_path):
    out = tmp_path/r.ARCHIVE
    out.mkdir(parents=True)
    for name in r.read(r.PACKAGE/'reference.json')['benchmark_files']:
        if name != 'final_pool.json':
            (out/name).symlink_to(r.ROOT/r.ARCHIVE/name)
    (out/'final_pool.json').write_text('{}')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        r.checked_benchmark(tmp_path)


def test_interrupted_episode_resume_does_not_repeat_committed_attack(tmp_path, network, monkeypatch):
    bases = checked_dev_pool(r.ROOT/'bases/dev-v2.json', 'dev')[:1]
    contract = {'views': [0, 1]}
    original = r.play_episode
    calls = []
    def interrupt_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return original(*args, **kwargs)
    monkeypatch.setattr(r, 'play_episode', interrupt_second)
    with pytest.raises(KeyboardInterrupt):
        r.run_units(network, bases, contract, tmp_path)
    with open_journal(tmp_path/'progress.sqlite3', contract, resume=True) as journal:
        saved = dict(journal.rows())
        assert len(saved) == 1
    calls.clear()
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(r, 'play_episode', counted)
    rows = r.run_units(network, bases, contract, tmp_path, resume=True)
    assert len(rows) == 2 and len(calls) == 1
    assert next(row for row in rows if row['view'] == 0) == next(iter(saved.values()))
    assert r.run_units(network, bases, contract, tmp_path, resume=True) == rows
    assert len(calls) == 1
    with pytest.raises(ValueError, match='contract changed'):
        r.run_units(network, bases, {'views': [0]}, tmp_path, resume=True)


def test_freeze_copies_exact_policy_and_refuses_tampering(tmp_path, network, monkeypatch):
    package = tmp_path/'package'
    package.mkdir()
    ref = r.read(r.PACKAGE/'reference.json')
    run = tmp_path/'synthetic-run'
    (run/'policies').mkdir(parents=True)
    contract = make_contract(replace(Settings(), learning_rate=3e-5), 100, 'pilot', 'cpu',
                             r.ROOT/'bases/train-v2.json', r.ROOT/'bases/dev-v2.json')
    publish_json(run/'contract.json', contract)
    policy = run/'policies'/f'{r.POLICY_NAME}.pt'
    torch.save({'algorithm': 'masked-double-dqn', 'steps': r.STEPS, 'online': network.state_dict()}, policy)
    publish_json(policy.with_suffix('.json'), {'sha256': sha256(policy), 'contract_sha256': sha256(run/'contract.json')})
    ref['policy_sha256'] = sha256(policy)
    ref['training_contract_sha256'] = sha256(run/'contract.json')
    publish_json(package/'reference.json', ref)
    monkeypatch.setattr(r, 'PACKAGE', package)
    out = tmp_path/'frozen'
    r.freeze(run, out)
    first = (out/'freeze.json').read_bytes()
    r.freeze(run, out)
    assert (out/'freeze.json').read_bytes() == first
    assert (out/'selected_model'/'policies'/policy.name).read_bytes() == policy.read_bytes()
    # Check export/receipt/resume wiring without evaluating a research checkpoint.
    synthetic_rows = [
        {'base_seed': b.seed, 'motif': b.motif, 'view': v, 'destruction': 0,
         'full_clear': False, 'town_hall': False, 'episode_return': -.3, 'spent': 200,
         'tick': 300, 'decisions': 56, 'finish_reason': 'tick_limit', 'measured_episode_seconds': 0}
        for b in r.checked_benchmark(r.ROOT) for v in range(8)]
    monkeypatch.setattr(r, 'run_units', lambda *a, **kw: synthetic_rows)
    r.evaluate(out)
    r.evaluate(out, resume=True)
    assert r.read(out/'complete.json')['episodes'] == 1880
    review = tmp_path/'review.zip'
    r.pack(out, review)
    with r.zipfile.ZipFile(review) as archive:
        assert 'summary.json' in archive.namelist()
        assert not any(n.endswith('.pt') for n in archive.namelist())
    with policy.open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        r.freeze(run, out)
    # Original run mutation cannot change the already frozen copy.
    r.selected_policy(out/'selected_model')
    with (out/'selected_model'/'policies'/policy.name).open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        r.selected_policy(out/'selected_model')


def synthetic_archive(path, duplicate=False, missing=False):
    rows = []
    for c in ('P', 'PA', 'heuristic'):
        for s in ([-1] if c == 'heuristic' else range(10)):
            for label in ([0] if c == 'heuristic' else (41001, 41002, 41003)):
                rows.append({'condition': c, 'learner_seed': s, 'sample_seed': label,
                             'motif': 'closed_ring', 'base_seed': 11200, 'view': 0,
                             'mode': 'heuristic' if c == 'heuristic' else 'DS', 'transitions': 4194304,
                             'destruction': .2 if c == 'heuristic' else .6,
                             'full_clear': 'False', 'town_hall': 'False', 'episode_return': .1, 'spent': 200})
    if duplicate:
        rows.append(rows[0])
    if missing:
        rows.pop()
    path.write_bytes(r.csv_payload(rows))
    return [{'motif': 'closed_ring', 'base_seed': 11200, 'view': 0,
             'destruction': .4, 'full_clear': 0, 'town_hall': 0, 'episode_return': .1, 'spent': 200}]


def test_comparison_uses_matching_scenarios_and_equal_learner_weights(tmp_path):
    file = tmp_path/'ppo.csv'
    ddqn = synthetic_archive(file)
    models, conditions, deltas = r.compare(ddqn, file)
    all_scope = {x['condition']: x for x in conditions if x['scope'] == 'all'}
    assert all_scope['P']['learners'] == 10
    assert all_scope['DDQN']['learners'] == 1
    assert all_scope['P']['mean_destruction_percent'] == pytest.approx(60)
    assert next(x for x in deltas if x['scope'] == 'all' and x['comparison'] == 'DDQN minus P')['destruction_percentage_points'] == pytest.approx(-20)
    ddqn[0]['view'] = 1
    with pytest.raises(ValueError, match='scenarios differ'):
        r.compare(ddqn, file)


@pytest.mark.parametrize('fault', ['duplicate', 'missing'])
def test_incomplete_or_duplicate_ppo_evidence_is_rejected(tmp_path, fault):
    file = tmp_path/'ppo.csv'
    rows = synthetic_archive(file, **{fault: True})
    with pytest.raises(ValueError, match='Duplicate|Missing'):
        r.compare(rows, file)


def test_archived_reference_means_reproduced_without_running_policies():
    bases = r.checked_benchmark(r.ROOT)
    # Synthetic DDQN zeros only to exercise matching; never exported as research results.
    rows = [{'motif': b.motif, 'base_seed': b.seed, 'view': v,
             'destruction': 0, 'full_clear': 0, 'town_hall': 0, 'episode_return': 0, 'spent': 0}
            for b in bases for v in range(8)]
    _, conditions, _ = r.compare(rows, r.ROOT/r.ARCHIVE/'episodes.csv')
    d = {x['condition']: x for x in conditions if x['scope'] == 'all'}
    assert d['P']['mean_destruction_percent'] == pytest.approx(92.9648, abs=.00005)
    assert d['PA']['mean_destruction_percent'] == pytest.approx(93.5310, abs=.00005)
    assert d['heuristic']['mean_destruction_percent'] == pytest.approx(56.8977, abs=.00005)

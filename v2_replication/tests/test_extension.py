"""Cohort isolation, evidence integrity, and interruption checks."""
import hashlib
import json
import pytest
from v2_replication import protocol, runner, workflow
from v2_replication.tests.test_recovery import tiny_run, same_runs
from experiments.storage import atomic_json, run_lock


def test_original_engine_and_new_cohort_produce_same_learning(tmp_path):
    from version2.tests.test_v2 import tiny_run as original_tiny_run
    original_tiny_run(tmp_path/'original', condition='PA', workers=2, dev=True)
    tiny_run(tmp_path/'replication', condition='PA', workers=2, dev=True)
    same_runs(tmp_path/'original', tmp_path/'replication')


def test_cohort_paths_and_recipe():
    plan = protocol.read_plan()
    assert plan['learner_seeds'] == list(range(3, 10))
    assert plan['total_steps'] * 14 == 58720256
    for c in plan['conditions']:
        for s in plan['learner_seeds']:
            assert runner.run_path(c, s) == protocol.ROOT/'runs/version2-seed-replication'/f'{c}-{s}'
    from version2.protocol import read_plan as old_plan
    assert old_plan()['learner_seeds'] == [0, 1, 2]
    assert runner.registration_path() == protocol.ROOT/'version2-records/seed-replication/registration.json'


def test_evidence_requires_exact_screen_and_phase_files(tmp_path, monkeypatch):
    package = tmp_path/'package'; package.mkdir()
    project = tmp_path/'project'; project.mkdir()
    def write(name, value):
        atomic_json(project/name, value)
        return protocol.digest(project/name)
    runs, models = {}, []
    for c in ('P', 'PA'):
        for s in range(3):
            name = f'{c}-{s}'; prefix = f'runs/original/{name}'
            contract = write(prefix+'/contract.json', {'example': name})
            manifest = write(prefix+'/manifest.json', {'status': 'completed', 'actual_steps': 4194304,
                                                       'contract_sha256': contract})
            model = write(prefix+'/model.zip', {'fixture': name})
            runs[name] = {'path': prefix, 'manifest_sha256': manifest, 'contract_sha256': contract}
            models.append({'path': prefix+'/model.zip', 'sha256': model})
    record = write('records/registration.json', {'fixture': 'screen'})
    pool = write('bases/dev.json', {'fixture': 'pool'})
    evidence = write('artifacts/phase/episodes.csv', {'fixture': 'phase'})
    write('artifacts/phase/complete.json', {'status': 'completed', 'new_episodes': 44928})
    atomic_json(package/'screen_reference.json', {'source_hashes': {}, 'version2_source_hashes': {},
        'registration_path': 'records/registration.json', 'registration_sha256': record,
        'dev_pool': {'path': 'bases/dev.json', 'sha256': pool}, 'runs': runs, 'models': models})
    atomic_json(package/'decision.json', {'decision': 'DS', 'source_directory': 'artifacts/phase',
        'source_files': {'episodes.csv': evidence}, 'required_completion': {'status': 'completed','new_episodes':44928}})
    monkeypatch.setattr(protocol, 'PACKAGE', package)
    assert protocol.verify_evidence(project)['completed_screen_learners'] == 6
    write('artifacts/phase/complete.json', {'status': 'paused', 'new_episodes': 1000})
    with pytest.raises(ValueError, match='completion differs'):
        protocol.verify_evidence(project)
    write('artifacts/phase/complete.json', {'status': 'completed', 'new_episodes': 44928})
    (project/'artifacts/phase/episodes.csv').write_text('changed')
    with pytest.raises(ValueError, match='Changed evidence'):
        protocol.verify_evidence(project)


def test_rejects_changed_extension_on_resume(tmp_path, monkeypatch):
    tiny_run(tmp_path/'run', pause=96)
    original = runner.checked_runtime
    def changed():
        current = original(); current['replication_source_hashes'] = {'changed': 'bad'}
        return current
    monkeypatch.setattr(runner, 'checked_runtime', changed)
    with pytest.raises(ValueError, match='replication_source_hashes changed'):
        runner.resume(tmp_path/'run')


def test_matrix_pause_stops_before_next_learner_and_resumes(tmp_path, monkeypatch):
    directory = tmp_path/'records'; directory.mkdir()
    atomic_json(directory/'registration.json', {})
    monkeypatch.setattr(workflow, 'records', lambda: directory)
    monkeypatch.setattr(runner, 'validate_registration', lambda: (directory/'registration.json', {}))
    monkeypatch.setattr(runner, 'run_path', lambda c, s: tmp_path/f'{c}-{s}')
    calls = []
    def fake_run(c, s, target):
        calls.append((c, s))
        if len(calls) == 1:
            (tmp_path/f'{c}-{s}').mkdir()  # Pause before the contract has been written.
            workflow.pause()
        return {'actual_steps': target, 'status': 'completed'}
    monkeypatch.setattr(runner, 'start_or_resume', fake_run)
    workflow.run_training(4194304)
    assert calls == [('P', 3)] and not (directory/'active.json').exists()
    workflow.run_training(4194304)
    assert len(calls) == 15 and calls[-1] == ('PA', 9)
    assert len(list((directory/'handled-stops').glob('*.json'))) == 1


def test_second_matrix_is_blocked_and_old_seed_rejected(tmp_path, monkeypatch):
    directory = tmp_path/'records'; directory.mkdir()
    monkeypatch.setattr(workflow, 'records', lambda: directory)
    monkeypatch.setattr(runner, 'validate_registration', lambda: (None, {}))
    with run_lock(directory, '.training.lock'):
        with pytest.raises(RuntimeError, match='active'):
            workflow.run_training(4194304)
    with pytest.raises(ValueError, match='seeds 3'):
        workflow.run_training(4194304, 'P', 0)


def test_global_pause_is_seen_before_first_update(tmp_path, monkeypatch):
    directory = tmp_path/'records'; directory.mkdir()
    monkeypatch.setattr(runner, 'registration_path', lambda: directory/'registration.json')
    atomic_json(directory/'STOP_MATRIX.json', {'requested': True})
    result = tiny_run(tmp_path/'run')
    assert result['status'] == 'paused' and result['pause_reason'] == 'user_request'
    assert result['actual_steps'] < 192


def test_review_archive_contains_evidence_but_no_weights(tmp_path, monkeypatch):
    directory = tmp_path/'version2-records/seed-replication'; directory.mkdir(parents=True)
    reg = directory/'registration.json'; atomic_json(reg, {'fixture': True})
    package = tmp_path/'v2_replication'; package.mkdir()
    atomic_json(package/'plan.json', {'fixture': True})
    monkeypatch.setattr(workflow, 'ROOT', tmp_path)
    monkeypatch.setattr(workflow, 'PACKAGE', package)
    monkeypatch.setattr(workflow, 'records', lambda: directory)
    monkeypatch.setattr(runner, 'validate_registration', lambda: (reg, {}))
    monkeypatch.setattr(runner, 'run_path', lambda c, s: tmp_path/'runs'/f'{c}-{s}')
    workflow.pack()
    import zipfile
    archive = next(tmp_path.glob('Base_Siege_V2_Replication_Review_*.zip'))
    with zipfile.ZipFile(archive) as z:
        receipt = json.loads(z.read('review/receipt.json'))
        assert receipt['completed_learners'] == 0 and receipt['expected_total_transitions'] == 58720256
        assert all(hashlib.sha256(z.read(n)).hexdigest() == sha for n, sha in receipt['files'].items())
        assert not any(name.endswith('model.zip') for name in z.namelist())


def test_status_refuses_to_report_missing_log_as_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, 'ROOT', tmp_path)
    atomic_json(tmp_path/'runs/version2-seed-replication/P-3/manifest.json',
                {'actual_steps': 8192, 'latest_checkpoint': 'checkpoint-missing'})
    with pytest.raises(ValueError, match='Missing committed episode log'):
        workflow.status_rows()

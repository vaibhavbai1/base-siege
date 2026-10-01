"""Plan and evidence checks without importing a learning framework."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def source_hashes_replication():
    paths = sorted(PACKAGE.rglob('*.py')) + sorted(PACKAGE.glob('*.json'))
    return {str(p.relative_to(ROOT)): digest(p) for p in paths if p.name != 'PACKAGE_SHA256.json'}


def read_plan():
    plan = read(PACKAGE/'plan.json')
    original = read(ROOT/'version2/plan.json')
    keys = ('placement_steps', 'total_steps', 'screen_steps', 'n_envs', 'rollout_steps',
            'batch_size', 'epochs', 'checkpoint_every_rollouts', 'keep_checkpoints',
            'development_interval', 'sample_seeds', 'milestones', 'initialization', 'unchanged')
    for key in keys:
        require(plan[key] == original[key], f'Training recipe differs from the screen: {key}')
    require(plan['experiment'] == 'version2-seed-replication', 'Wrong experiment identity')
    require(plan['conditions'] == ['P', 'PA'] and plan['learner_seeds'] == list(range(3, 10)),
            'This extension is only P/PA seeds 3–9')
    require(plan['selected_evaluation_mode'] == 'DS', 'Recorded acting decision changed')
    return plan


def check_stop(target, plan):
    quantum = plan['n_envs'] * plan['rollout_steps']
    require(type(target) is int and 0 < target <= plan['total_steps'] and target % quantum == 0,
            f'--to must be a positive multiple of {quantum} up to {plan["total_steps"]}')


def verify_hashes(root, expected):
    for relative, wanted in expected.items():
        path = root/relative
        require(path.is_file(), f'Missing evidence: {relative}')
        require(digest(path) == wanted, f'Changed evidence: {relative}; restore the recorded file, do not bypass this check')


def verify_evidence(root=ROOT):
    screen = read(PACKAGE/'screen_reference.json')
    required = {**screen['source_hashes'], **screen['version2_source_hashes'],
                screen['registration_path']: screen['registration_sha256'],
                screen['dev_pool']['path']: screen['dev_pool']['sha256']}
    for run in screen['runs'].values():
        required[run['path']+'/manifest.json'] = run['manifest_sha256']
        required[run['path']+'/contract.json'] = run['contract_sha256']
    for model in screen['models']:
        required[model['path']] = model['sha256']
    decision = read(PACKAGE/'decision.json')
    required.update({decision['source_directory']+'/'+name: value
                     for name, value in decision['source_files'].items()})
    verify_hashes(root, required)
    receipt = read(root/decision['source_directory']/'complete.json')
    for key, value in decision['required_completion'].items():
        require(receipt.get(key) == value, f'Phase diagnostic completion differs: {key}')
    for name, run in screen['runs'].items():
        manifest = read(root/run['path']/'manifest.json')
        require(manifest['status'] == 'completed' and manifest['actual_steps'] == 4194304,
                f'Screen learner is not complete: {name}')
        require(manifest['contract_sha256'] == run['contract_sha256'], f'Wrong screen contract: {name}')
    require(len(screen['runs']) == 6, 'Expected six completed screen learners')
    return {'verified_hashed_files': len(required), 'completed_screen_learners': 6,
            'selected_mode': decision['decision'], 'new_learners': 14}

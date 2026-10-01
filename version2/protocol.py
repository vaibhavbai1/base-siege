"""Dependency-free protocol and original source checks."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT/'version2'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes_v2():
    paths = list(PACKAGE.rglob('*.py')) + [PACKAGE/'plan.json', PACKAGE/'baseline.json']
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def read_plan():
    plan = json.loads((PACKAGE/'plan.json').read_text())
    quantum = plan['n_envs'] * plan['rollout_steps']
    if quantum < 2 or plan['keep_checkpoints'] < 3 or plan['checkpoint_every_rollouts'] < 1:
        raise ValueError('Invalid rollout or retention settings')
    counts = [plan['placement_steps'], plan['total_steps'], plan['screen_steps'],
              plan['development_interval'], *plan['milestones']]
    if any(type(n) is not int or n <= 0 or n % quantum for n in counts):
        raise ValueError('Budgets, milestones and evaluation interval must align with complete rollouts')
    if not plan['placement_steps'] < plan['screen_steps'] <= plan['total_steps']:
        raise ValueError('Invalid phase budgets')
    if max(plan['milestones']) != plan['total_steps']:
        raise ValueError('Final milestone must match total budget')
    return plan


def verify_baseline(root=ROOT):
    baseline = json.loads((PACKAGE/'baseline.json').read_text())
    expected = dict(baseline['source_hashes'])
    expected.update({item['path']: item['sha256'] for item in baseline['pools'].values()})
    failures = []
    for relative, wanted in expected.items():
        path = root/relative
        if not path.is_file():
            failures.append(f'Missing: {relative}')
        elif digest(path) != wanted:
            failures.append(f'Changed file or Git LFS pointer: {relative}')
    if failures:
        raise ValueError('\n'.join(failures)+'\nRestore verified original files; do not disable this check.')
    return baseline


def check_stop(target, plan):
    quantum = plan['n_envs'] * plan['rollout_steps']
    if target <= 0 or target > plan['total_steps'] or target % quantum:
        raise ValueError(f'--to must be a positive multiple of {quantum} up to {plan["total_steps"]}')

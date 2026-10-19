"""Read-only adapter for the development-selected T-1 checkpoint."""
from pathlib import Path
import hashlib
import json

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
STEPS = 4194304
LEARNER_SEED = 1
CHECKSUMS = {
    'model.zip': '277567ab4491aa3dda8b55d2b52545edc487761573f532103596d2fc0385f876',
    'manifest.json': 'fc57d9860e0e538e8c73014881aa5a8c445d9c4661bd270b001dc6e24cfc33cd',
    'contract.json': '00333fe38c5ecc9b365e619a92901b561970b2e92366de311b07d618adbfaa3a',
}
CANDIDATES = ('artifacts/trpo/selected-T1', 'runs/study-v3/T-1')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_checkpoint(root=ROOT):
    for name in CANDIDATES:
        folder = Path(root) / name
        if all((folder / file).is_file() for file in CHECKSUMS):
            return folder / 'model.zip', {
                'model_sha256': CHECKSUMS['model.zip'], 'actual_steps': STEPS,
                'training_contract_sha256': CHECKSUMS['contract.json'],
                'settings': {'condition': 'T', 'seed': LEARNER_SEED, 'mode': 'full'},
                'application_version': 'TRPO', 'algorithm': 'MaskableTRPO',
            }
    return None, {}


def validate_checkpoint(path, root=ROOT):
    """Check the pinned checkpoint, its records and the frozen inference source."""
    path, root = Path(path), Path(root)
    if path.name != 'model.zip':
        raise ValueError('Expected the selected T-1 model.zip.')
    for name, expected in CHECKSUMS.items():
        if digest(path.with_name(name)) != expected:
            raise ValueError('Selected TRPO file checksum differs: ' + name)
    manifest = json.loads(path.with_name('manifest.json').read_text())
    contract = json.loads(path.with_name('contract.json').read_text())
    if (manifest['status'] != 'completed' or manifest['actual_steps'] != STEPS
            or manifest['settings']['seed'] != LEARNER_SEED
            or manifest['settings']['condition'] != 'T'):
        raise ValueError('Unexpected TRPO learner identity or transition count.')
    from env import config as C
    config = json.loads(json.dumps({k: v for k, v in vars(C).items() if k.isupper()}))
    if manifest['config'] != config:
        raise ValueError('TRPO was trained with a different game configuration.')
    required = {n: h for n, h in manifest['source_hashes'].items()
                if n.startswith(('env/', 'agents/')) or n in ('train/features.py', 'experiments/rollout.py')}
    required['trpo/maskable_trpo.py'] = contract['source_hashes']['maskable_trpo.py']
    for name, expected in required.items():
        if digest(root / name) != expected:
            raise ValueError('Frozen TRPO inference source differs: ' + name)
    return manifest


@st.cache_resource(max_entries=1, show_spinner=False)
def _load_cached(path, modified_ns, size_bytes):
    from trpo.maskable_trpo import MaskableTRPO
    from ui.deployment import CpuPolicy
    from ui.policy_playback import isolated_rng
    with isolated_rng(0):
        model = MaskableTRPO.load(path, device='cpu')
        if model.num_timesteps != STEPS or model.seed != LEARNER_SEED:
            raise ValueError('TRPO checkpoint counter or learner seed differs.')
        model.policy.set_training_mode(False)
        return CpuPolicy(model)


def load_selected_policy(path):
    # Recheck files before consulting the cache, including source and sidecars.
    validate_checkpoint(path)
    stamp = Path(path).stat()
    return _load_cached(str(Path(path).resolve()), stamp.st_mtime_ns, stamp.st_size)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check'])
    parser.parse_args()
    path, _ = selected_checkpoint()
    if path is None:
        raise SystemExit('T-1 checkpoint files are missing.')
    load_selected_policy(path)
    print('TRPO ready: T-1, 4,194,304 transitions, CPU inference. Default acting mode: SS.')


if __name__ == '__main__':
    main()

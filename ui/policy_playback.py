"""Demo-only policy playback; does not modify the registered experiment pipeline."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import tempfile
import threading

import numpy as np
from env import config as C
from env.generator import generate_base
from ui.custom_bases import validate_custom_spec
from env.siege_env import SiegeEnv
from env.symmetry import SymmetryWrapper
from experiments.rollout import play_episode

# Python, NumPy and Torch CPU RNGs are process-global. Serialize whole sampled
# episodes across all cached models and sessions, then restore the caller's RNGs.
EPISODE_LOCK = threading.RLock()


def action_rng_seed(base, view, action_seed, scheme='legacy'):
    if scheme not in ('legacy','v2-final-ds-v1','v3-final-ds-v1'):
        raise ValueError('Unknown replay seed scheme.')
    text = (f'{scheme}:{action_seed}:{base.motif}:{base.seed}:{view}' if scheme!='legacy'
            else f'base-siege-action-v1:{base.seed}:{base.motif}:{view}:{action_seed}')
    if base.motif == 'custom':
        text += ':' + hashlib.sha256(base.to_json().encode()).hexdigest()
    value = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'big')
    return value if scheme!='legacy' else value % (2**31)


@contextmanager
def isolated_rng(seed):
    import torch
    with EPISODE_LOCK:
        python_state, numpy_state = random.getstate(), np.random.get_state()
        previous_determinism = torch.are_deterministic_algorithms_enabled()
        previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        with torch.random.fork_rng(devices=[]):
            try:
                random.seed(seed)
                np.random.seed(seed)
                torch.random.default_generator.manual_seed(seed)
                torch.set_num_threads(1)
                torch.use_deterministic_algorithms(True)
                yield
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)
                torch.use_deterministic_algorithms(previous_determinism, warn_only=previous_warn_only)


class ActingPolicy:
    def __init__(self, model, deterministic):
        self.model, self.deterministic = model, deterministic

    def act(self, observation, mask):
        action, _ = self.model.predict(observation, action_masks=mask, deterministic=self.deterministic)
        return int(action)


class PhaseActingPolicy:
    """Same phase decision as the frozen V2 evaluator; no extra RNG draws."""
    def __init__(self, model, mode):
        if mode not in ('DS','SS','DD'):
            raise ValueError('Choose DS, SS or DD.')
        self.model, self.mode = model, mode

    def act(self, observation, mask):
        phase = float(observation['vector'][6])
        if phase not in (0.0,1.0) or bool(mask[C.ACTION_FINISH_SHOPPING]) != (phase==0.0):
            raise ValueError('Phase and legal actions disagree.')
        deterministic = {'DS':(True,False),'SS':(False,False),'DD':(True,True)}[self.mode][int(phase)]
        action,_ = self.model.predict(observation, action_masks=mask, deterministic=deterministic)
        action = int(action)
        if not 0 <= action < len(mask) or not mask[action]:
            raise ValueError('Policy selected an illegal action.')
        return action


def record_policy(policy, *, base_seed, motif, view=0, action_seed=0, deterministic=False, spec=None,
                  acting_mode=None, seed_scheme='legacy'):
    selected_mode = acting_mode or ('DD' if deterministic else 'SS')
    if motif not in (*C.MOTIFS, 'custom') or view not in range(8) or not 0 <= action_seed <= 2**31-1:
        raise ValueError('Choose a supported demo structure, view and nonnegative action seed.')
    if motif == 'custom' and spec is None:
        raise ValueError('Provide a saved custom base.')
    base = spec if spec is not None else generate_base(base_seed, motif)
    if motif == 'custom':
        base = validate_custom_spec(base)
    if base.seed != base_seed or base.motif != motif:
        raise ValueError('The selected base does not match its seed and structure.')
    # SiegeEnv retains its normal protected-final-seed check.
    env = SymmetryWrapper(SiegeEnv(motifs=C.MOTIFS, record=True, record_ticks=True), transform=view)
    label = -1 if seed_scheme=='legacy' and selected_mode=='DD' else action_seed
    rng_seed = action_rng_seed(base, view, label, seed_scheme)
    try:
        with isolated_rng(rng_seed):
            replay = play_episode(env, PhaseActingPolicy(policy, selected_mode), seed=0, options={'spec':base})
    finally:
        env.close()
    replay['acting'] = {'mode':acting_mode or ('deterministic' if deterministic else 'sampled'), 'action_seed':action_seed,
                        'purchasing':'sampled' if selected_mode=='SS' else 'deterministic',
                        'deployment':'deterministic' if selected_mode=='DD' else 'sampled', 'seed_scheme':seed_scheme,
                        'view':view, 'rng_seed':rng_seed, 'display_coordinates':'original base orientation'}
    return replay


def model_catalog(paths):
    records = []
    for path in paths:
        try:
            manifest = json.loads(path.with_name('manifest.json').read_text())
            settings = manifest['settings']
            if manifest.get('status')=='completed' and settings.get('condition') in ('P','PA'):
                records.append((path, manifest))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(records, key=lambda item:(item[1]['settings']['condition'],
        item[1]['settings']['seed'], item[1]['settings'].get('mode')!='full', str(item[0])))


def validate_compatibility(manifest):
    if not manifest.get('model_sha256'):
        raise ValueError('This model needs a checksum in its manifest.')
    config = {k:v for k,v in vars(C).items() if k.isupper()}
    if 'config' in manifest and manifest['config'] != json.loads(json.dumps(config)):
        raise ValueError('This model was trained with a different game configuration. Open its matching project version.')
    if manifest.get('application_version') == 'V2':
        root = Path(__file__).resolve().parents[1]
        required = {name:digest for name,digest in manifest.get('source_hashes',{}).items()
                    if name.startswith(('env/','agents/')) or name in ('train/features.py','experiments/rollout.py')}
        if 'env/combat.py' not in required:
            raise ValueError('V2 inference provenance is missing.')
        for name,digest in required.items():
            path = root/name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Scientific source differs from the frozen V2 simulator: '+name)


def html_bytes(replay):
    from tools.export_replay_html import export_html
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'attack.html'
        export_html(replay,path)
        return path.read_bytes()

"""Fixed CPU diagnostics that preserve the learner's random-number streams."""
from io import BytesIO
import hashlib
import torch
from env import config as C
from env.siege_env import SiegeEnv
from env.symmetry import SymmetryWrapper
from experiments.design import development_panel
from experiments.rollout import play_episode
from train.checkpoints import capture_rng, restore_rng


def episode_seed(seed, base, view):
    key = f'v2-dev:{seed}:{base.motif}:{base.seed}:{view}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], 'big')


class ActingPolicy:
    def __init__(self, policy, deterministic):
        self.policy, self.deterministic = policy, deterministic

    def act(self, obs, mask):
        action, _ = self.policy.predict(obs, action_masks=mask, deterministic=self.deterministic)
        action = int(action)
        if not mask[action]:
            raise RuntimeError('Policy selected an illegal action')
        return action


def development_check(model, pool, history, interval, sample_seeds):
    if not pool or model.num_timesteps < history['next_step']:
        return
    rng = capture_rng()
    try:
        # Serialize only the policy's parameters and constructor settings, not its
        # cached action distribution/autograd graph or the learner optimizer.
        buffer = BytesIO()
        model.policy.save(buffer)
        buffer.seek(0)
        policy = type(model.policy).load(buffer, device='cpu')
        policy.set_training_mode(False)
        panel = development_panel(pool)
        if not panel:
            raise ValueError('Empty development panel')
        for mode, seeds in [('sampled', sample_seeds), ('deterministic', [0])]:
            episodes = []
            acting = ActingPolicy(policy, mode == 'deterministic')
            for base in panel:
                for view in C.DEV_DIAGNOSTIC_VIEWS:
                    for seed in seeds:
                        acting_seed = episode_seed(seed, base, view)
                        torch.manual_seed(acting_seed)
                        env = SymmetryWrapper(SiegeEnv(motifs=C.DEV_MOTIFS), transform=view)
                        try:
                            info = play_episode(env, acting, options={'spec': base})['summary']
                            episodes.append({**{key: info[key] for key in (
                                'base_seed', 'motif', 'destruction', 'town_hall', 'full_clear',
                                'episode_return', 'spent', 'finish_reason', 'decisions')},
                                'view': view, 'sample_seed': seed, 'acting_seed': acting_seed})
                        finally:
                            env.close()
            n = len(episodes)
            history['records'].append({'timesteps': model.num_timesteps, 'mode': mode,
                'device': 'cpu', 'episodes': n,
                'mean_destruction': sum(r['destruction'] for r in episodes)/n,
                'full_clear_rate': sum(r['full_clear'] for r in episodes)/n,
                'town_hall_rate': sum(r['town_hall'] for r in episodes)/n,
                'mean_return': sum(r['episode_return'] for r in episodes)/n,
                'mean_spent': sum(r['spent'] for r in episodes)/n, 'episode_results': episodes})
        while history['next_step'] <= model.num_timesteps:
            history['next_step'] += interval
    finally:
        restore_rng(rng)

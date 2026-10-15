"""Local UI adapter for the selected masked Double DQN pilot; no training changes."""
from pathlib import Path
import numpy as np

STEPS = 2097152
POLICY_SHA256 = 'c76232fc77ca43bf2cab616d2546324d9026d7a870df28f17ba4388b458fbdc9'
CONTRACT_SHA256 = 'da9cbc73ab732fb98f2b9c0ce35b2da58601cf6f789b04f99949d9dcd433bf86'
CANDIDATES = (
    'artifacts/ddqn/selected-lr3e5-seed100-v2-benchmark/selected_model',
    'runs/ddqn/pilot-lr3e5-seed100',
)


def selected_checkpoint(root):
    """Prefer the frozen inference copy of the exact selected checkpoint."""
    for name in CANDIDATES:
        run = Path(root)/name
        policy = run/'policies'/f'step-{STEPS:012d}.pt'
        if all(p.is_file() for p in (policy, policy.with_suffix('.json'), run/'contract.json')):
            return policy, {
                'model_sha256': POLICY_SHA256, 'actual_steps': STEPS,
                'settings': {'condition': 'DDQN', 'seed': 100, 'mode': 'pilot'},
                'application_version': 'DDQN', 'algorithm': 'masked-double-dqn',
                'training_contract_sha256': CONTRACT_SHA256,
            }
    return None, {}


class DDQNPolicy:
    """Same predict interface as the existing replay recorder expects."""
    def __init__(self, network):
        self.network = network
        self.rng = np.random.default_rng(0)  # Not drawn from at epsilon zero.

    def predict(self, observation, *, action_masks, deterministic=True):
        if not deterministic:
            raise ValueError('Selected DDQN replay uses greedy legal actions only.')
        from ddqn_experiment.core import choose_action
        return choose_action(self.network, observation, action_masks, self.rng, 0.0, 'cpu'), None


def load_selected_policy(policy_path):
    from ddqn_experiment.evaluation import load_policy
    from experiments.provenance import sha256
    from ui.policy_playback import isolated_rng
    path = Path(policy_path)
    run = path.parent.parent
    if path.name != f'step-{STEPS:012d}.pt' or sha256(path) != POLICY_SHA256:
        raise ValueError('Selected DDQN policy differs from the verified checkpoint.')
    if sha256(run/'contract.json') != CONTRACT_SHA256:
        raise ValueError('Selected DDQN training contract differs.')
    # Preserve the existing loader's scientific source, runtime and sidecar checks.
    # Initialising the network must not disturb another session's PPO sampling.
    with isolated_rng(0):
        network, identity = load_policy(run, steps=STEPS, device='cpu')
    if identity['learner_seed'] != 100 or identity['training_mode'] != 'pilot':
        raise ValueError('Unexpected DDQN learner identity.')
    return DDQNPolicy(network)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check'])
    parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path, _ = selected_checkpoint(root)
    if path is None:
        raise SystemExit('Selected DDQN files not found. Keep the original run or frozen selected_model folder in this project.')
    load_selected_policy(path)
    print(f'DDQN ready: seed 100, {STEPS:,} transitions, greedy legal acting.\n{path.relative_to(root)}')


if __name__ == '__main__':
    main()

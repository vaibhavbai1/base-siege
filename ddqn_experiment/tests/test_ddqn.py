from dataclasses import replace
import json
from pathlib import Path
import pickle

import numpy as np
import pytest
import torch

from env.siege_env import SiegeEnv
from experiments.rollout import load_replay
from ddqn_experiment.core import (Settings, QNetwork, ReplayBuffer, bootstrap_mask,
                                  choose_action, double_dqn_targets, epsilon_at, masked_argmax)
from ddqn_experiment.runner import train, load_latest
from ddqn_experiment.evaluation import evaluate, record_replay


def tiny():
    return Settings(total_steps=64, placement_steps=16, buffer_size=32, batch_size=4,
                    learning_starts=4, target_interval=8, exploration_steps=32,
                    checkpoint_interval=32, log_interval=16)


def test_double_target_uses_online_choice_target_value_and_mask():
    online = torch.tensor([[1., 3., 900.], [1., 2., 3.]])
    target = torch.tensor([[20., 4., 999.], [float('nan')]*3])
    mask = torch.tensor([[True, True, False], [False]*3])
    result = double_dqn_targets(torch.tensor([1., 7.]), torch.tensor([False, True]),
                                online, target, mask, 0.5)
    assert torch.equal(result, torch.tensor([3., 7.]))


def test_no_legal_nonterminal_action_is_an_error():
    with pytest.raises(ValueError):
        masked_argmax(torch.zeros(1, 3), torch.zeros(1, 3, dtype=torch.bool))


def test_both_exploration_and_greedy_are_legal():
    torch.set_num_threads(1)
    env = SiegeEnv()
    obs, _ = env.reset(seed=7)
    net = QNetwork(env.observation_space)
    mask = np.zeros(246, bool)
    mask[[1, 3]] = True
    with torch.no_grad():
        net.head[-1].weight.zero_()
        net.head[-1].bias.zero_()
        net.head[-1].bias[245] = 1000  # Highest value belongs to an illegal action.
        net.head[-1].bias[3] = 2
    rng = np.random.default_rng(5)
    assert choose_action(net, obs, mask, rng, 0, 'cpu') == 3
    chosen = [choose_action(net, obs, mask, rng, 1, 'cpu') for _ in range(200)]
    assert set(chosen) == {1, 3}


def test_truncation_bootstraps_without_mutating_environment():
    env = SiegeEnv()
    env.reset(seed=9)
    legal = env.action_masks().copy()
    env.truncated = True
    assert not env.action_masks().any()
    assert np.array_equal(bootstrap_mask(env, False, True), legal)
    assert env.truncated is True
    assert not bootstrap_mask(env, True, False).any()


def test_buffer_preserves_float_observations_and_ring_position():
    env = SiegeEnv()
    obs, _ = env.reset(seed=1)
    obs['grid'][0, 0, 0] = 0.1234567
    buffer = ReplayBuffer(3, env.observation_space)
    for i in range(5):
        buffer.add(obs, i, i/10, obs, False, env.action_masks())
    restored = ReplayBuffer(3, env.observation_space)
    restored.load_state_dict(buffer.state_dict())
    assert restored.pos == 2 and restored.size == 3
    assert np.array_equal(restored.obs['grid'], buffer.obs['grid'])
    assert np.array_equal(restored.actions, buffer.actions)
    obs['grid'][0, 0, 0] = 0
    assert np.all(restored.obs['grid'][:, 0, 0, 0] == np.float32(0.1234567))


def test_phase_exploration_is_independent_of_pause_point():
    s = Settings()
    assert epsilon_at(0, s) == 1
    assert epsilon_at(s.placement_steps, s) == 1
    assert epsilon_at(s.placement_steps + s.exploration_steps, s) == pytest.approx(.05)


@pytest.mark.parametrize('pause_at', [11, 16, 23])
def test_exact_cpu_resume_updates_weights_buffer_rng_and_logs(tmp_path, pause_at):
    full, split = tmp_path/'full', tmp_path/'split'
    s = tiny()
    train(full, s, seed=17, mode='smoke')
    train(split, s, seed=17, mode='smoke', until=pause_at)
    # Uncommitted log tail from a simulated abrupt stop must be removed on resume.
    with (split/'episodes.jsonl').open('a') as f:
        f.write('{"uncommitted": true}\n')
    train(split, s, seed=17, mode='smoke', until=64, resume=True)
    a, b = load_latest(full), load_latest(split)
    for net in ['online', 'target']:
        assert all(torch.equal(a[net][k], b[net][k]) for k in a[net])
    for key in ['steps', 'updates', 'episodes', 'phase', 'phase_steps', 'phase_cutoffs', 'last_loss']:
        assert a['progress'][key] == b['progress'][key]
    assert a['progress']['updates'] > 0
    initial = torch.load(full/'policies'/'step-000000000000.pt', weights_only=True)['online']
    assert any(not torch.equal(initial[k], a['online'][k]) for k in initial)
    assert a['progress']['phase'] == 'joint'
    for group in ['obs', 'next_obs']:
        for k in a['replay'][group]:
            assert np.array_equal(a['replay'][group][k], b['replay'][group][k])
    for key in ['actions', 'rewards', 'terminated', 'next_masks']:
        assert np.array_equal(a['replay'][key], b['replay'][key])
    for k in a['obs']:
        assert np.array_equal(a['obs'][k], b['obs'][k])
    assert pickle.dumps(a['rng']['generator']) == pickle.dumps(b['rng']['generator'])
    assert (full/'episodes.jsonl').read_bytes() == (split/'episodes.jsonl').read_bytes()
    assert len(list((split/'checkpoints').glob('step-*'))) == 2
    with pytest.raises(ValueError, match='Resume refused'):
        train(split, replace(s, learning_rate=.002), seed=17, mode='smoke', resume=True)


def test_corrupt_checkpoint_refused_and_existing_run_protected(tmp_path):
    root = tmp_path/'run'
    train(root, tiny(), seed=17, mode='smoke', until=3)
    with pytest.raises(FileExistsError):
        train(root, tiny(), seed=17, mode='smoke')
    latest = json.loads((root/'LATEST.json').read_text())['checkpoint']
    state = root/'checkpoints'/latest/'state.pt'
    with state.open('ab') as f:
        f.write(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        load_latest(root)


def test_dev_evaluation_resume_and_saved_replay(tmp_path):
    dev = Path('bases/dev-v2.json')
    if not dev.exists():
        pytest.skip('Existing development pool is required for this integration test')
    root = tmp_path/'learner'
    train(root, tiny(), seed=123, mode='smoke', dev_pool=dev)
    out = tmp_path/'evaluation'
    first = evaluate(root, dev, out, panel=True, views=[0])
    resumed = evaluate(root, dev, out, panel=True, views=[0], resume=True)
    assert first == resumed
    assert first['episodes'] == 12
    record_replay(root, dev, tmp_path/'replay.json')
    replay = load_replay(tmp_path/'replay.json')
    assert replay['policy']['algorithm'] == 'masked-double-dqn'
    assert replay['snapshots']
    with pytest.raises(ValueError, match='contract changed|differs'):
        evaluate(root, dev, out, panel=True, views=[1], resume=True)


def test_final_pools_are_not_accessible(tmp_path):
    from ddqn_experiment.runner import checked_pool
    data = json.loads(Path('bases/dev-v2.json').read_text())
    path = tmp_path/'final-copy.json'
    path.write_text(json.dumps(data))
    with pytest.raises(PermissionError):
        checked_pool(path, 'dev')

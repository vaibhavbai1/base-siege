from dataclasses import replace
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from train.checkpoints import (RecoverablePPO, latest_checkpoint, verify_checkpoint,
                               run_lock, capture_rng, restore_rng, atomic_json)
from train.train_ppo import TrainingSettings, phase_budgets, train, resume
from experiments.pools import build_pool
from env import config as C


def assert_same_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_same_tree(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_same_tree(left, right)
    else:
        assert a == b


def assert_same_runs(a, b):
    left = RecoverablePPO.load(a / "model.zip", device="cpu")
    right = RecoverablePPO.load(b / "model.zip", device="cpu")
    assert left.num_timesteps == right.num_timesteps
    assert left._n_updates == right._n_updates
    assert_same_tree(left.policy.state_dict(), right.policy.state_dict())
    assert_same_tree(left.policy.optimizer.state_dict(), right.policy.optimizer.state_dict())
    assert (a / "episodes.jsonl").read_bytes() == (b / "episodes.jsonl").read_bytes()
    assert json.loads((a / "development.json").read_text()) == json.loads((b / "development.json").read_text())


@pytest.mark.parametrize("condition,pause,workers", [("P", 32, 1), ("PA", 64, 1), ("PA", 96, 1), ("PA", 64, 2)])
def test_resume_matches_uninterrupted_updates(tmp_path, condition, pause, workers):
    settings = TrainingSettings(condition=condition, n_envs=workers, joint_steps=64)
    a, b = tmp_path / "continuous", tmp_path / "resumed"
    train(settings, a)
    paused = train(settings, b, pause_after_steps=pause)
    assert paused["status"] == "paused"
    assert paused["actual_steps"] == pause
    resumed = resume(b)
    assert resumed["status"] == "completed"
    assert_same_runs(a, b)
    assert resume(b)["attempts"] == resumed["attempts"]


def test_development_and_episode_history_survive_resume(tmp_path):
    pool = tmp_path / "dev.json"
    build_pool(10200, 10202, C.TRAIN_MOTIFS, pool, difficulty=False)
    settings = TrainingSettings(joint_steps=64)
    a, b = tmp_path / "a", tmp_path / "b"
    train(settings, a, dev_pool=pool, development_interval=64)
    train(settings, b, dev_pool=pool, development_interval=64, pause_after_steps=96)
    with (b / "episodes.jsonl").open("a") as handle:
        handle.write('{"uncommitted": true}\n')
    resume(b)
    assert_same_runs(a, b)
    assert "uncommitted" in (b / "recovery/attempt-002/previous-episodes.jsonl").read_text()


@pytest.mark.parametrize("condition", ["P", "PA"])
def test_compact_training_pool_resume_matches_continuous(tmp_path, condition):
    pool = tmp_path / "train.json"
    build_pool(20, 23, C.TRAIN_MOTIFS, pool, difficulty=False)
    settings = TrainingSettings(condition=condition, n_envs=2, joint_steps=64)
    a, b = tmp_path / "continuous", tmp_path / "resumed"
    train(settings, a, train_pool=pool)
    train(settings, b, train_pool=pool, pause_after_steps=64)
    resume(b)
    assert_same_runs(a, b)


def test_corrupt_latest_falls_back_and_ignores_partial_write(tmp_path):
    settings = TrainingSettings(joint_steps=64)
    a, b = tmp_path / "a", tmp_path / "b"
    train(settings, a)
    train(settings, b, pause_after_steps=96)
    newest, metadata, _ = latest_checkpoint(b)
    (newest / "state.pt").write_bytes(b"interrupted or damaged data")
    (b / "checkpoints/.staging-not-committed").mkdir()
    recovered, prior, rejected = latest_checkpoint(b)
    assert prior["actual_steps"] < metadata["actual_steps"] and rejected
    resume(b)
    assert_same_runs(a, b)


def test_portable_run_folder_and_changed_contract_rejected(tmp_path):
    a, b = tmp_path / "original", tmp_path / "moved"
    train(TrainingSettings(joint_steps=32), a, pause_after_steps=32)
    shutil.copytree(a, b)
    assert resume(b)["status"] == "completed"
    contract = json.loads((a / "contract.json").read_text())
    contract["settings"]["seed"] = 999
    atomic_json(a / "contract.json", contract)
    with pytest.raises(ValueError, match="contract changed"):
        resume(a)


def test_source_change_is_not_silently_resumed(tmp_path, monkeypatch):
    import train.train_ppo as runner
    train(TrainingSettings(joint_steps=32), tmp_path / "run", pause_after_steps=32)
    original = runner.runtime_manifest
    def changed():
        result = original()
        result["source_hashes"] = {"changed.py": "abc"}
        return result
    monkeypatch.setattr(runner, "runtime_manifest", changed)
    with pytest.raises(ValueError, match="source_hashes changed"):
        resume(tmp_path / "run")


def test_interrupted_initial_checkpoint_can_restart_before_any_learning(tmp_path, monkeypatch):
    import train.train_ppo as runner
    settings = TrainingSettings(joint_steps=32)
    a, b = tmp_path / "continuous", tmp_path / "interrupted"
    train(settings, a)
    with monkeypatch.context() as patch:
        patch.setattr(runner, "save_checkpoint", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("initial write interrupted")))
        with pytest.raises(RuntimeError, match="initial write"):
            train(settings, b)
    assert json.loads((b / "manifest.json").read_text())["actual_steps"] == 0
    assert resume(b)["status"] == "completed"
    assert_same_runs(a, b)


def test_run_lock_and_missing_legacy_contract(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="active training"):
            with run_lock(tmp_path):
                pass
    with run_lock(tmp_path):
        pass
    with pytest.raises(ValueError, match="Older model-only"):
        resume(tmp_path)


def test_rng_and_actual_budget_contract():
    state = capture_rng()
    a = torch.rand(5), np.random.random(5)
    restore_rng(state)
    assert_same_tree(a, (torch.rand(5), np.random.random(5)))
    settings = TrainingSettings(placement_steps=25000, joint_steps=225000, n_envs=4, rollout_steps=2048)
    assert phase_budgets(settings) == {"rollout_transitions": 8192, "placement_actual": 32768,
                                      "joint_actual": 229376, "total_actual": 262144}


def test_stop_on_last_rollout_step_does_not_export_untrained_data(tmp_path, monkeypatch):
    from train.train_ppo import ProgressRecorder, request_pause
    original = ProgressRecorder._on_step
    def stop_at_endpoint(callback):
        if callback.num_timesteps == 128:
            request_pause(callback.path.parent)
        return original(callback)
    settings = TrainingSettings(joint_steps=64)
    a, b = tmp_path / "a", tmp_path / "b"
    train(settings, a)
    with monkeypatch.context() as patch:
        patch.setattr(ProgressRecorder, "_on_step", stop_at_endpoint)
        stopped = train(settings, b)
    assert stopped["status"] == "paused"
    assert stopped["actual_steps"] == 96
    assert not (b / "model.zip").exists()
    resume(b)
    assert_same_runs(a, b)


def test_rounded_phase_budget_and_corrupt_pointer(tmp_path):
    settings = TrainingSettings(placement_steps=33, joint_steps=35)
    a, b = tmp_path / "a", tmp_path / "b"
    train(settings, a)
    train(settings, b, pause_after_steps=64)
    (b / "LATEST.json").write_text("incomplete pointer")
    result = resume(b)
    assert result["placement_actual"] == result["joint_actual"] == 64
    assert_same_runs(a, b)

"""Recovery tests use temporary files and training/development seeds only."""
from dataclasses import asdict
import json
import pickle
from pathlib import Path
import sqlite3
import select
import subprocess
import sys
import shutil

import pandas as pd
import pytest

from agents.heuristic import HeuristicCommander
from env import config as C
from env.generator import generate_base
from env.siege_env import SiegeEnv
from experiments.design import development_panel
from experiments.journal import open_journal
from experiments.pools import build_pool, load_pool
from experiments.provenance import runtime_manifest, sha256, source_hashes
from experiments.storage import atomic_json
from train.checkpoints import SnapshotMonitor
from train.train_ppo import phase_budgets, study_settings


def test_journal_contract_single_writer_and_checksum(tmp_path):
    path = tmp_path / "progress.sqlite3"
    with open_journal(path, {"seed": 3}) as journal:
        journal.put("a", {"score": .4})
        with pytest.raises(RuntimeError, match="active training or experiment"):
            with open_journal(path, {"seed": 3}, resume=True):
                pass
    with pytest.raises(FileExistsError):
        with open_journal(path, {"seed": 3}):
            pass
    with pytest.raises(ValueError, match="contract changed"):
        with open_journal(path, {"seed": 4}, resume=True):
            pass
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE items SET value='{}' WHERE key='a'")
    with open_journal(path, {"seed": 3}, resume=True) as journal:
        with pytest.raises(ValueError, match="checksum"):
            journal.get("a")


def test_killed_journal_writer_preserves_only_committed_units(tmp_path):
    path = tmp_path / "killed.sqlite3"
    script = """
import sys, time
from experiments.journal import open_journal
with open_journal(sys.argv[1], {"fixture": True}) as journal:
    journal.put("committed", {"value": 7})
    journal.connection.execute("BEGIN IMMEDIATE")
    journal.connection.execute("INSERT INTO items VALUES ('uncommitted', '{}', 'pending')")
    print("ready", flush=True)
    time.sleep(30)
"""
    child = subprocess.Popen([sys.executable, "-c", script, str(path)], stdout=subprocess.PIPE, text=True)
    try:
        ready, _, _ = select.select([child.stdout], [], [], 10)
        assert ready and child.stdout.readline().strip() == "ready"
        child.kill()
        child.wait(timeout=10)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        child.stdout.close()
    with open_journal(path, {"fixture": True}, resume=True) as journal:
        assert journal.get("committed") == {"value": 7}
        assert journal.get("uncommitted") is None
        journal.put("continued", {"value": 8})


def test_pool_resumes_without_revalidating_completed_candidates(tmp_path, monkeypatch):
    import experiments.pools as module
    uninterrupted, interrupted = tmp_path / "a.json", tmp_path / "b.json"
    expected = build_pool(20, 22, C.TRAIN_MOTIFS, uninterrupted, difficulty=False)
    original = module.generate_base
    calls = []
    def fail_once(seed, motif, **kwargs):
        calls.append((seed, motif))
        if len(calls) == 5:
            raise RuntimeError("simulated interruption")
        return original(seed, motif, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(module, "generate_base", fail_once)
        with pytest.raises(RuntimeError, match="interruption"):
            build_pool(20, 22, C.TRAIN_MOTIFS, interrupted, difficulty=False)
    assert not interrupted.exists()
    completed = set(calls[:4])
    resumed_calls = []
    def resumed(seed, motif, **kwargs):
        resumed_calls.append((seed, motif))
        return original(seed, motif, **kwargs)
    monkeypatch.setattr(module, "generate_base", resumed)
    result = build_pool(20, 22, C.TRAIN_MOTIFS, interrupted, difficulty=False, resume=True)
    assert result == expected
    assert not completed.intersection(resumed_calls)
    assert build_pool(20, 22, C.TRAIN_MOTIFS, interrupted, difficulty=False, resume=True) == expected
    assert uninterrupted.read_bytes() == interrupted.read_bytes()
    with pytest.raises(ValueError, match="contract changed"):
        build_pool(20, 23, C.TRAIN_MOTIFS, interrupted, difficulty=False, resume=True)


def test_transfer_not_admitted_to_current_development(tmp_path):
    path = tmp_path / "dev.json"
    build_pool(10200, 10209, (C.TRANSFER_MOTIF,), path, difficulty=False)
    with pytest.raises(ValueError, match="development pool"):
        load_pool(path, required_split="dev")


def test_development_panel_is_balanced_and_deterministic():
    pool = [generate_base(seed, motif) for seed in range(10) for motif in C.DEV_MOTIFS]
    panel = development_panel(pool)
    assert [(b.seed, b.motif) for b in panel] == [(b.seed, b.motif) for b in development_panel(pool[::-1])]
    assert len(panel) == 12
    for motif in C.DEV_MOTIFS:
        assert [b.seed for b in panel if b.motif == motif] == [0, 3, 6, 9]
    assert C.TRANSFER_MOTIF not in {b.motif for b in panel}


def test_compact_worker_snapshot_reuses_verified_frozen_pool():
    pool = [generate_base(seed, motif) for seed in range(50) for motif in C.TRAIN_MOTIFS]
    worker = SnapshotMonitor(SiegeEnv(pool=pool))
    worker.reset(seed=7)
    old_size = len(pickle.dumps(worker.__dict__, protocol=pickle.HIGHEST_PROTOCOL))
    saved = worker.checkpoint_state()
    assert len(saved) < old_size / 2
    restored = SnapshotMonitor(SiegeEnv(pool=pool))
    restored.reset(seed=999)
    restored.restore_checkpoint_state(saved)
    assert restored.unwrapped.pool is pool
    assert restored.unwrapped.spec == worker.unwrapped.spec
    other = SnapshotMonitor(SiegeEnv(pool=pool[:-1]))
    with pytest.raises(ValueError, match="pool changed"):
        other.restore_checkpoint_state(saved)


def test_pilot_and_full_budgets_are_explicit():
    assert len(C.STUDY_LEARNER_SEEDS) == 10
    assert phase_budgets(study_settings())["total_actual"] == 262144
    assert phase_budgets(study_settings(mode="pilot"))["total_actual"] == 57344


def test_episode_evaluation_resumes_exact_rows(tmp_path, monkeypatch):
    import train.evaluate as module
    monkeypatch.setattr(C, "RANDOM_AGENT_ACTION_SEEDS", (0, 1))
    bases = [generate_base(10200, motif) for motif in C.DEV_MOTIFS]
    expected = module.evaluate_study(bases, [], tmp_path / "a", views=(0, 4))
    original = module.episode_record
    completed = []
    def interrupted(policy, base, view, **kwargs):
        if len(completed) == 4:
            raise RuntimeError("interrupted episode")
        result = original(policy, base, view, **kwargs)
        completed.append((kwargs["condition"], base.motif, view))
        return result
    with monkeypatch.context() as patch:
        patch.setattr(module, "episode_record", interrupted)
        with pytest.raises(RuntimeError, match="interrupted"):
            module.evaluate_study(bases, [], tmp_path / "b", views=(0, 4))
    assert not (tmp_path / "b/episodes.csv").exists()
    resumed = []
    def trace(policy, base, view, **kwargs):
        resumed.append((kwargs["condition"], base.motif, view))
        return original(policy, base, view, **kwargs)
    monkeypatch.setattr(module, "episode_record", trace)
    actual = module.evaluate_study(bases, [], tmp_path / "b", views=(0, 4), resume=True)
    pd.testing.assert_frame_equal(actual, expected)
    assert not set(completed).intersection(resumed)
    assert (tmp_path / "a/episodes.csv").read_bytes() == (tmp_path / "b/episodes.csv").read_bytes()
    before = len(resumed)
    module.evaluate_study(bases, [], tmp_path / "b", views=(0, 4), resume=True)
    assert len(resumed) == before
    with pytest.raises(ValueError, match="contract changed"):
        module.evaluate_study(bases, [], tmp_path / "b", views=(0,), resume=True)


def test_final_resume_is_bound_to_freeze_and_output_without_real_final_layouts(tmp_path, monkeypatch):
    import experiments.protocol as protocol
    import train.evaluate as evaluator
    monkeypatch.setattr(C, "RANDOM_AGENT_ACTION_SEEDS", (0,))
    manifests, runs = {}, []
    for condition in ("P", "PA"):
        path = tmp_path / condition
        path.mkdir()
        (path / "model.zip").write_bytes(b"test fixture; not a model")
        manifest = {**json.loads(json.dumps(runtime_manifest())), "settings": asdict(study_settings(condition=condition)),
                    "status": "completed", "model_sha256": sha256(path / "model.zip"),
                    "actual_steps": 262144, "placement_actual": 32768, "joint_actual": 229376, "pools": {}}
        atomic_json(path / "manifest.json", manifest)
        manifests[condition] = manifest
        runs.append({"path": condition, "model_sha256": manifest["model_sha256"],
                     "manifest_sha256": sha256(path / "manifest.json")})
    freeze = tmp_path / "freeze.json"
    atomic_json(freeze, {"schema_version": 2, "source_hashes": source_hashes(), "runs": runs})
    # Substitute only the final pool source. No protected seed is generated or evaluated.
    def training_seed_fixture(start, stop, motifs, path, **kwargs):
        assert (start, stop) == C.FINAL_STUDY_SEED_RANGE
        return build_pool(20, 21, C.TRAIN_MOTIFS, path, difficulty=False, resume=kwargs["resume"])
    monkeypatch.setattr(protocol, "build_pool", training_seed_fixture)
    monkeypatch.setattr(evaluator, "load_run", lambda path: (HeuristicCommander(), manifests[Path(path).name]))
    original = evaluator.episode_record
    count = 0
    def fail(policy, base, view, **kwargs):
        nonlocal count
        count += 1
        if count == 4:
            raise RuntimeError("final interruption fixture")
        return original(policy, base, view, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(evaluator, "episode_record", fail)
        with pytest.raises(RuntimeError, match="interruption fixture"):
            protocol.final_evaluation(freeze, tmp_path / "final", confirm_final=True)
    with pytest.raises(FileExistsError, match="already started"):
        protocol.final_evaluation(freeze, tmp_path / "final", confirm_final=True)
    with pytest.raises(ValueError, match="same freeze and output"):
        protocol.final_evaluation(freeze, tmp_path / "different", confirm_final=True, resume=True)
    result = protocol.final_evaluation(freeze, tmp_path / "final", confirm_final=True, resume=True)
    assert set(result.base_seed) <= {20, 21}
    assert (tmp_path / "final/complete.json").exists()
    original_bytes = (tmp_path / "final/episodes.csv").read_bytes()
    protocol.final_evaluation(freeze, tmp_path / "final", confirm_final=True, resume=True)
    assert (tmp_path / "final/episodes.csv").read_bytes() == original_bytes


def test_study_matrix_skip_resume_and_source_guards(tmp_path, monkeypatch):
    import experiments.study as module
    registration = tmp_path / "registration.json"
    atomic_json(registration, {"learner_seeds": [0, 1], "source_hashes": source_hashes(),
                              "train_pool": "fixture", "dev_pool": "fixture"})
    plan = module.matrix(registration)
    assert [(r["seed"], r["condition"]) for r in plan] == [(0, "P"), (0, "PA"), (1, "PA"), (1, "P")]
    trained, resumed = [], []
    def fake_train(settings, output, **kwargs):
        output.mkdir()
        atomic_json(output / "contract.json", {"settings": asdict(settings),
            "inputs": {"registration": {"sha256": sha256(registration)}}})
        trained.append(output.name)
        return {"status": "completed"}
    monkeypatch.setattr(module, "train", fake_train)
    monkeypatch.setattr(module, "resume_run", lambda path: resumed.append(path.name) or {"status": "completed"})
    assert module.run_study(registration, tmp_path / "study", confirm_long_run=True, min_free_gib=1)["status"] == "completed"
    assert len(trained) == 4
    module.run_study(registration, tmp_path / "study", confirm_long_run=True, min_free_gib=1, resume=True)
    assert len(trained) == len(resumed) == 4
    with pytest.raises(ValueError, match="Existing output differs"):
        module.run_study(registration, tmp_path / "study", device="cuda", confirm_long_run=True, min_free_gib=1, resume=True)
    with pytest.raises(PermissionError):
        module.run_study(registration, tmp_path / "unapproved")


def test_registration_requires_complete_windows_and_freeze_paths_are_relative(tmp_path, monkeypatch):
    import experiments.pilot as pilot
    import experiments.pools as pools
    import experiments.protocol as protocol
    monkeypatch.setattr(C, "TRAIN_SEED_RANGE", (20, 29))
    monkeypatch.setattr(C, "DEV_SEED_RANGE", (10200, 10209))
    monkeypatch.setattr(C, "DEV_BASES_PER_MOTIF", 1)
    monkeypatch.setattr(pilot, "validate_review", lambda *args: {"summaries": [{"seed": 100}], "decision": "test fixture"})
    def fixture_validator(spec, **kwargs):
        valid, reason = pools.validate_structure(spec)
        return valid, .5, reason
    monkeypatch.setattr(pools, "validate_base", fixture_validator)
    train_pool, dev_pool = tmp_path / "train.json", tmp_path / "dev.json"
    build_pool(20, 29, C.TRAIN_MOTIFS, train_pool)
    build_pool(10200, 10209, C.DEV_MOTIFS, dev_pool)
    review = tmp_path / "review.json"
    review.write_text("synthetic review fixture")
    registration = tmp_path / "registration.json"
    protocol.register(registration, train_pool, dev_pool, pilot_review=review)
    reg = json.loads(registration.read_text())
    assert reg["learner_seeds"] == list(range(10))
    runs = []
    for seed in reg["learner_seeds"]:
        for condition in ("P", "PA"):
            path = tmp_path / "runs" / f"{condition}-{seed}"
            path.mkdir(parents=True)
            (path / "model.zip").write_bytes(b"synthetic full-run identity fixture")
            manifest = {**json.loads(json.dumps(runtime_manifest())),
                "settings": asdict(study_settings(condition=condition, seed=seed)),
                "status": "completed", "model_sha256": sha256(path / "model.zip"),
                "actual_steps": 262144, "placement_actual": 32768, "joint_actual": 229376,
                "registration_sha256": sha256(registration), "started_utc": "2099-01-01T00:00:00+00:00",
                "pools": {"train": {"sha256": sha256(train_pool)}, "dev": {"sha256": sha256(dev_pool)}}}
            atomic_json(path / "manifest.json", manifest)
            runs.append(path)
    frozen_path = tmp_path / "protocol" / "freeze.json"
    protocol.freeze(registration, runs, frozen_path)
    frozen = json.loads(frozen_path.read_text())
    assert frozen["paths_relative_to"] == "freeze_parent"
    assert all(not Path(run["path"]).is_absolute() for run in frozen["runs"])
    moved = tmp_path / "moved"
    shutil.copytree(tmp_path / "runs", moved / "runs")
    shutil.copytree(tmp_path / "protocol", moved / "protocol")
    for run in frozen["runs"]:
        assert sha256(moved / "protocol" / run["path"] / "model.zip") == run["model_sha256"]
    data = json.loads(dev_pool.read_text())
    data["validation"].pop()
    atomic_json(dev_pool, data)
    with pytest.raises(ValueError, match="candidate accounting"):
        protocol.register(tmp_path / "bad.json", train_pool, dev_pool, pilot_review=review)

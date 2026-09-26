from copy import deepcopy
import json
import numpy as np
import pandas as pd
import pytest
from env import config as C
from env.siege_env import SiegeEnv
from env.symmetry import SymmetryWrapper
from experiments.pools import build_pool, load_pool
from experiments.protocol import final_evaluation, register
from experiments.rollout import play_episode, save_replay, load_replay
from agents.random_agent import RandomAgent
from train.evaluate import paired_summary, validate_matched_runs, study_summary
from train.train_ppo import TrainingSettings, train


def test_augmentation_does_not_change_layout_sampling():
    plain = SiegeEnv()
    augmented = SymmetryWrapper(SiegeEnv())
    for index in range(20):
        seed = 9 if index == 0 else None
        a, ia = plain.reset(seed=seed)
        b, ib = augmented.reset(seed=seed)
        assert (ia["base_seed"], ia["motif"]) == (ib["base_seed"], ib["motif"])
        assert np.array_equal(a["vector"], b["vector"])


def rows():
    return pd.DataFrame([{"condition": condition, "learner_seed": learner, "base_seed": base,
        "motif": "closed_ring", "view": 0, "destruction": .2 + learner*.01 + (condition == "PA")*.1,
        "town_hall": 0, "full_clear": 0, "return": 0}
        for condition in ("P", "PA") for learner in range(5) for base in range(3)])


def test_paired_uncertainty_uses_learners_not_games():
    result = paired_summary(rows())
    assert result["n_independent_learners"] == 5
    assert result["mean_paired_difference"] == pytest.approx(.1)
    assert result["ci95_low"] == pytest.approx(.1)
    assert paired_summary(rows().query("learner_seed == 0"))["ci95_low"] is None
    assert "seen_structures" in study_summary(rows())["comparisons"]


def test_paired_units_must_match():
    frame = rows()
    with pytest.raises(ValueError, match="identical"):
        paired_summary(frame.iloc[1:])
    with pytest.raises(ValueError, match="Duplicate"):
        paired_summary(pd.concat([frame, frame.iloc[:1]]))


def manifest(condition):
    return {"settings": {"condition": condition, "seed": 0},
            **{key: 1 for key in ("actual_steps", "placement_actual", "joint_actual", "config", "pools", "source_hashes")}}


def test_matched_training_rejects_unfair_comparison():
    p, pa = manifest("P"), manifest("PA")
    assert validate_matched_runs([p, pa]) == [0]
    for field in ("actual_steps", "source_hashes", "pools"):
        changed = deepcopy(pa)
        changed[field] = 2
        with pytest.raises(ValueError, match="Unmatched"):
            validate_matched_runs([p, changed])
    with pytest.raises(ValueError, match="same learner"):
        validate_matched_runs([p])


def test_pool_provenance_split_and_duplicate_guards(tmp_path):
    path = tmp_path / "pool.json"
    build_pool(0, 8, C.TRAIN_MOTIFS, path, difficulty=False)
    assert load_pool(path, required_split="train")
    with pytest.raises(ValueError, match="outside"):
        load_pool(path, required_split="dev")
    data = json.loads(path.read_text())
    data["bases"].append(data["bases"][0])
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Duplicate"):
        load_pool(path)
    data["source_hashes"] = {}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="changed"):
        load_pool(path)


def test_protected_final_guards_do_not_generate_data(tmp_path):
    with pytest.raises(PermissionError):
        build_pool(10999, 11000, C.TRAIN_MOTIFS, tmp_path/"pool.json")
    assert not (tmp_path/"pool.json").exists()
    with pytest.raises(PermissionError):
        load_pool(tmp_path/"pool_final.json")
    with pytest.raises(PermissionError):
        final_evaluation(tmp_path/"missing.json", tmp_path/"results")
    assert not (tmp_path/"results").exists()


def test_no_accidental_full_run_or_invalid_registration(tmp_path):
    with pytest.raises(ValueError, match="require frozen"):
        train(TrainingSettings(mode="full"), tmp_path/"run")
    with pytest.raises(ValueError, match="requires an experiment registration"):
        train(TrainingSettings(mode="full"), tmp_path/"run", train_pool="missing", dev_pool="missing")
    with pytest.raises(ValueError, match="exactly five or ten"):
        register(tmp_path/"registration.json", "missing", "missing", [0])
    assert not (tmp_path/"run").exists()


def test_replay_json_roundtrip_and_overwrite_guard(tmp_path):
    replay = play_episode(SiegeEnv(base_seed=42, record=True), RandomAgent(5))
    path = tmp_path/"replay.json"
    save_replay(replay, path)
    assert load_replay(path) == json.loads(json.dumps(replay))
    with pytest.raises(FileExistsError):
        save_replay(replay, path)

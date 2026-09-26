"""Synthetic metadata fixtures test the review gate, not policy performance."""
from dataclasses import asdict
import json

import pytest

from experiments.pilot import review_pilots, validate_review
from experiments.provenance import runtime_manifest, sha256, source_hashes
from experiments.protocol import register
from experiments.storage import atomic_json
from train.train_ppo import study_settings


def fixtures(tmp_path):
    train_pool, dev_pool = tmp_path / "train.json", tmp_path / "dev.json"
    train_pool.write_text("synthetic training-pool hash fixture")
    dev_pool.write_text("synthetic development-pool hash fixture")
    runs, policies = [], []
    for condition in ("P", "PA"):
        path = tmp_path / condition
        path.mkdir()
        (path / "model.zip").write_bytes(b"synthetic identity; never deserialised")
        atomic_json(path / "development.json", {"records": [{"timesteps": 0}, {"timesteps": 50000}]})
        (path / "episodes.jsonl").write_text('{"spent": 199}\n{"spent": 0}\n')
        manifest = {**json.loads(json.dumps(runtime_manifest())),
            "settings": asdict(study_settings(condition=condition, mode="pilot")),
            "status": "completed", "model_sha256": sha256(path / "model.zip"),
            "actual_steps": 57344, "placement_actual": 32768, "joint_actual": 24576, "wall_seconds": 100,
            "pools": {"train": {"sha256": sha256(train_pool)}, "dev": {"sha256": sha256(dev_pool)}}}
        atomic_json(path / "manifest.json", manifest)
        runs.append(path)
        policies.append({"condition": condition, "learner_seed": 100, "model_sha256": manifest["model_sha256"],
                         "manifest_sha256": sha256(path / "manifest.json")})
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    (evaluation / "episodes.csv").write_text("synthetic CSV identity fixture")
    atomic_json(evaluation / "evaluation.json", {"mode": "development", "status": "completed",
        "views": list(range(8)), "episodes_sha256": sha256(evaluation / "episodes.csv"),
        "source_hashes": source_hashes(), "policies": policies, "identity": {"pool_sha256": sha256(dev_pool)}})
    return runs, evaluation, train_pool, dev_pool


def test_pilot_review_needs_real_decision_and_unchanged_evidence(tmp_path):
    runs, evaluation, train_pool, dev_pool = fixtures(tmp_path)
    review_path = tmp_path / "review.json"
    preview = review_pilots(runs, evaluation, parent=tmp_path)
    assert preview["decision"] == "pending_team_review"
    assert preview["planned_full_learners"] == 20
    assert preview["summaries"][0]["empty_army_fraction"] == .5
    atomic_json(review_path, preview)
    with pytest.raises(ValueError, match="recorded team"):
        validate_review(review_path, train_pool, dev_pool)
    approved = review_pilots(runs, evaluation, parent=tmp_path, note="Test fixture only; no actual experiment approved.")
    atomic_json(review_path, approved)
    assert validate_review(review_path, train_pool, dev_pool) == approved
    (evaluation / "episodes.csv").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        validate_review(review_path, train_pool, dev_pool)


def test_smoke_cannot_pass_pilot_review(tmp_path):
    runs, evaluation, _, _ = fixtures(tmp_path)
    for path in runs:
        manifest = json.loads((path / "manifest.json").read_text())
        manifest["settings"]["mode"] = "smoke"
        atomic_json(path / "manifest.json", manifest)
    with pytest.raises(ValueError):
        review_pilots(runs, evaluation, parent=tmp_path, note="Cannot approve smoke metadata.")


def test_registration_requires_review_before_accessing_pools(tmp_path):
    with pytest.raises(ValueError, match="pilot review"):
        register(tmp_path / "registration.json", "missing", "missing")
    assert not (tmp_path / "registration.json").exists()

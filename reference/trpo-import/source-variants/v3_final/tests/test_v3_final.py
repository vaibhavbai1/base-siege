"""Checks for the V3 final evaluation. No training, no pool generation."""
import json
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from env import config as C
from env.generator import generate_base
from train.features import SiegeFeatures
from train.train_ppo import make_env
from trpo import config as T
from trpo.maskable_trpo import MaskableTRPO
from v2_final.runner import acting_seed as v2_acting_seed
from v2_phases.runner import PhasePolicy
from v3_final import runner
from v3_final.analysis import summaries

ROOT = Path(__file__).resolve().parents[2]


def test_protocol_is_wellformed():
    spec = runner.protocol()
    assert spec["conditions"] == ["P", "PA", "T"]
    assert spec["mode"] == "DS"
    assert len(spec["views"]) == 8
    assert len(spec["learner_seeds"]) == 10
    low, high = spec["candidate_seed_interval"]
    assert (high - low + 1) * len(spec["motifs"]) == spec["candidate_layouts"]


def test_v3_pool_does_not_reuse_the_spent_v2_seeds():
    """The V2 final pool is spent; V3 must be a disjoint interval."""
    v2 = json.loads((ROOT / "v2_final/protocol.json").read_text())
    v3 = runner.protocol()
    v2_low, v2_high = v2["candidate_seed_interval"]
    v3_low, v3_high = v3["candidate_seed_interval"]
    assert set(range(v3_low, v3_high + 1)).isdisjoint(range(v2_low, v2_high + 1))


def test_acting_seed_is_deterministic_and_namespaced():
    base = generate_base(21200, "closed_ring")
    first = runner.acting_seed(41001, base, 3)
    assert first == runner.acting_seed(41001, base, 3)
    assert first != runner.acting_seed(41002, base, 3)
    assert first != runner.acting_seed(41001, base, 4)
    # A different namespace from V2, so the two studies never share a seed stream.
    assert first != v2_acting_seed(41001, base, 3)


def test_pool_candidates_cover_every_motif():
    spec = runner.protocol()
    candidates = runner.pool_candidates(spec)
    assert len(candidates) == spec["candidate_layouts"]
    assert {m for _s, m in candidates} == set(spec["motifs"])


def test_reference_models_merges_both_registries():
    models = runner.reference_models(ROOT)
    conditions = {m["condition"] for m in models}
    assert {"P", "PA"} <= conditions
    v2 = json.loads((ROOT / "ui/v2_checkpoints.json").read_text())
    assert sum(1 for m in models if m["condition"] in ("P", "PA")) == len(v2["models"])


def test_prepare_refuses_when_learners_are_missing():
    """A freeze must not be possible before every condition is trained."""
    with pytest.raises(ValueError, match="Missing approved checkpoints"):
        runner.prepare(root=ROOT)


def test_trpo_policy_drives_the_frozen_phase_policy():
    """PhasePolicy is what the evaluator uses; it must accept a TRPO policy."""
    ctors = [partial(make_env, seed=0, rank=0, symmetry=False, fixed_army=True)]
    env = DummyVecEnv(ctors)
    kwargs = T.model_kwargs(device="cpu", seed=100, n_steps=64)
    kwargs["batch_size"] = 64
    policy_kwargs = dict(T.POLICY_KWARGS)
    policy_kwargs["features_extractor_class"] = SiegeFeatures
    model = MaskableTRPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, **kwargs)
    model.policy.set_training_mode(False)

    acting = PhasePolicy(model.policy, "DS")
    env.close()

    # The evaluator drives a bare SiegeEnv, not the training Monitor wrapper.
    from env.siege_env import SiegeEnv

    inner = SiegeEnv(motifs=C.MOTIFS, allow_final=True)
    obs, _ = inner.reset(seed=0)
    for _ in range(40):
        mask = np.asarray(inner.action_masks())
        action = acting.act(obs, mask)
        assert mask[action], "PhasePolicy returned an illegal action for the TRPO policy"
        obs, _reward, terminated, truncated, _info = inner.step(action)
        if terminated or truncated:
            break
    inner.close()


def _row(condition, seed, motif, destruction):
    return {
        "condition": condition, "learner_seed": seed, "transitions": 4194304,
        "base_seed": 1, "motif": motif, "view": 0, "sample_seed": 41001,
        "mode": "heuristic" if condition == "heuristic" else "DS",
        "acting_seed": 0, "destruction": destruction, "full_clear": destruction == 1,
        "town_hall": 1.0, "episode_return": 1.0, "spent": 100, "tick": 50,
        "decisions": 10, "zero_destruction": float(destruction == 0),
        "finish_reason": "cleared",
    }


def test_summaries_pair_conditions_by_learner_seed():
    spec = runner.protocol()
    rows = []
    for seed in range(10):
        rows.append(_row("P", seed, "closed_ring", 0.90))
        rows.append(_row("PA", seed, "closed_ring", 0.92))
        rows.append(_row("T", seed, "closed_ring", 0.88))
    rows.append(_row("heuristic", -1, "closed_ring", 0.50))

    models, conditions, paired, comparisons = summaries(rows, spec)
    by_condition = {c["condition"]: c for c in conditions if c["scope"] == "all_final"}
    assert set(by_condition) == {"P", "PA", "T", "heuristic"}
    assert by_condition["T"]["learner_runs"] == 10

    primary = [c for c in comparisons if c["scope"] == "all_final" and c["role"] == "primary"]
    assert len(primary) == 1
    assert primary[0]["comparison"] == "T_minus_P"
    assert primary[0]["learner_observations"] == 10
    assert primary[0]["mean_destruction_change_points"] == pytest.approx(-2.0)

    labels = {c["comparison"] for c in comparisons if c["scope"] == "all_final"}
    assert {"PA_minus_P", "T_minus_PA", "T_minus_heuristic"} <= labels


def test_summaries_split_transfer_from_familiar():
    spec = runner.protocol()
    rows = [_row("T", s, "closed_ring", 0.9) for s in range(10)]
    rows += [_row("T", s, "layered_core", 0.8) for s in range(10)]
    models, conditions, _paired, _comparisons = summaries(rows, spec)
    scopes = {c["scope"] for c in conditions}
    assert {"all_final", "familiar", "transfer", "motif:layered_core"} <= scopes
    transfer = next(c for c in conditions if c["scope"] == "transfer" and c["condition"] == "T")
    familiar = next(c for c in conditions if c["scope"] == "familiar" and c["condition"] == "T")
    assert transfer["destruction_percent"] == pytest.approx(80.0)
    assert familiar["destruction_percent"] == pytest.approx(90.0)


def test_output_columns_match_v2():
    """V2 and V3 tables must be readable side by side without translation."""
    import csv
    spec = runner.protocol()
    rows = []
    for seed in range(10):
        for condition in ("P", "PA", "T"):
            rows.append(_row(condition, seed, "closed_ring", 0.92))
    rows.append(_row("heuristic", -1, "closed_ring", 0.57))
    models, conditions, paired, comparisons = summaries(rows, spec)

    artifacts = ROOT / "artifacts/version2-final-evaluation"
    for name, mine in (("condition_summary.csv", conditions),
                       ("comparisons.csv", comparisons),
                       ("model_summary.csv", models)):
        path = artifacts / name
        if not path.is_file():
            continue
        v2_columns = next(csv.reader(path.open()))
        missing = [c for c in v2_columns if c not in mine[0]]
        assert not missing, f"{name} is missing V2 columns: {missing}"

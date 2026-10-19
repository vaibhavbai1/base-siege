"""Checks for the acting-mode probe. No long evaluation runs here."""
import json
from pathlib import Path

import pytest

from v2_phases.runner import MODES
from v3_final import runner as final
from v3_probe import runner

ROOT = Path(__file__).resolve().parents[2]


def test_probe_varies_purchasing_only():
    """The whole claim rests on this: DS and SS differ in the purchasing phase
    and agree on deployment. If that stops being true the probe proves nothing."""
    baseline = MODES[runner.BASELINE_MODE]
    probe = MODES[runner.PROBE_MODE]
    assert baseline[0] != probe[0], "purchasing must differ between the two modes"
    assert baseline[1] == probe[1], "deployment must be identical between the two modes"
    assert probe[1] is False, "deployment is sampled in both modes"


def test_probe_writes_outside_the_frozen_evaluation():
    assert runner.OUTPUT != final.OUTPUT
    assert final.OUTPUT.name not in runner.OUTPUT.parts


def test_unit_key_includes_mode():
    """Probe rows must never be mistakable for the DS rows."""
    target = {"condition": "T", "learner_seed": 1, "transitions": 4194304}
    base = final.generate_base(21200, "closed_ring")
    ds = runner.unit_key(target, base, 0, 41001, "DS")
    ss = runner.unit_key(target, base, 0, 41001, "SS")
    assert ds != ss
    assert ":DS:" in ds and ":SS:" in ss


def test_acting_seed_is_shared_with_the_final_evaluation():
    """Holding the RNG stream fixed across modes is what makes the comparison
    paired; a fresh namespace here would confound mode with draw."""
    base = final.generate_base(21200, "closed_ring")
    assert runner.episode.__module__ == "v3_probe.runner"
    assert final.acting_seed(41001, base, 3) == final.acting_seed(41001, base, 3)


def test_frozen_contract_refuses_an_incomplete_evaluation(tmp_path):
    """A probe against a half-finished DS run would compare against numbers
    that do not exist yet."""
    (tmp_path / "freeze.json").write_text(json.dumps({"contract": {"protocol_sha256": "x"}}))
    with pytest.raises(ValueError, match="protocol.json changed|has not completed"):
        runner.frozen_contract(tmp_path)


def test_frozen_contract_requires_a_freeze(tmp_path):
    with pytest.raises(ValueError, match="No freeze"):
        runner.frozen_contract(tmp_path)


def test_probe_targets_excludes_the_heuristic():
    contract = {"models": [
        {"condition": "P", "learner_seed": 0, "run": "a"},
        {"condition": "T", "learner_seed": 0, "run": "b"},
    ]}
    selected = runner.probe_targets(contract, {"P", "T"})
    assert {m["condition"] for m in selected} == {"P", "T"}
    assert all(m["condition"] != "heuristic" for m in selected)


def test_probe_targets_refuses_a_missing_condition():
    contract = {"models": [{"condition": "P", "learner_seed": 0, "run": "a"}]}
    with pytest.raises(ValueError, match="no models for"):
        runner.probe_targets(contract, {"P", "T"})


def test_per_learner_averages_within_learner():
    rows = [
        {"condition": "T", "learner_seed": "1", "destruction": "0.5"},
        {"condition": "T", "learner_seed": "1", "destruction": "0.7"},
        {"condition": "T", "learner_seed": "2", "destruction": "0.9"},
    ]
    out = runner.per_learner(rows)
    assert out[("T", 1)] == pytest.approx(60.0)
    assert out[("T", 2)] == pytest.approx(90.0)

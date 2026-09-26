"""These tests also run from the standalone deployment bundle."""
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
from streamlit.testing.v1 import AppTest
from ui.deployment import asset_paths, load_cpu_policy

APP = ROOT / "ui/streamlit_app.py"


def open_app():
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    return app


def select(app, label, value):
    next(widget for widget in app.selectbox if widget.label == label).set_value(value).run(timeout=30)
    assert not app.exception


def test_public_app_renders_without_published_models():
    app = open_app()
    assert app.title[0].value == "Base Siege"
    assert any("CPU inference only" in caption.value for caption in app.caption)
    select(app, "Mode", "Policy replay")
    select(app, "Policy", "Saved Maskable PPO")
    if not asset_paths(ROOT, "model", public=True):
        assert any("No trained policy has been published" in item.value for item in app.info)
        assert next(button for button in app.button if button.label == "Generate replay").disabled
    else:
        models = next(widget for widget in app.selectbox if widget.label == "Published model")
        assert all(name.startswith("demo_assets/") for name in models.options)


@pytest.mark.parametrize("policy", ["Heuristic", "Valid random"])
def test_public_baseline_replay_completes(policy):
    app = open_app()
    select(app, "Mode", "Policy replay")
    select(app, "Policy", policy)
    next(button for button in app.button if button.label == "Generate replay").click().run(timeout=30)
    assert not app.exception
    replay = app.session_state["replay"]
    assert replay["policy"]["name"] == policy
    assert replay["summary"]["finish_reason"]
    assert replay["snapshots"]
    assert replay["summary"]["base_seed"] == 42


def test_public_evidence_modes_only_offer_demo_assets():
    app = open_app()
    for mode in ("Experiment results", "Saved replay"):
        select(app, "Mode", mode)
        for widget in app.selectbox:
            if widget.label in ("Run manifest", "Evaluation table", "Replay file"):
                assert all(value.startswith("demo_assets/") for value in widget.options)


def test_sessions_have_independent_games():
    first, second = open_app(), open_app()
    first.session_state["game"].step(0)
    assert first.session_state["game"].state.spent > 0
    assert second.session_state["game"].state.spent == 0
    assert first.session_state["game"] is not second.session_state["game"]


def test_reserved_seed_is_rejected():
    app = open_app()
    app.number_input[0].set_value(10000).run(timeout=30)
    assert not app.exception
    assert any("training seeds" in item.value for item in app.error)


@pytest.mark.parametrize("kind,relative,private", [
    ("model", "models/P/model.zip", "runs/P/model.zip"),
    ("manifest", "models/P/manifest.json", "artifacts/P/manifest.json"),
    ("replay", "replays/heuristic.json", "artifacts/example-replay.json"),
    ("evaluation", "evaluations/development.csv", "artifacts/episodes.csv"),
])
def test_public_asset_discovery_never_reads_local_runs(tmp_path, kind, relative, private):
    allowed, hidden = tmp_path / "demo_assets" / relative, tmp_path / private
    for path in (allowed, hidden):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test")
    assert asset_paths(tmp_path, kind, public=True) == [allowed]
    assert hidden in asset_paths(tmp_path, kind, public=False)


def test_public_asset_discovery_rejects_escaped_symlink(tmp_path):
    outside = tmp_path / "private/model.zip"
    outside.parent.mkdir()
    outside.write_text("not public")
    link = tmp_path / "demo_assets/models/leak"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside.parent, target_is_directory=True)
    assert not asset_paths(tmp_path, "model", public=True)


def test_cpu_policy_cache_checks_digest_and_reuses_weights(tmp_path):
    from sb3_contrib import MaskablePPO
    from env.siege_env import SiegeEnv
    from train.features import SiegeFeatures
    env = SiegeEnv(base_seed=42, motifs=("closed_ring",))
    model = MaskablePPO("MultiInputPolicy", env, n_steps=2, batch_size=2, device="cpu",
                       policy_kwargs={"features_extractor_class": SiegeFeatures, "normalize_images": False})
    path = tmp_path / "model.zip"
    model.save(path)
    stamp = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    load_cpu_policy.clear()
    policy = load_cpu_policy(str(path), stamp.st_mtime_ns, stamp.st_size, digest)
    assert policy is load_cpu_policy(str(path), stamp.st_mtime_ns, stamp.st_size, digest)
    assert str(policy.model.device) == "cpu"
    obs, _ = env.reset(seed=0)
    action, _ = policy.predict(obs, action_masks=env.action_masks(), deterministic=True)
    assert env.action_masks()[int(action)]
    with pytest.raises(ValueError, match="checksum"):
        load_cpu_policy(str(path), stamp.st_mtime_ns, stamp.st_size, "incorrect")
    load_cpu_policy.clear()
    env.close()

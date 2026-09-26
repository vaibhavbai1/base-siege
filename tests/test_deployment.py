import hashlib
import json
from pathlib import Path
import tomllib
import zipfile

import pytest
import yaml

from tools.build_deploy_bundle import build_bundle
from tools.export_demo_model import export_model

ROOT = Path(__file__).resolve().parents[1]


def test_cloud_config_keeps_provider_defaults_and_protections():
    config = tomllib.loads((ROOT / ".streamlit/config.toml").read_text())
    assert "port" not in config["server"]
    assert "address" not in config["server"]
    assert config["server"]["enableXsrfProtection"]
    assert config["server"]["enableCORS"]
    assert config["client"]["showErrorDetails"] == "none"


def test_bundle_is_an_allowlist_and_does_not_overwrite(tmp_path):
    output = tmp_path / "public-demo"
    result = build_bundle(output)
    paths = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    assert "ui/streamlit_app.py" in paths
    assert ".github/workflows/ci.yml" in paths
    assert "train/features.py" in paths
    assert "train/train_ppo.py" not in paths
    assert "train/checkpoints.py" not in paths
    assert "uv.lock" not in paths
    assert not any(path.startswith(("artifacts/", "runs/", "reference/", "notebooks/", ".venv/")) for path in paths)
    with zipfile.ZipFile(result["zip"]) as archive:
        assert set(archive.namelist()) == paths
        assert ".streamlit/config.toml" in archive.namelist()
    manifest = json.loads((output / "bundle-manifest.json").read_text())
    for path, digest in manifest["files"].items():
        assert hashlib.sha256((output / path).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        build_bundle(output)


def test_ci_has_read_only_permissions_and_no_deployment_secrets():
    for path in (ROOT / ".github/workflows/ci.yml", ROOT / "deployment/streamlit-ci.yml"):
        workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        assert workflow["permissions"] == {"contents": "read"}
        assert set(workflow["on"]) == {"push", "pull_request", "workflow_dispatch"}
        assert "secrets." not in path.read_text()
        assert "git push" not in path.read_text()


def test_runtime_dependencies_do_not_install_notebook_or_cuda_stack():
    requirements = (ROOT / "ui/requirements.txt").read_text()
    assert "download-r2.pytorch.org/whl/cpu/" in requirements
    assert "#sha256=" in requirements
    for package in ("tensorboard", "jupyterlab", "ipykernel", "nvidia-", "cuda-toolkit"):
        assert package not in requirements


def test_export_preserves_predictions_but_not_training_metadata(tmp_path):
    import torch
    from sb3_contrib import MaskablePPO
    from env.siege_env import SiegeEnv
    from train.features import SiegeFeatures

    run = tmp_path / "run"
    run.mkdir()
    env = SiegeEnv(base_seed=42, motifs=("closed_ring",))
    source = MaskablePPO("MultiInputPolicy", env, n_steps=2, batch_size=2, device="cpu",
                        policy_kwargs={"features_extractor_class": SiegeFeatures, "normalize_images": False,
                                       "net_arch": {"pi": [128], "vf": [128]}})
    source.tensorboard_log = "/private/training/location"
    source.save(run / "model.zip")
    manifest = {"status": "completed", "actual_steps": 0,
                "settings": {"condition": "P", "mode": "smoke", "seed": 0},
                "model_sha256": hashlib.sha256((run / "model.zip").read_bytes()).hexdigest(),
                "private_field": "/private/training/location"}
    (run / "manifest.json").write_text(json.dumps(manifest))
    destination = tmp_path / "export"
    public = export_model(run, destination)
    assert "private_field" not in public
    assert public["evidence_scope"] == "software smoke test"
    assert {p.name for p in destination.iterdir()} == {"manifest.json", "model.zip"}
    loaded = MaskablePPO.load(destination / "model.zip", device="cpu")
    assert loaded.tensorboard_log is None
    assert not loaded.policy.optimizer.state
    for key, value in source.policy.state_dict().items():
        assert torch.equal(value, loaded.policy.state_dict()[key])
    for seed in (0, 42, 10200):
        obs, _ = env.reset(seed=seed)
        kwargs = {"action_masks": env.action_masks(), "deterministic": True}
        assert int(source.predict(obs, **kwargs)[0]) == int(loaded.predict(obs, **kwargs)[0])
    with pytest.raises(FileExistsError):
        export_model(run, destination)
    manifest["model_sha256"] = "incorrect"
    (run / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        export_model(run, tmp_path / "bad-export")
    assert not (tmp_path / "bad-export").exists()
    env.close()

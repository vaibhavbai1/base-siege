"""Read-only demo assets and a bounded, CPU-only policy cache."""
import hashlib
from pathlib import Path
import threading

import streamlit as st


def asset_paths(root, kind, *, public):
    root = Path(root)
    patterns = {
        "manifest": "models/*/manifest.json", "model": "models/*/model.zip",
        "replay": "replays/*.json", "evaluation": "evaluations/*.csv",
    }
    assets = root / "demo_assets"
    paths = [p for p in assets.glob(patterns[kind])
             if p.is_file() and p.resolve().is_relative_to(assets.resolve())
             and not p.is_symlink()]
    if not public:
        pattern = {"manifest": "**/manifest.json", "model": "**/model.zip",
                   "replay": "**/*replay*.json", "evaluation": "**/episodes.csv"}[kind]
        paths.extend((root / "artifacts").glob(pattern))
        if kind == "model":
            paths.extend((root / "runs").glob(pattern))
        if kind == "manifest":
            paths.extend((root / "runs").glob(pattern))
        if kind == "evaluation":
            paths.extend((root / "artifacts").glob("**/sampled_episodes.csv"))
            paths.extend((root / "artifacts").glob("**/combined_episodes.csv"))
    return sorted(set(paths))


class CpuPolicy:
    """Shared weights, serial prediction; each visitor keeps a separate environment."""

    def __init__(self, model):
        self.model = model
        self.lock = threading.Lock()

    def predict(self, *args, **kwargs):
        with self.lock:
            return self.model.predict(*args, **kwargs)


@st.cache_resource(max_entries=2, show_spinner=False)
def load_cpu_policy(path, modified_ns, size_bytes, expected_sha256=None, expected_steps=None):
    # File stamp and digest are cache keys, so a replaced export is reloaded.
    path = Path(path)
    if expected_sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("The published policy checksum does not match its manifest.")
    import torch
    from sb3_contrib import MaskablePPO
    from ui.policy_playback import isolated_rng
    torch.set_num_threads(1)
    # Loading a second model must not disturb another session's sampled episode.
    with isolated_rng(0):
        model = MaskablePPO.load(path, device="cpu")
        if expected_steps is not None and model.num_timesteps != expected_steps:
            raise ValueError('Model timestep counter differs from its published manifest.')
        model.policy.set_training_mode(False)
        return CpuPolicy(model)

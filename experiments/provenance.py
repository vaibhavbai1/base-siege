"""Small, inspectable manifests instead of implicit experiment state."""
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from env import config as C

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {str(p.relative_to(ROOT)): sha256(p) for folder in ("env", "agents", "train", "experiments")
            for p in sorted((ROOT / folder).glob("*.py"))}


def config_snapshot():
    return {name: value for name, value in vars(C).items() if name.isupper()}


def runtime_manifest():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {p: version(p) for p in ("numpy", "torch", "gymnasium", "stable-baselines3", "sb3-contrib", "streamlit")},
            "source_hashes": source_hashes(), "config": config_snapshot()}


def write_json(path, value, *, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)

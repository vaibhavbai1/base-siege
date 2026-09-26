"""Committed, checksummed PPO recovery points at completed update boundaries."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import random
import tempfile
import uuid

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from experiments.provenance import sha256
from experiments.storage import atomic_bytes, atomic_json, run_lock, sync_directory

SCHEMA = 3


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class SnapshotMonitor(Monitor):
    """Serialize the worker, including its Gym RNG, D4 view and current battle."""

    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        pool = self.unwrapped.pool
        self._pool_digest = hashlib.sha256(
            json.dumps([json.loads(base.to_json()) for base in pool], sort_keys=True).encode()
        ).hexdigest() if pool is not None else None

    def checkpoint_state(self):
        if self.results_writer is not None:
            raise RuntimeError("SnapshotMonitor requires centralized episode logging")
        pool, digest = self.unwrapped.pool, self._pool_digest
        class StatePickler(pickle.Pickler):
            def persistent_id(self, obj):
                return ("frozen_pool", digest) if pool is not None and obj is pool else None
        buffer = io.BytesIO()
        StatePickler(buffer, protocol=pickle.HIGHEST_PROTOCOL).dump(self.__dict__)
        return buffer.getvalue()

    def restore_checkpoint_state(self, state):
        pool, digest = self.unwrapped.pool, self._pool_digest
        class StateUnpickler(pickle.Unpickler):
            def persistent_load(self, identity):
                if identity != ("frozen_pool", digest) or pool is None:
                    raise ValueError("Checkpoint frozen layout pool changed")
                return pool
        # Only immutable pool data is external; the complete dynamic worker state is saved.
        restored = StateUnpickler(io.BytesIO(state)).load()
        self.__dict__.clear()
        self.__dict__.update(restored)


class RecoverablePPO(MaskablePPO):
    """Use upstream PPO unchanged; notify persistence only after its update ends."""

    after_update = None

    def train(self):
        super().train()
        if self.after_update is not None:
            self.after_update(self)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["after_update"]


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG topology changed; use the same number of visible GPUs")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def capture_vector(vector):
    return {"workers": vector.env_method("checkpoint_state"),
            "seeds": deepcopy(vector._seeds), "options": deepcopy(vector._options),
            "reset_infos": deepcopy(vector.reset_infos)}


def restore_vector(vector, state):
    if len(state["workers"]) != vector.num_envs:
        raise ValueError("Checkpoint worker count does not match the run")
    for index, worker in enumerate(state["workers"]):
        vector.env_method("restore_checkpoint_state", worker, indices=index)
    vector._seeds = state["seeds"]
    vector._options = state["options"]
    vector.reset_infos = state["reset_infos"]


def save_checkpoint(model, vector, directory, *, phase, contract_hash, development=None):
    """Publish a whole generation before replacing LATEST.json; keep older ones."""
    directory = Path(directory)
    checkpoint_root = directory / "checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    name = f"checkpoint-{model.num_timesteps:012d}-{phase}-{uuid.uuid4().hex[:10]}"
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=checkpoint_root))
    model.save(staging / "model.zip")
    state = {"schema_version": SCHEMA, "rng": capture_rng(),
             "vector": capture_vector(vector), "development": development,
             "last_obs": deepcopy(model._last_obs),
             "last_episode_starts": deepcopy(model._last_episode_starts)}
    torch.save(state, staging / "state.pt")
    episodes = directory / "episodes.jsonl"
    (staging / "episodes.jsonl").write_bytes(episodes.read_bytes() if episodes.exists() else b"")
    files = {path.name: sha256(path) for path in staging.iterdir() if path.is_file()}
    metadata = {"schema_version": SCHEMA, "created_utc": utc_now(), "phase": phase,
                "actual_steps": model.num_timesteps, "optimizer_updates": model._n_updates,
                "contract_sha256": contract_hash, "files": files,
                "boundary": "after_completed_update" if model.num_timesteps else "initial_state"}
    (staging / "checkpoint.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    for path in staging.iterdir():
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    sync_directory(staging)
    committed = checkpoint_root / name
    os.rename(staging, committed)
    sync_directory(checkpoint_root)
    atomic_json(directory / "LATEST.json", {"schema_version": SCHEMA, "checkpoint": name,
        "checkpoint_sha256": sha256(committed / "checkpoint.json"), "actual_steps": model.num_timesteps})
    return committed, metadata


def verify_checkpoint(path):
    path = Path(path)
    metadata = json.loads((path / "checkpoint.json").read_text())
    if metadata.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported checkpoint schema")
    if set(metadata["files"]) != {"model.zip", "state.pt", "episodes.jsonl"}:
        raise ValueError("Incomplete checkpoint file set")
    for name, digest in metadata["files"].items():
        if sha256(path / name) != digest:
            raise ValueError(f"Checksum mismatch: {name}")
    if metadata["phase"] not in ("placement", "joint") or metadata["actual_steps"] < 0:
        raise ValueError("Invalid checkpoint progress")
    return metadata


def latest_checkpoint(directory):
    """Recover from an interrupted write or bad pointer using a verified generation."""
    directory = Path(directory)
    rejected = []
    candidates = []
    pointer = directory / "LATEST.json"
    if pointer.exists():
        try:
            data = json.loads(pointer.read_text())
            name = data["checkpoint"]
            if Path(name).name != name or not name.startswith("checkpoint-"):
                raise ValueError("Unsafe checkpoint pointer")
            path = directory / "checkpoints" / name
            if sha256(path / "checkpoint.json") != data["checkpoint_sha256"]:
                raise ValueError("Pointer checksum mismatch")
            candidates.append(path)
        except (OSError, ValueError, KeyError) as exc:
            rejected.append(f"LATEST.json: {exc}")
    remaining = sorted((directory / "checkpoints").glob("checkpoint-*"),
                       key=lambda p: (p.name.split("-")[1], (p / "checkpoint.json").stat().st_mtime_ns
                                      if (p / "checkpoint.json").exists() else 0), reverse=True)
    # Directory rename commits a generation. A crash before LATEST is replaced
    # must not hide a newer, fully committed generation.
    candidates = sorted(set(candidates + remaining),
                        key=lambda p: (p.name.split("-")[1], (p / "checkpoint.json").stat().st_mtime_ns
                                       if (p / "checkpoint.json").exists() else 0), reverse=True)
    for path in candidates:
        try:
            return path, verify_checkpoint(path), rejected
        except (OSError, ValueError, KeyError) as exc:
            rejected.append(f"{path.name}: {exc}")
    raise ValueError("No intact committed checkpoint is available: " + "; ".join(rejected))


def restore_checkpoint(path, model, vector):
    verify_checkpoint(path)
    state = torch.load(Path(path) / "state.pt", map_location="cpu", weights_only=False)
    restore_vector(vector, state["vector"])
    model._last_obs = state["last_obs"]
    model._last_episode_starts = state["last_episode_starts"]
    return state

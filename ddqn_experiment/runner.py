"""Training contracts and atomic recovery checkpoints; never writes PPO outputs."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import random
import shutil
import signal
import tempfile
from time import perf_counter
import uuid

import numpy as np
import torch

from env.siege_env import SiegeEnv
from experiments.pools import load_pool
from experiments.provenance import sha256, source_hashes
from experiments.storage import atomic_json, run_lock, sync_directory
from ddqn_experiment.core import (Settings, QNetwork, ReplayBuffer, choose_action,
                                  epsilon_at, bootstrap_mask, update)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 1


def now():
    return datetime.now(timezone.utc).isoformat()


def sources():
    return {**source_hashes(), **{str(p.relative_to(ROOT)): sha256(p)
            for p in sorted((ROOT / "ddqn_experiment").glob("*.py"))}}


def runtime():
    return {"python": platform.python_version(), "packages": {n: version(n) for n in
            ("numpy", "torch", "gymnasium", "stable-baselines3", "sb3-contrib")}}


def configure(seed, device):
    if device not in ("cpu", "cuda"):
        raise ValueError("Device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def checked_pool(path, split):
    data = json.loads(Path(path).read_text())
    if data.get("filter") != "structure-and-difficulty":
        raise ValueError("Experiments require the existing difficulty-filtered pool")
    return load_pool(path, required_split=split)


def make_contract(settings, seed, mode, device, train_pool, dev_pool):
    settings.validate()
    if mode not in ("pilot", "main", "smoke") or seed < 0:
        raise ValueError("Invalid mode or seed")
    if mode == "pilot" and seed in range(10):
        raise ValueError("Use pilot seed 100 (or another seed outside 0–9)")
    if mode == "main" and seed not in range(10):
        raise ValueError("Main learner seeds are 0–9")
    if mode != "smoke" and (train_pool is None or dev_pool is None):
        raise ValueError("Provide frozen training and development pools")
    pools = {}
    for split, path in (("train", train_pool), ("dev", dev_pool)):
        if path is not None:
            bases = checked_pool(path, split)
            pools[split] = {"sha256": sha256(path), "layouts": len(bases)}
    return {"schema": SCHEMA, "algorithm": "masked-double-dqn", "condition": "no-augmentation",
            "mode": mode, "seed": seed, "settings": asdict(settings), "device": device,
            "runtime": runtime(), "sources": sources(), "pools": pools,
            "n_envs": 1, "evaluation_acting": "greedy-legal-lowest-index-tie",
            "curriculum": "placement then joint; reset episode and replay buffer at boundary",
            "bootstrap": "bootstrap safety-guard truncation; no bootstrap after termination"}


def append_json(path, row):
    with Path(path).open("a") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")


@contextmanager
def stop_signals(stop):
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    for s in previous:
        signal.signal(s, lambda *_: stop.update(requested=True))
    try:
        yield
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


def rng_state(rng):
    return {"generator": deepcopy(rng.bit_generator.state), "python": random.getstate(),
            "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(rng, state):
    rng.bit_generator.state = state["generator"]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Visible GPU count changed")
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


def verify_generation(folder, contract_hash):
    meta = json.loads((folder / "checkpoint.json").read_text())
    if meta.get("schema") != SCHEMA or meta["contract_sha256"] != contract_hash:
        raise ValueError("Checkpoint contract mismatch")
    if set(meta["files"]) != {"state.pt", "policy.pt"}:
        raise ValueError("Incomplete checkpoint generation")
    for name, digest in meta["files"].items():
        if sha256(folder / name) != digest:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    return meta


def load_latest(output):
    output = Path(output)
    pointer = json.loads((output / "LATEST.json").read_text())
    if Path(pointer["checkpoint"]).name != pointer["checkpoint"]:
        raise ValueError("Invalid checkpoint path")
    folder = output / "checkpoints" / pointer["checkpoint"]
    if sha256(folder / "checkpoint.json") != pointer["metadata_sha256"]:
        raise ValueError("Checkpoint metadata checksum mismatch")
    verify_generation(folder, sha256(output / "contract.json"))
    # Recovery files are trusted local files, containing the simulator state.
    return torch.load(folder / "state.pt", map_location="cpu", weights_only=False)


def commit(output, contract_hash, online, target, optimizer, replay, env, obs, rng, progress):
    pointer = output / "LATEST.json"
    existing_policy = output / "policies" / f"step-{progress['steps']:012d}.pt"
    if (pointer.exists() and existing_policy.exists()
            and json.loads(pointer.read_text())["actual_steps"] == progress["steps"]):
        return  # Avoid two identical full-buffer generations at a pause boundary.
    root = output / "checkpoints"
    root.mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=root))
    offsets = {}
    for name in ("episodes.jsonl", "progress.jsonl"):
        path = output / name
        with path.open("ab") as f:
            f.flush()
            os.fsync(f.fileno())
        offsets[name] = path.stat().st_size
    state = {"online": online.state_dict(), "target": target.state_dict(),
             "optimizer": optimizer.state_dict(), "replay": replay.state_dict(),
             "env": {k: v for k, v in env.__dict__.items() if k != "pool"},
             "obs": obs, "rng": rng_state(rng), "progress": progress, "log_offsets": offsets}
    torch.save(state, staging / "state.pt")
    torch.save({"schema": SCHEMA, "algorithm": "masked-double-dqn", "steps": progress["steps"],
                "contract_sha256": contract_hash, "online": online.state_dict()}, staging / "policy.pt")
    metadata = {"schema": SCHEMA, "created_utc": now(), "actual_steps": progress["steps"],
                "contract_sha256": contract_hash,
                "files": {name: sha256(staging / name) for name in ("state.pt", "policy.pt")}}
    atomic_json(staging / "checkpoint.json", metadata)
    for path in staging.iterdir():
        with path.open("rb") as f:
            os.fsync(f.fileno())
    sync_directory(staging)
    folder = root / f"step-{progress['steps']:012d}-{uuid.uuid4().hex[:8]}"
    os.rename(staging, folder)
    sync_directory(root)
    atomic_json(output / "LATEST.json", {"checkpoint": folder.name,
                "metadata_sha256": sha256(folder / "checkpoint.json"), "actual_steps": progress["steps"]})
    # Keep compact, numbered policies for development curves, but only two full buffers.
    policy_dir = output / "policies"
    policy_dir.mkdir(exist_ok=True)
    policy = policy_dir / f"step-{progress['steps']:012d}.pt"
    if not policy.exists():
        temporary = policy.with_suffix(".tmp")
        shutil.copyfile(folder / "policy.pt", temporary)
        os.replace(temporary, policy)
        atomic_json(policy.with_suffix(".json"), {"sha256": sha256(policy),
                    "contract_sha256": contract_hash, "actual_steps": progress["steps"]})
    generations = sorted(root.glob("step-*"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for old in generations[2:]:
        shutil.rmtree(old)
    print(f"Saved {progress['steps']:,} transitions; {progress['episodes']:,} completed episodes", flush=True)


def train(output, settings, *, seed=100, mode="pilot", device="cpu", train_pool=None,
          dev_pool=None, resume=False, until=None):
    output = Path(output).resolve()
    configure(seed, device)
    contract = make_contract(settings, seed, mode, device, train_pool, dev_pool)
    until = settings.total_steps if until is None else until
    if type(until) is not int or not 0 < until <= settings.total_steps:
        raise ValueError("--until must be a positive absolute transition count within the plan")
    if resume:
        if json.loads((output / "contract.json").read_text()) != contract:
            raise ValueError("Resume refused: settings, source, pools or runtime changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "contract.json", contract)
        atomic_json(output / "created.json", {"created_utc": now(), "scope": "new post-PPO experiment"})
    with run_lock(output):
        pool = checked_pool(train_pool, "train") if train_pool is not None else None
        env = SiegeEnv(pool=pool, fixed_army=True)
        obs, _ = env.reset(seed=seed)
        online, target = QNetwork(env.observation_space).to(device), QNetwork(env.observation_space).to(device)
        target.load_state_dict(online.state_dict())
        target.eval()
        target.requires_grad_(False)
        optimizer = torch.optim.Adam(online.parameters(), lr=settings.learning_rate)
        replay = ReplayBuffer(settings.buffer_size, env.observation_space)
        rng = np.random.default_rng(seed)
        progress = {"steps": 0, "updates": 0, "episodes": 0, "phase": "placement",
                    "phase_steps": 0, "phase_cutoffs": 0, "recent_destruction": [],
                    "loop_seconds": 0.0, "last_loss": None}
        if resume:
            state = load_latest(output)
            online.load_state_dict(state["online"])
            target.load_state_dict(state["target"])
            optimizer.load_state_dict(state["optimizer"])
            for entry in optimizer.state.values():
                for key, value in entry.items():
                    if torch.is_tensor(value):
                        entry[key] = value.to(device)
            replay.load_state_dict(state["replay"])
            env.__dict__.update(state["env"])
            obs, progress = state["obs"], state["progress"]
            for name, offset in state["log_offsets"].items():
                path = output / name
                if not path.exists() or path.stat().st_size < offset:
                    raise ValueError("A committed training log is missing or shortened")
                with path.open("r+b") as f:
                    f.truncate(offset)
            restore_rng(rng, state["rng"])
        if until < progress["steps"]:
            raise ValueError("--until precedes the saved transition count")
        # Clear a handled pause request only on an explicitly resumed invocation.
        if resume and (output / "STOP_REQUESTED").exists():
            (output / "STOP_REQUESTED").unlink()
        contract_hash = sha256(output / "contract.json")
        stop = {"requested": False}
        append_json(output / "attempts.jsonl", {"started_utc": now(), "resume": resume,
                    "from_step": progress["steps"], "until": until})
        try:
            if not resume:
                commit(output, contract_hash, online, target, optimizer, replay, env, obs, rng, progress)
            with stop_signals(stop):
                while progress["steps"] < until and not stop["requested"]:
                    if (output / "STOP_REQUESTED").exists():
                        break
                    started = perf_counter()
                    action = choose_action(online, obs, env.action_masks(), rng,
                                           epsilon_at(progress["steps"], settings), device)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    replay.add(obs, action, reward, next_obs, terminated,
                               bootstrap_mask(env, terminated, truncated))
                    obs = next_obs
                    progress["steps"] += 1
                    progress["phase_steps"] += 1
                    if terminated or truncated:
                        progress["episodes"] += 1
                        progress["recent_destruction"] = (progress["recent_destruction"] + [info["destruction"]])[-100:]
                        append_json(output / "episodes.jsonl", {"steps": progress["steps"],
                                    "phase": progress["phase"], "terminated": terminated,
                                    "truncated": truncated, **info})
                        obs, _ = env.reset()
                    if (progress["phase_steps"] >= settings.learning_starts
                            and replay.size >= settings.batch_size
                            and progress["steps"] % settings.train_freq == 0):
                        progress["last_loss"] = update(online, target, optimizer, replay, rng, settings, device)
                        progress["updates"] += 1
                    if progress["steps"] % settings.target_interval == 0:
                        target.load_state_dict(online.state_dict())
                    boundary = progress["steps"] == settings.placement_steps
                    if boundary:
                        # Match PPO's scheduled curriculum reset. The last sampled
                        # transition still bootstraps its true placement continuation.
                        if not (terminated or truncated):
                            progress["phase_cutoffs"] += 1
                        env.set_fixed_army(False)
                        obs, _ = env.reset()
                        replay.pos = replay.size = 0
                        progress.update(phase="joint", phase_steps=0, recent_destruction=[])
                    progress["loop_seconds"] += perf_counter() - started
                    if progress["steps"] % settings.log_interval == 0 or boundary:
                        row = {k: v for k, v in progress.items() if k != "recent_destruction"}
                        row.update(epsilon=epsilon_at(progress["steps"], settings),
                                   recent_training_destruction=(float(np.mean(progress["recent_destruction"]))
                                                               if progress["recent_destruction"] else None))
                        append_json(output / "progress.jsonl", row)
                        print(f"{progress['phase']} {progress['steps']:,}/{settings.total_steps:,} "
                              f"epsilon={row['epsilon']:.3f} updates={progress['updates']:,}", flush=True)
                    if boundary or progress["steps"] % settings.checkpoint_interval == 0:
                        commit(output, contract_hash, online, target, optimizer, replay, env, obs, rng, progress)
                # Includes graceful pauses between fully completed transitions/updates.
                commit(output, contract_hash, online, target, optimizer, replay, env, obs, rng, progress)
            status = "complete" if progress["steps"] == settings.total_steps else "paused"
            atomic_json(output / "status.json", {"status": status, "updated_utc": now(), **progress})
            return progress
        except Exception as exc:
            # Do not publish potentially half-updated state. The previous generation
            # remains the recovery point; resume rolls logs back to that point.
            atomic_json(output / "last_error.json", {"utc": now(), "error": repr(exc),
                        "recovery": "resume from last committed checkpoint"})
            raise
        finally:
            env.close()

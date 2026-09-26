"""Two-stage Maskable PPO with local, post-update recovery checkpoints."""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from functools import partial
import json
import math
import os
from pathlib import Path
import signal
import threading
from time import perf_counter
import uuid

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env import config as C
from env.siege_env import SiegeEnv
from env.symmetry import SymmetryWrapper
from experiments.pools import load_pool
from experiments.provenance import runtime_manifest, sha256, source_hashes
from experiments.rollout import play_episode
from experiments.design import development_panel
from train.checkpoints import (RecoverablePPO, SnapshotMonitor, atomic_bytes, atomic_json,
    capture_rng, restore_rng, latest_checkpoint, restore_checkpoint, run_lock, save_checkpoint, utc_now)
from train.features import SiegeFeatures


@dataclass(frozen=True)
class TrainingSettings:
    condition: str = "P"
    seed: int = 0
    placement_steps: int = 64
    joint_steps: int = 192
    n_envs: int = 1
    rollout_steps: int = 32
    batch_size: int = 32
    epochs: int = 2
    mode: str = "smoke"
    device: str = "cpu"

    def validate(self):
        if self.condition not in ("P", "PA") or self.mode not in ("smoke", "pilot", "full"):
            raise ValueError("Unknown condition or training mode")
        if min(self.placement_steps, self.joint_steps, self.n_envs, self.rollout_steps, self.batch_size, self.epochs) < 1:
            raise ValueError("Training counts must be positive")
        if self.rollout_steps * self.n_envs < 2 or self.batch_size < 2:
            raise ValueError("PPO needs at least two samples per update")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("Choose cpu or cuda explicitly")


def phase_budgets(settings):
    quantum = settings.n_envs * settings.rollout_steps
    placement = math.ceil(settings.placement_steps / quantum) * quantum
    joint = math.ceil(settings.joint_steps / quantum) * quantum
    return {"rollout_transitions": quantum, "placement_actual": placement,
            "joint_actual": joint, "total_actual": placement + joint}


def study_settings(*, condition="P", seed=None, device="cpu", mode="full"):
    if mode not in ("pilot", "full"):
        raise ValueError("Choose pilot or full study settings")
    seed = (C.PILOT_LEARNER_SEED if mode == "pilot" else 0) if seed is None else seed
    return TrainingSettings(condition=condition, seed=seed, device=device, mode=mode,
        placement_steps=C.PILOT_PLACEMENT_TIMESTEPS if mode == "pilot" else C.CURRICULUM_PLACEMENT_TIMESTEPS,
        joint_steps=C.PILOT_JOINT_TIMESTEPS if mode == "pilot" else C.CURRICULUM_JOINT_TIMESTEPS,
        n_envs=C.N_TRAINING_ENVS, rollout_steps=C.PPO_N_STEPS,
        batch_size=C.PPO_BATCH_SIZE, epochs=C.PPO_N_EPOCHS)


def make_env(pool_path=None, *, seed=0, rank=0, symmetry=False, fixed_army=True, split="train"):
    pool = load_pool(pool_path, required_split=split) if pool_path else None
    env = SiegeEnv(pool=pool, fixed_army=fixed_army)
    if symmetry:
        env = SymmetryWrapper(env)
    env.reset(seed=seed + rank)
    return SnapshotMonitor(env)


def validate_inputs(settings, train_pool, dev_pool, registration):
    settings.validate()
    if settings.mode in ("pilot", "full") and (not train_pool or not dev_pool):
        raise ValueError("Full experiments require frozen training and development pools")
    if settings.mode == "pilot":
        if settings.seed in C.STUDY_LEARNER_SEEDS:
            raise ValueError("Keep pilot learner seeds separate from the main study")
        if settings != study_settings(condition=settings.condition, seed=settings.seed, device=settings.device, mode="pilot"):
            raise ValueError("Pilot settings differ from the declared pilot curriculum")
        if registration:
            raise ValueError("Pilots are development evidence, not registered full runs")
        for path in (train_pool, dev_pool):
            if json.loads(Path(path).read_text()).get("filter") != "structure-and-difficulty":
                raise ValueError("Pilots require difficulty-filtered pools")
    if settings.mode == "full":
        if not registration:
            raise ValueError("Full training requires an experiment registration created before training")
        reg = json.loads(Path(registration).read_text())
        if settings.seed not in reg["learner_seeds"] or reg["source_hashes"] != source_hashes():
            raise ValueError("Learner seed or source differs from registration")
        planned = {"placement_steps": C.CURRICULUM_PLACEMENT_TIMESTEPS,
                   "joint_steps": C.CURRICULUM_JOINT_TIMESTEPS, "n_envs": C.N_TRAINING_ENVS,
                   "rollout_steps": C.PPO_N_STEPS, "batch_size": C.PPO_BATCH_SIZE, "epochs": C.PPO_N_EPOCHS}
        if any(getattr(settings, key) != value for key, value in planned.items()):
            raise ValueError("Full training settings differ from the registered curriculum")
        for split, path in (("train", train_pool), ("dev", dev_pool)):
            if sha256(path) != reg[f"{split}_sha256"]:
                raise ValueError("Training pool differs from registration")
            if json.loads(Path(path).read_text()).get("filter") != "structure-and-difficulty":
                raise ValueError("Full runs require the planned difficulty-filtered pools")
    for split, path in (("train", train_pool), ("dev", dev_pool)):
        if path:
            load_pool(path, required_split=split)


def configure_device(settings):
    if settings.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this notebook kernel")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class PauseTraining(Exception):
    pass


@contextmanager
def stop_signals(stop):
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, lambda number, frame: stop.update(requested=True))
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def request_pause(output):
    output = Path(output)
    if not (output / "contract.json").is_file():
        raise ValueError("Not a recoverable training run")
    atomic_json(output / "STOP_REQUESTED.json", {"requested_utc": utc_now()})


class ProgressRecorder(BaseCallback):
    def __init__(self, path, stop, on_start):
        super().__init__()
        self.path, self.stop, self.on_start = Path(path), stop, on_start

    def _on_training_start(self):
        self.on_start()

    def _on_step(self):
        with self.path.open("a") as handle:
            for info, done in zip(self.locals["infos"], self.locals["dones"]):
                for key, value in info["reward_components"].items():
                    self.logger.record_mean(f"reward/{key}", value)
                if done:
                    handle.write(json.dumps({"timesteps": self.num_timesteps,
                        "destruction": info["destruction"], "return": info["episode_return"],
                        "spent": info["spent"], "finish_reason": info["finish_reason"]}) + "\n")
        # A partial rollout is discarded; the last committed update remains intact.
        return not (self.stop["requested"] or (self.path.parent / "STOP_REQUESTED.json").exists())


def development_check(model, pool, history, interval):
    if pool and model.num_timesteps >= history["next_step"]:
        rng = capture_rng()
        try:
            records = []
            panel = development_panel(pool)
            for base in panel:
                for view in C.DEV_DIAGNOSTIC_VIEWS:
                    env = SymmetryWrapper(SiegeEnv(motifs=C.DEV_MOTIFS), transform=view)
                    try:
                        records.append(play_episode(env, model, options={"spec": base})["summary"])
                    finally:
                        env.close()
            if not records:
                raise ValueError("Development pool has no permitted layouts")
            history["records"].append({"timesteps": model.num_timesteps,
                "mean_return": sum(r["episode_return"] for r in records) / len(records),
                "mean_destruction": sum(r["destruction"] for r in records) / len(records),
                "mean_spent": sum(r["spent"] for r in records) / len(records),
                "episodes": len(records), "views": list(C.DEV_DIAGNOSTIC_VIEWS),
                "layouts": [{"seed": b.seed, "motif": b.motif} for b in panel]})
            while history["next_step"] <= model.num_timesteps:
                history["next_step"] += interval
        finally:
            restore_rng(rng)


def _execute(settings, output, contract, manifest, *, recovery=None, pause_after_steps=None):
    configure_device(settings)
    budgets = phase_budgets(settings)
    train_pool = output / contract["inputs"]["train"]["path"] if contract["inputs"]["train"] else None
    dev_pool = load_pool(output / contract["inputs"]["dev"]["path"], required_split="dev") if contract["inputs"]["dev"] else None
    constructors = [partial(make_env, train_pool, seed=settings.seed, rank=rank,
                    symmetry=settings.condition == "PA") for rank in range(settings.n_envs)]
    vector = None
    started = perf_counter()
    stop = {"requested": False}
    phase = "placement"
    history = {"next_step": 0, "records": []}
    committed = None
    needs_initial = recovery is None
    attempt = len(manifest.get("attempts", [])) + 1
    manifest.setdefault("attempts", []).append({"number": attempt, "started_utc": utc_now()})
    manifest.update(status="running", last_error=None)
    atomic_json(output / "manifest.json", manifest)
    try:
        vector = SubprocVecEnv(constructors, start_method="spawn") if settings.n_envs > 1 else DummyVecEnv(constructors)
        if recovery:
            committed, metadata, rejected = recovery
            if metadata["contract_sha256"] != sha256(output / "contract.json"):
                raise ValueError("Checkpoint contract differs from this run")
            phase = metadata["phase"]
            model = RecoverablePPO.load(committed / "model.zip", env=vector, device=settings.device, force_reset=False)
            state = restore_checkpoint(committed, model, vector)
            history = state["development"]
            archive = output / "recovery" / f"attempt-{attempt:03d}"
            archive.mkdir(parents=True, exist_ok=False)
            if (output / "episodes.jsonl").exists():
                atomic_bytes(archive / "previous-episodes.jsonl", (output / "episodes.jsonl").read_bytes())
            atomic_bytes(output / "episodes.jsonl", (committed / "episodes.jsonl").read_bytes())
            atomic_json(archive / "resume.json", {"checkpoint": committed.name, "actual_steps": model.num_timesteps,
                "rejected": rejected, "resumed_utc": utc_now()})
            restore_rng(state["rng"])
            print(f"Resuming {phase} at {model.num_timesteps:,} / {budgets['total_actual']:,} transitions", flush=True)
        else:
            model = RecoverablePPO("MultiInputPolicy", vector, seed=settings.seed,
                learning_rate=C.PPO_LEARNING_RATE, n_steps=settings.rollout_steps,
                batch_size=settings.batch_size, n_epochs=settings.epochs, gamma=C.PPO_GAMMA,
                gae_lambda=C.PPO_GAE_LAMBDA, ent_coef=C.PPO_ENT_COEF, device=settings.device,
                verbose=0, policy_kwargs={"features_extractor_class": SiegeFeatures, "normalize_images": False,
                    "net_arch": {"pi": [C.POLICY_HEAD_DIM], "vf": [C.VALUE_HEAD_DIM]}})
        # Attempts have separate event files; canonical history rolls back with the model.
        model.tensorboard_log = str(output / "tensorboard" / f"attempt-{attempt:03d}")
        if (output / "STOP_REQUESTED.json").exists():
            destination = output / "recovery" / f"handled-stop-{uuid.uuid4().hex}.json"
            destination.parent.mkdir(exist_ok=True)
            os.replace(output / "STOP_REQUESTED.json", destination)

        def commit():
            nonlocal committed
            committed, metadata = save_checkpoint(model, vector, output, phase=phase,
                contract_hash=sha256(output / "contract.json"), development=history)
            manifest.update(actual_steps=model.num_timesteps, latest_checkpoint=committed.name,
                            current_phase=phase, optimizer_updates=model._n_updates)
            atomic_json(output / "development.json", history)
            atomic_json(output / "manifest.json", manifest)
            print(f"Saved {phase}: {model.num_timesteps:,} / {budgets['total_actual']:,} transitions", flush=True)

        def on_start():
            nonlocal needs_initial
            if needs_initial:
                development_check(model, dev_pool, history, contract["development_interval"])
                commit()
                needs_initial = False

        def after_update(updated):
            development_check(updated, dev_pool, history, contract["development_interval"])
            pause = stop["requested"] or (output / "STOP_REQUESTED.json").exists()
            pause = pause or (pause_after_steps is not None and updated.num_timesteps >= pause_after_steps)
            end = budgets["placement_actual"] if phase == "placement" else budgets["total_actual"]
            rollout = updated.num_timesteps // budgets["rollout_transitions"]
            if rollout % contract["checkpoint_every_rollouts"] == 0 or updated.num_timesteps >= end or pause:
                commit()
            if pause:
                raise PauseTraining()

        model.after_update = after_update
        with stop_signals(stop):
            for next_phase, requested_end in (("placement", settings.placement_steps),
                    ("joint", budgets["placement_actual"] + settings.joint_steps)):
                if phase == "joint" and next_phase == "placement":
                    continue
                if next_phase == "joint" and phase == "placement":
                    if committed:
                        atomic_bytes(output / "placement.zip", (committed / "model.zip").read_bytes())
                    phase = "joint"
                    vector.env_method("set_fixed_army", False)
                    model.set_env(vector, force_reset=True)
                    needs_initial = True
                remaining = requested_end - model.num_timesteps
                if remaining > 0:
                    model.learn(remaining, reset_num_timesteps=False,
                        callback=ProgressRecorder(output / "episodes.jsonl", stop, on_start), tb_log_name="learning")
                    if stop["requested"] or (output / "STOP_REQUESTED.json").exists():
                        raise PauseTraining()
                if model.num_timesteps < requested_end:
                    raise PauseTraining()
        if model.num_timesteps != budgets["total_actual"]:
            raise RuntimeError("Unexpected final training budget")
        atomic_bytes(output / "model.zip", (committed / "model.zip").read_bytes())
        check = SiegeEnv()
        try:
            obs, _ = check.reset(seed=123)
            loaded = MaskablePPO.load(output / "model.zip", device="cpu")
            action, _ = loaded.predict(obs, action_masks=check.action_masks(), deterministic=True)
            assert check.action_masks()[int(action)]
        finally:
            check.close()
        manifest.update(status="completed", actual_steps=model.num_timesteps,
            placement_actual=budgets["placement_actual"], joint_actual=budgets["joint_actual"],
            model_sha256=sha256(output / "model.zip"), save_reload_verified=True)
    except PauseTraining:
        path, metadata, _ = latest_checkpoint(output)
        manifest.update(status="paused", actual_steps=metadata["actual_steps"], latest_checkpoint=path.name,
            note="Resume the same run directory; any partial rollout will be replayed.")
    except BaseException as exc:
        manifest.update(status="failed", last_error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        elapsed = perf_counter() - started
        manifest["wall_seconds"] = manifest.get("wall_seconds", 0) + elapsed
        manifest["attempts"][-1].update(finished_utc=utc_now(), wall_seconds=elapsed, status=manifest["status"])
        atomic_json(output / "manifest.json", manifest)
        if vector is not None:
            vector.close()
    return manifest


def train(settings, output, *, train_pool=None, dev_pool=None, registration=None,
          checkpoint_every_rollouts=1, development_interval=C.DEV_EVAL_INTERVAL_TIMESTEPS, pause_after_steps=None):
    validate_inputs(settings, train_pool, dev_pool, registration)
    if checkpoint_every_rollouts < 1 or development_interval < 1:
        raise ValueError("Checkpoint and development intervals must be positive")
    configure_device(settings)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = {}
    for name, path in (("train", train_pool), ("dev", dev_pool), ("registration", registration)):
        inputs[name] = None
        if path:
            relative = f"inputs/{name}.json"
            atomic_bytes(output / relative, Path(path).read_bytes())
            inputs[name] = {"path": relative, "sha256": sha256(output / relative)}
    contract = {"schema_version": 2, "created_utc": utc_now(), "settings": asdict(settings),
        "inputs": inputs, "checkpoint_every_rollouts": checkpoint_every_rollouts,
        "development_interval": development_interval, "budgets": phase_budgets(settings), **runtime_manifest()}
    atomic_json(output / "contract.json", contract)
    manifest = {**runtime_manifest(), "settings": asdict(settings), "started_utc": utc_now(),
        "registration_sha256": inputs["registration"]["sha256"] if inputs["registration"] else None,
        "contract_sha256": sha256(output / "contract.json"), "pools": {k: inputs[k] for k in ("train", "dev")},
        "requested_steps": settings.placement_steps + settings.joint_steps, "actual_steps": 0,
        "evidence_scope": {"smoke": "software smoke test", "pilot": "development pilot; not final evidence",
                           "full": "registered training run"}[settings.mode]}
    with run_lock(output):
        return _execute(settings, output, contract, manifest, pause_after_steps=pause_after_steps)


def resume(output, *, pause_after_steps=None):
    output = Path(output).expanduser().resolve()
    if not (output / "contract.json").is_file():
        raise ValueError("No recovery contract. Older model-only runs support inference, not safe resume.")
    with run_lock(output):
        contract = json.loads((output / "contract.json").read_text())
        manifest = json.loads((output / "manifest.json").read_text())
        if sha256(output / "contract.json") != manifest["contract_sha256"]:
            raise ValueError("Run contract changed")
        current = json.loads(json.dumps(runtime_manifest()))
        for key in ("source_hashes", "config", "python", "packages"):
            if contract[key] != current[key]:
                raise ValueError(f"Cannot resume: {key} changed. Restore the original code/environment.")
        for item in contract["inputs"].values():
            if item and sha256(output / item["path"]) != item["sha256"]:
                raise ValueError("Frozen input changed")
        try:
            recovery = latest_checkpoint(output)
        except ValueError:
            # No learning can precede the initial committed snapshot. A killed
            # initial write can safely restart, but damaged committed history cannot.
            if manifest["actual_steps"] != 0 or any((output / "checkpoints").glob("checkpoint-*")):
                raise
            recovery = None
        if (manifest["status"] == "completed" and recovery is not None and recovery[1]["actual_steps"] == contract["budgets"]["total_actual"]
                and (output / "model.zip").exists() and sha256(output / "model.zip") == manifest["model_sha256"]):
            print("This run is already complete; no training repeated.", flush=True)
            return manifest
        return _execute(TrainingSettings(**contract["settings"]), output, contract, manifest,
                        recovery=recovery, pause_after_steps=pause_after_steps)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", choices=["P", "PA"])
    p.add_argument("--seed", type=int)
    p.add_argument("--device", choices=["cpu", "cuda"])
    p.add_argument("--out", required=True, help="Persistent Studio project run directory; never /tmp")
    p.add_argument("--train-pool")
    p.add_argument("--dev-pool")
    p.add_argument("--registration")
    p.add_argument("--full", action="store_true")
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--confirm-long-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--request-pause", action="store_true")
    p.add_argument("--checkpoint-every-rollouts", type=int, default=1)
    args = p.parse_args()
    if args.full and args.pilot:
        p.error("Choose --pilot or --full, not both")
    if args.resume or args.request_pause:
        if any(v is not None for v in (args.condition, args.seed, args.device, args.train_pool, args.dev_pool, args.registration)) or args.full or args.pilot:
            p.error("Resume/pause reads the existing run settings; do not supply new training settings")
        if args.request_pause:
            request_pause(args.out)
            print("Pause requested. Wait for status=paused before stopping the Studio space.")
            return
        result = resume(args.out)
    else:
        if (args.full or args.pilot) and not args.confirm_long_run:
            p.error("Pilot/full training requires --confirm-long-run")
        values = {"condition": args.condition or "P",
                  "seed": args.seed if args.seed is not None else C.PILOT_LEARNER_SEED if args.pilot else 0,
                  "device": args.device or "cpu"}
        if args.full or args.pilot:
            values = asdict(study_settings(**values, mode="pilot" if args.pilot else "full"))
        result = train(TrainingSettings(**values), args.out, train_pool=args.train_pool, dev_pool=args.dev_pool,
            registration=args.registration, checkpoint_every_rollouts=args.checkpoint_every_rollouts)
    print(json.dumps({k: result.get(k) for k in ("status", "evidence_scope", "requested_steps", "actual_steps", "wall_seconds", "save_reload_verified")}, indent=2))


if __name__ == "__main__":
    main()

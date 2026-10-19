"""Training execution for the masked TRPO arm.

Mirrors train/train_ppo.py: same curriculum, same pools, same checkpoint and
pause/resume contract. It imports those modules; it does not modify them, so
`python -m ui.v2_models check` keeps passing and the twenty approved V2
checkpoints keep verifying.

    python -m trpo.runner pilot --out runs/trpo-pilot-100 --device cuda
    python -m trpo.runner resume --out runs/trpo-pilot-100
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import uuid
from functools import partial
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env import config as C
from experiments.pools import load_pool
from experiments.provenance import runtime_manifest, sha256
from train.checkpoints import (
    latest_checkpoint,
    restore_checkpoint,
    restore_rng,
    save_checkpoint,
    utc_now,
)
from train.features import SiegeFeatures
from train.train_ppo import (
    PauseTraining,
    ProgressRecorder,
    make_env,
    stop_signals,
)
# V2's check, not the V1-era one in train_ppo: it records BOTH sampled and
# deterministic acting per interval, which is what the published PPO
# development curves contain. Using the other one produces a deterministic-only
# number that looks far worse than PPO's headline figure for no real reason.
from version2.evaluation import development_check
from trpo import config as T
from trpo.maskable_trpo import RecoverableTRPO
from trpo.retention import retain_and_export

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def build_vector(n_envs, *, seed, pool_path, fixed_army):
    constructors = [
        partial(make_env, pool_path, seed=seed, rank=i, symmetry=False, fixed_army=fixed_army)
        for i in range(n_envs)
    ]
    return SubprocVecEnv(constructors, start_method="spawn") if n_envs > 1 else DummyVecEnv(constructors)


def budgets(total_steps, n_envs, rollout_steps):
    """Round each phase to whole vector rollouts, as the PPO trainer does."""
    rollout = rollout_steps * n_envs
    total = (total_steps // rollout) * rollout
    if total < rollout:
        # Without this the budget floors to zero and the run "completes"
        # having trained nothing, which looks like success.
        raise ValueError(
            f"--steps {total_steps} is smaller than one vector rollout "
            f"({rollout_steps} rollout-steps x {n_envs} workers = {rollout} transitions). "
            f"Use --steps {rollout} or more, or lower --rollout-steps / --n-envs."
        )
    # Capped at the total so a short smoke run cannot overshoot its own budget
    # on the placement phase alone.
    placement = min((T.CURRICULUM_PLACEMENT_TIMESTEPS // rollout) * rollout, total)
    return {"rollout_transitions": rollout, "placement_actual": placement, "total_actual": total}


def write_contract(output, *, seed, device, total_steps, n_envs, rollout_steps, train_pool, dev_pool,
                   development_interval, sample_seeds,
                   checkpoint_every_rollouts=8, keep_checkpoints=3):
    from importlib.metadata import version

    contract = {
        "algorithm": "MaskableTRPO",
        "condition": "T",
        "created_utc": utc_now(),
        "checkpoint_every_rollouts": checkpoint_every_rollouts,
        "keep_checkpoints": keep_checkpoints,
        "milestones": [32768, 262144, 524288, 1048576, 2097152, 4194304],
        "development_interval": development_interval,
        "sample_seeds": list(sample_seeds),
        "hardware": {
            "cuda_available": torch.cuda.is_available(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "torch_cuda_build": torch.version.cuda,
        },
        "hyperparameters": T.model_kwargs(device=device, seed=seed, n_steps=rollout_steps),
        "inputs": {
            "train_pool": str(train_pool) if train_pool else None,
            "train_pool_sha256": sha256(Path(train_pool)) if train_pool else None,
            "dev_pool": str(dev_pool) if dev_pool else None,
            "dev_pool_sha256": sha256(Path(dev_pool)) if dev_pool else None,
        },
        "packages": {p: version(p) for p in ("torch", "numpy", "stable-baselines3", "sb3-contrib", "gymnasium")},
        "platform": platform.platform(),
        "python": platform.python_version(),
        "settings": {"seed": seed, "device": device, "n_envs": n_envs, "total_steps": total_steps},
        "source_hashes": {
            name: sha256(ROOT / "trpo" / name) for name in ("config.py", "maskable_trpo.py", "runner.py")
        },
    }
    atomic_json(output / "contract.json", contract)
    return contract


def execute(output, contract, *, resume=False, pause_after_steps=None):
    settings = contract["settings"]
    device, seed, n_envs = settings["device"], settings["seed"], settings["n_envs"]
    rollout_steps = contract["hyperparameters"]["n_steps"]
    plan = budgets(settings["total_steps"], n_envs, rollout_steps)
    train_pool = contract["inputs"]["train_pool"]
    dev_pool_path = contract["inputs"]["dev_pool"]
    # development_check expects a loaded pool, not a path.
    dev_pool = load_pool(dev_pool_path, required_split="dev") if dev_pool_path else None

    manifest_path = output / "manifest.json"
    manifest = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except ValueError:
            # A disk-full event truncates the manifest mid-write. The manifest
            # is bookkeeping, not recovery state -- checkpoints/ and LATEST.json
            # carry that -- so rebuild it rather than refusing to resume.
            print(f"manifest.json is unreadable ({manifest_path.stat().st_size} bytes); "
                  "rebuilding it from the committed checkpoints", flush=True)
            damaged = output / "recovery" / f"manifest-damaged-{uuid.uuid4().hex[:8]}.json"
            damaged.parent.mkdir(parents=True, exist_ok=True)
            os.replace(manifest_path, damaged)
    if manifest is None:
        # Same shape the PPO trainer writes: runtime_manifest supplies
        # source_hashes, config, python and packages, which ui/policy_playback
        # validate_compatibility and the V3 registry both read.
        manifest = {
            **runtime_manifest(),
            "settings": {
                "condition": "T",
                "seed": seed,
                "device": device,
                "n_envs": n_envs,
                "mode": "full",
                "algorithm": "MaskableTRPO",
            },
            "application_version": "V3",
            "started_utc": utc_now(),
        }
    attempt = len(manifest.get("attempts", [])) + 1
    manifest.setdefault("attempts", []).append({"number": attempt, "started_utc": utc_now()})
    manifest.update(status="running", last_error=None)
    atomic_json(manifest_path, manifest)

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing to train on CPU silently")

    phase = "placement"
    committed = None
    history = {"next_step": 0, "records": [], "trpo": []}
    needs_initial = True
    stop = {"requested": False}

    vector = build_vector(n_envs, seed=seed, pool_path=train_pool, fixed_army=True)
    try:
        recovery = latest_checkpoint(output) if resume else None
        if recovery:
            committed, metadata, rejected = recovery
            if metadata["contract_sha256"] != sha256(output / "contract.json"):
                raise ValueError("Checkpoint contract differs from this run")
            phase = metadata["phase"]
            model = RecoverableTRPO.load(
                committed / "model.zip", env=vector, device=device, force_reset=False
            )
            state = restore_checkpoint(committed, model, vector)
            history = state["development"]
            archive = output / "recovery" / f"attempt-{attempt:03d}"
            archive.mkdir(parents=True, exist_ok=True)
            if (output / "episodes.jsonl").exists():
                (archive / "previous-episodes.jsonl").write_bytes((output / "episodes.jsonl").read_bytes())
            (output / "episodes.jsonl").write_bytes((committed / "episodes.jsonl").read_bytes())
            atomic_json(
                archive / "resume.json",
                {
                    "checkpoint": committed.name,
                    "actual_steps": model.num_timesteps,
                    "rejected": rejected,
                    "resumed_utc": utc_now(),
                },
            )
            restore_rng(state["rng"])
            print(f"Resuming {phase} at {model.num_timesteps:,} / {plan['total_actual']:,}", flush=True)
        else:
            policy_kwargs = dict(T.POLICY_KWARGS)
            policy_kwargs["features_extractor_class"] = SiegeFeatures
            model = RecoverableTRPO(
                "MultiInputPolicy",
                vector,
                policy_kwargs=policy_kwargs,
                **T.model_kwargs(device=device, seed=seed, n_steps=rollout_steps),
            )

        model.tensorboard_log = str(output / "tensorboard" / f"attempt-{attempt:03d}")
        if (output / "STOP_REQUESTED.json").exists():
            destination = output / "recovery" / f"handled-stop-{uuid.uuid4().hex}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(output / "STOP_REQUESTED.json", destination)

        def commit():
            nonlocal committed
            committed, _ = save_checkpoint(
                model, vector, output, phase=phase,
                contract_hash=sha256(output / "contract.json"), development=history,
            )
            manifest.update(
                actual_steps=model.num_timesteps,
                latest_checkpoint=committed.name,
                current_phase=phase,
                optimizer_updates=model._n_updates,
            )
            atomic_json(output / "development.json", history)
            atomic_json(manifest_path, manifest)
            # Without this every generation is kept and the volume fills.
            retain_and_export(output, committed, contract)
            print(f"Saved {phase}: {model.num_timesteps:,} / {plan['total_actual']:,}", flush=True)

        def on_start():
            nonlocal needs_initial
            if needs_initial:
                development_check(model, dev_pool, history, contract["development_interval"],
                                  contract["sample_seeds"])
                commit()
                needs_initial = False

        def after_update(updated):
            development_check(updated, dev_pool, history, contract["development_interval"],
                              contract["sample_seeds"])
            # TRPO-specific diagnostics: without these there is no way to tell
            # from the logs whether the trust region is actually binding.
            recorded = updated.logger.name_to_value
            history["trpo"].append({
                "timesteps": updated.num_timesteps,
                "kl_divergence": recorded.get("train/kl_divergence_loss"),
                "line_search_success": recorded.get("train/is_line_search_success"),
                "policy_objective": recorded.get("train/policy_objective"),
                "value_loss": recorded.get("train/value_loss"),
            })
            pause = stop["requested"] or (output / "STOP_REQUESTED.json").exists()
            pause = pause or (pause_after_steps is not None and updated.num_timesteps >= pause_after_steps)
            end = plan["placement_actual"] if phase == "placement" else plan["total_actual"]
            rollout = updated.num_timesteps // plan["rollout_transitions"]
            if rollout % contract["checkpoint_every_rollouts"] == 0 or updated.num_timesteps >= end or pause:
                commit()
            if pause:
                raise PauseTraining()

        model.after_update = after_update
        with stop_signals(stop):
            for next_phase, requested_end in (
                ("placement", plan["placement_actual"]),
                ("joint", plan["total_actual"]),
            ):
                if phase == "joint" and next_phase == "placement":
                    continue
                if next_phase == "joint" and phase == "placement":
                    if committed:
                        (output / "placement.zip").write_bytes((committed / "model.zip").read_bytes())
                    phase = "joint"
                    vector.env_method("set_fixed_army", False)
                    model.set_env(vector, force_reset=True)
                    needs_initial = True
                remaining = requested_end - model.num_timesteps
                if remaining > 0:
                    model.learn(
                        remaining,
                        reset_num_timesteps=False,
                        callback=ProgressRecorder(output / "episodes.jsonl", stop, on_start),
                        tb_log_name="learning",
                    )
                if model.num_timesteps < requested_end:
                    raise PauseTraining()

        model.save(output / "model.zip")
        # Reload before publishing: a checkpoint that cannot be loaded back is
        # not a finished run, and the registry refuses anything not completed.
        RecoverableTRPO.load(output / "model.zip", device="cpu")
        manifest.update(status="completed", actual_steps=model.num_timesteps,
                        completed_utc=utc_now(), model_sha256=sha256(output / "model.zip"),
                        save_reload_verified=True)
        atomic_json(manifest_path, manifest)
        print(f"Completed: {model.num_timesteps:,} transitions", flush=True)

    except PauseTraining:
        manifest.update(status="paused", paused_utc=utc_now())
        atomic_json(manifest_path, manifest)
        print("Paused; resume with: python -m trpo.runner resume --out", output, flush=True)
    except BaseException as error:
        manifest.update(status="failed", last_error=f"{type(error).__name__}: {error}")
        atomic_json(manifest_path, manifest)
        raise
    finally:
        vector.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Masked TRPO training")
    sub = parser.add_subparsers(dest="command", required=True)

    pilot = sub.add_parser("pilot", help="fresh run; refuses to overwrite an existing directory")
    pilot.add_argument("--out", required=True)
    pilot.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    pilot.add_argument("--seed", type=int, default=T.PILOT_SEED)
    pilot.add_argument("--steps", type=int, default=T.PILOT_TOTAL_TIMESTEPS)
    pilot.add_argument("--n-envs", type=int, default=T.N_TRAINING_ENVS)
    pilot.add_argument("--rollout-steps", type=int, default=T.TRPO_N_STEPS)
    pilot.add_argument("--train-pool", default=None)
    pilot.add_argument("--dev-pool", default=None)
    pilot.add_argument("--development-interval", type=int, default=32768,
                       help="transitions between development checks; 262144 matches the V2 study exactly")
    pilot.add_argument("--sample-seeds", type=int, nargs="+", default=[31001, 31002, 31003],
                       help="acting seeds for the sampled development mode; V2 used 31001 31002 31003")
    pilot.add_argument("--checkpoint-every-rollouts", type=int, default=8,
                       help="rollouts between committed generations; V2 used 8")
    pilot.add_argument("--keep-checkpoints", type=int, default=3,
                       help="recovery generations retained on disk; minimum 3")
    pilot.add_argument("--pause-after-steps", type=int, default=None)

    resume = sub.add_parser("resume", help="continue an interrupted run from its last committed checkpoint")
    resume.add_argument("--out", required=True)
    resume.add_argument("--pause-after-steps", type=int, default=None)

    args = parser.parse_args(argv)
    output = Path(args.out)

    if args.command == "pilot":
        if output.exists() and any(output.iterdir()):
            raise SystemExit(f"{output} exists and is not empty; pass a new --out or use resume")
        # Validate the budget before creating anything, so a bad --steps does
        # not leave a half-made run directory behind.
        try:
            budgets(args.steps, args.n_envs, args.rollout_steps)
        except ValueError as error:
            raise SystemExit(str(error))
        output.mkdir(parents=True, exist_ok=True)
        contract = write_contract(
            output, seed=args.seed, device=args.device, total_steps=args.steps,
            n_envs=args.n_envs, rollout_steps=args.rollout_steps,
            train_pool=args.train_pool, dev_pool=args.dev_pool,
            development_interval=args.development_interval, sample_seeds=args.sample_seeds,
            checkpoint_every_rollouts=args.checkpoint_every_rollouts,
            keep_checkpoints=args.keep_checkpoints,
        )
        execute(output, contract, resume=False, pause_after_steps=args.pause_after_steps)
    else:
        contract = json.loads((output / "contract.json").read_text())
        execute(output, contract, resume=True, pause_after_steps=args.pause_after_steps)


if __name__ == "__main__":
    sys.exit(main())

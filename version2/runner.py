"""V2 engine adapted from frozen train/train_ppo.py. V1 files are unchanged."""
from dataclasses import dataclass, asdict
from functools import partial
from pathlib import Path
from time import perf_counter
import json
import os
import uuid
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from env import config as C
from env.siege_env import SiegeEnv
from experiments.pools import load_pool
from experiments.provenance import runtime_manifest, sha256
from train.features import SiegeFeatures
from train.train_ppo import (TrainingSettings, phase_budgets, make_env, configure_device,
    PauseTraining, stop_signals, ProgressRecorder, request_pause)
from train.checkpoints import (RecoverablePPO, atomic_bytes, atomic_json, capture_rng,
    restore_rng, latest_checkpoint, restore_checkpoint, run_lock, save_checkpoint, utc_now)
from version2.protocol import source_hashes_v2, read_plan, verify_baseline, ROOT
from version2.evaluation import development_check
from version2.retention import retain_and_export

@dataclass(frozen=True)
class V2Settings(TrainingSettings):
    mode: str = "v2-control"

    def validate(self):
        if self.mode != "v2-control":
            raise ValueError("This entry point only supports the new control experiment")
        TrainingSettings(**{**asdict(self), "mode": "smoke"}).validate()


def hardware():
    return {"torch_cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}

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
    manifest.update(status="running", last_error=None, pause_reason=None)
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
            retain_and_export(output, committed, contract)
            print(f"Saved {phase}: {model.num_timesteps:,} / {budgets['total_actual']:,} transitions", flush=True)

        def on_start():
            nonlocal needs_initial
            if needs_initial:
                development_check(model, dev_pool, history, contract["development_interval"], contract["sample_seeds"])
                commit()
                needs_initial = False

        def after_update(updated):
            development_check(updated, dev_pool, history, contract["development_interval"], contract["sample_seeds"])
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
            pause_reason="user_request" if stop["requested"] or (output / "STOP_REQUESTED.json").exists() else "screen_boundary",
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



def checked_runtime():
    baseline = verify_baseline()
    current = json.loads(json.dumps(runtime_manifest()))
    if current['python'] != baseline['python']:
        raise ValueError(f"Use baseline Python {baseline['python']}; found {current['python']}")
    for name in ('torch', 'numpy', 'gymnasium', 'stable-baselines3', 'sb3-contrib'):
        if current['packages'][name] != baseline['packages'][name]:
            raise ValueError(f"Restore baseline {name}=={baseline['packages'][name]}")
    if current['config'] != baseline['config']:
        raise ValueError('Original configuration changed')
    return {**current, 'version2_source_hashes': source_hashes_v2(), 'hardware': hardware()}


def registration_path():
    return ROOT/'version2-records'/'longer-control'/'registration.json'


def register(device):
    plan = read_plan()
    current = checked_runtime()
    configure_device(V2Settings(device=device))
    path = registration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {'experiment': plan['experiment'], 'plan': plan, 'device': device, **current}
    with run_lock(path.parent):
        if path.exists():
            previous = json.loads(path.read_text())
            for key, value in record.items():
                if previous[key] != value:
                    raise ValueError(f'Existing registration differs: {key}. Do not overwrite it.')
        else:
            atomic_json(path, {**record, 'created_utc': utc_now()})
    return path


def validate_registration():
    path = registration_path()
    if not path.is_file():
        raise ValueError('First run: python -m version2 register --device cuda')
    reg = json.loads(path.read_text())
    for key, value in checked_runtime().items():
        if reg[key] != value:
            raise ValueError(f'Registered {key} changed; restore the registered environment')
    if reg['plan'] != read_plan():
        raise ValueError('Registered plan changed')
    return path, reg


def run_path(condition, seed):
    return ROOT/'runs'/'version2-longer-control'/f'{condition}-{seed}'


def start_or_resume(condition, seed, target):
    from version2.protocol import check_stop
    registration, reg = validate_registration()
    plan = reg['plan']
    if condition not in plan['conditions'] or seed not in plan['learner_seeds']:
        raise ValueError('Condition/seed outside the registered development screen')
    check_stop(target, plan)
    output = run_path(condition, seed)
    if output.exists():
        return resume(output, pause_after_steps=target)
    settings = V2Settings(condition=condition, seed=seed, device=reg['device'],
        placement_steps=plan['placement_steps'], joint_steps=plan['total_steps']-plan['placement_steps'],
        n_envs=plan['n_envs'], rollout_steps=plan['rollout_steps'], batch_size=plan['batch_size'], epochs=plan['epochs'])
    settings.validate()
    configure_device(settings)
    output.mkdir(parents=True, exist_ok=False)
    with run_lock(output):
        baseline = verify_baseline()
        inputs = {}
        for name, path in [('train', ROOT/baseline['pools']['train']['path']),
                           ('dev', ROOT/baseline['pools']['dev']['path']), ('registration', registration)]:
            relative = f'inputs/{name}.json'
            atomic_bytes(output/relative, path.read_bytes())
            inputs[name] = {'path': relative, 'sha256': sha256(output/relative)}
        contract = {'schema_version': 2, 'experiment': plan['experiment'], 'created_utc': utc_now(),
            'settings': asdict(settings), 'inputs': inputs, 'budgets': phase_budgets(settings),
            **{key: plan[key] for key in ('checkpoint_every_rollouts', 'keep_checkpoints',
                'development_interval', 'sample_seeds', 'milestones')}, **checked_runtime()}
        atomic_json(output/'contract.json', contract)
        manifest = {**checked_runtime(), 'settings': asdict(settings), 'started_utc': utc_now(),
            'registration_sha256': sha256(registration), 'contract_sha256': sha256(output/'contract.json'),
            'pools': {k: inputs[k] for k in ('train', 'dev')}, 'requested_steps': plan['total_steps'],
            'actual_steps': 0, 'evidence_scope': plan['evidence_scope'],
            'timing_scope': 'Measured process wall time including evaluation, checkpoint I/O and recovery'}
        atomic_json(output/'manifest.json', manifest)
        return _execute(settings, output, contract, manifest,
            pause_after_steps=target if target != plan['total_steps'] else None)


def resume(output, *, pause_after_steps=None):
    output = Path(output).resolve()
    with run_lock(output):
        contract = json.loads((output/'contract.json').read_text())
        manifest = json.loads((output/'manifest.json').read_text())
        if sha256(output/'contract.json') != manifest['contract_sha256']:
            raise ValueError('Run contract changed')
        if contract.get('experiment') != 'version2-longer-control':
            raise ValueError('Not this Version 2 experiment')
        current = checked_runtime()
        for key in ('source_hashes', 'version2_source_hashes', 'config', 'python', 'packages', 'hardware'):
            if contract[key] != current[key]:
                raise ValueError(f'Cannot resume: {key} changed')
        for item in contract['inputs'].values():
            if item and sha256(output/item['path']) != item['sha256']:
                raise ValueError('Frozen input changed')
        try:
            recovery = latest_checkpoint(output)
        except ValueError:
            if manifest['actual_steps'] != 0 or any((output/'checkpoints').glob('checkpoint-*')):
                raise
            recovery = None
        total = contract['budgets']['total_actual']
        if manifest.get('status') == 'completed':
            if recovery is None or recovery[1]['actual_steps'] != total or sha256(output/'model.zip') != manifest['model_sha256']:
                raise ValueError('Completed run is incomplete or damaged')
            return manifest
        if recovery and pause_after_steps and recovery[1]['actual_steps'] >= pause_after_steps and pause_after_steps < total:
            print(f'Already at requested screen: {recovery[1]["actual_steps"]:,} transitions', flush=True)
            return manifest
        return _execute(V2Settings(**contract['settings']), output, contract, manifest,
                        recovery=recovery, pause_after_steps=pause_after_steps if pause_after_steps != total else None)

"""V3 final evaluation: MaskablePPO (P, PA) and MaskableTRPO (T) on one fresh pool.

Built alongside v2_final rather than inside it. The V2 final pool is spent and its
freeze marker stands; this generates a new pool on a disjoint seed interval and
evaluates all three conditions plus the heuristic baseline under one frozen protocol.

Nothing in env/, agents/, experiments/ or train/ is modified -- those are imported,
and all 28 files hashed by ui/v2_checkpoints.json stay byte-identical.

    python -m v3_final freeze
    python -m v3_final pool
    python -m v3_final run [--resume] [--max-new N]
    python -m v3_final export
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from agents.heuristic import HeuristicCommander
from env import config as C
from env.generator import BaseSpec, generate_base
from env.siege_env import SiegeEnv
from env.symmetry import SymmetryWrapper
from experiments.journal import open_journal
from experiments.pools import validate_base
from experiments.provenance import runtime_manifest, sha256
from experiments.rollout import play_episode
from experiments.storage import atomic_json, publish_bytes, publish_json, run_lock
from trpo.maskable_trpo import MaskableTRPO
from v2_phases.runner import PhasePolicy

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts/version3-final-evaluation"

METRICS = ("destruction", "full_clear", "town_hall", "episode_return", "spent", "tick", "decisions")
KEY_FIELDS = ("condition", "learner_seed", "transitions", "base_seed", "motif", "view", "mode", "sample_seed")

# Condition -> the class that can load that condition's checkpoints. P and PA are
# the frozen V2 PPO learners; T is the TRPO arm.
LOADERS = {"P": MaskablePPO, "PA": MaskablePPO, "T": MaskableTRPO}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def protocol():
    return read(PACKAGE / "protocol.json")


# --------------------------------------------------------------------------
# Models


def reference_models(root=ROOT):
    """Every approved checkpoint, from the V2 registry plus the V3 registry.

    The V2 entries are read from ui/v2_checkpoints.json so this package never
    carries a second copy of their hashes.
    """
    models = []
    v2 = read(Path(root) / "ui/v2_checkpoints.json")
    for entry in v2["models"]:
        models.append({
            "condition": entry["condition"],
            "learner_seed": entry["learner_seed"],
            "transitions": entry["transitions"],
            "path": entry["path"],
            "sha256": entry["sha256"],
            "run": entry["run"],
        })
    v3_registry = Path(root) / "ui/v3_checkpoints.json"
    if v3_registry.exists():
        for entry in read(v3_registry)["models"]:
            models.append({
                "condition": entry["condition"],
                "learner_seed": entry["learner_seed"],
                "transitions": entry["transitions"],
                "path": entry["path"],
                "sha256": entry["sha256"],
                "run": entry["run"],
            })
    return sorted(models, key=lambda m: (m["condition"], m["learner_seed"]))


def load_target(target, root=ROOT):
    if target["condition"] == "heuristic":
        return None
    path = Path(root) / target["path"]
    require(sha256(path) == target["sha256"], "Model changed after preflight: " + str(path))
    model = LOADERS[target["condition"]].load(path, device="cpu")
    require(
        model.num_timesteps == target["transitions"],
        "Model timestep counter differs from the frozen checkpoint",
    )
    model.policy.set_training_mode(False)
    return model


def targets(contract):
    return [
        *sorted(contract["models"], key=lambda m: (m["learner_seed"], m["condition"])),
        {"condition": "heuristic", "learner_seed": -1, "transitions": 0, "run": "heuristic"},
    ]


# --------------------------------------------------------------------------
# Freeze


def prepare(root=ROOT, output=OUTPUT):
    """Read-only provenance checks. Generates and evaluates nothing."""
    spec = protocol()
    models = reference_models(root)
    expected = {(c, s) for c in spec["conditions"] for s in spec["learner_seeds"]}
    present = {(m["condition"], m["learner_seed"]) for m in models}
    missing = sorted(expected - present)
    require(
        not missing,
        "Missing approved checkpoints for: "
        + ", ".join(f"{c}-{s}" for c, s in missing)
        + ". Train them and register them in ui/v3_checkpoints.json before freezing.",
    )
    for model in models:
        require(
            model["transitions"] == spec["transitions"],
            f"{model['run']} was trained for {model['transitions']} transitions, not {spec['transitions']}",
        )
        require((Path(root) / model["path"]).is_file(), "Missing checkpoint file: " + model["path"])

    runtime = json.loads(json.dumps(runtime_manifest()))
    torch.set_num_threads(spec["runtime"]["torch_threads"])
    torch.use_deterministic_algorithms(spec["runtime"]["deterministic_algorithms"])

    contract = {
        "schema": 1,
        "kind": spec["kind"],
        "protocol": spec,
        "models": models,
        "scientific_runtime": runtime,
        "protocol_sha256": sha256(PACKAGE / "protocol.json"),
        "device": spec["runtime"]["device"],
        "torch_threads": spec["runtime"]["torch_threads"],
        "output_relative": "artifacts/version3-final-evaluation",
    }
    attacks = len(spec["views"]) * len(spec["sample_seeds"])
    report = {
        "status": "ready",
        "models": len(models),
        "conditions": spec["conditions"],
        "candidate_layouts": spec["candidate_layouts"],
        "candidate_seed_interval": spec["candidate_seed_interval"],
        "mode": spec["mode"],
        "attacks_per_accepted_layout_per_learner": attacks,
        "accepted_layouts": "not generated yet",
        "device": spec["runtime"]["device"],
    }
    return contract, report


def freeze(output=OUTPUT):
    output = Path(output)
    contract, report = prepare(output=output)
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output, ".pipeline.lock"):
        target = output / "freeze.json"
        if target.exists():
            require(read(target)["contract"] == contract, "Frozen protocol differs. Do not overwrite the freeze.")
        else:
            require(
                not list(output.glob("*.sqlite3")) and not (output / "final_pool.json").exists(),
                "Results exist without a freeze; stop and review.",
            )
            publish_json(
                target,
                {
                    "frozen_utc": now(),
                    "contract": contract,
                    "basis": "V2 study complete and TRPO pilot gates passed; fresh V3 layouts not yet generated",
                },
            )
    print(json.dumps({**report, "status": "frozen", "freeze_sha256": sha256(target)}, indent=2))


# --------------------------------------------------------------------------
# Pool


def pool_candidates(spec):
    low, high = spec["candidate_seed_interval"]
    return [(seed, motif) for seed in range(low, high + 1) for motif in spec["motifs"]]


def build_pool(output=OUTPUT, resume=False):
    """Generate the fresh V3 pool. Resumable; accepted layouts are never replaced."""
    output = Path(output)
    frozen = read(output / "freeze.json")
    contract = frozen["contract"]
    spec = contract["protocol"]
    candidates = pool_candidates(spec)
    require(
        len(candidates) == spec["candidate_layouts"],
        f"Protocol says {spec['candidate_layouts']} candidates but the interval yields {len(candidates)}",
    )

    with run_lock(output, ".pipeline.lock"):
        with open_journal(output / "pool.progress.sqlite3", contract, resume=resume) as journal:
            for seed, motif in candidates:
                key = f"{seed}:{motif}"
                if journal.get(key) is not None:
                    continue
                base = generate_base(seed, motif)
                accepted, score, reason = validate_base(base, allow_final=True)
                journal.put(key, {
                    "seed": seed, "motif": motif, "accepted": bool(accepted),
                    "score": None if score is None else float(score), "reason": reason,
                })
            records = [journal.get(f"{seed}:{motif}") for seed, motif in candidates]

    accepted = [r for r in records if r["accepted"]]
    bases = [generate_base(r["seed"], r["motif"]) for r in accepted]
    for motif in spec["motifs"]:
        require(
            any(b.motif == motif for b in bases),
            f"No accepted layouts for {motif}. Stop; do not add replacement candidates.",
        )
    payload = {
        "kind": spec["kind"],
        "generated_utc": now(),
        "attempted": len(records),
        "accepted": len(bases),
        "validation": records,
        "bases": [json.loads(b.to_json()) if hasattr(b, "to_json") else b.__dict__ for b in bases],
    }
    publish_json(output / "final_pool.json", payload)
    print(json.dumps({
        "status": "pool ready", "attempted": len(records), "accepted": len(bases),
        "familiar": sum(1 for b in bases if b.motif != spec["transfer_motif"]),
        "transfer": sum(1 for b in bases if b.motif == spec["transfer_motif"]),
        "pool_sha256": sha256(output / "final_pool.json"),
    }, indent=2))


def load_pool(output=OUTPUT):
    data = read(Path(output) / "final_pool.json")
    bases = [generate_base(b["seed"], b["motif"]) for b in data["bases"]]
    return sorted(bases, key=lambda b: (b.seed, b.motif))


# --------------------------------------------------------------------------
# Evaluation


def acting_seed(label, base, view):
    value = f"v3-final-ds-v1:{label}:{base.motif}:{base.seed}:{view}"
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "big")


def episode(target, model, base, view, label):
    seed = acting_seed(label, base, view)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    policy = HeuristicCommander() if target["condition"] == "heuristic" else PhasePolicy(model.policy, "DS")
    env = SymmetryWrapper(SiegeEnv(motifs=C.MOTIFS, allow_final=True), transform=view)
    try:
        return play_episode(env, policy, options={"spec": base})
    finally:
        env.close()


def evaluate_one(target, model, base, view, label):
    info = episode(target, model, base, view, label)["summary"]
    return {
        "condition": target["condition"], "learner_seed": target["learner_seed"],
        "transitions": target["transitions"], "base_seed": base.seed, "motif": base.motif,
        "view": view, "sample_seed": label,
        "mode": "heuristic" if target["condition"] == "heuristic" else "DS",
        "acting_seed": acting_seed(label, base, view),
        **{k: info[k] for k in METRICS},
        "zero_destruction": float(info["destruction"] == 0),
        "finish_reason": info["finish_reason"],
    }


def unit_key(target, base, view, label):
    mode = "heuristic" if target["condition"] == "heuristic" else "DS"
    return ":".join(map(str, (
        target["condition"], target["learner_seed"], target["transitions"],
        base.seed, base.motif, view, mode, label,
    )))


def units(contract, pool):
    spec = contract["protocol"]
    for target in targets(contract):
        for base in pool:
            for view in spec["views"]:
                for label in ([0] if target["condition"] == "heuristic" else spec["sample_seeds"]):
                    yield target, base, view, label


def validate_row(row):
    d = row["destruction"]
    require(math.isfinite(d) and 0 <= d <= 1, "Invalid destruction")
    require(row["full_clear"] == (d == 1), "Inconsistent full_clear flag")
    require(row["zero_destruction"] == float(d == 0), "Inconsistent zero_destruction flag")
    require(0 <= row["spent"] <= 200 and 0 <= row["tick"] <= 300, "Invalid episode accounting")
    require(math.isfinite(row["episode_return"]), "Invalid return")
    identity = type("Identity", (), {"seed": row["base_seed"], "motif": row["motif"]})()
    require(
        row["acting_seed"] == acting_seed(row["sample_seed"], identity, row["view"]),
        "Wrong acting seed",
    )


def execute(output=OUTPUT, resume=False, max_new=None):
    output = Path(output)
    frozen = read(output / "freeze.json")
    contract = frozen["contract"]
    # The protocol the freeze captured must still be the protocol on disk;
    # otherwise results would be collected under settings nobody froze.
    require(
        contract["protocol_sha256"] == sha256(PACKAGE / "protocol.json"),
        "protocol.json changed after the freeze. Restore it or start a new evaluation directory.",
    )
    pool = load_pool(output)
    torch.set_num_threads(contract["torch_threads"])
    torch.use_deterministic_algorithms(contract["protocol"]["runtime"]["deterministic_algorithms"])

    started = perf_counter()
    done = 0
    with run_lock(output, ".pipeline.lock"):
        with open_journal(output / "evaluation.progress.sqlite3", contract, resume=resume) as journal:
            current_run, model = None, None
            for target, base, view, label in units(contract, pool):
                key = unit_key(target, base, view, label)
                if journal.get(key) is not None:
                    continue
                if target["run"] != current_run:
                    model = load_target(target)
                    current_run = target["run"]
                row = evaluate_one(target, model, base, view, label)
                validate_row(row)
                journal.put(key, row)
                done += 1
                if done % 500 == 0:
                    print(f"{done} new attacks in {perf_counter() - started:.0f}s", flush=True)
                if max_new is not None and done >= max_new:
                    stopped_early = True
                    break
            else:
                stopped_early = False
    elapsed = perf_counter() - started
    record_timing(output, done, elapsed)
    status = "partial" if stopped_early else "complete"
    print(json.dumps({"status": status, "new_attacks": done, "seconds": round(elapsed, 1)}, indent=2))


def record_timing(output, done, elapsed):
    """Timing accumulates across resumes, so it is not a write-once artifact:
    a long run legitimately happens in several sittings, and a partial run's
    work counts towards the measured total just as much as the last one's."""
    if not done:
        return
    path = Path(output) / "timing_summary.json"
    record = read(path) if path.exists() else {"segments": []}
    record["segments"].append({"measured_utc": now(), "new_attacks": done, "seconds": elapsed})
    record["total_seconds"] = sum(s["seconds"] for s in record["segments"])
    record["total_attacks"] = sum(s["new_attacks"] for s in record["segments"])
    record["note"] = "execution only; excludes pauses and preflight checks"
    atomic_json(path, record)


# --------------------------------------------------------------------------
# Export


def csv_bytes(rows):
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=sorted(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode()


def export(output=OUTPUT):
    from v3_final.analysis import summaries

    output = Path(output)
    frozen = read(output / "freeze.json")
    contract = frozen["contract"]
    pool = load_pool(output)

    with open_journal(output / "evaluation.progress.sqlite3", contract, resume=True) as journal:
        rows = [journal.get(unit_key(t, b, v, l)) for t, b, v, l in units(contract, pool)]
    missing = sum(1 for r in rows if r is None)
    require(missing == 0, f"{missing} attacks are still missing; finish the run before exporting")

    models, conditions, paired, comparisons = summaries(rows, contract["protocol"])
    publish_bytes(output / "episodes.csv", csv_bytes(rows))
    publish_bytes(output / "model_summary.csv", csv_bytes(models))
    publish_bytes(output / "condition_summary.csv", csv_bytes(conditions))
    publish_bytes(output / "comparisons_by_learner.csv", csv_bytes(paired))
    publish_bytes(output / "comparisons.csv", csv_bytes(comparisons))
    publish_json(output / "complete.json", {
        "completed_utc": now(),
        "attacks": len(rows),
        "accepted_layouts": len(pool),
        "conditions": contract["protocol"]["conditions"],
        "outputs": {
            name: sha256(output / name)
            for name in ("episodes.csv", "model_summary.csv", "condition_summary.csv",
                         "comparisons_by_learner.csv", "comparisons.csv")
        },
    })
    print(json.dumps({"status": "exported", "attacks": len(rows), "layouts": len(pool)}, indent=2))

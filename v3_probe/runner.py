"""Acting-mode probe: the frozen V3 models and pool, re-evaluated under SS.

The V3 final evaluation measured one acting mode, DS -- deterministic purchasing,
sampled deployment. Under it condition T scored 56.1% destruction against P's 92.9%,
with T's army eliminated in 72% of attacks while it spent 199.7 of 200 resources.
Development, which sampled both phases, had put T at 92.5% against P's 92.3%.

Those two facts are only compatible if the deficit lives in the purchasing phase.
This probe tests that directly:

    DS = (purchasing deterministic, deployment sampled)   <- the frozen protocol
    SS = (purchasing sampled,       deployment sampled)   <- this probe

Deployment is sampled in both, so purchasing is the single variable. If T recovers
under SS while P is unchanged, the V3 result is acting-mode sensitivity rather than
a difference in learned policy quality.

THIS IS NOT PART OF THE FROZEN PROTOCOL. It is a post-hoc analysis written after
seeing the V3 result, and the write-up must say so. It is kept in its own package
writing to its own directory so that artifacts/version3-final-evaluation/ is read
but never modified -- the freeze marker there stands.

    python -m v3_probe run [--resume] [--max-new N] [--conditions P T]
    python -m v3_probe export
"""
from __future__ import annotations

import csv
import io
import json
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from env import config as C
from env.siege_env import SiegeEnv
from env.symmetry import SymmetryWrapper
from experiments.journal import open_journal
from experiments.provenance import runtime_manifest, sha256
from experiments.rollout import play_episode
from experiments.storage import atomic_json, publish_bytes, run_lock
from v2_phases.runner import MODES, PhasePolicy
from v3_final import runner as final

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "artifacts/version3-final-evaluation"
OUTPUT = ROOT / "artifacts/version3-mode-probe"

PROBE_MODE = "SS"
BASELINE_MODE = "DS"
DEFAULT_CONDITIONS = ("P", "T")

METRICS = final.METRICS


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def require(ok, message):
    if not ok:
        raise ValueError(message)


# --------------------------------------------------------------------------
# Inputs: read from the completed final evaluation, never written back


def frozen_contract(final_dir=FINAL):
    """The same contract the final evaluation ran under, plus the checks that it
    really is the same one. A probe comparing against DS numbers is only honest
    if it uses the identical models and the identical pool."""
    freeze_path = Path(final_dir) / "freeze.json"
    require(freeze_path.is_file(), f"No freeze at {freeze_path}. Run the V3 final evaluation first.")
    contract = final.read(freeze_path)["contract"]
    require(
        contract["protocol_sha256"] == sha256(final.PACKAGE / "protocol.json"),
        "protocol.json changed since the freeze; the probe would not be comparable.",
    )
    complete = Path(final_dir) / "complete.json"
    require(
        complete.is_file(),
        "The V3 final evaluation has not completed. Finish it before probing, "
        "otherwise there are no DS numbers to compare against.",
    )
    return contract


def probe_targets(contract, conditions):
    """Learner checkpoints only. The heuristic has no acting mode to vary, so
    including it would just re-run identical episodes under a new label."""
    selected = [m for m in contract["models"] if m["condition"] in conditions]
    require(selected, f"No models for conditions {sorted(conditions)} in the freeze.")
    missing = set(conditions) - {m["condition"] for m in selected}
    require(not missing, f"Freeze has no models for: {sorted(missing)}")
    return sorted(selected, key=lambda m: (m["learner_seed"], m["condition"]))


# --------------------------------------------------------------------------
# Evaluation


def episode(target, model, base, view, label, mode):
    # Same acting seed as the DS run, deliberately: the RNG stream is then held
    # fixed across modes, so a difference between DS and SS is the mode and not
    # a different draw. final.acting_seed keeps the v3-final-ds-v1 namespace.
    seed = final.acting_seed(label, base, view)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = SymmetryWrapper(SiegeEnv(motifs=C.MOTIFS, allow_final=True), transform=view)
    try:
        return play_episode(env, PhasePolicy(model.policy, mode), options={"spec": base})
    finally:
        env.close()


def evaluate_one(target, model, base, view, label, mode):
    info = episode(target, model, base, view, label, mode)["summary"]
    return {
        "condition": target["condition"], "learner_seed": target["learner_seed"],
        "transitions": target["transitions"], "base_seed": base.seed, "motif": base.motif,
        "view": view, "sample_seed": label, "mode": mode,
        "purchasing": "deterministic" if MODES[mode][0] else "sampled",
        "deployment": "deterministic" if MODES[mode][1] else "sampled",
        "acting_seed": final.acting_seed(label, base, view),
        **{k: info[k] for k in METRICS},
        "zero_destruction": float(info["destruction"] == 0),
        "finish_reason": info["finish_reason"],
    }


def unit_key(target, base, view, label, mode):
    # mode is in the key, so probe rows can never collide with the DS rows even
    # if someone points this at the final directory by mistake.
    return ":".join(map(str, (
        target["condition"], target["learner_seed"], target["transitions"],
        base.seed, base.motif, view, mode, label,
    )))


def units(contract, pool, targets):
    spec = contract["protocol"]
    for target in targets:
        for base in pool:
            for view in spec["views"]:
                for label in spec["sample_seeds"]:
                    yield target, base, view, label


def execute(output=OUTPUT, final_dir=FINAL, conditions=DEFAULT_CONDITIONS,
            mode=PROBE_MODE, resume=False, max_new=None):
    require(mode in MODES, f"Unknown acting mode {mode}")
    output, final_dir = Path(output), Path(final_dir)
    contract = frozen_contract(final_dir)
    pool = final.load_pool(final_dir)
    targets = probe_targets(contract, set(conditions))

    output.mkdir(parents=True, exist_ok=True)
    probe_contract = {
        "probe": "acting-mode",
        "status": "exploratory; declared outside the frozen V3 protocol",
        "baseline_mode": BASELINE_MODE,
        "probe_mode": mode,
        "varies": "purchasing phase only (deployment is sampled in both modes)",
        "conditions": sorted(conditions),
        "source_evaluation": str(final_dir.relative_to(ROOT)),
        "freeze_sha256": sha256(final_dir / "freeze.json"),
        "pool_sha256": sha256(final_dir / "final_pool.json"),
        "models": targets,
        "protocol": contract["protocol"],
        "torch_threads": contract["torch_threads"],
        "runtime": runtime_manifest(),
        "created_utc": now(),
    }
    contract_path = output / "probe_contract.json"
    if contract_path.exists():
        existing = final.read(contract_path)
        for field in ("freeze_sha256", "pool_sha256", "probe_mode", "conditions"):
            require(
                existing[field] == probe_contract[field],
                f"This directory holds a probe with a different {field}. Use a new --out.",
            )
        # Adopt the STORED contract for the rest of the run. The journal checks
        # byte equality against the contract it was created with, and the one
        # built above carries a fresh created_utc and runtime manifest, so
        # passing it would fail every resume with "Progress contract changed".
        # The four fields checked above are the ones that must not drift; the
        # timestamp and the runtime record are allowed to differ between
        # sittings and are kept as first written.
        probe_contract = existing
    else:
        atomic_json(contract_path, probe_contract)

    torch.set_num_threads(contract["torch_threads"])
    torch.use_deterministic_algorithms(contract["protocol"]["runtime"]["deterministic_algorithms"])

    started = perf_counter()
    done = 0
    stopped_early = False
    with run_lock(output, ".pipeline.lock"):
        with open_journal(output / "probe.progress.sqlite3", probe_contract, resume=resume) as journal:
            current_run, model = None, None
            for target, base, view, label in units(contract, pool, targets):
                key = unit_key(target, base, view, label, mode)
                if journal.get(key) is not None:
                    continue
                if target["run"] != current_run:
                    model = final.load_target(target)
                    current_run = target["run"]
                row = evaluate_one(target, model, base, view, label, mode)
                final.validate_row(row)
                journal.put(key, row)
                done += 1
                if done % 500 == 0:
                    print(f"{done} new attacks in {perf_counter() - started:.0f}s", flush=True)
                if max_new is not None and done >= max_new:
                    stopped_early = True
                    break

    elapsed = perf_counter() - started
    if done:
        path = output / "timing_summary.json"
        record = final.read(path) if path.exists() else {"segments": []}
        record["segments"].append({"measured_utc": now(), "new_attacks": done, "seconds": elapsed})
        record["total_seconds"] = sum(s["seconds"] for s in record["segments"])
        record["total_attacks"] = sum(s["new_attacks"] for s in record["segments"])
        atomic_json(path, record)

    print(json.dumps({
        "status": "partial" if stopped_early else "complete",
        "mode": mode, "new_attacks": done, "seconds": round(elapsed, 1),
    }, indent=2))


# --------------------------------------------------------------------------
# Export: the probe rows, and the DS-vs-SS difference per learner


def ds_rows(final_dir=FINAL):
    """The DS numbers come from the final evaluation's published episodes.csv,
    not from its journal -- that file is the artifact of record and is hashed in
    complete.json, so reading it keeps the comparison anchored to what was
    actually reported."""
    path = Path(final_dir) / "episodes.csv"
    require(path.is_file(), "No episodes.csv in the final evaluation; run its export first.")
    published = final.read(Path(final_dir) / "complete.json")["outputs"]["episodes.csv"]
    require(sha256(path) == published, "episodes.csv does not match its recorded hash.")
    with path.open() as handle:
        return list(csv.DictReader(handle))


def per_learner(rows, key="destruction"):
    out = {}
    grouped = {}
    for r in rows:
        grouped.setdefault((r["condition"], int(r["learner_seed"])), []).append(float(r[key]))
    for ident, values in grouped.items():
        out[ident] = 100.0 * statistics.fmean(values)
    return out


def csv_bytes(rows):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=sorted(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode()


def export(output=OUTPUT, final_dir=FINAL):
    output, final_dir = Path(output), Path(final_dir)
    probe_contract = final.read(output / "probe_contract.json")
    mode = probe_contract["probe_mode"]

    # journal.rows() is a generator yielding (key, row) pairs, and it reads the
    # connection lazily. Materialise it inside the with-block or it raises
    # "Cannot operate on a closed database" after the context exits.
    with open_journal(output / "probe.progress.sqlite3", probe_contract, resume=True) as journal:
        probe = [row for _key, row in journal.rows()]
    require(probe, "The probe journal is empty; run the probe first.")

    publish_bytes(output / "episodes.csv", csv_bytes(probe))

    baseline = [r for r in ds_rows(final_dir) if r["condition"] in probe_contract["conditions"]]
    ss = per_learner(probe)
    ds = per_learner(baseline)

    paired = []
    for ident in sorted(set(ss) & set(ds)):
        condition, seed = ident
        paired.append({
            "condition": condition, "learner_seed": seed,
            f"{BASELINE_MODE}_destruction_percent": round(ds[ident], 4),
            f"{mode}_destruction_percent": round(ss[ident], 4),
            "mode_change_points": round(ss[ident] - ds[ident], 4),
        })
    publish_bytes(output / "mode_by_learner.csv", csv_bytes(paired))

    summary = []
    for condition in probe_contract["conditions"]:
        deltas = [p["mode_change_points"] for p in paired if p["condition"] == condition]
        if not deltas:
            continue
        summary.append({
            "condition": condition,
            "learners": len(deltas),
            f"mean_{BASELINE_MODE}_percent": round(
                statistics.fmean([p[f"{BASELINE_MODE}_destruction_percent"]
                                  for p in paired if p["condition"] == condition]), 4),
            f"mean_{mode}_percent": round(
                statistics.fmean([p[f"{mode}_destruction_percent"]
                                  for p in paired if p["condition"] == condition]), 4),
            "mean_change_points": round(statistics.fmean(deltas), 4),
            "learners_improved": sum(1 for d in deltas if d > 0),
        })
    publish_bytes(output / "mode_summary.csv", csv_bytes(summary))

    atomic_json(output / "complete.json", {
        "status": "exploratory probe; outside the frozen V3 protocol",
        "completed_utc": now(),
        "baseline_mode": BASELINE_MODE, "probe_mode": mode,
        "attacks": len(probe),
        "outputs": {
            name: sha256(output / name)
            for name in ("episodes.csv", "mode_by_learner.csv", "mode_summary.csv")
        },
    })
    print(json.dumps(summary, indent=2))

"""Keep a bounded number of recovery generations, and export milestone models.

Modelled on version2/retention.py, which is hard-scoped to the V2 experiment and
so cannot be reused directly. Without this the trainer keeps every committed
generation, and because each one carries a copy of the growing episode history,
disk use climbs quadratically: a 4,194,304-transition learner writes 512
generations and fills a 50 GB Studio volume several learners in.
"""
import shutil
from pathlib import Path

from experiments.provenance import sha256
from train.checkpoints import atomic_bytes, atomic_json, verify_checkpoint


def retain_and_export(output, committed, contract):
    output, committed = Path(output), Path(committed)
    keep_count = contract.get("keep_checkpoints", 3)
    if keep_count < 3:
        raise ValueError("Keep at least three generations so a corrupt latest can fall back twice")

    root = output / "checkpoints"
    if root.is_symlink() or committed.is_symlink() or committed.parent != root:
        raise ValueError("Unexpected checkpoint location")

    metadata = verify_checkpoint(committed)
    if metadata["contract_sha256"] != sha256(output / "contract.json"):
        raise ValueError("Wrong checkpoint contract")

    # Milestones are published outside checkpoints/ so pruning cannot reach them.
    if metadata["actual_steps"] in contract.get("milestones", []):
        destination = output / "milestones" / f"step-{metadata['actual_steps']:012d}.zip"
        atomic_bytes(destination, (committed / "model.zip").read_bytes())
        atomic_json(destination.with_suffix(".json"), {
            "actual_steps": metadata["actual_steps"],
            "phase": metadata["phase"],
            "sha256": sha256(destination),
            "contract_sha256": metadata["contract_sha256"],
            "purpose": "inference and development comparisons",
        })

    valid = []
    for path in root.glob("checkpoint-*"):
        if path.is_symlink() or not path.is_dir():
            continue
        try:
            item = verify_checkpoint(path)
            if item["contract_sha256"] == metadata["contract_sha256"]:
                valid.append((item["actual_steps"], (path / "checkpoint.json").stat().st_mtime_ns, path))
        except (ValueError, OSError, KeyError):
            continue

    valid.sort(reverse=True)
    keep = {item[2] for item in valid[:keep_count]}
    keep.add(committed)
    # Corrupt or unrecognised generations are left alone for inspection.
    if len(keep) >= keep_count:
        for _steps, _mtime, path in valid:
            if path not in keep:
                shutil.rmtree(path)
    return len(keep)

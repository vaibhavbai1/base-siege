"""Report and reclaim disk used by finished TRPO runs.

    python3 reclaim_space.py              # report only, deletes nothing
    python3 reclaim_space.py --apply      # delete what the report listed

Only touches runs whose manifest says "completed". For those, checkpoints/ and
inputs/ exist to support resuming and are not needed again: model.zip,
manifest.json, development.json, episodes.jsonl, contract.json, placement.zip,
milestones/ and tensorboard/ are all kept, and those are what registration, the
V3 evaluation and the written record read.

A run that is running, paused, failed or unparseable is left completely alone.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path("runs/study-v3")
# Safe to remove once a run is complete: recovery state and a copy of the pool.
RECLAIMABLE = ("checkpoints", "inputs")


def size(path):
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def status_of(run):
    try:
        return json.loads((run / "manifest.json").read_text()).get("status")
    except (OSError, ValueError):
        return None


def main():
    apply = "--apply" in sys.argv
    if not ROOT.is_dir():
        print(f"{ROOT} not found; run this from the repo root.")
        return 1

    total = shutil.disk_usage(Path.home())
    print(f"volume: {human(total.used)} used, {human(total.free)} free of {human(total.total)}\n")

    reclaimable = 0
    for run in sorted(ROOT.glob("T-*")):
        state = status_of(run)
        parts = {name: size(run / name) for name in RECLAIMABLE}
        run_total = size(run)
        if state == "completed":
            freeable = sum(parts.values())
            reclaimable += freeable
            detail = ", ".join(f"{n} {human(s)}" for n, s in parts.items() if s)
            print(f"  {run.name:<6} completed   {human(run_total):>8}   free {human(freeable):>8}"
                  f"   ({detail})" if detail else f"  {run.name:<6} completed   {human(run_total):>8}   nothing to free")
            if apply and freeable:
                for name in RECLAIMABLE:
                    target = run / name
                    if target.exists():
                        shutil.rmtree(target)
        else:
            label = state or "no readable manifest"
            print(f"  {run.name:<6} {label:<11} {human(run_total):>8}   SKIPPED (not completed)")

    print()
    if apply:
        after = shutil.disk_usage(Path.home())
        print(f"freed {human(reclaimable)}; now {human(after.free)} free")
    else:
        print(f"{human(reclaimable)} reclaimable. Re-run with --apply to delete it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

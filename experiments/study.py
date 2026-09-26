"""Plan or run a registered learner matrix, without implicit final-test access."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import threading
import uuid

from experiments.provenance import sha256, source_hashes
from experiments.storage import atomic_json, publish_json, run_lock
from train.train_ppo import study_settings, train, resume as resume_run, request_pause


def matrix(registration, device="cpu"):
    reg = json.loads(Path(registration).read_text())
    if reg["source_hashes"] != source_hashes():
        raise ValueError("Code changed after registration")
    result = []
    for index, seed in enumerate(reg["learner_seeds"]):
        # Alternate the order within pairs; both methods still use every seed.
        for condition in (("P", "PA") if index % 2 == 0 else ("PA", "P")):
            result.append(asdict(study_settings(condition=condition, seed=seed, device=device)))
    return result


def request_study_pause(output):
    output = Path(output)
    if not (output / "study.json").is_file():
        raise ValueError("Not a registered study directory")
    atomic_json(output / "STOP_REQUESTED.json", {"requested_utc": datetime.now(timezone.utc).isoformat()})


def run_study(registration, output, *, device="cpu", resume=False, confirm_long_run=False, min_free_gib=8):
    if not confirm_long_run:
        raise PermissionError("Study execution requires --confirm-long-run")
    if min_free_gib < 1:
        raise ValueError("Keep at least 1 GiB of free-space headroom")
    registration, output = Path(registration).resolve(), Path(output).resolve()
    settings = matrix(registration, device)
    reg = json.loads(registration.read_text())
    contract = {"schema_version": 1, "registration_sha256": sha256(registration),
                "matrix": settings, "source_hashes": source_hashes(), "min_free_gib": min_free_gib}
    output.mkdir(parents=True, exist_ok=resume)
    with run_lock(output, ".study.lock"):
        if resume and not (output / "study.json").exists():
            raise ValueError("Missing study contract; do not reuse an unrelated directory")
        publish_json(output / "study.json", contract)
        stop_path = output / "STOP_REQUESTED.json"
        if resume and stop_path.exists():
            (output / "recovery").mkdir(exist_ok=True)
            os.replace(stop_path, output / "recovery" / f"handled-stop-{uuid.uuid4().hex}.json")
        completed = []
        for entry in settings:
            if stop_path.exists():
                break
            if shutil.disk_usage(output).free < min_free_gib * 1024**3:
                atomic_json(output / "progress.json", {"status": "paused", "reason": "insufficient free storage",
                                                     "completed": completed, "total": len(settings)})
                raise RuntimeError("Insufficient free storage; existing progress is preserved")
            run_path = output / f"{entry['condition']}-{entry['seed']}"
            atomic_json(output / "progress.json", {"status": "running", "active_run": run_path.name,
                                                "completed": completed, "total": len(settings)})
            finished = threading.Event()
            def watch_pause():
                while not finished.wait(.25):
                    if stop_path.exists() and (run_path / "contract.json").exists():
                        request_pause(run_path)
                        return
            watcher = threading.Thread(target=watch_pause, daemon=True)
            watcher.start()
            try:
                if run_path.exists():
                    saved = json.loads((run_path / "contract.json").read_text())
                    if saved["settings"] != entry or saved["inputs"]["registration"]["sha256"] != sha256(registration):
                        raise ValueError("Existing learner does not belong to this registered matrix")
                    result = resume_run(run_path)
                else:
                    result = train(study_settings(condition=entry["condition"], seed=entry["seed"], device=device),
                        run_path, train_pool=reg["train_pool"], dev_pool=reg["dev_pool"], registration=registration)
            except BaseException as exc:
                atomic_json(output / "progress.json", {"status": "failed", "active_run": run_path.name,
                    "completed": completed, "total": len(settings), "error": f"{type(exc).__name__}: {exc}"})
                raise
            finally:
                finished.set()
                watcher.join()
            if result["status"] != "completed":
                break
            completed.append(run_path.name)
        state = {"status": "completed" if len(completed) == len(settings) else "paused",
                 "completed": completed, "total": len(settings)}
        atomic_json(output / "progress.json", state)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--registration", required=True)
    plan.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run = sub.add_parser("run")
    run.add_argument("--registration", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--confirm-long-run", action="store_true")
    run.add_argument("--min-free-gib", type=float, default=8)
    pause = sub.add_parser("pause")
    pause.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        rows = matrix(args.registration, args.device)
        print(json.dumps({"learners": len(rows), "total_transitions": sum(r["placement_steps"] + r["joint_steps"] for r in rows),
                          "matrix": rows}, indent=2))
    elif args.command == "pause":
        request_study_pause(args.out)
    else:
        print(json.dumps(run_study(args.registration, args.out, device=args.device, resume=args.resume,
            confirm_long_run=args.confirm_long_run, min_free_gib=args.min_free_gib), indent=2))


if __name__ == "__main__":
    main()

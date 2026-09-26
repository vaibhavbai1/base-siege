"""Summarise development pilots; a team decision is required before registration."""
import argparse
from collections import deque
import json
import os
from pathlib import Path

from env import config as C
from experiments.provenance import sha256, source_hashes
from experiments.storage import publish_json
from train.evaluate import inspect_run, validate_matched_runs


def review_pilots(runs, evaluation, *, parent, note=""):
    parent, evaluation = Path(parent).resolve(), Path(evaluation).resolve()
    manifests = [inspect_run(run) for run in runs]
    validate_matched_runs(manifests)
    receipt = json.loads((evaluation / "evaluation.json").read_text())
    if receipt["mode"] != "development" or receipt["views"] != list(range(8)):
        raise ValueError("Review the completed eight-view development evaluation")
    if receipt["status"] != "completed" or receipt["episodes_sha256"] != sha256(evaluation / "episodes.csv"):
        raise ValueError("Development evaluation is incomplete or changed")
    if receipt["source_hashes"] != source_hashes():
        raise ValueError("Pilot evaluation used different source")
    expected_models = {(m["settings"]["condition"], m["settings"]["seed"], m["model_sha256"],
                        sha256(Path(path) / "manifest.json")) for path, m in zip(runs, manifests)}
    actual_models = {(p["condition"], p["learner_seed"], p["model_sha256"], p["manifest_sha256"])
                     for p in receipt["policies"] if p["condition"] in ("P", "PA")}
    if actual_models != expected_models:
        raise ValueError("Development evaluation does not match these pilots")
    summaries, evidence = [], []
    for path, manifest in zip(runs, manifests):
        path = Path(path).resolve()
        if manifest["settings"]["mode"] != "pilot" or manifest["settings"]["seed"] in C.STUDY_LEARNER_SEEDS:
            raise ValueError("Review separate development pilots, not smoke or main-study learners")
        if manifest["source_hashes"] != source_hashes():
            raise ValueError("Pilot source changed; rerun the pilots after development changes")
        if receipt.get("identity", {}).get("pool_sha256") != manifest["pools"]["dev"]["sha256"]:
            raise ValueError("Pilot evaluation used a different development pool")
        history = json.loads((path / "development.json").read_text())["records"]
        with (path / "episodes.jsonl").open() as handle:
            tail = [json.loads(line) for line in deque(handle, maxlen=100)]
        seconds = manifest["wall_seconds"]
        if seconds <= 0 or manifest["actual_steps"] <= 0:
            raise ValueError("Missing pilot runtime measurement")
        rate = manifest["actual_steps"] / seconds
        summaries.append({"condition": manifest["settings"]["condition"], "seed": manifest["settings"]["seed"],
            "transitions": manifest["actual_steps"], "wall_seconds": seconds, "steps_per_second": rate,
            "initial_development": history[0] if history else None, "last_development": history[-1] if history else None,
            "last_100_episodes": len(tail), "empty_army_fraction": sum(r["spent"] == 0 for r in tail) / len(tail) if tail else None,
            "projected_one_full_run_hours_at_this_rate": C.TOTAL_TRAINING_TIMESTEPS / rate / 3600})
        evidence.append({"path": os.path.relpath(path, parent), "manifest_sha256": sha256(path / "manifest.json"),
                         "model_sha256": manifest["model_sha256"], "development_sha256": sha256(path / "development.json"),
                         "episodes_sha256": sha256(path / "episodes.jsonl")})
    return {"schema_version": 1, "decision": "proceed" if note.strip() else "pending_team_review",
            "review_note": note.strip(), "source_hashes": source_hashes(), "runs": evidence,
            "evaluation": os.path.relpath(evaluation, parent), "evaluation_sha256": sha256(evaluation / "evaluation.json"),
            "train_pool_sha256": manifests[0]["pools"]["train"]["sha256"],
            "dev_pool_sha256": manifests[0]["pools"]["dev"]["sha256"], "summaries": summaries,
            "planned_full_learners": len(C.STUDY_LEARNER_SEEDS) * 2,
            "projected_training_hours": sum(s["projected_one_full_run_hours_at_this_rate"] for s in summaries) / len(summaries) * len(C.STUDY_LEARNER_SEEDS) * 2,
            "limits": "Pilot-based extrapolation, not a guarantee. Excludes full pool generation and separate evaluations. The team must inspect curves, baselines, failures and storage; this tool does not decide whether learning is good."}


def validate_review(path, train_pool, dev_pool):
    path = Path(path).resolve()
    saved = json.loads(path.read_text())
    if saved.get("decision") != "proceed" or not saved.get("review_note", "").strip():
        raise ValueError("A recorded team pilot review is required before full registration")
    fresh = review_pilots([path.parent / entry["path"] for entry in saved["runs"]],
                         path.parent / saved["evaluation"], parent=path.parent, note=saved["review_note"])
    if fresh != saved:
        raise ValueError("Pilot review evidence changed")
    if saved["train_pool_sha256"] != sha256(train_pool) or saved["dev_pool_sha256"] != sha256(dev_pool):
        raise ValueError("Registration pools differ from the reviewed pilots")
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--approve-study", action="store_true")
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    if args.approve_study and (not args.note.strip() or args.out is None):
        parser.error("--approve-study needs --out and an honest --note explaining the team's decision")
    if args.note and not args.approve_study:
        parser.error("Use --approve-study to record a decision; otherwise this is a read-only report")
    parent = args.out.resolve().parent if args.out else Path.cwd()
    result = review_pilots(args.runs, args.evaluation, parent=parent, note=args.note if args.approve_study else "")
    if args.out:
        publish_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

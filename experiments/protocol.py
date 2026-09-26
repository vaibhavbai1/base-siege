"""Explicit registration, freeze and one-time final evaluation entrypoints."""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from env import config as C
from experiments.pools import load_pool, build_pool
from env.generator import BaseSpec
from experiments.provenance import source_hashes, sha256, write_json
from experiments.storage import publish_json, run_lock
from train.evaluate import inspect_run, validate_matched_runs, evaluate_study


def register(path, train_pool, dev_pool, learner_seeds=C.STUDY_LEARNER_SEEDS, *, pilot_review=None):
    if len(set(learner_seeds)) != len(learner_seeds) or len(learner_seeds) not in (5, 10):
        raise ValueError("Register exactly five or ten distinct learner seeds")
    if not pilot_review:
        raise ValueError("A recorded pilot review is required before full registration")
    from experiments.pilot import validate_review
    review = validate_review(pilot_review, train_pool, dev_pool)
    if any(not isinstance(seed, int) or seed < 0 for seed in learner_seeds):
        raise ValueError("Learner seeds must be nonnegative integers")
    if set(learner_seeds) & {row["seed"] for row in review["summaries"]}:
        raise ValueError("Main learner seeds must not overlap the reviewed pilots")
    for split, pool, seed_range, motifs in (("train", train_pool, C.TRAIN_SEED_RANGE, C.TRAIN_MOTIFS),
                                           ("dev", dev_pool, C.DEV_SEED_RANGE, C.DEV_MOTIFS)):
        bases = load_pool(pool, required_split=split)
        data = json.loads(Path(pool).read_text())
        if data["filter"] != "structure-and-difficulty":
            raise ValueError("Registered pools must use the planned difficulty filter")
        if data["seed_interval"] != list(seed_range) or data["motifs"] != list(motifs):
            raise ValueError("Registration requires the complete declared candidate windows, not smoke pools")
        expected = {(seed, motif) for seed in range(seed_range[0], seed_range[1] + 1) for motif in motifs}
        records = data["validation"]
        if len(records) != len(expected) or {(r["seed"], r["motif"]) for r in records} != expected:
            raise ValueError("Pool candidate accounting is incomplete")
        if any(sum(b.motif == motif for b in bases) < C.DEV_BASES_PER_MOTIF for motif in motifs):
            raise ValueError("Too few accepted layouts in a structure; review the design before registering")
    write_json(path, {"schema_version": 2, "registered_utc": datetime.now(timezone.utc).isoformat(),
        "learner_seeds": sorted(learner_seeds), "source_hashes": source_hashes(),
        "train_pool": str(Path(train_pool).resolve()), "dev_pool": str(Path(dev_pool).resolve()),
        "train_sha256": sha256(train_pool), "dev_sha256": sha256(dev_pool),
        "final_candidate_seeds": list(C.FINAL_STUDY_SEED_RANGE), "views": list(range(8)),
        "filter": "structure-and-difficulty", "requested_budget": C.TOTAL_TRAINING_TIMESTEPS,
        "pilot_review_sha256": sha256(pilot_review), "pilot_review": review,
        "actual_budget_rule": "32 complete vector rollouts: 4 placement, 28 joint",
        "primary_metric": "destruction", "primary_comparison": "PA minus P, averaged within learner then across learners",
        "transfer_claim": "Held out from training; excluded from current development. Earlier software checks included this structure.",
        "selection_rule": "Use every registered learner and its fixed-budget endpoint; never select using final scores."})


def freeze(registration, runs, output):
    reg = json.loads(Path(registration).read_text())
    if reg["source_hashes"] != source_hashes():
        raise ValueError("Source changed after registration; explicitly re-register before training")
    manifests = [inspect_run(run) for run in runs]
    seeds = validate_matched_runs(manifests)
    if seeds != reg["learner_seeds"]:
        raise ValueError("Learner seeds differ from registration")
    for run, manifest in zip(runs, manifests):
        if manifest.get("registration_sha256") != sha256(registration):
            raise ValueError("Run did not use this registration")
        if datetime.fromisoformat(manifest["started_utc"]) < datetime.fromisoformat(reg["registered_utc"]):
            raise ValueError("Registration must precede training")
        if manifest["settings"]["mode"] != "full" or manifest["source_hashes"] != reg["source_hashes"]:
            raise ValueError("Only completed full runs from the registered code can be frozen")
        for split in ("train", "dev"):
            if manifest["pools"][split]["sha256"] != reg[f"{split}_sha256"]:
                raise ValueError("Run pool differs from registered pool")
    write_json(output, {"schema_version": 2, "registration": reg, "paths_relative_to": "freeze_parent",
        "registration_sha256": sha256(registration), "source_hashes": source_hashes(),
        "runs": [{"path": os.path.relpath(Path(run).resolve(), Path(output).resolve().parent),
                  "manifest_sha256": sha256(Path(run)/"manifest.json"),
                  "model_sha256": sha256(Path(run)/"model.zip")} for run in runs]})


def final_evaluation(freeze_path, output, *, confirm_final=False, resume=False):
    if not confirm_final:
        raise PermissionError("Final evaluation requires explicit confirmation")
    freeze_path, output = Path(freeze_path).resolve(), Path(output).resolve()
    frozen = json.loads(freeze_path.read_text())
    if frozen.get("schema_version") != 2:
        raise ValueError("Create a version-2 freeze before using resumable final evaluation")
    if frozen["source_hashes"] != source_hashes():
        raise ValueError("Frozen source hashes no longer match")
    runs = [freeze_path.parent / run["path"] for run in frozen["runs"]]
    for run, path in zip(frozen["runs"], runs):
        if run["model_sha256"] != sha256(path/"model.zip") or run["manifest_sha256"] != sha256(path/"manifest.json"):
            raise ValueError("A frozen learner artifact changed")
    marker = freeze_path.with_suffix(".final-started.json")
    binding = {"schema_version": 2, "freeze_sha256": sha256(freeze_path),
               "output": os.path.relpath(output, freeze_path.parent)}
    with run_lock(freeze_path.parent, f".{freeze_path.name}.final.lock"):
        if marker.exists():
            if not resume:
                raise FileExistsError("Final evaluation already started; use --resume for the same frozen experiment")
            if json.loads(marker.read_text()) != binding:
                raise ValueError("Final resume must use the same freeze and output directory")
        else:
            if resume:
                raise FileNotFoundError("No authorised final evaluation to resume")
            if output.exists():
                raise FileExistsError("Choose a new final output directory")
            publish_json(marker, binding)
        output.mkdir(parents=True, exist_ok=True)
        pool_path = output / "final_pool.json"
        data = build_pool(*C.FINAL_STUDY_SEED_RANGE, C.MOTIFS, pool_path, allow_final=True,
                         resume=pool_path.with_suffix(".progress.sqlite3").exists())
        bases = [BaseSpec.from_json(json.dumps(base)) for base in data["bases"]]
        if not bases:
            raise ValueError("No final candidates passed the registered filter; do not consume reserve seeds")
        frame = evaluate_study(bases, runs, output, views=range(8), allow_final=True, mode="final",
            resume=(output / "evaluation.progress.sqlite3").exists(), identity=binding)
        publish_json(output/"complete.json", {"freeze_sha256": sha256(freeze_path),
            "episodes_sha256": sha256(output/"episodes.csv"), "accepted_layouts": len(bases),
            "candidate_layouts": data["attempted"], "reserve_seeds_used": False})
        return frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    reg = commands.add_parser("register")
    reg.add_argument("--train-pool", required=True); reg.add_argument("--dev-pool", required=True)
    reg.add_argument("--pilot-review", required=True)
    reg.add_argument("--seeds", nargs="+", type=int, default=list(C.STUDY_LEARNER_SEEDS)); reg.add_argument("--out", required=True)
    fr = commands.add_parser("freeze")
    fr.add_argument("--registration", required=True); fr.add_argument("--runs", nargs="+", required=True); fr.add_argument("--out", required=True)
    final = commands.add_parser("final")
    final.add_argument("--freeze", required=True); final.add_argument("--out", required=True)
    final.add_argument("--confirm-final", action="store_true")
    final.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.command == "register": register(args.out, args.train_pool, args.dev_pool, args.seeds, pilot_review=args.pilot_review)
    elif args.command == "freeze": freeze(args.registration, args.runs, args.out)
    else: final_evaluation(args.freeze, args.out, confirm_final=args.confirm_final, resume=args.resume)


if __name__ == "__main__":
    main()

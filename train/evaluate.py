"""Matched-layout evaluation and learner-level uncertainty estimates."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t

from env import config as C
from env.siege_env import SiegeEnv
from env.symmetry import SymmetryWrapper
from agents.heuristic import HeuristicCommander
from agents.random_agent import RandomAgent
from experiments.pools import load_pool
from experiments.provenance import sha256, runtime_manifest
from experiments.rollout import play_episode
from experiments.journal import open_journal
from experiments.storage import json_bytes, publish_json, publish_bytes


def episode_record(policy, base, view, *, condition, learner_seed=-1, allow_final=False, mode="development"):
    env = SymmetryWrapper(SiegeEnv(motifs=C.MOTIFS, allow_final=allow_final), transform=view)
    try:
        result = play_episode(env, policy, options={"spec": base})["summary"]
        return {"condition": condition, "learner_seed": learner_seed,
            "base_seed": base.seed, "motif": base.motif, "view": view,
            "destruction": result["destruction"], "town_hall": int(result["town_hall"]),
            "full_clear": int(result["full_clear"]), "return": result["episode_return"],
            "spent": result["spent"], "ticks": result["tick"], "finish_reason": result["finish_reason"],
            "evidence_scope": mode}
    finally:
        env.close()


def evaluate(policy, bases, *, condition, learner_seed=-1, views=(0,), allow_final=False, mode="development"):
    return pd.DataFrame([episode_record(policy, base, view, condition=condition, learner_seed=learner_seed,
                         allow_final=allow_final, mode=mode) for base in bases for view in views])


def evaluate_random(bases, *, views=(0,), action_seeds=C.RANDOM_AGENT_ACTION_SEEDS, allow_final=False):
    rows = []
    for action_seed in action_seeds:
        frame = evaluate(RandomAgent(action_seed), bases, condition="random", views=views, allow_final=allow_final)
        frame["action_seed"] = action_seed
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def inspect_run(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("status") != "completed" or manifest["model_sha256"] != sha256(path / "model.zip"):
        raise ValueError("Incomplete run or checkpoint checksum mismatch")
    return manifest


def load_run(path):
    from sb3_contrib import MaskablePPO
    manifest = inspect_run(path)
    return MaskablePPO.load(Path(path) / "model.zip", device="cpu"), manifest


def validate_matched_runs(manifests):
    if not manifests:
        raise ValueError("No learner runs supplied")
    groups = {condition: {} for condition in ("P", "PA")}
    for manifest in manifests:
        settings = manifest["settings"]
        condition, seed = settings["condition"], settings["seed"]
        if seed in groups[condition]:
            raise ValueError("Duplicate learner seed")
        groups[condition][seed] = manifest
    if not groups["P"] or groups["P"].keys() != groups["PA"].keys():
        raise ValueError("P and PA must have exactly the same learner seeds")
    for seed in groups["P"]:
        a, b = groups["P"][seed], groups["PA"][seed]
        for key in ("actual_steps", "placement_actual", "joint_actual", "config", "pools", "source_hashes"):
            left, right = a[key], b[key]
            if key == "pools" and isinstance(left, dict) and isinstance(right, dict):
                left = {k: v["sha256"] if v else None for k, v in left.items()}
                right = {k: v["sha256"] if v else None for k, v in right.items()}
            if left != right:
                raise ValueError(f"Unmatched {key} for learner seed {seed}")
        sa = {k: v for k, v in a["settings"].items() if k != "condition"}
        sb = {k: v for k, v in b["settings"].items() if k != "condition"}
        if sa != sb:
            raise ValueError("Learning settings differ beyond augmentation")
    return sorted(groups["P"])


def paired_summary(frame, metric="destruction"):
    keys = ["learner_seed", "base_seed", "motif", "view"]
    p = frame[frame.condition == "P"]
    pa = frame[frame.condition == "PA"]
    if p.duplicated(keys).any() or pa.duplicated(keys).any():
        raise ValueError("Duplicate evaluation units")
    paired = p.merge(pa, on=keys, suffixes=("_p", "_pa"), how="outer", indicator=True)
    if paired.empty or (paired["_merge"] != "both").any():
        raise ValueError("The two conditions do not have identical evaluation units")
    paired["difference"] = paired[f"{metric}_pa"] - paired[f"{metric}_p"]
    differences = paired.groupby("learner_seed").difference.mean()
    n = len(differences)
    mean = float(differences.mean())
    half = float(t.ppf(.975, n-1) * differences.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    return {"metric": metric, "n_independent_learners": n, "mean_paired_difference": mean,
            "ci95_low": mean-half if half is not None else None,
            "ci95_high": mean+half if half is not None else None,
            "learner_differences": {str(k): float(v) for k, v in differences.items()},
            "note": "Confidence interval uses learner seeds, not individual games."}


def study_summary(frame):
    """Report transfer and orientation separately from the pooled headline."""
    subsets = {"overall": frame, "seen_structures": frame[frame.motif.isin(C.TRAIN_MOTIFS)],
               "transfer_structure": frame[~frame.motif.isin(C.TRAIN_MOTIFS)]}
    subsets.update({f"motif/{motif}": rows for motif, rows in frame.groupby("motif")})
    comparisons = {}
    for label, rows in subsets.items():
        if {"P", "PA"}.issubset(set(rows.condition)):
            comparisons[label] = {metric: paired_summary(rows, metric)
                                 for metric in ("destruction", "town_hall", "full_clear", "return")}
    means = frame.groupby(["condition", "learner_seed", "motif", "view"])[["destruction", "town_hall", "full_clear", "return"]].mean().reset_index()
    gap = means.groupby(["condition", "learner_seed", "motif"]).destruction.agg(lambda x: x.max()-x.min()).reset_index(name="orientation_gap")
    return {"comparisons": comparisons, "means": means.to_dict("records"),
            "orientation_gap": gap.to_dict("records"),
            "orientation_note": "Max minus min mean destruction across tested views; informative only when all eight views were evaluated.",
            "baseline_note": "Random uses ten action seeds per base/view. Baseline game counts are not independent PPO learners."}


def evaluate_study(bases, runs, output, *, views=tuple(range(8)), resume=False,
                   allow_final=False, mode="development", identity=None):
    """Resume individual episodes under an unchanged pool/model/runtime contract."""
    import torch
    torch.set_num_threads(1)
    bases = sorted(bases, key=lambda base: (base.seed, base.motif))
    views = tuple(views)
    if not bases or len({(b.seed, b.motif) for b in bases}) != len(bases):
        raise ValueError("Evaluation needs nonempty, distinct layouts")
    if not views or len(set(views)) != len(views) or any(view not in range(8) for view in views):
        raise ValueError("Choose distinct D4 views in 0..7")
    if allow_final != (mode == "final"):
        raise ValueError("Final access and evidence labels must agree")
    if not allow_final and any(base.motif not in C.DEV_MOTIFS for base in bases):
        raise ValueError("Transfer structure is excluded from development evaluation")
    current = json.loads(json.dumps(runtime_manifest()))
    manifests = [inspect_run(path) for path in runs]
    if manifests:
        validate_matched_runs(manifests)
    for manifest in manifests:
        for key in ("source_hashes", "config", "python", "packages"):
            if manifest[key] != current[key]:
                raise ValueError(f"Learner {key} differs from the evaluation environment")
        if allow_final and manifest["settings"]["mode"] != "full":
            raise ValueError("Final evaluation accepts only registered full learners")
    ordered = sorted(zip(runs, manifests), key=lambda pair: (pair[1]["settings"]["seed"], pair[1]["settings"]["condition"]))
    policies = [{"condition": "heuristic", "learner_seed": -1, "action_seed": -1}]
    policies.extend({"condition": "random", "learner_seed": -1, "action_seed": seed}
                    for seed in C.RANDOM_AGENT_ACTION_SEEDS)
    paths = {}
    for path, manifest in ordered:
        entry = {"condition": manifest["settings"]["condition"], "learner_seed": manifest["settings"]["seed"],
                 "action_seed": -1, "model_sha256": manifest["model_sha256"],
                 "manifest_sha256": sha256(Path(path) / "manifest.json"), "training_mode": manifest["settings"]["mode"]}
        paths[len(policies)] = path
        policies.append(entry)
    contract = {"schema_version": 1, "kind": "episode-evaluation", "mode": mode,
                "identity": identity, "views": list(views), "policies": policies, "torch_threads": 1,
                "bases_sha256": hashlib.sha256(json_bytes([json.loads(b.to_json()) for b in bases])).hexdigest(),
                **{key: current[key] for key in ("source_hashes", "config", "python", "packages")}}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=resume or allow_final)
    progress = output / "evaluation.progress.sqlite3"
    with open_journal(progress, contract, resume=resume) as journal:
        for policy_index, entry in enumerate(policies):
            policy = None
            for base_index, base in enumerate(bases):
                for view in views:
                    key = f"{policy_index:04d}:{base_index:06d}:{view}"
                    if journal.get(key) is not None:
                        continue
                    if policy is None:
                        policy = (HeuristicCommander() if entry["condition"] == "heuristic" else
                                  RandomAgent(entry["action_seed"]) if entry["condition"] == "random" else
                                  load_run(paths[policy_index])[0])
                    record = episode_record(policy, base, view, condition=entry["condition"],
                        learner_seed=entry["learner_seed"], allow_final=allow_final, mode=mode)
                    record["action_seed"] = entry["action_seed"]
                    record["training_mode"] = entry.get("training_mode", "not trained")
                    journal.put(key, record)
        frame = pd.DataFrame([record for _, record in journal.rows()])
        expected = len(bases) * len(views) * len(policies)
        if len(frame) != expected:
            raise ValueError("Evaluation journal contains an unexpected episode count")
        publish_bytes(output / "episodes.csv", frame.to_csv(index=False).encode())
        if manifests:
            publish_json(output / "paired_summary.json", paired_summary(frame))
        publish_json(output / "study_summary.json", study_summary(frame))
        publish_json(output / "evaluation.json", {"schema_version": 1, "status": "completed", "mode": mode,
            "contract_sha256": hashlib.sha256(json_bytes(contract)).hexdigest(),
            "episodes_sha256": sha256(output / "episodes.csv"), "rows": expected,
            "accepted_layouts": len(bases), "views": list(views),
            "policies": policies, "source_hashes": current["source_hashes"], "identity": identity,
            "note": "Development and pilot rows are not final research evidence." if not allow_final else "Frozen final evaluation."})
        return frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", required=True)
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--out", required=True)
    p.add_argument("--all-views", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    bases = load_pool(args.pool, required_split="dev")
    views = tuple(range(8)) if args.all_views else (0,)
    frame = evaluate_study(bases, args.runs, args.out, views=views, resume=args.resume,
                           identity={"pool_sha256": sha256(args.pool)})
    print(frame.groupby("condition")["destruction"].agg(["mean", "count"]).to_string())


if __name__ == "__main__":
    main()

"""Offline layout validation with an explicit, recorded sampling distribution."""
import argparse
import json
from collections import Counter
from pathlib import Path
import math

from env import config as C
from env.access import ensure_seed_allowed
from env.generator import BaseSpec, generate_base, placement_region, placement_compartments, _inner_region, wall_tiles
from experiments.provenance import config_snapshot, source_hashes, runtime_manifest
from experiments.journal import open_journal
from experiments.storage import publish_json


def pool_source_hashes():
    return {path: digest for path, digest in source_hashes().items()
            if path.startswith(("env/", "agents/")) or path in
            ("experiments/pools.py", "experiments/storage.py", "experiments/journal.py")}


def validate_structure(spec):
    if spec.motif not in C.MOTIFS or set(spec.walls) != wall_tiles(spec.motif):
        return False, "wall geometry"
    positions = [(r, c) for _, r, c in spec.buildings]
    if len(set(positions)) != len(positions) or len(set(spec.walls)) != len(spec.walls):
        return False, "overlapping entities"
    if set(positions) & set(spec.walls):
        return False, "building on wall"
    if Counter(t for t, _, _ in spec.buildings) != Counter(C.BUILDING_COUNTS):
        return False, "building composition"
    if any(tile not in placement_region(spec.motif) for tile in positions):
        return False, "outside placement region"
    th = next((r, c) for t, r, c in spec.buildings if t == "town_hall")
    if th not in _inner_region(spec.motif):
        return False, "Town Hall outside core"
    if any(not (set(positions) & compartment) for compartment in placement_compartments(spec.motif)):
        return False, "empty compartment"
    coverage = sum(C.BUILDING_STATS[t]["min_range"] <= math.dist(th, (r, c)) <= C.BUILDING_STATS[t]["range"]
                   for t, r, c in spec.buildings if t in C.DEFENCE_TYPES)
    if coverage < 2:
        return False, "Town Hall covered by fewer than two defences"
    return True, "accepted"


def validate_base(spec, *, allow_final=False):
    from agents.heuristic import evaluate_from_all_sides
    valid, reason = validate_structure(spec)
    if not valid:
        return False, None, reason
    score = evaluate_from_all_sides(spec, allow_final=allow_final)
    return C.DIFFICULTY_MIN <= score <= C.DIFFICULTY_MAX, score, "difficulty"


def build_pool(start, stop, motifs, path, *, difficulty=True, resume=False, allow_final=False):
    if stop < start or not motifs or any(m not in C.MOTIFS for m in motifs):
        raise ValueError("Invalid seed interval or motif list")
    # Check the entire requested interval before generating any base.
    for seed in range(start, stop + 1):
        ensure_seed_allowed(seed, allow_final=allow_final)
    path = Path(path)
    if path.exists() and not resume:
        raise FileExistsError(f"Pool already exists: {path}")
    contract = {"schema_version": 1, "filter": "structure-and-difficulty" if difficulty else "structure-only",
                "source_hashes": pool_source_hashes(), "config": config_snapshot(),
                "seed_interval": [start, stop], "motifs": list(motifs), "allow_final": allow_final}
    runtime = runtime_manifest()
    contract["runtime"] = {key: runtime[key] for key in ("python", "packages")}
    with open_journal(path.with_suffix(".progress.sqlite3"), contract, resume=resume) as journal:
        kept, rejected, records = [], Counter(), []
        for seed in range(start, stop + 1):
            for motif in motifs:
                key = f"{seed}:{motif}"
                item = journal.get(key)
                if item is None:
                    spec = generate_base(seed, motif, allow_final=allow_final)
                    if difficulty:
                        valid, score, reason = validate_base(spec, allow_final=allow_final)
                    else:
                        valid, reason = validate_structure(spec)
                        score = None
                    item = {"record": {"seed": seed, "motif": motif, "accepted": valid,
                            "best_heuristic": score, "reason": reason},
                            "base": json.loads(spec.to_json()) if valid else None}
                    journal.put(key, item)
                records.append(item["record"])
                if item["base"] is not None:
                    kept.append(item["base"])
                else:
                    rejected[item["record"]["reason"]] += 1
        manifest = {k: v for k, v in contract.items() if k != "allow_final"}
        manifest.update(accepted=len(kept), attempted=len(records), rejections=dict(rejected),
                        bases=kept, validation=records)
        publish_json(path, manifest)
        return manifest


def load_pool(path, *, required_split=None):
    path = Path(path)
    if "final" in path.name.lower():
        raise PermissionError("Final pool access is restricted to the frozen evaluation protocol")
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1 or not data.get("bases"):
        raise ValueError("Empty or unsupported base pool")
    if data.get("config") != json.loads(json.dumps(config_snapshot())):
        raise ValueError("Pool was built with different game configuration; rebuild it")
    if data.get("source_hashes") != pool_source_hashes():
        raise ValueError("Pool generator or validation code changed; rebuild the pool")
    bases = [BaseSpec.from_json(json.dumps(row)) for row in data["bases"]]
    if len({(b.seed, b.motif) for b in bases}) != len(bases):
        raise ValueError("Duplicate layouts in pool")
    for spec in bases:
        ensure_seed_allowed(spec.seed)
        valid, reason = validate_structure(spec)
        if not valid:
            raise ValueError(f"Invalid pool layout: {reason}")
        if required_split:
            low, high = {"train": C.TRAIN_SEED_RANGE, "dev": C.DEV_SEED_RANGE}[required_split]
            if not low <= spec.seed <= high:
                raise ValueError(f"Pool contains a seed outside {required_split}")
            if required_split == "train" and spec.motif not in C.TRAIN_MOTIFS:
                raise ValueError("Transfer motif must not enter the training pool")
            if required_split == "dev" and spec.motif not in C.DEV_MOTIFS:
                raise ValueError("Transfer motif must not enter the development pool")
    return bases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "dev"], required=True)
    parser.add_argument("--count", type=int, default=20, help="Number of candidate seeds")
    parser.add_argument("--out", required=True)
    parser.add_argument("--structure-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    low, high = C.TRAIN_SEED_RANGE if args.split == "train" else C.DEV_SEED_RANGE
    if not 1 <= args.count <= high - low + 1:
        parser.error("count is outside the selected partition")
    result = build_pool(low, low + args.count - 1, C.TRAIN_MOTIFS if args.split == "train" else C.DEV_MOTIFS,
                        args.out, difficulty=not args.structure_only, resume=args.resume)
    print(f"Accepted {result['accepted']}/{result['attempted']} candidates")


if __name__ == "__main__":
    main()

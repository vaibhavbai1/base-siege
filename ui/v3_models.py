"""V3 application registry: the TRPO arm, kept separate from the V2 registry.

ui/v2_checkpoints.json and ui/v2_models.py are left exactly as they are. V3
checkpoints live in ui/v3_checkpoints.json and are read here, so a problem with
the TRPO arm can never invalidate the twenty approved V2 models.

    python -m ui.v3_models check
    python -m ui.v3_models register --run runs/study-v3/T-0 --condition T --seed 0
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = Path(__file__).with_name("v3_checkpoints.json")

# Conditions this registry is allowed to hold. P and PA stay in the V2 registry.
V3_CONDITIONS = ("T",)


def read_registry(path=REGISTRY):
    path = Path(path)
    if not path.is_file():
        return {"schema": 1, "version": "V3", "mode": "DS", "models": [], "source_hashes": {}}
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label_v3(manifest, target, registry):
    """Tag a manifest so the application can tell V3 models apart from V2."""
    result = dict(manifest)
    result["application_version"] = "V3"
    result["algorithm"] = target.get("algorithm", "MaskableTRPO")
    result["source_hashes"] = registry.get("source_hashes", {})
    result["settings"] = {
        **manifest.get("settings", {}),
        "condition": target["condition"],
        "seed": target["learner_seed"],
    }
    return result


def application_catalog(root=ROOT, *, registry=None):
    """Approved V3 checkpoints, as (path, manifest) pairs the app can show."""
    root = Path(root)
    registry = registry or read_registry()
    catalog, issues = [], []
    for target in registry.get("models", []):
        path = root / target["path"]
        try:
            if not path.is_file():
                raise FileNotFoundError("checkpoint file is missing")
            if digest(path) != target["sha256"]:
                raise ValueError("checkpoint checksum differs")
            manifest_path = root / target["manifest_path"]
            if digest(manifest_path) != target["manifest_sha256"]:
                raise ValueError("manifest checksum differs")
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("status") != "completed":
                raise ValueError("run is incomplete")
            if manifest.get("actual_steps") != target["transitions"]:
                raise ValueError("transition count differs from the approved checkpoint")
            settings = manifest.get("settings", {})
            if settings.get("condition") != target["condition"] or settings.get("seed") != target["learner_seed"]:
                raise ValueError("learner identity differs")
            catalog.append((path, label_v3(manifest, target, registry)))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append(f"{target.get('run', target.get('path'))}: {exc}")
    catalog.sort(key=lambda item: (item[1]["settings"]["condition"], item[1]["settings"]["seed"]))
    return catalog, issues


def check(root=ROOT):
    """Validate the V3 registry without training or writing research data."""
    registry = read_registry()
    catalog, problems = application_catalog(root, registry=registry)
    report = {
        "status": "ready" if not problems and catalog else ("empty" if not catalog else "problems"),
        "registry": str(REGISTRY.relative_to(ROOT)),
        "verified_v3_models": len(catalog),
        "conditions": sorted({m["settings"]["condition"] for _, m in catalog}),
        "problems": problems,
    }
    if not catalog and not problems:
        report["note"] = "No V3 checkpoints registered yet. Train the TRPO arm, then register it."
    return report


def register(run_dir, *, condition, learner_seed, root=ROOT, registry_path=REGISTRY, algorithm="MaskableTRPO"):
    """Add a completed TRPO run to the V3 registry.

    Refuses anything that is not finished, so a half-trained learner cannot reach
    the application.
    """
    if condition not in V3_CONDITIONS:
        raise ValueError(f"condition must be one of {V3_CONDITIONS}; P and PA belong to the V2 registry")
    root = Path(root).resolve()
    # Accept either a path relative to the project root or an absolute one.
    run_dir = Path(run_dir)
    run_dir = (run_dir if run_dir.is_absolute() else root / run_dir).resolve()
    if root not in run_dir.parents:
        raise ValueError(f"{run_dir} is outside the project root; registered paths must be inside it")
    model_path = run_dir / "model.zip"
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "completed":
        raise ValueError(f"run status is {manifest.get('status')!r}, not 'completed'")
    # Refuse at registration rather than letting `check` fail later with a
    # registry that already claims the learner.
    settings = manifest.get("settings", {})
    if settings.get("condition") != condition:
        raise ValueError(f"run was trained as condition {settings.get('condition')!r}, not {condition!r}")
    if settings.get("seed") != learner_seed:
        raise ValueError(f"run was trained with seed {settings.get('seed')!r}, not {learner_seed!r}")

    registry = read_registry(registry_path)
    entry = {
        "algorithm": algorithm,
        "condition": condition,
        "learner_seed": learner_seed,
        "manifest_path": str(manifest_path.relative_to(root)),
        "manifest_sha256": digest(manifest_path),
        "path": str(model_path.relative_to(root)),
        "run": run_dir.name,
        "sha256": digest(model_path),
        "transitions": manifest["actual_steps"],
    }
    models = [m for m in registry.get("models", [])
              if not (m["condition"] == condition and m["learner_seed"] == learner_seed)]
    models.append(entry)
    registry["models"] = sorted(models, key=lambda m: (m["condition"], m["learner_seed"]))
    registry.setdefault("schema", 1)
    registry.setdefault("version", "V3")
    registry.setdefault("mode", "DS")

    from experiments.provenance import source_hashes
    registry["source_hashes"] = source_hashes()

    Path(registry_path).write_text(json.dumps(registry, indent=1, sort_keys=True) + "\n")
    return entry


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(prog="python -m ui.v3_models")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    add = sub.add_parser("register")
    add.add_argument("--run", required=True)
    add.add_argument("--condition", default="T")
    add.add_argument("--seed", type=int, required=True)

    args = parser.parse_args(argv)
    if args.command == "check":
        print(json.dumps(check(), indent=2))
    else:
        print(json.dumps(register(args.run, condition=args.condition, learner_seed=args.seed), indent=2))


if __name__ == "__main__":
    sys.exit(main())

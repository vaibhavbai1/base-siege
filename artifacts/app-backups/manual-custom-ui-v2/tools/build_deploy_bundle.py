"""Build a small standalone source bundle; never publish or alter Git state."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]

# Explicit list: a newly added notebook, archive, secret or checkpoint is not deployable.
SOURCE_FILES = (
    ".streamlit/config.toml", "ui/streamlit_app.py", "ui/app.py", "ui/deployment.py",
    "ui/policy_playback.py", "tools/export_replay_html.py", "env/symmetry.py",
    "ui/requirements.txt", "ui/frontend/index.html", "ui/frontend/app.js",
    "ui/frontend/theme.css", "ui/frontend/playback.css", "ui/frontend/glyphs.js", "env/__init__.py",
    "env/access.py", "env/config.py", "env/entities.py", "env/pathfinding.py",
    "env/generator.py", "env/combat.py", "env/siege_env.py", "agents/__init__.py",
    "agents/heuristic.py", "agents/random_agent.py", "experiments/__init__.py",
    "experiments/rollout.py", "train/__init__.py", "train/features.py",
    "demo_assets/README.md", "docs/DEPLOYMENT.md",
    "tools/check_public_demo.py", "tests/test_public_app.py",
    "deployment/requirements-test.txt", "deployment/streamlit-ci.yml",
)


def build_bundle(output, *, root=ROOT):
    output, root = Path(output).resolve(), Path(root).resolve()
    if output.exists() or output.with_suffix(".zip").exists():
        raise FileExistsError("Choose a new bundle folder; previous bundles are preserved.")
    files = [(Path(name), Path(name)) for name in SOURCE_FILES]
    for pattern in ("models/*/model.zip", "models/*/manifest.json", "replays/*.json", "evaluations/*.csv"):
        files.extend((p.relative_to(root), p.relative_to(root))
                     for p in sorted((root / "demo_assets").glob(pattern)))
    for source, _ in files:
        path = root / source
        if not path.is_file() or any((root / parent).is_symlink() for parent in (source, *source.parents)):
            raise ValueError(f"Missing file or symbolic link in bundle input: {source}")
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for source, target in files:
        if str(source) == "deployment/streamlit-ci.yml":
            target = Path(".github/workflows/ci.yml")
        destination = output / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / source, destination)
        hashes[str(target)] = hashlib.sha256(destination.read_bytes()).hexdigest()
    (output / "README.md").write_text(
        "# Base Siege public demo\n\nUTS 43008, Project 12, Dihedrals.\n\n"
        "Deploy with Python 3.12 and entrypoint `ui/streamlit_app.py`. "
        "No Docker, S3 or secrets are needed.\n\n"
        "Read [Deployment](docs/DEPLOYMENT.md). Training code and recovery checkpoints "
        "remain in the main project. The baseline policies are not learned policies.\n")
    (output / "bundle-manifest.json").write_text(json.dumps({"schema_version": 1, "files": hashes}, indent=2) + "\n")
    with zipfile.ZipFile(output.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output))
    return {"folder": str(output), "zip": str(output.with_suffix(".zip")), "files": len(hashes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/streamlit-deploy")
    args = parser.parse_args()
    print(json.dumps(build_bundle(args.out), indent=2))


if __name__ == "__main__":
    main()

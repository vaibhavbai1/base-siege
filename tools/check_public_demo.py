"""Short public-app preflight. No training, model export or network writes."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
from time import perf_counter


def check(root):
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    import torch
    from streamlit.testing.v1 import AppTest
    from ui.deployment import asset_paths, load_cpu_policy
    from env.siege_env import SiegeEnv
    from experiments.rollout import play_episode

    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use Python 3.12 to match training and the pinned deployment wheel.")
    if sys.platform == "linux" and torch.version.cuda is not None:
        raise RuntimeError("The Linux demo must use the CPU-only PyTorch wheel.")
    app = AppTest.from_file(str(root / "ui/streamlit_app.py")).run(timeout=30)
    if app.exception or app.title[0].value != "Base Siege":
        raise RuntimeError("Public app did not render successfully.")
    mode = next(widget for widget in app.selectbox if widget.label == "Mode")
    mode.set_value("Policy replay").run(timeout=30)
    next(button for button in app.button if button.label == "Generate replay").click().run(timeout=30)
    if app.exception or not app.session_state["replay"]["summary"]["finish_reason"]:
        raise RuntimeError("Heuristic replay did not finish.")
    models = []
    for path in asset_paths(root, "model", public=True):
        manifest = json.loads(path.with_name("manifest.json").read_text())
        stamp = path.stat()
        policy = load_cpu_policy(str(path), stamp.st_mtime_ns, stamp.st_size, manifest["model_sha256"])
        start = perf_counter()
        env = SiegeEnv(base_seed=42, motifs=("closed_ring",), record=True)
        try:
            replay = play_episode(env, policy)
        finally:
            env.close()
        models.append({"name": path.parent.name, "seconds": round(perf_counter() - start, 3),
                       "decisions": len(replay["actions"]), "evidence_scope": manifest["evidence_scope"]})
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"status": "passed", "python": sys.version.split()[0], "platform": sys.platform,
            "torch": importlib.metadata.version("torch"), "cuda_runtime": torch.version.cuda,
            "peak_process_rss_mib": round(peak / (1024 ** 2 if sys.platform == "darwin" else 1024), 1),
            "heuristic_replay": "completed", "published_models": models,
            "scope": "Local single-process preflight, not a cloud capacity or policy-quality benchmark"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = check(args.root)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))

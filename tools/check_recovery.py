"""Short CPU recovery experiment; compare continuous, resumed and killed runs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
from train.checkpoints import RecoverablePPO, atomic_json, latest_checkpoint
from train.train_ppo import TrainingSettings, train, resume


def same_tree(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same_tree(x, y) for x, y in zip(a, b))
    return a == b


def compare_runs(left, right):
    a = RecoverablePPO.load(Path(left) / "model.zip", device="cpu")
    b = RecoverablePPO.load(Path(right) / "model.zip", device="cpu")
    return {"weights_equal": same_tree(a.policy.state_dict(), b.policy.state_dict()),
            "optimizer_equal": same_tree(a.policy.optimizer.state_dict(), b.policy.optimizer.state_dict()),
            "episode_history_equal": (Path(left) / "episodes.jsonl").read_bytes() == (Path(right) / "episodes.jsonl").read_bytes(),
            "timesteps_equal": a.num_timesteps == b.num_timesteps,
            "updates_equal": a._n_updates == b._n_updates}


def recovery_lab(output, *, kill_process=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    settings = TrainingSettings(condition="PA", placement_steps=64, joint_steps=192)
    train(settings, output / "continuous")
    train(settings, output / "resumed", pause_after_steps=96)
    resume(output / "resumed")
    checks = {"pause_resume": compare_runs(output / "continuous", output / "resumed")}
    if kill_process:
        run = output / "killed"
        command = [sys.executable, "-m", "train.train_ppo", "--condition", "PA", "--out", str(run)]
        with (output / "killed-process.log").open("w") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 45
                while time.monotonic() < deadline:
                    if child.poll() is not None:
                        raise RuntimeError("Child finished before the interruption test")
                    if (run / "LATEST.json").exists():
                        point = json.loads((run / "LATEST.json").read_text())
                        if point["actual_steps"] >= 32:
                            child.kill()  # Only the test process created immediately above.
                            child.wait(timeout=10)
                            break
                    time.sleep(.005)
                else:
                    raise TimeoutError("No completed checkpoint published within 45 seconds")
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=10)
        checkpoint, metadata, rejected = latest_checkpoint(run)
        resume(run)
        checks["process_kill_resume"] = compare_runs(output / "continuous", run)
        atomic_json(output / "interruption.json", {"signal": "SIGKILL", "recovered_steps": metadata["actual_steps"],
            "checkpoint": checkpoint.name, "rejected": rejected})
    result = {"scope": "short CPU software recovery check; not a trained-policy result", "checks": checks,
              "passed": all(all(row.values()) for row in checks.values())}
    atomic_json(output / "recovery_report.json", result)
    if not result["passed"]:
        raise AssertionError(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--kill-process", action="store_true")
    args = parser.parse_args()
    print(json.dumps(recovery_lab(args.out, kill_process=args.kill_process), indent=2))


if __name__ == "__main__":
    main()

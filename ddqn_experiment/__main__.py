"""Run from the Base Siege repository root: python -m ddqn_experiment --help."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from ddqn_experiment.core import Settings
from ddqn_experiment.runner import checked_pool, train, runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="Validate existing frozen pools; no training")
    check.add_argument("--train-pool", default="bases/train-v2.json")
    check.add_argument("--dev-pool", default="bases/dev-v2.json")
    training = sub.add_parser("train", help="Start or resume one independent learner")
    training.add_argument("--out", required=True)
    training.add_argument("--seed", type=int, default=100)
    training.add_argument("--mode", choices=["pilot", "main"], default="pilot")
    training.add_argument("--train-pool", default="bases/train-v2.json")
    training.add_argument("--dev-pool", default="bases/dev-v2.json")
    training.add_argument("--settings", help="JSON overrides of Settings; fixed for the run")
    training.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    training.add_argument("--until", type=int, help="Absolute pause point; total plan remains unchanged")
    training.add_argument("--resume", action="store_true")
    smoke = sub.add_parser("smoke", help="64-transition integration check; not performance evidence")
    smoke.add_argument("--out", required=True)
    for name in ("status", "pause"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
    for name in ("evaluate", "replay"):
        p = sub.add_parser(name, help="Development only; never reads final pools")
        p.add_argument("--run", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--dev-pool", default="bases/dev-v2.json")
        p.add_argument("--steps", type=int, help="Saved numbered policy; default latest")
        p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
        if name == "evaluate":
            p.add_argument("--panel", action="store_true", help="Fixed 12-layout development panel")
            p.add_argument("--views", type=int, nargs="+", default=[0, 1, 4, 5])
            p.add_argument("--resume", action="store_true")
        else:
            p.add_argument("--index", type=int, default=0)
            p.add_argument("--view", type=int, default=0)
    args = parser.parse_args()
    if args.command == "check":
        print(json.dumps({"runtime": runtime(), "settings": asdict(Settings()),
              "training_layouts": len(checked_pool(args.train_pool, "train")),
              "development_layouts": len(checked_pool(args.dev_pool, "dev"))}, indent=2))
    elif args.command == "train":
        settings = Settings(**json.loads(Path(args.settings).read_text())) if args.settings else Settings()
        train(args.out, settings, seed=args.seed, mode=args.mode, device=args.device,
              train_pool=args.train_pool, dev_pool=args.dev_pool, resume=args.resume, until=args.until)
    elif args.command == "smoke":
        settings = Settings(total_steps=64, placement_steps=16, buffer_size=64, batch_size=4,
                            learning_starts=4, train_freq=4, target_interval=8,
                            exploration_steps=32, checkpoint_interval=32, log_interval=16)
        train(args.out, settings, seed=123, mode="smoke")
    elif args.command == "status":
        root = Path(args.run)
        result = {"last_committed": json.loads((root / "LATEST.json").read_text())}
        if (root / "status.json").exists():
            result["last_exit"] = json.loads((root / "status.json").read_text())
        print(json.dumps(result, indent=2))
    elif args.command == "pause":
        root = Path(args.run)
        if not (root / "contract.json").is_file():
            raise ValueError("Not a DDQN run folder")
        (root / "STOP_REQUESTED").touch()
        print("Pause requested; wait for the trainer to print Saved and exit.")
    elif args.command == "evaluate":
        from ddqn_experiment.evaluation import evaluate
        evaluate(args.run, args.dev_pool, args.out, steps=args.steps, views=args.views,
                 panel=args.panel, resume=args.resume, device=args.device)
    elif args.command == "replay":
        from ddqn_experiment.evaluation import record_replay
        print(record_replay(args.run, args.dev_pool, args.out, steps=args.steps,
                            index=args.index, view=args.view, device=args.device))


if __name__ == "__main__":
    main()

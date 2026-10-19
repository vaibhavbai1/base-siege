"""Command line for the V3 final evaluation."""
import argparse
import sys

from v3_final import runner


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m v3_final", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze", help="record the protocol and approved checkpoints; generates nothing")
    pool = sub.add_parser("pool", help="generate the fresh V3 layout pool")
    pool.add_argument("--resume", action="store_true")
    run = sub.add_parser("run", help="evaluate every condition on the frozen pool")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--max-new", type=int, default=None)
    sub.add_parser("export", help="write episode, model, condition and comparison tables")

    args = parser.parse_args(argv)
    if args.command == "freeze":
        runner.freeze()
    elif args.command == "pool":
        runner.build_pool(resume=args.resume)
    elif args.command == "run":
        runner.execute(resume=args.resume, max_new=args.max_new)
    else:
        runner.export()


if __name__ == "__main__":
    sys.exit(main())

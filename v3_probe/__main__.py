"""Command line for the acting-mode probe."""
import argparse
import sys

from v3_probe import runner


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m v3_probe", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help=f"re-evaluate the frozen V3 models under {runner.PROBE_MODE}")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--max-new", type=int, default=None,
                     help="stop after N new attacks; use a small value for a smoke test")
    run.add_argument("--conditions", nargs="+", default=list(runner.DEFAULT_CONDITIONS),
                     help="conditions to probe (default: P T -- T is the question, P is the control)")
    run.add_argument("--mode", default=runner.PROBE_MODE, choices=sorted(runner.MODES),
                     help="acting mode to probe against the frozen DS baseline")
    run.add_argument("--out", default=str(runner.OUTPUT))

    export = sub.add_parser("export", help="write the probe tables and the DS-vs-SS difference")
    export.add_argument("--out", default=str(runner.OUTPUT))

    args = parser.parse_args(argv)
    if args.command == "run":
        runner.execute(output=args.out, conditions=args.conditions, mode=args.mode,
                       resume=args.resume, max_new=args.max_new)
    else:
        runner.export(output=args.out)


if __name__ == "__main__":
    sys.exit(main())

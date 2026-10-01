"""Separately registered Version 2 training for P/PA seeds 3–9."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('preflight')
    register = sub.add_parser('register')
    register.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    for name in ('train', 'matrix'):
        command = sub.add_parser(name)
        command.add_argument('--to', type=int, default=4194304)
        if name == 'train':
            command.add_argument('--condition', choices=['P', 'PA'], required=True)
            command.add_argument('--seed', type=int, choices=range(3, 10), required=True)
    for name in ('pause', 'status', 'report', 'pack'):
        sub.add_parser(name)
    args = parser.parse_args()
    from v2_replication import workflow
    if args.command in ('pause', 'status', 'report', 'pack'):
        return getattr(workflow, args.command)()
    if args.command in ('train', 'matrix'):
        return workflow.run_training(args.to, getattr(args, 'condition', None), getattr(args, 'seed', None))
    from v2_replication import runner
    if args.command == 'preflight':
        from v2_replication.protocol import verify_evidence, read_plan
        runtime = runner.checked_runtime()
        plan = read_plan()
        print(json.dumps({**verify_evidence(), 'hardware': runtime['hardware'],
                          'transitions_per_learner': plan['total_steps'],
                          'total_new_transitions': 14*plan['total_steps']}, indent=2))
    else:
        print(runner.register(args.device))


if __name__ == '__main__':
    main()

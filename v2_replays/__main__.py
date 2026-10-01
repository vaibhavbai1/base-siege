import argparse
import json
from . import runner as R


def main():
    parser = argparse.ArgumentParser(description='Reproduce reviewed V2 development attacks; no training or mechanics changes.')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('preflight')
    commands.add_parser('pack')
    run = commands.add_parser('run')
    run.add_argument('--max-new-cases', type=int)
    args = parser.parse_args()
    if args.command == 'pack':
        R.pack()
        return
    if args.command == 'run' and args.max_new_cases is not None and args.max_new_cases < 1:
        parser.error('--max-new-cases must be positive')
    contract, plan, pool = R.prepare()
    if args.command == 'preflight':
        print(json.dumps({'status': 'ready', 'cases': len(plan['cases']), 'pairs': len(plan['pairs']),
                          'models': len(contract['models']), 'new_training': False}, indent=2))
    else:
        R.execute(contract, plan, pool, max_cases=args.max_new_cases)


if __name__ == '__main__':
    main()

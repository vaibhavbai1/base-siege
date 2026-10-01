import argparse
import json
from . import runner as R


def main():
    parser = argparse.ArgumentParser(description='Phase-specific action selection with existing checkpoints. No training.')
    commands = parser.add_subparsers(dest='command',required=True)
    for name in ('preflight','status','pause','pack'):
        commands.add_parser(name)
    run = commands.add_parser('run')
    run.add_argument('--max-new-episodes',type=int)
    args = parser.parse_args()
    if args.command == 'status':
        path = R.OUTPUT/'progress.json'
        print(json.dumps(R.read(path),indent=2) if path.exists() else 'No phase evaluation progress yet.')
    elif args.command == 'pause':
        if not (R.OUTPUT/'evaluation_contract.json').exists():
            raise SystemExit('Phase evaluation has not started.')
        R.atomic_json(R.OUTPUT/'STOP_REQUESTED.json',{'requested_utc':R.E.now()})
        print('Pause requested. Wait for paused status and the original terminal prompt before stopping SageMaker.')
    elif args.command == 'pack':
        R.pack()
    else:
        if args.command == 'run' and args.max_new_episodes is not None and args.max_new_episodes < 1:
            parser.error('--max-new-episodes must be positive')
        contract,pool,baselines = R.prepare()
        if args.command == 'preflight':
            print(json.dumps({'status':'ready','models':len(contract['models']),'layouts':len(pool),
                              'new_episodes':contract['expected_episodes'],'reused_baseline_episodes':len(baselines),
                              'new_modes':contract['new_modes'],'device':'cpu'},indent=2))
        else:
            R.execute(contract,pool,baselines,max_new=args.max_new_episodes)


if __name__ == '__main__':
    main()

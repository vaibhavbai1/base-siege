import argparse
import json
from . import runner as R


def main():
    p=argparse.ArgumentParser(description='Evaluate existing V2 milestones on the full development pool; no training.')
    commands=p.add_subparsers(dest='command',required=True)
    commands.add_parser('preflight')
    commands.add_parser('status')
    commands.add_parser('pause')
    run=commands.add_parser('run')
    run.add_argument('--max-new-episodes',type=int,help='Pause after this many additional committed evaluations')
    args=p.parse_args()
    if args.command=='status':
        path=R.OUTPUT/'progress.json'
        print(json.dumps(R.read(path),indent=2) if path.exists() else 'Evaluation has not recorded progress yet.')
    elif args.command=='pause':
        if not (R.OUTPUT/'evaluation_contract.json').exists():
            raise SystemExit('Evaluation has not started.')
        R.atomic_json(R.OUTPUT/'STOP_REQUESTED.json',{'requested_utc':R.now()})
        print('Pause requested. Wait for status=paused; the current episode will finish first.')
    else:
        if args.command=='run' and args.max_new_episodes is not None and args.max_new_episodes<1:
            p.error('--max-new-episodes must be positive')
        contract,pool=R.prepare()
        if args.command=='preflight':
            print(json.dumps({'model_milestones_verified':len(contract['models']),
                'development_layouts':len(pool),'views':contract['views'],
                'expected_episodes':contract['expected_episodes'],'device':'cpu',
                'output':str(R.OUTPUT)},indent=2))
        else:
            R.execute(contract,pool,max_new=args.max_new_episodes)


if __name__=='__main__':
    main()

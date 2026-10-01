import argparse
import json
from . import runner as R


def main():
    parser=argparse.ArgumentParser(description='Frozen Version 2 final evaluation. No training.')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('preflight','freeze','status','pause','pack'):sub.add_parser(name)
    run=sub.add_parser('run');run.add_argument('--max-new-episodes',type=int)
    args=parser.parse_args()
    if args.command=='preflight':
        _,report=R.prepare();print(json.dumps(report,indent=2))
    elif args.command=='freeze':R.freeze()
    elif args.command=='run':
        if args.max_new_episodes is not None and args.max_new_episodes<1:parser.error('--max-new-episodes must be positive')
        R.run(args.max_new_episodes)
    elif args.command=='status':
        path=R.OUTPUT/'progress.json'
        print(json.dumps(R.read(path),indent=2) if path.exists() else 'No final evaluation progress yet.')
    elif args.command=='pause':
        if not (R.OUTPUT/'freeze.json').exists():raise SystemExit('Final evaluation has not been frozen.')
        R.atomic_json(R.OUTPUT/'STOP_REQUESTED.json',{'requested_utc':R.now()})
        print('Pause requested. Wait for paused status and the original terminal prompt before stopping SageMaker.')
    elif args.command=='pack':R.pack()


if __name__=='__main__':main()

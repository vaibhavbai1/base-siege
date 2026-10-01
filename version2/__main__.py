"""Run from the project root: python -m version2 --help."""
import argparse
import csv
import json
import sys
from version2.protocol import ROOT, read_plan, verify_baseline


def status():
    plan = read_plan()
    for condition in plan['conditions']:
        for seed in plan['learner_seeds']:
            path = ROOT/'runs'/plan['experiment']/f'{condition}-{seed}'
            if not (path/'manifest.json').exists():
                print(f'{condition}-{seed}: not started')
                continue
            info = json.loads((path/'manifest.json').read_text())
            checkpoint = path/'checkpoints'/info.get('latest_checkpoint', '')/'episodes.jsonl'
            episodes = sum(1 for _ in checkpoint.open()) if checkpoint.is_file() else 0
            print(f'{condition}-{seed}: {info["status"]}; {info["actual_steps"]:,} committed transitions; '
                  f'{episodes:,} completed episodes; {info.get("wall_seconds", 0):.1f}s recorded wall time')


def report():
    plan = read_plan()
    fields = ['condition','learner_seed','transitions','mode','episodes',
              'destruction_percent','full_clear_percent','town_hall_percent']
    writer = csv.DictWriter(sys.stdout, fieldnames=fields)
    writer.writeheader()
    for condition in plan['conditions']:
        for seed in plan['learner_seeds']:
            path = ROOT/'runs'/plan['experiment']/f'{condition}-{seed}'
            if not (path/'development.json').is_file():
                continue
            info = json.loads((path/'manifest.json').read_text())
            for row in json.loads((path/'development.json').read_text())['records']:
                if row['timesteps'] > info['actual_steps']:
                    continue
                writer.writerow(dict(condition=condition, learner_seed=seed, transitions=row['timesteps'],
                    mode=row['mode'], episodes=row['episodes'], destruction_percent=100*row['mean_destruction'],
                    full_clear_percent=100*row['full_clear_rate'], town_hall_percent=100*row['town_hall_rate']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('preflight', help='Check original files and scientific dependencies')
    reg = sub.add_parser('register', help='Record the development plan before training')
    reg.add_argument('--device', choices=['cpu','cuda'], required=True)
    train = sub.add_parser('train', help='Start or resume one model')
    train.add_argument('--condition', choices=['P','PA'], required=True)
    train.add_argument('--seed', type=int, required=True)
    train.add_argument('--to', type=int, default=1048576, help='Absolute transition count to reach')
    matrix = sub.add_parser('matrix', help='Sequentially start/resume the six screening models')
    matrix.add_argument('--to', type=int, default=1048576)
    pause = sub.add_parser('pause', help='Ask the current model to pause safely')
    pause.add_argument('--condition', choices=['P','PA'], required=True)
    pause.add_argument('--seed', type=int, required=True)
    sub.add_parser('status')
    sub.add_parser('report', help='Print development diagnostics as CSV')
    args = parser.parse_args()
    if args.command == 'status':
        return status()
    if args.command == 'report':
        return report()
    from version2 import runner
    if args.command == 'preflight':
        baseline = verify_baseline()
        current = runner.checked_runtime()
        plan = read_plan()
        print(json.dumps({'verified_original_source_files': len(baseline['source_hashes']),
            'verified_pools': list(baseline['pools']), 'hardware': current['hardware'],
            'screen_steps_per_model': plan['screen_steps'], 'total_steps_per_model': plan['total_steps']}, indent=2))
    elif args.command == 'register':
        print(runner.register(args.device))
    elif args.command == 'pause':
        runner.request_pause(runner.run_path(args.condition, args.seed))
        print('Pause requested. Wait for status=paused before stopping the SageMaker space.')
    elif args.command == 'train':
        result = runner.start_or_resume(args.condition, args.seed, args.to)
        print(json.dumps({k: result.get(k) for k in ('status','actual_steps','wall_seconds','pause_reason')}, indent=2))
    elif args.command == 'matrix':
        plan = read_plan()
        for condition in plan['conditions']:
            for seed in plan['learner_seeds']:
                result = runner.start_or_resume(condition, seed, args.to)
                if result['actual_steps'] < args.to or result.get('pause_reason') == 'user_request':
                    print('Matrix stopped after the current model paused.')
                    return


if __name__ == '__main__':
    main()

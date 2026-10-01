"""One training process, whole-matrix pause, and compact review archives."""
import csv
import io
import json
import uuid
import zipfile
from datetime import datetime, timezone
from experiments.storage import atomic_json, run_lock
from v2_replication.protocol import ROOT, PACKAGE, read, read_plan, digest, require, check_stop


def records():
    return ROOT/'version2-records/seed-replication'


def archive_stop():
    stop = records()/'STOP_MATRIX.json'
    if stop.exists():
        dest = records()/'handled-stops'/f'{uuid.uuid4().hex}.json'
        dest.parent.mkdir(parents=True, exist_ok=True)
        stop.rename(dest)


def run_training(target, condition=None, seed=None):
    from v2_replication import runner
    plan = read_plan()
    check_stop(target, plan)
    require((condition is None) == (seed is None), 'Specify both condition and seed')
    if condition is not None:
        require(condition in plan['conditions'] and seed in plan['learner_seeds'], 'Only P/PA seeds 3–9 are allowed')
    runner.validate_registration()
    selected = [(condition, seed)] if condition is not None else [
        (c, s) for c in plan['conditions'] for s in plan['learner_seeds']]
    directory = records()
    with run_lock(directory, '.training.lock'):
        archive_stop()
        try:
            for c, s in selected:
                if (directory/'STOP_MATRIX.json').exists():
                    print('Matrix paused before the next learner.', flush=True)
                    return
                atomic_json(directory/'active.json', {'condition': c, 'seed': s, 'target': target})
                if (directory/'STOP_MATRIX.json').exists():
                    return
                print(f'Learner {c}-{s}: target {target:,} transitions', flush=True)
                result = runner.start_or_resume(c, s, target)
                if result['actual_steps'] < target or result.get('pause_reason') == 'user_request':
                    print('Training paused. Repeat the same command to resume.', flush=True)
                    return
            print('All requested learners reached the requested target.', flush=True)
        finally:
            (directory/'active.json').unlink(missing_ok=True)


def pause():
    from train.train_ppo import request_pause
    from v2_replication.runner import run_path
    directory = records()
    require((directory/'registration.json').is_file(), 'Register the cohort first')
    atomic_json(directory/'STOP_MATRIX.json', {'requested_utc': datetime.now(timezone.utc).isoformat()})
    try:
        item = read(directory/'active.json')
    except FileNotFoundError:
        item = None
    if item:
        path = run_path(item['condition'], item['seed'])
        if (path/'contract.json').is_file():
            request_pause(path)
    print('Pause requested for the cohort. Wait for the training command to return before shutdown.')


def status_rows():
    plan = read_plan()
    rows = []
    for condition in plan['conditions']:
        for seed in plan['learner_seeds']:
            path = ROOT/'runs'/plan['experiment']/f'{condition}-{seed}'
            row = {'condition': condition, 'learner_seed': seed, 'status': 'not_started',
                   'committed_transitions': 0, 'completed_episodes': 0,
                   'recorded_wall_seconds': 0, 'unfinished_attempts': 0}
            if (path/'manifest.json').is_file():
                manifest = read(path/'manifest.json')
                episodes = path/'checkpoints'/manifest.get('latest_checkpoint', '')/'episodes.jsonl'
                require(manifest['actual_steps'] == 0 or episodes.is_file(),
                        f'Missing committed episode log for {condition}-{seed}; restore/recover before reporting counts')
                count = 0
                if episodes.is_file():
                    with episodes.open() as handle:
                        count = sum(1 for _ in handle)
                row.update(status=manifest['status'], committed_transitions=manifest['actual_steps'],
                           completed_episodes=count, recorded_wall_seconds=manifest.get('wall_seconds', 0),
                           unfinished_attempts=sum('wall_seconds' not in x for x in manifest.get('attempts', [])))
            rows.append(row)
    return rows


def status():
    rows = status_rows()
    for row in rows:
        print(f'{row["condition"]}-{row["learner_seed"]}: {row["status"]}; '
              f'{row["committed_transitions"]:,} committed transitions; '
              f'{row["completed_episodes"]:,} completed episodes; '
              f'{row["recorded_wall_seconds"]:.1f}s recorded; '
              f'{row["unfinished_attempts"]} attempts without an end time')
    print(f'Total: {sum(r["committed_transitions"] for r in rows):,} committed transitions; '
          f'{sum(r["completed_episodes"] for r in rows):,} completed episodes')
    print('Status is the last saved record, not proof a process is alive. Ended-attempt wall time includes evaluation and I/O.')


def csv_text(rows):
    result = io.StringIO(newline='')
    writer = csv.DictWriter(result, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    return result.getvalue()


def report():
    import sys
    plan = read_plan()
    writer = csv.DictWriter(sys.stdout, fieldnames=['condition','learner_seed','transitions','mode',
        'episodes','destruction_percent','full_clear_percent','town_hall_percent'])
    writer.writeheader()
    for c in plan['conditions']:
        for s in plan['learner_seeds']:
            path = ROOT/'runs'/plan['experiment']/f'{c}-{s}'
            if not (path/'development.json').is_file():
                continue
            manifest = read(path/'manifest.json')
            for row in read(path/'development.json')['records']:
                if row['timesteps'] <= manifest['actual_steps']:
                    writer.writerow(dict(condition=c, learner_seed=s, transitions=row['timesteps'], mode=row['mode'],
                        episodes=row['episodes'], destruction_percent=100*row['mean_destruction'],
                        full_clear_percent=100*row['full_clear_rate'], town_hall_percent=100*row['town_hall_rate']))


def pack():
    """Metadata for review, never represented as a complete recovery backup."""
    from v2_replication import runner
    from train.checkpoints import latest_checkpoint
    reg_path, _ = runner.validate_registration()
    with run_lock(records(), '.training.lock'):
        plan = read_plan()
        files = [reg_path, *PACKAGE.rglob('*.py'), *PACKAGE.glob('*.json'), *PACKAGE.glob('*.md')]
        for c in plan['conditions']:
            for s in plan['learner_seeds']:
                path = runner.run_path(c, s)
                if not path.exists():
                    continue
                manifest = read(path/'manifest.json')
                require(digest(path/'contract.json') == manifest['contract_sha256'], 'Run contract changed')
                require(manifest['registration_sha256'] == digest(reg_path), 'Wrong run registration')
                contract = read(path/'contract.json')
                for item in contract['inputs'].values():
                    if item:
                        require(digest(path/item['path']) == item['sha256'], 'Frozen input changed')
                if manifest['actual_steps'] > 0:
                    _, metadata, _ = latest_checkpoint(path)
                    require(metadata['actual_steps'] == manifest['actual_steps'],
                            'Latest checkpoint differs from manifest; resume/recover before packing')
                if manifest['status'] == 'completed':
                    require(manifest['actual_steps'] == plan['total_steps'], 'Incorrect completion budget')
                    require(digest(path/'model.zip') == manifest['model_sha256'], 'Final model changed')
                files.extend(p for p in [path/'manifest.json', path/'contract.json', path/'development.json',
                                        path/'LATEST.json', *path.glob('milestones/*.json')] if p.is_file())
        payloads = {str(p.relative_to(ROOT)): p.read_bytes() for p in sorted(set(files))}
        rows = status_rows()
        payloads['review/status.csv'] = csv_text(rows).encode()
        payloads['review/receipt.json'] = json.dumps({
            'scope': 'Compact review metadata, not a model or recovery backup',
            'completed_learners': sum(r['status'] == 'completed' for r in rows),
            'planned_learners': 14, 'expected_total_transitions': 14*4194304,
            'committed_transitions': sum(r['committed_transitions'] for r in rows),
            'completed_episodes': sum(r['completed_episodes'] for r in rows),
            'recorded_wall_seconds': sum(r['recorded_wall_seconds'] for r in rows),
            'attempts_without_end_times': sum(r['unfinished_attempts'] for r in rows),
            'files': {name: __import__('hashlib').sha256(data).hexdigest() for name, data in payloads.items()}
        }, indent=2).encode()
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        destination = ROOT/f'Base_Siege_V2_Replication_Review_{stamp}.zip'
        with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in payloads.items():
                archive.writestr(name, data)
        print(destination)
        print('Review ZIP excludes model weights, raw episode logs and recovery states. Keep the full runs folder separately.')

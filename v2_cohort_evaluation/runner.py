"""Read-only development inference; no training or final-pool access."""
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import csv
import hashlib
import io
import json
import math
import os
import uuid
import zipfile

from experiments.journal import open_journal
from experiments.storage import atomic_json, publish_json, publish_bytes, run_lock
from train.train_ppo import stop_signals
from version2.evaluation import episode_seed
from v2_development import runner as E
from v2_phases import runner as F
from v2_phases.runner import PhasePolicy, episode, evaluate_one, row_key, KEY_FIELDS

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
OUTPUT = ROOT/'artifacts/version2-cohort-development'
MODES = {'DS': (True, False), 'SS': (False, False), 'DD': (True, True)}


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def package_hashes():
    files = sorted(PACKAGE.rglob('*.py')) + sorted(PACKAGE.glob('*.json'))
    return {str(p.relative_to(PACKAGE)): E.sha256(p) for p in files if p.name != 'PACKAGE_SHA256.json'}


def verify_files(root, mapping):
    for name, wanted in mapping.items():
        path = root/name
        require(path.is_file(), f'Missing required file: {name}')
        require(E.sha256(path) == wanted, f'File changed: {name}; restore it, do not edit expected hashes')


def verify_replicas(root, reference):
    required = {reference['registration_path']: reference['registration_sha256'],
                **reference['replication_source_hashes']}
    for item in reference['runs'].values():
        required[item['path']+'/manifest.json'] = item['manifest_sha256']
        required[item['path']+'/contract.json'] = item['contract_sha256']
        required[item['path']+'/milestones/step-000004194304.json'] = item['final_metadata_sha256']
    for model in reference['models']:
        required[model['path']] = model['sha256']
    verify_files(root, required)
    registration = read(root/reference['registration_path'])
    for model in reference['models']:
        item = reference['runs'][model['run']]
        manifest = read(root/item['path']/'manifest.json')
        contract = read(root/item['path']/'contract.json')
        metadata = read(root/item['path']/'milestones/step-000004194304.json')
        require(manifest['status'] == 'completed' and manifest['actual_steps'] == 4194304,
                f'Incomplete learner: {model["run"]}')
        require(manifest['contract_sha256'] == item['contract_sha256'], 'Wrong training contract')
        require(manifest['registration_sha256'] == reference['registration_sha256'], 'Wrong cohort registration')
        require(manifest['model_sha256'] == metadata['sha256'] == model['sha256'], 'Wrong final model')
        require(metadata['contract_sha256'] == item['contract_sha256'], 'Wrong milestone contract')
        require(contract['settings']['seed'] == model['learner_seed'] and
                contract['settings']['condition'] == model['condition'], 'Model identity differs from training settings')
        for key in ('source_hashes','version2_source_hashes','replication_source_hashes','config','python','packages'):
            require(manifest[key] == contract[key] == registration[key], 'Training provenance differs: '+key)
    return registration


def labels(contract, mode):
    return [0] if mode == 'DD' else contract['sample_seeds']


def expected_keys(contract, models):
    return {(m['condition'], m['learner_seed'], m['transitions'], b['seed'], b['motif'], v, mode, label)
            for m in models for b in contract['layouts'] for v in contract['views']
            for mode in contract['modes'] for label in labels(contract, mode)}


def validate_rows(rows, contract, models):
    wanted = expected_keys(contract, models)
    require(len(rows) == len(wanted) and {row_key(r) for r in rows} == wanted,
            'Missing, duplicate or unexpected evaluation units')
    panel = {(b['seed'],b['motif']) for b in contract['monitor_panel']}
    for row in rows:
        d = row['destruction']
        require(math.isfinite(d) and 0 <= d <= 1, 'Invalid destruction value')
        require(bool(row['full_clear']) == (d == 1) and row['zero_destruction'] == float(d == 0), 'Inconsistent outcome flags')
        require(row['monitor_layout'] == ((row['base_seed'], row['motif']) in panel), 'Incorrect monitor membership')
        seed = int.from_bytes(hashlib.sha256(
            f'v2-dev:{row["sample_seed"]}:{row["motif"]}:{row["base_seed"]}:{row["view"]}'.encode()).digest()[:4], 'big')
        require(row['acting_seed'] == seed, 'Incorrect action seed')
        require(0 <= row['spent'] <= 200 and 0 <= row['tick'] <= 300 and 0 < row['decisions'] <= 70,
                'Invalid episode accounting')


def read_baselines(folder, contract):
    rows = []
    with (Path(folder)/'combined_episodes.csv').open(newline='') as handle:
        for raw in csv.DictReader(handle):
            if raw['mode'] not in contract['modes']:
                continue
            row = dict(raw)
            for name in ('learner_seed','transitions','base_seed','view','sample_seed','acting_seed','spent','tick','decisions'):
                row[name] = int(row[name])
            for name in ('destruction','episode_return','zero_destruction'):
                row[name] = float(row[name])
            for name in ('full_clear','town_hall','monitor_layout'):
                require(row[name] in ('True','False'), 'Invalid boolean in baseline CSV')
                row[name] = row[name] == 'True'
            rows.append(row)
    validate_rows(rows, contract, contract['screen_models'])
    return rows


def prepare():
    protocol = read(PACKAGE/'protocol.json')
    reference = read(PACKAGE/'replication_reference.json')
    baseline_reference = read(PACKAGE/'baseline_reference.json')
    baseline_folder = ROOT/baseline_reference['directory']
    verify_files(baseline_folder, baseline_reference['files'])
    phase = read(baseline_folder/'evaluation_contract.json')
    require(read(baseline_folder/'complete.json')['status'] == 'completed', 'Phase diagnostic not complete')
    require(F.package_hashes() == phase['package_hashes'], 'Phase inference code changed')
    original, pool = E.prepare()
    require(original['scientific_runtime'] == phase['scientific_runtime'], 'Scientific evaluation runtime changed')
    require(original['layouts'] == phase['layouts'] and original['pool_sha256'] == phase['pool_sha256'], 'Development pool changed')
    registration = verify_replicas(ROOT, reference)
    for key in ('source_hashes','config','python','packages'):
        require(registration[key] == original['scientific_runtime'][key], 'Replication runtime differs: '+key)
    planned = registration['plan']['post_training_development']
    require(planned['modes'] == protocol['modes'] == ['DS','SS','DD'], 'Modes differ from registered follow-on plan')
    require(planned['views'] == protocol['views'] == phase['views'], 'View design changed')
    require(planned['sample_seeds'] == protocol['sample_seeds'] == phase['sample_seeds'], 'Acting labels changed')
    screen = [m for m in original['models'] if m['transitions'] == protocol['transitions']]
    require(screen == phase['models'], 'Screen checkpoints changed')
    models = screen + reference['models']
    require(len(models) == 20 and {(m['condition'],m['learner_seed']) for m in models} ==
            {(c,s) for c in ('P','PA') for s in range(10)}, 'Expected exactly twenty final checkpoints')
    require(all(m['transitions'] == 4194304 for m in models), 'Only final checkpoints are eligible')
    contract = {'schema': 1, 'kind': protocol['kind'], 'scope': protocol['scope'], 'protocol': protocol,
        'package_hashes': package_hashes(), 'replication_reference': reference, 'baseline_reference': baseline_reference,
        'scientific_runtime': original['scientific_runtime'], 'phase_code_hashes': phase['package_hashes'],
        'development_code_hashes': original['package_hashes'], 'pool_sha256': original['pool_sha256'],
        'models': models, 'new_models': reference['models'], 'screen_models': screen,
        'layouts': original['layouts'], 'monitor_panel': original['monitor_panel'],
        'modes': protocol['modes'], 'views': protocol['views'], 'sample_seeds': protocol['sample_seeds'],
        'device': 'cpu', 'torch_threads': 1, 'expected_episodes': protocol['expected_new_episodes']}
    require(len(pool) == 156 and len(contract['monitor_panel']) == 12, 'Unexpected development design')
    require(len(expected_keys(contract, contract['new_models'])) == contract['expected_episodes'] == 122304,
            'Unexpected new evaluation size')
    baselines = read_baselines(baseline_folder, contract)
    require(len(baselines) == protocol['expected_reused_episodes'] == 52416, 'Unexpected reused evaluation size')
    require(len(baselines)+contract['expected_episodes'] == protocol['expected_combined_episodes'] == 174720,
            'Unexpected combined size')
    return contract, pool, baselines


def units(contract, pool):
    for target in contract['new_models']:
        for mode in contract['modes']:
            for base in pool:
                for view in contract['views']:
                    for label in labels(contract, mode):
                        yield target, base, view, mode, label


def csv_bytes(rows, fields=None):
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text, fieldnames=fields or sorted(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    return text.getvalue().encode()


def export(journal, contract, pool, baselines, output):
    from .analysis import summaries
    pairs = list(journal.rows())
    wanted = {E.unit_key(*unit) for unit in units(contract,pool)}
    require(len(pairs) == contract['expected_episodes'] and {k for k,_ in pairs} == wanted,
            'Incomplete or unexpected journal units')
    fresh = [r for _,r in pairs]
    validate_rows(fresh, contract, contract['new_models'])
    for key, row in pairs:
        require(key == ':'.join(str(row[k]) for k in KEY_FIELDS), 'Journal key and payload disagree')
    combined = sorted([*fresh, *baselines], key=row_key)
    validate_rows(combined, contract, contract['models'])
    def annotate(row):
        return {**row, 'cohort': 'screening' if row['learner_seed'] < 3 else 'replication'}
    files = {'new_episodes.csv': csv_bytes([annotate(r) for r in fresh]),
             'combined_episodes.csv': csv_bytes([annotate(r) for r in combined])}
    for name, rows in summaries(combined, contract['protocol']['intervals']).items():
        files[name] = csv_bytes(rows)
    for name, payload in files.items():
        publish_bytes(output/name, payload)
    publish_json(output/'complete.json', {'status': 'completed', 'new_episodes': len(fresh),
        'reused_baseline_episodes': len(baselines), 'combined_episodes': len(combined), 'scope': contract['scope'],
        'contract_sha256': E.sha256(output/'evaluation_contract.json'),
        'outputs': {name:E.sha256(output/name) for name in files}})


def pack(output=OUTPUT):
    output = Path(output)
    with run_lock(output, '.progress.sqlite3.lock'):
        receipt = read(output/'complete.json')
        require(receipt['status'] == 'completed', 'Evaluation is not complete')
        required = {**receipt['outputs'], 'evaluation_contract.json': receipt['contract_sha256']}
        verify_files(output, required)
        names = [*required, 'complete.json', 'progress.json', 'attempts.jsonl']
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        target = output.parent/f'Base_Siege_V2_Cohort_Dev_Results_{stamp}.zip'
        with zipfile.ZipFile(target, 'x', zipfile.ZIP_DEFLATED) as archive:
            for name in names:
                archive.write(output/name, 'version2-cohort-development/'+name)
        print(target)
        return target


def execute(contract, pool, baselines, output=OUTPUT, root=ROOT, max_new=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stop = {'requested': False}
    completed = new_count = 0
    started = perf_counter()
    target_name, status = 'initializing', 'running'
    attempts = output/'attempts.jsonl'
    panel = {(b['seed'],b['motif']) for b in contract['monitor_panel']}
    path = output/'progress.sqlite3'
    with open_journal(path, contract, resume=path.exists()) as journal:
        publish_json(output/'evaluation_contract.json', contract)
        if (output/'STOP_REQUESTED.json').exists():
            handled = output/'stop_requests'/f'handled-{uuid.uuid4().hex}.json'
            handled.parent.mkdir(exist_ok=True)
            os.replace(output/'STOP_REQUESTED.json', handled)
        completed = journal.count()
        require(completed <= contract['expected_episodes'], 'Unexpected journal size')
        def progress(state):
            elapsed = perf_counter()-started
            speed = elapsed/new_count if new_count >= 100 else None
            value = {'status': state, 'completed_episodes': completed, 'expected_episodes': contract['expected_episodes'],
                     'current_target': target_name, 'updated_utc': E.now(), 'new_episodes_this_attempt': new_count,
                     'attempt_wall_seconds': elapsed, 'rough_remaining_seconds':
                         (contract['expected_episodes']-completed)*speed if speed is not None else None}
            atomic_json(output/'progress.json', value)
            return value
        with attempts.open('a') as handle:
            handle.write(json.dumps({'event':'started','utc':E.now(),'already_committed':completed})+'\n')
        try:
            loaded, loaded_key = None, None
            with stop_signals(stop):
                for target, base, view, mode, label in units(contract,pool):
                    target_name = f'{target["run"]} / {mode}'
                    key = E.unit_key(target,base,view,mode,label)
                    if journal.get(key) is not None:
                        continue
                    if stop['requested'] or (output/'STOP_REQUESTED.json').exists() or (max_new is not None and new_count >= max_new):
                        status = 'paused'
                        value = progress(status)
                        print(json.dumps(value,indent=2),flush=True)
                        return value
                    model_key = target['condition'], target['learner_seed'], target['transitions']
                    if loaded_key != model_key:
                        loaded = E.load_target(target,root)
                        loaded_key = model_key
                    row = evaluate_one(target,loaded,base,view,mode,label,panel)
                    journal.put(key,row)
                    completed += 1; new_count += 1
                    if new_count == 1 or new_count % 100 == 0:
                        value = progress('running'); eta = value['rough_remaining_seconds']
                        estimate = f'rough remaining {eta/3600:.2f} hours' if eta is not None else 'estimating speed'
                        print(f'{completed:,}/{contract["expected_episodes"]:,} new attacks; {target_name}; {estimate}',flush=True)
                export(journal,contract,pool,baselines,output)
                status = 'completed'
                value = progress(status)
                print(json.dumps(value,indent=2),flush=True)
                return value
        except BaseException as error:
            status = 'failed'
            progress(status)
            atomic_json(output/'last_error.json',{'utc':E.now(),'error':f'{type(error).__name__}: {error}'})
            raise
        finally:
            with attempts.open('a') as handle:
                handle.write(json.dumps({'event':'finished','utc':E.now(),'status':status,'new_episodes':new_count,
                    'completed_episodes':completed,'wall_seconds':perf_counter()-started})+'\n')


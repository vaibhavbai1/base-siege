from collections import defaultdict
from datetime import datetime, timezone
import csv
import io
import json
import os
from pathlib import Path
import random
from statistics import mean, stdev
from time import perf_counter
import uuid
import zipfile

import numpy as np
import torch
from env import config as C
from env.siege_env import SiegeEnv
from env.symmetry import SymmetryWrapper
from experiments.journal import open_journal
from experiments.rollout import play_episode
from experiments.storage import atomic_json, publish_json, publish_bytes
from train.train_ppo import stop_signals
from version2.evaluation import episode_seed
from v2_development import runner as E

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
BASELINES = ROOT/'artifacts/version2-full-development'
OUTPUT = ROOT/'artifacts/version2-phase-evaluation'
# First letter = purchasing; second = deployment. True means deterministic.
MODES = {'SS': (False, False), 'DD': (True, True), 'DS': (True, False), 'SD': (False, True)}
NEW_MODES = ('DS', 'SD')
KEY_FIELDS = ('condition', 'learner_seed', 'transitions', 'base_seed', 'motif', 'view', 'mode', 'sample_seed')


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def package_hashes():
    files = sorted(PACKAGE.rglob('*.py')) + [PACKAGE/'protocol.json', PACKAGE/'baseline_reference.json']
    return {str(p.relative_to(PACKAGE)): E.sha256(p) for p in files}


def row_key(row):
    return tuple(row[k] for k in KEY_FIELDS)


def read_baselines(folder, contract):
    """Keep one DD result per layout/view; do not duplicate it for three labels."""
    wanted = {(m['condition'], m['learner_seed'], m['transitions']) for m in contract['models']}
    result = []
    with (Path(folder)/'episodes.csv').open(newline='') as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for name in ('learner_seed', 'transitions', 'base_seed', 'view', 'sample_seed', 'acting_seed', 'spent', 'tick', 'decisions'):
                row[name] = int(row[name])
            if (row['condition'], row['learner_seed'], row['transitions']) not in wanted:
                continue
            for name in ('destruction', 'episode_return', 'zero_destruction'):
                row[name] = float(row[name])
            for name in ('full_clear', 'town_hall', 'monitor_layout'):
                require(row[name] in ('True', 'False'), 'Invalid boolean in baseline CSV')
                row[name] = row[name] == 'True'
            require(row['mode'] in ('sampled', 'deterministic'), 'Unexpected baseline mode')
            row['mode'] = 'SS' if row['mode'] == 'sampled' else 'DD'
            result.append(row)
    expected = set()
    for model in contract['models']:
        for base in contract['layouts']:
            for view in contract['views']:
                for mode, labels in [('SS', contract['sample_seeds']), ('DD', [0])]:
                    for label in labels:
                        expected.add((model['condition'], model['learner_seed'], model['transitions'],
                                      base['seed'], base['motif'], view, mode, label))
    require(len(result) == len(expected) and {row_key(r) for r in result} == expected,
            'Missing or duplicate baseline units')
    return result


def prepare():
    protocol = read(PACKAGE/'protocol.json')
    reference = read(PACKAGE/'baseline_reference.json')
    for name, digest in reference['files'].items():
        path = BASELINES/name
        require(path.is_file(), f'Missing completed development file: {path}')
        require(E.sha256(path) == digest, f'Baseline file changed: {name}. Do not edit reference hashes.')
    original, pool = E.prepare()
    require(original == read(BASELINES/'evaluation_contract.json'),
            'Completed development inputs, code or software environment changed')
    require(read(BASELINES/'complete.json')['status'] == 'completed', 'Full development evaluation is incomplete')
    models = [m for m in original['models'] if m['transitions'] == protocol['transitions']]
    require({(m['condition'], m['learner_seed']) for m in models} == {(c,s) for c in ('P','PA') for s in range(3)},
            'Expected six final V2 checkpoints')
    contract = {'schema': 1, 'kind': 'version2-phase-evaluation', 'scope': protocol['scope'],
                'protocol': protocol, 'package_hashes': package_hashes(), 'baseline_reference': reference,
                'baseline_contract_sha256': E.sha256(BASELINES/'evaluation_contract.json'),
                'scientific_runtime': original['scientific_runtime'], 'pool_sha256': original['pool_sha256'],
                'models': models, 'layouts': original['layouts'], 'monitor_panel': original['monitor_panel'],
                'views': original['views'], 'sample_seeds': original['sample_seeds'], 'new_modes': list(NEW_MODES),
                'device': 'cpu', 'torch_threads': 1,
                'expected_episodes': len(models)*len(pool)*len(original['views'])*len(original['sample_seeds'])*len(NEW_MODES)}
    require(contract['expected_episodes'] == 44928, 'Unexpected mixed-mode design size')
    baseline_rows = read_baselines(BASELINES, contract)
    require(len(baseline_rows) == 29952, 'Expected 29,952 reused baseline attacks')
    return contract, pool, baseline_rows


class PhasePolicy:
    def __init__(self, policy, mode):
        require(mode in MODES, 'Unknown phase mode')
        self.policy, self.mode = policy, mode

    def act(self, obs, mask):
        phase = float(obs['vector'][6])
        require(phase in (0.0, 1.0), 'Unexpected phase observation')
        shopping = phase == 0.0
        require(bool(mask[C.ACTION_FINISH_SHOPPING]) == shopping, 'Phase and legal actions disagree')
        deterministic = MODES[self.mode][0 if shopping else 1]
        action, _ = self.policy.predict(obs, action_masks=mask, deterministic=deterministic)
        action = int(action)
        require(0 <= action < len(mask) and bool(mask[action]), 'Policy selected an illegal action')
        return action


def episode(model, base, view, mode, label):
    seed = episode_seed(label, base, view)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = SymmetryWrapper(SiegeEnv(motifs=C.DEV_MOTIFS), transform=view)
    try:
        return play_episode(env, PhasePolicy(model.policy, mode), options={'spec': base})
    finally:
        env.close()


def evaluate_one(target, model, base, view, mode, label, panel):
    info = episode(model, base, view, mode, label)['summary']
    return {'condition': target['condition'], 'learner_seed': target['learner_seed'], 'transitions': target['transitions'],
            'base_seed': base.seed, 'motif': base.motif, 'view': view, 'mode': mode, 'sample_seed': label,
            'acting_seed': episode_seed(label, base, view), 'monitor_layout': (base.seed,base.motif) in panel,
            **{k: info[k] for k in E.METRICS if k != 'zero_destruction'},
            'zero_destruction': float(info['destruction'] == 0), 'finish_reason': info['finish_reason']}


def units(contract, pool):
    for target in contract['models']:
        for mode in contract['new_modes']:
            for base in pool:
                for view in contract['views']:
                    for label in contract['sample_seeds']:
                        yield target, base, view, mode, label


def csv_bytes(rows, fields=None):
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text, fieldnames=fields or sorted(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    return text.getvalue().encode()


def summaries(rows):
    models, conditions, _ = E.summarize(rows)
    grouped = defaultdict(list)
    for row in rows:
        for scope in ['all_dev'] + ([] if row['monitor_layout'] else ['outside_monitor_panel']):
            grouped[scope, row['condition'], row['learner_seed'], row['transitions'], row['mode']].append(row)
    for summary in models:
        group = grouped[tuple(summary[k] for k in ('scope','condition','learner_seed','transitions','mode'))]
        summary['below_half_percent'] = 100*mean(r['destruction'] < .5 for r in group)
        summary['purchasing'] = 'deterministic' if MODES[summary['mode']][0] else 'sampled'
        summary['deployment'] = 'deterministic' if MODES[summary['mode']][1] else 'sampled'
        summary['origin'] = 'new_mixed_evaluation' if summary['mode'] in NEW_MODES else 'reused_full_development'
    for summary in conditions:
        group = [r for r in models if all(r[k] == summary[k] for k in ('scope','condition','transitions','mode'))]
        summary['below_half_percent'] = mean(r['below_half_percent'] for r in group)
        summary['purchasing'], summary['deployment'], summary['origin'] = (group[0][k] for k in ('purchasing','deployment','origin'))
    lookup = {(r['scope'],r['condition'],r['learner_seed'],r['transitions'],r['mode']): r for r in models}
    comparisons = []
    for row in models:
        if row['mode'] not in NEW_MODES:
            continue
        for reference in ('SS', 'DD'):
            base = lookup[row['scope'],row['condition'],row['learner_seed'],row['transitions'],reference]
            entry = {k: row[k] for k in ('scope','condition','learner_seed','transitions','mode')}
            entry.update(reference_mode=reference, destruction_percent=row['destruction_percent'],
                         reference_destruction_percent=base['destruction_percent'])
            for metric in ('destruction_percent','full_clear_percent','town_hall_percent','zero_destruction_percent','below_half_percent'):
                entry['change_'+metric+'_points'] = row[metric]-base[metric]
            comparisons.append(entry)
    groups = defaultdict(list)
    for row in comparisons:
        groups[row['scope'],row['condition'],row['transitions'],row['mode'],row['reference_mode']].append(row)
    condition_comparisons = []
    for group_key, group in sorted(groups.items()):
        entry = dict(zip(('scope','condition','transitions','mode','reference_mode'), group_key))
        changes = [r['change_destruction_percent_points'] for r in group]
        entry.update(learner_runs=len(group), learners_with_higher_destruction=sum(v > 0 for v in changes),
                     destruction_change_sd_across_learners=stdev(changes) if len(changes)>1 else 0)
        for metric in ('destruction_percent','full_clear_percent','town_hall_percent','zero_destruction_percent','below_half_percent'):
            entry['mean_change_'+metric+'_points'] = mean(r['change_'+metric+'_points'] for r in group)
        condition_comparisons.append(entry)
    return models, conditions, comparisons, condition_comparisons


def export(journal, contract, pool, baselines, output):
    expected = {E.unit_key(*unit) for unit in units(contract,pool)}
    pairs = list(journal.rows())
    require(len(pairs) == contract['expected_episodes'] and {k for k,_ in pairs} == expected, 'Incomplete or unexpected mixed-mode units')
    mixed = [r for _,r in pairs]
    combined = sorted([*mixed, *baselines], key=row_key)
    require(len({row_key(r) for r in combined}) == len(combined), 'Duplicate combined units')
    model_rows, condition_rows, comparisons, condition_comparisons = summaries(combined)
    files = {'mixed_episodes.csv': csv_bytes(mixed), 'combined_episodes.csv': csv_bytes(combined),
             'model_summary.csv': csv_bytes(model_rows), 'condition_summary.csv': csv_bytes(condition_rows),
             'comparisons_by_learner.csv': csv_bytes(comparisons), 'condition_comparisons.csv': csv_bytes(condition_comparisons)}
    for name, data in files.items():
        publish_bytes(output/name, data)
    publish_json(output/'complete.json', {'status': 'completed', 'new_episodes': len(mixed), 'reused_baseline_episodes': len(baselines),
                 'combined_episodes': len(combined), 'scope': contract['scope'],
                 'contract_sha256': E.sha256(output/'evaluation_contract.json'),
                 'outputs': {name:E.sha256(output/name) for name in files}})


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


def pack(output=OUTPUT):
    output = Path(output)
    receipt = read(output/'complete.json')
    require(receipt['status'] == 'completed', 'Evaluation is not complete')
    required = {**receipt['outputs'], 'evaluation_contract.json':receipt['contract_sha256']}
    for name,digest in required.items():
        require(E.sha256(output/name) == digest, 'Result hash mismatch: '+name)
    names = [*required, 'complete.json', 'progress.json', 'attempts.jsonl']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    target = output.parent/f'Base_Siege_V2_Phase_Results_{stamp}.zip'
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(output/name, 'version2-phase-evaluation/'+name)
    print(target)
    return target

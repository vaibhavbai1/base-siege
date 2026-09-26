#!/usr/bin/env python3
"""Base Siege: inference-only, resumable action-selection diagnostic.

Place this file in the project root, next to pyproject.toml. Run:
    uv run python diagnose_policy_sampling.py
After interruption, use the identical command with --resume.

Defaults: all P/PA seeds 0..9, 12 layouts per train/dev split, views 0/1/4/5,
one deterministic and three sampled attacks per model/layout/view: 7,680 games.
This is exploratory evidence, not a replacement for registered evaluation.
No training, model updates, protected final-pool access, or source edits occur.
Outputs go only to artifacts/policy-sampling-v1 (or --out).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
from statistics import mean
import sys
from time import perf_counter
from types import SimpleNamespace

VERSION = 1
VIEWS = (0, 1, 4, 5)
ACTION_SEEDS = (0, 1, 2)
FIELDS = ('destruction', 'town_hall', 'full_clear', 'return', 'spent', 'ticks',
          'zero_destruction', 'empty_army', 'army_size', 'purchase_concentration',
          'deploy_unique_tiles', 'deployment_concentration', 'wait_actions')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_api(root):
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    from env import config as C
    from env.siege_env import SiegeEnv
    from env.symmetry import SymmetryWrapper
    from experiments.design import development_panel
    from experiments.journal import open_journal
    from experiments.pools import load_pool
    from experiments.provenance import runtime_manifest
    from experiments.rollout import play_episode
    from experiments.storage import atomic_json, json_bytes, publish_bytes, publish_json
    from train.evaluate import inspect_run, load_run, validate_matched_runs
    from scipy.stats import t
    return SimpleNamespace(**locals())


def select_train_panel(pool, motifs, count=4):
    """Score-independent hash ordering, stratified by structure; exactly 12 bases."""
    result = []
    for motif in motifs:
        candidates = [b for b in pool if b.motif == motif]
        require(len(candidates) >= count, f'Too few training bases for {motif}')
        candidates.sort(key=lambda b: hashlib.sha256(
            f'base-siege-sampling-panel-v1:{b.motif}:{b.seed}'.encode()).hexdigest())
        result.extend(candidates[:count])
    return result


def reference_key(row):
    return (row['condition'], int(row['learner_seed']), int(row['base_seed']),
            row['motif'], int(row['view']))


def validate_reference(directory, manifests, api):
    receipt = read_json(directory / 'evaluation.json')
    require(receipt['status'] == 'completed' and receipt['mode'] == 'development',
            'Reference must be the completed development evaluation.')
    require(digest(directory / 'episodes.csv') == receipt['episodes_sha256'],
            'Reference episode CSV checksum mismatch.')
    require(receipt['source_hashes'] == manifests[0]['source_hashes'],
            'Reference evaluation used different source code.')
    with (directory / 'episodes.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == receipt['rows'], 'Reference episode count mismatch.')
    records = {}
    for row in rows:
        if row['condition'] not in ('P', 'PA'):
            continue
        key = reference_key(row)
        require(key not in records, 'Duplicate learner episode in reference evaluation.')
        records[key] = row
    return receipt, records


def prepare(args, api):
    root = args.project_root.resolve()
    runs = [root / args.runs_root / f'{c}-{s}' for s in range(10) for c in ('P', 'PA')]
    print('Checking 20 models, frozen inputs, and the existing evaluation...', flush=True)
    manifests = [api.inspect_run(p) for p in runs]
    require(api.validate_matched_runs(manifests) == list(range(10)), 'Expected all ten matched learner seeds.')
    current = json.loads(json.dumps(api.runtime_manifest()))
    pool_paths = {'train': root / args.train_pool, 'dev': root / args.dev_pool}
    pool_hashes = {k: digest(p) for k, p in pool_paths.items()}
    registrations = set()
    for run, m in zip(runs, manifests):
        require(m['settings']['mode'] == 'full' and m['actual_steps'] == 262144,
                f'{run.name}: expected completed full-budget model.')
        for key in ('source_hashes', 'config', 'python', 'packages'):
            require(m[key] == current[key], f'{run.name}: {key} changed. Restore the original environment.')
        for split, h in pool_hashes.items():
            require(m['pools'][split]['sha256'] == h, f'{run.name}: {split} pool differs from training.')
        registration_hash = m.get('registration_sha256')
        require(bool(registration_hash), f'{run.name}: registration identity missing.')
        registrations.add(registration_hash)
        c = read_json(run / 'contract.json')
        require(digest(run / 'contract.json') == m['contract_sha256'], f'{run.name}: contract changed.')
        require(c['settings'] == m['settings'], f'{run.name}: inconsistent settings.')
    require(len(registrations) == 1, 'All learners must share the original registration.')
    receipt, reference = validate_reference(root / args.reference_evaluation, manifests, api)
    require(receipt['identity']['pool_sha256'] == pool_hashes['dev'], 'Reference uses another development pool.')
    receipt_models = {(p['condition'], p['learner_seed']): p for p in receipt['policies'] if p['condition'] in ('P','PA')}
    require(len(receipt_models) == 20, 'Reference must contain all 20 learners.')
    for run, m in zip(runs, manifests):
        p = receipt_models.get((m['settings']['condition'], m['settings']['seed']))
        require(p is not None and p['model_sha256'] == m['model_sha256'] and
                p['manifest_sha256'] == digest(run / 'manifest.json'), f'{run.name}: reference model differs.')
    pools = {k: api.load_pool(path, required_split=k) for k, path in pool_paths.items()}
    panels = {'train': select_train_panel(pools['train'], api.C.TRAIN_MOTIFS),
              'dev': api.development_panel(pools['dev'])}
    require(all(len(v) == 12 for v in panels.values()), 'Expected 12 bases in each panel.')
    # Confirm every required deterministic reference unit exists before starting.
    for m in manifests:
        for b in panels['dev']:
            for view in VIEWS:
                require((m['settings']['condition'], m['settings']['seed'], b.seed, b.motif, view) in reference,
                        'Reference is missing a required development game.')
    contract = {
        'schema_version': VERSION, 'kind': 'policy-sampling-diagnostic',
        'evidence_scope': 'exploratory diagnostic; not registered final evaluation',
        'script_sha256': digest(__file__), 'device': 'cpu', 'torch_threads': 1,
        'runtime': {k: current[k] for k in ('source_hashes','config','python','packages')},
        'pool_sha256': pool_hashes, 'registration_sha256': next(iter(registrations)),
        'reference_receipt_sha256': digest(root / args.reference_evaluation / 'evaluation.json'),
        'reference_episodes_sha256': receipt['episodes_sha256'],
        'panels': {k: [json.loads(b.to_json()) for b in v] for k,v in panels.items()},
        'panel_selection': {'train': '4 per motif by fixed SHA256 ordering', 'dev': 'existing development_panel'},
        'views': list(VIEWS), 'sampled_action_seeds': list(ACTION_SEEDS),
        'rng_rule': 'SHA256 of base seed, motif, view, action seed; independent reset per episode',
        'models': [{'condition':m['settings']['condition'], 'learner_seed':m['settings']['seed'],
                    'model_sha256':m['model_sha256'], 'manifest_sha256':digest(p/'manifest.json')}
                   for p,m in zip(runs,manifests)],
        'expected_episodes': len(runs)*sum(len(v) for v in panels.values())*len(VIEWS)*(1+len(ACTION_SEEDS)),
    }
    return runs, manifests, panels, reference, contract


def episode_seed(base, view, action_seed):
    text = f'base-siege-action-v1:{base.seed}:{base.motif}:{view}:{action_seed}'
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'big') % (2**31)


class ActingPolicy:
    def __init__(self, model, deterministic):
        self.model, self.deterministic = model, deterministic

    def act(self, observation, mask):
        action, _ = self.model.predict(observation, action_masks=mask, deterministic=self.deterministic)
        return int(action)


def action_statistics(actions, C):
    buys = Counter(a for a in actions if C.ACTION_BUY_FIRST <= a <= C.ACTION_BUY_LAST)
    deployments = [a-C.ACTION_DEPLOY_FIRST for a in actions if C.ACTION_DEPLOY_FIRST <= a <= C.ACTION_DEPLOY_LAST]
    tiles = Counter(a % C.N_DEPLOY_TILES for a in deployments)
    bought = sum(buys.values())
    result = {'army_size': bought, 'empty_army': int(bought == 0),
              'purchase_concentration': max(buys.values(), default=0)/max(1,bought),
              'deploy_unique_tiles':len(tiles),
              'deployment_concentration': max(tiles.values(),default=0)/max(1,len(deployments)),
              'wait_actions':sum(a == C.ACTION_WAIT for a in actions)}
    for i, troop in enumerate(C.TROOP_TYPES):
        result['bought_'+troop] = buys[i+C.ACTION_BUY_FIRST]
        result['deployed_'+troop] = sum(a//C.N_DEPLOY_TILES == i for a in deployments)
    return result


def run_episode(api, model, base, view, mode, action_seed):
    seed = episode_seed(base, view, action_seed)
    random.seed(seed)
    api.np.random.seed(seed)
    api.torch.manual_seed(seed)
    env = api.SymmetryWrapper(api.SiegeEnv(motifs=api.C.TRAIN_MOTIFS, fixed_army=False), transform=view)
    try:
        replay = api.play_episode(env, ActingPolicy(model, mode == 'deterministic'), seed=0, options={'spec':base})
    finally:
        env.close()
    s = replay['summary']
    return {'base_seed':base.seed, 'motif':base.motif, 'view':view, 'mode':mode,
            'action_seed':action_seed, 'rng_seed':seed, 'destruction':s['destruction'],
            'town_hall':int(s['town_hall']), 'full_clear':int(s['full_clear']),
            'return':s['episode_return'], 'spent':s['spent'], 'ticks':s['tick'],
            'finish_reason':s['finish_reason'], 'zero_destruction':int(s['destruction']==0),
            **action_statistics(replay['actions'],api.C)}


def compare_reference(row, reference):
    ref = reference[reference_key(row)]
    for metric in ('destruction','town_hall','full_clear','return','spent','ticks'):
        require(math.isclose(float(row[metric]),float(ref[metric]),rel_tol=0,abs_tol=1e-9),
                f'Deterministic reference mismatch for {reference_key(row)}, {metric}: '
                f"new={row[metric]}, previous={ref[metric]}. Stop and investigate; do not overwrite reference results.")
    require(row['finish_reason']==ref['finish_reason'], 'Deterministic finish reason differs from reference.')


def groups_mean(rows, keys, fields=FIELDS):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return [{**dict(zip(keys,key)), 'episodes':len(values),
             **{f:mean(float(r[f]) for r in values) for f in fields}}
            for key,values in sorted(groups.items())]


def summaries(rows, api):
    by_learner = groups_mean(rows, ('condition','learner_seed','split','mode'))
    lookup = {(r['condition'],r['learner_seed'],r['split'],r['mode']):r for r in by_learner}
    effects = []
    for condition in ('P','PA'):
        for effect, subset, left, right in [
            ('sampled_minus_deterministic','train',('train','sampled'),('train','deterministic')),
            ('sampled_minus_deterministic','dev',('dev','sampled'),('dev','deterministic')),
            ('dev_minus_train','deterministic',('dev','deterministic'),('train','deterministic')),
            ('dev_minus_train','sampled',('dev','sampled'),('train','sampled'))]:
            for metric in ('destruction','return','town_hall','full_clear'):
                differences = {str(seed): lookup[condition,seed,*left][metric]-lookup[condition,seed,*right][metric]
                               for seed in range(10)}
                values = api.np.array(list(differences.values()))
                average = float(values.mean())
                half = float(api.t.ppf(.975,9)*values.std(ddof=1)/(10**.5))
                effects.append({'condition':condition,'effect':effect,'subset':subset,'metric':metric,
                                'mean_difference':average,'ci95_low':average-half,'ci95_high':average+half,
                                'n_independent_learners':10,'learner_differences':differences})
    return {'overall':groups_mean(rows,('condition','split','mode')), 'by_learner':by_learner,
            'by_view':groups_mean(rows,('condition','split','mode','view')),
            'by_motif':groups_mean(rows,('condition','split','mode','motif')),
            'effects':effects,
            'limits':['Exploratory fixed-panel diagnostic; retain the registered deterministic evaluation.',
                      'Intervals describe variation across ten learners on these fixed panels, not all possible bases.',
                      'Train/dev differences also reflect which 12 layouts were selected in each split.',
                      'Sampled games are averaged within learner; repeats are not independent trained agents.',
                      'View 0 is reported separately because unaugmented P trained in that coordinate view.',
                      'Concentration is largest troop-type or deployment-tile share; defined as 0 when none occur.']}


def csv_bytes(rows):
    buffer = io.StringIO(newline='')
    fields = sorted({k for row in rows for k in row})
    writer = csv.DictWriter(buffer,fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def evaluate(args, api, runs, manifests, panels, reference, contract):
    output = (args.project_root / args.out).resolve()
    progress_path = output / 'diagnostic.progress.sqlite3'
    if args.resume:
        require(progress_path.is_file(), 'No diagnostic journal to resume. Use a new output folder for a new diagnostic.')
    output.mkdir(parents=True,exist_ok=args.resume)
    expected = contract['expected_episodes']
    modes = [('deterministic',-1)]+[('sampled',s) for s in ACTION_SEEDS]
    with api.open_journal(progress_path,contract,resume=args.resume) as journal:
        api.publish_json(output/'diagnostic_contract.json',contract)
        done = journal.count()
        require(done <= expected, 'Journal has too many episodes.')
        start_done, start_time = done, perf_counter()
        print(f'Diagnostic ready: {done:,}/{expected:,} saved episodes. Inference only; models are not trained.',flush=True)

        def progress(label, status='running'):
            seconds = perf_counter()-start_time
            rate = (done-start_done)/seconds if done>start_done and seconds>0 else None
            eta = (expected-done)/rate if rate else None
            api.atomic_json(output/'progress.json',{'status':status,'completed':done,'total':expected,
                'current':label,'new_episodes_this_session':done-start_done,'session_seconds':seconds,
                'episodes_per_second':rate,'estimated_remaining_seconds':eta})
            eta_text = f'{eta/60:.1f} min' if eta is not None else 'measuring'
            print(f'{done:,}/{expected:,} ({100*done/expected:.1f}%) | {label} | ETA {eta_text}',flush=True)

        label = 'starting'
        progress(label)
        try:
            for index,(run,m) in enumerate(zip(runs,manifests)):
                model = None
                for split,bases in panels.items():
                    for mode,action_seed in modes:
                        label = f'{run.name} {split} {mode} action-seed={action_seed}'
                        for base_index,base in enumerate(bases):
                            for view in VIEWS:
                                key = f'{index:02d}:{split}:{mode}:{action_seed}:{base_index:02d}:{view}'
                                saved = journal.get(key)
                                if saved is not None:
                                    continue
                                if model is None:
                                    model,_ = api.load_run(run)
                                row = run_episode(api,model,base,view,mode,action_seed)
                                row.update(condition=m['settings']['condition'],learner_seed=m['settings']['seed'],split=split,
                                           evidence_scope='exploratory diagnostic')
                                if split=='dev' and mode=='deterministic':
                                    compare_reference(row,reference)
                                journal.put(key,row)
                                done += 1
                                if done % 25 == 0 or done == expected:
                                    progress(label)
                del model
        except BaseException:
            progress(label,status='interrupted')
            raise
        rows = [r for _,r in journal.rows()]
        require(len(rows)==expected, 'Unexpected final episode count.')
        reference_checks = len(runs)*len(panels['dev'])*len(VIEWS)
        require(sum(r['split']=='dev' and r['mode']=='deterministic' for r in rows)==reference_checks,
                'Missing deterministic reference checks.')
        # Recheck even on completed resumes, including all saved deterministic development records.
        for row in rows:
            if row['split']=='dev' and row['mode']=='deterministic':
                compare_reference(row,reference)
        summary = summaries(rows,api)
        api.publish_bytes(output/'diagnostic_episodes.csv',csv_bytes(rows))
        api.publish_bytes(output/'diagnostic_summary.csv',csv_bytes(summary['overall']))
        api.publish_json(output/'diagnostic_summary.json',summary)
        api.publish_json(output/'diagnostic_complete.json',{'status':'completed','episodes':len(rows),
            'evidence_scope':'exploratory diagnostic; not registered final evaluation',
            'contract_sha256':hashlib.sha256(api.json_bytes(contract)).hexdigest(),
            'episodes_sha256':digest(output/'diagnostic_episodes.csv'),
            'deterministic_dev_reference_matches':reference_checks})
        progress('complete',status='completed')
        print('\nMean destruction (not accuracy):',flush=True)
        for row in summary['overall']:
            print(f"{row['condition']:2} {row['split']:5} {row['mode']:13} {row['destruction']*100:6.2f}%  n={row['episodes']}")
        print(f'\nResults: {output}\nSend diagnostic_summary.json, diagnostic_episodes.csv, and diagnostic_complete.json.')


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root',type=Path,default=Path.cwd())
    parser.add_argument('--runs-root',type=Path,default=Path('runs/study-v2'))
    parser.add_argument('--train-pool',type=Path,default=Path('bases/train-v2.json'))
    parser.add_argument('--dev-pool',type=Path,default=Path('bases/dev-v2.json'))
    parser.add_argument('--reference-evaluation',type=Path,default=Path('artifacts/study-dev-v2'))
    parser.add_argument('--out',type=Path,default=Path('artifacts/policy-sampling-v1'))
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    require((args.project_root/'train/evaluate.py').is_file(), 'Run this from the Base Siege project root.')
    require(Path(__file__).resolve().parent not in [args.project_root/p for p in ('env','train','agents','experiments')],
            'Place the diagnostic in the project root, not a registered source folder.')
    api = load_api(args.project_root)
    api.torch.set_num_threads(1)
    api.torch.use_deterministic_algorithms(True)
    prepared = prepare(args,api)
    evaluate(args,api,*prepared)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped. Saved episodes are retained. Re-run with --resume.',file=sys.stderr)
        sys.exit(130)

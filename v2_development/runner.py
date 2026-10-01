"""CPU development evaluation only. No training or final-pool access."""
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

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from agents.heuristic import HeuristicCommander
from env import config as C
from env.siege_env import SiegeEnv
from env.symmetry import SymmetryWrapper
from experiments.design import development_panel
from experiments.journal import open_journal
from experiments.pools import load_pool
from experiments.provenance import sha256, runtime_manifest, source_hashes
from experiments.rollout import play_episode
from experiments.storage import atomic_json, publish_bytes, publish_json
from train.train_ppo import stop_signals
from version2.evaluation import ActingPolicy, episode_seed

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
OUTPUT = ROOT/'artifacts/version2-full-development'
SEEDS = [31001,31002,31003]
VIEWS = list(range(8))
METRICS = ['destruction','full_clear','town_hall','zero_destruction','episode_return','spent','tick','decisions']


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def require(ok,message):
    if not ok:
        raise ValueError(message)


def configure_cpu():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def package_hashes():
    paths = sorted(PACKAGE.rglob('*.py')) + [PACKAGE/'expected.json']
    return {str(p.relative_to(PACKAGE)):sha256(p) for p in paths}


def verify_files(root, expected):
    required = {**expected['source_hashes'], **expected['version2_source_hashes'],
        expected['registration_path']:expected['registration_sha256'],
        expected['dev_pool']['path']:expected['dev_pool']['sha256']}
    for name,entry in expected['runs'].items():
        required[entry['path']+'/manifest.json'] = entry['manifest_sha256']
        required[entry['path']+'/contract.json'] = entry['contract_sha256']
    for model in expected['models']:
        required[model['path']] = model['sha256']
    for relative,wanted in required.items():
        path = root/relative
        require(path.is_file(),f'Missing {relative}. Keep the original completed run folders.')
        require(sha256(path)==wanted,f'Hash mismatch: {relative}. Do not replace the recorded hash or bypass this check.')
    for name,entry in expected['runs'].items():
        manifest = read(root/entry['path']/'manifest.json')
        require(manifest['status']=='completed' and manifest['actual_steps']==4194304,
                f'{name} is not a completed four-million run')
        require(manifest['contract_sha256']==entry['contract_sha256'], f'{name}: wrong contract')
    for model in expected['models']:
        metadata = read((root/model['path']).with_suffix('.json'))
        require(metadata['sha256']==model['sha256'] and metadata['actual_steps']==model['transitions'],
                f'Milestone metadata differs: {model["path"]}')
        require(metadata['contract_sha256']==expected['runs'][model['run']]['contract_sha256'],
                'Milestone belongs to a different training contract')


def prepare():
    configure_cpu()
    expected = read(PACKAGE/'expected.json')
    verify_files(ROOT,expected)
    current = json.loads(json.dumps(runtime_manifest()))
    require(current['source_hashes']==expected['source_hashes'],'Scientific source set changed')
    require(current['config']==expected['config'],'Original simulator configuration changed')
    require(current['python']==expected['python'],'Use recorded Python '+expected['python'])
    for name in ('torch','numpy','gymnasium','stable-baselines3','sb3-contrib'):
        require(current['packages'][name]==expected['packages'][name],f'Restore recorded {name} version')
    pool = sorted(load_pool(ROOT/expected['dev_pool']['path'],required_split='dev'),key=lambda b:(b.seed,b.motif))
    require(len(pool)==156,'Expected all 156 frozen development layouts')
    panel = {(b.seed,b.motif) for b in development_panel(pool)}
    require(len(panel)==12,'Unexpected monitor panel')
    models = expected['models']
    ids = {(m['condition'],m['learner_seed'],m['transitions']) for m in models}
    require(ids=={(c,s,t) for c in ('P','PA') for s in range(3) for t in (262144,1048576,2097152,4194304)},
            'Expected exactly 24 planned model milestones')
    require(len(models)==len(ids),'Duplicate model milestones')
    contract = {'schema':1, 'kind':'version2-full-development',
        'scope':'Development comparison specified after the four-million diagnostic results; not a final test',
        'expected_records_sha256':sha256(PACKAGE/'expected.json'), 'package_hashes':package_hashes(),
        'scientific_runtime':{k:current[k] for k in ('python','packages','platform','config','source_hashes')},
        'device':'cpu','torch_threads':1,'pool_sha256':expected['dev_pool']['sha256'],
        'models':models,'views':VIEWS,'sample_seeds':SEEDS,
        'acting_seed_rule':'version2.evaluation.episode_seed; shared across learners, conditions and budgets',
        'layouts':[{'seed':b.seed,'motif':b.motif} for b in pool],
        'monitor_panel':[{'seed':s,'motif':m} for s,m in sorted(panel)],
        'expected_episodes':len(models)*len(pool)*len(VIEWS)*(len(SEEDS)+1)+len(pool)*len(VIEWS),
        'report_scopes':['all_dev','outside_monitor_panel'], 'baseline':'HeuristicCommander() once per layout/view'}
    return contract,pool


def targets(contract):
    return [*contract['models'], {'run':'heuristic','condition':'heuristic','learner_seed':-1,'transitions':0}]


def units(target,contract,pool):
    modes = [('deterministic',[0])] if target['condition']=='heuristic' else [('sampled',contract['sample_seeds']),('deterministic',[0])]
    for mode,seeds in modes:
        for base in pool:
            for view in contract['views']:
                for label in seeds:
                    yield base,view,mode,label


def unit_key(target,base,view,mode,label):
    return f'{target["condition"]}:{target["learner_seed"]}:{target["transitions"]}:{base.seed}:{base.motif}:{view}:{mode}:{label}'


def evaluate_one(target,policy,base,view,mode,label,panel):
    seed = episode_seed(label,base,view)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    acting = policy if target['condition']=='heuristic' else ActingPolicy(policy.policy,mode=='deterministic')
    env = SymmetryWrapper(SiegeEnv(motifs=C.DEV_MOTIFS),transform=view)
    try:
        info = play_episode(env,acting,options={'spec':base})['summary']
    finally:
        env.close()
    return {'condition':target['condition'],'learner_seed':target['learner_seed'],
        'transitions':target['transitions'],'base_seed':base.seed,'motif':base.motif,'view':view,
        'mode':mode,'sample_seed':label,'acting_seed':seed,'monitor_layout':(base.seed,base.motif) in panel,
        **{k:info[k] for k in METRICS if k!='zero_destruction'},
        'zero_destruction':float(info['destruction']==0),'finish_reason':info['finish_reason']}


def load_target(target,root):
    if target['condition']=='heuristic':
        return HeuristicCommander()
    path = root/target['path']
    require(sha256(path)==target['sha256'],'Model changed after preflight: '+str(path))
    model = MaskablePPO.load(path,device='cpu')
    require(model.num_timesteps==target['transitions'],'Model timestep counter differs from milestone metadata')
    model.policy.set_training_mode(False)
    return model


def csv_bytes(rows,fields=None):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer,fieldnames=fields or list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    return buffer.getvalue().encode()


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        for scope in ['all_dev']+(['outside_monitor_panel'] if not row['monitor_layout'] else []):
            groups[(scope,row['condition'],row['learner_seed'],row['transitions'],row['mode'])].append(row)
    model_rows = []
    for (scope,condition,seed,steps,mode),items in sorted(groups.items()):
        result = {'scope':scope,'condition':condition,'learner_seed':seed,'transitions':steps,'mode':mode,
            'layouts':len({(r['base_seed'],r['motif']) for r in items}),'episodes':len(items)}
        result.update({name+'_percent':100*mean(float(r[name]) for r in items)
                       for name in ('destruction','full_clear','town_hall','zero_destruction')})
        result.update({'mean_'+name:mean(r[name] for r in items) for name in ('episode_return','spent','tick','decisions')})
        model_rows.append(result)
    condition_groups = defaultdict(list)
    for row in model_rows:
        condition_groups[(row['scope'],row['condition'],row['transitions'],row['mode'])].append(row)
    means = []
    for (scope,condition,steps,mode),items in sorted(condition_groups.items()):
        result = {'scope':scope,'condition':condition,'transitions':steps,'mode':mode,
                  'learner_runs':len(items) if condition!='heuristic' else 0,'episodes':sum(r['episodes'] for r in items)}
        for metric in ('destruction_percent','full_clear_percent','town_hall_percent','zero_destruction_percent'):
            result[metric]=mean(r[metric] for r in items)
        result['destruction_sd_across_learners']=stdev(r['destruction_percent'] for r in items) if len(items)>1 else 0
        means.append(result)
    lookup={(r['scope'],r['condition'],r['learner_seed'],r['transitions'],r['mode']):r for r in model_rows}
    comparisons=[]
    for row in model_rows:
        if row['condition']=='heuristic' or row['transitions']==262144:
            continue
        base=lookup.get((row['scope'],row['condition'],row['learner_seed'],262144,row['mode']))
        if base:
            comparisons.append({k:row[k] for k in ('scope','condition','learner_seed','transitions','mode')} |
                {'reference_transitions':262144,'reference_destruction_percent':base['destruction_percent'],
                 'destruction_percent':row['destruction_percent'],
                 'change_percentage_points':row['destruction_percent']-base['destruction_percent']})
    return model_rows,means,comparisons


def export(journal,contract,pool,output):
    expected = {unit_key(t,b,v,m,s) for t in targets(contract) for b,v,m,s in units(t,contract,pool)}
    pairs = list(journal.rows())
    require({k for k,_ in pairs}==expected and len(pairs)==contract['expected_episodes'],'Incomplete or unexpected evaluation units')
    rows = [r for _,r in pairs]
    model_rows,means,comparisons = summarize(rows)
    files = {'episodes.csv':csv_bytes(rows),'model_summary.csv':csv_bytes(model_rows),
             'condition_summary.csv':csv_bytes(means), 'budget_comparison.csv':csv_bytes(comparisons,
                ['scope','condition','learner_seed','transitions','mode','reference_transitions',
                 'reference_destruction_percent','destruction_percent','change_percentage_points'])}
    for name,payload in files.items():
        publish_bytes(output/name,payload)
    publish_json(output/'complete.json',{'status':'completed','episodes':len(rows),
        'contract_sha256':sha256(output/'evaluation_contract.json'),
        'outputs':{name:sha256(output/name) for name in files},'scope':contract['scope']})


def execute(contract,pool,output=OUTPUT,root=ROOT,max_new=None):
    output.mkdir(parents=True,exist_ok=True)
    stop={'requested':False}
    completed=0; new_count=0; started=perf_counter(); current_target='initializing'
    attempts=output/'attempts.jsonl'
    panel={(b['seed'],b['motif']) for b in contract['monitor_panel']}
    progress_path=output/'progress.sqlite3'
    with open_journal(progress_path,contract,resume=progress_path.exists()) as journal:
        publish_json(output/'evaluation_contract.json',contract)
        if (output/'STOP_REQUESTED.json').exists():
            handled=output/'stop_requests'/f'handled-{uuid.uuid4().hex}.json'
            handled.parent.mkdir(exist_ok=True)
            os.replace(output/'STOP_REQUESTED.json',handled)
        completed=journal.count()
        require(completed<=contract['expected_episodes'],'Unexpected journal size')
        def progress(status):
            elapsed=perf_counter()-started
            seconds_per_episode=elapsed/new_count if new_count>=100 else None
            value={'status':status,'completed_episodes':completed,'expected_episodes':contract['expected_episodes'],
                'current_target':current_target,'updated_utc':now(),'new_episodes_this_attempt':new_count,
                'attempt_wall_seconds':elapsed,'rough_remaining_seconds':
                (contract['expected_episodes']-completed)*seconds_per_episode if seconds_per_episode else None}
            atomic_json(output/'progress.json',value)
            return value
        with attempts.open('a') as f:
            f.write(json.dumps({'event':'started','utc':now(),'already_committed':completed})+'\n')
        try:
            with stop_signals(stop):
                for target in targets(contract):
                    current_target=f'{target["run"]} / {target["transitions"]}'
                    policy=None
                    for base,view,mode,label in units(target,contract,pool):
                        if stop['requested'] or (output/'STOP_REQUESTED.json').exists() or (max_new is not None and new_count>=max_new):
                            value=progress('paused')
                            print(json.dumps(value,indent=2),flush=True)
                            return value
                        key=unit_key(target,base,view,mode,label)
                        if journal.get(key) is not None:
                            continue
                        if policy is None:
                            policy=load_target(target,root)
                        row=evaluate_one(target,policy,base,view,mode,label,panel)
                        journal.put(key,row)
                        completed+=1; new_count+=1
                        if new_count==1 or new_count%100==0:
                            value=progress('running')
                            eta=value['rough_remaining_seconds']
                            estimate=f'rough remaining {eta/3600:.2f} hours' if eta is not None else 'estimating speed'
                            print(f'{completed:,}/{contract["expected_episodes"]:,} episodes; {current_target}; {estimate}',flush=True)
                    del policy
                export(journal,contract,pool,output)
                value=progress('completed')
                print(json.dumps(value,indent=2),flush=True)
                return value
        except BaseException as error:
            value=progress('failed')
            atomic_json(output/'last_error.json',{'utc':now(),'error':f'{type(error).__name__}: {error}'})
            raise
        finally:
            with attempts.open('a') as f:
                f.write(json.dumps({'event':'finished','utc':now(),'new_episodes':new_count,
                    'completed_episodes':completed,'wall_seconds':perf_counter()-started})+'\n')

"""Frozen final evaluation. Core simulator, learned weights and training stay unchanged."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import csv
import hashlib
import io
import json
import math
import os
import random
import uuid
import zipfile

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from agents.heuristic import HeuristicCommander
from env import config as C
from env.generator import BaseSpec, generate_base
from env.siege_env import SiegeEnv
from env.symmetry import SymmetryWrapper
from experiments.journal import open_journal
from experiments.pools import validate_base
from experiments.provenance import runtime_manifest, sha256
from experiments.rollout import play_episode
from experiments.storage import atomic_json, publish_json, publish_bytes, run_lock
from train.train_ppo import stop_signals
from v2_phases.runner import PhasePolicy

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
OUTPUT = ROOT/'artifacts/version2-final-evaluation'
KEY_FIELDS = ('condition','learner_seed','transitions','base_seed','motif','view','mode','sample_seed')
METRICS = ('destruction','full_clear','town_hall','episode_return','spent','tick','decisions')


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def package_hashes():
    paths = sorted(PACKAGE.rglob('*.py')) + sorted(PACKAGE.glob('*.json'))
    return {str(p.relative_to(PACKAGE)): sha256(p) for p in paths if p.name != 'PACKAGE_SHA256.json'}


def verify_files(root, mapping):
    for name, digest in mapping.items():
        path = Path(root)/name
        require(path.is_file(), f'Missing required file: {name}')
        require(sha256(path) == digest, f'File changed: {name}. Restore it; do not edit expected hashes.')


def overlap(value, low, high):
    """Inspect saved pool/contract metadata, without generating fresh layouts."""
    if isinstance(value, dict):
        for key, v in value.items():
            if key in ('base_seed','seed') and isinstance(v,int) and low <= v <= high:
                return True
            if key in ('seed_interval','candidate_seed_interval','final_candidate_seeds') and isinstance(v,list) and len(v)==2 and all(isinstance(x,int) for x in v):
                if max(low,v[0]) <= min(high,v[1]): return True
            if overlap(v,low,high): return True
    elif isinstance(value,list):
        return any(overlap(v,low,high) for v in value)
    return False


def freshness_check(root, output, protocol):
    low, high = protocol['candidate_seed_interval']
    require(C.FINAL_SEED_RANGE[0] <= low <= high <= C.FINAL_SEED_RANGE[1], 'Candidate range must stay in reserved final partition')
    require(high < C.FINAL_STUDY_SEED_RANGE[0] or low > C.FINAL_STUDY_SEED_RANGE[1], 'Candidate range overlaps original final study')
    checked = []
    for directory in ('bases','artifacts','experiments'):
        for path in sorted((Path(root)/directory).rglob('*.json')):
            if path.is_relative_to(Path(output)): continue
            name = path.name.lower()
            if directory != 'bases' and not any(k in name for k in ('pool','contract','freeze','registration')): continue
            require(not overlap(read(path),low,high), f'New seed range appears in earlier metadata: {path}. Stop and review prior use.')
            checked.append(str(path.relative_to(root)))
    return checked


def prepare(root=ROOT, output=OUTPUT):
    """Read-only provenance checks. Does not generate or evaluate test layouts."""
    protocol, reference = read(PACKAGE/'protocol.json'), read(PACKAGE/'reference.json')
    verify_files(PACKAGE, read(PACKAGE/'PACKAGE_SHA256.json'))
    verify_files(root, reference['required_files'])
    current = json.loads(json.dumps(runtime_manifest()))
    recorded = reference['scientific_runtime']
    for key in ('python','source_hashes','config'):
        require(current[key] == recorded[key], f'Scientific {key} changed; restore the recorded environment.')
    for name in ('numpy','torch','gymnasium','stable-baselines3','sb3-contrib'):
        require(current['packages'][name] == recorded['packages'][name], f'Restore recorded {name} {recorded["packages"][name]}')
    models = reference['models']
    require(len(models)==20 and {(m['condition'],m['learner_seed']) for m in models} == {(c,s) for c in ('P','PA') for s in range(10)}, 'Expected all twenty final learners')
    for m in models:
        require(m['transitions']==4194304, 'Wrong final training budget')
        item = reference['runs'][m['run']]
        manifest, training = read(Path(root)/item['path']/'manifest.json'), read(Path(root)/item['path']/'contract.json')
        require(manifest['status']=='completed' and manifest['actual_steps']==4194304, 'Incomplete training run: '+m['run'])
        require(manifest['contract_sha256']==item['contract_sha256'], 'Training contract mismatch')
        require(training['settings']['condition']==m['condition'] and training['settings']['seed']==m['learner_seed'], 'Learner identity mismatch')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checked = freshness_check(Path(root), Path(output), protocol)
    contract = {'schema':1,'kind':protocol['kind'],'protocol':protocol,'models':models,
                'scientific_runtime':recorded,'package_hashes':package_hashes(),
                'reference_sha256':sha256(PACKAGE/'reference.json'),
                'device':'cpu','torch_threads':1,'output_relative':'artifacts/version2-final-evaluation'}
    return contract, {'status':'ready','models':20,'candidate_layouts':protocol['candidate_layouts'],
                      'candidate_seed_interval':protocol['candidate_seed_interval'],'mode':'DS',
                      'attacks_per_accepted_layout':488,'accepted_layouts':'not generated yet',
                      'prior_metadata_files_checked':len(checked),'device':'cpu'}


def freeze(output=OUTPUT):
    output = Path(output)
    contract, report = prepare(output=output)
    output.mkdir(parents=True,exist_ok=True)
    with run_lock(output,'.pipeline.lock'):
        target = output/'freeze.json'
        if target.exists():
            require(read(target)['contract']==contract, 'Frozen protocol differs. Do not overwrite the freeze.')
        else:
            require(not list(output.glob('*.sqlite3')) and not (output/'final_pool.json').exists(), 'Results exist without a freeze; stop and review.')
            publish_json(target, {'frozen_utc':now(),'contract':contract,'basis':'V2 development complete; fresh final layouts not yet generated'})
    print(json.dumps({**report,'status':'frozen','freeze_sha256':sha256(target)},indent=2))


def acting_seed(label, base, view):
    value = f'v2-final-ds-v1:{label}:{base.motif}:{base.seed}:{view}'
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4],'big')


def episode(target, model, base, view, label):
    seed = acting_seed(label,base,view)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    policy = HeuristicCommander() if target['condition']=='heuristic' else PhasePolicy(model.policy,'DS')
    env = SymmetryWrapper(SiegeEnv(motifs=C.MOTIFS,allow_final=True),transform=view)
    try:
        return play_episode(env,policy,options={'spec':base})
    finally:
        env.close()


def evaluate_one(target, model, base, view, label):
    info = episode(target,model,base,view,label)['summary']
    return {'condition':target['condition'],'learner_seed':target['learner_seed'],'transitions':target['transitions'],
            'base_seed':base.seed,'motif':base.motif,'view':view,'sample_seed':label,
            'mode':'heuristic' if target['condition']=='heuristic' else 'DS',
            'acting_seed':acting_seed(label,base,view), **{k:info[k] for k in METRICS},
            'zero_destruction':float(info['destruction']==0),'finish_reason':info['finish_reason']}


def load_target(target, root=ROOT):
    if target['condition']=='heuristic':return None
    path=Path(root)/target['path']
    require(sha256(path)==target['sha256'],'Model changed after preflight: '+str(path))
    model=MaskablePPO.load(path,device='cpu')
    require(model.num_timesteps==target['transitions'],'Model timestep counter differs from frozen checkpoint')
    model.policy.set_training_mode(False)
    return model


def targets(contract):
    return [*sorted(contract['models'],key=lambda m:(m['learner_seed'],m['condition'])),
            {'condition':'heuristic','learner_seed':-1,'transitions':0,'run':'heuristic'}]


def units(contract, pool):
    p=contract['protocol']
    for target in targets(contract):
        for base in pool:
            for view in p['views']:
                for label in ([0] if target['condition']=='heuristic' else p['sample_seeds']):
                    yield target,base,view,label


def unit_key(target,base,view,label):
    mode='heuristic' if target['condition']=='heuristic' else 'DS'
    return ':'.join(map(str,(target['condition'],target['learner_seed'],target['transitions'],base.seed,base.motif,view,mode,label)))


def row_key(row):
    return ':'.join(str(row[k]) for k in KEY_FIELDS)


def validate_row(row):
    d=row['destruction']
    require(math.isfinite(d) and 0<=d<=1,'Invalid destruction')
    require(row['full_clear']==(d==1) and row['zero_destruction']==float(d==0),'Inconsistent outcome flags')
    require(0<=row['spent']<=200 and 0<=row['tick']<=300 and 0<row['decisions']<=70,'Invalid episode accounting')
    require(math.isfinite(row['episode_return']),'Invalid return')
    base=type('Identity',(),{'seed':row['base_seed'],'motif':row['motif']})()
    require(row['acting_seed']==acting_seed(row['sample_seed'],base,row['view']),'Wrong acting seed')


def csv_bytes(rows):
    buf=io.StringIO(newline='');writer=csv.DictWriter(buf,fieldnames=sorted(rows[0]))
    writer.writeheader();writer.writerows(rows);return buf.getvalue().encode()


def pool_candidates(contract):
    p=contract['protocol']; low,high=p['candidate_seed_interval']
    return [(seed,motif) for seed in range(low,high+1) for motif in p['motifs']]


def validate_pool_data(data, contract):
    expected=set(pool_candidates(contract));records=data['validation']
    require(len(records)==len(expected) and {(r['seed'],r['motif']) for r in records}==expected,'Pool candidate accounting mismatch')
    accepted={(r['seed'],r['motif']) for r in records if r['accepted']}
    bases=[BaseSpec.from_json(json.dumps(x)) for x in data['bases']]
    require(len(bases)==len(accepted) and {(b.seed,b.motif) for b in bases}==accepted,'Accepted pool mismatch')
    require(data['attempted']==len(expected) and data['accepted']==len(bases),'Wrong pool counts')
    require(all(any(b.motif==m for b in bases) for m in contract['protocol']['motifs']),'No accepted layouts in a motif. Stop; do not add replacement candidates.')
    return sorted(bases,key=lambda b:(b.seed,b.motif))


def export(journal, contract, pool, output, frozen_hash):
    from .analysis import summaries
    pairs=list(journal.rows());wanted={unit_key(*u) for u in units(contract,pool)}
    require(len(pairs)==len(wanted) and {k for k,_ in pairs}==wanted,'Missing or unexpected final attacks')
    rows=[]
    for key,row in pairs:
        require(key==row_key(row),'Journal key and payload disagree');validate_row(row);rows.append(row)
    files={'episodes.csv':csv_bytes(rows)}
    files.update({name:csv_bytes(items) for name,items in summaries(rows,contract['protocol']['intervals']).items()})
    for name,payload in files.items():publish_bytes(output/name,payload)
    protected=['freeze.json','final_pool.json','evaluation_contract.json',*files]
    publish_json(output/'complete.json',{'status':'completed','episodes':len(rows),'accepted_layouts':len(pool),
                 'candidate_layouts':contract['protocol']['candidate_layouts'],'freeze_sha256':frozen_hash,
                 'outputs':{name:sha256(output/name) for name in protected}})


def execute(contract, output=OUTPUT, root=ROOT, max_new=None):
    """Internal runner also supports small development-only software fixtures."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    stop={'requested':False};stage='starting';status='running';completed=expected=0
    fresh=pool_new=0;target_name='';started=perf_counter();eval_started=None
    attempts=output/'attempts.jsonl';attempt_id=uuid.uuid4().hex
    with run_lock(output,'.pipeline.lock'):
        require((output/'freeze.json').is_file(),'Run freeze before accessing final layouts.')
        require(read(output/'freeze.json')['contract']==contract,'Frozen contract changed')
        frozen_hash=sha256(output/'freeze.json')
        stop_path=output/'STOP_REQUESTED.json'
        if stop_path.exists():
            handled=output/'stop_requests'/f'handled-{uuid.uuid4().hex}.json';handled.parent.mkdir(exist_ok=True);os.replace(stop_path,handled)
        def progress(state):
            elapsed=perf_counter()-started
            eta=None
            if eval_started is not None and fresh>=100:eta=(expected-completed)*(perf_counter()-eval_started)/fresh
            value={'status':state,'stage':stage,'completed_units':completed,'expected_units':expected,
                   'current_target':target_name,'new_attacks_this_attempt':fresh,'new_pool_candidates_this_attempt':pool_new,
                   'attempt_wall_seconds':elapsed,'rough_remaining_seconds':eta,'updated_utc':now()}
            atomic_json(output/'progress.json',value);return value
        def should_stop():return stop['requested'] or stop_path.exists()
        def log(value):
            with attempts.open('a') as h:
                h.write(json.dumps({'attempt_id':attempt_id,**value})+'\n');h.flush();os.fsync(h.fileno())
        log({'event':'started','utc':now()})
        try:
            with stop_signals(stop):
                stage='pool_generation';candidates=pool_candidates(contract);expected=len(candidates)
                pool_path=output/'pool.progress.sqlite3'
                binding={'stage':'final_pool','freeze_sha256':frozen_hash,'contract':contract}
                with open_journal(pool_path,binding,resume=pool_path.exists()) as journal:
                    wanted={f'{s}:{m}' for s,m in candidates}
                    require({k for k,_ in journal.rows()} <= wanted,'Unexpected pool journal entry')
                    completed=journal.count();progress('running')
                    for seed,motif in candidates:
                        key=f'{seed}:{motif}'
                        if journal.get(key) is not None:continue
                        if should_stop():status='paused';return progress(status)
                        base=generate_base(seed,motif,allow_final=True)
                        valid,score,reason=validate_base(base,allow_final=True)
                        item={'record':{'seed':seed,'motif':motif,'accepted':valid,'best_heuristic':score,'reason':reason},
                              'base':json.loads(base.to_json()) if valid else None}
                        journal.put(key,item);completed+=1;pool_new+=1
                        if pool_new==1 or pool_new%25==0:
                            progress('running');print(f'Pool: {completed}/{expected} candidates checked',flush=True)
                    items=[journal.get(f'{s}:{m}') for s,m in candidates]
                    pool_data={'schema':1,'freeze_sha256':frozen_hash,'seed_interval':contract['protocol']['candidate_seed_interval'],
                               'motifs':contract['protocol']['motifs'],'filter':contract['protocol']['filter'],
                               'attempted':len(items),'accepted':sum(x['record']['accepted'] for x in items),
                               'validation':[x['record'] for x in items],'bases':[x['base'] for x in items if x['base'] is not None]}
                    pool=validate_pool_data(pool_data,contract);publish_json(output/'final_pool.json',pool_data)
                stage='evaluation';expected=sum(1 for _ in units(contract,pool))
                evaluation={'freeze_sha256':frozen_hash,'pool_sha256':sha256(output/'final_pool.json'),
                            'contract':contract,'expected_episodes':expected}
                publish_json(output/'evaluation_contract.json',evaluation)
                print(f'Pool ready: {len(pool)} accepted layouts; {expected:,} evaluation attacks',flush=True)
                eval_started=perf_counter();path=output/'progress.sqlite3'
                with open_journal(path,evaluation,resume=path.exists()) as journal:
                    completed=journal.count();require(completed<=expected,'Unexpected evaluation journal size')
                    loaded=None;loaded_id=None;progress('running')
                    for target,base,view,label in units(contract,pool):
                        key=unit_key(target,base,view,label);target_name=target['run']
                        if journal.get(key) is not None:continue
                        if should_stop() or (max_new is not None and fresh>=max_new):status='paused';return progress(status)
                        if loaded_id!=target_name:loaded=load_target(target,root);loaded_id=target_name
                        row=evaluate_one(target,loaded,base,view,label);validate_row(row)
                        require(row_key(row)==key,'Attack identity mismatch')
                        journal.put(key,row);completed+=1;fresh+=1
                        if fresh==1 or fresh%100==0:
                            value=progress('running');eta=value['rough_remaining_seconds']
                            tail=f'rough remaining {eta/3600:.2f} hours' if eta is not None else 'estimating speed'
                            print(f'{completed:,}/{expected:,} attacks; {target_name}; {tail}',flush=True)
                    stage='export';progress('running');export(journal,contract,pool,output,frozen_hash)
                status='completed';return progress(status)
        except BaseException as error:
            status='failed';progress(status);atomic_json(output/'last_error.json',{'utc':now(),'error':f'{type(error).__name__}: {error}'})
            raise
        finally:
            log({'event':'finished','utc':now(),'status':status,'stage':stage,'wall_seconds':perf_counter()-started,
                 'new_attacks':fresh,'new_pool_candidates':pool_new})


def run(max_new=None):
    contract,_=prepare()
    result=execute(contract,max_new=max_new)
    print(json.dumps(result,indent=2))


def timing(output):
    events=[json.loads(x) for x in (output/'attempts.jsonl').read_text().splitlines()]
    starts={x['attempt_id'] for x in events if x['event']=='started'}
    finishes=[x for x in events if x['event']=='finished']
    return {'measured_finished_attempt_seconds':sum(x['wall_seconds'] for x in finishes),
            'attempts_missing_end_record':sorted(starts-{x['attempt_id'] for x in finishes}),
            'scope':'pool generation, attack execution and export inside execute; excludes pauses and preflight; hard-killed attempts may lack durations'}


def pack(output=OUTPUT):
    output=Path(output)
    with run_lock(output,'.pipeline.lock'):
        receipt=read(output/'complete.json');require(receipt['status']=='completed','Final evaluation is incomplete')
        verify_files(output,receipt['outputs'])
        require(read(output/'progress.json')['status']=='completed','Run has not exited with completed progress')
        names=[*receipt['outputs'],'complete.json','progress.json','attempts.jsonl']
        target=output.parent/('Base_Siege_V2_Final_Results_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')+'.zip')
        with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
            for name in names:z.write(output/name,'version2-final-evaluation/'+name)
            z.writestr('version2-final-evaluation/timing_summary.json',json.dumps(timing(output),indent=2))
        print(target);return target

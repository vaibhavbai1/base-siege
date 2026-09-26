#!/usr/bin/env python3
"""Full development evaluation of the existing sampled Base Siege policies.

Upload beside diagnose_policy_sampling.py in the original project root.
    uv run python evaluate_sampled_development.py
    uv run python evaluate_sampled_development.py --resume
    uv run python evaluate_sampled_development.py --status

All 20 models, all 156 development bases, 8 views, and 3 fixed action seeds:
74,880 new sampled episodes. Existing deterministic and baseline results are
reused only for comparisons. No training or protected final evaluation occurs.
Results are supplemental development evidence selected after the diagnostic.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from time import perf_counter

HELPER_SHA256 = '420e07d5a741b42be1afd0993c0f5fd7c5c190de02540a9a7b306ea715808261'
VIEWS = tuple(range(8))
ACTION_SEEDS = (0,1,2)
METRICS = ('destruction','town_hall','full_clear','return','spent','ticks','zero_destruction')


def require(ok,message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text())


def csv_read(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def load_helper(root):
    path=root/'diagnose_policy_sampling.py'
    require(path.is_file(), 'Keep the original diagnose_policy_sampling.py in the project root.')
    require(digest(path)==HELPER_SHA256,
            'The diagnostic helper differs from the validated version. Restore the original downloaded file.')
    spec=importlib.util.spec_from_file_location('base_siege_sampling_helper',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unit_key(row):
    return (row['condition'],int(row['learner_seed']),int(row['base_seed']),row['motif'],int(row['view']))


def sampled_key(row):
    return (*unit_key(row),int(row['action_seed']))


def validated_diagnostic(directory,guard,api):
    receipt=json_read(directory/'diagnostic_complete.json')
    saved=json_read(directory/'diagnostic_contract.json')
    require(receipt['status']=='completed', 'The earlier diagnostic is incomplete.')
    require(receipt['episodes']==guard['expected_episodes']==7680, 'Unexpected diagnostic episode count.')
    require(receipt['deterministic_dev_reference_matches']==960, 'Diagnostic reference checks are incomplete.')
    require(digest(directory/'diagnostic_episodes.csv')==receipt['episodes_sha256'], 'Diagnostic CSV checksum mismatch.')
    require(hashlib.sha256(api.json_bytes(saved)).hexdigest()==receipt['contract_sha256'], 'Diagnostic contract checksum mismatch.')
    require(saved==guard, 'The diagnostic used different models, pools, code, runtime, or settings.')
    rows=csv_read(directory/'diagnostic_episodes.csv')
    require(len(rows)==7680,'Diagnostic CSV count mismatch.')
    lookup={}
    for row in rows:
        if row['split']=='dev' and row['mode']=='sampled':
            key=sampled_key(row)
            require(key not in lookup,'Duplicate sampled diagnostic unit.')
            lookup[key]=row
    require(len(lookup)==2880,'Expected 2,880 sampled development diagnostic games.')
    expected={(p['condition'],p['learner_seed'],int(b['seed']),b['motif'],v,s)
              for p in saved['models'] for b in saved['panels']['dev']
              for v in saved['views'] for s in saved['sampled_action_seeds']}
    require(set(lookup)==expected,'Sampled diagnostic units are incomplete or unexpected.')
    return receipt,lookup


def validate_reference_units(rows,bases,models):
    units={(b.seed,b.motif,v) for b in bases for v in VIEWS}
    groups=defaultdict(list)
    for row in rows:
        groups[(row['condition'],int(row['learner_seed']),int(row['action_seed']))].append(row)
    expected={(m['condition'],m['learner_seed'],-1) for m in models}
    expected.add(('heuristic',-1,-1))
    expected.update(('random',-1,s) for s in range(10))
    require(set(groups)==expected,'Reference policy groups differ from the planned study.')
    for key,group in groups.items():
        actual={(int(r['base_seed']),r['motif'],int(r['view'])) for r in group}
        require(len(group)==len(units) and actual==units,f'Reference has missing/duplicate games: {key}')


def prepare(args,D,api):
    # Reuse every guard from the tested diagnostic: manifests, model hashes,
    # both pool hashes, source/runtime identity, matched seeds and registration.
    runs,manifests,_,_,guard=D.prepare(args,api)
    root=args.project_root
    diagnostic_receipt,diagnostic=validated_diagnostic(root/args.diagnostic,guard,api)
    bases=sorted(api.load_pool(root/args.dev_pool,required_split='dev'),key=lambda b:(b.seed,b.motif))
    reference=csv_read(root/args.reference_evaluation/'episodes.csv')
    validate_reference_units(reference,bases,guard['models'])
    require(len(bases)==156,'Expected the existing frozen development pool of 156 layouts.')
    contract={k:guard[k] for k in ('runtime','pool_sha256','registration_sha256','models','device','torch_threads',
                                 'reference_receipt_sha256','reference_episodes_sha256','rng_rule')}
    contract.update(schema_version=1,kind='full-development-sampled-policy-evaluation',
        evidence_scope='supplemental development evaluation, specified after the sampling diagnostic; not final-test evidence',
        script_sha256=digest(__file__),helper_sha256=HELPER_SHA256,
        diagnostic_receipt_sha256=digest(root/args.diagnostic/'diagnostic_complete.json'),
        diagnostic_contract_sha256=diagnostic_receipt['contract_sha256'],
        diagnostic_episodes_sha256=diagnostic_receipt['episodes_sha256'],
        development_bases=[json.loads(b.to_json()) for b in bases],
        views=list(VIEWS),sampled_action_seeds=list(ACTION_SEEDS),
        expected_episodes=len(runs)*len(bases)*len(VIEWS)*len(ACTION_SEEDS),
        expected_diagnostic_matches=len(diagnostic),
        primary_metric='destruction',
        comparisons=['sampled PA minus sampled P','sampled minus deterministic within condition','sampled minus heuristic'],
        aggregation='Average action seeds and matched base/view games within each learner, then average learners.',
        uncertainty='Paired Student-t intervals over 10 independent learner seeds; fixed development pool.',
        model_selection='Every original full-budget endpoint; no model selection or retraining.',
        references='Original deterministic and baseline episode records; not rerun or replaced.')
    return runs,manifests,bases,reference,diagnostic,contract


def compare_diagnostic(row,reference):
    key=sampled_key(row)
    for field,value in reference.items():
        if field=='evidence_scope':
            continue
        require(field in row,f'Diagnostic field absent: {field}')
        actual=row[field]
        if isinstance(actual,(float,int)):
            require(math.isclose(float(actual),float(value),rel_tol=0,abs_tol=1e-9),
                    f'Sampled diagnostic mismatch: {key}, {field}. Stop and investigate.')
        else:
            require(str(actual)==str(value),f'Sampled diagnostic mismatch: {key}, {field}.')


def paired_effect(api,name,condition,metric,differences):
    require(set(differences)==set(range(10)),'An effect must include all ten learner seeds.')
    values=api.np.array([differences[s] for s in range(10)],dtype=float)
    average=float(values.mean())
    half=float(api.t.ppf(.975,9)*values.std(ddof=1)/(10**.5))
    return {'comparison':name,'condition':condition,'metric':metric,'mean_difference':average,
            'ci95_low':average-half,'ci95_high':average+half,'n_independent_learners':10,
            'learner_differences':{str(k):float(v) for k,v in differences.items()}}


def summarize(rows,reference,D,api):
    reused=[]
    for source in reference:
        row={**source,**{m:float(source[m]) for m in METRICS if m!='zero_destruction'}}
        row.update(learner_seed=int(row['learner_seed']),view=int(row['view']),
                   zero_destruction=int(row['destruction']==0),
                   mode='deterministic' if row['condition'] in ('P','PA') else 'baseline')
        reused.append(row)
    all_rows=rows+reused
    by_learner=D.groups_mean([r for r in all_rows if r['condition'] in ('P','PA')],
                            ('condition','learner_seed','mode'),fields=METRICS)
    lookup={(r['condition'],r['learner_seed'],r['mode']):r for r in by_learner}
    overall=D.groups_mean(all_rows,('condition','mode'),fields=METRICS)
    heuristic=next(r for r in overall if r['condition']=='heuristic')
    effects=[]
    for metric in ('destruction','town_hall','full_clear','return'):
        differences={s:lookup['PA',s,'sampled'][metric]-lookup['P',s,'sampled'][metric] for s in range(10)}
        effects.append(paired_effect(api,'sampled PA minus sampled P','PA-P',metric,differences))
        for condition in ('P','PA'):
            differences={s:lookup[condition,s,'sampled'][metric]-lookup[condition,s,'deterministic'][metric] for s in range(10)}
            effects.append(paired_effect(api,'sampled minus deterministic',condition,metric,differences))
            differences={s:lookup[condition,s,'sampled'][metric]-heuristic[metric] for s in range(10)}
            effects.append(paired_effect(api,'sampled minus heuristic',condition,metric,differences))
    return {'overall':overall,'by_learner':by_learner,'effects':effects,
        'by_view':D.groups_mean(all_rows,('condition','mode','view'),fields=METRICS),
        'by_motif':D.groups_mean(all_rows,('condition','mode','motif'),fields=METRICS),
        'by_action_seed':D.groups_mean(rows,('condition','action_seed'),fields=METRICS),
        'sampled_behaviour':D.groups_mean(rows,('condition',),fields=D.FIELDS+tuple(
            f'{prefix}_{troop}' for prefix in ('bought','deployed') for troop in api.C.TROOP_TYPES)),
        'limits':['Supplemental development evidence chosen after inspecting the diagnostic; not a held-out final result.',
                  'Deterministic and baseline rows are reused from the checksum-verified original evaluation.',
                  'Confidence intervals use ten learner seeds on the fixed pool; action repeats are not independent learners.',
                  'The development pool includes the earlier diagnostic panel, so this extends coverage rather than constituting a new untouched test.',
                  'Primary registered deterministic results remain unchanged.']}


def evaluate(args,D,api,runs,manifests,bases,reference,diagnostic,contract):
    output=(args.project_root/args.out).resolve()
    journal_path=output/'sampled.progress.sqlite3'
    if args.resume:
        require(journal_path.is_file(),'No sampled-evaluation journal to resume. Check the output folder.')
    output.mkdir(parents=True,exist_ok=args.resume)
    total=contract['expected_episodes']
    with api.open_journal(journal_path,contract,resume=args.resume) as journal:
        api.publish_json(output/'sampled_contract.json',contract)
        done=journal.count()
        require(done<=total,'Unexpected journal size.')
        start_done,started=done,perf_counter()
        print(f'Ready: {done:,}/{total:,} saved sampled episodes. Models are not retrained.',flush=True)

        def progress(label,status='running'):
            elapsed=perf_counter()-started
            rate=(done-start_done)/elapsed if done>start_done and elapsed>0 else None
            remaining=(total-done)/rate if rate else None
            state={'status':status,'completed':done,'total':total,'current':label,
                   'updated_utc':datetime.now(timezone.utc).isoformat(),
                   'new_episodes_this_session':done-start_done,'session_seconds':elapsed,
                   'episodes_per_second':rate,'estimated_remaining_seconds':remaining}
            api.atomic_json(output/'progress.json',state)
            eta=f'{remaining/60:.1f} min' if remaining is not None else 'measuring'
            print(f'{done:,}/{total:,} ({100*done/total:.1f}%) | {label} | ETA {eta}',flush=True)

        label='starting'
        progress(label)
        try:
            for index,(run,m) in enumerate(zip(runs,manifests)):
                model=None
                for action_seed in ACTION_SEEDS:
                    label=f'{run.name}, action seed {action_seed}'
                    for base_index,base in enumerate(bases):
                        for view in VIEWS:
                            key=f'{index:02d}:{action_seed}:{base_index:04d}:{view}'
                            if journal.get(key) is not None:
                                continue
                            if model is None:
                                model,_=api.load_run(run)
                            row=D.run_episode(api,model,base,view,'sampled',action_seed)
                            row.update(condition=m['settings']['condition'],learner_seed=m['settings']['seed'],split='dev',
                                       evidence_scope='supplemental development evaluation')
                            old=diagnostic.get(sampled_key(row))
                            if old is not None:
                                compare_diagnostic(row,old)
                            journal.put(key,row)
                            done+=1
                            if done%100==0 or done==total:
                                progress(label)
                del model
        except BaseException:
            progress(label,status='interrupted')
            raise
        rows=[r for _,r in journal.rows()]
        require(len(rows)==total,'Incomplete sampled episode count.')
        expected={(m['settings']['condition'],m['settings']['seed'],b.seed,b.motif,v,s)
                  for m in manifests for b in bases for v in VIEWS for s in ACTION_SEEDS}
        require(len(expected)==len(rows) and {sampled_key(r) for r in rows}==expected,
                'Sampled episode identities are incomplete or duplicated.')
        matches=0
        for row in rows:
            old=diagnostic.get(sampled_key(row))
            if old is not None:
                compare_diagnostic(row,old)
                matches+=1
        require(matches==len(diagnostic)==contract['expected_diagnostic_matches'],
                'Missing comparisons with the earlier sampled diagnostic.')
        progress('exporting results')
        summary=summarize(rows,reference,D,api)
        api.publish_bytes(output/'sampled_episodes.csv',D.csv_bytes(rows))
        api.publish_bytes(output/'sampled_summary.csv',D.csv_bytes(summary['overall']))
        api.publish_json(output/'sampled_summary.json',summary)
        api.publish_json(output/'sampled_complete.json',{'status':'completed','episodes':len(rows),
            'evidence_scope':contract['evidence_scope'],'accepted_layouts':len(bases),'views':list(VIEWS),
            'action_seeds':list(ACTION_SEEDS),'diagnostic_reference_matches':matches,
            'contract_sha256':hashlib.sha256(api.json_bytes(contract)).hexdigest(),
            'episodes_sha256':digest(output/'sampled_episodes.csv'),
            'summary_sha256':digest(output/'sampled_summary.json'),
            'original_evaluation_episodes_sha256':contract['reference_episodes_sha256']})
        progress('complete',status='completed')
        print('\nMean destruction (not accuracy):')
        for r in summary['overall']:
            print(f"{r['condition']:9} {r['mode']:13} {r['destruction']*100:6.2f}%  n={r['episodes']}")
        print(f'\nResults: {output}\nSend sampled_summary.json, sampled_episodes.csv, sampled_complete.json, and sampled_contract.json.')


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root',type=Path,default=Path.cwd())
    p.add_argument('--runs-root',type=Path,default=Path('runs/study-v2'))
    p.add_argument('--train-pool',type=Path,default=Path('bases/train-v2.json'))
    p.add_argument('--dev-pool',type=Path,default=Path('bases/dev-v2.json'))
    p.add_argument('--reference-evaluation',type=Path,default=Path('artifacts/study-dev-v2'))
    p.add_argument('--diagnostic',type=Path,default=Path('artifacts/policy-sampling-v1'))
    p.add_argument('--out',type=Path,default=Path('artifacts/sampled-dev-v1'))
    p.add_argument('--resume',action='store_true')
    p.add_argument('--status',action='store_true',help='Read progress.json without opening the evaluation database.')
    args=p.parse_args()
    args.project_root=args.project_root.resolve()
    if args.status:
        require(not args.resume,'Choose --status or --resume.')
        progress=args.project_root/args.out/'progress.json'
        require(progress.is_file(),'No progress file yet; start the evaluation first.')
        print(progress.read_text())
        return
    require((args.project_root/'train/evaluate.py').is_file(),'Run from the original Base Siege project root.')
    require(Path(__file__).resolve().parent not in [args.project_root/p for p in ('env','train','agents','experiments')],
            'Place this file in the project root, not in a registered source folder.')
    D=load_helper(args.project_root)
    api=D.load_api(args.project_root)
    api.torch.set_num_threads(1)
    api.torch.use_deterministic_algorithms(True)
    prepared=prepare(args,D,api)
    evaluate(args,D,api,*prepared)


if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped. Committed episodes remain saved. Run again with --resume.',file=sys.stderr)
        sys.exit(130)

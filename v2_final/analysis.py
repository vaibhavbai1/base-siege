"""Final summaries: learners are the unit of replication, not attacks."""
from collections import defaultdict
from statistics import mean, stdev
import numpy as np

PERCENT_METRICS=('destruction','full_clear','town_hall','zero_destruction','below_half')


def scopes(row):
    return ('all_final','transfer' if row['motif']=='layered_core' else 'familiar','motif:'+row['motif'])


def interval(values, options):
    values=np.asarray(values,dtype=float)
    if len(values)<5:return {'ci95_low_points':'','ci95_high_points':'','interval_note':'omitted: fewer than five learner observations'}
    rng=np.random.default_rng(options['seed'])
    samples=values[rng.integers(0,len(values),size=(options['resamples'],len(values)))].mean(axis=1)
    low,high=np.quantile(samples,[.025,.975])
    return {'ci95_low_points':float(low),'ci95_high_points':float(high),
            'interval_note':'learner percentile bootstrap; fixed final layouts/views/action labels; secondary intervals unadjusted'}


def summaries(rows, options):
    groups=defaultdict(list)
    for row in rows:
        for scope in scopes(row):groups[scope,row['condition'],row['learner_seed']].append(row)
    models=[]
    for (scope,condition,seed),items in sorted(groups.items()):
        entry={'scope':scope,'condition':condition,'learner_seed':seed,'mode':items[0]['mode'],
               'transitions':items[0]['transitions'],'episodes':len(items),
               'layouts':len({(r['base_seed'],r['motif']) for r in items}),
               'training_cohort':'baseline' if condition=='heuristic' else ('screening' if seed<3 else 'replication')}
        entry.update({m+'_percent':100*mean(float(r[m]) for r in items) for m in PERCENT_METRICS if m!='below_half'})
        entry['below_half_percent']=100*mean(r['destruction']<.5 for r in items)
        entry.update({'mean_'+m:mean(r[m] for r in items) for m in ('episode_return','spent','tick','decisions')})
        models.append(entry)
    conditions=[]
    for scope,condition in sorted({(r['scope'],r['condition']) for r in models}):
        items=[r for r in models if r['scope']==scope and r['condition']==condition]
        values=[r['destruction_percent'] for r in items]
        entry={'scope':scope,'condition':condition,'mode':items[0]['mode'],'learner_runs':0 if condition=='heuristic' else len(items),
               'episodes':sum(r['episodes'] for r in items),'layouts':items[0]['layouts'],
               'destruction_sd_across_learners':stdev(values) if len(values)>1 else ''}
        entry.update({m+'_percent':mean(r[m+'_percent'] for r in items) for m in PERCENT_METRICS})
        entry.update(interval(values,options) if condition!='heuristic' else {'ci95_low_points':'','ci95_high_points':'','interval_note':'deterministic baseline; no independent learner runs'})
        conditions.append(entry)
    lookup={(r['scope'],r['condition'],r['learner_seed']):r for r in models}
    paired=[]
    for scope in sorted({r['scope'] for r in models}):
        seeds=sorted(r['learner_seed'] for r in models if r['scope']==scope and r['condition']=='P')
        for seed in seeds:
            for method,reference in [('PA','P'),('P','heuristic'),('PA','heuristic')]:
                a=lookup[scope,method,seed];b=lookup[scope,reference,-1 if reference=='heuristic' else seed]
                entry={'scope':scope,'comparison':method+'_minus_'+reference,'learner_seed':seed,
                       'role':'primary' if scope=='all_final' and reference=='P' else 'secondary'}
                entry.update({m+'_change_points':a[m+'_percent']-b[m+'_percent'] for m in PERCENT_METRICS})
                paired.append(entry)
    comparisons=[]
    for scope,comparison in sorted({(r['scope'],r['comparison']) for r in paired}):
        items=[r for r in paired if r['scope']==scope and r['comparison']==comparison]
        values=[r['destruction_change_points'] for r in items]
        entry={'scope':scope,'comparison':comparison,'role':items[0]['role'],'learner_observations':len(items),
               'learners_with_positive_destruction_difference':sum(x>0 for x in values)}
        entry.update({'mean_'+m+'_change_points':mean(r[m+'_change_points'] for r in items) for m in PERCENT_METRICS})
        entry.update(interval(values,options));comparisons.append(entry)
    return {'model_summary.csv':models,'condition_summary.csv':conditions,
            'comparisons_by_learner.csv':paired,'comparisons.csv':comparisons}

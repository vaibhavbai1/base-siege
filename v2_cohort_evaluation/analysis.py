"""Equal learner weights; intervals resample learner pairs, never attacks."""
from collections import defaultdict
from statistics import mean, stdev
import numpy as np

METRICS = ('destruction_percent','full_clear_percent','town_hall_percent',
           'zero_destruction_percent','below_half_percent')
COHORTS = ('screening','replication','all')


def cohort(seed):
    return 'screening' if seed < 3 else 'replication'


def members(rows, selected):
    return [r for r in rows if selected == 'all' or r['cohort'] == selected]


def interval(values, options):
    n = len(values)
    if n < options['minimum_learner_pairs']:
        return {'ci95_low_points':'', 'ci95_high_points':'', 'interval_note':'omitted: fewer than 5 learner pairs'}
    rng = np.random.default_rng(options['seed'])
    values = np.asarray(values, dtype=float)
    boot = values[rng.integers(0,n,size=(options['resamples'],n))].mean(axis=1)
    low, high = np.quantile(boot, [.025,.975])
    return {'ci95_low_points':float(low), 'ci95_high_points':float(high),
            'interval_note':'paired learner percentile bootstrap; fixed dev layouts; unadjusted'}


def summaries(rows, options):
    groups = defaultdict(list)
    for row in rows:
        for scope in ['all_dev'] + ([] if row['monitor_layout'] else ['outside_monitor_panel']):
            groups[scope,row['condition'],row['learner_seed'],row['transitions'],row['mode']].append(row)
    models = []
    for (scope,condition,seed,steps,mode), items in sorted(groups.items()):
        entry = {'scope':scope,'cohort':cohort(seed),'condition':condition,'learner_seed':seed,
                 'transitions':steps,'mode':mode,'episodes':len(items),
                 'layouts':len({(r['base_seed'],r['motif']) for r in items})}
        entry.update({k+'_percent':100*mean(float(r[k]) for r in items)
                      for k in ('destruction','full_clear','town_hall','zero_destruction')})
        entry['below_half_percent'] = 100*mean(r['destruction'] < .5 for r in items)
        entry.update({'mean_'+k:mean(r[k] for r in items) for k in ('episode_return','spent','tick','decisions')})
        models.append(entry)
    conditions = []
    dimensions = sorted({(r['scope'],r['condition'],r['transitions'],r['mode']) for r in models})
    for scope,condition,steps,mode in dimensions:
        group = [r for r in models if (r['scope'],r['condition'],r['transitions'],r['mode']) == (scope,condition,steps,mode)]
        for selected in COHORTS:
            items = members(group, selected)
            if not items:
                continue
            entry = {'scope':scope,'cohort':selected,'condition':condition,'transitions':steps,'mode':mode,
                     'learner_runs':len(items),'episodes':sum(r['episodes'] for r in items)}
            entry.update({k:mean(r[k] for r in items) for k in METRICS})
            entry['destruction_sd_across_learners'] = stdev(r['destruction_percent'] for r in items) if len(items)>1 else 0
            conditions.append(entry)
    lookup = {(r['scope'],r['condition'],r['learner_seed'],r['mode']):r for r in models}
    mode_rows = []
    for row in models:
        if row['mode'] != 'DS':
            continue
        for reference in ('SS','DD'):
            base = lookup[row['scope'],row['condition'],row['learner_seed'],reference]
            entry = {k:row[k] for k in ('scope','cohort','condition','learner_seed','transitions','mode')}
            entry['reference_mode'] = reference
            entry.update({'change_'+k+'_points':row[k]-base[k] for k in METRICS})
            mode_rows.append(entry)
    mode_aggregates = []
    dimensions = sorted({(r['scope'],r['condition'],r['mode'],r['reference_mode']) for r in mode_rows})
    for scope,condition,mode,reference in dimensions:
        group = [r for r in mode_rows if (r['scope'],r['condition'],r['mode'],r['reference_mode']) == (scope,condition,mode,reference)]
        for selected in COHORTS:
            items = members(group,selected)
            if not items:
                continue
            changes = [r['change_destruction_percent_points'] for r in items]
            entry = {'scope':scope,'cohort':selected,'condition':condition,'mode':mode,'reference_mode':reference,
                     'learner_pairs':len(items),'learners_with_higher_destruction':sum(v>0 for v in changes)}
            entry.update({'mean_change_'+k+'_points':mean(r['change_'+k+'_points'] for r in items) for k in METRICS})
            entry.update(interval(changes,options))
            mode_aggregates.append(entry)
    augmentation = []
    for row in models:
        if row['condition'] != 'PA':
            continue
        base = lookup[row['scope'],'P',row['learner_seed'],row['mode']]
        entry = {k:row[k] for k in ('scope','cohort','learner_seed','transitions','mode')}
        entry.update({'pa_minus_p_'+k+'_points':row[k]-base[k] for k in METRICS})
        augmentation.append(entry)
    augmentation_aggregates = []
    for scope, mode in sorted({(r['scope'],r['mode']) for r in augmentation}):
        group = [r for r in augmentation if (r['scope'],r['mode']) == (scope,mode)]
        for selected in COHORTS:
            items = members(group,selected)
            if not items:
                continue
            changes = [r['pa_minus_p_destruction_percent_points'] for r in items]
            entry = {'scope':scope,'cohort':selected,'mode':mode,'learner_pairs':len(items),
                     'pairs_with_higher_pa_destruction':sum(v>0 for v in changes)}
            entry.update({'mean_pa_minus_p_'+k+'_points':mean(r['pa_minus_p_'+k+'_points'] for r in items) for k in METRICS})
            entry.update(interval(changes,options))
            augmentation_aggregates.append(entry)
    return {'model_summary.csv':models,'condition_summary.csv':conditions,
            'mode_comparisons_by_learner.csv':mode_rows,'mode_comparisons.csv':mode_aggregates,
            'augmentation_by_seed.csv':augmentation,'augmentation_comparisons.csv':augmentation_aggregates}

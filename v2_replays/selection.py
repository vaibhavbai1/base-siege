"""Deterministic, explicitly outcome-selected cases for qualitative inspection."""
import csv
from pathlib import Path

KEY = ('condition', 'learner_seed', 'transitions', 'base_seed', 'motif', 'view', 'mode', 'sample_seed')
INTS = ('acting_seed', 'base_seed', 'decisions', 'learner_seed', 'sample_seed', 'spent', 'tick', 'transitions', 'view')
FLOATS = ('destruction', 'episode_return', 'zero_destruction')
BOOLS = ('full_clear', 'monitor_layout', 'town_hall')


def read_episodes(path):
    with Path(path).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for name in INTS:
            row[name] = int(row[name])
        for name in FLOATS:
            row[name] = float(row[name])
        for name in BOOLS:
            if row[name] not in ('True', 'False'):
                raise ValueError('Invalid boolean in evaluation CSV')
            row[name] = row[name] == 'True'
    return rows


def key(row):
    return tuple(row[name] for name in KEY)


def select_cases(rows):
    lookup = {key(row): row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError('Duplicate evaluation units')
    cases, by_key, pairs = [], {}, []

    def add(row, role):
        identity = key(row)
        if identity not in by_key:
            case = {'id': f'case-{len(cases)+1:03d}', 'roles': [], 'expected': row}
            cases.append(case)
            by_key[identity] = case
        case = by_key[identity]
        if role not in case['roles']:
            case['roles'].append(role)
        return case['id']

    for condition in ('P', 'PA'):
        for seed in range(3):
            subset = [r for r in rows if r['condition'] == condition and r['learner_seed'] == seed
                      and r['transitions'] == 4194304 and r['mode'] == 'sampled']
            categories = [
                ('lowest army-eliminated attack', [r for r in subset if r['finish_reason'] == 'army_eliminated'], False),
                ('lowest time-limit attack', [r for r in subset if r['finish_reason'] == 'tick_limit'], False),
                ('middle-ranked non-clear attack', [r for r in subset if not r['full_clear']], True),
                ('full clear with middle-ranked finish time', [r for r in subset if r['full_clear']], True),
            ]
            for role, candidates, middle in categories:
                if not candidates:
                    raise ValueError(f'No cases for {condition}-{seed}: {role}')
                order = 'tick' if role.startswith('full clear') else 'destruction'
                candidates.sort(key=lambda r: (r[order], key(r)))
                chosen = candidates[(len(candidates)-1)//2 if middle else 0]
                sampled_id = add(chosen, role)
                counterpart = dict(chosen, mode='deterministic', sample_seed=0)
                deterministic_id = add(lookup[key(counterpart)], 'deterministic counterpart on the same layout and view')
                pairs.append({'condition': condition, 'learner_seed': seed, 'selection_role': role,
                              'sampled_id': sampled_id, 'deterministic_id': deterministic_id})
    return {'schema': 1,
            'scope': 'Outcome-selected development cases for qualitative inspection; not an unbiased performance estimate',
            'selection_rule': 'Per learner: lowest army-eliminated, lowest time-limit, lower median non-clear by destruction, lower median full-clear by tick; ties use the full evaluation key. Pair each with deterministic actions on the same layout/view.',
            'cases': cases, 'pairs': pairs}

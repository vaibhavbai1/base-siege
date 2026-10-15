"""Explicit final-benchmark adapter; leaves all scientific training modules intact."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
from time import perf_counter
import zipfile

from ddqn_experiment.evaluation import GreedyPolicy, load_policy
from ddqn_experiment.runner import configure, runtime, sources, stop_signals
from env import config as C
from env.generator import BaseSpec
from env.siege_env import SiegeEnv
from env.symmetry import SymmetryWrapper
from experiments.journal import open_journal
from experiments.provenance import sha256
from experiments.rollout import play_episode
from experiments.storage import publish_bytes, publish_json, run_lock

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
DEFAULT_RUN = 'runs/ddqn/pilot-lr3e5-seed100'
DEFAULT_OUT = 'artifacts/ddqn/selected-lr3e5-seed100-v2-benchmark'
ARCHIVE = 'artifacts/version2-final-evaluation'
STEPS = 2097152
POLICY_NAME = f'step-{STEPS:012d}'
METRICS = ('destruction', 'full_clear', 'town_hall', 'episode_return', 'spent')
LIMITATIONS = [
    'One development-selected DDQN pilot learner (seed 100); no across-DDQN-learner interval.',
    'Existing V2 benchmark and PPO outcomes were known before this comparison; not a newly unseen final set.',
    'DDQN greedy legal acting; PPO DS deterministic buying and sampled deployment/waiting.',
    'DDQN selected checkpoint 2097152 transitions; PPO fixed endpoints 4194304. Not equal checkpoint budgets.',
    'Two DDQN tuning runs each consumed 4194304 transitions (8388608 total), including later discarded checkpoints.',
    'P and PA each represent ten learners; DDQN represents one selected pilot, not an algorithm-wide estimate.',
    'Filtered layouts and fixed simulator including inaccessible-target limitation; not arbitrary custom bases.',
    'No tuning or checkpoint replacement based on these benchmark outcomes; preserve this result if later work is proposed.'
]


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(path, expected):
    require(Path(path).is_file(), f'Missing required file: {path}')
    require(sha256(path) == expected, f'Checksum mismatch: {path}. Keep the evidence; do not edit expected hashes.')


def adapter_hashes():
    return {str(p.relative_to(PACKAGE)): sha256(p)
            for p in sorted(PACKAGE.rglob('*.py')) + [PACKAGE/'reference.json']}


def selected_policy(run):
    """Preserve the original loader's source, runtime and checksum checks."""
    ref = read(PACKAGE/'reference.json')
    run = Path(run)
    verify(run/'contract.json', ref['training_contract_sha256'])
    verify(run/'policies'/f'{POLICY_NAME}.pt', ref['policy_sha256'])
    network, identity = load_policy(run, steps=STEPS, device='cpu')
    require(identity['learner_seed'] == 100 and identity['training_mode'] == 'pilot', 'Wrong selected learner')
    return network, identity


def checked_benchmark(root):
    ref = read(PACKAGE/'reference.json')
    directory = Path(root)/ARCHIVE
    for name, digest in ref['benchmark_files'].items():
        verify(directory/name, digest)
    data = read(directory/'final_pool.json')
    require(data['accepted'] == 235 and len(data['bases']) == 235, 'Expected 235 frozen V2 layouts')
    require(data['seed_interval'] == [11200, 11399], 'Wrong final seed range')
    bases = [BaseSpec.from_json(json.dumps(b)) for b in data['bases']]
    keys = {(b.motif, b.seed) for b in bases}
    require(len(keys) == 235 and all(11200 <= b.seed <= 11399 for b in bases), 'Invalid or duplicate bases')
    require(dict(Counter(b.motif for b in bases)) == ref['motif_counts'], 'Unexpected motif coverage')
    return sorted(bases, key=lambda b: (b.motif, b.seed))


def scenario(base, view):
    return f'{base.motif}:{base.seed}:{view}'


def csv_payload(rows):
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode()


def freeze(run, output, root=ROOT):
    configure(0, 'cpu')
    output, run = Path(output), Path(run)
    _, identity = selected_policy(run)
    bases = checked_benchmark(root)
    ref = read(PACKAGE/'reference.json')
    contract = {
        'schema': 1, 'scope': 'selected DDQN on existing V2 final benchmark',
        'policy': identity, 'views': list(range(8)), 'layouts': len(bases), 'attacks': len(bases)*8,
        'pool_sha256': ref['benchmark_files']['final_pool.json'],
        'ppo_episodes_sha256': ref['benchmark_files']['episodes.csv'],
        'benchmark_files': ref['benchmark_files'], 'adapter_hashes': adapter_hashes(),
        'sources': sources(), 'runtime': runtime(), 'device': 'cpu', 'torch_threads': 1,
        'selection': 'Highest mean destruction among six full-development checkpoints; two learning rates, seed 100.',
        'development_mean_destruction_percent': 68.40880893300249,
        'aggregation': 'Equal views within layout, equal layouts; PPO equal three repeats then equal ten learners.',
        'primary_comparison': 'Selected DDQN versus V2 P DS; PA DS and heuristic are secondary descriptive references.',
        'limitations': LIMITATIONS,
    }
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output, '.benchmark.lock'):
        target = output/'freeze.json'
        if target.exists():
            require(read(target)['contract'] == contract, 'Existing freeze differs; do not replace the selected checkpoint.')
        else:
            require(not (output/'progress.sqlite3').exists() and not (output/'episodes.csv').exists(),
                    'Results exist without freeze; stop and review')
        # Immutable inference copy: subsequent evaluation does not depend on a mutable training run folder.
        for name in ('contract.json', f'policies/{POLICY_NAME}.pt', f'policies/{POLICY_NAME}.json'):
            publish_bytes(output/'selected_model'/name, (run/name).read_bytes())
        publish_bytes(output/'final_pool.json', (Path(root)/ARCHIVE/'final_pool.json').read_bytes())
        if not target.exists():
            publish_json(target, {'frozen_utc': datetime.now(timezone.utc).isoformat(), 'contract': contract})
    print(f'Frozen selected checkpoint at {STEPS:,} transitions. Ready for {len(bases)*8:,} attacks.')
    return contract


def validate_row(row):
    d = row['destruction']
    require(math.isfinite(d) and 0 <= d <= 1, 'Invalid destruction')
    require(row['full_clear'] == (d == 1), 'Inconsistent full-clear result')
    require(row['town_hall'] in (0, 1), 'Invalid Town Hall result')
    require(math.isfinite(row['episode_return']), 'Invalid return')
    require(0 <= row['spent'] <= 200 and 0 <= row['tick'] <= 300 and 0 < row['decisions'] <= 70,
            'Invalid episode accounting')


def run_units(network, bases, contract, output, resume=False):
    """Transactional episode loop, also used by bounded integration tests."""
    output = Path(output)
    policy = GreedyPolicy(network, 'cpu')
    stop = {'requested': False}
    expected = {scenario(b, v) for b in bases for v in contract['views']}
    with open_journal(output/'progress.sqlite3', contract, resume=resume) as journal, stop_signals(stop):
        existing = dict(journal.rows())
        require(set(existing) <= expected, 'Unexpected scenarios in saved progress')
        for base in bases:
            for view in contract['views']:
                key = scenario(base, view)
                row = journal.get(key)
                if row is None:
                    if stop['requested']:
                        print('Paused safely. Run the same evaluate command with --resume.', flush=True)
                        return None
                    env = SymmetryWrapper(SiegeEnv(motifs=C.MOTIFS, allow_final=True), transform=view)
                    started = perf_counter()
                    try:
                        result = play_episode(env, policy, seed=0, options={'spec': base})['summary']
                    finally:
                        env.close()
                    row = {'base_seed': base.seed, 'motif': base.motif, 'view': view,
                           **{m: result[m] for m in (*METRICS, 'tick', 'decisions', 'finish_reason')},
                           'measured_episode_seconds': perf_counter()-started}
                    validate_row(row)
                    journal.put(key, row)
                    count = journal.count()
                    if count == 1 or count % 50 == 0 or count == len(expected):
                        print(f'DDQN benchmark {count}/{len(expected)} attacks saved', flush=True)
                require(key == f"{row['motif']}:{row['base_seed']}:{row['view']}", 'Journal identity mismatch')
                validate_row(row)
        pairs = list(journal.rows())
        require({k for k, _ in pairs} == expected, 'Incomplete evaluation')
        return [r for _, r in pairs]


def scopes(motif):
    return ('all', 'transfer' if motif == 'layered_core' else 'familiar', f'motif:{motif}')


def mean(values):
    return sum(values)/len(values)


def metric_number(value):
    # The original PPO CSV writes boolean outcomes as True/False.
    if value in ('True', 'False'):
        return float(value == 'True')
    return float(value)


def compare(rows, ppo_path):
    """Match scenario identities before aggregation; no false learner pairing."""
    wanted = {(r['motif'], r['base_seed'], r['view']) for r in rows}
    require(len(wanted) == len(rows), 'Duplicate DDQN scenario')
    groups = defaultdict(list)
    seen = set()
    with Path(ppo_path).open(newline='') as f:
        for raw in csv.DictReader(f):
            condition = raw['condition']
            learner = int(raw['learner_seed'])
            label = int(raw['sample_seed'])
            key = (raw['motif'], int(raw['base_seed']), int(raw['view']))
            require(key in wanted, 'PPO and DDQN scenarios differ')
            if condition == 'heuristic':
                require(learner == -1 and label == 0 and raw['mode'] == 'heuristic', 'Wrong heuristic record')
            else:
                require(condition in ('P', 'PA') and learner in range(10) and label in (41001,41002,41003)
                        and raw['mode'] == 'DS' and int(raw['transitions']) == 4194304, 'Wrong PPO record')
            identity = (condition, learner, *key, label)
            require(identity not in seen, 'Duplicate PPO record')
            seen.add(identity)
            item = {m: metric_number(raw[m]) for m in METRICS}
            require(all(math.isfinite(v) for v in item.values()), 'Nonfinite archived metric')
            groups[(condition, learner, key)].append(item)
    targets = [('P', s) for s in range(10)] + [('PA', s) for s in range(10)] + [('heuristic', -1)]
    scenario_means = {}
    for condition, learner in targets:
        for key in wanted:
            items = groups[(condition, learner, key)]
            require(len(items) == (1 if condition == 'heuristic' else 3), 'Missing archived attacks')
            scenario_means[(condition, learner, key)] = {m: mean([r[m] for r in items]) for m in METRICS}
    by_model = defaultdict(list)
    for (condition, learner, key), values in scenario_means.items():
        for scope in scopes(key[0]):
            by_model[(condition, learner, scope)].append(values)
    for row in rows:
        for scope in scopes(row['motif']):
            by_model[('DDQN', 100, scope)].append(row)
    models = []
    for (condition, learner, scope), items in sorted(by_model.items()):
        models.append({'condition': condition, 'learner_seed': learner, 'scope': scope,
                       'layout_views': len(items),
                       **{m: mean([r[m] for r in items]) for m in METRICS}})
    conditions = []
    for scope in sorted({r['scope'] for r in models}):
        for condition in ('DDQN', 'P', 'PA', 'heuristic'):
            items = [r for r in models if r['condition'] == condition and r['scope'] == scope]
            conditions.append({'condition': condition, 'scope': scope,
                               'learners': 0 if condition == 'heuristic' else len(items),
                               'mean_destruction_percent': 100*mean([r['destruction'] for r in items]),
                               'full_clear_percent': 100*mean([r['full_clear'] for r in items]),
                               'town_hall_percent': 100*mean([r['town_hall'] for r in items]),
                               'mean_return': mean([r['episode_return'] for r in items])})
    deltas = []
    for scope in sorted({r['scope'] for r in conditions}):
        items = {r['condition']: r for r in conditions if r['scope'] == scope}
        for reference in ('P', 'PA', 'heuristic'):
            deltas.append({'scope': scope, 'comparison': f'DDQN minus {reference}',
                           'destruction_percentage_points': items['DDQN']['mean_destruction_percent']-items[reference]['mean_destruction_percent'],
                           'full_clear_percentage_points': items['DDQN']['full_clear_percent']-items[reference]['full_clear_percent'],
                           'inference': 'descriptive only; one DDQN learner, no paired-learner confidence interval'})
    return models, conditions, deltas


def evaluate(output, resume=False, root=ROOT):
    output = Path(output)
    configure(0, 'cpu')
    with run_lock(output, '.benchmark.lock'):
        frozen = read(output/'freeze.json')
        contract = frozen['contract']
        require(contract['adapter_hashes'] == adapter_hashes(), 'Evaluation adapter changed after freeze')
        require(contract['sources'] == sources() and contract['runtime'] == runtime(), 'Scientific source or runtime changed')
        bases = checked_benchmark(root)
        verify(output/'final_pool.json', contract['pool_sha256'])
        network, identity = selected_policy(output/'selected_model')
        require(identity == contract['policy'], 'Frozen policy changed')
        rows = run_units(network, bases, contract, output, resume=resume)
        if rows is None:
            return
        models, conditions, deltas = compare(rows, Path(root)/ARCHIVE/'episodes.csv')
        summary = {'scope': contract['scope'], 'policy': identity, 'layouts': len(bases),
                   'views': contract['views'], 'episodes': len(rows),
                   'mean_destruction_percent': 100*mean([r['destruction'] for r in rows]),
                   'full_clear_percent': 100*mean([r['full_clear'] for r in rows]),
                   'town_hall_percent': 100*mean([r['town_hall'] for r in rows]),
                   'full_clears': sum(r['full_clear'] for r in rows),
                   'zero_destruction_attacks': sum(r['destruction'] == 0 for r in rows),
                   'mean_return': mean([r['episode_return'] for r in rows]),
                   'measured_episode_seconds_sum': sum(r['measured_episode_seconds'] for r in rows),
                   'timing_scope': 'Episode loops only; excludes preflight, checkpoint loading, comparison and export.',
                   'limitations': LIMITATIONS}
        files = {'episodes.csv': csv_payload(rows), 'model_summary.csv': csv_payload(models),
                 'condition_summary.csv': csv_payload(conditions), 'comparisons.csv': csv_payload(deltas)}
        for name, payload in files.items():
            publish_bytes(output/name, payload)
        publish_json(output/'summary.json', summary)
        protected = ['freeze.json', 'final_pool.json', 'summary.json', *files]
        publish_json(output/'complete.json', {'status': 'completed', 'episodes': len(rows),
                                            'outputs': {name: sha256(output/name) for name in protected}})
        print(json.dumps(summary, indent=2))
        print('Completed. Keep this result frozen; send the review ZIP for analysis.')


def pack(output, destination):
    output, destination = Path(output), Path(destination)
    with run_lock(output, '.benchmark.lock'):
        receipt = read(output/'complete.json')
        require(receipt['status'] == 'completed' and receipt['episodes'] == 1880,
                'Evaluation must finish before making the review ZIP')
        for name, digest in receipt['outputs'].items():
            require(Path(name).name == name, 'Invalid receipt filename')
            verify(output/name, digest)
        with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for name in [*receipt['outputs'], 'complete.json']:
                archive.write(output/name, arcname=name)
    print(f'Review ZIP saved: {destination}. It excludes model weights and recovery files.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    f = sub.add_parser('freeze', help='Verify and copy the selected checkpoint; no attacks')
    f.add_argument('--run', type=Path, default=Path(DEFAULT_RUN))
    f.add_argument('--out', type=Path, default=Path(DEFAULT_OUT))
    e = sub.add_parser('evaluate', help='Run 1880 final-benchmark attacks after freeze')
    e.add_argument('--out', type=Path, default=Path(DEFAULT_OUT))
    e.add_argument('--resume', action='store_true')
    p = sub.add_parser('pack', help='Make a compact review ZIP after completion')
    p.add_argument('--out', type=Path, default=Path(DEFAULT_OUT))
    p.add_argument('--zip', type=Path, default=Path('ddqn_selected_v2_benchmark_review.zip'))
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(args.run, args.out)
    elif args.command == 'evaluate':
        evaluate(args.out, args.resume)
    else:
        pack(args.out, args.zip)

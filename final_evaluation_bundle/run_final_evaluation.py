#!/usr/bin/env python3
"""Prepare a pre-final amendment, then run original and sampled final evaluations.

Run from the unchanged Base Siege project root. No training is performed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import tempfile
import threading
import zipfile
from time import perf_counter
from types import SimpleNamespace

BUNDLE = Path(__file__).resolve().parent
F_HASH = '9380ec90389a86f84eb6b42af7a5f7798d3c9120fe5fae08d2fa1e906f134ce8'
FREEZE = Path('experiments/final-freeze-v2.json')
AMENDMENT = Path('experiments/final-sampling-amendment-v1.json')
ORIGINAL = Path('artifacts/study-final-v2')
SAMPLED = Path('artifacts/sampled-final-v1')
STATE = Path('artifacts/final-evaluation-v1-progress.json')
VIEWS = tuple(range(8))
ACTION_SEEDS = (0, 1, 2)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dependencies():
    path = BUNDLE / 'evaluate_sampled_development.py'
    require(digest(path) == F_HASH, 'Bundled development helper changed; restore the downloaded bundle.')
    spec = importlib.util.spec_from_file_location('final_sampling_summary_helper', path)
    F = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(F)
    return F, F.load_helper(BUNDLE)


def development_inputs(root, F, D, api):
    args = SimpleNamespace(project_root=root, runs_root=Path('runs/study-v2'),
        train_pool=Path('bases/train-v2.json'), dev_pool=Path('bases/dev-v2.json'),
        reference_evaluation=Path('artifacts/study-dev-v2'), diagnostic=Path('artifacts/policy-sampling-v1'))
    runs, manifests, _, _, _, expected = F.prepare(args, D, api)
    directory = root / 'artifacts/sampled-dev-v1'
    receipt = read(directory / 'sampled_complete.json')
    require(receipt['status'] == 'completed' and receipt['episodes'] == 74880 and
            receipt['diagnostic_reference_matches'] == 2880, 'Full sampled development evaluation is incomplete.')
    require(read(directory / 'sampled_contract.json') == expected, 'Development contract differs from current inputs.')
    for filename, field in [('sampled_episodes.csv', 'episodes_sha256'),
                            ('sampled_summary.json', 'summary_sha256'),
                            ('sampled_contract.json', 'contract_sha256')]:
        require(digest(directory / filename) == receipt[field], f'Development checksum mismatch: {filename}')
    files = [Path('experiments/registration-v2.json'),
             Path('artifacts/study-dev-v2/evaluation.json'),
             Path('artifacts/study-dev-v2/episodes.csv'),
             Path('artifacts/policy-sampling-v1/diagnostic_complete.json')]
    files += [Path('artifacts/sampled-dev-v1') / name for name in
              ('sampled_complete.json', 'sampled_contract.json', 'sampled_episodes.csv', 'sampled_summary.json')]
    evidence = {str(p): digest(root / p) for p in files}
    require(evidence['experiments/registration-v2.json'] == expected['registration_sha256'],
            'The original registration file differs from the trained models.')
    return runs, manifests, expected, evidence


def specification(root, F, D, api):
    runs, manifests, development, evidence = development_inputs(root, F, D, api)
    spec = {
        'schema_version': 1, 'kind': 'pre-final-sampled-evaluation-amendment',
        'decision_timing': 'Specified after inspecting development results, before this final evaluation starts.',
        'reason': 'Sampling the learned action distribution substantially outperformed greedy inference on development bases.',
        'original_study': 'Preserve the registered deterministic evaluation and its primary PA-minus-P comparison.',
        'extension': 'Separately evaluate stochastic inference using all existing fixed-budget models; no retraining or selection.',
        'script_sha256': digest(__file__), 'development_helper_sha256': F_HASH,
        'diagnostic_helper_sha256': F.HELPER_SHA256,
        'runtime': development['runtime'], 'models': development['models'],
        'development_evidence_sha256': evidence,
        'registration_sha256': development['registration_sha256'],
        'freeze_path': str(FREEZE), 'original_output': str(ORIGINAL), 'sampled_output': str(SAMPLED),
        'final_candidate_seeds': list(api.C.FINAL_STUDY_SEED_RANGE),
        'motifs': list(api.C.MOTIFS), 'views': list(VIEWS), 'sampled_action_seeds': list(ACTION_SEEDS),
        'filter': 'Unchanged registered structure-and-difficulty filter; no additional exclusions; no reserve seeds.',
        'inference': {'device': 'cpu', 'torch_threads': 1, 'deterministic': False,
                      'action_masking': True, 'rng_rule': development['rng_rule']},
        'primary_extension_metric': 'destruction',
        'extension_comparisons': ['sampled PA minus sampled P', 'sampled minus deterministic', 'sampled minus heuristic'],
        'secondary_metrics': ['town_hall', 'full_clear', 'return', 'spent', 'zero_destruction'],
        'aggregation': 'Average three action seeds and base/view games within learner; compare ten matched learner seeds.',
        'uncertainty': '95% paired Student-t intervals over ten learner seeds, conditional on the accepted final pool. Secondary comparisons are descriptive, without multiplicity correction.',
        'reporting': 'Report overall, familiar structures, transfer structure, and each motif separately. Do not choose models or inference settings using final scores.',
        'transfer_limit': 'layered_core was excluded from training and current development; earlier software checks included this structure.',
        'scope_limit': 'Findings apply to this simplified simulator and difficulty-filtered base distribution, not the commercial game.',
    }
    return runs, manifests, spec


def validate_frozen(root, runs, spec):
    frozen = read(root / FREEZE)
    require(frozen['schema_version'] == 2, 'Unsupported freeze version.')
    require(frozen['registration_sha256'] == spec['registration_sha256'] and
            frozen['source_hashes'] == spec['runtime']['source_hashes'], 'Freeze differs from the amendment.')
    expected = {(r.resolve(), m['model_sha256'], m['manifest_sha256']) for r, m in zip(runs, spec['models'])}
    actual = {((root / FREEZE).parent.joinpath(r['path']).resolve(), r['model_sha256'], r['manifest_sha256'])
              for r in frozen['runs']}
    require(len(frozen['runs']) == 20 and actual == expected, 'Freeze does not contain exactly these twenty models.')


def prepare(root, F, D, api):
    from experiments.protocol import freeze
    from experiments.storage import run_lock
    (root / 'experiments').mkdir(exist_ok=True)
    with run_lock(root / 'experiments', '.final-extension-v1.lock'):
        runs, _, spec = specification(root, F, D, api)
        if (root / AMENDMENT).exists():
            saved = read(root / AMENDMENT)
            require(saved['specification'] == spec, 'Existing amendment differs; do not overwrite it.')
            validate_frozen(root, runs, spec)
            require(saved['freeze_sha256'] == digest(root / FREEZE), 'Frozen manifest changed.')
            print('Already prepared with identical inputs. No files changed.')
            return
        markers = list((root / 'experiments').rglob('*.final-started.json'))
        require(not markers and not (root / ORIGINAL).exists() and not (root / SAMPLED).exists(),
                'Final evaluation appears to have started already. Stop: a pre-final amendment cannot be backdated.')
        if not (root / FREEZE).exists():
            # Original freeze validation, then atomic publication under its original relative-path semantics.
            fd, temporary = tempfile.mkstemp(prefix='freeze-', suffix='.json', dir=root / 'experiments')
            os.close(fd)
            Path(temporary).unlink()
            try:
                freeze(root / 'experiments/registration-v2.json', runs, temporary)
                api.publish_bytes(root / FREEZE, Path(temporary).read_bytes())
            finally:
                Path(temporary).unlink(missing_ok=True)
        validate_frozen(root, runs, spec)
        api.publish_json(root / AMENDMENT, {'created_utc': datetime.now(timezone.utc).isoformat(),
            'freeze_sha256': digest(root / FREEZE), 'specification': spec})
        print(f'Prepared {AMENDMENT} and {FREEZE}. No final bases generated or evaluated.')


@contextmanager
def heartbeat(root, api, phase):
    stop = threading.Event()
    started = perf_counter()
    def update():
        elapsed = perf_counter() - started
        api.atomic_json(root / STATE, {'phase': phase, 'status': 'running',
            'updated_utc': datetime.now(timezone.utc).isoformat(), 'phase_seconds_this_session': elapsed})
        print(f'{phase} | active | {elapsed / 60:.1f} min this session', flush=True)
    def worker():
        while not stop.wait(30):
            update()
    update()
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def final_inputs(root, F, api, amendment):
    """Only called after the original confirmed, frozen final protocol completes."""
    from env.generator import BaseSpec
    receipt = read(root / ORIGINAL / 'evaluation.json')
    complete = read(root / ORIGINAL / 'complete.json')
    marker = read((root / FREEZE).with_suffix('.final-started.json'))
    expected_binding = {'schema_version': 2, 'freeze_sha256': digest(root / FREEZE),
                        'output': os.path.relpath(root / ORIGINAL, (root / FREEZE).parent)}
    require(marker == expected_binding and receipt['identity'] == expected_binding,
            'Original final authorization/output binding differs.')
    require(receipt['status'] == 'completed' and receipt['mode'] == 'final' and
            complete['freeze_sha256'] == amendment['freeze_sha256'] and not complete['reserve_seeds_used'],
            'Original final evaluation is incomplete or mismatched.')
    require(receipt['episodes_sha256'] == complete['episodes_sha256'] == digest(root / ORIGINAL / 'episodes.csv'),
            'Original final episode checksum mismatch.')
    pool = read(root / ORIGINAL / 'final_pool.json')
    require(pool['seed_interval'] == list(api.C.FINAL_STUDY_SEED_RANGE) and pool['motifs'] == list(api.C.MOTIFS)
            and pool['filter'] == 'structure-and-difficulty', 'Unexpected final pool definition.')
    bases = sorted([BaseSpec.from_json(json.dumps(b)) for b in pool['bases']], key=lambda b: (b.seed, b.motif))
    require(len(bases) == complete['accepted_layouts'] == receipt['accepted_layouts'] and bases,
            'Final pool size mismatch.')
    low, high = api.C.FINAL_STUDY_SEED_RANGE
    require(len({(b.seed, b.motif) for b in bases}) == len(bases) and
            all(low <= b.seed <= high and b.motif in api.C.MOTIFS for b in bases), 'Unexpected final base identities.')
    # Bind the pool's exact serialized bases to the immutable original evaluation journal.
    from experiments.provenance import runtime_manifest
    current = json.loads(json.dumps(runtime_manifest()))
    original_contract = {'schema_version': 1, 'kind': 'episode-evaluation', 'mode': 'final',
        'identity': expected_binding, 'views': list(VIEWS), 'policies': receipt['policies'], 'torch_threads': 1,
        'bases_sha256': hashlib.sha256(api.json_bytes([json.loads(b.to_json()) for b in bases])).hexdigest(),
        **{k: current[k] for k in ('source_hashes', 'config', 'python', 'packages')}}
    require(hashlib.sha256(api.json_bytes(original_contract)).hexdigest() == receipt['contract_sha256'],
            'Final pool or original evaluation contract changed.')
    models = amendment['specification']['models']
    policies = [p for p in receipt['policies'] if p['condition'] in ('P', 'PA')]
    require([{k: p[k] for k in ('condition', 'learner_seed', 'model_sha256', 'manifest_sha256')} for p in policies]
            == models, 'Original final evaluation used different models.')
    reference = F.csv_read(root / ORIGINAL / 'episodes.csv')
    F.validate_reference_units(reference, bases, models)
    require(len(reference) == receipt['rows'], 'Original final episode count mismatch.')
    contract = {'schema_version': 1, 'kind': 'amended-sampled-final-evaluation',
        'evidence_scope': 'Final-test extension specified after development but before final access; preserve original registered results.',
        'amendment_sha256': digest(root / AMENDMENT), 'freeze_sha256': digest(root / FREEZE),
        'script_sha256': digest(__file__), 'runtime': amendment['specification']['runtime'], 'models': models,
        'final_pool_sha256': digest(root / ORIGINAL / 'final_pool.json'),
        'reference_receipt_sha256': digest(root / ORIGINAL / 'evaluation.json'),
        'reference_episodes_sha256': receipt['episodes_sha256'],
        'views': list(VIEWS), 'sampled_action_seeds': list(ACTION_SEEDS),
        'expected_episodes': len(bases) * len(models) * len(VIEWS) * len(ACTION_SEEDS),
        'rng_rule': amendment['specification']['inference']['rng_rule']}
    return bases, reference, contract


def run_episode(api, D, model, base, view, action_seed):
    # Explicit protected access only inside the confirmed final orchestration.
    seed = D.episode_seed(base, view, action_seed)
    random.seed(seed)
    api.np.random.seed(seed)
    api.torch.manual_seed(seed)
    env = api.SymmetryWrapper(api.SiegeEnv(motifs=api.C.MOTIFS, fixed_army=False, allow_final=True), transform=view)
    try:
        replay = api.play_episode(env, D.ActingPolicy(model, False), seed=0, options={'spec': base})
    finally:
        env.close()
    s = replay['summary']
    return {'base_seed': base.seed, 'motif': base.motif, 'view': view, 'mode': 'sampled',
        'action_seed': action_seed, 'rng_seed': seed, 'destruction': s['destruction'],
        'town_hall': int(s['town_hall']), 'full_clear': int(s['full_clear']),
        'return': s['episode_return'], 'spent': s['spent'], 'ticks': s['tick'],
        'finish_reason': s['finish_reason'], 'zero_destruction': int(s['destruction'] == 0),
        **D.action_statistics(replay['actions'], api.C)}


def summarize(rows, reference, F, D, api):
    result = F.summarize(rows, reference, D, api)
    result['limits'] = [
        'Sampled extension specified after development and before final access; original deterministic study retained.',
        'Intervals describe ten learner seeds on this fixed difficulty-filtered final pool; action repeats are not independent learners.',
        'Secondary comparisons and motif analyses are descriptive; no multiple-comparison correction.',
        'Transfer structure was held out from training/current development but appeared in earlier software checks.',
        'Pooled means weight accepted bases equally, so motif weights depend on filter acceptance.']
    predicates = {'seen_structures': lambda m: m in api.C.TRAIN_MOTIFS,
                  'transfer_structure': lambda m: m not in api.C.TRAIN_MOTIFS}
    predicates.update({f'motif/{m}': lambda actual, expected=m: actual == expected for m in api.C.MOTIFS})
    result['subsets'] = {}
    for label, predicate in predicates.items():
        subset = [r for r in rows if predicate(r['motif'])]
        refs = [r for r in reference if predicate(r['motif'])]
        result['subsets'][label] = ({k: v for k, v in F.summarize(subset, refs, D, api).items()
                                    if k in ('overall', 'by_learner', 'effects')} if subset else
                                   {'status': 'no accepted layouts; do not consume reserve seeds'})
    per_view = D.groups_mean(rows, ('condition', 'learner_seed', 'motif', 'view'), fields=('destruction',))
    groups = {}
    for row in per_view:
        groups.setdefault((row['condition'], row['learner_seed'], row['motif']), []).append(row['destruction'])
    result['sampled_orientation_gap'] = [dict(condition=c, learner_seed=s, motif=m, orientation_gap=max(v)-min(v))
                                        for (c, s, m), v in sorted(groups.items())]
    return result


def sampled(root, F, D, api, runs, manifests, bases, reference, contract):
    output = root / SAMPLED
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'sampled.progress.sqlite3'
    with api.open_journal(path, contract, resume=path.exists()) as journal:
        api.publish_json(output / 'sampled_contract.json', contract)
        done, total = journal.count(), contract['expected_episodes']
        require(done <= total, 'Unexpected journal size.')
        initial, started = done, perf_counter()
        def progress(label, status='running'):
            seconds = perf_counter() - started
            rate = (done-initial) / seconds if done > initial and seconds > 0 else None
            eta = (total-done) / rate if rate else None
            state = {'phase': 'sampled final evaluation', 'status': status, 'completed': done, 'total': total,
                'current': label, 'updated_utc': datetime.now(timezone.utc).isoformat(),
                'episodes_per_second': rate, 'estimated_remaining_seconds': eta}
            api.atomic_json(output / 'progress.json', state)
            api.atomic_json(root / STATE, state)
            estimate = f'{eta/60:.1f} min' if eta is not None else 'measuring'
            print(f'{done:,}/{total:,} ({100*done/total:.1f}%) | {label} | ETA {estimate}', flush=True)
        label = 'starting'
        progress(label)
        try:
            for index, (run, manifest) in enumerate(zip(runs, manifests)):
                model = None
                for action_seed in ACTION_SEEDS:
                    label = f'{run.name}, action seed {action_seed}'
                    for base_index, base in enumerate(bases):
                        for view in VIEWS:
                            key = f'{index:02d}:{action_seed}:{base_index:04d}:{view}'
                            if journal.get(key) is not None:
                                continue
                            if model is None:
                                model, _ = api.load_run(run)
                            row = run_episode(api, D, model, base, view, action_seed)
                            row.update(condition=manifest['settings']['condition'], learner_seed=manifest['settings']['seed'],
                                split='final', evidence_scope=contract['evidence_scope'])
                            journal.put(key, row)
                            done += 1
                            if done % 100 == 0 or done == total:
                                progress(label)
                del model
        except BaseException:
            progress(label, 'interrupted')
            raise
        rows = [r for _, r in journal.rows()]
        expected = {(m['settings']['condition'], m['settings']['seed'], b.seed, b.motif, v, s)
                    for m in manifests for b in bases for v in VIEWS for s in ACTION_SEEDS}
        require(len(rows) == len(expected) == total and {F.sampled_key(r) for r in rows} == expected,
                'Missing, duplicate or unexpected sampled episode identities.')
        progress('exporting results')
        summary = summarize(rows, reference, F, D, api)
        api.publish_bytes(output / 'sampled_episodes.csv', D.csv_bytes(rows))
        api.publish_bytes(output / 'sampled_summary.csv', D.csv_bytes(summary['overall']))
        api.publish_json(output / 'sampled_summary.json', summary)
        api.publish_json(output / 'sampled_complete.json', {'status': 'completed', 'episodes': len(rows),
            'accepted_layouts': len(bases), 'contract_sha256': digest(output / 'sampled_contract.json'),
            'episodes_sha256': digest(output / 'sampled_episodes.csv'),
            'summary_sha256': digest(output / 'sampled_summary.json'),
            'amendment_sha256': contract['amendment_sha256'], 'freeze_sha256': contract['freeze_sha256'],
            'reserve_seeds_used': False, 'evidence_scope': contract['evidence_scope']})
        progress('complete', 'completed')
        for row in summary['overall']:
            print(f"{row['condition']:9} {row['mode']:13} destruction={row['destruction']*100:.2f}% n={row['episodes']}")


def run(root, F, D, api, *, confirm_final, resume):
    require(confirm_final, 'Use --confirm-final to authorize the frozen final evaluation.')
    from experiments.protocol import final_evaluation
    from experiments.storage import run_lock
    require((root / AMENDMENT).is_file(), 'Run the prepare command first.')
    with run_lock(root / 'experiments', '.final-extension-v1.lock'):
        runs, manifests, spec = specification(root, F, D, api)
        amendment = read(root / AMENDMENT)
        require(amendment['specification'] == spec, 'Amendment inputs changed; restore the original files.')
        validate_frozen(root, runs, spec)
        require(amendment['freeze_sha256'] == digest(root / FREEZE), 'Freeze changed after amendment.')
        marker = (root / FREEZE).with_suffix('.final-started.json')
        if marker.exists():
            require(resume, 'Final evaluation already started. Use run --confirm-final --resume.')
        else:
            require(not (root / ORIGINAL).exists() and not (root / SAMPLED).exists(),
                    'Final output exists without its authorization marker. Stop and investigate.')
        try:
            with heartbeat(root, api, 'original final protocol: pool generation and deterministic evaluation'):
                # Original protocol owns protected pool generation, locking and resume binding.
                final_evaluation(root / FREEZE, root / ORIGINAL, confirm_final=True, resume=marker.exists())
            bases, reference, contract = final_inputs(root, F, api, amendment)
            sampled(root, F, D, api, runs, manifests, bases, reference, contract)
        except BaseException:
            state = read(root / STATE) if (root / STATE).exists() else {}
            api.atomic_json(root / STATE, {**state, 'status': 'interrupted',
                'updated_utc': datetime.now(timezone.utc).isoformat()})
            raise
        print(f'Both final evaluations completed. Results: {ORIGINAL} and {SAMPLED}', flush=True)


def collect(root):
    """Package completed outputs only; never archive a live progress database."""
    original = read(root / ORIGINAL / 'complete.json')
    sampled_receipt = read(root / SAMPLED / 'sampled_complete.json')
    require(sampled_receipt['status'] == 'completed', 'Sampled final evaluation is incomplete.')
    require(original['freeze_sha256'] == sampled_receipt['freeze_sha256'] == digest(root / FREEZE),
            'Final receipts do not share the current freeze.')
    require(sampled_receipt['amendment_sha256'] == digest(root / AMENDMENT), 'Amendment checksum mismatch.')
    require(original['episodes_sha256'] == digest(root / ORIGINAL / 'episodes.csv'), 'Original final CSV checksum mismatch.')
    for filename, field in [('sampled_episodes.csv','episodes_sha256'), ('sampled_summary.json','summary_sha256'),
                            ('sampled_contract.json','contract_sha256')]:
        require(sampled_receipt[field] == digest(root / SAMPLED / filename), f'Checksum mismatch: {filename}')
    files = [FREEZE, AMENDMENT, FREEZE.with_suffix('.final-started.json'), Path('experiments/registration-v2.json')]
    files += [ORIGINAL / name for name in ('episodes.csv','evaluation.json','paired_summary.json',
                                          'study_summary.json','complete.json','final_pool.json')]
    files += [SAMPLED / name for name in ('sampled_episodes.csv','sampled_summary.json',
                                         'sampled_summary.csv','sampled_complete.json','sampled_contract.json')]
    require(all((root / p).is_file() for p in files), 'Some completed result files are missing.')
    output = root / 'Base_Siege_Final_Results.zip'
    # Re-running collection safely regenerates just this derived ZIP, leaving evidence intact.
    fd, temporary = tempfile.mkstemp(prefix='final-results-', suffix='.zip', dir=root)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(root / path, str(path))
            for name in ('run_final_evaluation.py','diagnose_policy_sampling.py','evaluate_sampled_development.py','README.md'):
                archive.write(BUNDLE / name, f'final_evaluation_bundle/{name}')
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f'Created {output.name}. Download this ZIP and upload it for analysis.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'status', 'collect'))
    parser.add_argument('--confirm-final', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = Path.cwd().resolve()
    if args.command == 'collect':
        collect(root)
        return
    if args.command == 'status':
        require((root / STATE).exists(), 'No progress yet. Start the final evaluation first.')
        print((root / STATE).read_text())
        return
    require((root / 'train/evaluate.py').is_file(), 'Run this command from the original Base Siege project root.')
    F, D = dependencies()
    api = D.load_api(root)
    api.torch.set_num_threads(1)
    api.torch.use_deterministic_algorithms(True)
    if args.command == 'prepare':
        require(not args.resume and not args.confirm_final, 'prepare does not accept run flags.')
        prepare(root, F, D, api)
    else:
        run(root, F, D, api, confirm_final=args.confirm_final, resume=args.resume)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted. Resume with: uv run python final_evaluation_bundle/run_final_evaluation.py run --confirm-final --resume')
        raise SystemExit(130)

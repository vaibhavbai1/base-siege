"""Reproduce reviewed development rows using their original CPU acting seeds."""
from collections import Counter
import csv
import html
import io
import json
from pathlib import Path
import random
import tempfile
from time import perf_counter
import zipfile

import numpy as np
import torch
from env import config as C
from env.combat import in_range, _approach_goals
from env.siege_env import SiegeEnv, describe_action
from env.symmetry import SymmetryWrapper
from experiments.rollout import play_episode
from experiments.storage import publish_json, publish_bytes, run_lock
from v2_development import runner as E
from version2.evaluation import ActingPolicy, episode_seed
from .selection import read_episodes, select_cases
from .renderer import export_html

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
OUTPUT = ROOT / 'artifacts/version2-replay-inspection'
RESULTS = ROOT / 'artifacts/version2-full-development'
SUMMARY_FIELDS = ('base_seed', 'motif', 'view', 'destruction', 'full_clear', 'town_hall',
                  'episode_return', 'spent', 'tick', 'decisions', 'finish_reason')


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def hashes(folder):
    return {str(p.relative_to(folder)): E.sha256(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py', '.json', '.js', '.css', '.html')}


def prepare():
    provenance = read(PACKAGE/'reviewed_results.json')
    for name, digest in provenance['files'].items():
        path = RESULTS/name
        require(path.is_file(), f'Missing completed development result: {path}')
        require(E.sha256(path) == digest, f'Reviewed result changed: {name}')
    evaluation_contract, pool = E.prepare()
    saved_contract = read(RESULTS/'evaluation_contract.json')
    require(evaluation_contract == saved_contract, 'Evaluation code, inputs or runtime changed. Use the completed evaluation environment.')
    require(read(RESULTS/'complete.json')['status'] == 'completed', 'Development evaluation is incomplete')
    plan = read(PACKAGE/'selection_plan.json')
    require(select_cases(read_episodes(RESULTS/'episodes.csv')) == plan, 'Case selection differs from the reviewed data')
    contract = {'schema': 1, 'scope': plan['scope'], 'package_hashes': hashes(PACKAGE),
                'reviewed_results': provenance, 'selection_plan_sha256': E.sha256(PACKAGE/'selection_plan.json'),
                'models': [t for t in evaluation_contract['models'] if t['transitions'] == 4194304],
                'runtime': evaluation_contract['scientific_runtime'], 'device': 'cpu',
                'torch_threads': 1, 'case_count': len(plan['cases']), 'pair_count': len(plan['pairs'])}
    return contract, plan, pool


class InspectionEnv(SiegeEnv):
    """Adds passive annotations to presentation frames, never to observations."""
    def _add_visual_movement(self, frame):
        super()._add_visual_movement(frame)
        troops = {t.entity_id: t for t in self.state.troops}
        for entry in frame['troops']:
            troop = troops[entry['entity_id']]
            target = self.state.grid.building_at(troop.target) if troop.target is not None else None
            alive = target is not None and target.is_alive
            entry['inspection_target'] = list(troop.target) if troop.target is not None else None
            entry['inspection_target_alive'] = bool(alive)
            entry['inspection_target_in_range'] = bool(alive and in_range(troop, target))
            entry['inspection_approach_goals'] = len(_approach_goals(self.state, troop.target)) if alive else None


def validate_summary(replay, expected):
    actual = replay['summary']
    for field in SUMMARY_FIELDS:
        if field in ('destruction', 'episode_return'):
            same = abs(actual[field] - expected[field]) < 1e-10
        else:
            same = actual[field] == expected[field]
        require(same, f'Replay mismatch for {field}: expected {expected[field]!r}, got {actual[field]!r}. Stop; do not change the reference result.')


def record(case, model, base):
    expected = case['expected']
    seed = episode_seed(expected['sample_seed'], base, expected['view'])
    require(seed == expected['acting_seed'], 'Acting-seed derivation differs')
    E.configure_cpu()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = SymmetryWrapper(InspectionEnv(motifs=C.DEV_MOTIFS, record=True, record_ticks=True), transform=expected['view'])
    try:
        replay = play_episode(env, ActingPolicy(model.policy, expected['mode'] == 'deterministic'), options={'spec': base})
    finally:
        env.close()
    validate_summary(replay, expected)
    replay['inspection_case'] = case
    replay['acting'] = {'mode': expected['mode'], 'view': expected['view'], 'sample_seed': expected['sample_seed'],
                        'rng_seed': seed, 'display_coordinates': 'original base orientation'}
    replay['policy'] = {'name': f"{expected['condition']}-{expected['learner_seed']} | {expected['transitions']:,} transitions | {expected['mode']} | {case['id']}"}
    replay['verification'] = 'All recorded scalar evaluation metrics match; original action sequences were not stored in the evaluation CSV.'
    return replay


def diagnostics(replay):
    expected = replay['inspection_case']['expected']
    history = replay['snapshots']
    require(len(history) == len(replay['actions'])+1, 'Unexpected decision snapshot count')
    purchases, deployed = Counter(), Counter()
    timeline = []
    for step, (view_action, before, after) in enumerate(zip(replay['actions'], history, history[1:]), 1):
        action = after['last_action']  # Original-base coordinates, already inverse transformed.
        if before['phase'] == 'shopping' and 0 <= action < 4:
            purchases[C.TROOP_TYPES[action]] += 1
        if before['phase'] == 'battle' and C.ACTION_DEPLOY_FIRST <= action <= C.ACTION_DEPLOY_LAST:
            deployed[C.TROOP_TYPES[(action-C.ACTION_DEPLOY_FIRST)//C.N_DEPLOY_TILES]] += 1
        timeline.append({'decision': step, 'tick_before': before['tick'], 'tick_after': after['tick'],
                         'view_action': view_action, 'base_action': action, 'action': describe_action(action),
                         'destruction_percent_after': after['destruction']*100})
    # Keep the first actual post-tick frame for each positive tick, avoiding
    # duplicate decision-boundary and next-deployment snapshots at that tick.
    ticks = {}
    for frame in replay['visual_frames']:
        if frame['tick'] > 0:
            ticks.setdefault(frame['tick'], frame)
    no_goals, flagged_ticks, flagged_ids = 0, set(), set()
    for tick, frame in ticks.items():
        for troop in frame['troops']:
            if troop['inspection_target_alive'] and not troop['inspection_target_in_range'] and troop['inspection_approach_goals'] == 0:
                no_goals += 1
                flagged_ticks.add(tick)
                flagged_ids.add(troop['entity_id'])
    final = history[-1]
    remaining = Counter(b['type'] for b in final['buildings'] if b['type'] != 'wall')
    deploy_ticks = [r['tick_before'] for r in timeline if r['action'].startswith('Deploy ')]
    return {'case_id': replay['inspection_case']['id'], **expected,
            'roles': replay['inspection_case']['roles'],
            'purchased_army': {t: purchases[t] for t in C.TROOP_TYPES},
            'deployed_army': {t: deployed[t] for t in C.TROOP_TYPES},
            'reserve_at_end': final['reserve'], 'alive_at_end': final['troops_alive'],
            'remaining_nonwall_buildings': dict(sorted(remaining.items())),
            'first_deployment_tick': min(deploy_ticks) if deploy_ticks else None,
            'last_deployment_tick': max(deploy_ticks) if deploy_ticks else None,
            'post_tick_troop_observations_no_approach_goal': no_goals,
            'ticks_with_no_approach_goal_observation': len(flagged_ticks),
            'troops_with_no_approach_goal_observation': len(flagged_ids),
            'annotation_limit': 'Post-tick observations only. A missing approach goal is a diagnostic flag, not proof that it caused failure. No path search, retargeting or fallback was added.',
            'timeline': timeline, 'metrics_match': True}


STYLE = '<style>body{font:16px system-ui;margin:32px auto;max-width:1200px;padding:0 20px;color:#182536;background:#f5f7fa}table{border-collapse:collapse;width:100%;background:white;margin:18px 0}th,td{padding:10px;border:1px solid #d5dce5;text-align:left}th{background:#e7edf5}a{color:#164eaf}code{white-space:pre-wrap}p{line-height:1.6}.scroll{overflow:auto}</style>'


def document(title, content):
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>'+html.escape(title)+'</title>'+STYLE+'<body><h1>'+html.escape(title)+'</h1>'+content+'</body></html>').encode()


def details_html(d):
    content = '<p><a href="../index.html">All cases</a> · <a href="'+d['case_id']+'.html">Watch this replay</a></p>'
    content += f"<p>{html.escape(d['condition'])}-{d['learner_seed']} · {html.escape(d['mode'])} · {100*d['destruction']:.2f}% building value destroyed · {html.escape(d['finish_reason'])}</p>"
    content += '<p>Base '+str(d['base_seed'])+' / '+html.escape(d['motif'])+'; view '+str(d['view'])+'. Deployment coordinates below use the original base orientation.</p>'
    content += '<table><tr><th>Troop</th><th>Purchased</th><th>Deployed</th><th>Reserve at end</th></tr>'
    for t in C.TROOP_TYPES:
        content += f'<tr><td>{html.escape(t)}</td><td>{d["purchased_army"][t]}</td><td>{d["deployed_army"][t]}</td><td>{d["reserve_at_end"][t]}</td></tr>'
    content += '</table><p>Remaining non-wall buildings: '+html.escape(json.dumps(d['remaining_nonwall_buildings']))+'. Surviving troops: '+str(d['alive_at_end'])+'.</p>'
    content += '<p>Out-of-range troops with no approach goal were observed in '+str(d['ticks_with_no_approach_goal_observation'])+' post-tick frames. '+html.escape(d['annotation_limit'])+'</p>'
    content += '<h2>Decision timeline</h2><table><tr><th>Decision</th><th>Tick before</th><th>Action</th><th>Tick after</th><th>Destruction after</th></tr>'
    for row in d['timeline']:
        content += f'<tr><td>{row["decision"]}</td><td>{row["tick_before"]}</td><td>{html.escape(row["action"])}</td><td>{row["tick_after"]}</td><td>{row["destruction_percent_after"]:.2f}%</td></tr>'
    return document(d['case_id']+' attack details', content+'</table>')


def index_html(plan, summaries):
    by_id = {r['case_id']: r for r in summaries}
    text = '<p>Selected development attacks from existing 4,194,304-transition checkpoints. Every replay matches its recorded scalar evaluation results. This collection deliberately includes failures and successes; do not use its average as a performance score.</p>'
    text += '<p>Open a replay, press Play, and use the timeline, speed, Ranges and Paths controls. Details show purchases, deployment timing and the remaining army/buildings. Both replays in a pair use the same learner, layout and view.</p>'
    text += '<div class="scroll"><table><tr><th>Learner</th><th>Selection reason</th><th>Layout / view</th><th>Sampled attack</th><th>Deterministic counterpart</th></tr>'
    for pair in plan['pairs']:
        a, b = by_id[pair['sampled_id']], by_id[pair['deterministic_id']]
        text += '<tr><td>'+html.escape(pair['condition'])+'-'+str(pair['learner_seed'])+'</td><td>'+html.escape(pair['selection_role'])+'</td><td>'+html.escape(a['motif'])+' / '+str(a['base_seed'])+' / '+str(a['view'])+'</td>'
        for d in (a, b):
            text += '<td><a href="cases/'+d['case_id']+'.html">Watch: '+f"{100*d['destruction']:.2f}%"+'</a><br>'+html.escape(d['finish_reason'])+'<br><a href="cases/'+d['case_id']+'.details.html">Army and decisions</a></td>'
        text += '</tr>'
    return document('Base Siege V2 replay inspection', text+'</table></div>')


def csv_bytes(rows):
    text = io.StringIO(newline='')
    fields = sorted(k for k in rows[0] if k not in ('timeline', 'annotation_limit'))
    writer = csv.DictWriter(text, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(row[k], sort_keys=True) if isinstance(row[k], (dict, list)) else row[k] for k in fields})
    return text.getvalue().encode()


def execute(contract, plan, pool, output=OUTPUT, root=ROOT, max_cases=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    began = perf_counter()
    with run_lock(output):
        publish_json(output/'inspection_contract.json', contract)
        publish_json(output/'selection_plan.json', plan)
        (output/'cases').mkdir(exist_ok=True)
        models = {(m['condition'], m['learner_seed']): m for m in contract['models']}
        bases = {(b.seed, b.motif): b for b in pool}
        loaded, summaries = {}, []
        done = 0
        for case in plan['cases']:
            name = case['id']
            receipt_path = output/'cases'/f'{name}.receipt.json'
            if receipt_path.exists():
                receipt = read(receipt_path)
                require(receipt['case'] == case, 'Existing replay case changed')
                for filename, digest in receipt['files'].items():
                    require(E.sha256(output/'cases'/filename) == digest, 'Existing replay output changed: '+filename)
                summaries.append(read(output/'cases'/f'{name}.diagnostic.json'))
                continue
            if max_cases is not None and done >= max_cases:
                print(f'Paused after {done} new cases. Run the same command without --max-new-cases to continue.', flush=True)
                return {'status': 'paused', 'verified_cases': len(summaries)}
            expected = case['expected']
            model_key = expected['condition'], expected['learner_seed']
            if model_key not in loaded:
                loaded[model_key] = E.load_target(models[model_key], root)
            replay = record(case, loaded[model_key], bases[expected['base_seed'], expected['motif']])
            diagnostic = diagnostics(replay)
            publish_json(output/'cases'/f'{name}.json', replay)
            publish_json(output/'cases'/f'{name}.diagnostic.json', diagnostic)
            with tempfile.TemporaryDirectory() as temp:
                rendered = Path(temp)/'replay.html'
                export_html(replay, rendered)
                publish_bytes(output/'cases'/f'{name}.html', rendered.read_bytes())
            publish_bytes(output/'cases'/f'{name}.details.html', details_html(diagnostic))
            files = [f'{name}.json', f'{name}.diagnostic.json', f'{name}.html', f'{name}.details.html']
            publish_json(receipt_path, {'case': case, 'files': {n: E.sha256(output/'cases'/n) for n in files}})
            summaries.append(diagnostic)
            done += 1
            print(f'{len(summaries)}/{len(plan["cases"])} verified: {name}, {expected["condition"]}-{expected["learner_seed"]}, {expected["mode"]}, {100*expected["destruction"]:.2f}% destruction', flush=True)
        publish_bytes(output/'diagnostics.csv', csv_bytes(summaries))
        publish_bytes(output/'index.html', index_html(plan, summaries))
        exported = [p for p in sorted(output.rglob('*')) if p.is_file() and p.name not in ('.run.lock', 'complete.json')]
        receipt = {'status': 'completed', 'verified_cases': len(summaries), 'pairs': len(plan['pairs']),
                   'scope': plan['scope'], 'files': {str(p.relative_to(output)): E.sha256(p) for p in exported}}
        publish_json(output/'complete.json', receipt)
        print(f'Completed. New replays this attempt: {done}; elapsed {perf_counter()-began:.1f} seconds. Open {output/"index.html"}', flush=True)
        return receipt


def pack(output=OUTPUT):
    output = Path(output)
    receipt = read(output/'complete.json')
    require(receipt['status'] == 'completed', 'Replay inspection is not complete')
    for name, digest in receipt['files'].items():
        require(E.sha256(output/name) == digest, 'Output checksum mismatch: '+name)
    target = output.parent/'Base_Siege_V2_Replay_Inspection_Results.zip'
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted([*receipt['files'], 'complete.json']):
            # Fixed timestamps make repeat packaging idempotent.
            info = zipfile.ZipInfo('version2-replay-inspection/'+name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (output/name).read_bytes())
    publish_bytes(target, buffer.getvalue())
    print(target)
    return target

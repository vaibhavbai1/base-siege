#!/usr/bin/env python3
"""Record a sampled trained-agent attack and export the existing smooth HTML player.

Run from the unchanged Base Siege project root:
    uv run python watch_trained_agent.py
    uv run python watch_trained_agent.py --condition PA

Defaults: learner 0, first accepted closed-ring development base, view 0,
action seed 0. These are fixed illustrative choices, not selected for a high score.
The actual outcome must match its saved sampled-development episode.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random

HELPER_SHA = '420e07d5a741b42be1afd0993c0f5fd7c5c190de02540a9a7b306ea715808261'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def record(api, helper, model, base, view, action_seed):
    seed = helper.episode_seed(base, view, action_seed)
    random.seed(seed)
    api.np.random.seed(seed)
    api.torch.manual_seed(seed)
    env = api.SymmetryWrapper(api.SiegeEnv(motifs=api.C.TRAIN_MOTIFS,
        fixed_army=False, record=True, record_ticks=True), transform=view)
    try:
        return api.play_episode(env, helper.ActingPolicy(model, False), seed=0, options={'spec': base})
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--condition', choices=['P','PA'], default='P')
    p.add_argument('--learner-seed', type=int, choices=range(10), default=0)
    p.add_argument('--motif', choices=['closed_ring','ns_gates','ew_gates'], default='closed_ring')
    p.add_argument('--base-seed', type=int, help='An accepted seed in the existing development pool.')
    p.add_argument('--view', type=int, choices=range(8), default=0)
    p.add_argument('--action-seed', type=int, choices=range(3), default=0)
    args = p.parse_args()
    root = Path.cwd()
    require((root/'train/evaluate.py').is_file(), 'Run this from your Base Siege project root.')
    helper_path = root/'final_evaluation_bundle/diagnose_policy_sampling.py'
    require(helper_path.is_file() and digest(helper_path)==HELPER_SHA,
            'Keep the original final_evaluation_bundle in the project root.')
    spec = importlib.util.spec_from_file_location('watch_sampling_helper', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    api = helper.load_api(root)
    api.torch.set_num_threads(1)
    api.torch.use_deterministic_algorithms(True)
    from tools.export_replay_html import export_html
    from experiments.storage import publish_bytes, publish_json

    run = root/f'runs/study-v2/{args.condition}-{args.learner_seed}'
    manifest = api.inspect_run(run)
    runtime = json.loads(json.dumps(api.runtime_manifest()))
    for key in ('source_hashes','config','python','packages'):
        require(manifest[key]==runtime[key], f'{key} differs from training; restore the original environment.')
    pool_path = root/'bases/dev-v2.json'
    require(manifest['pools']['dev']['sha256']==digest(pool_path), 'Development pool changed.')
    bases = sorted(api.load_pool(pool_path, required_split='dev'), key=lambda b:(b.seed,b.motif))
    choices = [b for b in bases if b.motif==args.motif and (args.base_seed is None or b.seed==args.base_seed)]
    require(bool(choices), 'No accepted development base matches these options. Use an accepted development seed.')
    base = choices[0]

    directory = root/'artifacts/sampled-dev-v1'
    receipt = json.loads((directory/'sampled_complete.json').read_text())
    require(receipt['status']=='completed', 'Sampled development evaluation is incomplete.')
    require(digest(directory/'sampled_contract.json')==receipt['contract_sha256'], 'Development contract checksum mismatch.')
    contract = json.loads((directory/'sampled_contract.json').read_text())
    entry = next(m for m in contract['models'] if m['condition']==args.condition and m['learner_seed']==args.learner_seed)
    require(entry['model_sha256']==manifest['model_sha256'] and entry['manifest_sha256']==digest(run/'manifest.json'),
            'Model differs from the completed sampled development evaluation.')
    require(digest(directory/'sampled_episodes.csv')==receipt['episodes_sha256'], 'Development episode checksum mismatch.')
    with (directory/'sampled_episodes.csv').open(newline='') as handle:
        matches = [r for r in csv.DictReader(handle) if r['condition']==args.condition and
                   int(r['learner_seed'])==args.learner_seed and int(r['base_seed'])==base.seed and
                   r['motif']==base.motif and int(r['view'])==args.view and int(r['action_seed'])==args.action_seed]
    require(len(matches)==1, 'Expected exactly one matching saved development episode.')
    expected = matches[0]
    print(f'Recording {run.name}, {base.motif}, base {base.seed}, view {args.view}, action seed {args.action_seed}...',flush=True)
    model, _ = api.load_run(run)
    replay = record(api, helper, model, base, args.view, args.action_seed)
    summary = replay['summary']
    for field, target in [('destruction','destruction'),('town_hall','town_hall'),('full_clear','full_clear'),
                          ('episode_return','return'),('spent','spent'),('tick','ticks')]:
        require(math.isclose(float(summary[field]),float(expected[target]),rel_tol=0,abs_tol=1e-9),
                f'Recorded result differs from saved evaluation: {target}. No replay exported.')
    require(summary['finish_reason']==expected['finish_reason'], 'Recorded finish reason differs.')
    require(bool(replay.get('visual_frames')), 'Smooth simulator frames are missing.')
    replay['policy'] = {'name':f'Trained {args.condition} PPO — sampled actions — learner {args.learner_seed}',
        'model':str(run.relative_to(root))+'/model.zip','model_sha256':manifest['model_sha256'],
        'action_seed':args.action_seed,'view':args.view,'rng_seed':helper.episode_seed(base,args.view,args.action_seed),
        'evidence_scope':'Illustrative development replay; not a new final evaluation',
        'reference_match':True}
    output = root/'artifacts/demo'
    output.mkdir(parents=True,exist_ok=True)
    stem = f'{run.name}-{base.motif}-{base.seed}-view{args.view}-action{args.action_seed}'
    replay_path = output/(stem+'-replay.json')
    html_path = output/(stem+'.html')
    publish_json(replay_path,replay)
    # Render a temporary HTML file, then publish without replacing different output.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        temporary = Path(tmp)/'replay.html'
        export_html(replay,temporary)
        publish_bytes(html_path,temporary.read_bytes())
    print(f"Destruction: {summary['destruction']:.2%} | Town Hall: {bool(summary['town_hall'])} | Full clear: {bool(summary['full_clear'])}")
    print('Result matches the saved development evaluation.')
    print(f'HTML: {html_path.relative_to(root)}')
    print(f'Replay JSON: {replay_path.relative_to(root)}')
    print('Download the HTML file, open it in Chrome or Edge, and press Play.')


if __name__=='__main__':
    main()

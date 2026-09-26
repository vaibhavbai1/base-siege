"""Export one Python-recorded episode to an offline, animated HTML player.

Example: python -m tools.export_replay_html --out demo_preview.html
The default is a labelled heuristic, not a learned policy. Use --replay for
an episode downloaded from the app, including a saved PPO policy episode.
"""
import argparse
import hashlib
import json
from pathlib import Path
from env import config as C
from env.siege_env import SiegeEnv
from agents.heuristic import HeuristicCommander
from experiments.rollout import play_episode, load_replay

ROOT=Path(__file__).resolve().parents[1]


def export_html(replay,output):
    frontend=ROOT/'ui/frontend'
    payload={'snapshot':replay['snapshots'][0], 'troop_types':C.TROOP_TYPES,
        'troops':C.TROOP_STATS,'buildings':C.BUILDING_STATS,'readonly':True,'log':[],
        'replay':{'frames':replay.get('visual_frames') or replay['snapshots'],
                  'tick_recorded':bool(replay.get('visual_frames')),
                  'id':hashlib.sha256(json.dumps(replay,sort_keys=True).encode()).hexdigest(),
                  'label':replay.get('policy',{}).get('name','Recorded attack')}}
    page=(frontend/'index.html').read_text()
    for filename in ('theme.css','playback.css'):
        page=page.replace(f'<link rel="stylesheet" href="{filename}">','<style>'+(frontend/filename).read_text()+'</style>')
    page=page.replace('<main class="wrap">','<main class="wrap"><h1>Base Siege</h1><p class="sub">Recorded attack · press Play to watch</p>')
    for filename in ('glyphs.js','manual_engine.js','app.js'):
        page=page.replace(f'<script src="{filename}"></script>','<script>'+(frontend/filename).read_text().replace('</script','<\\/script')+'</script>')
    # Escape HTML-significant characters in JSON, including any replay labels.
    encoded=json.dumps(payload).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    page=page.replace('</body>',f'<script>window.postMessage({{type:"streamlit:render",args:{{payload:{encoded}}}}},"*");</script></body>')
    Path(output).write_text(page)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Choose a new output file')
    if args.replay:replay=load_replay(args.replay)
    else:
        replay=play_episode(SiegeEnv(base_seed=42,motifs=('closed_ring',),record=True,record_ticks=True),HeuristicCommander())
        replay['policy']={'name':'Heuristic baseline — not a trained agent'}
    export_html(replay,args.out)
    print(args.out)


if __name__=='__main__':main()

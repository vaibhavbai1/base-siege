"""Frozen copy of the existing offline replay exporter, with local assets."""
import hashlib
import json
from pathlib import Path
from env import config as C

def export_html(replay,output):
    frontend=Path(__file__).resolve().parent/'frontend'
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


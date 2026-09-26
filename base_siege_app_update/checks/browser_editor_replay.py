from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
from playwright.sync_api import sync_playwright,expect
from env import config as C
from env.generator import generate_base
from env.siege_env import SiegeEnv
from agents.heuristic import HeuristicCommander
from experiments.rollout import play_episode
from tools.export_replay_html import export_html
from ui.custom_bases import layout_from_spec,custom_spec
ROOT=Path.cwd()
OUTPUT=ROOT/'artifacts/ui-v3-checks'
OUTPUT.mkdir(parents=True,exist_ok=True)
layout=layout_from_spec(generate_base(42,'layered_core'))
replay=play_episode(SiegeEnv(record=True,record_ticks=True),HeuristicCommander(),options={'spec':custom_spec(layout)})
replay['policy']={'name':'Custom-base heuristic test'}
export_html(replay,Path(str(OUTPUT/'custom-replay-test.html')))
with sync_playwright() as p:
    b=p.chromium.launch(headless=True,args=['--no-sandbox','--no-zygote','--single-process','--disable-gpu'])
    page=b.new_page(viewport={'width':1100,'height':950});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto((ROOT/'ui/editor/index.html').as_uri())
    page.evaluate("p=>window.postMessage({type:'streamlit:render',args:{payload:p}},'*')",{'layout':layout,'counts':C.BUILDING_COUNTS,'version':1})
    expect(page.locator('#save')).to_be_enabled()
    page.locator('summary').click()
    original=layout['buildings'][0]
    free=next((r,c) for r in range(1,15) for c in range(1,15) if (r,c) not in custom_spec(layout).occupied)
    for r,c in (original[1:],free):
        page.locator('#row').fill(str(r));page.locator('#col').fill(str(c));page.locator('#place').click()
    assert [original[0],*free] in page.evaluate('draft.buildings')
    page.locator('#undo').click();assert original in page.evaluate('draft.buildings')
    page.locator('[data-tool=wall]').click();before=page.evaluate('draft.walls.length')
    page.locator('#cv').click(position={'x':20,'y':20});assert page.evaluate('draft.walls.length')==before
    page.locator('#clear-walls').click()
    box=page.locator('#cv').bounding_box()
    page.mouse.move(box['x']+box['width']*1.5/16,box['y']+box['height']*1.5/16)
    page.mouse.down();page.mouse.move(box['x']+box['width']*4.5/16,box['y']+box['height']*1.5/16,steps=9);page.mouse.up()
    assert page.evaluate('draft.walls.length')==4
    page.goto(Path(str(OUTPUT/'custom-replay-test.html')).resolve().as_uri())
    expect(page.locator('#btn-play')).to_be_visible()
    page.locator('#btn-play').click();page.wait_for_function('cursor>100')
    page.locator('#btn-play').click();assert not page.evaluate('playing')
    page.locator('#timeline').fill('1000');page.locator('#timeline').dispatch_event('input')
    assert page.evaluate('shown.finish_reason')
    expect(page.locator('#result-banner')).to_be_visible()
    page.screenshot(path=str(OUTPUT/'replay-preview.png'),full_page=True)
    assert not errors,errors
    print('Browser: move/undo, perimeter guard, drag painting and custom replay play/pause/seek/end passed.')
    b.close()

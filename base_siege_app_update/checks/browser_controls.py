import copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from playwright.sync_api import sync_playwright, expect
from ui.manual_game import ManualGame
from ui.custom_bases import layout_from_spec, custom_spec
from env.generator import generate_base
from env import config as C
from tests.test_browser_manual_parity import manual_payload
ROOT=Path.cwd()
OUTPUT=ROOT/'artifacts/ui-v3-checks'
OUTPUT.mkdir(parents=True,exist_ok=True)
game=ManualGame(base_seed=42,motifs=('layered_core',),record=True,record_ticks=True);game.reset(seed=0)
ack=None;logs=[];start=0;gamekey='browser-1';commands=[]
def payload():
    return {**manual_payload(game),'readonly':False,'game_key':gamekey,'ack_id':None,'log':[],'error':''}
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox','--no-zygote','--single-process','--disable-gpu'])
    page=browser.new_page(viewport={'width':1100,'height':1000})
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.add_init_script("window.manualRequests=[];window.addEventListener('message',e=>{if(e.data.type==='streamlit:setComponentValue')window.manualRequests.push(e.data.value)});")
    page.goto((ROOT/'ui/frontend/index.html').as_uri())
    page.evaluate("p=>window.postMessage({type:'streamlit:render',args:{payload:p}},'*')",payload())
    expect(page.locator('#btn-main')).to_be_disabled()
    for _ in range(7):page.get_by_role('button',name='Buy melee',exact=True).click()
    expect(page.locator('#s-budget')).to_have_text(str(200-7*C.TROOP_COSTS['melee']))
    page.get_by_role('button',name='Buy melee',exact=True).click(modifiers=['Shift'])
    expect(page.locator('#s-budget')).to_have_text(str(200-6*C.TROOP_COSTS['melee']))
    page.locator('#btn-main').click()
    expect(page.locator('#btn-main')).to_have_text('Run the clock')
    page.locator('#cv').click(position={'x':255,'y':17},modifiers=['Shift'])
    page.wait_for_function('payload.snapshot.troops_total===5')
    assert page.evaluate("manual.tick")==0 and page.evaluate("manual.total")==5
    page.locator('#speed').select_option('0.4')
    page.locator('#cv').focus();page.keyboard.press('Space')
    page.wait_for_function('payload.snapshot.tick>0')
    # A deployment is accepted while clock animations are running.
    page.locator('#cv').click(position={'x':280,'y':17})
    page.wait_for_function('payload.snapshot.troops_total===6')
    page.wait_for_function('payload.snapshot.tick===10 && !running && !draining',timeout=15000)
    assert page.evaluate("manual.tick")==10 and page.evaluate("manual.total")==6
    page.wait_for_timeout(300);assert page.evaluate("manual.tick")==10
    page.locator('#stepmode').uncheck()
    page.wait_for_timeout(250);assert page.evaluate("manual.tick")==10 # Checkbox does not start clock.
    page.locator('#cv').focus();page.keyboard.press('Digit1')
    expected=C.TROOP_TYPES[0];assert page.evaluate('selected')==expected
    page.keyboard.press('Space');page.wait_for_function('payload.snapshot.tick>=13')
    page.keyboard.press('Space');page.wait_for_function('!running && !draining')
    paused=page.evaluate("manual.tick");page.wait_for_timeout(350);assert page.evaluate("manual.tick")==paused
    assert page.evaluate('manualRequests.length')==0
    page.screenshot(path=str(OUTPUT/'manual-v3-preview.png'),full_page=True)
    assert not errors,errors
    print('Browser-local manual controls (zero server tick/action requests): buy, refund, batch deployment, running deployment, 10-tick pause, continuous clock, Space, troop keys passed.')
    editor=page
    editor.set_viewport_size({'width':1100,'height':900})
    editor.on('pageerror',lambda e:errors.append(str(e)))
    editor.goto((ROOT/'ui/editor/index.html').as_uri())
    data=layout_from_spec(generate_base(42,'layered_core'))
    editor.evaluate("p=>window.postMessage({type:'streamlit:render',args:{payload:p}},'*')",{'layout':data,'version':1,'counts':C.BUILDING_COUNTS})
    expect(editor.locator('#save')).to_be_enabled()
    editor.locator('#clear-walls').click()
    assert editor.evaluate('draft.walls.length')==0
    editor.locator('#undo').click()
    assert editor.evaluate('draft.walls.length')==len(data['walls'])
    editor.locator('#clear-all').click();expect(editor.locator('#save')).to_be_disabled()
    # Rebuild from a completely blank board via keyboard placement controls.
    editor.locator('summary').click()
    for i,(kind,r,c) in enumerate(data['buildings']):
        editor.locator(f'[data-tool={kind}]').click()
        editor.locator('#row').fill(str(r));editor.locator('#col').fill(str(c));editor.locator('#place').click()
    expect(editor.locator('#save')).to_be_enabled()
    occupied={(r,c) for _,r,c in data['buildings']}
    free=next((r,c) for r in range(1,15) for c in range(1,15) if (r,c) not in occupied)
    editor.locator('[data-tool=wall]').click()
    editor.locator('#row').fill(str(free[0]));editor.locator('#col').fill(str(free[1]));editor.locator('#place').click()
    assert editor.evaluate('draft.walls.length')==1
    editor.locator('#name').fill('My layered fortress')
    editor.evaluate("window.saved=[];window.addEventListener('message',e=>{if(e.data.type==='streamlit:setComponentValue')window.saved.push(e.data.value)})")
    editor.locator('#manual').click();editor.wait_for_function('saved.length===1')
    saved=editor.evaluate('saved[0]');assert saved['target']=='Manual play'
    spec=custom_spec(saved['layout']);assert spec.motif=='custom'
    editor.screenshot(path=str(OUTPUT/'editor-v3-preview.png'),full_page=True)
    assert not errors,errors
    print('Editor browser: clear, undo, rebuild all buildings, paint wall, rename and submit a validated custom base passed.')
    browser.close()

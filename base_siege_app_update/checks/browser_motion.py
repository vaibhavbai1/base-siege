from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd()))
from playwright.sync_api import sync_playwright,expect
from tests.test_browser_manual_parity import manual_payload
from ui.manual_game import ManualGame
from env.generator import generate_base,BaseSpec
ROOT=Path.cwd()
OUTPUT=ROOT/'artifacts/ui-v3-checks'
OUTPUT.mkdir(parents=True,exist_ok=True)
base=generate_base(42,'layered_core');base=BaseSpec(0,'custom',base.buildings,())
game=ManualGame();game.reset(seed=0,options={'spec':base})
payload={**manual_payload(game),'readonly':False,'game_key':'motion-test','log':[],'error':''}
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox','--no-zygote','--single-process','--disable-gpu'])
    page=browser.new_page(viewport={'width':1100,'height':1050});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto((ROOT/'ui/frontend/index.html').as_uri())
    page.evaluate("p=>{window.requests=[];window.addEventListener('message',e=>{if(e.data.type==='streamlit:setComponentValue')requests.push(e.data)});window.postMessage({type:'streamlit:render',args:{payload:p}},'*')}",payload)
    page.get_by_role('button',name='Buy tank',exact=True).click()
    page.locator('#btn-main').click();page.locator('#cv').click(position={'x':255,'y':17})
    page.evaluate("""()=>{window.motion=[];function sample(now){
      const t=shown?.troops.find(t=>t.entity_id===1);
      if(t){const a=displayPositions[index]?.get(1)||t,b=displayPositions[index+1]?.get(1)||a;
        motion.push({time:now,row:a.row+(b.row-a.row)*blend,col:a.col+(b.col-a.col)*blend,tick:payload.snapshot.tick});}
      requestAnimationFrame(sample);
    }requestAnimationFrame(sample)}""")
    page.locator('#cv').focus();page.keyboard.press('Space')
    page.wait_for_function('manual.tick===10&&!running&&!draining',timeout=5000)
    samples=page.evaluate('motion');unique={(round(s['row'],4),round(s['col'],4)) for s in samples}
    jumps=[((b['row']-a['row'])**2+(b['col']-a['col'])**2)**.5 for a,b in zip(samples,samples[1:])]
    assert len(unique)>15,(len(unique),samples)
    assert max(jumps)<.25,max(jumps)
    assert page.evaluate('requests.length')==0
    # Repeated server renders cannot rewind the locally played battle.
    page.evaluate("p=>window.postMessage({type:'streamlit:render',args:{payload:p}},'*')",payload)
    page.wait_for_timeout(80);assert page.evaluate('manual.tick')==10
    # Pause freezes fractional position; resume continues without a tile jump.
    page.keyboard.press('Space');page.wait_for_function('manual.tick>=13');page.keyboard.press('Space')
    paused=page.evaluate('[manual.tick,blend]');page.wait_for_timeout(150);assert page.evaluate('[manual.tick,blend]')==paused
    assert not errors,errors
    page.screenshot(path=str(OUTPUT/'manual-v3-motion.png'),full_page=True)
    print(json.dumps({'samples':len(samples),'distinct_positions_in_10_ticks':len(unique),'largest_frame_jump_tiles':round(max(jumps),4),'server_action_requests':page.evaluate('requests.length'),'errors':errors}))
    browser.close()

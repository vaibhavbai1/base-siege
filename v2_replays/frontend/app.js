/* Agent replays use Python-recorded frames. Manual play runs locally with
   the parity-tested browser engine; rendering never feeds back into combat. */
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
const GRID=16,CELL=cv.width/GRID,$=id=>document.getElementById(id);
const colours={town_hall:'#6C63C8',cannon:'#C74E77',mortar:'#CE5A2E',storage:'#7A8B8E',wall:'#6b6560',tank:'#1E8F6E',melee:'#3179C6',ranged:'#5E8F22',breaker:'#D99425'};
let payload=null,selected='melee',pending=false,sequence=0,key='',manualKey='';
let frames=[],times=[],cursor=0,index=0,playing=false,lastTime=0,shown=null,visualNext=null,blend=0,uiStamp='';
let running=false,chunk=0,manual=null,manualElapsed=0,manualDuration=60,draining=false;
let armyStamp='';
let effects=[],lastHitTick=-1,lastHeight=0,hover=null,displayPositions=[];
const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
const perimeter=[];
for(let c=0;c<16;c++)perimeter.push([0,c]);for(let r=1;r<16;r++)perimeter.push([r,15]);
for(let c=14;c>=0;c--)perimeter.push([15,c]);for(let r=14;r>=1;r--)perimeter.push([r,0]);
function send(type,fields={}){parent.postMessage({isStreamlitMessage:true,type,...fields},'*');}
// Human actions and ticks are local. No Streamlit round-trip is on this path.
function command(kind,fields={}){
  if(!payload||payload.readonly||!manual)return;
  try{
    if(kind==='reset'){
      running=false;draining=false;chunk=0;manualElapsed=0;effects=[];lastHitTick=-1;
      manual.command(kind);payload.snapshot=manual.snapshot();payload.log=[];
      frames=[payload.snapshot];shown=visualNext=payload.snapshot;displayPositions=[manual.visualPositions()];index=0;blend=0;
    }else{
      manual.command(kind,fields);payload.snapshot=manual.snapshot();payload.log=manual.log;
      // Keep in-progress interpolation for existing troops when a new unit is
      // deployed. New troops and reserve changes appear immediately.
      const known=new Set((shown?.troops||[]).map(t=>t.entity_id));
      if(frames.length>1){
        const spawned=payload.snapshot.troops.filter(t=>!known.has(t.entity_id));
        shown={...shown,phase:payload.snapshot.phase,reserve:payload.snapshot.reserve,spent:payload.snapshot.spent,troops:[...shown.troops,...spawned]};
        frames[0]=shown;frames[1]=payload.snapshot;visualNext=payload.snapshot;
        for(const t of spawned)displayPositions[0].set(t.entity_id,{row:t.row,col:t.col});
        displayPositions[1]=manual.visualPositions();
      }else{shown=visualNext=payload.snapshot;frames=[shown];displayPositions=[manual.visualPositions()];}
    }
    payload.error='';syncUI(true);
  }catch(e){payload.error=e.message;syncUI(true);}
}
function tickMillis(){return {'0.2':220,'0.4':120,'1':60,'2':26,'5':10}[$('speed').value]||60;}
function advanceManual(){
  const previous=manual.snapshot(),positions=manual.visualPositions();
  manual.command('tick');payload.snapshot=manual.snapshot();payload.log=manual.log;
  frames=[previous,payload.snapshot];displayPositions=[positions,manual.visualPositions()];index=0;
  shown=previous;visualNext=payload.snapshot;blend=0;manualDuration=tickMillis();
  emit(payload.snapshot,1);chunk++;
  if(payload.snapshot.finish_reason||($('stepmode').checked&&chunk>=10)){running=false;draining=true;chunk=0;}
  syncUI(true);
}
function deploy(i,count=1){
  if(!payload||payload.readonly||payload.snapshot.phase!=='battle'||payload.snapshot.finish_reason)return;
  if(Number.isInteger(i)&&i>=0&&i<60&&payload.snapshot.reserve[selected]>0)
    command('deploy',{troop:selected,tile:i,count});
}
function mainControl(){
  if(!payload||payload.readonly||payload.snapshot.finish_reason)return;
  if(payload.snapshot.phase==='shopping'){
    if(Object.values(payload.snapshot.reserve).some(n=>n>0))command('start');
  }else{
    running=!running;
    draining=false;
    if(running&&(frames.length<2||manualElapsed>=manualDuration)){manualElapsed=0;advanceManual();}
    syncUI(true);
  }
}
function total(){return times.length?times[times.length-1]:0;}
function clock(ms){const s=Math.floor(ms/1000);return `${Math.floor(s/60)}:${String(s%60).padStart(2,'0')}`;}
function height(){const h=Math.ceil(document.querySelector('.wrap').getBoundingClientRect().height);if(h!==lastHeight){lastHeight=h;send('streamlit:setFrameHeight',{height:h});}}
function install(sequenceFrames,autoplay,keepEffects=false){
  frames=sequenceFrames;times=[0];
  for(let i=1;i<frames.length;i++){
    const dt=frames[i].tick-frames[i-1].tick;
    times.push(times[i-1]+(dt>0?dt*60:frames[i-1].phase==='shopping'?140:1));
  }
  // Look ahead only to movements that actually occurred. Never extrapolate
  // through walls or invent a route for a unit that stayed still.
  displayPositions=frames.map(()=>new Map());
  const future=new Map();
  for(let i=frames.length-1;i>=0;i--)for(const t of frames[i].troops){
    let f=future.get(t.entity_id);
    if(f&&f.row===t.row&&f.col===t.col)f=f.next;
    const fraction=t.movement_fraction||0;
    const adjacent=f&&Math.abs(f.row-t.row)+Math.abs(f.col-t.col)===1;
    const upcoming=adjacent&&f.tick-frames[i].tick<=Math.ceil(1/payload.troops[t.type].speed)+1;
    displayPositions[i].set(t.entity_id,{row:t.row+(upcoming?(f.row-t.row)*fraction:0),col:t.col+(upcoming?(f.col-t.col)*fraction:0)});
    future.set(t.entity_id,{row:t.row,col:t.col,tick:frames[i].tick,next:f});
  }
  cursor=0;index=0;
  if(!keepEffects){effects=[];lastHitTick=-1;lastTime=0;}
  playing=autoplay&&frames.length>1;uiStamp='';updatePosition(false);
}
function seek(value){cursor=Math.max(0,Math.min(total(),value));effects=[];lastHitTick=-1;updatePosition(false);syncUI(true);}
function updatePosition(animate){
  let next=0;while(next+1<times.length&&times[next+1]<=cursor)next++;
  if(animate){for(let i=index+1;i<=next;i++)emit(frames[i],i);}
  index=next;shown=frames[index]||payload?.snapshot;visualNext=frames[index+1]||shown;
  const duration=(times[index+1]||total())-(times[index]||0);
  blend=duration>0?(cursor-times[index])/duration:0;
  syncUI();
}
function emit(s,frameIndex){
  if(!s||s.tick===lastHitTick)return;lastHitTick=s.tick;
  const seen=new Set();
  for(const a of s.attacks||[]){
    const to=a.impact||a.to,token=`${a.kind}:${a.from}:${to}`;
    if(!seen.has(token)){
      seen.add(token);effects.push({kind:'shot',from:a.from,to,age:0,life:a.attacker_type==='mortar'?390:170,arc:a.attacker_type==='mortar',colour:a.kind==='troop'?'#f7da75':'#d76732'});
      effects.push({kind:'ring',to,age:0,life:340,colour:a.attacker_type==='mortar'?'#ec994e':'#f5d885'});
    }
    if($('damage').checked)effects.push({kind:'number',to:a.to,damage:a.damage,age:0,life:540});
  }
  const prior=frames[frameIndex-1];if(!prior)return;
  const alive=new Set(s.buildings.map(b=>`${b.row},${b.col}`));
  for(const b of prior.buildings)if(!alive.has(`${b.row},${b.col}`)){
    // Deterministic decorative debris; no game RNG is consumed.
    for(let j=0;j<7;j++)effects.push({kind:'debris',to:[b.row,b.col],angle:j*Math.PI*2/7,age:0,life:600,colour:colours[b.type]});
  }
  if(effects.length>420)effects=effects.slice(-420);
}
function draw(now){
  if(!shown||!payload)return;const s=shown,S=CELL,next=visualNext||s;
  ctx.clearRect(0,0,cv.width,cv.height);ctx.globalAlpha=1;
  for(let r=0;r<16;r++)for(let c=0;c<16;c++){const edge=r===0||c===0||r===15||c===15;ctx.fillStyle=edge?((r+c)%2?'#7fa65c':'#789e55'):((r+c)%2?'#cdbb96':'#c7b48d');ctx.fillRect(c*S,r*S,S,S);}
  if(hover&&!payload.readonly&&s.phase==='battle'){ctx.fillStyle='#ffffff44';ctx.fillRect(hover[1]*S,hover[0]*S,S,S);}
  if($('ranges').checked)for(const b of s.buildings){const stat=payload.buildings[b.type];if(!stat.range)continue;
    ctx.strokeStyle='#ce5a2e88';ctx.setLineDash([5,4]);ctx.lineWidth=1.4;ctx.beginPath();ctx.arc((b.col+.5)*S,(b.row+.5)*S,stat.range*S,0,Math.PI*2);ctx.stroke();
    if(stat.min_range){ctx.fillStyle='#ce5a2e22';ctx.beginPath();ctx.arc((b.col+.5)*S,(b.row+.5)*S,stat.min_range*S,0,Math.PI*2);ctx.fill();}}
  ctx.setLineDash([]);
  if($('paths').checked)for(const t of s.troops){if(!t.path)continue;ctx.strokeStyle='#29374677';ctx.lineWidth=1.5;ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo((t.col+.5)*S,(t.row+.5)*S);for(const [r,c]of t.path)ctx.lineTo((c+.5)*S,(r+.5)*S);ctx.stroke();}ctx.setLineDash([]);
  const nextBuildings=new Map(next.buildings.map(b=>[`${b.row},${b.col}`,b]));
  for(const b of s.buildings){const x=b.col*S,y=b.row*S,n=nextBuildings.get(`${b.row},${b.col}`);
    const shot=effects.find(e=>e.kind==='shot'&&e.from[0]===b.row&&e.from[1]===b.col);
    const aim=shot?Math.atan2(shot.to[0]-b.row,shot.to[1]-b.col):-Math.PI/4,recoil=shot?1-shot.age/shot.life:0;
    if(b.type==='wall')gWall(x,y,S);else{
      ctx.fillStyle='#00000016';ctx.beginPath();ctx.ellipse(x+S*.5,y+S*.90,S*.34,S*.11,0,0,Math.PI*2);ctx.fill();
      BGLYPH[b.type](x,y,S,colours[b.type],aim,recoil);
      const hp=n?b.hp+(n.hp-b.hp)*blend:b.hp;drawHP(x+S*.09,y+S*.02,S*.82,hp/b.max_hp);
    }
  }
  const nextTroops=new Map(next.troops.map(t=>[t.entity_id,t]));
  const canTween=(!payload.readonly||!reduced)&&(next.tick-s.tick)<=1;
  for(const t of s.troops){const n=nextTroops.get(t.entity_id)||t;
    const f=canTween?blend:0,p0=displayPositions[index]?.get(t.entity_id)||t,p1=displayPositions[index+1]?.get(t.entity_id)||p0,r=p0.row+(p1.row-p0.row)*f,c=p0.col+(p1.col-p0.col)*f;
    // Stable tiny offsets make stacked troops visible without changing recorded positions.
    const offset=((t.entity_id%3)-1)*S*.07,moving=p0.row!==p1.row||p0.col!==p1.col;
    const bob=moving&&!reduced?Math.sin(now/95+t.entity_id)*S*.025:0,x=(c+.5)*S+offset,y=(r+.5)*S+bob;
    const radius=S*(t.type==='tank'?.32:.26);
    ctx.fillStyle='#00000026';ctx.beginPath();ctx.ellipse(x,(r+.5)*S+radius*.9,radius*.9,radius*.32,0,0,Math.PI*2);ctx.fill();
    TGLYPH[t.type](x,y,radius,colours[t.type]);drawHP(x-radius,y-radius*1.3-S*.08,radius*2,(t.hp+(n.hp-t.hp)*f)/t.max_hp);
  }
  if(!reduced)for(const e of effects){const q=Math.min(1,e.age/e.life),x=(e.to[1]+.5)*S,y=(e.to[0]+.5)*S;
    ctx.globalAlpha=1-q;
    if(e.kind==='shot'){const x0=(e.from[1]+.5)*S,y0=(e.from[0]+.5)*S;ctx.fillStyle=e.colour;
      const px=x0+(x-x0)*q,py=y0+(y-y0)*q-(e.arc?Math.sin(q*Math.PI)*Math.hypot(x-x0,y-y0)*.3:0);
      ctx.beginPath();ctx.arc(px,py,S*(e.arc?.12:.06),0,Math.PI*2);ctx.fill();
    }else if(e.kind==='ring'){ctx.strokeStyle=e.colour;ctx.lineWidth=S*.06;ctx.beginPath();ctx.arc(x,y,S*(.12+.55*q),0,Math.PI*2);ctx.stroke();
    }else if(e.kind==='number'){ctx.fillStyle='#892c21';ctx.font='bold 11px sans-serif';ctx.fillText(`−${e.damage}`,x+5,y-8-q*23);
    }else{ctx.fillStyle=e.colour;ctx.fillRect(x+Math.cos(e.angle)*q*S*.8,y+Math.sin(e.angle)*q*S*.8+q*q*12,4,4);}}
  ctx.globalAlpha=1;
}
function syncUI(force=false){
  if(!shown||!payload)return;const s=payload.readonly?shown:payload.snapshot,over=Boolean(s.finish_reason),shop=s.phase==='shopping';
  $('btn-play').textContent=playing?'Ⅱ Pause':'▶ Play';$('btn-play').setAttribute('aria-label',playing?'Pause replay':'Play replay');
  $('timeline').value=total()?Math.round(cursor/total()*1000):0;$('replay-clock').textContent=`${clock(cursor)} / ${clock(total())}`;
  $('playback-status').textContent=cursor>=total()&&total()>0?'Finished':playing?'Playing':'Paused';
  const stamp=`${index}:${key}:${playing}:${pending}:${selected}:${running}:${s.revision}`;if(!force&&stamp===uiStamp)return;uiStamp=stamp;
  $('s-phase').textContent=over?'Finished':shop?'Shopping':'Battle';$('s-budget').textContent=200-s.spent;
  const destruction=s.destroyed_fraction??s.destruction??0;$('s-dest').textContent=`${(destruction*100).toFixed(1)}%`;
  $('s-bar').style.width=`${destruction*100}%`;$('s-tick').textContent=300-s.tick;
  $('paneltitle').textContent=payload.readonly?'Recorded army':shop?'Buy your army':over?'Battle complete':'Deploy your troops';
  const nextArmyStamp=JSON.stringify([payload.readonly,shop,over,s.spent,s.reserve,selected]);
  if(nextArmyStamp!==armyStamp){armyStamp=nextArmyStamp;
  $('troops').replaceChildren();payload.troop_types.forEach((kind,i)=>{
    const stat=payload.troops[kind],row=document.createElement('button');row.className=`troop ${selected===kind?'sel':''}`;row.style.width='100%';row.style.textAlign='left';
    row.disabled=payload.readonly||over||(shop?(!payload.snapshot.legal_actions.includes(i)&&!s.reserve[kind]):!s.reserve[kind]);row.setAttribute('aria-label',`${shop?'Buy':'Select'} ${kind}`);
    row.innerHTML=`<span class="swatch" style="background:${colours[kind]}"></span><span><span class="nm">${kind}</span><br><span class="ds">${stat.hp} hp · ${stat.damage_per_tick} damage · range ${stat.range}</span></span><span class="ct"><span class="cost">${stat.cost}r</span><span class="own">×${s.reserve[kind]||0}</span></span>`;
    row.onclick=e=>{selected=kind;if(shop)command(e.shiftKey?'refund':'buy',{troop:kind});else syncUI(true);};$('troops').append(row);
  });}
  $('btn-main').textContent=shop?'Start battle':running?'Pause':s.tick===0?'Run the clock':'Continue';$('btn-main').disabled=payload.readonly||over||(shop&&!Object.values(s.reserve).some(n=>n>0));
  $('btn-reset').disabled=payload.readonly;$('btn-deploy').disabled=payload.readonly||shop||over||!(s.reserve[selected]>0);
  $('hint').textContent=over?`Ended: ${s.finish_reason.replaceAll('_',' ')}.`:payload.readonly?'Watch the recorded attack. Pause or drag the timeline to inspect a moment.':shop?'Spend up to 200. Max 25 troops. Shift-click a row to refund one.':running?'Running — Space to pause. You can still select and deploy troops.':'Click a green edge to deploy (Shift-click for 5). Space runs the clock. Keys 1–4 select troops.';
  $('error').textContent=payload.error||'';
  $('result-banner').hidden=!over;if(over)$('result-banner').innerHTML=`<strong>${(destruction*100).toFixed(1)}% destroyed</strong><span>${s.full_clear?'Full clear':s.town_hall?'Town Hall destroyed':'Battle complete'} · ${s.spent} resources spent</span>`;
  document.querySelector('.evidence').hidden=!payload.readonly;
  $('reward').replaceChildren();if(s.reward_totals)for(const[label,value]of Object.entries({...s.reward_totals,return:s.episode_return})){const tr=document.createElement('tr'),a=document.createElement('td'),b=document.createElement('td');a.textContent=label.replaceAll('_',' ');b.textContent=Number(value).toFixed(3);tr.append(a,b);$('reward').append(tr);}
  $('last-action').textContent=s.last_action!=null?`Decision ${s.decisions} · action ${s.last_action}`:'Battle tick '+s.tick;
  $('badges').replaceChildren();for(const label of [destruction>=.5?'50% reached':null,s.town_hall?'Town Hall':null,s.full_clear?'Full clear':null].filter(Boolean)){const b=document.createElement('span');b.className='badge';b.textContent=label;$('badges').append(b);}
  $('log').replaceChildren();let lines;
  if(payload.readonly){lines=frames.slice(0,index+1).filter(f=>f.last_action!=null).slice(-16).map(f=>`Decision ${f.decisions} · tick ${f.tick} · ${((f.destroyed_fraction??0)*100).toFixed(1)}% destroyed`);}
  else lines=payload.log||[];
  for(const line of lines){const div=document.createElement('div');div.textContent=line;$('log').append(div);}$('log').scrollTop=$('log').scrollHeight;height();
}
function animate(now){
  const dt=lastTime?Math.min(100,now-lastTime):0;lastTime=now;
  if(payload?.readonly&&playing){
    const rate=Number($('speed').value);cursor=Math.min(total(),cursor+dt*rate);
    for(const e of effects)e.age+=dt*rate;effects=effects.filter(e=>e.age<e.life);updatePosition(true);
    if(cursor>=total()){playing=false;syncUI(true);}
  }
  if(payload&&!payload.readonly&&manual){
    for(const e of effects)e.age+=dt;effects=effects.filter(e=>e.age<e.life);
    if(running||draining){
      manualElapsed+=dt;
      while(manualElapsed>=manualDuration&&(running||draining)){
        if(running&&!manual.finish){manualElapsed-=manualDuration;advanceManual();}
        else{manualElapsed=manualDuration;draining=false;break;}
      }
      blend=Math.min(1,manualElapsed/manualDuration);
      if(blend>=1){shown=visualNext=payload.snapshot;index=frames.length-1;blend=0;}
    }
  }
  draw(payload?.readonly?cursor:now);requestAnimationFrame(animate);
}
$('btn-main').onclick=mainControl;$('btn-reset').onclick=()=>command('reset');$('btn-deploy').onclick=e=>deploy(Number($('tile').value),e.shiftKey?5:1);
$('btn-play').onclick=()=>{if(cursor>=total())seek(0);playing=!playing;lastTime=0;syncUI(true);};
$('btn-restart').onclick=()=>{playing=false;seek(0);};
$('btn-step').onclick=()=>{playing=false;let i=index+1;while(i<frames.length&&frames[i].decisions==null)i++;seek(times[Math.min(i,frames.length-1)]);};
$('timeline').oninput=()=>{playing=false;seek(Number($('timeline').value)/1000*total());};
$('stepmode').onchange=()=>{chunk=0;syncUI(true);};
$('speed').onchange=()=>{const fraction=manualElapsed/manualDuration;manualDuration=tickMillis();manualElapsed=fraction*manualDuration;};
window.addEventListener('keydown',e=>{
  if(!payload||payload.readonly||e.repeat||['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName)||e.target.isContentEditable)return;
  if(e.code==='Space'){e.preventDefault();mainControl();return;}
  const n=['Digit1','Digit2','Digit3','Digit4'].indexOf(e.code);
  if(n>=0&&!payload.snapshot.finish_reason){selected=payload.troop_types[n];syncUI(true);}
});
$('btn-fullscreen').onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.querySelector('.boardwrap').requestFullscreen();}catch{ $('recording-note').textContent='Fullscreen is unavailable in this browser view.';}};
cv.onclick=e=>{if(!payload||payload.readonly||shown.phase!=='battle')return;const b=cv.getBoundingClientRect(),c=Math.floor((e.clientX-b.left)/b.width*16),r=Math.floor((e.clientY-b.top)/b.height*16);const i=perimeter.findIndex(([rr,cc])=>rr===r&&cc===c);if(i>=0){$('tile').value=i;deploy(i,e.shiftKey?5:1);}};
cv.onmousemove=e=>{if(!shown)return;const b=cv.getBoundingClientRect(),c=Math.floor((e.clientX-b.left)/b.width*16),r=Math.floor((e.clientY-b.top)/b.height*16);hover=perimeter.some(([rr,cc])=>rr===r&&cc===c)?[r,c]:null;const building=shown.buildings.find(b=>b.row===r&&b.col===c);cv.title=building?`${building.type.replaceAll('_',' ')} · ${building.hp}/${building.max_hp} HP`:`Tile (${r}, ${c})`;};cv.onmouseleave=()=>hover=null;
window.addEventListener('message',event=>{
  if(event.source!==parent||event.data.type!=='streamlit:render')return;
  const incoming=event.data.args.payload;if(!incoming)return;
  const newKey=incoming.replay?incoming.replay.id:`manual:${incoming.game_key}`;
  const sameManual=manual&&!incoming.readonly&&newKey===key;
  payload=incoming;
  document.body.classList.toggle('is-replay',!!payload.readonly);
  $('playback').hidden=!payload.readonly;$('manual-controls').hidden=payload.readonly;
  $('recording-note').textContent=payload.readonly?(payload.replay.tick_recorded?'Recorded battle · speed changes playback only.':'Older replay: decision frames only. Generate a new replay for smooth movement.'):'Smooth manual play · Runs in your browser. Pause every decision stops after 10 ticks.';
  $('replay-label').textContent=payload.replay?.label||'Recorded attack';
  if(payload.readonly){
    running=false;draining=false;manual=null;
    if(newKey!==key){key=newKey;install(payload.replay.frames,false);}
  }else if(sameManual){
    // A sidebar rerun must never overwrite the live browser-owned game with
    // the server's initial board. Only a changed game key starts a new base.
    payload.snapshot=manual.snapshot();payload.log=manual.log;
  }else{
    key=newKey;running=false;playing=false;draining=false;chunk=0;manualElapsed=0;
    manual=new SiegeManualEngine(payload);payload.snapshot=manual.snapshot();payload.log=[];
    install([payload.snapshot],false);displayPositions=[manual.visualPositions()];
  }
  syncUI(true);
});
window.addEventListener('resize',height);new ResizeObserver(height).observe(document.querySelector('.wrap'));
document.addEventListener('visibilitychange',()=>{if(document.hidden){playing=false;running=false;draining=false;lastTime=0;syncUI(true);}});
send('streamlit:componentReady',{apiVersion:1});send('streamlit:setFrameHeight',{height:780});requestAnimationFrame(animate);

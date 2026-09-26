/* Base geometry editor only. Combat always runs in Python. */
const cv=document.getElementById('cv'),ctx=cv.getContext('2d'),$=id=>document.getElementById(id),S=34;
const colours={town_hall:'#6C63C8',cannon:'#C74E77',mortar:'#CE5A2E',storage:'#7A8B8E',wall:'#6b6560'};
const labels={move:'Move building',town_hall:'Town Hall',cannon:'Cannon',mortar:'Mortar',storage:'Storage',wall:'Paint wall',erase:'Erase'};
let draft=null,version=null,counts={},tool='move',picked=null,history=[],painting=false,lastTile='',height=0;
function send(type,fields={}){parent.postMessage({isStreamlitMessage:true,type,...fields},'*');}
function status(text,error=false){$('status').textContent=text;$('status').classList.toggle('error',error);}
function saveUndo(){history.push(JSON.stringify(draft));if(history.length>100)history.shift();}
function count(kind){return draft.buildings.filter(b=>b[0]===kind).length;}
function sync(){
  if(!draft)return;
  const palette=$('palette');palette.replaceChildren();
  for(const [kind,label] of Object.entries(labels)){
    const b=document.createElement('button');b.className=tool===kind?'sel':'';b.dataset.tool=kind;
    b.innerHTML=(colours[kind]?`<span class="swatch" style="background:${colours[kind]}"></span>`:'')+label;
    if(kind in counts)b.append(` ${count(kind)}/${counts[kind]}`);
    b.onclick=()=>{tool=kind;picked=null;sync();};palette.append(b);
  }
  $('inventory').textContent=`${draft.buildings.length}/10 buildings · ${draft.walls.length} walls`;
  $('help').textContent=tool==='move'?(picked?'Click an empty interior tile to place this building.':'Click a building, then click its new location.'):tool==='wall'?'Click or drag to paint walls on empty interior tiles.':tool==='erase'?'Click or drag to erase walls or buildings.':`Click an empty interior tile to place a ${labels[tool]}.`;
  $('undo').disabled=!history.length;
  const complete=Object.entries(counts).every(([k,n])=>count(k)===n);
  for(const id of ['save','manual','agent'])$(id).disabled=!complete;
  draw();reportHeight();
}
function reportHeight(){const h=Math.ceil(document.querySelector('.wrap').getBoundingClientRect().height)+8;if(height!==h){height=h;send('streamlit:setFrameHeight',{height:h});}}
function draw(){
  if(!draft)return;
  for(let r=0;r<16;r++)for(let c=0;c<16;c++){
    const edge=r===0||c===0||r===15||c===15;
    ctx.fillStyle=edge?((r+c)%2?'#7fa65c':'#789e55'):((r+c)%2?'#cdbb96':'#c7b48d');ctx.fillRect(c*S,r*S,S,S);
    ctx.strokeStyle='#7f715a22';ctx.strokeRect(c*S,r*S,S,S);
  }
  for(const [r,c] of draft.walls)gWall(c*S,r*S,S);
  for(const [kind,r,c] of draft.buildings)BGLYPH[kind](c*S,r*S,S,colours[kind],-Math.PI/4,0);
  if(picked){ctx.strokeStyle='#fff';ctx.lineWidth=3;ctx.strokeRect(picked[2]*S+2,picked[1]*S+2,S-4,S-4);}
}
function apply(r,c){
  if(!draft||!Number.isInteger(r)||!Number.isInteger(c)||r<1||r>14||c<1||c>14){status('Keep the green perimeter clear for troop deployment.',true);return;}
  const bi=draft.buildings.findIndex(b=>b[1]===r&&b[2]===c),wi=draft.walls.findIndex(w=>w[0]===r&&w[1]===c);
  if(tool==='move'){
    if(bi>=0){picked=draft.buildings[bi];sync();return;}
    if(!picked){status('Select a building to move first.',true);return;}
    if(wi>=0){status('Erase the wall here or choose an empty tile.',true);return;}
    picked[1]=r;picked[2]=c;picked=null;
  }else if(tool==='erase'){
    if(bi>=0)draft.buildings.splice(bi,1);
    if(wi>=0)draft.walls.splice(wi,1);
  }else{
    if(bi>=0||wi>=0){status('This tile is occupied. Use Move or Erase.',true);return;}
    if(tool==='wall')draft.walls.push([r,c]);
    else if(count(tool)<counts[tool])draft.buildings.push([tool,r,c]);
    else{status(`All ${labels[tool]} buildings are placed. Move or erase one first.`,true);return;}
  }
  status('Unsaved changes. Place all 10 buildings, then save or play.');sync();
}
function tileAt(e){const b=cv.getBoundingClientRect();return [Math.floor((e.clientY-b.top)/b.height*16),Math.floor((e.clientX-b.left)/b.width*16)];}
cv.onpointerdown=e=>{if(e.button!==0||!draft)return;e.preventDefault();cv.focus();saveUndo();painting=true;cv.setPointerCapture(e.pointerId);const t=tileAt(e);lastTile=t.join(',');apply(...t);};
cv.onpointermove=e=>{if(!painting||!['wall','erase'].includes(tool))return;const t=tileAt(e),id=t.join(',');if(id!==lastTile){lastTile=id;apply(...t);}};
cv.onpointerup=cv.onpointercancel=()=>{painting=false;lastTile='';};
$('place').onclick=()=>{saveUndo();apply(Number($('row').value),Number($('col').value));};
$('undo').onclick=()=>{if(history.length){draft=JSON.parse(history.pop());picked=null;$('name').value=draft.name;status('Last edit undone.');sync();}};
$('clear-walls').onclick=()=>{saveUndo();draft.walls=[];picked=null;status('Walls cleared. Undo is available.');sync();};
$('clear-all').onclick=()=>{saveUndo();draft.walls=[];draft.buildings=[];picked=null;status('Place all 10 buildings to save or play. Undo is available.');sync();};
$('name').oninput=()=>{if(draft){draft.name=$('name').value;status('Unsaved name change.');}};
function submit(target){
  if(!draft)return;
  draft.name=$('name').value.trim();
  if(!draft.name){status('Give your base a name first.',true);return;}
  send('streamlit:setComponentValue',{value:{id:`${Date.now()}-${Math.random()}`,version,layout:draft,target},dataType:'json'});
  status(target?'Opening your saved layout…':'Base submitted. Download its JSON below to keep a copy.');
}
$('save').onclick=()=>submit(null);$('manual').onclick=()=>submit('Manual play');$('agent').onclick=()=>submit('Policy replay');
window.addEventListener('message',e=>{if(e.source!==parent||e.data.type!=='streamlit:render')return;const p=e.data.args.payload;if(!p)return;if(version!==p.version){version=p.version;draft=JSON.parse(JSON.stringify(p.layout));counts=p.counts;history=[];picked=null;$('name').value=draft.name;status('Edit this template, then save or play.');}sync();});
window.addEventListener('resize',reportHeight);
send('streamlit:componentReady',{apiVersion:1});send('streamlit:setFrameHeight',{height:660});

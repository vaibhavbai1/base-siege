// Visual primitives retained from Vaibhav's supplied HTML.
function shade(hex,f){
  const n=parseInt(hex.slice(1),16);
  const ch=v=>Math.min(255,Math.max(0,Math.round(v*f)));
  return '#'+(((ch(n>>16)<<16)|(ch((n>>8)&255)<<8)|ch(n&255))>>>0)
    .toString(16).padStart(6,'0');
}
function roundRect(x,y,w,h,r){ctx.beginPath();ctx.moveTo(x+r,y);
  ctx.arcTo(x+w,y,x+w,y+h,r);ctx.arcTo(x+w,y+h,x,y+h,r);
  ctx.arcTo(x,y+h,x,y,r);ctx.arcTo(x,y,x+w,y,r);ctx.closePath();}


function gTownHall(x,y,S,c){
  const d=shade(c,.68), l=shade(c,1.24);
  ctx.fillStyle=c; ctx.beginPath();
  ctx.moveTo(x+.20*S,y+.90*S); ctx.lineTo(x+.27*S,y+.40*S);
  ctx.lineTo(x+.73*S,y+.40*S); ctx.lineTo(x+.80*S,y+.90*S);
  ctx.closePath(); ctx.fill();
  ctx.fillStyle=l; ctx.fillRect(x+.22*S,y+.32*S,.56*S,.10*S);
  for(const o of[.22,.44,.66])ctx.fillRect(x+o*S,y+.24*S,.12*S,.10*S);
  ctx.fillStyle=d; ctx.beginPath();
  ctx.moveTo(x+.42*S,y+.90*S); ctx.lineTo(x+.42*S,y+.66*S);
  ctx.quadraticCurveTo(x+.5*S,y+.56*S,x+.58*S,y+.66*S);
  ctx.lineTo(x+.58*S,y+.90*S); ctx.closePath(); ctx.fill();
  ctx.fillStyle=d; ctx.fillRect(x+.47*S,y+.03*S,.045*S,.22*S);
  ctx.beginPath(); ctx.moveTo(x+.515*S,y+.05*S);
  ctx.lineTo(x+.76*S,y+.11*S); ctx.lineTo(x+.515*S,y+.17*S);
  ctx.closePath(); ctx.fill();
}
function gCannon(x,y,S,c,aim,recoil){
  const d=shade(c,.66), l=shade(c,1.26);
  ctx.save(); ctx.translate(x+.5*S,y+.62*S);
  ctx.rotate((aim===undefined?-Math.PI/4:aim)+Math.PI/2);
  ctx.translate(0,(recoil||0)*S*.14);
  ctx.fillStyle=d; roundRect(-.10*S,-.56*S,.20*S,.56*S,.05*S); ctx.fill();
  ctx.fillStyle=shade(c,.5); roundRect(-.13*S,-.60*S,.26*S,.10*S,.04*S); ctx.fill();
  ctx.restore();
  ctx.fillStyle=c; ctx.beginPath();
  ctx.arc(x+.5*S,y+.66*S,.30*S,Math.PI,0); ctx.fill();
  ctx.fillRect(x+.20*S,y+.66*S,.60*S,.20*S);
  ctx.fillStyle=l; ctx.beginPath();
  ctx.arc(x+.5*S,y+.66*S,.13*S,Math.PI,0); ctx.fill();
}
function gMortar(x,y,S,c,aim,recoil){
  const d=shade(c,.60), l=shade(c,1.26);
  ctx.save(); ctx.translate(0,(recoil||0)*S*.10);
  ctx.fillStyle=c; ctx.beginPath();
  ctx.moveTo(x+.22*S,y+.24*S); ctx.lineTo(x+.78*S,y+.24*S);
  ctx.lineTo(x+.65*S,y+.74*S); ctx.lineTo(x+.35*S,y+.74*S);
  ctx.closePath(); ctx.fill();
  ctx.fillStyle=d; ctx.beginPath();
  ctx.ellipse(x+.5*S,y+.25*S,.28*S,.09*S,0,0,7); ctx.fill();
  ctx.fillStyle=l; roundRect(x+.18*S,y+.72*S,.64*S,.18*S,.06*S); ctx.fill();
  ctx.restore();
}
function gStorage(x,y,S,c){
  const d=shade(c,.62), l=shade(c,1.28);
  ctx.fillStyle=c; roundRect(x+.14*S,y+.30*S,.72*S,.58*S,.06*S); ctx.fill();
  ctx.strokeStyle=d; ctx.lineWidth=Math.max(1.4,.06*S);
  ctx.beginPath();
  ctx.moveTo(x+.19*S,y+.34*S); ctx.lineTo(x+.81*S,y+.84*S);
  ctx.moveTo(x+.81*S,y+.34*S); ctx.lineTo(x+.19*S,y+.84*S);
  ctx.stroke();
  ctx.fillStyle=l; roundRect(x+.09*S,y+.22*S,.82*S,.14*S,.05*S); ctx.fill();
}
function gWall(x,y,S){
  ctx.fillStyle='#5f5952'; ctx.fillRect(x,y+.22*S,S,.78*S);
  ctx.fillStyle='#8a837c'; ctx.fillRect(x,y+.22*S,S,.15*S);
  ctx.fillStyle='#5f5952';
  for(const o of[.03,.39,.75])ctx.fillRect(x+o*S,y+.02*S,.22*S,.23*S);
  ctx.strokeStyle='#4b4640'; ctx.lineWidth=1; ctx.beginPath();
  ctx.moveTo(x,y+.60*S); ctx.lineTo(x+S,y+.60*S);
  ctx.moveTo(x+.5*S,y+.37*S); ctx.lineTo(x+.5*S,y+.60*S);
  ctx.moveTo(x+.25*S,y+.60*S); ctx.lineTo(x+.25*S,y+S);
  ctx.moveTo(x+.75*S,y+.60*S); ctx.lineTo(x+.75*S,y+S);
  ctx.stroke();
}
function tTank(x,y,r,c){
  const d=shade(c,.60), l=shade(c,1.32);
  ctx.fillStyle=d;
  roundRect(x-r*1.02,y-r*.82,r*2.04,r*.44,r*.2); ctx.fill();
  roundRect(x-r*1.02,y+r*.38,r*2.04,r*.44,r*.2); ctx.fill();
  ctx.fillStyle=c; roundRect(x-r*.86,y-r*.58,r*1.72,r*1.16,r*.26); ctx.fill();
  ctx.fillStyle=d; ctx.fillRect(x,y-r*.13,r*1.3,r*.26);
  ctx.fillStyle=l; ctx.beginPath(); ctx.arc(x,y,r*.44,0,7); ctx.fill();
}
function tMelee(x,y,r,c){
  const d=shade(c,.55), l=shade(c,1.35);
  ctx.fillStyle=d; ctx.fillRect(x+r*.34,y-r*1.0,r*.17,r*1.5);
  ctx.fillRect(x+r*.14,y+r*.3,r*.57,r*.16);
  ctx.fillStyle=c; ctx.beginPath();
  ctx.moveTo(x-r*.82,y-r*.85); ctx.lineTo(x+r*.22,y-r*.85);
  ctx.lineTo(x+r*.22,y+r*.18);
  ctx.quadraticCurveTo(x-r*.3,y+r*1.0,x-r*.82,y+r*.18);
  ctx.closePath(); ctx.fill();
  ctx.fillStyle=l; ctx.fillRect(x-r*.62,y-r*.52,r*.62,r*.15);
}
function tRanged(x,y,r,c){
  const d=shade(c,.52);
  ctx.lineCap='round';
  ctx.strokeStyle=c; ctx.lineWidth=r*.34;
  ctx.beginPath(); ctx.arc(x-r*.2,y,r*.82,-Math.PI*.44,Math.PI*.44); ctx.stroke();
  ctx.strokeStyle=d; ctx.lineWidth=r*.11;
  const a=Math.PI*.44, cx=x-r*.2;
  ctx.beginPath();
  ctx.moveTo(cx+r*.82*Math.cos(-a),y+r*.82*Math.sin(-a));
  ctx.lineTo(cx+r*.82*Math.cos(a),y+r*.82*Math.sin(a)); ctx.stroke();
  ctx.lineWidth=r*.17;
  ctx.beginPath(); ctx.moveTo(x-r*.75,y); ctx.lineTo(x+r*.8,y); ctx.stroke();
  ctx.lineCap='butt';
}
function tBreaker(x,y,r,c){
  const d=shade(c,.5), l=shade(c,1.35);
  ctx.fillStyle=c; ctx.beginPath(); ctx.arc(x,y+r*.18,r*.82,0,7); ctx.fill();
  ctx.fillStyle=l; ctx.beginPath(); ctx.arc(x-r*.28,y-r*.12,r*.26,0,7); ctx.fill();
  ctx.fillStyle=d; ctx.fillRect(x-r*.24,y-r*.86,r*.48,r*.3);
  ctx.strokeStyle=d; ctx.lineWidth=r*.15; ctx.lineCap='round';
  ctx.beginPath(); ctx.moveTo(x,y-r*.82);
  ctx.quadraticCurveTo(x+r*.55,y-r*1.3,x+r*.38,y-r*1.55); ctx.stroke();
  ctx.lineCap='butt';
  ctx.fillStyle='#FFC94D'; ctx.beginPath();
  ctx.arc(x+r*.38,y-r*1.62,r*.22,0,7); ctx.fill();
}
const BGLYPH={town_hall:gTownHall,cannon:gCannon,mortar:gMortar,storage:gStorage};
const TGLYPH={tank:tTank,melee:tMelee,ranged:tRanged,breaker:tBreaker};

function drawHP(x,y,w,frac){
  ctx.fillStyle='#00000030'; roundRect(x,y,w,4,2); ctx.fill();
  ctx.fillStyle=frac>0.6?'#5E8F22':frac>0.25?'#D99425':'#D6423F';
  roundRect(x,y,w*Math.max(0,frac),4,2); ctx.fill();
}


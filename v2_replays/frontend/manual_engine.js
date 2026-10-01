/* Browser-local manual simulation. Stats and base geometry are supplied by Python.
 * Mirrors env/combat.py, entities.py and pathfinding.py; checked tick-for-tick.
 * Training and learned-policy evaluation continue to use the Python engine.
 */
(function(root){
'use strict';
const clone=x=>JSON.parse(JSON.stringify(x)), key=(r,c)=>r*16+c;
const neighbours=[[-1,0],[1,0],[0,-1],[0,1]];
const distance=(a,b)=>Math.hypot(a.row-b.row,a.col-b.col);
const posOrder=(a,b)=>a.row-b.row||a.col-b.col;
const pyRound=x=>{const f=Math.floor(x),d=x-f;return d===.5?(f%2?f+1:f):Math.round(x);};
class Heap{
  constructor(){this.a=[];}
  less(a,b){return a[0]<b[0]||(a[0]===b[0]&&a[1]<b[1]);}
  push(v){let i=this.a.length;this.a.push(v);while(i){const p=(i-1)>>1;if(!this.less(v,this.a[p]))break;this.a[i]=this.a[p];i=p;}this.a[i]=v;}
  pop(){const result=this.a[0],v=this.a.pop();if(this.a.length){let i=0;while(i*2+1<this.a.length){let c=i*2+1;if(c+1<this.a.length&&this.less(this.a[c+1],this.a[c]))c++;if(!this.less(this.a[c],v))break;this.a[i]=this.a[c];i=c;}this.a[i]=v;}return result;}
}
class ManualEngine{
  constructor(payload){
    this.config=clone(payload.manual_config);this.types=payload.troop_types.slice();
    this.stats=clone(payload.troops);this.buildingStats=clone(payload.buildings);
    this.initial=clone(payload.snapshot);this.reset();
  }
  reset(){
    const s=this.initial;this.buildings=new Map(s.buildings.map(b=>[key(b.row,b.col),{...clone(b),cooldown:0}]));
    this.troops=[];this.reserve=Object.fromEntries(this.types.map(t=>[t,0]));
    this.tick=0;this.phase='shopping';this.spent=0;this.total=0;this.destroyed=0;this.nextId=1;
    this.milestones={half:false,town_hall:false,full_clear:false};this.finish=null;this.attacks=[];
    this.topology=0;this.seenVersion=null;this.cache=new Map();this.revision=0;this.log=[];
    this.totalValue=[...this.buildings.values()].reduce((a,b)=>a+this.buildingStats[b.type].value,0);
  }
  command(kind,{troop,tile,count=1}={}){
    if(kind==='reset'){this.reset();return;}
    if(this.finish)throw Error('The battle has finished. Start a new base.');
    if(kind==='buy'||kind==='refund'){
      if(this.phase!=='shopping'||!this.types.includes(troop))throw Error('Buy or refund troops during shopping only.');
      const cost=this.stats[troop].cost;
      if(kind==='buy'){
        if(this.spent+cost>this.config.budget||Object.values(this.reserve).reduce((a,b)=>a+b,0)>=this.config.max_army)throw Error('Army size or budget limit reached.');
        this.reserve[troop]++;this.spent+=cost;
      }else{
        if(!this.reserve[troop])throw Error('There is no troop of this type to refund.');
        this.reserve[troop]--;this.spent-=cost;
      }
      this.log.push(`${kind==='buy'?'Buy':'Refund'} ${troop}`);
    }else if(kind==='start'){
      if(this.phase!=='shopping'||!Object.values(this.reserve).some(n=>n>0))throw Error('Buy at least one troop before starting.');
      this.phase='battle';this.log.push('Deploy your opening wave, then run the clock.');
    }else if(kind==='deploy'){
      if(this.phase!=='battle'||!this.types.includes(troop)||!Number.isInteger(tile)||tile<0||tile>=60||![1,5].includes(count))throw Error('Choose a troop and a green perimeter tile.');
      const amount=Math.min(count,this.reserve[troop]);if(!amount)throw Error('No selected troops remain in reserve.');
      const [row,col]=this.config.deploy_tiles[tile];
      for(let i=0;i<amount;i++)this.troops.push({type:troop,row,col,hp:this.stats[troop].hp,max_hp:this.stats[troop].hp,entity_id:this.nextId++,credit:0,target:null,path:null});
      this.reserve[troop]-=amount;this.total+=amount;this.log.push(`${amount} × ${troop} deployed at (${row}, ${col})`);
    }else if(kind==='tick'){
      if(this.phase!=='battle')throw Error('Start the battle before running the clock.');
      this.step();return;
    }else throw Error('Unknown manual command.');
    this.attacks=[];this.revision++;this.log=this.log.slice(-60);
  }
  blocked(r,c){const b=this.buildings.get(key(r,c));return b&&b.hp>0&&b.type!=='wall';}
  inBounds(r,c){return r>=0&&r<16&&c>=0&&c<16;}
  pathTo(t,target){
    const goals=neighbours.map(([dr,dc])=>[target.row+dr,target.col+dc]).filter(([r,c])=>this.inBounds(r,c)&&!this.blocked(r,c)).map(([r,c])=>key(r,c)).sort((a,b)=>a-b);
    if(!goals.length)return null;
    const start=key(t.row,t.col),cacheKey=`${t.type}:${start}:${goals.join(',')}`;
    if(this.cache.has(cacheKey))return clone(this.cache.get(cacheKey));
    const openCost=1/this.stats[t.type].speed,damage=this.stats[t.type].damage_per_tick;
    const h=n=>openCost*Math.min(...goals.map(g=>Math.abs((n>>4)-(g>>4))+Math.abs((n%16)-(g%16))));
    const frontier=new Heap();frontier.push([h(start),start]);
    const best=new Map([[start,0]]),came=new Map(),closed=new Set();let result=null;
    while(frontier.a.length){
      const [,current]=frontier.pop();if(closed.has(current))continue;closed.add(current);
      if(goals.includes(current)){let path=[current];while(came.has(path.at(-1)))path.push(came.get(path.at(-1)));result=path.reverse().map(n=>[n>>4,n%16]);break;}
      const r=current>>4,c=current%16;
      for(const [dr,dc]of neighbours){
        const rr=r+dr,cc=c+dc,n=key(rr,cc);if(!this.inBounds(rr,cc)||closed.has(n)||this.blocked(rr,cc))continue;
        const b=this.buildings.get(n),cost=b&&b.hp>0&&b.type==='wall'?(damage>0?openCost+b.hp/damage:Infinity):openCost;
        const candidate=best.get(current)+cost;
        if(candidate<(best.get(n)??Infinity)){best.set(n,candidate);came.set(n,current);frontier.push([candidate+h(n),n]);}
      }
    }
    this.cache.set(cacheKey,result);return clone(result);
  }
  purge(){
    let changed=false;
    for(const [k,b]of this.buildings)if(b.hp<=0){
      this.buildings.delete(k);changed=true;
      if(b.type!=='wall'){this.destroyed+=this.buildingStats[b.type].value;if(b.type==='town_hall')this.milestones.town_hall=true;}
    }
    if(changed)this.topology++;
    this.troops=this.troops.filter(t=>t.hp>0);
  }
  invalidate(){if(this.seenVersion===this.topology)return;this.seenVersion=this.topology;this.cache.clear();for(const t of this.troops)t.path=null;}
  acquire(t){
    let rule=this.stats[t.type].target_rule;
    while(rule){
      const options=[...this.buildings.values()].filter(b=>b.hp>0&&(rule==='nearest_wall'?b.type==='wall':rule==='nearest_defence'?this.config.defence_types.includes(b.type):b.type!=='wall'));
      if(options.length){options.sort((a,b)=>distance(t,a)-distance(t,b)||posOrder(a,b));return key(options[0].row,options[0].col);}
      rule=this.config.target_fallback[rule];
    }
    return null;
  }
  step(){
    if(this.finish)return;
    const pending=new Map(),attacks=[];
    const add=(entity,amount)=>{if(amount>0)pending.set(entity,(pending.get(entity)||0)+amount);};
    this.purge();this.invalidate();
    for(const t of this.troops){
      if(t.target===null)t.target=this.acquire(t);
      else if(!this.buildings.has(t.target)||this.buildings.get(t.target).hp<=0){t.target=this.acquire(t);t.path=null;}
    }
    const strike=(t,b)=>{add(b,this.stats[t.type].damage_per_tick);attacks.push({from:[t.row,t.col],to:[b.row,b.col],kind:'troop',damage:this.stats[t.type].damage_per_tick});if(this.stats[t.type].dies_after_attack)add(t,t.hp);};
    for(const t of [...this.troops].sort((a,b)=>a.entity_id-b.entity_id)){
      const target=this.buildings.get(t.target),stat=this.stats[t.type];
      if(t.hp<=0||!target||target.hp<=0||stat.walls_only&&target.type!=='wall')continue;
      if(distance(t,target)<=stat.range){strike(t,target);continue;}
      // No approach goals leaves the old cached path untouched in Python.
      const goals=neighbours.some(([dr,dc])=>this.inBounds(target.row+dr,target.col+dc)&&!this.blocked(target.row+dr,target.col+dc));
      if(!goals)continue;
      const path=this.pathTo(t,target);t.path=path;t.visual_path=path;if(!path||path.length<2)continue;
      const [r,c]=path[1],obstacle=this.buildings.get(key(r,c));
      if(obstacle&&obstacle.hp>0){if(obstacle.type==='wall')strike(t,obstacle);continue;}
      t.credit+=pyRound(stat.speed*this.config.speed_scale);
      if(t.credit>=this.config.speed_scale){t.credit-=this.config.speed_scale;t.row=r;t.col=c;}
    }
    const defences=[...this.buildings.values()].filter(b=>b.hp>0&&this.config.defence_types.includes(b.type)).sort(posOrder);
    for(const b of defences){
      const stat=this.buildingStats[b.type];if(b.cooldown>0){b.cooldown--;continue;}
      const targets=this.troops.filter(t=>t.hp>0&&distance(b,t)>=stat.min_range&&distance(b,t)<=stat.range).sort((a,c)=>distance(b,a)-distance(b,c)||a.entity_id-c.entity_id);
      const target=targets[0];if(!target)continue;
      const hit=stat.splash_radius>0?this.troops.filter(t=>t.hp>0&&distance(target,t)<=stat.splash_radius):[target];
      for(const t of hit){
        const splash=t.row!==target.row||t.col!==target.col;
        const damage=splash?Math.max(1,pyRound(stat.damage_per_shot*this.config.splash_falloff)):stat.damage_per_shot;
        add(t,damage);attacks.push({from:[b.row,b.col],to:[t.row,t.col],kind:'defence',damage,attacker_type:b.type,impact:[target.row,target.col],target_id:t.entity_id,splash});
      }
      b.cooldown=stat.attack_cooldown-1;
    }
    for(const [e,amount]of pending)e.hp=Math.max(0,e.hp-amount);
    this.purge();this.invalidate();this.tick++;this.attacks=attacks;
    if(this.destroyed/Math.max(1,this.totalValue)>=this.config.half_threshold)this.milestones.half=true;
    if(![...this.buildings.values()].some(b=>b.type!=='wall')){this.milestones.full_clear=true;this.finish='full_clear';}
    else if(this.tick>=this.config.tick_limit)this.finish='tick_limit';
    else if(!this.troops.length&&!Object.values(this.reserve).some(n=>n>0))this.finish='army_eliminated';
    this.revision++;
    if(attacks.length||this.finish)this.log.push(`Tick ${this.tick} · ${(100*this.destroyed/Math.max(1,this.totalValue)).toFixed(1)}% destroyed${this.finish?' · '+this.finish.replaceAll('_',' '):''}`);
    this.log=this.log.slice(-60);
  }
  snapshot(){
    const legal=[];
    if(!this.finish){
      if(this.phase==='shopping'){
        if(Object.values(this.reserve).reduce((a,b)=>a+b,0)<this.config.max_army)this.types.forEach((t,i)=>{if(this.spent+this.stats[t].cost<=this.config.budget)legal.push(i);});legal.push(4);
      }else{this.types.forEach((t,i)=>{if(this.reserve[t])for(let n=0;n<60;n++)legal.push(5+i*60+n);});legal.push(245);}
    }
    const destruction=this.destroyed/Math.max(1,this.totalValue);
    return {tick:this.tick,phase:this.phase,spent:this.spent,reserve:{...this.reserve},troops_alive:this.troops.length,troops_total:this.total,
      buildings:[...this.buildings.values()].sort(posOrder).map(b=>({type:b.type,row:b.row,col:b.col,hp:b.hp,max_hp:b.max_hp})),
      troops:[...this.troops].sort((a,b)=>a.entity_id-b.entity_id).map(t=>({type:t.type,row:t.row,col:t.col,hp:t.hp,max_hp:t.max_hp,entity_id:t.entity_id,path:clone(t.path),movement_fraction:t.credit/this.config.speed_scale})),
      attacks:clone(this.attacks),destroyed_fraction:destruction,destruction,town_hall:this.milestones.town_hall,full_clear:this.milestones.full_clear,
      finish_reason:this.finish,decisions:Math.floor(this.tick/this.config.ticks_per_decision),revision:this.revision,last_action:null,legal_actions:legal,
      base_seed:this.initial.base_seed,motif:this.initial.motif};
  }
  // Presentation positions spend movement credit continuously along an open
  // path. They never feed back into targeting, damage or integer tile motion.
  visualPositions(){
    const poses=new Map();
    for(const t of this.troops){
      const path=t.path||t.visual_path||[],at=path.findIndex(([r,c])=>r===t.row&&c===t.col),next=at>=0?path[at+1]:null;
      const clear=next&&!this.buildings.has(key(...next));
      const fraction=clear?t.credit/this.config.speed_scale:0;
      poses.set(t.entity_id,{row:t.row+(clear?(next[0]-t.row)*fraction:0),col:t.col+(clear?(next[1]-t.col)*fraction:0)});
    }
    return poses;
  }
}
root.SiegeManualEngine=ManualEngine;
if(typeof module!=='undefined')module.exports=ManualEngine;
})(typeof globalThis!=='undefined'?globalThis:this);

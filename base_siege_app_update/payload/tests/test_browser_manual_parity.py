"""The browser-local manual engine must match real Python combat each tick.

Node runs the exact JS shipped to the browser. No mocked simulator is involved.
"""
import json
from pathlib import Path
import random
import shutil
import subprocess
import pytest
from env import config as C
from env.generator import generate_base, BaseSpec
from ui.manual_game import ManualGame

ROOT=Path(__file__).resolve().parents[1]
NODE=shutil.which('node')
pytestmark=pytest.mark.skipif(NODE is None,reason='Node is needed to run the browser-engine parity checks')
FIELDS=('tick','phase','spent','reserve','troops_alive','troops_total','buildings','troops','attacks',
        'destroyed_fraction','destruction','town_hall','full_clear','finish_reason','decisions',
        'revision','last_action','legal_actions','base_seed','motif')


def manual_payload(env):
    return {'snapshot':env.snapshot(),'troop_types':C.TROOP_TYPES,'troops':C.TROOP_STATS,'buildings':C.BUILDING_STATS,
        'manual_config':{'budget':C.RESOURCE_BUDGET,'max_army':C.MAX_ARMY_SIZE,'tick_limit':C.BATTLE_TICK_LIMIT,
            'ticks_per_decision':C.TICKS_PER_DECISION,'speed_scale':C.SPEED_SCALE,'splash_falloff':C.SPLASH_FALLOFF,
            'half_threshold':C.HALF_DESTRUCTION_THRESHOLD,'defence_types':C.DEFENCE_TYPES,
            'target_fallback':C.TARGET_RULE_FALLBACK,'deploy_tiles':C.DEPLOY_TILES}}


def compare_sequence(spec,army,seed):
    env=ManualGame(record=True,record_ticks=True);env.reset(seed=0,options={'spec':spec})
    payload=manual_payload(env);commands=[];expected=[];rng=random.Random(seed)
    def act(kind,**fields):
        env.command(kind,**fields);commands.append({'kind':kind,**fields})
        frame=env.visual_history[-1];expected.append({k:frame[k] for k in FIELDS})
    for troop,count in army.items():
        for _ in range(count):act('buy',troop=troop)
    first=next(iter(army));act('refund',troop=first);act('buy',troop=first);act('start')
    for tick in range(C.BATTLE_TICK_LIMIT):
        for troop in C.TROOP_TYPES:
            if env.state.reserve.get(troop,0) and (tick%13==0 or rng.random()<.035):
                act('deploy',troop=troop,tile=rng.randrange(60),count=5 if tick%2 else 1)
        act('tick')
        if env.state.is_finished:break
    program="""const fs=require('fs'),Engine=require(process.argv[1]);
const data=JSON.parse(fs.readFileSync(0,'utf8')),env=new Engine(data.payload),out=[];
for(const c of data.commands){env.command(c.kind,c);out.push(env.snapshot());}
process.stdout.write(JSON.stringify(out));"""
    result=subprocess.run([NODE,'-e',program,str(ROOT/'ui/frontend/manual_engine.js')],
        input=json.dumps({'payload':payload,'commands':commands}),text=True,capture_output=True,check=True)
    actual=json.loads(result.stdout)
    assert len(actual)==len(expected)
    for i,(a,e) in enumerate(zip(actual,expected)):
        assert {k:a[k] for k in FIELDS}==e, f'Diverged after command {i}: {commands[i]}; tick {e["tick"]}'
    return expected[-1]


@pytest.mark.parametrize('motif',C.MOTIFS)
@pytest.mark.parametrize('seed',[0,7,42])
def test_every_tick_matches_python(motif,seed):
    compare_sequence(generate_base(seed,motif),{'tank':4,'melee':5,'ranged':4,'breaker':1},seed)


@pytest.mark.parametrize('army',[
    {'melee':25}, {'tank':8}, {'ranged':20}, {'breaker':16}, dict(C.HEURISTIC_ARMY)
])
def test_extreme_armies_and_full_budget(army):
    compare_sequence(generate_base(42,'layered_core'),army,15)


def test_custom_dense_and_wall_free_bases():
    base=generate_base(3,'closed_ring')
    for walls in ((),tuple((r,c) for r in range(1,15) for c in range(1,15) if (r,c) not in {(r,c) for _,r,c in base.buildings})):
        spec=BaseSpec(0,'custom',base.buildings,walls)
        compare_sequence(spec,dict(C.HEURISTIC_ARMY),9)

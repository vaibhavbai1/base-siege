"""Behavioral checks for prototype controls and demo-only custom geometry."""
import copy
import json
from pathlib import Path
import pytest
from env import config as C
from env.combat import BattleState, deploy_troop, record_snapshot, step_tick
from env.generator import generate_base
from env.siege_env import SiegeEnv
from experiments.rollout import play_episode
from agents.heuristic import HeuristicCommander
from ui.manual_game import ManualGame
from ui.custom_bases import custom_spec, layout_from_spec, layout_id, parse_layout, validate_layout

ROOT=Path(__file__).resolve().parents[1]


def game(spec=None):
    env=ManualGame(base_seed=42, record=True, record_ticks=True)
    env.reset(seed=0, options={'spec':spec} if spec else None)
    return env


def test_manual_shopping_refund_batch_and_separate_clock():
    env=game()
    with pytest.raises(ValueError,match='at least one'):env.command('start')
    for _ in range(7):env.command('buy',troop='melee')
    env.command('refund',troop='melee')
    assert env.state.spent==6*C.TROOP_COSTS['melee']
    env.command('start')
    env.command('deploy',troop='melee',tile=7,count=5)
    assert env.state.tick==0 and env.state.troops_total==5 and env.state.reserve['melee']==1
    env.command('tick')
    env.command('deploy',troop='melee',tile=8,count=5)
    assert env.state.tick==1 and env.state.troops_total==6
    for _ in range(9):env.command('tick')
    assert env.state.tick==10
    assert 'reward_totals' not in env.snapshot()
    with pytest.raises(ValueError):env.command('refund',troop='melee')


def test_manual_budget_limit_and_tick_equivalence_to_combat():
    env=game()
    for _ in range(C.RESOURCE_BUDGET//C.TROOP_COSTS['tank']):env.command('buy',troop='tank')
    with pytest.raises(ValueError):env.command('buy',troop='tank')
    env.command('refund',troop='tank')
    env.command('buy',troop='tank')
    state=BattleState.from_spec(env.spec,reserve=dict(env.state.reserve))
    env.command('start')
    for tile in (0,8,20):
        env.command('deploy',troop='tank',tile=tile)
        deploy_troop(state,'tank',C.DEPLOY_TILES[tile])
    for _ in range(24):
        env.command('tick');step_tick(state)
        assert record_snapshot(env.state)==record_snapshot(state)
    # The registered agent interface still advances ten ticks on a deployment.
    agent=SiegeEnv(base_seed=42);agent.reset(seed=0)
    agent.step(0);agent.step(C.ACTION_FINISH_SHOPPING);agent.step(C.ACTION_DEPLOY_FIRST)
    assert agent.state.tick==C.TICKS_PER_DECISION


def layout():
    return layout_from_spec(generate_base(42,'layered_core'))


def test_custom_roundtrip_content_identity_and_play():
    base=layout();renamed=copy.deepcopy(base);renamed['name']='My fortress'
    assert layout_id(base)==layout_id(renamed)
    assert parse_layout(json.dumps(base).encode())==base
    spec=custom_spec(base)
    manual=game(spec);assert manual.spec==spec
    env=SiegeEnv(record=True,record_ticks=True)
    replay=play_episode(env,HeuristicCommander(),options={'spec':spec})
    assert replay['summary']['motif']=='custom' and replay['summary']['finish_reason']
    assert {(b['type'],b['row'],b['col']) for b in replay['snapshots'][0]['buildings']}==set(spec.buildings)|{('wall',r,c) for r,c in spec.walls}


@pytest.mark.parametrize('change',[
    lambda d:d['buildings'].pop(),
    lambda d:d['buildings'].__setitem__(0,['town_hall',0,0]),
    lambda d:d['buildings'].__setitem__(0,['unknown',7,7]),
    lambda d:d['walls'].append(d['buildings'][0][1:]),
    lambda d:d['walls'].append([True,2]),
    lambda d:d['walls'].append([2.5,2]),
    lambda d:d.update(name=''),
    lambda d:d.update(schema='other'),
])
def test_invalid_custom_geometry_is_rejected(change):
    data=layout();change(data)
    with pytest.raises(ValueError):validate_layout(data)


def test_invalid_import():
    for raw in (b'{bad', b'\xff',b' '*32769):
        with pytest.raises(ValueError):parse_layout(raw)


def test_layered_and_custom_app_flow():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(ROOT/'ui/app.py')).run(timeout=30)
    def select(label,value):
        next(w for w in app.selectbox if w.label==label).set_value(value).run(timeout=30)
        assert not app.exception
    select('Base structure','layered_core')
    assert app.session_state['game'].spec.motif=='layered_core'
    select('Mode','Policy replay')
    # Exercise base/editor behavior independently of available checkpoints.
    select('Policy','Heuristic')
    next(b for b in app.button if b.label=='Generate replay').click().run(timeout=30)
    assert not app.error, [error.value for error in app.error]
    assert not app.exception and app.session_state['replay']['summary']['motif']=='layered_core'
    select('Mode','Base editor')
    assert app.get('component_instance')
    draft=copy.deepcopy(app.session_state['editor_draft']);draft['name']='Test custom base'
    app.session_state['base_editor']={'id':'test-save-1','version':app.session_state['editor_version'],
        'layout':draft,'target':'Manual play'}
    app.run(timeout=30)
    assert not app.exception
    assert app.session_state['mode']=='Manual play'
    assert app.session_state['game'].spec==custom_spec(draft)
    select('Mode','Policy replay')
    select('Policy','Heuristic')
    next(b for b in app.button if b.label=='Generate replay').click().run(timeout=30)
    assert not app.exception
    assert not app.error, [error.value for error in app.error]
    assert app.session_state['replay']['summary']['motif']=='custom'
    assert app.session_state['replay']['policy']['custom_base_sha256']==layout_id(draft)

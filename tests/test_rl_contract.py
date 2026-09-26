import json
import numpy as np
import pytest
import gymnasium as gym
from env import config as C
from env.entities import Building, Troop
from env.pathfinding import GridState
from env.combat import BattleState, step_tick
from env.generator import BaseSpec, generate_base
from env.siege_env import SiegeEnv
from env.symmetry import SymmetryWrapper, action_permutation, transform_observation, transform_tile
from agents.heuristic import HeuristicCommander
from agents.random_agent import RandomAgent
from experiments.rollout import play_episode


def test_live_wall_breaching_and_no_overlap():
    buildings = {(r, 1): Building.spawn("wall", (r, 1)) for r in range(16)}
    buildings[(2, 2)] = Building.spawn("storage", (2, 2))
    grid = GridState(buildings)
    troop = Troop.spawn("melee", (2, 0), 1)
    state = BattleState(grid, troops=[troop], total_base_value=10)
    wall = grid.building_at((2, 1))
    for _ in range(25):
        step_tick(state)
        assert not grid.is_wall(troop.position)
    assert wall.hp == 0


def test_terminal_kill_finalises_score_and_milestones():
    building = Building.spawn("town_hall", (2, 2))
    building.hp = 15
    state = BattleState(GridState({(2, 2): building}), troops=[Troop.spawn("melee", (2, 1), 1)], total_base_value=30)
    step_tick(state)
    assert state.destroyed_fraction == 1
    assert all(state.milestones.values())
    assert state.finish_reason == "full_clear"
    step_tick(state)
    assert state.destroyed_value == 30


def test_same_tick_army_death_and_splash_event_coverage():
    cannon = Building.spawn("cannon", (2, 2))
    troop = Troop.spawn("melee", (2, 1), 1)
    troop.hp = 5
    state = BattleState(GridState({(2, 2): cannon}), troops=[troop])
    step_tick(state)
    assert state.finish_reason == "army_eliminated"
    assert state.tick == 1 and not state.troops
    mortar = Building.spawn("mortar", (5, 5))
    units = [Troop.spawn("tank", p, i) for i, p in enumerate([(5, 7), (5, 8), (6, 7)])]
    state = BattleState(GridState({(5, 5): mortar}), troops=units)
    step_tick(state)
    hits = [a for a in state.last_attacks if a["kind"] == "defence"]
    assert sorted(a["damage"] for a in hits) == [18, 18, 45]
    assert len({a["target_id"] for a in hits}) == 3


@pytest.mark.parametrize("seed", range(8))
def test_complete_random_episode_spaces_masks_rewards(seed):
    env = SiegeEnv(record=True)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    total = 0
    while True:
        assert env.observation_space.contains(obs)
        mask = env.action_masks()
        assert mask.dtype == bool and mask.shape == (246,) and mask.any()
        illegal = np.flatnonzero(~mask)
        if len(illegal):
            before = json.dumps(env.snapshot(), sort_keys=True)
            with pytest.raises(ValueError):
                env.step(int(illegal[0]))
            assert json.dumps(env.snapshot(), sort_keys=True) == before
        action = int(rng.choice(np.flatnonzero(mask)))
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward == pytest.approx(sum(info["reward_components"].values()))
        total += reward
        assert not any(env.state.grid.is_wall(t.position) for t in env.state.troops)
        if terminated or truncated:
            assert not truncated
            assert env.observation_space.contains(obs)
            assert info["episode_return"] == pytest.approx(total)
            assert info["reward_totals"]["destruction"] == pytest.approx(info["destruction"])
            assert env.decisions <= 56
            break
    with pytest.raises(gym.error.ResetNeeded):
        env.step(245)


def test_budget_mask_empty_army_and_cost_once():
    env = SiegeEnv()
    env.reset(seed=0)
    for _ in range(25):
        env.step(1)
    assert np.flatnonzero(env.action_masks()).tolist() == [4]
    _, reward, done, _, _ = env.step(4)
    assert reward == -0.3 and not done
    assert env.step(245)[1] == 0
    env.reset(seed=0)
    _, reward, done, _, info = env.step(4)
    assert done and reward == 0 and info["tick"] == 0


def test_heuristic_army_and_repeatability():
    episodes = [play_episode(SiegeEnv(base_seed=42, record=True), HeuristicCommander()) for _ in range(2)]
    assert episodes[0] == episodes[1]
    assert episodes[0]["summary"]["spent"] == 199
    assert len([a for a in episodes[0]["actions"] if a < 4]) == 16


@pytest.mark.parametrize("transform", range(8))
def test_d4_bijection_and_physical_trajectory(transform):
    perm = action_permutation(transform)
    assert sorted(perm) == list(range(246))
    assert np.array_equal(perm[:5], np.arange(5)) and perm[245] == 245
    original = SiegeEnv(base_seed=42, record=True)
    wrapped = SymmetryWrapper(SiegeEnv(base_seed=42, record=True), transform=transform)
    a, _ = original.reset(seed=0)
    b, _ = wrapped.reset(seed=0)
    policy = HeuristicCommander()
    while True:
        assert np.array_equal(b["grid"], transform_observation(a, transform)["grid"])
        assert np.array_equal(wrapped.action_masks()[perm], original.action_masks())
        action = policy.act(a, original.action_masks())
        a, ra, ta, ca, ia = original.step(action)
        b, rb, tb, cb, ib = wrapped.step(int(perm[action]))
        assert (ra, ta, ca) == (rb, tb, cb)
        assert original.snapshot() == wrapped.unwrapped.snapshot()
        if ta or ca:
            break


def test_rotation_index_and_spatial_mapping():
    for i, tile in enumerate(C.DEPLOY_TILES):
        assert transform_tile(tile, 1) == C.DEPLOY_TILES[(i + 15) % 60]
    obs = {"grid": np.arange(8 * 16 * 16).reshape(8, 16, 16), "vector": np.arange(9)}
    for transform in range(8):
        result = transform_observation(obs, transform)
        for r in range(16):
            for c in range(16):
                nr, nc = transform_tile((r, c), transform)
                assert np.array_equal(result["grid"][:, nr, nc], obs["grid"][:, r, c])


@pytest.mark.parametrize("seed", [11000, 11199, 11999])
def test_protected_seed_rejected_before_generation(seed):
    with pytest.raises(PermissionError):
        generate_base(seed, "closed_ring")
    with pytest.raises(PermissionError):
        SiegeEnv(base_seed=seed)
    env = SiegeEnv()
    with pytest.raises(PermissionError):
        env.reset(options={"spec": BaseSpec(seed, "closed_ring", (), ())})


def test_render_and_fixed_army_reset():
    env = SiegeEnv(fixed_army=True, render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    assert obs["vector"][6] == 1 and env.state.spent == 199
    assert env.render().shape[2] == 3
    assert env.step(245)[1] == pytest.approx(-0.2985)
    assert env.step(245)[1] == 0

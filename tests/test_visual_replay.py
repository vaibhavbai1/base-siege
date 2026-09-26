"""Presentation recording must not alter actions, rewards, or combat outcomes."""
import json
import pytest
from env.siege_env import SiegeEnv
from agents.heuristic import HeuristicCommander
from experiments.rollout import play_episode, save_replay, load_replay


@pytest.mark.parametrize('seed,motif', [(42,'closed_ring'),(3,'ns_gates'),(7,'ew_gates')])
def test_tick_recording_preserves_complete_episode(seed,motif):
    plain = play_episode(SiegeEnv(base_seed=seed,motifs=(motif,),record=True),HeuristicCommander())
    visual = play_episode(SiegeEnv(base_seed=seed,motifs=(motif,),record=True,record_ticks=True),HeuristicCommander())
    assert {k:visual[k] for k in plain} == plain
    frames=visual['visual_frames']
    ticks=[f['tick'] for f in frames]
    assert ticks==sorted(ticks)
    assert set(ticks)==set(range(visual['summary']['tick']+1))
    assert frames[-1]['episode_return']==plain['summary']['episode_return']
    assert frames[-1]['destroyed_fraction']==plain['summary']['destruction']
    for frame in frames:
        walls={(b['row'],b['col']) for b in frame['buildings'] if b['type']=='wall'}
        assert all((t['row'],t['col']) not in walls for t in frame['troops'])
    json.dumps(visual,allow_nan=False)


def test_tick_recording_disabled_by_default_and_roundtrip(tmp_path):
    env=SiegeEnv(record=True)
    ordinary=play_episode(env,HeuristicCommander())
    assert env.visual_history==[] and 'visual_frames' not in ordinary
    visual=play_episode(SiegeEnv(record=True,record_ticks=True),HeuristicCommander())
    for name,replay in [('legacy',ordinary),('visual',visual)]:
        path=tmp_path/f'{name}.json';save_replay(replay,path)
        assert load_replay(path)==replay

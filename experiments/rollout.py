"""All policies run through the same Gymnasium decision loop."""
import json
from pathlib import Path


def play_episode(env, policy, *, seed=0, options=None):
    obs, info = env.reset(seed=seed, options=options)
    if hasattr(policy, "reset"):
        policy.reset()
    actions = []
    while True:
        mask = env.action_masks()
        if hasattr(policy, "act"):
            action = policy.act(obs, mask)
        else:
            action, _ = policy.predict(obs, action_masks=mask, deterministic=True)
            action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(int(action))
        if terminated or truncated:
            break
    result = {"schema_version": 1, "summary": info, "actions": actions,
              "snapshots": list(env.unwrapped.history)}
    if getattr(env.unwrapped, "record_ticks", False):
        result["visual_frames"] = list(env.unwrapped.visual_history)
        result["visual_frame_scope"] = "actual simulator ticks; presentation only"
    return result


def save_replay(replay, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(replay, handle, indent=2, allow_nan=False)


def load_replay(path):
    from env.access import ensure_seed_allowed
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != 1 or not data.get("snapshots"):
        raise ValueError("Expected a recorded Base Siege replay")
    ensure_seed_allowed(data["summary"]["base_seed"])
    for frame in data["snapshots"]:
        ensure_seed_allowed(frame["base_seed"])
    return data

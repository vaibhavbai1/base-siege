"""Shared local/public interface; the canvas sends actions to SiegeEnv."""
from pathlib import Path
import hashlib
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st
import streamlit.components.v1 as components
from env import config as C
from env.siege_env import SiegeEnv, describe_action
from agents.heuristic import HeuristicCommander
from agents.random_agent import RandomAgent
from experiments.rollout import play_episode, load_replay
from ui.deployment import asset_paths, load_cpu_policy

PUBLIC_DEMO = globals().get("PUBLIC_DEMO", False)

st.set_page_config(page_title="Base Siege | Project 12", page_icon="🏰", layout="wide")
st.markdown("""<style>.stApp{background:#eeeae0;color:#1d2530} .block-container{max-width:1140px;padding-top:1.7rem}
h1{font-size:1.6rem!important} [data-testid=stSidebar]{background:#fbfaf7} .stButton button{border-radius:8px}
</style>""", unsafe_allow_html=True)
st.title("Base Siege")
st.caption("Project 12 · Dihedrals · Army selection and troop deployment")
if PUBLIC_DEMO:
    st.caption("Interactive demo · CPU inference only · Sessions reset when the app restarts")
component = components.declare_component("siege_board", path=str(ROOT / "ui/frontend"))


def new_game(seed, motif):
    env = SiegeEnv(base_seed=seed, motifs=(motif,), record=True, record_ticks=True)
    env.reset(seed=0)
    return env


with st.sidebar:
    st.header("Session")
    mode = st.selectbox("Mode", ["Manual play", "Policy replay", "Saved replay", "Experiment results"])
    motif = st.selectbox("Base structure", C.TRAIN_MOTIFS)
    seed = st.number_input("Base seed", min_value=0, max_value=10399, value=42, step=1)
    if 10000 <= seed < 10200:
        st.error("Use training seeds 0–9999 or development seeds 10200–10399.")
        st.stop()
    st.caption("Training and development seeds only. The transfer structure and final seeds are reserved for evaluation.")
    st.divider()
    st.markdown("**Game rules**\n\n200 resources · 25 troops maximum · 300 ticks\n\nWalls have no score. Troops move and target automatically after deployment.")

key = (int(seed), motif)
if st.session_state.get("layout_key") != key:
    st.session_state.layout_key = key
    st.session_state.game = new_game(*key)
    st.session_state.game_log = []
    st.session_state.last_event = None
    st.session_state.visual_start = 0
    st.session_state.pop("replay", None)
env = st.session_state.game
error = ""

if mode == "Experiment results":
    st.subheader("Saved experiment evidence")
    paths = asset_paths(ROOT, "manifest", public=PUBLIC_DEMO)
    if not paths:
        st.info("No experiment results have been published yet. The game and baseline agents are available in the other modes."
                if PUBLIC_DEMO else "No experiment manifests have been saved. Use the training notebook or command line to run a software smoke check.")
    else:
        labels = [str(p.relative_to(ROOT)) for p in paths]
        chosen = st.selectbox("Run manifest", labels)
        manifest = json.loads((ROOT / chosen).read_text())
        if manifest["settings"]["mode"] == "smoke":
            st.warning("Software smoke check only. This checkpoint does not establish a learned strategy or a research result.")
        cols = st.columns(4)
        for col, label, value in zip(cols, ["Condition", "Learner seed", "Actual steps", "Status"],
                [manifest["settings"]["condition"], manifest["settings"]["seed"], f"{manifest.get('actual_steps', 0):,}", manifest["status"]]):
            col.metric(label, value)
        episodes_path = (ROOT / chosen).parent / "episodes.jsonl"
        if episodes_path.exists() and episodes_path.stat().st_size:
            episodes = pd.read_json(episodes_path, lines=True)
            st.caption("Training episodes. These curves are not held-out evaluation scores.")
            st.line_chart(episodes.groupby("timesteps")[["destruction", "return"]].mean())
        with st.expander("Configuration and provenance"):
            st.json(manifest)
    evaluation_paths = asset_paths(ROOT, "evaluation", public=PUBLIC_DEMO)
    if evaluation_paths:
        st.subheader("Recorded evaluation")
        chosen_eval = st.selectbox("Evaluation table", [str(p.relative_to(ROOT)) for p in evaluation_paths])
        frame = pd.read_csv(ROOT / chosen_eval)
        st.caption("Evidence scope: " + ", ".join(sorted(frame.evidence_scope.unique())))
        motifs = st.multiselect("Structures", sorted(frame.motif.unique()), default=sorted(frame.motif.unique()))
        filtered = frame[frame.motif.isin(motifs)]
        st.dataframe(filtered.groupby("condition")[["destruction", "town_hall", "full_clear", "return"]].mean(), width="stretch")
        st.caption("Means above are descriptive. For P versus PA uncertainty, use the learner-paired summary from notebook 04.")
        with st.expander("Individual episodes"):
            st.dataframe(filtered, width="stretch")
    st.stop()

readonly = mode != "Manual play"
if mode == "Policy replay":
    policy_name = st.selectbox("Policy", ["Heuristic", "Valid random", "Saved Maskable PPO"])
    model_paths = asset_paths(ROOT, "model", public=PUBLIC_DEMO)
    chosen_model = None
    details = {}
    if policy_name == "Saved Maskable PPO":
        if not model_paths:
            st.info("No trained policy has been published yet. Try the Heuristic or Valid random policy."
                    if PUBLIC_DEMO else "No saved model is available. Run a smoke check or a planned training experiment first.")
        else:
            chosen_model = st.selectbox("Published model" if PUBLIC_DEMO else "Local model",
                                        [str(p.relative_to(ROOT)) for p in model_paths])
            st.caption("Team-selected policy. No model uploads are accepted." if PUBLIC_DEMO
                       else "Load only your own trusted model files.")
            manifest_path = (ROOT / chosen_model).parent / "manifest.json"
            if manifest_path.exists():
                details = json.loads(manifest_path.read_text())
                st.caption(f"{details['evidence_scope']} · {details.get('actual_steps', 0):,} training steps")
                if details.get("settings", {}).get("mode") == "smoke":
                    st.warning("Software smoke check only. This checkpoint does not establish a learned strategy or a research result.")
    if st.button("Generate replay", disabled=policy_name == "Saved Maskable PPO" and not chosen_model):
        with st.spinner("Running one episode in Python"):
            if policy_name == "Heuristic":
                policy = HeuristicCommander()
            elif policy_name == "Valid random":
                policy = RandomAgent(0)
            else:
                model_path = ROOT / chosen_model
                if PUBLIC_DEMO and not details.get("model_sha256"):
                    st.error("This policy needs a valid public export manifest before it can be used.")
                    st.stop()
                try:
                    stamp = model_path.stat()
                    policy = load_cpu_policy(str(model_path), stamp.st_mtime_ns, stamp.st_size,
                                             details.get("model_sha256"))
                except Exception:
                    st.error("The saved policy could not be loaded. Try a baseline policy while the team checks its export.")
                    st.stop()
            replay_env = new_game(*key)
            st.session_state.replay = play_episode(replay_env, policy)
            st.session_state.replay["policy"] = {"name": policy_name,
                "model": chosen_model if policy_name == "Saved Maskable PPO" else None}
            st.session_state.replay_policy = policy_name
    if "replay" not in st.session_state:
        st.info("Choose a policy and generate one recorded episode.")
        st.stop()
    replay = st.session_state.replay
elif mode == "Saved replay":
    candidates = asset_paths(ROOT, "replay", public=PUBLIC_DEMO)
    if not candidates:
        st.info("No saved replays have been published yet. Generate one in Policy replay."
                if PUBLIC_DEMO else "No saved replays found in artifacts. Generate a policy replay or run the replay notebook.")
        st.stop()
    chosen = st.selectbox("Replay file", [str(p.relative_to(ROOT)) for p in candidates])
    try:
        replay = load_replay(ROOT / chosen)
    except (ValueError, PermissionError, KeyError) as exc:
        st.error(str(exc))
        st.stop()

if readonly:
    frames = replay["snapshots"]
    st.caption(f"Recorded base {replay['summary']['base_seed']} · {replay['summary']['motif']} · Use Play below to watch the attack.")
    recorded_policy = replay.get("policy", {})
    st.caption("Recorded policy: " + recorded_policy.get("name", "not specified in this replay") +
               (" · " + recorded_policy["model"] if recorded_policy.get("model") else ""))
    snapshot = frames[0]
    log = [f"Decision {f['decisions']}: {describe_action(f['last_action'])} → {f['destruction']:.1%}"
           for f in frames if f["last_action"] is not None][-40:]
    st.download_button("Download recorded episode", json.dumps(replay, indent=2),
                       file_name="base-siege-replay.json", mime="application/json")
else:
    snapshot, log = env.snapshot(), st.session_state.game_log

payload = {"snapshot": snapshot, "troop_types": C.TROOP_TYPES, "troops": C.TROOP_STATS,
           "buildings": C.BUILDING_STATS, "readonly": readonly, "log": log,
           "action_label": describe_action(snapshot["last_action"]) if snapshot["last_action"] is not None else None,
           "error": st.session_state.get("game_error", "")}
payload["game_key"] = f"{seed}:{motif}:{id(env)}"
if readonly:
    visual = replay.get("visual_frames") or replay["snapshots"]
    payload["replay"] = {"frames": visual, "tick_recorded": bool(replay.get("visual_frames")),
                         "id": hashlib.sha256(json.dumps(replay, sort_keys=True).encode()).hexdigest() + f":{id(replay)}",
                         "label": replay.get("policy", {}).get("name", "Recorded attack")}
else:
    # The browser animates only the newly executed decision, not the entire game.
    payload["manual_frames"] = env.visual_history[st.session_state.get("visual_start", 0):]
event = component(payload=payload, key="battlefield", default=None)
if not readonly and event and event.get("id") != st.session_state.last_event:
    st.session_state.last_event = event["id"]
    try:
        if event["kind"] == "reset":
            st.session_state.game = new_game(*key)
            st.session_state.visual_start = 0
            st.session_state.game_log = []
        elif event["kind"] == "action":
            st.session_state.visual_start = len(env.visual_history)
            _, reward, _, _, info = env.step(event["action"])
            st.session_state.game_log.append(f"{env.decisions}: {describe_action(event['action'])} | reward {reward:+.3f}")
            for hit in env.events:
                st.session_state.game_log.append(f"tick {hit['tick']}: {hit['kind']} hit {hit['to']} for {hit['damage']}")
            st.session_state.game_log = st.session_state.game_log[-60:]
        st.session_state.game_error = ""
    except (ValueError, KeyError, RuntimeError) as exc:
        st.session_state.game_error = str(exc)
    st.rerun()

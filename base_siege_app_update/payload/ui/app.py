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
from env.generator import generate_base
from ui.manual_game import ManualGame
from ui.custom_bases import custom_spec, layout_id, layout_from_spec, validate_layout, parse_layout
from agents.heuristic import HeuristicCommander
from agents.random_agent import RandomAgent
from experiments.rollout import play_episode, load_replay
from ui.deployment import asset_paths, load_cpu_policy
from ui.policy_playback import model_catalog, validate_compatibility, record_policy, html_bytes

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


def new_game(seed, motif, spec=None, *, manual=True):
    cls = ManualGame if manual else SiegeEnv
    env = cls(base_seed=seed, motifs=(motif,) if motif in C.MOTIFS else C.MOTIFS, record=True, record_ticks=True)
    env.reset(seed=0, options={'spec': spec} if spec else {'motif': motif})
    return env


# Apply editor navigation before creating the sidebar widgets.
if 'next_mode' in st.session_state:
    st.session_state.mode = st.session_state.pop('next_mode')
    st.session_state.base_source = 'My custom base'

with st.sidebar:
    st.header("Session")
    mode = st.selectbox("Mode", ["Manual play", "Policy replay", "Base editor", "Saved replay", "Experiment results"], key='mode')
    custom = False
    base_spec = None
    if mode in ('Manual play', 'Policy replay'):
        custom = st.selectbox('Play on', ['Generated base', 'My custom base'], key='base_source') == 'My custom base'
    if custom:
        if 'custom_layout' not in st.session_state:
            st.info('Open Base editor to create or import a custom base first.')
            st.stop()
        base_spec = custom_spec(st.session_state.custom_layout)
        seed, motif = base_spec.seed, base_spec.motif
        st.caption(st.session_state.custom_layout['name'])
        st.caption('Custom demo layout. Agent performance may differ from the evaluated layouts.')
    else:
        motif = st.selectbox("Base structure", C.MOTIFS)
        pool_path = ROOT / "bases/dev-v2.json"
        use_pool = False
        if mode == "Policy replay" and motif in C.TRAIN_MOTIFS and not PUBLIC_DEMO and pool_path.is_file():
            use_pool = st.selectbox("Base source", ["Development pool", "Custom seed"]) == "Development pool"
        if use_pool:
            from experiments.pools import load_pool
            try:
                pool = [b for b in load_pool(pool_path, required_split="dev") if b.motif == motif]
                if not pool:
                    st.error("No accepted development bases for this structure.")
                    st.stop()
                seed = st.selectbox("Development base seed", sorted(b.seed for b in pool))
                base_spec = next(b for b in pool if b.seed == seed)
            except (OSError, ValueError, PermissionError) as exc:
                st.error(f"Cannot load the development pool: {exc}")
                st.stop()
        else:
            seed = st.number_input("Base seed", min_value=0, max_value=10399, value=42, step=1)
        if 10000 <= seed < 10200:
            st.error("Use training seeds 0–9999 or development seeds 10200–10399.")
            st.stop()
        st.caption("All four structures are available for illustrative demo play. Final evaluation seeds remain protected.")
    st.divider()
    st.markdown("**Game rules**\n\n200 resources · 25 troops maximum · 300 ticks\n\nWalls have no score. Troops move and target automatically after deployment.")

if mode == 'Base editor':
    st.subheader('Build your own base')
    st.caption('Move buildings, paint walls, or erase tiles. Keep 1 Town Hall, 3 cannons, 2 mortars and 4 storages. The green edge stays open for deployment.')
    def set_draft(layout):
        st.session_state.editor_draft = layout
        st.session_state.editor_version = st.session_state.get('editor_version', 0) + 1
    if 'editor_draft' not in st.session_state:
        set_draft(st.session_state.get('custom_layout') or layout_from_spec(generate_base(int(seed), motif)))
    cols = st.columns(2)
    if cols[0].button('Load selected structure as template'):
        set_draft(layout_from_spec(generate_base(int(seed), motif)))
    if cols[1].button('Edit saved custom base', disabled='custom_layout' not in st.session_state):
        set_draft(st.session_state.custom_layout)
    imported = st.file_uploader('Import a custom-base JSON', type=['json'])
    if imported is not None and st.button('Load imported base'):
        try:
            set_draft(parse_layout(imported.getvalue()))
        except ValueError as exc:
            st.error(str(exc))
    editor = components.declare_component('siege_editor', path=str(ROOT / 'ui/editor'))
    event = editor(payload={'layout': st.session_state.editor_draft,
                            'version': st.session_state.editor_version,
                            'counts': C.BUILDING_COUNTS}, key='base_editor', default=None)
    if event and event.get('id') != st.session_state.get('editor_last_event'):
        st.session_state.editor_last_event = event.get('id')
        try:
            if event.get('version') != st.session_state.editor_version:
                raise ValueError('The editor template changed. Apply your current draft again.')
            layout = validate_layout(event.get('layout'))
            st.session_state.custom_layout = layout
            st.session_state.editor_draft = layout
            st.session_state.custom_saved = layout['name']
            if event.get('target') in ('Manual play', 'Policy replay'):
                st.session_state.next_mode = event['target']
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    if 'custom_layout' in st.session_state:
        st.success('Saved custom base: ' + st.session_state.custom_layout['name'])
        st.download_button('Download custom base JSON', json.dumps(st.session_state.custom_layout, indent=2),
                           file_name='base-siege-custom-base.json', mime='application/json')
    st.caption('Download your saved base to keep it across app restarts or share it. Template/import buttons replace the editor draft.')
    st.stop()

key = (int(seed), motif, layout_id(st.session_state.custom_layout) if custom else
       hashlib.sha256(base_spec.to_json().encode()).hexdigest() if base_spec else '')
if st.session_state.get("layout_key") != key:
    st.session_state.layout_key = key
    st.session_state.game = new_game(int(seed), motif, base_spec)
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
    model_paths = asset_paths(ROOT, "model", public=PUBLIC_DEMO)
    catalog = model_catalog(model_paths)
    policy_name = st.selectbox("Policy", ["Heuristic", "Valid random", "Saved Maskable PPO"], index=2 if catalog else 0)
    chosen_model = None
    details = {}
    deterministic, action_seed, view = False, 0, 0
    if policy_name == "Saved Maskable PPO":
        if not catalog:
            st.info("No trained policy has been published yet. Try the Heuristic or Valid random policy."
                    if PUBLIC_DEMO else "No saved model is available. Run a smoke check or a planned training experiment first.")
        else:
            conditions = sorted({m['settings']['condition'] for _,m in catalog})
            condition = st.selectbox("Condition", conditions)
            seeds = sorted({m['settings']['seed'] for _,m in catalog if m['settings']['condition']==condition})
            learner_seed = st.selectbox("Learner seed", seeds)
            matching = [(p,m) for p,m in catalog if m['settings']['condition']==condition and m['settings']['seed']==learner_seed]
            chosen_model = st.selectbox("Published model" if PUBLIC_DEMO else "Local model",
                                        [str(p.relative_to(ROOT)) for p,m in matching])
            st.caption("Team-selected policy. No model uploads are accepted." if PUBLIC_DEMO
                       else "Load only your own trusted model files.")
            manifest_path = (ROOT / chosen_model).parent / "manifest.json"
            if manifest_path.exists():
                details = json.loads(manifest_path.read_text())
                st.caption(f"{details['evidence_scope']} · {details.get('actual_steps', 0):,} training steps")
                if details.get("settings", {}).get("mode") == "smoke":
                    st.warning("Software smoke check only. This checkpoint does not establish a learned strategy or a research result.")
            acting_mode = st.selectbox("Action selection", ["Sampled", "Deterministic"])
            deterministic = acting_mode == "Deterministic"
            action_seed = int(st.number_input("Action seed", min_value=0, max_value=2147483647, value=0, step=1,
                                             disabled=deterministic))
            view = st.selectbox("Agent view", list(range(8)))
            st.caption("Sampled uses the learned action probabilities. The same model, base, view and action seed reproduce the same attack.")
            st.caption("Agent view changes the coordinates seen by the policy. Playback shows the original base orientation.")
    settings_key = (key, policy_name, chosen_model, details.get('model_sha256'), deterministic, action_seed, view)
    if st.session_state.get('replay_settings') != settings_key:
        st.session_state.pop('replay', None)
        st.session_state.replay_settings = settings_key
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
                    validate_compatibility(details)
                    stamp = model_path.stat()
                    policy = load_cpu_policy(str(model_path), stamp.st_mtime_ns, stamp.st_size,
                                             details.get("model_sha256"))
                except Exception as exc:
                    st.error("The saved policy could not be loaded. Check its model and manifest files." if PUBLIC_DEMO
                             else f"Could not load model: {exc}")
                    st.stop()
            try:
                if policy_name == "Saved Maskable PPO":
                    generated = record_policy(policy, base_seed=int(seed), motif=motif, spec=base_spec,
                                              view=view, action_seed=action_seed, deterministic=deterministic)
                else:
                    replay_env = new_game(int(seed), motif, base_spec, manual=False)
                    try:
                        generated = play_episode(replay_env, policy, options={'spec':base_spec} if base_spec else None)
                    finally:
                        replay_env.close()
            except Exception as exc:
                st.error("Could not generate this attack. Check the selected model and base." if PUBLIC_DEMO
                         else f"Could not generate attack: {exc}")
                st.stop()
            st.session_state.replay = generated
            replay_name = (f"{details['settings']['condition']} PPO · learner {details['settings']['seed']} · {acting_mode.lower()}"
                           if policy_name == "Saved Maskable PPO" else policy_name)
            st.session_state.replay["policy"] = {"name": replay_name,
                "model": chosen_model if policy_name == "Saved Maskable PPO" else None,
                "model_sha256": details.get('model_sha256'), "acting": generated.get('acting'),
                "evidence_scope": "Illustrative custom-base replay" if custom else "Illustrative generated-base replay",
                "custom_base": st.session_state.custom_layout if custom else None,
                "custom_base_sha256": layout_id(st.session_state.custom_layout) if custom else None}
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
    custom_label = replay.get('policy', {}).get('custom_base')
    if isinstance(custom_label, dict):
        st.caption('Custom base: ' + custom_label.get('name', 'My custom base') + ' · Use Play below to watch the attack.')
    else:
        st.caption(f"Recorded base {replay['summary']['base_seed']} · {replay['summary']['motif']} · Use Play below to watch the attack.")
    recorded_policy = replay.get("policy", {})
    st.caption("Recorded policy: " + recorded_policy.get("name", "not specified in this replay") +
               (" · " + recorded_policy["model"] if recorded_policy.get("model") else ""))
    snapshot = frames[0]
    log = [f"Decision {f['decisions']}: {describe_action(f['last_action'])} → {f['destruction']:.1%}"
           for f in frames if f["last_action"] is not None][-40:]
    st.download_button("Download recorded episode", json.dumps(replay, indent=2),
                       file_name="base-siege-replay.json", mime="application/json")
    st.download_button("Download smooth HTML replay", html_bytes(replay),
                       file_name="base-siege-attack.html", mime="text/html")
    with st.expander("Attack result"):
        result = replay['summary']
        cols = st.columns(4)
        for col,label,value in zip(cols,["Destruction","Town Hall","Full clear","Budget spent"],
            [f"{result['destruction']:.2%}","Destroyed" if result['town_hall'] else "Standing",
             "Yes" if result['full_clear'] else "No",f"{result['spent']} / {C.RESOURCE_BUDGET}"]):
            col.metric(label,value)
        st.caption("One illustrative episode; aggregate evaluation scores are reported separately.")
else:
    snapshot, log = env.snapshot(), st.session_state.game_log
    st.caption("Smooth manual play · Click to buy, Shift-click to refund. Deploy without advancing time; Space runs or pauses the clock. Agent replays retain 10-tick decisions.")

payload = {"snapshot": snapshot, "troop_types": C.TROOP_TYPES, "troops": C.TROOP_STATS,
           "buildings": C.BUILDING_STATS, "readonly": readonly, "log": log,
           "action_label": describe_action(snapshot["last_action"]) if snapshot["last_action"] is not None else None,
           "error": st.session_state.get("game_error", "")}
payload["game_key"] = f"{seed}:{motif}:{id(env)}"
payload["ack_id"] = st.session_state.last_event
if not readonly:
    payload['manual_config'] = {'budget': C.RESOURCE_BUDGET, 'max_army': C.MAX_ARMY_SIZE,
        'tick_limit': C.BATTLE_TICK_LIMIT, 'ticks_per_decision': C.TICKS_PER_DECISION,
        'speed_scale': C.SPEED_SCALE, 'splash_falloff': C.SPLASH_FALLOFF,
        'half_threshold': C.HALF_DESTRUCTION_THRESHOLD, 'defence_types': C.DEFENCE_TYPES,
        'target_fallback': C.TARGET_RULE_FALLBACK, 'deploy_tiles': C.DEPLOY_TILES}
if readonly:
    visual = replay.get("visual_frames") or replay["snapshots"]
    payload["replay"] = {"frames": visual, "tick_recorded": bool(replay.get("visual_frames")),
                         "id": hashlib.sha256(json.dumps(replay, sort_keys=True).encode()).hexdigest(),
                         "label": replay.get("policy", {}).get("name", "Recorded attack")}
else:
    # The browser animates only the newly executed decision, not the entire game.
    payload["manual_frames"] = env.visual_history[st.session_state.get("visual_start", 0):]
event = component(payload=payload, key="battlefield", default=None)
if (not readonly and event and event.get("id") != st.session_state.last_event
        and event.get("game_key") == payload["game_key"]):
    st.session_state.last_event = event["id"]
    try:
        if event["kind"] == "reset":
            st.session_state.game = new_game(int(seed), motif, base_spec)
            st.session_state.visual_start = 0
            st.session_state.game_log = []
        else:
            st.session_state.visual_start = len(env.visual_history)
            message = env.command(event['kind'], troop=event.get('troop'),
                                  tile=event.get('tile'), count=event.get('count', 1))
            st.session_state.game_log.append(message)
            for hit in env.events:
                st.session_state.game_log.append(f"tick {hit['tick']}: {hit['kind']} hit {hit['to']} for {hit['damage']}")
            st.session_state.game_log = st.session_state.game_log[-60:]
        st.session_state.game_error = ""
    except (ValueError, KeyError, RuntimeError) as exc:
        st.session_state.game_error = str(exc)
    st.rerun()

"""Study narrative and bounded experiments for the Base Siege notebooks."""

SETUP = '''
from pathlib import Path
import os
import sys
import json
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

candidate = Path(os.environ.get("BASE_SIEGE_ROOT", Path.cwd())).expanduser()
ROOT = next((p for p in [candidate, *candidate.parents] if (p / "env/siege_env.py").is_file()), None)
if ROOT is None:
    raise FileNotFoundError("Open the complete project, or set BASE_SIEGE_ROOT to its folder.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from env import config as C
from env.siege_env import SiegeEnv, describe_action, deployment_action
from env.render import draw_snapshot
from env.generator import generate_base
from experiments.provenance import runtime_manifest, sha256
RUNS = Path(os.environ.get("BASE_SIEGE_RUNS_DIR", ROOT / "runs")).expanduser().resolve()
OUTPUT = ROOT / "artifacts" / "notebook-sessions" / uuid.uuid4().hex[:12]
OUTPUT.mkdir(parents=True, exist_ok=False)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 12, "axes.labelsize": 10, "font.size": 10})
def save_figure(name):
    plt.savefig(OUTPUT / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.show()
print(f"Project: {ROOT.name} | Python: {sys.version.split()[0]}")
print(f"This notebook's outputs: {OUTPUT}")
'''

NOTEBOOKS = {
"00_sagemaker_setup.ipynb": [
"""# 00. SageMaker Studio setup and run management

UTS 43008 Reinforcement Learning · Assignment 3 · Project 12 · Dihedrals

Check the kernel, project files and storage before spending time on training.
The project uses local files in your persistent Studio project folder. It does not use S3.

The current default study has twenty full learners, preceded by a separate pilot pair and team
review. Read the [team master](../docs/TEAM_MASTER.md) and [runbook](../docs/RUNBOOK.md) before
starting that workflow. The study controller and evaluation journals support compatible resumes.

The supplied Studio screenshots show these choices:

| Instance | CPU / system RAM | GPU | Role |
|---|---|---|---|
| ml.t3.medium | 2 vCPU / 4 GiB | None | Light reading and checks |
| ml.g4dn.xlarge | 4 vCPU / 16 GiB | One T4 | Starting point for the pilot |
| ml.g4dn.2xlarge | 8 vCPU / 32 GiB | One T4 | More CPU/RAM if measured simulation load needs it |

The screenshots show 50 GB of persistent storage and an administrator-controlled 60-minute idle
setting. That is not ordinarily a one-hour training limit; confirm the course lifecycle script's
behaviour rather than disabling it. Read the image-upgrade warning and back up files before
updating the image. A GPU selection alone does not prove this kernel has compatible CUDA.

Read 00, 01 and 02 first. Complete the short recovery lab in 03b before starting full experiments
in 03. Notebooks 04 and 05 cover evaluation and the demonstration.

## Before the first code cell

Open the complete project, not just the notebooks. Run these commands in a Studio terminal from
the project folder. They install a project environment; this notebook does not run them automatically.

~~~bash
uv sync --frozen
uv run python -m ipykernel install --user --name base-siege --display-name "Python 3 (Base Siege)"
~~~

Select **Python 3 (Base Siege)** in the kernel menu. Installation needs internet access, Python
3.12 and enough disk space. A GPU instance alone does not prove that the kernel can use CUDA.
""",
("md", """## 1. What needs to stay on disk

A Studio JupyterLab space uses an attached EBS volume for its files. Stopping and later starting
the same space preserves that volume. Studio Classic has a different storage layout, so confirm
where your project is stored. Do not use a temporary folder for the only copy of a run.

| Keep together | Purpose |
|---|---|
| Source, notebooks and uv.lock | Recreate the program and packages |
| Pools and registration | Preserve the agreed layouts and protocol |
| Each complete run folder | Recover model, optimiser, games, RNGs and history |
| Evaluation CSVs and manifests | Trace figures back to individual episodes |

Stopping compute is different from deleting a space or its storage. Local checkpoints are not an
independent backup. This project does not copy them to another service.

Source: [AWS Studio JupyterLab spaces](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-jl-user-guide.html).
"""),
("code", '''
import platform
import shutil
import torch
from importlib.metadata import version

packages = ["numpy", "torch", "gymnasium", "stable-baselines3", "sb3-contrib", "nbformat"]
display(pd.DataFrame({"package": packages, "version": [version(p) for p in packages]}))
print("Kernel:", sys.executable)
print("Platform:", platform.platform())
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), "| CUDA:", torch.version.cuda)
print(f"Free project-volume space: {shutil.disk_usage(ROOT).free / 1024**3:.1f} GiB")
assert sys.version_info[:2] == (3, 12), "Use the project's Python 3.12 kernel."
'''),
("md", """## 2. Choose persistent run storage

The default is the runs folder inside this project. Set BASE_SIEGE_RUNS_DIR before launching
Jupyter only if you need a different persistent location. Avoid temporary folders, container
caches and scratch disks.

This cell tests access; it cannot prove your AWS retention settings. It creates and removes
only its own tiny storage-probe file.
"""),
("code", '''
import tempfile
RUNS.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(prefix=".storage-probe-", dir=RUNS) as handle:
    handle.write(b"base-siege-storage-check")
    handle.flush()
    os.fsync(handle.fileno())
print("Run directory:", RUNS)
if str(RUNS).startswith(("/tmp/", "/private/tmp/")):
    raise RuntimeError("Choose persistent project storage, not /tmp.")
print("Writable. Confirm volume persistence in Studio.")
'''),
("md", """## 3. Check the project

Run the complete tests and a short recovery drill in a terminal. They do not start an AWS instance,
create cloud resources, open final layouts or run the full study.

~~~bash
uv run pytest -q
uv run python -m tools.check_recovery --out artifacts/my-recovery-check --kill-process
~~~

Use a new output folder each time. The second command kills only its own short test process,
then compares resumed training against uninterrupted training.
"""),
("code", '''
required = ["env/siege_env.py", "train/checkpoints.py", "train/train_ppo.py",
            "experiments/protocol.py", "uv.lock", "reference/base_siege_stage5_sync.zip"]
inventory = pd.DataFrame({"file": required, "present": [(ROOT / p).is_file() for p in required]})
display(inventory)
assert inventory.present.all()
print("Original ZIP SHA-256:", sha256(ROOT / required[-1]))
'''),
("md", """## 4. Understand the units

A simulator tick is one movement/combat update. An environment transition is one agent decision.
Shopping consumes decisions but no combat ticks. PPO gathers a rollout from every worker and
then updates the policy using several passes over that data.

The GPU handles neural-network operations. The Python simulator runs on CPUs. This small policy
may be limited by simulation or process overhead, so measure short CPU and CUDA runs on the target
machine before fixing the study setup.
"""),
("code", '''
from train.train_ppo import TrainingSettings, phase_budgets
full = TrainingSettings(placement_steps=C.CURRICULUM_PLACEMENT_TIMESTEPS,
    joint_steps=C.CURRICULUM_JOINT_TIMESTEPS, n_envs=C.N_TRAINING_ENVS,
    rollout_steps=C.PPO_N_STEPS, batch_size=C.PPO_BATCH_SIZE, epochs=C.PPO_N_EPOCHS, mode="full")
display(pd.Series(phase_budgets(full), name="transitions"))
print("Default checkpoint spacing:", phase_budgets(full)["rollout_transitions"], "transitions")
'''),
("md", """## 5. Stop and resume deliberately

For a planned break, request a pause from a second terminal. Wait for status=paused in the run
manifest, then stop Studio compute. An incomplete rollout is discarded; the last committed
update remains recoverable.

~~~bash
uv run python -m train.train_ppo --request-pause --out runs/P-0
uv run python -m train.train_ppo --resume --out runs/P-0
~~~

After an unexpected shutdown, reopen the same project and resume the same run directory.
The trainer checks saved settings, source, packages and input hashes. A completed run is not
trained again.

With the default cadence, an abrupt stop can lose unsaved work from one rollout and its
in-progress update. The full configuration has 8,192 transitions per rollout. Zero lost work
cannot be promised.
"""),
("code", '''
records = []
for path in sorted(RUNS.glob("*/manifest.json")):
    data = json.loads(path.read_text())
    records.append({"run": path.parent.name, "status": data.get("status"),
        "committed_steps": data.get("actual_steps", 0), "phase": data.get("current_phase"),
        "attempts": len(data.get("attempts", []))})
display(pd.DataFrame(records) if records else pd.DataFrame(columns=["run", "status", "committed_steps", "phase", "attempts"]))
'''),
("md", """## 6. Before a long run

- Run the recovery lab again in the Studio kernel you will train with.
- Confirm CUDA works if you choose the CUDA device; there is no silent CPU fallback.
- Keep the same source and locked packages throughout the experiment.
- Check disk space. Older checkpoint generations are kept; nothing is pruned automatically.
- Register learner seeds and filtered pools before full training.
- Do not use smoke policies as performance evidence.

Same-machine CPU continuation is tested. Equality across GPU types, drivers or package versions
is not promised. Read [the recovery guide](../docs/SAGEMAKER_STUDIO.md).
"""),
],
"01_environment_and_layouts.ipynb": [
"""# 01. Environment, layouts and the RL problem

UTS 43008 · Assignment 3 · Project 12 · Dihedrals

Imagine a small board game. You have 200 coins to buy an army. You choose where and when troops
enter a defended base. Troops then move and fight using fixed rules. The learner is the commander,
not the individual soldier.

This notebook connects that picture to the actual observations, actions and rewards. It uses
training layouts only. Vaibhav's Stage 5 simulator is the source; changes are recorded in
[Decisions](../docs/DECISIONS.md). The original ZIP is preserved unchanged.
""",
("md", """## 1. The learning problem

| RL term | Meaning here |
|---|---|
| Environment | Layout, army, movement, combat and episode rules |
| Agent | Commander choosing purchases and deployment |
| Observation | Eight board layers and nine summary values |
| Action | Buy, finish shopping, deploy or wait |
| Reward | New destruction and milestones, minus army cost |
| Episode | Shopping followed by one battle |
| Policy | A rule mapping observations to action probabilities |

Training means repeatedly playing, measuring rewards and adjusting the policy. A deterministic
simulator can produce varied experience because layouts, actions and augmented views are sampled.
"""),
("code", '''
troops = pd.DataFrame(C.TROOP_STATS).T
display(troops[["hp", "damage_per_tick", "range", "speed", "cost", "target_rule"]])
display(pd.DataFrame(C.BUILDING_STATS).T)
print("Board:", C.GRID_SIZE, "by", C.GRID_SIZE)
print("Budget:", C.RESOURCE_BUDGET, "| Army limit:", C.MAX_ARMY_SIZE)
'''),
("md", """## 2. Why four troop types?

The tank absorbs damage and prefers defences. Melee troops are cheap attackers. Ranged troops
can attack from four tiles away, including across walls. Breakers target walls and are consumed
by their attack. There is no manual steering after deployment.

Movement uses four-neighbour weighted A*. Combat range uses Euclidean distance. Walls slow
movement but do not block shots. These are separate rules, not inconsistent distance formulas.
"""),
("code", '''
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for motif, ax in zip(C.TRAIN_MOTIFS, axes):
    board = SiegeEnv(base_seed=42, motifs=(motif,))
    board.reset(seed=0)
    draw_snapshot(board.snapshot(), ax, style="presentation", show_hud=False)
    ax.set_title(motif.replace("_", " ").title())
    board.close()
plt.tight_layout()
save_figure("training-layout-families")
'''),
("md", """## 3. Procedural layouts, not a downloaded dataset

A seed and a wall structure determine a base. The same pair recreates the same layout. Building
positions must be unique, inside the allowed region and away from walls and deployment tiles.
The three structures above appear in training and current development, with separate seeds.
Layered-core is reserved for the final stage of the current workflow. Earlier software checks
included it in development; that history is disclosed, not erased. Final-layout seeds remain protected.

Generation creates candidates. Some fail additional structural requirements, such as Town Hall
coverage. The table records those rejections instead of assuming every candidate is accepted.
"""),
("code", '''
from experiments.pools import validate_structure
layout_rows = []
for seed in range(20):
    for motif in C.TRAIN_MOTIFS:
        spec = generate_base(seed, motif)
        valid, reason = validate_structure(spec)
        layout_rows.append({"seed": seed, "motif": motif, "walls": len(spec.walls),
                            "buildings": len(spec.buildings), "valid": valid, "reason": reason})
layouts = pd.DataFrame(layout_rows)
display(layouts.groupby("motif").agg(candidates=("seed", "size"), valid=("valid", "sum"),
                                     walls=("walls", "mean"), buildings=("buildings", "mean")))
display(layouts.loc[~layouts.valid].groupby(["motif", "reason"]).size().rename("rejected_candidates"))
assert len(layouts) == 20 * len(C.TRAIN_MOTIFS)
'''),
("md", """## 4. What the policy sees

The spatial input has shape (8, 16, 16). Five layers locate building types; the remaining layers
show building health, defence coverage and troop density. The vector contains four reserve counts,
remaining resources, remaining time, battle phase, destruction and living troops.

Inputs are float32 values in [0, 1]. Normalisation gives quantities comparable numerical scales.
The policy does not receive the renderer's complete internal state.
"""),
("code", '''
env = SiegeEnv(base_seed=42, motifs=("closed_ring",))
obs, info = env.reset(seed=0)
display(pd.DataFrame([{"input": k, "shape": str(v.shape), "dtype": str(v.dtype),
                       "minimum": float(v.min()), "maximum": float(v.max())} for k, v in obs.items()]))
assert env.observation_space.contains(obs)
names = ["Town Hall", "Cannon", "Mortar", "Storage", "Wall", "Health", "Coverage", "Troop density"]
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for layer, name, ax in zip(obs["grid"], names, axes.flat):
    ax.imshow(layer, vmin=0, vmax=1, cmap="viridis")
    ax.set_title(name)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
save_figure("observation-layers")
'''),
("code", '''
vector_names = ["tank reserve", "melee reserve", "ranged reserve", "breaker reserve",
                "budget remaining", "time remaining", "battle phase", "destruction", "troops alive"]
display(pd.DataFrame({"field": vector_names, "normalised_value": obs["vector"]}))
'''),
("md", """## 5. Is this fully observable?

No. Troop density does not separately identify each troop's type, health, target or movement
progress. Different internal states can look identical to the policy. Strictly, this is a
partially observed decision problem.

The feed-forward CNN is a compact baseline, without memory of previous observations. A recurrent
policy could be a future experiment; it is not implemented or claimed here.
"""),
("md", """## 6. Decode the 246 actions

| IDs | Meaning | Count |
|---|---|---:|
| 0–3 | Buy tank, melee, ranged or breaker | 4 |
| 4 | Finish shopping | 1 |
| 5–244 | Four troop types × 60 perimeter positions | 240 |
| 245 | Wait | 1 |

The mask prevents illegal actions, not poor strategy. Deploying where a cannon can hit may be
legal even when it is a bad choice.
"""),
("code", '''
ids = [0, 3, 4, 5, 64, 65, 244, 245]
display(pd.DataFrame({"id": ids, "meaning": [describe_action(a) for a in ids]}))
trace = []
for action in [0, 2, 4, deployment_action("tank", 7), C.ACTION_WAIT]:
    legal_before = int(env.action_masks().sum())
    _, reward, terminated, truncated, step_info = env.step(action)
    trace.append({"action": describe_action(action), "legal_before": legal_before,
        "tick": step_info["tick"], "spent": step_info["spent"], "reward": reward,
        "terminated": terminated, "truncated": truncated})
display(pd.DataFrame(trace))
env.close()
'''),
("md", """## 7. A worked reward example

Reward is not the same as destruction. Destroyed non-wall value contributes its fraction of 155
total value. Crossing 50% destruction, destroying the Town Hall and clearing the base each add
0.25 once. Army cost contributes −0.30 × spent / 200, charged once as battle begins.

Destroying walls gives no direct reward, but can help troops reach valuable targets. Gamma is 1.0,
so rewards within this finite episode are not discounted.
"""),
("code", '''
example = pd.DataFrame([
    {"component": "Destroyed value: 80 of 155", "reward": 80 / C.TOTAL_BASE_VALUE},
    {"component": "Half-destruction milestone", "reward": C.MILESTONE_HALF_DESTRUCTION},
    {"component": "Other milestones not achieved", "reward": 0.0},
    {"component": "Army cost: 100 resources", "reward": -C.COST_PENALTY_WEIGHT * 100 / C.RESOURCE_BUDGET},
])
display(example)
print(f"Illustrative total return: {example.reward.sum():.4f}")
print(f"Full clear with 200 spent: {1 + .25*3 - .30:.2f}")
'''),
("md", """## 8. Episode endings

A cleared base, battle deadline or exhausted army is a normal terminal outcome. The 300-tick
deadline belongs to the task, so it is termination. The 70-decision safety guard is truncation;
value bootstrapping treats it differently.

Battle decisions advance up to 10 ticks, stopping early if the battle ends. Shopping advances
zero combat ticks. PPO transitions and combat ticks are therefore not interchangeable.
"""),
("code", '''
empty = SiegeEnv(base_seed=42)
empty.reset(seed=0)
_, reward, terminated, truncated, end = empty.step(C.ACTION_FINISH_SHOPPING)
display(pd.Series({"return": reward, "terminated": terminated, "truncated": truncated,
                   "finish_reason": end["finish_reason"], "ticks": end["tick"]}))
assert terminated and not truncated and end["tick"] == 0
empty.close()
'''),
("md", """## 9. Follow the source

1. env/generator.py constructs a seeded base.
2. env/siege_env.py implements reset, masks, step, observation and reward.
3. env/combat.py moves troops, chooses attacks, applies damage and checks endings.
4. env/pathfinding.py estimates travel costs including walls.

Explain why a wall can be strategically important when wall reward is zero.
"""),
],
"02_baselines_and_validation.ipynb": [
"""# 02. Baselines, validation and exploratory analysis

UTS 43008 · Assignment 3 · Project 12 · Dihedrals

Before asking whether PPO learns, check that the game is reproducible and that the comparison
policies behave as intended. This notebook uses small development samples. Its plots are
diagnostics, not final assignment results.

How strong is the scripted commander? How variable is valid-random play? Which layouts survive
the planned difficulty filter?
""",
("md", """## 1. Two policies with different purposes

Valid-random samples legal actions only. It does not sample illegal actions and receive an
invented penalty. The heuristic buys a fixed 199-resource army, chooses a side using defence
coverage and follows a scripted deployment order.

Neither policy learns. PPO must be compared with both on identical layouts and views.
"""),
("code", '''
from agents.heuristic import HeuristicCommander
from agents.random_agent import RandomAgent
from experiments.rollout import play_episode
from time import perf_counter

army = pd.Series(C.HEURISTIC_ARMY, name="count")
display(pd.DataFrame({"count": army, "unit_cost": pd.Series(C.TROOP_COSTS)}))
print("Army cost:", sum(C.HEURISTIC_ARMY[t] * C.TROOP_COSTS[t] for t in C.TROOP_TYPES))
'''),
("md", """## 2. Matched development episodes

Use six seed/structure pairs and three random action seeds. Repeated random seeds describe
action variability; they are not independent trained learners. Save raw rows before averaging.
"""),
("code", '''
rows = []
for base_seed in range(10200, 10206):
    for label, action_seed in [("heuristic", -1)] + [("random", s) for s in range(3)]:
        policy = HeuristicCommander() if label == "heuristic" else RandomAgent(action_seed)
        board = SiegeEnv(base_seed=base_seed, motifs=("closed_ring",))
        started = perf_counter()
        result = play_episode(board, policy)["summary"]
        rows.append({"agent": label, "base_seed": base_seed, "action_seed": action_seed,
            "destruction": result["destruction"], "return": result["episode_return"],
            "spent": result["spent"], "ticks": result["tick"], "finish_reason": result["finish_reason"],
            "seconds": perf_counter() - started})
        board.close()
results = pd.DataFrame(rows)
results.to_csv(OUTPUT / "baseline-development-episodes.csv", index=False)
display(results.head(8))
'''),
("code", '''
display(results.groupby("agent").agg(episodes=("base_seed", "size"),
    mean_destruction=("destruction", "mean"), std_destruction=("destruction", "std"),
    mean_return=("return", "mean"), mean_spent=("spent", "mean"), seconds=("seconds", "sum")))
display(pd.crosstab(results.agent, results.finish_reason))
'''),
("md", """## 3. Look beyond a single mean

The left plot compares policies on matching layouts. The right shows the spread of episode
outcomes. Six layouts do not establish general performance. These unfiltered demonstration
bases are a different population from the filtered study pool.
"""),
("code", '''
per_base = results.groupby(["base_seed", "agent"]).destruction.mean().unstack()
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
per_base.plot.bar(ax=axes[0], rot=0)
axes[0].set(xlabel="Development layout seed", ylabel="Mean destruction", ylim=(0, 1))
for label, group in results.groupby("agent"):
    values = np.sort(group.destruction.to_numpy())
    axes[1].step(values, np.arange(1, len(values)+1)/len(values), where="post", label=label)
axes[1].set(xlabel="Episode destruction", ylabel="Fraction of episodes at or below value", xlim=(0, 1))
axes[1].legend()
plt.tight_layout()
save_figure("baseline-diagnostics")
'''),
("md", """## 4. Test determinism

An identical layout, policy and action seed should reproduce the actions and outcome. Runtime
measurements are not deterministic and are deliberately excluded from the equality check.
"""),
("code", '''
def diagnostic_episode(action_seed):
    board = SiegeEnv(base_seed=10200, motifs=("closed_ring",))
    try:
        return play_episode(board, RandomAgent(action_seed))
    finally:
        board.close()
first, repeat = diagnostic_episode(17), diagnostic_episode(17)
assert first == repeat
display(pd.Series({"actions": len(first["actions"]), "same_seed_identical": first == repeat,
    "different_seed_actions_differ": first["actions"] != diagnostic_episode(18)["actions"]}))
'''),
("md", """## 5. Structural validity is not difficulty

Structural checks reject overlaps, buildings on walls, incorrect building counts, an unprotected
Town Hall or empty required compartments.

The difficulty test attacks each valid base from four sides with a heuristic and takes the best
destruction fraction. The study keeps values from 0.25 to 0.85 inclusive. That four-side maximum
is a layout-filter rule, not the heuristic's reported baseline score.
"""),
("code", '''
from experiments.pools import validate_structure, validate_base
checks = []
for base_seed in range(12):
    spec = generate_base(base_seed, "closed_ring")
    structural, _ = validate_structure(spec)
    accepted, score, reason = validate_base(spec)
    checks.append({"base_seed": base_seed, "structural": structural, "accepted": accepted,
                   "best_side_destruction": score, "reason": reason})
filter_results = pd.DataFrame(checks)
display(filter_results)
display(filter_results.groupby("accepted").size().rename("candidate_count"))
'''),
("code", '''
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.axhspan(C.DIFFICULTY_MIN, C.DIFFICULTY_MAX, alpha=.12, color="green", label="Accepted interval")
ax.scatter(filter_results.base_seed, filter_results.best_side_destruction,
           c=np.where(filter_results.accepted, "#2a8065", "#b14c45"))
ax.set(xlabel="Candidate layout seed", ylabel="Best four-side heuristic destruction", ylim=(0, 1))
ax.legend()
save_figure("difficulty-filter")
'''),
("md", """## 6. The filter changes the population

A result on accepted bases is not evidence that the policy handles every generated base.
Report candidate counts, acceptance rates and rejection reasons by structure. Do not keep
sampling until a preferred policy gets an attractive score.

Small diagnostic pools are for software checks. Full pools must be built and frozen separately.
"""),
("code", '''
display(pd.DataFrame([
    {"partition": "training", "first": C.TRAIN_SEED_RANGE[0], "last": C.TRAIN_SEED_RANGE[1], "purpose": "learn parameters"},
    {"partition": "development", "first": C.DEV_SEED_RANGE[0], "last": C.DEV_SEED_RANGE[1], "purpose": "choose the setup"},
    {"partition": "final", "first": C.FINAL_STUDY_SEED_RANGE[0], "last": C.FINAL_STUDY_SEED_RANGE[1], "purpose": "one frozen evaluation"}
]))
print("Ranges only. No final layouts were generated.")
'''),
("md", """## 7. Regression checks and viva questions

Run the full suite with the command below. It checks mechanics, masks, reward charging, D4
mappings and protected seed ranges. Passing tests is not evidence of a good learned policy.

~~~bash
uv run pytest -q
~~~

Explain the difference between a layout seed, action seed and learner seed. Why can ten random
action seeds not replace five independent PPO training runs?
"""),
],
}


NOTEBOOKS.update({
"03_maskable_ppo_training.ipynb": [
"""# 03. Maskable PPO: architecture and training workflow

UTS 43008 · Assignment 3 · Project 12 · Dihedrals

Does training with rotated and reflected coordinate views improve generalisation when the
learner, layouts and actual training budget are otherwise matched?

P is standard Maskable PPO. PA is the same method with D4 view augmentation at reset. Our custom
work covers the game, RL interface, feature extractor, transformations and experiment workflow.
PPO itself is supplied by SB3 Contrib.

Full training is disabled here. Run the short recovery lab in **03b** before the long experiments.
""",
("md", """## 1. Observation to decision

The CNN extracts features from eight board layers. A linear layer processes the nine summary
values. Their outputs are concatenated into 256 shared features. Separate heads estimate action
preferences and state value.

The action head has 246 outputs. Illegal actions are masked before sampling. The value head
estimates expected future return, not the current destruction percentage.
"""),
("code", '''
import torch
from train.features import SiegeFeatures
from train.train_ppo import TrainingSettings, phase_budgets, make_env, train, resume
from train.checkpoints import RecoverablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from dataclasses import asdict

torch.set_num_threads(1)
board = SiegeEnv()
observation, _ = board.reset(seed=0)
extractor = SiegeFeatures(board.observation_space)
display(extractor)
with torch.no_grad():
    features = extractor({k: torch.as_tensor(v).unsqueeze(0) for k, v in observation.items()})
print("Feature output shape:", tuple(features.shape))
print("Feature extractor parameters:", sum(p.numel() for p in extractor.parameters()))
board.close()
'''),
("code", '''
display(pd.DataFrame([
    ["Grid input", "8 × 16 × 16"], ["Conv 3×3, stride 2, ReLU", "32 × 8 × 8"],
    ["Conv 3×3, stride 2, ReLU", "64 × 4 × 4"], ["Flatten", "1,024"],
    ["Vector linear + ReLU", "9 → 32"], ["Concatenate + linear + ReLU", "1,056 → 256"],
    ["Policy head", "256 → 128 → 246"], ["Value head", "256 → 128 → 1"]
], columns=["operation", "dimensions"]))
'''),
("md", """## 2. What PPO changes

PPO collects actions, rewards, old action probabilities and value estimates. It computes
advantages: was an outcome better or worse than expected? It updates the policy using a clipped
probability ratio, discouraging large changes from one batch.

Positive advantage encourages an action; negative advantage discourages it. Clipping is a
training objective, not a strict guarantee on every action probability.
"""),
("code", '''
ratio = np.linspace(.5, 1.5, 201)
epsilon = .2
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for advantage, ax in zip([1., -1.], axes):
    ordinary = ratio * advantage
    clipped = np.clip(ratio, 1-epsilon, 1+epsilon) * advantage
    ax.plot(ratio, ordinary, "--", label="Unclipped")
    ax.plot(ratio, np.minimum(ordinary, clipped), label="PPO clipped")
    ax.axvline(1, color="grey", alpha=.4)
    ax.set(title=f"Advantage = {advantage:+.0f}", xlabel="New / old action probability", ylabel="Objective to maximise")
    ax.legend()
plt.tight_layout()
save_figure("ppo-clipping-example")
'''),
("md", """## 3. A worked advantage calculation

The one-step residual is reward + gamma × next_value − current_value. Generalised advantage
estimation combines residuals backwards through the rollout using gamma and lambda. Terminal
states have no future value. Truncated transitions may need bootstrapping; the library uses
the environment's ending signals.

These numbers illustrate the mathematics. They are not training results.
"""),
("code", '''
rewards = np.array([0.0, .2, .5])
values = np.array([.1, .15, .3, 0.0])
residuals = rewards + C.PPO_GAMMA * values[1:] - values[:-1]
advantages = np.zeros_like(rewards)
carry = 0.0
for i in reversed(range(len(rewards))):
    carry = residuals[i] + C.PPO_GAMMA * C.PPO_GAE_LAMBDA * carry
    advantages[i] = carry
display(pd.DataFrame({"reward": rewards, "value": values[:-1], "TD residual": residuals,
    "GAE advantage": advantages, "return target": advantages + values[:-1]}))
'''),
("md", """## 4. Inspect a masked prediction

This initialised model has not learned. We only check that the selected action is legal.
Masks must be supplied during evaluation as well as training.
"""),
("code", '''
vector = DummyVecEnv([make_env])
initial = RecoverablePPO("MultiInputPolicy", vector, seed=0, device="cpu", n_steps=32,
    batch_size=32, n_epochs=2, policy_kwargs={"features_extractor_class": SiegeFeatures,
    "normalize_images": False, "net_arch": {"pi": [128], "vf": [128]}})
observation = vector.reset()
mask = np.stack(vector.env_method("action_masks"))
action, _ = initial.predict(observation, action_masks=mask, deterministic=True)
print("Initial legal choice:", describe_action(int(action[0])))
print("Legal actions:", int(mask.sum()))
assert mask[0, action[0]]
vector.close()
'''),
("md", """## 5. What D4 augmentation does

A square has eight rotation/reflection views. PA samples one at reset and transforms the board,
deployment coordinates and mask together. Shopping actions and summary values stay unchanged.

The physical battle remains in its original coordinates. The simulator's coordinate tie-breaks
are not claimed to be rotation-equivariant. A separate view RNG prevents augmentation from
changing the sequence of sampled layouts.
"""),
("code", '''
from env.symmetry import action_permutation, transform_observation
board = SiegeEnv(base_seed=42, motifs=("closed_ring",))
observation, _ = board.reset(seed=0)
fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
mapping = []
for view, ax in enumerate(axes.flat):
    permutation = action_permutation(view)
    assert np.array_equal(np.sort(permutation), np.arange(C.N_ACTIONS))
    transformed = transform_observation(observation, view)
    ax.imshow(transformed["grid"][:5].sum(axis=0), cmap="cividis")
    ax.set_title(f"View {view}")
    ax.set_xticks([])
    ax.set_yticks([])
    mapping.append({"view": view, "physical_action": 12, "view_action": int(permutation[12])})
plt.tight_layout()
save_figure("d4-views")
display(pd.DataFrame(mapping))
board.close()
'''),
("md", """## 6. Curriculum and matched budgets

Placement training supplies the heuristic army. Joint training starts new shopping episodes and
learns purchases as well as deployment. Policy and optimiser continue across the phase boundary.

Revision 2 directly requests 32,768 + 229,376 = 262,144 transitions. These are four placement
rollouts and twenty-eight joint rollouts with four workers and 2,048 steps per worker.
The old draft requested 250,000, which rounded to this count. Both methods receive equal budgets.
"""),
("code", '''
full_settings = TrainingSettings(mode="full", device="cuda",
    placement_steps=C.CURRICULUM_PLACEMENT_TIMESTEPS, joint_steps=C.CURRICULUM_JOINT_TIMESTEPS,
    n_envs=C.N_TRAINING_ENVS, rollout_steps=C.PPO_N_STEPS, batch_size=C.PPO_BATCH_SIZE, epochs=C.PPO_N_EPOCHS)
display(pd.Series(asdict(full_settings), name="full_run_settings"))
display(pd.Series(phase_budgets(full_settings), name="actual_budget"))
display(pd.Series({name: getattr(C, name) for name in ["PPO_LEARNING_RATE", "PPO_GAMMA",
    "PPO_GAE_LAMBDA", "PPO_ENT_COEF", "PPO_N_EPOCHS", "PPO_BATCH_SIZE"]}, name="learning_parameters"))
'''),
("md", """## 7. Recovery is part of the training loop

After each completed rollout update, the trainer saves model and Adam optimiser, worker games,
RNG states, last observations, phase and committed history. Each generation has checksums.
The latest pointer is replaced only after the complete generation is written.

The exported model is the fixed-budget endpoint, not a checkpoint selected using final scores.
Development checks run at the first completed update at or after each 25,000-transition interval.
They use up to four spread layouts from each of three development structures across four views:
48 episodes when the panel is full. An initial check runs before learning. These diagnostics
are not the complete final evaluation. Pool data is referenced from verified run inputs rather
than copied into each worker checkpoint.
"""),
("md", """## 8. Optional short software run

This switch runs tiny CPU learners only. It cannot launch full training. New output folders
protect existing evidence. Smoke policies are not expected to play well.
"""),
("code", '''
RUN_SOFTWARE_SMOKE = False
if RUN_SOFTWARE_SMOKE:
    for condition in ("P", "PA"):
        report = train(TrainingSettings(condition=condition), OUTPUT / f"smoke-{condition}")
        print(condition, report["status"], report["actual_steps"])
else:
    print("Smoke training not requested. Notebook 03b contains the recovery exercise.")
'''),
("md", """## 9. Prepare the study first

Build the full filtered pools. Run the separate 57,344-step pilot pair at learner seed 100, evaluate
it and record the team's evidence-based review. Registration verifies that review. The default
main study uses ten learner seeds per condition: 20 learners and 5,242,880 transitions.

See [Team master](../docs/TEAM_MASTER.md) and [Runbook](../docs/RUNBOOK.md). The next cell constructs
the study-controller command but does not execute it. Source changes after registration require
a new registration and matching fresh runs; the controller does not access final layouts.
"""),
("code", '''
import shlex
full_command = [sys.executable, "-m", "experiments.study", "run", "--confirm-long-run",
    "--device", "cuda", "--registration", str(ROOT / "experiments/registration-v2.json"),
    "--out", str(RUNS / "study-v2")]
print(shlex.join(full_command))
print("Command only. Full training has NOT started.")
'''),
("md", """### Pilot budget and runtime review

The pilot uses the full network, optimiser and four-worker rollout configuration, with a shorter
joint phase. It is not included among the main learners. The command-line pilot review estimates
main-training hours from measured wall time and checks that the development evaluation belongs
to those pilots. The team still needs to assess learning curves and failures.

No runtime below is an observed SageMaker measurement. The table shows only the planned work.
"""),
("code", '''
from train.train_ppo import study_settings
display(pd.DataFrame([
    {"stage": "pilot pair", "learners": 2, "steps_each": phase_budgets(study_settings(mode="pilot"))["total_actual"]},
    {"stage": "main study", "learners": 2 * len(C.STUDY_LEARNER_SEEDS),
     "steps_each": phase_budgets(study_settings())["total_actual"]},
]).assign(total_steps=lambda frame: frame.learners * frame.steps_each))
'''),
("md", """## 10. Inspect a real history

Select an existing run explicitly. After recovery, canonical episode history matches the
committed model state. Previous uncommitted history is archived separately. Compare returns
only when their reward definitions match.
"""),
("code", '''
RUN_TO_INSPECT = None
if RUN_TO_INSPECT is not None:
    selected = Path(RUN_TO_INSPECT)
    manifest = json.loads((selected / "manifest.json").read_text())
    display(pd.Series({k: manifest.get(k) for k in ["status", "actual_steps", "current_phase", "evidence_scope"]}))
    history_file = selected / "episodes.jsonl"
    if history_file.exists() and history_file.stat().st_size:
        history = pd.read_json(history_file, lines=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
        for column, ax in zip(["return", "destruction"], axes):
            ax.scatter(history.timesteps, history[column], s=8, alpha=.35, label="Episode")
            ax.plot(history.timesteps, history[column].rolling(20, min_periods=1).mean(), label="20-episode mean")
            ax.set(xlabel="Environment transitions", ylabel=column.title())
            ax.legend()
        plt.tight_layout()
        save_figure("training-history")
else:
    print("No run selected. No learning curve has been invented.")
'''),
("md", """## 11. Questions to explain

Why mask actions rather than penalising invalid moves? Why keep P and PA otherwise identical?
Why does recovery need more than weights? Why use independent learner seeds for uncertainty?
Why select the final model by budget rather than its final-test score?

Primary references: [Maskable PPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)
and [SB3 save format](https://stable-baselines3.readthedocs.io/en/master/guide/save_format.html).
"""),
],
"03b_checkpoint_recovery.ipynb": [
"""# 03b. Checkpoint and recovery lab

UTS 43008 · Assignment 3 · Project 12 · Dihedrals

Before the long study, train once without a break, train again with a pause, resume it and compare
the resulting state. This is a software test, not evidence of a useful strategy.

This follows the shared-helper structure in our Semester 3 Deep Learning Assignment 3. RL adds
game-state and random-state recovery because training experience is generated while the agent plays.

The default exercise performs only two 256-transition CPU learners. It uses no AWS services,
S3, final layouts or full training budgets.
""",
("md", """## 1. Recovery point versus final model

| Artifact | Purpose |
|---|---|
| contract.json | Immutable settings, source, versions, input hashes and budgets |
| Checkpoint model.zip | Policy, value network and optimiser from a safe boundary |
| state.pt | Worker games, RNGs, last observations and development state |
| Checkpoint episodes.jsonl | History prefix consistent with those weights |
| checkpoint.json | Phase, completed steps and checksums |
| LATEST.json | Pointer to a published generation |
| Run-root model.zip | Completed fixed-budget model for inference |

These are Python serialised objects. Load only artifacts from trusted code and people.
Checksums detect damage; they do not make an unknown model safe.
"""),
("code", '''
from tools.check_recovery import recovery_lab, compare_runs
from train.checkpoints import latest_checkpoint, verify_checkpoint
from train.train_ppo import resume

lab = OUTPUT / "recovery-lab"
report = recovery_lab(lab)
display(pd.DataFrame(report["checks"]).T)
assert report["passed"]
'''),
("md", """## 2. Inspect the evidence

The second learner pauses at 96 transitions during joint training, then resumes to 256.
The comparison checks every model and optimiser tensor, counters and canonical episode history.
Matching only the number of steps would not establish correct continuation.
"""),
("code", '''
records = []
for label in ["continuous", "resumed"]:
    manifest = json.loads((lab / label / "manifest.json").read_text())
    records.append({"run": label, "status": manifest["status"], "actual_steps": manifest["actual_steps"],
        "attempts": len(manifest["attempts"]), "save_reload_verified": manifest["save_reload_verified"]})
display(pd.DataFrame(records))
display(pd.DataFrame(json.loads((lab / "resumed/manifest.json").read_text())["attempts"]))
'''),
("md", """## 3. Inspect a complete generation

The trainer writes a staging directory, flushes its files, renames it to a committed generation
and atomically updates the pointer. Incomplete staging directories are ignored.

If the newest generation is damaged, recovery can use an earlier intact one. That fallback may
repeat more work but preserves a consistent training state.
"""),
("code", '''
checkpoint, metadata, rejected = latest_checkpoint(lab / "resumed")
assert verify_checkpoint(checkpoint) == metadata
display(pd.Series({k: metadata[k] for k in ["phase", "actual_steps", "optimizer_updates", "boundary"]}))
display(pd.DataFrame([{"file": name, "size_MiB": (checkpoint / name).stat().st_size / 1024**2,
    "sha256": digest} for name, digest in metadata["files"].items()]))
print("Rejected generations:", rejected)
'''),
("md", """## 4. Completed runs are not repeated

Resume again. It should return the completion record without adding another training attempt.
This matters when reopening and rerunning notebook cells.
"""),
("code", '''
before = json.loads((lab / "resumed/manifest.json").read_text())
after = resume(lab / "resumed")
assert len(before["attempts"]) == len(after["attempts"])
print("No training repeated.")
'''),
("md", """## 5. Abrupt shutdown test

The exercise above pauses cleanly. This terminal command also kills its own test process after
a checkpoint is published, then verifies recovery. It never kills another run or the notebook kernel.

~~~bash
uv run python -m tools.check_recovery --out artifacts/studio-kill-check --kill-process
~~~

The test suite also covers a corrupt latest checkpoint, partial writes, two-worker PA, changed
source, moved run folders, and recovery of episode and development history.
"""),
("code", '''
rows = []
for path in sorted((lab / "resumed/checkpoints").glob("checkpoint-*")):
    item = json.loads((path / "checkpoint.json").read_text())
    rows.append({"steps": item["actual_steps"], "phase": item["phase"],
        "MiB": sum(p.stat().st_size for p in path.iterdir() if p.is_file()) / 1024**2})
generations = pd.DataFrame(rows)
display(generations)
print(f"Checkpoint storage in this run: {generations.MiB.sum():.1f} MiB")
'''),
("md", """## 6. Storage and limits

Older generations are retained; nothing is deleted automatically. The default full budget has
32 completed rollout updates plus phase-start checkpoints. Sizes depend on workers, state and
history. Use measurements and leave room for packages, multiple learners and report artifacts.

A shutdown can lose unsaved work. A deleted volume can lose every local checkpoint.
No local-file implementation prevents that, and we are not adding S3.

CPU equality here is evidence for this environment. Run the lab on Studio too. Hardware,
GPU kernels and package changes can affect numerical reproducibility.
"""),
("md", """## 7. Recovery checklist

1. Reopen the same persistent project and Python environment.
2. Confirm the complete run folder and contract are present.
3. Resume the existing folder without changing its learning settings.
4. Read the recovered phase and transition count.
5. If checks fail, preserve the artifacts and investigate instead of silently starting over.
6. Treat the study run as finished only when the manifest says completed.

Read train/checkpoints.py and train/train_ppo.py to follow the implementation.
"""),
],
})


NOTEBOOKS.update({
"04_evaluation_and_statistics.ipynb": [
"""# 04. Evaluation, uncertainty and report evidence

UTS 43008 · Assignment 3 · Project 12 · Dihedrals

The question is whether PA improves over P, not whether one replay looks convincing.
Evaluate matched learners on identical layouts and views. Keep training, development and final
evidence separate.

This notebook neither trains nor runs final evaluation. It inspects an explicitly selected CSV.
Its small numerical example tests the statistics only; it is not a project result.
""",
("md", """## 1. The evaluation unit

One row is one policy on one layout and view. A layout is identified by seed and structure.
Each independent learner seed needs both P and PA, with matched settings, source, pools and
actual phase budgets.

Games from the same learner are correlated. Thousands of games are not thousands of independent
learners. Average matched differences within each learner, then estimate uncertainty across
independent learner seeds.
"""),
("code", '''
from train.evaluate import paired_summary, study_summary, validate_matched_runs, load_run

display(pd.DataFrame([
    ["condition", "P, PA, heuristic or random"], ["learner_seed", "Independent training run"],
    ["base_seed + motif + view", "Matched evaluation situation"],
    ["destruction", "Fraction of non-wall value destroyed"],
    ["town_hall / full_clear", "Binary outcomes"], ["return", "Reward including milestones and cost"],
    ["evidence_scope", "Development, software check or authorised final"]
], columns=["field", "meaning"]))
'''),
("md", """## 2. Select actual evidence

Set the CSV path after development evaluation. Leave it unset until results exist.
Do not automatically pick the newest file; that can mix a smoke check with study evidence.
"""),
("code", '''
EVALUATION_CSV = None
frame = None
if EVALUATION_CSV is not None:
    frame = pd.read_csv(Path(EVALUATION_CSV))
    required = {"condition", "learner_seed", "base_seed", "motif", "view",
                "destruction", "town_hall", "full_clear", "return", "evidence_scope"}
    assert required.issubset(frame.columns), f"Missing: {required - set(frame.columns)}"
    display(frame.groupby(["evidence_scope", "condition"]).size().rename("episodes"))
    display(frame.head())
else:
    print("No results selected. No policy-performance claim is available.")
'''),
("md", """## 3. Check provenance before a headline

Validate the corresponding run folders, not just the CSV. A CSV alone cannot prove equal training
budgets, source or checkpoint provenance. Retain every raw episode row and report accepted versus
candidate counts and coverage by structure.
"""),
("code", '''
RUN_FOLDERS = []
if RUN_FOLDERS:
    manifests = [load_run(Path(path))[1] for path in RUN_FOLDERS]
    print("Matched independent learner seeds:", validate_matched_runs(manifests))
else:
    print("No run folders selected for provenance validation.")
'''),
("md", """## 4. Understand the paired interval

For each learner, average PA destruction minus P destruction over matched evaluation situations.
The headline is the mean of those learner differences. The Student-t interval uses their sample
standard deviation and the number of independent learners.

The five differences below are invented values for testing the estimator. They are not saved
as project results. With only one learner pair, the implementation deliberately returns no interval.
"""),
("code", '''
toy_differences = [.01, .04, -.02, .03, .02]
toy_rows = []
for learner, difference in enumerate(toy_differences):
    for base in range(3):
        for condition in ["P", "PA"]:
            toy_rows.append({"condition": condition, "learner_seed": learner, "base_seed": base,
                "motif": "numerical_example", "view": 0,
                "destruction": .4 + (difference if condition == "PA" else 0)})
toy = pd.DataFrame(toy_rows)
toy_summary = paired_summary(toy)
display(pd.Series(toy_summary, name="Numerical example only"))
assert toy_summary["n_independent_learners"] == 5
assert np.isclose(toy_summary["mean_paired_difference"], np.mean(toy_differences))
assert paired_summary(toy.query("learner_seed == 0"))["ci95_low"] is None
'''),
("md", """## 5. Analyse the real paired results

A positive difference favours PA. If the interval includes zero, the evidence does not establish
a consistent direction at that confidence level. Even an interval above zero does not prove the
method works in all games or beyond the filtered study population.
"""),
("code", '''
if frame is not None and {"P", "PA"}.issubset(set(frame.condition)):
    summary = paired_summary(frame)
    display(pd.Series(summary))
    differences = pd.Series(summary["learner_differences"], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.axhline(0, color="grey", linewidth=1)
    ax.scatter(differences.index, differences.values)
    ax.set(xlabel="Independent learner seed", ylabel="Mean destruction difference: PA − P",
           title=", ".join(sorted(frame.evidence_scope.unique())))
    save_figure("paired-learner-differences")
else:
    print("Real paired analysis awaits a P/PA evaluation CSV.")
'''),
("md", """## 6. Transfer and orientation

Report training structures and layered-core transfer separately. Transfer is absent from training
and current development evaluation. Earlier software checks included this structure, so a claim
that it was never inspected during development would be inaccurate.

For each learner and structure, orientation gap is maximum minus minimum mean destruction across
eight views. A smaller gap suggests less sensitivity to presentation. It is meaningful only
when all eight views were evaluated.
"""),
("code", '''
if frame is not None:
    means = frame.groupby(["condition", "motif", "view"]).destruction.mean().reset_index()
    display(means)
    view_counts = frame.groupby(["condition", "learner_seed", "base_seed", "motif"]).view.nunique()
    print("Every evaluation unit has eight views:", bool((view_counts == 8).all()))
    if {"P", "PA"}.issubset(set(frame.condition)):
        report = study_summary(frame)
        display(pd.DataFrame(report["orientation_gap"]))
else:
    print("Structure and orientation tables will use your selected CSV.")
'''),
("md", """## 7. Report-ready evidence

The experiment tools save raw episodes, paired summaries, structure summaries and provenance.
Notebook plots go into the printed output folder. Record that path when moving figures into
the report. Do not overwrite inconvenient results.

Include:

- Source/configuration and actual training budgets.
- Independent learner count, accepted layouts, views and random baseline action seeds.
- Main metric with learner-level uncertainty, plus Town Hall and full-clear outcomes.
- Structure results, orientation sensitivity and representative failures.
- Runtime details, interruptions and the filtered-population limitation.
"""),
("md", """## 8. Final evaluation is a separate gate

Finish development and every registered learner pair, then freeze verified artifacts.
The final command requires confirmation and binds one freeze to one output. This notebook never
invokes it. An interrupted final job can resume the same committed pool and episode journal;
different code, weights or output paths are rejected. Do not tune from final results, delete
the binding to obtain another attempt, or consume reserve seeds.

See [Runbook](../docs/RUNBOOK.md) for the registration, freeze and final workflow.
"""),
("md", """### Resuming development evaluation

The evaluator commits one complete episode at a time. If the process stops, add --resume to
the same command. Source, package versions, pool, model identities and views must still match.
A changed experiment needs a new output and must not reuse an old journal.

The following cell prints commands only. It does not load models, evaluate games or access
the final partition.
"""),
("code", '''
import shlex
run_paths = [str(RUNS / "study-v2" / f"{condition}-{seed}")
             for seed in C.STUDY_LEARNER_SEEDS for condition in ("P", "PA")]
command = [sys.executable, "-m", "train.evaluate", "--pool", str(ROOT / "bases/dev-v2.json"),
           "--runs", *run_paths, "--all-views", "--out", str(ROOT / "artifacts/study-dev-v2")]
print("New evaluation:", shlex.join(command))
print("Compatible resume:", shlex.join(command + ["--resume"]))
'''),
],
"05_replay_and_viva.ipynb": [
"""# 05. Replay analysis and explaining the project

UTS 43008 · Assignment 3 · Project 12 · Dihedrals

Connect the game on screen with the code using one recorded development episode. The web app,
baselines and PPO step the same Python environment. The canvas displays returned state;
it is not a second learning environment.

This notebook runs a heuristic episode, not a trained PPO strategy.
""",
("md", """## 1. Record an episode

Replays store actions, snapshots, reward components and combat events. They help debug and
explain failures, but one episode is not a statistical comparison.
"""),
("code", '''
from agents.heuristic import HeuristicCommander
from experiments.rollout import play_episode, save_replay, load_replay

board = SiegeEnv(base_seed=10200, motifs=("closed_ring",), record=True)
replay = play_episode(board, HeuristicCommander())
board.close()
save_replay(replay, OUTPUT / "heuristic-replay.json")
assert load_replay(OUTPUT / "heuristic-replay.json") == json.loads(json.dumps(replay))
display(pd.Series(replay["summary"]))
'''),
("md", """## 2. Beginning, middle and end

Look for deployment direction, damage taken before reaching defences, wall breaches and whether
the Town Hall survives. These observations suggest development questions, not a reason to inspect
protected final layouts.
"""),
("code", '''
snapshots = replay["snapshots"]
indices = [0, len(snapshots)//2, len(snapshots)-1]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for index, ax in zip(indices, axes):
    draw_snapshot(snapshots[index], ax, style="presentation", show_hud=False)
    ax.set_title(f"Frame {index} · tick {snapshots[index]['tick']}")
plt.tight_layout()
save_figure("replay-story")
'''),
("md", """## 3. Follow actions and rewards

Buying changes reserves and spend. Finishing shopping enters battle and charges army cost.
Deployment adds a troop and advances combat. Waiting advances combat without deployment.
Each milestone pays only on the transition when it is first achieved.
"""),
("code", '''
trace = pd.DataFrame([{"decision": i, "action": describe_action(frame["last_action"]),
    "tick": frame["tick"], "destruction": frame["destruction"], "return": frame["episode_return"],
    **frame["reward_components"]} for i, frame in enumerate(snapshots) if frame["last_action"] is not None])
display(trace.head(12))
display(trace.tail(5))
components = trace[list(C.REWARD_COMPONENT_KEYS)].sum()
display(components.rename("episode reward total"))
assert np.isclose(components.sum(), replay["summary"]["episode_return"])
'''),
("code", '''
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].step(trace.decision, trace.destruction, where="post")
axes[0].set(xlabel="Agent decisions", ylabel="Destroyed value fraction", ylim=(0, 1))
axes[1].step(trace.decision, trace["return"], where="post")
axes[1].set(xlabel="Agent decisions", ylabel="Cumulative return")
plt.tight_layout()
save_figure("replay-progress")
'''),
("md", """## 4. Trace one click through the architecture

| Layer | Responsibility |
|---|---|
| ui/app.py and canvas | Read actions and draw state |
| env/siege_env.py | Check legality, advance the episode and calculate reward |
| env/combat.py | Move, choose attacks, apply simultaneous damage and check endings |
| env/pathfinding.py | Weighted four-neighbour routes |
| train/features.py and PPO | Convert observation into a learned action |
| experiments/rollout.py | Consistent episode loop for policy evaluation |
| train/checkpoints.py | Save enough state for continuation |

Python is authoritative for the current app. Vaibhav's visual theme is retained. His preserved
prototype is reference material, not another source of assignment results.
"""),
("md", """## 5. A short, honest pitch

We are studying whether rotation and reflection augmentation helps an RL commander generalise.
It buys troops and chooses deployment positions in a small base-attack game. Movement and combat
follow fixed rules. We compare the same Maskable PPO learner with and without augmentation,
alongside random and scripted baselines.

The simulator, web demonstration, training and evaluation code are implemented. Short software
checks do not establish a strong strategy. Long matched-seed experiments and final findings
still need to be completed.
"""),
("md", """## 6. Questions the professor may ask

| Question | Plain-English answer |
|---|---|
| What is learned? | Army purchases, deployment positions and timing. Troop movement is scripted. |
| Why RL? | Early purchases and deployment choices affect later combat rewards. |
| Why PPO? | A practical policy-gradient baseline; masking handles phase-dependent legal actions. |
| What is custom? | The game/interface, feature fusion, D4 mapping and experiment/recovery workflow. |
| Why two phases? | Learn deployment with a fixed army, then joint purchasing and deployment. |
| Is the observation Markov? | Not fully. Some per-troop internal state is hidden. |
| Is the comparison fair? | Same layouts, seeds, network, settings and actual budgets; augmentation differs. |
| What if Studio stops? | Resume the last intact update with optimiser, games and RNGs. Unsaved work can repeat. |
| Does it need S3? | No. It uses the persistent Studio project folder. |
| Is the research finished? | The software exists; convergence and final multi-seed findings remain unverified. |

Practise these answers in your own words, and follow the relevant source functions for detail.
"""),
("md", """## 7. What to demonstrate

Buy a small army in the app and explain the legal perimeter choices. Show a heuristic replay,
observation layers and D4 views. Use notebook 03b to explain recovery. Label any short-run PPO
replay as a software check.

Do not promise augmentation improves performance before the matched study provides evidence.
"""),
],
})

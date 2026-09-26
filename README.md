# Smooth playback update

Start with [DEMO_START_HERE.md](DEMO_START_HERE.md), or open `demo_preview.html` for the included recorded heuristic attack.

# Base Siege · Assignment 3

UTS 43008 Reinforcement Learning · Project 12 · Dihedrals

A custom strategy-game RL project. The agent buys troops with 200 resources, then chooses where
and when to deploy them against a defended 16 × 16 base. Movement and combat follow fixed rules.

Start with the [team master document](docs/TEAM_MASTER.md). It explains the project, Vaibhav's
contribution, the study, remaining work and questions for the oral defence.

## Source and scope

This project starts from Vaibhav's Stage 5 ZIP and retains his visual theme. The original archive
is preserved unchanged in reference/. The current design improves on the initial A/B drafts;
older demos, submissions and recorded software artifacts are not rewritten.

P means standard Maskable PPO. PA means the same learner trained with rotated/reflected coordinate
views. PPO comes from SB3 Contrib. Our custom work is the task, environment integration,
representations, augmentation, features, recovery and controlled experiment.

## Current study

| Item | Working design |
|---|---|
| Main comparison | P versus PA, plus random and heuristic baselines |
| Main learners | Ten seeds per method: 20 full runs |
| Budget per learner | 32,768 placement + 229,376 joint = 262,144 transitions |
| Main total | 5,242,880 transitions |
| Development gate | Separate 57,344-step P/PA pilot pair at learner seed 100, followed by a recorded team review |
| Training/development | Three structures, with separate layout seeds |
| Transfer | Layered core excluded from training and current development; historical development exposure disclosed |
| Evaluation | Eight views per accepted layout; uncertainty across independent learner seeds |

Read [EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) before registering or comparing results.
The compact observation remains partial. There is no claim of a novel PPO algorithm, an optimal
policy or a measured augmentation improvement.

## Start locally

Python 3.12 is required. From this folder:

~~~bash
uv sync --frozen
uv run streamlit run ui/app.py --server.address 127.0.0.1 --server.port 8512
~~~

Open http://127.0.0.1:8512. On this Mac, Start Base Siege.command is the existing double-click
launcher. The current app uses Python as its only battle engine and preserves the supplied
board artwork, glyphs and colour theme.

The app supports manual play, baseline/PPO replays, saved replay inspection and result display.
Baselines are not learned policies; software-check weights are not research results.

## Read the notebooks in order

1. [00: SageMaker setup](notebooks/00_sagemaker_setup.ipynb): kernel, GPU, storage and shutdown checks.
2. [01: Environment and layouts](notebooks/01_environment_and_layouts.ipynb): game rules and the RL contract.
3. [02: Baselines and validation](notebooks/02_baselines_and_validation.ipynb): diagnostic games and filtering.
4. [03: PPO training](notebooks/03_maskable_ppo_training.ipynb): network, curriculum, pilot and study commands.
5. [03b: Checkpoint recovery](notebooks/03b_checkpoint_recovery.ipynb): short CPU recovery exercise.
6. [04: Evaluation and statistics](notebooks/04_evaluation_and_statistics.ipynb): paired results and reporting.
7. [05: Replay and viva](notebooks/05_replay_and_viva.ipynb): connect visible decisions with code.

Run uv run jupyter lab notebooks. Long training is disabled by default. Shared Python modules
are authoritative; notebooks do not contain a separate copy of the simulator or PPO implementation.

## Recovery across the workflow

Training saves completed updates, optimiser state, worker games, random states and history.
Schema-3 snapshots refer to verified immutable pools instead of embedding a copy in every worker.

Pool construction journals one candidate at a time. Evaluation journals one episode at a time.
The study controller plans, runs, pauses and resumes the registered matrix, verifies completed
learners and checks free disk space. Final evaluation resumes only the same freeze/output.

All persistence is local to the project storage. No S3 is used. Keep a separate backup against
volume loss. See the [runbook](docs/RUNBOOK.md) and [Studio guide](docs/SAGEMAKER_STUDIO.md).

## Bounded verification

~~~bash
uv run pytest -q
uv run python tools/execute_notebooks.py --out artifacts/my-notebook-check
uv run python tools/smoke_pipeline.py --out artifacts/my-smoke-check
uv run python -m tools.check_recovery --out artifacts/my-recovery-check --kill-process
~~~

Choose new output folders. The smoke pipeline trains P and PA for only 256 steps each using two
workers, reloads them and tests resumable development evaluation. The recovery drill kills only
its own short-lived test process. These checks are not learning-quality experiments.

See [VERIFICATION.md](docs/VERIFICATION.md) for observed results and artifact paths.
SageMaker/CUDA execution, long pilots, the full study and protected final evaluation have not run.
Research findings, final report, presentation and individual defence preparation remain.

## Public Streamlit demo

The public entrypoint is ui/streamlit_app.py. Its CPU-only dependency profile and selected-asset
bundle exclude notebooks, training state, local research outputs and private configuration.

~~~bash
.venv/bin/python tools/build_deploy_bundle.py --out artifacts/my-streamlit-deploy
~~~

Use the generated folder for a dedicated deployment repository, not for SageMaker training.
The public app keeps the same simulator and theme. See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for
asset export and later hosting. No Docker, S3, paid API or public deployment is performed here.

## Project map

| Folder | Purpose |
|---|---|
| env/ | Configuration, procedural layouts, simulation, Gymnasium and D4 |
| agents/ | Random and heuristic policies |
| train/ | Features, Maskable PPO, checkpoints and evaluation |
| experiments/ | Pools, journals, pilot review, study controller and final protocol |
| ui/ | Streamlit app and canvas component |
| notebooks/ | Seven teaching and experiment notebooks |
| tests/ | Mechanics, recovery, provenance, notebook and application regressions |
| tools/ | Bounded checks, notebook generation/execution and deployment export |
| docs/ | Team master, experiment plan, runbook, architecture, viva and verification |
| reference/ | Unchanged source archive |
| artifacts/ | Local outputs, excluded from Git |

The inherited BUILD_ORDER_v2.md, prototype/ and original figures are reference material.
The current workflow is defined by the team master and experiment plan, not the archived HTML
engine. The separately referenced v3 project-plan file was not present in the ZIP.

API references: [Gymnasium](https://gymnasium.farama.org/introduction/create_custom_env/),
[Maskable PPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html),
[Streamlit components](https://docs.streamlit.io/develop/concepts/custom-components/components-v1/intro).

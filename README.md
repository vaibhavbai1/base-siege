# Base Siege

**A reinforcement learning commander for army purchasing and troop deployment in a custom base-attack simulator.**

University of Technology Sydney · 43008 Reinforcement Learning · Assignment 3 · Project 12 · Team Dihedrals

Base Siege asks a learning agent to assemble an army within a fixed budget and deploy it against a defended base. Troop movement, target selection and defensive combat follow programmed rules. The agent learns the purchasing and deployment strategy through interaction with the simulator.

The completed study compares **MaskablePPO without symmetry augmentation (P)** with **the same learner trained using rotated and reflected coordinate views (PA)**. Twenty main models were trained independently. With sampled action selection, the saved policies achieved approximately **86% mean destruction** on the final evaluation layouts. Symmetry augmentation did not demonstrate a clear additional mean benefit under the tested settings.

This is a custom educational simulator inspired by base-attack strategy games. It is not connected to the commercial Clash of Clans game.

## Application preview

![Base Siege recorded battle interface](docs/verification/smooth-playback-desktop.png)

*Recorded interface screenshot showing the heuristic baseline. This image illustrates the replay controls, not a trained PPO result.*

The application supports:

- **Manual play:** purchase an army and deploy troops using browser combat controls.
- **Policy replay:** watch a heuristic, random policy or saved MaskablePPO model.
- **Base editor:** create compatible custom layouts for manual play or agent demonstrations.
- **Saved replay:** inspect a previously recorded attack.
- **Experiment results:** browse saved training manifests and evaluation tables.

Manual combat runs in the browser for smoother interaction. Trained-agent replays are generated from Python simulator trajectories. Playback speed changes the animation, not the recorded battle outcome.

## Quick start

### Requirements

- **Python 3.12**; the recorded training environment used **3.12.14**.
- **uv** for installing the dependencies in `uv.lock`.
- **Git and Git LFS** for downloading code, models and recovery data.
- Linux or macOS. On Windows, use **WSL2 with Ubuntu** for the complete research workflow because recovery modules use Unix file locking.
- Internet access for the initial dependency installation and model download.

The local application loads saved policies on the **CPU**. A training GPU is not required for replay or manual play. The complete experiment folders occupy several gigabytes, and dependency installation requires additional space.

### 1. Clone the repository and retrieve large files

Open a terminal on your computer, or the Ubuntu terminal on Windows. Install Git LFS first if `git lfs version` is unavailable. On Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y git-lfs
```

Then run:

```bash
git lfs install
git clone https://github.com/vaibhavbai1/base-siege.git
cd base-siege
git lfs pull
```

For a private repository, your GitHub account must have access and Git authentication must be configured. If you already cloned the repository, enter its folder and run `git lfs pull` instead of cloning again.

**Use Git LFS when retrieving saved models.** An unresolved LFS pointer is a small text file, not a usable model or database. The repository's `.gitattributes` defines which files use LFS. Model and results availability depends on the experiment-data upload having completed; if a required path is absent after pulling, obtain the corresponding experiment bundle from the project owner. Retraining is not required to recover an existing model file.

### 2. Install dependencies

Run these commands from the repository root, where `pyproject.toml` and `uv.lock` are located:

```bash
uv python install 3.12.14
uv sync --frozen --python 3.12.14
```

If uv is not installed, one option is to install it in a separate tools environment:

```bash
python3 -m venv "$HOME/.venvs/base-siege-tools"
"$HOME/.venvs/base-siege-tools/bin/python" -m pip install uv
export PATH="$HOME/.venvs/base-siege-tools/bin:$PATH"
```

Then repeat the installation commands above. On Ubuntu, a missing `venv` module can be installed with `sudo apt-get install python3-venv`. Repeat the PATH export in a new terminal if required.

`--frozen` uses the existing lock file without changing dependency choices. Keep the recorded versions when reproducing the study. A package-installation failure should be diagnosed before changing version pins.

### 3. Launch the local application

```bash
uv run --frozen streamlit run ui/app.py --server.address 127.0.0.1 --server.port 8512
```

Open **http://127.0.0.1:8512** in your browser. Keep the terminal running; press **Ctrl+C** there to stop the server.

Use **`ui/app.py`** for the complete local project. `ui/streamlit_app.py` enables the separate public-demo mode, which looks for selected exports in `demo_assets/` rather than the full research run folders.

These localhost instructions are for a local computer or WSL. If you launch the app on SageMaker, use the application/port access mechanism provided by your SageMaker workspace; your computer's localhost URL does not refer to the remote SageMaker server.

## Watch a trained agent

In the application:

1. Select **Mode → Policy replay**.
2. Select **Generated base**, structure **closed_ring**, and **Development pool**.
3. Choose development base seed **10200**.
4. Set **Policy → Saved Maskable PPO**.
5. Choose condition **P** and learner seed **0**.
6. Select **`runs/study-v2/P-0/model.zip`** as the local model.
7. Choose **Sampled**, action seed **0**, and agent view **0**.
8. Click **Generate replay**, then **Play**.

These are fixed demonstration choices, not a selection of the best-performing learner. Other smoke-test or recovery-test models may also appear in the selector; use the `study-v2` path for a main research model.

Generating a replay uses saved weights and does not train the agent. Sampling chooses actions according to learned probabilities. The action seed controls that sampling; matching outcomes also depends on a compatible numerical runtime.

### Watch without installing Python dependencies

When the experiment artifacts are available, open this file directly in Chrome, Edge or Firefox:

```text
artifacts/demo/P-0-closed_ring-10200-view0-action0.html
```

It contains an already-recorded sampled P-0 attack. The accompanying replay records **87.10% destruction** on that development layout. This is one illustrative attack, not the final study average.

The top-level `demo_preview.html` is a separate **heuristic baseline** demonstration.

## Final results

The final evaluation used **213 accepted layouts** from 800 candidates, with eight coordinate views per layout. The sampled extension used three action seeds per model/layout/view combination.

| Policy | Action selection | Mean destruction | Full-clear rate | Town Hall destruction rate |
| --- | --- | ---: | ---: | ---: |
| P | Sampled | **85.95%** | 54.64% | 85.12% |
| PA | Sampled | **85.45%** | 53.24% | 85.14% |
| Heuristic | Programmed baseline | 55.10% | 2.58% | 59.39% |
| P | Deterministic / greedy | 43.53% | 6.78% | 43.72% |
| PA | Deterministic / greedy | 44.59% | 6.68% | 45.26% |
| Random | Uniform legal-action baseline | 6.39% | 0.00% | 6.34% |

**Destruction measures the fraction of building value destroyed.** It is not win rate, damage to building health, or simply the percentage of buildings removed. Full-clear rate and Town Hall destruction rate are separate metrics.

The strongest observed difference came from **how the existing policy selected actions**. Sampling produced much higher mean destruction than always choosing the highest-probability legal action. This improvement used the existing checkpoints; no additional training was performed for the sampled extension.

For sampled inference, **PA minus P was −0.50 percentage points**, with a **95% paired learner-level interval of −2.06 to +1.06 points**. This does not establish an additional mean benefit from symmetry augmentation, and it does not prove that the methods are equivalent.

The sampled extension was motivated by development results and documented **before final evaluation**. It was not part of the original pre-training registration. The original deterministic evaluation remains part of the study.

Sources: [final summary CSV](artifacts/sampled-final-v1/sampled_summary.csv), [detailed final analysis](artifacts/sampled-final-v1/sampled_summary.json), and [sampling amendment](experiments/final-sampling-amendment-v1.json).

## Environment and learning task

| Component | Configuration |
| --- | --- |
| Battlefield | 16 × 16 grid |
| Starting resources | 200 |
| Troop types | Tank, melee, ranged and breaker |
| Deployment positions | 60 perimeter tiles |
| Observation | Eight-channel spatial grid and nine normalised summary values |
| Action space | 246 discrete actions |
| Legal-action masks | Exclude actions such as unaffordable purchases and unavailable troop deployments |
| Battle progression | Ten simulator ticks per RL battle decision |
| Battle limit | 300 ticks |
| Layout structures | Closed ring, north–south gates, east–west gates and layered core |

Purchasing decisions do not advance combat time. The action encoding is:

| Action IDs | Meaning |
| --- | --- |
| `0–3` | Buy one troop of the corresponding type |
| `4` | Finish shopping |
| `5–244` | Deploy one of four troop types at one of 60 perimeter positions |
| `245` | Wait and advance combat |

The eight grid channels describe the Town Hall, cannons, mortars, storages, walls, building health, defensive coverage and troop density. The summary vector supplies resource, reserve and phase/time information. This observation is **partial**: it does not expose all internal troop state.

The reward combines newly destroyed building value, one-time bonuses for reaching half destruction, destroying the Town Hall and clearing the base, and a resource-spending penalty. Each milestone bonus is **0.25**, and the spending penalty is **0.30 × resources spent / 200**. Wall shaping is disabled. Reward and destruction are therefore related but different quantities.

See [environment configuration](env/config.py), [Gymnasium interface](env/siege_env.py) and [combat implementation](env/combat.py).

## Algorithm and experiment design

The learner is **SB3 Contrib MaskablePPO**, with a custom convolutional feature extractor for the grid and a separate branch for the summary vector. Legal-action masking is used during learning and inference.

| Condition | Meaning |
| --- | --- |
| **P** | MaskablePPO without symmetry augmentation |
| **PA** | The same MaskablePPO setup with randomly rotated/reflected coordinate views during training |

PA uses the eight symmetries of a square, with observations, masks and deployment coordinates mapped consistently. It changes the coordinate view rather than the underlying combat rules.

There are **ten independently trained models per condition**: `P-0` through `P-9` and `PA-0` through `PA-9`. Learner seeds `0–9` control training randomness. These are repeated runs of the same two conditions, not twenty different algorithms or an ensemble. Layout seeds generate bases; action seeds control sampled inference.

### Training settings

| Setting | Recorded value |
| --- | ---: |
| Parallel environments per learner | 4 |
| Rollout steps per environment | 2,048 |
| Total transitions per rollout | 8,192 |
| Minibatch size | 256 |
| Optimisation epochs per rollout | 10 |
| Learning rate | 0.0003 |
| Discount factor γ | 1.0 |
| GAE λ | 0.95 |
| Entropy coefficient | 0.01 |
| Main placement-only phase | 32,768 transitions |
| Main joint purchasing/deployment phase | 229,376 transitions |
| Total per main model | **262,144 transitions** |

The twenty main runs produced **5,242,880 transitions** and **143,887 completed training episodes**. Summed measured training-process time was **9,772.874 seconds**, approximately **2 hours 42 minutes 53 seconds**. This is the sum of run timers, not the duration of the entire project or all evaluations. The exact GPU model was not established by the recovered records.

Separate pilot models `P-100` and `PA-100` each used **57,344 transitions**: 32,768 placement-only and 24,576 joint transitions. Main training runs started independently rather than resuming the pilots.

Training and development used the first three layout structures with separate seed partitions. Layered core was excluded from the main training and development pools and included in final transfer analysis; earlier historical development exposure is disclosed in the study documentation. Final evaluation used the accepted layouts from a protected seed partition. Comparisons quantify variability across independent learners, rather than treating every attack as an independent training run.

## Completed experiments and saved evidence

| Stage | Saved location | Evaluation episodes |
| --- | --- | ---: |
| Pilot training | `runs/pilot-v2/` | Training stage |
| Pilot development evaluation | `artifacts/pilot-dev-v2/` | 16,224 |
| Twenty main training runs | `runs/study-v2/` | Training stage |
| Original deterministic development evaluation | `artifacts/study-dev-v2/` | 38,688 |
| Deterministic-versus-sampled diagnostic | `artifacts/policy-sampling-v1/` | 7,680 |
| Full sampled development evaluation | `artifacts/sampled-dev-v1/` | 74,880 |
| Original final evaluation | `artifacts/study-final-v2/` | 52,824 |
| Sampled final extension | `artifacts/sampled-final-v1/` | 102,240 |

Evaluation episode counts are separate from training episodes and transitions. Some scenarios are intentionally reused across diagnostic and evaluation stages.

Important evidence files include:

- Per-run `model.zip`, `manifest.json`, `contract.json`, `episodes.jsonl` and `development.json`.
- `experiments/registration-v2.json`, `final-freeze-v2.json` and `final-sampling-amendment-v1.json`.
- `bases/train-v2.json`, `bases/dev-v2.json` and `artifacts/study-final-v2/final_pool.json`.
- Evaluation CSVs, summaries, completion records and, in the full backup, SQLite progress databases.

The detailed **Base Siege Complete Experiment Handbook** accompanies the project handover and explains the mechanics, experiments, results, timing evidence and professor questions. When included in the repository, it is stored under `docs/` as `Base_Siege_Complete_Experiment_Handbook.docx`.

## Repository map

| Path | Purpose |
| --- | --- |
| `env/` | Simulator, configuration, layout generator, observations and symmetry wrapper |
| `agents/` | Random and heuristic commanders |
| `train/` | Feature extractor, PPO training, model loading, checkpoints and evaluation |
| `experiments/` | Study design, registration, base pools, journals and final protocol |
| `ui/` | Streamlit application, browser manual engine, replay player and base editor |
| `runs/` | Saved model runs and training/recovery records |
| `artifacts/` | Evaluation outputs, diagnostics, saved replays and verification evidence |
| `bases/` | Saved training/development pools and pool-construction records |
| `notebooks/` | Setup, environment, baseline, training, recovery, analysis and replay notebooks |
| `tests/` | Simulator, interface, provenance, recovery and application checks |
| `tools/` | Replay export, diagnostics, notebook tools and deployment helpers |
| `final_evaluation_bundle/` | Final evaluation and sampled-inference extension scripts |
| `docs/` | Architecture, experiment plan, runbook, deployment and verification notes |
| `demo_assets/` | Selected exports for public-demo mode, when prepared |
| `base_siege_app_update/` | Packaged application update and its verification material |
| `prototype/`, `reference/` | Earlier prototype and historical handoff material |

## Notebooks and verification

From the repository root, open the notebooks with:

```bash
uv run --frozen jupyter lab notebooks
```

Read them in this order:

1. [SageMaker setup](notebooks/00_sagemaker_setup.ipynb)
2. [Environment and layouts](notebooks/01_environment_and_layouts.ipynb)
3. [Baselines and validation](notebooks/02_baselines_and_validation.ipynb)
4. [MaskablePPO training](notebooks/03_maskable_ppo_training.ipynb)
5. [Checkpoint recovery](notebooks/03b_checkpoint_recovery.ipynb)
6. [Evaluation and statistics](notebooks/04_evaluation_and_statistics.ipynb)
7. [Replay and viva](notebooks/05_replay_and_viva.ipynb)

Read notebook cells before running them. Some cells start training, pool generation or evaluation; none of those operations is needed to view saved results or use the application.

For focused application checks:

```bash
uv run --frozen pytest -q tests/test_manual_custom_ui.py tests/test_policy_playback.py tests/test_browser_manual_parity.py
```

Browser-engine parity checks require Node.js and are skipped when it is unavailable. The full suite is available through `uv run --frozen pytest -q`; it includes short learner and recovery exercises. Passing software checks does not establish policy quality or eliminate every simulator limitation.

## Reproducibility and recovery

Manifests and contracts record configurations, runtime versions, source hashes, pool identities and model hashes. SHA-256 hashes help detect changed files. Final freezes and completion records bind the recorded evaluation to its inputs and outputs.

**An exported model and an exact recovery checkpoint serve different purposes.** `model.zip` supports saved-policy inference. Exact training recovery also needs the committed checkpoint directory referenced by `LATEST.json`, including `checkpoint.json`, `state.pt`, the checkpoint model and episode history. SQLite journals support compatible continuation of interrupted pool generation and evaluation.

A lightweight essentials bundle may omit intermediate checkpoints and databases while retaining final models and results. Use the complete backup for recovery and confirm that the referenced files exist. Do not bypass configuration or provenance checks to force a resume.

The recorded runtime includes Python 3.12.14, Gymnasium 1.3.0, Stable-Baselines3 2.9.0, SB3 Contrib 2.9.0, PyTorch 2.13.0 and NumPy 2.5.2. Matching settings and seeds supports reproducibility, but bitwise agreement across different hardware and numerical libraries is not guaranteed.

For future experiments, preserve this completed study and assign a new version to changes in rewards, mechanics, observations or training settings. If the current final results guide those changes, use a fresh held-out evaluation for the new final claim.

## Known limitations

- **Inaccessible targets:** the frozen scientific movement code can return without a reachable-target fallback when the selected building has no approach goals. The reported results describe that simulator revision.
- **Partial observations:** the policy does not see every internal troop variable, and the network is feed-forward.
- **Selected task distribution:** evaluation uses difficulty-filtered procedural layouts. Results do not guarantee the same performance on arbitrary custom bases.
- **Manual play differs from RL:** browser players can deploy with different timing, so casual manual scores are not a controlled human-versus-agent benchmark.
- **No demonstrated augmentation advantage:** the sampled P–PA interval includes zero. This does not prove equivalence or rule out benefits at other budgets.
- **Timing evidence has limits:** training timers were measured; some evaluation durations are archive-timestamp estimates, and other durations were not preserved.

Some earlier documents, including `TEAM_START_HERE.md` and `DEMO_START_HERE.md`, describe the project before the full experiments or latest manual-play update. Their pending-training statements and older manual-timing descriptions are historical. Use this README for the current overview and the saved source, manifests, results and complete handbook for detailed evidence.

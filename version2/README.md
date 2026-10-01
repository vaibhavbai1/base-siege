# Base Siege Version 2 — longer-training control

Prepared for Vaibhav Bairathi, Master of Artificial Intelligence, UTS.

This is a **new development experiment**. It tests whether longer training improves the existing agent before changing observations, rewards or architecture. Better results are possible, not guaranteed.

Movement and targeting stay unchanged: **no reachable-target fallback** is added. The original inaccessible-target limitation remains. Original scientific source, models and results are preserved. The older folders already have names such as `study-v2`; this package uses the distinct output name `version2-longer-control`.

## Planned experiment

| Setting | Value |
|---|---|
| Conditions | P and PA, with their original meanings |
| Learner seeds | 0, 1, 2 per condition: six independent screening models |
| Initialization | Fresh models, not old checkpoints |
| Placement-only phase | 32,768 transitions |
| First review / automatic pause | 1,048,576 total transitions per model |
| Planned maximum | 4,194,304 total transitions per model |
| Environments × rollout | 4 × 2,048 = 8,192 transitions per update |
| PPO settings | Original batch size 256, 10 epochs and original learning settings |
| Recovery snapshots | Every 8 updates, plus phase endings and pause boundaries |
| Retention | Last 3 verified recovery generations per run |
| Model milestones | 32,768; 262,144; 524,288; 1,048,576; 2,097,152; 4,194,304 |
| Development diagnostic | Initially and every 262,144 transitions |
| Sampled evaluation | Fixed 12-layout × 4-view panel × 3 action-seed labels: 144 episodes |
| Deterministic evaluation | Same layouts/views: 48 episodes |
| Final evaluation | Not included; requires a new held-out protocol |

The network, observation, reward, action masks, simulator, PPO hyperparameters, original train/dev pools and placement-only duration remain unchanged. The planned change to learning is the duration of the joint purchasing/deployment phase. Evaluation and storage are improved separately.

A **transition** is an agent decision; an **episode** is a complete attack. The old twenty main runs already logged **143,887 episodes combined**, not per model. The new runner counts actual completed episodes. Four million transitions do not guarantee a particular episode count or better performance.

The action-seed labels are 31001, 31002 and 31003. Each episode's actual action seed is derived from its label, layout and view using SHA-256 and recorded in its result. The same seeds are used across models/checkpoints. Evaluation uses a separate CPU policy and restores the learner's Python, NumPy and PyTorch random-number states.

## 1. Extract on SageMaker

Download `Base_Siege_Version2_Starter.zip` and upload it into your existing project folder with SageMaker's file browser.

Open a **SageMaker terminal**, not a Python notebook cell. Run:

```bash
cd ~/Base_Siege_Smooth_Demo/base-siege-assignment3
pwd
ls pyproject.toml uv.lock bases/train-v2.json bases/dev-v2.json
```

If your project is elsewhere, change `cd` to its actual location. The correct directory contains `env/`, `train/`, `bases/`, `pyproject.toml` and `uv.lock`.

Create a separate Git branch:

```bash
git switch -c version2-longer-control
```

If it already exists, use `git switch version2-longer-control`.

Extract the package:

```bash
unzip -n Base_Siege_Version2_Starter.zip
```

The ZIP adds only `version2/`; it does not overwrite the old project. `-n` keeps existing files. If you already have an edited `version2/` directory, this is not an upgrade command; inspect that directory before replacing anything.

## 2. Check the environment

Use your existing locked project environment. If dependencies are not installed yet:

```bash
uv python install 3.12.14
uv sync --frozen --python 3.12.14
```

These commands use the project's existing `uv.lock`. Do not update dependency versions for this control. If `uv` is unavailable, install it using your existing SageMaker setup before running these commands; do not paste them into a Python cell.

Run:

```bash
uv run --frozen python -m version2 preflight
uv run --frozen python -m pytest version2/tests -q
```

Preflight checks all 28 original scientific source files, both base-pool hashes, the original configuration, Python 3.12.14 and the scientific package versions. Streamlit's version is recorded but is not required to match the old study before registration. All recorded package versions must then remain unchanged for recovery.

If preflight finds modified source, stop and inspect the differences. Do not disable the checks. The package is grounded in the uploaded archive and handbook; it has not independently verified your live GitHub or SageMaker checkout. A changed application file outside the scientific source set will not fail this check.

If a pool is a small Git LFS pointer, retrieve the actual file using your existing Git LFS setup. Do not regenerate the frozen pools.

The tests train tiny temporary CPU models solely to verify software. They are not research evidence. See `VALIDATION.md` for the local test results and limits. Run the tests on SageMaker before the long experiment as well.

## 3. Record the plan and start P-0

On a GPU-backed SageMaker instance, from the same project-root terminal:

```bash
uv run --frozen python -m version2 register --device cuda
uv run --frozen python -m version2 train --condition P --seed 0 --to 1048576
```

If intentionally training on CPU, use `--device cpu` when registering. Device and hardware are recorded; do not silently switch them during recovery. CUDA registration stops if no GPU is available.

Registration writes `version2-records/longer-control/registration.json` before training. It records the source hashes, plan, baseline pool hashes, runtime and detected GPU name when available. This is a **new development protocol**. It does not change the documented timing or status of the original sampled extension.

The model pauses automatically at **1,048,576 transitions**. `status=paused` there is expected. The four-million maximum was recorded in advance so a later continuation does not require editing the run contract.

Keep SageMaker running. If you have `tmux`, use it to keep the terminal session available:

```bash
tmux new -s base-siege-v2
```

Run training inside that session. Detach with **Ctrl+B, then D**. Reconnect with:

```bash
tmux attach -t base-siege-v2
```

tmux does not prevent SageMaker instance shutdown. Recovery snapshots protect against an interrupted process, but up to 65,536 transitions since the previous scheduled snapshot may need replaying after an abrupt failure.

## 4. Check progress, pause and resume

In another terminal, first enter the same project directory:

```bash
cd ~/Base_Siege_Smooth_Demo/base-siege-assignment3
uv run --frozen python -m version2 status
```

Status shows the latest manifest's committed transitions and completed episodes. Its wall time updates at the end of an attempt, so it is not a live stopwatch. A killed process can leave status saying `running`; the next training command checks the lock and recovers the actual checkpoint. Wall time includes evaluation, checkpoint I/O and recovery, and may omit the unfinished attempt after a hard kill. Do not report it as pure optimizer time or assume missing time is zero.

To request a safe pause:

```bash
uv run --frozen python -m version2 pause --condition P --seed 0
```

Wait for `status=paused` before stopping the SageMaker space. Partial rollouts are discarded and replayed. Resume by repeating:

```bash
uv run --frozen python -m version2 train --condition P --seed 0 --to 1048576
```

It resumes the same model; it does not create another independent model. Repeating it after the screen has been reached does not train extra updates.

Keep `contract.json`, `inputs/`, `LATEST.json` and retained `checkpoints/`. A milestone ZIP is an inference model, **not a full recovery checkpoint**. Corrupt or unrecognized checkpoint directories are retained for inspection; the runner can recover from an earlier verified generation when available.

## 5. Run the six-model screen

After the first model runs correctly:

```bash
uv run --frozen python -m version2 matrix --to 1048576
```

It runs P-0, P-1, P-2, PA-0, PA-1 and PA-2 sequentially. It resumes incomplete runs and skips already reached screens. Do not run two matrix commands concurrently.

Export the scores:

```bash
uv run --frozen python -m version2 status
uv run --frozen python -m version2 report > version2-records/longer-control/development_summary.csv
```

Compare sampled destruction at **262,144 versus 1,048,576 transitions within these new runs**, then compare paired P/PA learner seeds. Destruction means building **value** destroyed, not win rate. Full-clear rate is exported separately.

This is a small development panel, not the full development set or final test. Do not compare its scores directly to the old 85.95% / 85.45% final means as though they were the same evaluation. Three learner seeds are screening evidence, and hundreds of evaluation episodes do not become hundreds of independent trained agents.

Review these curves and failure cases before spending more compute. If continuation is justified, the following reaches the recorded maximum:

```bash
uv run --frozen python -m version2 matrix --to 4194304
```

Do not edit the plan, Python code or dependencies after registration and silently resume. Changed rewards, observations or training methods belong to another recorded experiment. A later confirmatory study needs a separately frozen design and independent learner runs. Because the old final results have already been examined, protect a new held-out evaluation set before making new final generalization claims.

## Output and team handover

New files go under:

```text
version2-records/longer-control/
runs/version2-longer-control/P-0/
runs/version2-longer-control/P-1/
runs/version2-longer-control/P-2/
runs/version2-longer-control/PA-0/
runs/version2-longer-control/PA-1/
runs/version2-longer-control/PA-2/
```

Each run records copied inputs, its contract, manifest, episode log, development episode results, TensorBoard events, retained recovery checkpoints and milestone models. At the one-million screen, the model is:

```text
milestones/step-000001048576.zip
```

At the completed four-million budget, the run also exports `model.zip`. This package does not automatically change the app's selected model or replace an old results table. Existing manual play is not a controlled human-versus-agent benchmark.

Commit the source and registration after creating the registration:

```bash
git add version2 version2-records/longer-control/registration.json
git diff --cached --stat
git commit -m "Add Version 2 longer-training control"
git push -u origin version2-longer-control
```

This pushes a review branch; it does not merge into main. Before registration exists, commit just `version2`.

Back up `runs/version2-longer-control/` separately, including recovery state. The archived `.gitignore` ignores `runs/`, and your live rules may differ. Do not assume `git add .` uploads results. Check Git LFS rules and storage before uploading large files. Share selected milestone models, contracts, manifests and development results with the team; keep full recovery state in the project backup. No automatic large-file upload is performed.

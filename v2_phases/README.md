# V2 purchasing and deployment action-selection diagnostic

This package tests two new ways of using the existing six final V2 models. It does not train, change checkpoints or simulator rules, or access the final-test pool. Keep the original `version2/`, `v2_development/` and scientific source files unchanged.

## What is being tested

The first letter identifies purchasing; the second identifies deployment. **S** means sample from the model's action probabilities. **D** means choose its most likely legal action.

| Code | Purchasing | Deployment | Work performed |
|---|---|---|---|
| SS | Sampled | Sampled | Reuse completed full-development results |
| DD | Deterministic | Deterministic | Reuse completed full-development results |
| DS | Deterministic | Sampled | New evaluation |
| SD | Sampled | Deterministic | New evaluation |

The question is whether sampling helps more during purchasing, deployment or both. An improvement is not guaranteed. A mode selected here remains development-selected and requires a subsequent fixed final-test protocol.

This diagnostic was specified after reviewing the full-development results and selected replays. It was not preregistered before training. `protocol.json` records the plan before this evaluation begins.

## Fixed design

- Six independent models: P-0, P-1, P-2, PA-0, PA-1 and PA-2, each at 4,194,304 transitions.
- All 156 frozen development layouts, in all eight views.
- Three action-seed labels per mixed mode: 31001, 31002 and 31003.
- **44,928 new attacks:** 6 models × 156 layouts × 8 views × 3 seeds × 2 modes.
- **29,952 reused attacks:** 22,464 SS and 7,488 DD, from the same six final checkpoints.
- **74,880 combined comparison rows.** These include reused evidence; they are not all new attacks and none are training episodes.
- Report all development layouts and, separately, the 144 layouts outside the original monitoring panel. Both scopes remain development data.

The primary metric is percentage of building value destroyed. Secondary results include full clears, Town Hall destruction, zero destruction, and attacks below 50% destruction. Comparisons are calculated within each learner, then averaged equally over the three learners per condition. Attacks and orientations are not additional independent trained models.

DD is evaluated once per layout/view. It is not duplicated into three pretend independent samples. SS, DS and SD average their three action samples before the balanced learner-level comparisons. No significance or equivalence claim is made from this package's descriptive summaries.

## 1. Upload and install

Upload `Base_Siege_V2_Phase_Evaluation.zip` into the existing SageMaker project folder. Open a **SageMaker terminal**, not a notebook cell:

```bash
cd ~/Base_Siege_Smooth_Demo/base-siege-assignment3
unzip -n Base_Siege_V2_Phase_Evaluation.zip
```

Adjust `cd` only if you moved the project. The correct folder contains `env/`, `runs/`, `artifacts/`, `version2/`, `v2_development/`, `pyproject.toml` and `uv.lock`. The ZIP adds only `v2_phases/`. Keep the existing dependency versions.

## 2. Verify the inputs and software

```bash
uv run --frozen python -m v2_phases preflight
uv run --frozen python -m pytest v2_phases/tests -q
```

Expected preflight output includes:

```text
status: ready
models: 6
layouts: 156
new_episodes: 44928
reused_baseline_episodes: 29952
new_modes: [DS, SD]
device: cpu
```

Preflight verifies the completed result files against the reviewed hashes and checks the frozen evaluation contract. It calls the previous evaluator's preflight, which verifies original scientific source, V2 source, run records, development pool, scientific software and all 24 retained milestones used by that evaluation. Only the six 4,194,304-transition checkpoints are used for the new attacks.

The thirteen software tests use the existing original `runs/study-v2/P-0/model.zip`. They perform a few software-check attacks without training. Keep that original checkpoint available. If either command fails, share the complete error before continuing. Do not edit reference hashes or overwrite original results.

## 3. Begin with 1,000 attacks

```bash
uv run --frozen python -m v2_phases run --max-new-episodes 1000
uv run --frozen python -m v2_phases status
```

These are the first 1,000 real attacks, retained for the full comparison. Expected status is `paused` with 1,000 out of 44,928 completed. The terminal reports approximate remaining time every 100 new attacks. The estimate is based on this attempt's measured speed and can change between modes and layouts.

The earlier full-development run averaged about 0.094 seconds per attack, suggesting roughly 70 minutes for 44,928 attacks at the same speed. This is an extrapolation from a different evaluation, not a measured duration or guaranteed completion time for this diagnostic. CPU inference and one Torch thread are intentional.

Continue with:

```bash
uv run --frozen python -m v2_phases run
```

Do not start two copies. This command reuses completed attacks and automatically resumes saved progress. It does not restart training.

## 4. Pause and resume safely

In a **second terminal**, from the same project folder:

```bash
cd ~/Base_Siege_Smooth_Demo/base-siege-assignment3
uv run --frozen python -m v2_phases pause
uv run --frozen python -m v2_phases status
```

The evaluator finishes and saves its current attack, then pauses. Wait for `paused` and the original terminal prompt before stopping SageMaker. After signing back in, check whether the original process is still running before starting another copy.

When the earlier process has stopped, resume using:

```bash
cd ~/Base_Siege_Smooth_Demo/base-siege-assignment3
uv run --frozen python -m v2_phases run
```

Each completed attack is saved in a checksummed SQLite transaction. An abrupt interruption can repeat the unfinished attack; committed attacks are reused. The status file is a last-written snapshot, not a live process detector. The output lock prevents simultaneous writers. Keep code, dependencies and inputs unchanged while paused.

## 5. Package the completed results

Once the evaluator reports `completed`, run:

```bash
uv run --frozen python -m v2_phases pack
```

The command checks the completion receipt and exported hashes, then prints a ZIP path under `artifacts/`, with a name beginning:

```text
Base_Siege_V2_Phase_Results_
```

Refresh the SageMaker file browser, open `artifacts`, right-click the new ZIP and choose **Download**. Upload it here for analysis.

The original results stay under `artifacts/version2-full-development/`. New outputs are stored separately under `artifacts/version2-phase-evaluation/`:

| File | Meaning |
|---|---|
| `evaluation_contract.json` | Frozen design, source/model/result hashes and runtime |
| `progress.sqlite3` | Per-attack recovery database; keep in your project backup |
| `progress.json` | Latest progress snapshot and approximate remaining time |
| `attempts.jsonl` | Attempt starts, finishes and measured durations |
| `mixed_episodes.csv` | The 44,928 new DS/SD attacks |
| `combined_episodes.csv` | New attacks plus the 29,952 reused SS/DD records |
| `model_summary.csv` | Four modes for each learner and reporting scope |
| `condition_summary.csv` | Equal-weight condition means and learner variability |
| `comparisons_by_learner.csv` | DS/SD changes against SS/DD within each learner |
| `condition_comparisons.csv` | Mean changes, learner variability and count of learners improving |
| `complete.json` | Completion receipt and exported hashes |

The compact results ZIP omits the SQLite database. Retain that database for recovery and backup. An unmatched attempt-start event after a hard interruption indicates missing timing; do not fill it using time between sessions.

## How phase switching and random seeds work

The recorded observation's seventh summary value (`vector[6]`) identifies shopping or battle. The tool checks this against the legal Finish shopping action. Choosing Finish shopping follows the purchasing rule; subsequent deployment and wait decisions follow the deployment rule. The policy receives the same observations and legal-action masks as before.

Python, NumPy and Torch are seeded once per episode with the existing V2 episode-seed function. There is no reseeding at the phase boundary. Deterministic actions consume no sampling draws. Therefore switching the purchasing rule changes which random draws remain for deployment. The four modes compare complete strategies; they do not hold the army, battle states or deployment random draws identical across modes.

Uniform SS and DD behavior is tested against the original acting-policy implementation, including exact action sequences in the small tests. The mixed-mode purchasing prefixes are also checked against their corresponding uniform-mode purchasing rules.

## Interpretation and next decision

Inspect condition means, changes for each learner, full clears and low-destruction failures. Avoid selecting a mode from one dramatic replay or one learner. The purpose is to decide which action-selection strategy deserves a subsequent confirmatory evaluation, and whether another training experiment is justified. No checkpoint is automatically promoted and no final evaluation or new training is launched by this package.

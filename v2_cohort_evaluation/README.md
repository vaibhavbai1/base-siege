# Base Siege Version 2: full development evaluation of all twenty learners

Prepared for Vaibhav Bairathi, Master of Artificial Intelligence, UTS, 30 September 2026.

This package evaluates the completed models. It does **not** train, change weights, alter the simulator, or access final layouts. It adds only `v2_cohort_evaluation/` and writes new results under `artifacts/version2-cohort-development/`.

## What will run

| Item | Value |
|---|---|
| Existing screening learners | P-0–P-2 and PA-0–PA-2 |
| New replication learners | P-3–P-9 and PA-3–PA-9 |
| Checkpoint | Final 4,194,304-transition checkpoint of every learner |
| Development layouts | All 156 existing frozen layouts |
| Coordinate views | All 8 |
| DS | Deterministic purchases, sampled deployment |
| SS | Sampled purchases, sampled deployment |
| DD | Deterministic purchases and deployment |
| DS/SS action labels | 31001, 31002 and 31003 per layout/view |
| DD repetitions | Once per layout/view; not triplicated |
| New attacks on fourteen learners | **122,304** |
| Reused attacks on six screening learners | **52,416** |
| Combined attacks | **174,720** |
| Inference | CPU, one PyTorch thread, deterministic computation settings |

Every new learner receives 3,744 DS attacks, 3,744 SS attacks and 1,248 DD attacks. These are evaluation attacks, not additional training episodes. Earlier SD results remain preserved in the phase-diagnostic outputs but are not part of this DS/SS/DD comparison.

The package verifies the earlier phase results before reusing their rows. It keeps the original result files unchanged. Python, NumPy and PyTorch action RNGs are reset once per attack using the existing seed rule; there is no phase-boundary reseeding. DS includes a deterministic Finish Shopping decision and samples all subsequent legal battle decisions. Sampling follows learned probabilities, not a uniform random policy.

## Why report the cohorts separately?

DS was selected using seeds 0–2 after full-development evaluation, replay review and the phase diagnostic. The DS/SS/DD follow-on comparison and final-checkpoint choice were recorded before training seeds 3–9. This implementation and the interval details were specified after training and before viewing full-development results for the new cohort. Do not describe all these choices as preregistered before the initial six-model screen.

The reports distinguish `screening` (seeds 0–2), `replication` (seeds 3–9), and `all` (seeds 0–9). The seven new learners per condition are the prospective replication cohort. The combined ten-learner result includes the three learners used for strategy selection.

Results cover both all 156 development layouts and the 144 layouts outside the training monitoring panel. Those 144 are still development layouts, not a new final test. Models are fixed to their final checkpoint; the code does not select a best seed or best milestone.

## 1. Upload and extract

Upload `Base_Siege_V2_Cohort_Development_Evaluation.zip` into the existing project folder in SageMaker's file browser. Open a **SageMaker terminal**, not a Python notebook cell:

```bash
cd ~/Base_Siege_Smooth_Demo/base-siege-assignment3
unzip -n Base_Siege_V2_Cohort_Development_Evaluation.zip
```

Adjust `cd` if your project is elsewhere. The correct folder contains `pyproject.toml`, `uv.lock`, `env/`, `version2/`, `v2_development/`, `v2_phases/`, `v2_replication/`, `runs/`, and `artifacts/`.

`-n` preserves existing files. If `v2_cohort_evaluation/` already contains a different revision, do not combine revisions. Keep the registered training packages unchanged.

## 2. Check your files and run the software tests

Run these commands one at a time. Continue only when each succeeds:

```bash
uv run --frozen python -m v2_cohort_evaluation preflight
uv run --frozen python -m pytest v2_cohort_evaluation/tests -q
```

Preflight checks the original scientific source and development pool, recorded runtime, original screen checkpoints, all fourteen new final models and their training records, and the earlier phase results. It should report 20 models, 156 layouts, 122304 new episodes, and 52416 reused episodes.

The tests use a previously saved model for a few tiny inference checks and temporary output directories. They do not train. Their fixtures deliberately share one policy under synthetic identities to test bookkeeping; fixture scores are not research results. Tests use the original P-0 checkpoint if available, otherwise the completed Version 2 P-0 final milestone.

If preflight reports a missing or changed file, preserve the error and restore the matching recorded file. Do not change reference hashes. Keep the original screen's reviewed milestone files as well as its six final checkpoints: their existing provenance check is reused. Use your current locked environment rather than upgrading packages.

## 3. Run a short check, then continue

First run 1,000 new attacks:

```bash
uv run --frozen python -m v2_cohort_evaluation run --max-new-episodes 1000
```

It should finish with `status: paused` and `completed_episodes: 1000` on a fresh evaluation. This is intentional. If resuming existing progress, the limit means up to 1,000 additional attacks in this invocation, not 1,000 total.

Then continue without a limit:

```bash
uv run --frozen python -m v2_cohort_evaluation run
```

The same command resumes progress. Completed attacks are reused rather than rerun. The evaluation contract is written before its first attack and checked on every restart. Each completed attack is committed to a checksummed SQLite journal. Run only one evaluator process for this output folder; a lock blocks concurrent writers.

Based on the previous phase-evaluation rate, roughly **3–4 hours** is a planning estimate for the new attacks, not a measured duration or guarantee. The terminal prints a rough remaining-time estimate after its first 100 new attacks. Inference uses CPU even when SageMaker has a GPU, matching earlier evaluations.

Keep SageMaker running. If you already use tmux, run the evaluation inside a session such as `tmux new -s base-siege-evaluation`; reconnect with `tmux attach -t base-siege-evaluation`. A terminal session does not keep a stopped SageMaker instance running.

## 4. Progress, safe pause and recovery

In another terminal, enter the same project folder:

```bash
cd ~/Base_Siege_Smooth_Demo/base-siege-assignment3
uv run --frozen python -m v2_cohort_evaluation status
```

Progress counts **new** attacks and should eventually reach 122304/122304. The 52416 reused attacks are reported separately in the completion receipt. Status is a saved record, not proof a process is still alive after an abrupt termination.

To pause:

```bash
uv run --frozen python -m v2_cohort_evaluation pause
```

Wait for `status: paused` and for the evaluation terminal to return to its prompt before shutting down SageMaker. The current attack is allowed to finish and commit. If a shutdown interrupts an attack, that uncommitted attack is rerun on restart.

After restarting, enter the same project folder and repeat:

```bash
uv run --frozen python -m v2_cohort_evaluation run
```

Keep the entire `artifacts/version2-cohort-development/` directory, especially `progress.sqlite3` and `evaluation_contract.json`. Do not edit source, protocol, dependencies, inputs or checkpoints during the evaluation. An interrupted process can leave an unmatched start record in the timing log; missing durations must be reported as missing rather than zero.

## 5. Export the completed results

When status says `completed`, run:

```bash
uv run --frozen python -m v2_cohort_evaluation pack
```

It prints a timestamped ZIP path under `artifacts/`, named `Base_Siege_V2_Cohort_Dev_Results_*.zip`. Download that ZIP and attach it in this chat. Packing checks result hashes and refuses incomplete output or an active evaluator.

The results ZIP includes all new and combined raw attack rows, summaries, comparisons, the evaluation contract, completion receipt, progress and timing attempts. It omits model weights and the SQLite recovery journal. Keep your full project folders separately as the backup.

## Reading the outputs

| File | Meaning |
|---|---|
| `new_episodes.csv` | The 122,304 new attack results |
| `combined_episodes.csv` | All 174,720 attacks, with cohort labels |
| `model_summary.csv` | Each learner/mode, on all layouts and outside the monitor panel |
| `condition_summary.csv` | Equal-weight learner means for screening, replication and all learners |
| `mode_comparisons_by_learner.csv` | DS minus SS and DS minus DD for each learner |
| `mode_comparisons.csv` | Cohort-level mode differences and learner-pair intervals |
| `augmentation_by_seed.csv` | PA minus P for each matching learner seed and mode |
| `augmentation_comparisons.csv` | Cohort-level PA minus P differences and intervals |
| `evaluation_contract.json` | Frozen inputs, models, settings, code hashes and interpretation |
| `complete.json` | Completion counts and output checksums |
| `attempts.jsonl` | Recorded starts, finishes and measured process wall times |

Destruction is the fraction of building **value** destroyed. Full-clear rate is the fraction of attacks that destroy all scoring buildings. Below-half rate counts attacks with destruction strictly below 50%. None of these is classification accuracy.

Summaries give every independent learner equal weight. Destruction-difference intervals use 20,000 paired learner bootstrap resamples, seed 43001, percentile limits 2.5% and 97.5%. For mode comparisons, the pair is the same learner under two acting modes. For PA minus P, it is matching learner seed across conditions. The three-seed screening cohort does not receive an interval because fewer than five learner pairs are available.

These are descriptive, unadjusted intervals conditional on the fixed development layouts and action labels. They quantify variation across trained learners, not uncertainty over every possible unseen base. They are not corrected for multiple comparisons. The ten-seed aggregate includes the strategy-selection cohort; prefer the separately reported seven-seed replication for the new evidence. A large number of attacks does not create a large number of independently trained models.

This remains a development evaluation. After its review, a separately frozen held-out final protocol is the next stage. No simulator changes, reachable-target fallback, application update or further training are performed here.

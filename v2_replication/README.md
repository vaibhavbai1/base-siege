# Base Siege Version 2: additional learner seeds

Prepared for Vaibhav Bairathi, Master of Artificial Intelligence, UTS. Package prepared 29 September 2026; DS selected from development results on 28 September 2026.

This adds **P-3 through P-9 and PA-3 through PA-9**: fourteen independent training runs. Together with the six completed Version 2 screen learners, this gives ten learners per condition and twenty Version 2 learners. They are repeated runs of the same algorithm, not an ensemble.

## Recorded plan

| Item | Choice |
|---|---|
| New output folder | `runs/version2-seed-replication/` |
| New learner seeds | 3–9 in both P and PA |
| Initialization | Each learner starts independently from scratch |
| Transitions per learner | 4,194,304 |
| Placement-only phase | 32,768 transitions |
| Joint phase | 4,161,536 transitions |
| Total new transitions | 58,720,256 across fourteen learners |
| Environments × rollout | 4 × 2,048 = 8,192 transitions/update |
| PPO batch size / epochs | 256 / 10 |
| Training monitor | Original small SS/DD development panel |
| Selected follow-on evaluation strategy | DS: deterministic purchasing, sampled deployment |
| Checkpoint for follow-on comparisons | Final 4,194,304-transition checkpoint of every learner |
| Final evaluation | Separate future protocol; no final-layout access in this package |

P means PPO without symmetry augmentation. PA adds randomly rotated/reflected coordinate views during training. **PPO training still samples actions in both phases.** DS controls how a trained policy acts; it does not change the learning algorithm. The in-training monitor intentionally stays identical to the six-model screen and reports sampled and deterministic modes, not DS.

An episode is a complete attack. A transition is one decision. The runner counts actual completed episodes; it does not assume an episode count from the transition budget.

The network, rewards, observation, masks, simulator, movement rules, hyperparameters and frozen pools stay the same. No reachable-target fallback is introduced. Previous models and scientific results remain in their existing folders.

## Decision and evidence boundaries

On the six existing learners, DS achieved 93.33% mean destruction for P and 93.43% for PA. SS achieved 92.87% and 93.00%. All six learners improved mean destruction under DS. This is a modest improvement; destruction measures building value destroyed, not win rate.

DS was selected **after** full-development evaluation, replay inspection and the phase diagnostic. It was not preregistered before training the six screen models. The decision is now recorded before the new cohort starts. New learner seeds provide prospective replication on the existing development distribution, not a fresh final-layout test.

Action RNGs are reset once per episode, not at the phase boundary. Changing purchasing sampling changes subsequent random-stream consumption. Interpret this as a comparison of complete acting strategies, not an exact matched-army causal experiment.

`plan.json` records follow-on development evaluation with DS, SS and DD on all 156 frozen development layouts, eight views, and three action labels per sampled mode. DD runs once per layout/view. Report new seeds 3–9 separately from screening seeds 0–2, and show all ten per condition with those cohort labels. Give learners equal weight. Use every learner's final checkpoint; do not report a selected best seed as the condition result.

This package provides training and review export. The full follow-on evaluator and a new held-out final protocol are subsequent stages. Define and freeze that final protocol before examining its results. Do not describe the previously inspected final layouts as unseen.

## 1. Upload and extract

Download `Base_Siege_V2_Seed_Replication.zip`. Upload it through SageMaker JupyterLab's file browser into the existing project folder containing `pyproject.toml`, `uv.lock`, `env/`, `version2/`, `runs/` and `artifacts/`.

Open a **SageMaker terminal**, not a Python notebook cell:

```bash
cd ~/Base_Siege_Smooth_Demo/base-siege-assignment3
pwd
ls pyproject.toml uv.lock version2/plan.json
unzip -n Base_Siege_V2_Seed_Replication.zip
```

If your project is elsewhere, adjust `cd`. The ZIP adds only `v2_replication/`. `-n` keeps existing files. If that folder already contains another version, do not mix versions. Keep the original Version 2 source, run folders and phase results under `artifacts/version2-phase-evaluation/`.

Use your existing locked SageMaker environment. Do not upgrade dependencies or regenerate frozen base pools.

## 2. Check before training

From the same terminal:

```bash
uv run --frozen python -m v2_replication preflight
uv run --frozen python -m pytest v2_replication/tests -q
```

Preflight verifies original scientific source and pools, Version 2 source, six completed screen manifests/contracts, 24 reviewed model milestones, and the phase-result files supporting DS. It checks the recorded Python and scientific dependencies. The tests use tiny temporary CPU models, solely for software validation, without modifying research results.

If a check fails, preserve its exact output. Restore the matching file rather than replacing expected hashes. The package is grounded in uploaded evidence, not a live inspection of your SageMaker checkout. Read `VALIDATION.md` for local verification and limitations.

## 3. Register and run the fourteen learners

On the same GPU-backed SageMaker setup used for Version 2:

```bash
uv run --frozen python -m v2_replication register --device cuda
uv run --frozen python -m v2_replication matrix --to 4194304
```

Registration creates `version2-records/seed-replication/registration.json` before research training. It records the plan, evidence, source hashes, dependencies, device and detected hardware. Repeating registration with matching settings is safe; differing settings are rejected.

The matrix runs P-3 through P-9, then PA-3 through PA-9, sequentially. It resumes incomplete runs and verifies/skips completed ones. A shared lock prevents concurrent new-cohort training commands. It rejects seeds 0–2.

Keep SageMaker running. If you already use `tmux`, start `tmux new -s base-siege-replication` before the matrix command and run training inside it. Detach with Ctrl+B, then D; reconnect with `tmux attach -t base-siege-replication`. A terminal session does not prevent instance shutdown.

For one learner instead of the matrix:

```bash
uv run --frozen python -m v2_replication train --condition P --seed 3 --to 4194304
```

Do not run it concurrently with the matrix. An optional temporary stop at `--to 1048576` does not change the registered final budget; later repeat with `--to 4194304`.

## 4. Monitor, pause and restart

In another SageMaker terminal:

```bash
cd ~/Base_Siege_Smooth_Demo/base-siege-assignment3
uv run --frozen python -m v2_replication status
```

Status shows committed transitions, completed episodes, recorded wall time and attempts without end times. A hard-killed process can leave `running` in the last manifest; status is not a live process check. Recorded time includes monitoring, checkpoint I/O and recovery. Missing attempt durations are not treated as zero-length attempts.

To pause the active learner and stop the matrix before another starts:

```bash
uv run --frozen python -m v2_replication pause
```

**Wait for the training command to return**, then confirm the active learner is paused or completed before shutting down SageMaker. A monitor pass may need to finish first. Already completed learners remain completed.

After restarting, open a terminal and repeat:

```bash
cd ~/Base_Siege_Smooth_Demo/base-siege-assignment3
uv run --frozen python -m v2_replication matrix --to 4194304
```

Keep the registered source, dependencies, device and hardware unchanged for recovery. The command archives the handled stop marker and restores verified recovery state. Scheduled checkpoints occur every 65,536 transitions, plus phase endings and safe pauses. Abrupt failures can require replaying work since the previous committed snapshot. Partial rollouts are not counted as committed learning.

Keep `contract.json`, `manifest.json`, `inputs/`, `LATEST.json` and retained `checkpoints/` for each run. A model ZIP alone is not a full recovery backup. Three verified recovery generations are retained, alongside milestone inference models; corrupt/unrecognized generations remain available for inspection.

## 5. Export for review when finished

Status should show fourteen completed learners at 4,194,304 transitions each, totalling 58,720,256 new transitions:

```bash
uv run --frozen python -m v2_replication status
uv run --frozen python -m v2_replication report > version2-records/seed-replication/development_summary.csv
uv run --frozen python -m v2_replication pack
```

The last command prints a timestamped `Base_Siege_V2_Replication_Review_*.zip` path. Download that ZIP from SageMaker and attach it in this chat. It includes source, plan, decision, registration, contracts/manifests, monitor histories, milestone metadata, status totals and checksums. Packing also supports safely paused partial cohorts and records how many learners completed.

The compact ZIP **excludes model weights, raw training episode logs and recovery states**. It is not a backup. Keep the complete `runs/version2-seed-replication/` and `version2-records/seed-replication/` folders on persistent storage alongside the original results. Episode counts are computed from committed logs on SageMaker; the compact ZIP alone cannot independently recount omitted raw logs.

This stage does not change the browser application's selected policy. After reviewing the new training runs, run the planned DS/SS/DD development evaluation and prepare the separately frozen final evaluation.

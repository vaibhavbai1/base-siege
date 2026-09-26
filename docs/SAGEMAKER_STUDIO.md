# SageMaker Studio: local checkpoints and recovery

Project 12 · Base Siege · 8 September 2026

This workflow uses the persistent project folder in SageMaker Studio. It has no S3 integration,
AWS SDK dependency, bucket configuration or automatic cloud backup.

## What carries over from Deep Learning Assignment 3

The UTSWork Semester 3 Assignment 3 notebooks were inspected read-only on 8 September 2026.
The ResNet50 notebook calls shared training code; that code saves stage progress, model and
optimiser state, scheduler/scaler state, history and a training signature. The EDA and evaluation
notebooks separate explanation, executable checks, tables and figures.

The RL notebooks follow that shared-code structure. They do not duplicate the simulator.
Recovery is adapted to PPO: it also preserves active games, observations and random generators.
No driver-classification code, dataset, trained weights, ClearML integration or S3 workflow was copied.

Reference files in the private UTSWork repository:

- Semester 3/Deep Learning and CNN/assignments/deepLearning-assignment1/assignment3/code/notebooks/01_EDA.ipynb
- The same notebooks folder: 02a_Training_ResNet50.ipynb and 03_Evaluation.ipynb
- The corresponding src/train.py, Git blob 4a45e3229a75babc4288d5afe4a75290cd748382

## Hardware and administrator settings

The supplied screenshots show ml.g4dn.xlarge (4 vCPUs, 16 GiB RAM, one T4) and
ml.g4dn.2xlarge (8 vCPUs, 32 GiB RAM, the same GPU). Start with xlarge and measure the
four-worker workload before changing machines. See [AWS G4 specifications](https://aws.amazon.com/ec2/instance-types/g4/).

The screenshots show 50 GB of storage, with an administrator-imposed 50 GB limit. Monitor free
space; the study controller checks for 8 GiB headroom before each learner. Do not assume this
threshold proves the remaining study will fit, or delete recovery evidence to make space.

The 60-minute setting is an idle timeout, not ordinarily a one-hour job limit. AWS's native
JupyterLab rule checks active kernel and terminal sessions. This course also attaches a named
lifecycle configuration; confirm its behaviour with the administrator before unattended work.
Do not disable or bypass it. See [AWS idle shutdown](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-idle-shutdown.html).

The screenshots also show an image-upgrade warning partly obscured by the instance menu.
Read the full warning, back up existing work and use an approved patched image before the
study. An image update changes the environment; repeat preflight before collecting evidence.

## 1. Use the complete project

Keep source, notebooks, uv.lock, pools, registration and run folders together in the Studio
project. Open notebook 00 and select the project kernel.

~~~bash
uv sync --frozen
uv run python -m ipykernel install --user --name base-siege --display-name "Python 3 (Base Siege)"
~~~

The lock requires Python 3.12. Keep that environment unchanged after starting a study.
This setup has not been installed or tested inside the user's AWS account. Check the kernel's
torch.cuda.is_available() and run a short CUDA smoke test before a long CUDA run:

~~~bash
uv run python -m train.train_ppo --condition P --device cuda --out runs/studio-cuda-smoke
uv run python -m train.train_ppo --resume --out runs/studio-cuda-smoke
~~~

These are commands for later use, not cloud actions already performed. CUDA is explicit:
if unavailable, the trainer raises an error rather than silently training on CPU.

## 2. Confirm storage persistence

The default is project/runs. Notebook paths also support BASE_SIEGE_ROOT and BASE_SIEGE_RUNS_DIR.
For terminal commands, pass the chosen persistent directory with --out.

Modern Studio JupyterLab spaces attach EBS storage to the space. Stopping compute and restarting
the same space preserves that storage. Studio Classic uses EFS-backed home storage and has a
different lifecycle. Confirm which application and volume you use. Do not rely on a temporary
folder or container scratch area. See [AWS JupyterLab spaces](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-jl-user-guide.html)
and [Studio Classic storage](https://docs.aws.amazon.com/sagemaker/latest/dg/security-access-control-studio-nb.html).

Check free disk space before every long run. A GPU accelerates the network; game simulation is
CPU work. Measure throughput on the target machine before choosing how many learners to run
concurrently. Each learner already uses four CPU worker processes in the planned configuration.

Local checkpoints survive only while their storage survives. Deleting the space or its volume,
losing the filesystem, or removing the run folder can lose all recovery points. No independent
backup is created by this project. No S3 service is required.

## 3. Start a fresh run or resume an existing one

New runs refuse to overwrite an existing directory. Full runs require pools, registration and
explicit long-run confirmation. Follow RUNBOOK.md to prepare those first.

~~~bash
uv run python -m experiments.study run --registration experiments/registration-v2.json --device cuda --out runs/study-v2 --confirm-long-run
~~~

After reopening the same Studio project:

~~~bash
uv run python -m experiments.study run --registration experiments/registration-v2.json --device cuda --out runs/study-v2 --confirm-long-run --resume
~~~

Resume uses the saved settings; do not supply a new seed, condition, device or budget. It verifies
the contract, source, configuration, Python and package versions, frozen inputs and checkpoint
checksums. Restore the original environment if compatibility checks fail. Older model-only runs
from before this upgrade support inference, but do not contain enough state for safe continuation.

The controller verifies completed learners and skips retraining them. It resumes the interrupted
learner and continues the registered matrix. Individual learner resume is still available with
train.train_ppo --resume --out followed by that learner's folder.

## 4. Plan a break

From a second Studio terminal:

~~~bash
uv run python -m experiments.study pause --out runs/study-v2
~~~

Watch the study progress.json until status becomes paused. Then stop Studio compute. SIGINT and SIGTERM
also request a pause while training. If an update is underway it can finish and save; if the
rollout is incomplete, the trainer discards it and uses the preceding checkpoint.

If the process dies abruptly, manifest.json may still say running. That is not proof that compute
is active. Resume checks committed checkpoint generations, and the operating-system lock prevents
a second writer while the original trainer really is active.

## 5. What is saved

~~~text
runs/study-v2/P-0/
  contract.json               immutable settings and provenance
  manifest.json               current status, attempts and committed progress
  inputs/                     copies of pools and registration
  checkpoints/
    checkpoint-.../
      model.zip               policy, value network and optimiser
      state.pt                worker games, observations and RNG states
      episodes.jsonl          committed training-history prefix
      checkpoint.json         phase, progress and SHA-256 checksums
  LATEST.json                 pointer to a published checkpoint
  episodes.jsonl              canonical history for the resumed run
  development.json            checkpoint-consistent development history
  recovery/                   resume records and previous uncommitted history
  tensorboard/attempt-.../     separate event files per attempt
  placement.zip               placement-phase endpoint
  model.zip                   completed fixed-budget endpoint
~~~

Schema-3 snapshots reference the immutable pool in inputs/ through a verified digest. The pool
is not embedded once per worker per generation. Each snapshot still includes the dynamic
worker environment and Monitor wrapper, reserve/active troops,
battle state, layout and D4 RNGs, Python/NumPy/PyTorch RNGs, CUDA RNGs when applicable, last
observations and episode-start flags. Upstream SB3 saves the network and optimiser together.
See [SB3 save format](https://stable-baselines3.readthedocs.io/en/master/guide/save_format.html).

PPO's algorithm is unchanged. A small subclass calls persistence after the upstream optimiser
update finishes. The next rollout can be generated from that saved state, so a partially filled
on-policy buffer does not need to be recovered.

## 6. Atomic publication and rollback

1. Write a new staging generation in the checkpoint directory.
2. Save model, state and the current episode-history prefix.
3. Hash and flush files and their directory.
4. Rename the complete generation into place.
5. Replace LATEST.json atomically.

Recovery ignores staging directories and validates committed generations newest-first. A broken
pointer can be reconstructed from committed folders. A corrupt latest generation causes fallback
to an earlier intact generation. Previous uncommitted episode history is archived before restoring
the matching prefix. TensorBoard attempts stay separate, so an abandoned tail is not quietly
mixed into canonical analysis.

Checksums detect incomplete or damaged artifacts, not malicious code. Model and state files use
Python serialisation. Only load trusted local training artifacts.

## 7. What can be lost?

Default saving occurs after every completed rollout update. Full training collects
4 workers × 2,048 decisions = 8,192 transitions per update. A sudden stop can lose the current
rollout and in-progress update, but not require restarting all previous completed training.
Fallback from an already damaged generation can repeat more than one rollout.

The optional --checkpoint-every-rollouts setting increases the maximum unsaved interval.
Its value is stored in the run contract and does not change during resume.

Generations are not pruned automatically. Older size measurements refer to the earlier checkpoint
schema, not a full revision-2 AWS run. Measure current checkpoint and pool sizes during the pilot.
Full runs have 32 completed updates plus phase-start checkpoints. Account for all learner pairs,
state/history growth, packages, root exports and results when choosing disk capacity.

## 8. Verify recovery before the study

~~~bash
uv run pytest -q
uv run python -m tools.check_recovery --out artifacts/studio-recovery-check --kill-process
~~~

The recovery drill performs short CPU runs. It verifies controlled pause/resume and a SIGKILL
of its own child process against an uninterrupted learner: weights, optimiser, episode history,
step counter and update counter must match. Notebook 03b runs the clean-pause version.

Local tests cover both P and PA, placement and joint phases, phase boundaries, rounded budgets,
two workers, corrupted files/pointers, history rollback, moved folders, source mismatch and
single-writer locking.

The tests ran on this Mac's CPU, not an AWS GPU. CUDA deterministic settings and RNG capture
are implemented, but exact equality across different hardware, GPU drivers or library versions
is not promised. Validate the actual Studio kernel before the assessed run.

## 9. Pool and evaluation recovery

Pool generation stores committed candidate records in a companion .progress.sqlite3 file.
Development and final evaluation store each completed episode in evaluation.progress.sqlite3.
Transactions and row checksums prevent a partially written unit from becoming accepted evidence.
The last uncommitted unit may repeat after a stop.

Use --resume with the original command. Changing code, runtime, inputs, views or models is not
a compatible resume. Final evaluation additionally binds its freeze to one output directory.
Keep the journals and final marker with their artifacts; do not delete a marker to retest.

The study controller never invokes final evaluation. Follow the separate final commands in
[RUNBOOK.md](RUNBOOK.md) only after development is complete and the artifacts have been frozen.

## 10. Completion is not convergence

A completed manifest means the requested curriculum ran to the recorded complete-rollout budget,
its endpoint was exported, and save/reload passed. It does not mean the policy converged or won.

The 57,344-step pilot pair, default twenty-learner study, freeze/final execution, report
findings and submission remain separate work. Smoke results must stay labelled as software checks.

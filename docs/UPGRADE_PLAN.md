# Upgrade history

## Current revision: study reliability and team handoff

The 8 September 2026 study revision adds transactional pool/evaluation recovery, compatible
final-evaluation resume, compact pool references in checkpoint schema 3, a study controller,
broader development diagnostics and an explicit pilot-review gate. It directly requests the
262,144-step budget and defaults to ten seeds per condition.

The team master, experiment plan, runbook and notebooks now describe this workflow. Original
source archives, earlier software outputs and draft submissions remain historical evidence.
The compact partial observation and original visual design are deliberately retained.
No long pilot, full study, final-layout access, SageMaker execution or public deployment is
performed as part of this local revision.

## Earlier notebook and training-checkpoint upgrade

The original build supplied the simulator, RL interface, basic training and app. Its notebooks were short demonstrations and its training only saved phase-end models. They were not sufficient for a long, interruption-safe Studio workflow.

Implemented in this revision:

1. Fix Markdown indentation and test the rendered Markdown, not just executable cells.
2. Expand the notebooks into a coherent investigation: setup, mechanics, baselines, training, recovery, evaluation and replay.
3. Save model/optimizer state, worker/environment state, RNG state, phase and committed progress after PPO updates.
4. Publish checksummed checkpoint generations atomically, keep previous generations, and reject incompatible resumes.
5. Support the project's persistent SageMaker Studio folder. No S3 integration or cloud-resource changes.
6. Test uninterrupted versus resumed CPU learning, phase transitions, corrupt/incomplete checkpoints, process interruption, subprocess workers and all notebook cells.

Long training and actual AWS execution remain outside this local verification. Studio storage and GPU preflight must be checked in the user's AWS space before the main study. Keeping the Studio space is essential: local checkpoints do not protect against deleting its storage.

Reference reviewed: the UTSWork Semester 3 Deep Learning Assignment 3 EDA, ResNet50 training and evaluation notebooks, plus their shared `src/train.py`. The workflow informed the notebook structure and local recovery design; the old repository and Drive sources were not edited. See `SAGEMAKER_STUDIO.md` and `VERIFICATION.md` for implementation details and observed checks.

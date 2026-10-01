# Validation record

Package finalized 29 September 2026. Software checks are not research training results.

## Executed checks

Command, from the existing project root:

```bash
uv run --frozen python -m pytest v2_replication/tests -q
```

Result: **17 passed in 37.76 seconds** on CPU. Temporary files were isolated under the test runner's temporary directories.

The tests cover:

- Uninterrupted versus paused/resumed learning: identical network weights, optimizer state, completed episode logs and monitoring history, including multiple workers and symmetry augmentation.
- Equivalence of the original Version 2 and new-cohort learning engines on a small PA run with development monitoring.
- Recovery from a corrupted latest checkpoint using an earlier retained generation.
- Retention of three verified recovery generations and model milestones.
- Sampled evaluation preserving the learner's random states and policy parameters.
- Public registration/start/resume with four workers and temporary test budgets.
- Rejection of changed original Version 2 or new-cohort source on resume.
- Separate paths and seeds 3–9; original screen seeds remain 0–2.
- Evidence hashes and completion checks, including rejection of altered phase evidence.
- Matrix pause between learners, pause before the first optimizer update, restart after a handled stop, and prevention of concurrent matrix commands.
- Compact review-archive checksums and exclusion of model weights.
- Refusal to report a missing committed episode log as zero episodes.

Additional checks confirmed all 28 original scientific source files and frozen pools, and all nine original Version 2 source/protocol hashes, match their recorded references. The seven pinned phase-result files and required completion fields match the uploaded phase-results archive. CLI help and empty-cohort status execute successfully.

## Runtime

Python 3.12.14; NumPy 2.5.2; PyTorch 2.13.0; Gymnasium 1.3.0; Stable-Baselines3 2.9.0; SB3 Contrib 2.9.0; Streamlit 1.63.0. PyTorch reports CUDA build 13.0 but no available GPU in the local validation environment. Tests ran on CPU.

## Scope and limits

The new runner copies the verified Version 2 engine, changes the experiment paths/identity and registered seed set, adds extension-source provenance and a whole-cohort pause marker, and reuses the original environment, feature extractor, PPO settings, monitoring evaluator and recoverable-checkpoint primitives. No movement, reward, observation or learning-rule changes were made.

The 14 research learners were **not** trained here. Long CUDA training, GPU recovery and validation of the user's complete live SageMaker directory require execution there. The full six-model checkpoint collection is not present in this local workspace, so a production registration was not created. Unit tests use explicit temporary evidence fixtures for their tiny public-entrypoint test; production commands do not bypass evidence validation.

The compact review export is not a complete backup and omits raw episode logs. Its episode totals are calculated from the retained committed logs on the machine producing it. Local tests do not establish a future score or a guaranteed training duration.

Before research training, run preflight and the included tests in the existing SageMaker project. Registration then records that machine's actual source, runtime, device and hardware. Do not edit code, plan or dependencies after registration and resume under the same experiment identity.

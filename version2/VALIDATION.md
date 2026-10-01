# Validation record

Verified locally on 27 September 2026.

- **8 tests passed** (`python -m pytest version2/tests -q`, 72.75 seconds for the test suite).
- P and PA pause/resume matched uninterrupted CPU runs: policy weights, optimizer state, episode logs and development history.
- Repeating a reached screening target did not train additional updates.
- Sampled and deterministic diagnostics survived pause/resume with identical recorded results.
- Evaluation preserved the learner weights, train/eval mode and random-number state.
- A damaged latest checkpoint recovered from an earlier retained generation and reproduced the uninterrupted run.
- Resume rejected changed V2 source provenance.
- Invalid transition targets and missing baseline files were rejected.
- Registration and the public start/resume entry point were exercised with four worker processes and the real frozen train/dev input files, using tiny test-only budgets. That test skipped the full development panel; separate tests exercised evaluation on a one-layout panel.
- Original scientific source: all 28 file hashes match the original P-0 contract.
- Frozen train and development pool hashes passed preflight.
- Syntax checks passed for all supplied Python files.

## Scope and limits

Local runtime: Python 3.12.14, PyTorch 2.13.0, NumPy 2.5.2, Gymnasium 1.3.0, Stable-Baselines3 2.9.0, SB3-Contrib 2.9.0. Streamlit was 1.64.0 locally; the old record was 1.63.0. It is not used to run the learner.

The local machine had **no available CUDA GPU**. GPU execution, GPU recovery equivalence, million-transition learning behaviour, resource usage and SageMaker-specific process lifecycle have not been verified here. Repeat the included checks in the actual SageMaker environment and start with one model.

All local learning used tiny temporary software-test models. **No new scientific training run or final evaluation was conducted.** No performance improvement is claimed. The supplied ZIP is an additive package for the existing project, not a complete standalone repository.

During development, evaluation initially attempted to deep-copy the live policy. An after-update test exposed cached autograd tensors that could not be copied. The final implementation instead serializes the policy constructor settings and parameter state into memory and loads a separate CPU policy. The passing evaluation/recovery tests cover the corrected implementation.

The original simulator, training source, observations, reward, network, old checkpoints, evaluation results and handbook were not modified. Movement/targeting limitations remain as documented.

# Validation record

Prepared 30 September 2026.

## Executed checks

From the recovered original project with the exact dependency lock:

```bash
uv run --frozen python -m pytest v2_cohort_evaluation/tests -q
```

**14 passed in 9.99 seconds.** No model training was performed.

Checks covered:

- DS/SS/DD phase flags, including Finish Shopping and battle Wait.
- SS and DD reproducing the original uniform-mode action sequences, summaries and PyTorch RNG state.
- DS retaining the deterministic purchasing prefix.
- Byte-identical result exports after uninterrupted versus paused/resumed inference.
- Completed-unit reuse, safe pause after the current attack, and archiving handled stop markers.
- Rejection of changed contracts, corrupted SQLite rows and concurrent writers.
- DD occurring once per layout/view and duplicate baseline rejection.
- All 122,304 new unit identities, 52,416 reused units and 174,720 combined unit identities.
- Import/validation of the actual uploaded phase-diagnostic rows, including action seed derivation, outcome flags and monitor membership.
- Equal learner weighting even with uneven attack counts in a synthetic test; screening/replication/all cohort separation; direction of DS-reference and PA-P differences.
- Reproducible paired learner intervals without changing the global NumPy RNG; omission for the three-seed screen.
- Rejection of changed model files and checksummed result packaging.

Additional checks confirmed the 28 frozen scientific source files, the original six screen model references against the phase contract, and the fourteen new run-contract/manifest/milestone references against the uploaded replication review. CLI help runs successfully.

## Evidence and scope

References come from `Base_Siege_V2_Replication_Review_20260930_012951_255215.zip` and `Base_Siege_V2_Phase_Results_20260928_125120_734496.zip`, plus the verified earlier Version 2 evaluator packages and original project archive. The attached handbook was used as project context; current implementation details follow the source and recorded contracts.

The phase policy is imported unchanged from the verified `v2_phases` package. Model loading, base loading, CPU configuration and unit identifiers reuse `v2_development`. The transactional execution loop is adapted from `v2_phases`, with a new design, references, output path, validation and cohort-level statistics. Scientific source, trained weights and earlier results are not modified.

Software tests use an existing original 262,144-transition checkpoint in temporary inference fixtures. They cannot establish the performance of the twenty Version 2 models. Those full model files are on SageMaker and are not included in the compact uploaded review. The complete live preflight and 122,304-attack research evaluation must therefore run there. Local full preflight is not claimed to have passed.

The planned inference environment is the recorded Python 3.12.14, PyTorch 2.13.0, NumPy 2.5.2, Gymnasium 1.3.0, Stable-Baselines3 2.9.0 and SB3 Contrib 2.9.0, on CPU with one PyTorch thread. Runtime/config/source checks run before research evaluation and are frozen in its contract.

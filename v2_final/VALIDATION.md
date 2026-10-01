# Validation record — 30 September 2026

The final-evaluation add-on was tested against the recovered Base Siege source
and the uploaded completed cohort-development evidence. All recovered scientific
source files matched the hashes in that evidence. The twenty V2 final model
identities and SHA-256 digests are pinned in `reference.json`.

## Executed software checks

**15 tests passed in 7.54 seconds** in an isolated test environment. The checks
cover:

1. Real DS inference: reproducible actions, unchanged PhasePolicy behavior,
   matching Torch RNG state, and deterministic purchase prefix including the
   finish-shopping action.
2. The ordinary heuristic's actions and summaries match the original policy.
3. Interrupted/resumed and uninterrupted runs export byte-identical scientific
   results; completed runs are idempotent; packing verifies receipt hashes.
4. A pause during an attack commits it once before pausing.
5. A pause during pool filtering retains completed candidates without rechecking.
6. Freeze required, changed frozen settings rejected, concurrent writer rejected.
7. Corrupt evaluation-journal payloads rejected.
8. Injected pool/inference failures leave resumable progress.
9. Equal learner weights, separate familiar/transfer scopes, and independent
   reconstruction of paired bootstrap intervals from a synthetic fixture.
10. Full design has 800 candidate identities and 488 distinct attacks per accepted
    layout, with all twenty model identities and the specified seed derivation.
11. Saved earlier metadata overlapping the new seed range is rejected.
12. Altered/missing checkpoint files fail hash verification.
13. Missing attempt-end timing is flagged, not silently counted as measured time.
14. Freeze is idempotent, generates no layouts, and rejects altered plans.
15. Preflight checks the registered model count and rejects changed scientific
    source and core-package versions using controlled provenance fixtures.

Software inference tests used development seed 10200, two familiar motifs and
an existing **Version 1 P-0 checkpoint at 262,144 transitions** from the essentials
archive. P and PA identities in the small recovery fixture intentionally share
that one policy: these are software tests, not scientific V2 comparisons. They
are written to temporary test directories, never historical result directories.
No training was run. No candidate in 11200–11399 was generated or evaluated.

Test runtime: Python 3.12.14, NumPy 2.5.2, Gymnasium 1.3.0, Stable-Baselines3 2.9.0,
SB3-Contrib 2.9.0, Torch 2.13.0+cpu and pytest 9.1.1. The CPU test build is separate
from the recorded SageMaker Torch 2.13.0 environment. User-facing preflight keeps
its strict recorded-version checks; the test environment is not supplied as a
replacement for the SageMaker environment.

## Checks that must run on SageMaker

The uploaded result archive contains identities and hashes, not the twenty V2
checkpoint files. Consequently the complete production preflight and actual
final evaluation cannot be executed here. `v2_final preflight` on SageMaker must
verify those files, their training records, prior development outputs, source,
helper packages and installed versions. Every model is hashed again at load time,
its internal timestep counter checked, and its policy placed in inference mode.

## Scientific boundaries

The algorithm, rewards, observations, movement and troop targeting remain frozen.
The new code adds final orchestration, a fresh acting-seed namespace, provenance,
resumable persistence and analysis. It imports the original `PhasePolicy`,
`HeuristicCommander`, simulator, structural/difficulty validator and rollout.

Final estimates and confidence intervals do not exist yet. The primary analysis,
secondary analyses, candidate range, acceptance rule, seeds and checkpoint list
are documented before the SageMaker final run. This is a final protocol selected
after development; it is not a preregistration before V2 training.

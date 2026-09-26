# What is built, and what remains

Project 12 · Base Siege · study revision 2 · 8 September 2026

The executable project extends Vaibhav's supplied Stage 5 simulator. It is not yet a completed
research study. [TEAM_MASTER.md](TEAM_MASTER.md) is the current team entry point.

| Area | Current state |
|---|---|
| Simulator, Gymnasium contract and baselines | Implemented, with original ZIP preserved |
| D4 views, custom features and Maskable PPO | Implemented; bounded CPU integration exercised |
| Main experiment | Default ten seeds per condition, exactly 262,144 transitions per learner |
| Development split | Three structures only; historical layered-core exposure explicitly disclosed |
| Training diagnostic | Up to 12 layouts across four views, with recorded identities and spending |
| Training recovery | Update-boundary checkpoints, corruption fallback and schema-3 compact pool references |
| Pool/evaluation recovery | Checksummed transactional journals; explicit same-contract resume |
| Study orchestration | Plan/run/pause/resume controller, completed-run verification and disk headroom checks |
| Pilot and approval | Explicit 57,344-step pilot mode, measured-runtime review and registration gate |
| Final protocol | Relative freeze paths; one freeze/output binding; resumable candidates and episodes |
| Notebooks and documentation | Updated to the revision-2 workflow; shared modules remain authoritative |
| Web application | Existing Vaibhav-style design retained; public Streamlit source preparation retained |
| Long pilots and full study | Not run |
| SageMaker/CUDA verification | Not run |
| Protected final layouts and evaluation | Not accessed |
| Learned-policy research findings, final report and submission | Not completed |
| Public hosting and GitHub CI execution | Not performed |

See [VERIFICATION.md](VERIFICATION.md) for observed local checks. Historical artifacts remain
under their original paths and must not be cited as results of the new source/configuration.

## Decisions intentionally left as scientific limitations

The observation remains compact and partial, with a feed-forward policy. Reward shaping for walls
remains off. Heuristic difficulty filtering defines a restricted procedural population. The study
compares two PPO variants, not two unrelated RL algorithms. These choices keep the central
augmentation question controlled; the pilot may motivate a documented redesign before registration.

## Next execution stage

1. Verify the selected Studio image, CUDA, persistent storage and administrator shutdown policy.
2. Run the local checks and recovery drill on that machine.
3. Build the declared pools, run the pilot pair and review its development evidence.
4. Register the approved design; execute all matched learners with the controller.
5. Finish development analysis, freeze, run/resume the final evaluation and write actual findings.

The code does not claim policy quality, a fixed SageMaker runtime, or cloud reliability from local
software tests. No S3 is required; backups are still needed against loss of the storage itself.

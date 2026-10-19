# TRPO handoff

Condition **T**: MaskableTRPO, ten learners, evaluated against the two PPO
conditions on one frozen pool.

**Start with `EXPERIMENT_DETAILS.md`.** It answers the eleven experiment questions
and the masking question, with values taken from the run contracts rather than
from memory.

## Run one attack

```bash
python handoff/predict_example.py
```

Loads `runs/study-v3/T-1/model.zip`, builds the environment, reads the observation
and the legal-action mask, chooses actions by the frozen evaluation procedure and
plays one complete attack. No preprocessing files, no saved statistics, no
`VecNormalize`.

```bash
python handoff/predict_example.py --mode SS     # sampled purchasing
```

Worth running both. The difference is the main finding; see section 8 of
`EXPERIMENT_DETAILS.md`.

## Where things are

| Looking for | Path |
| --- | --- |
| The algorithm | `trpo/maskable_trpo.py` |
| Hyperparameters | `trpo/config.py`, and per run in `contract.json` |
| Training loop, resume, checkpointing | `trpo/runner.py` |
| Evaluation | `v3_final/` (frozen protocol), `v3_probe/` (acting-mode probe) |
| Acting procedure at evaluation | `v2_phases/runner.py`, class `PhasePolicy` |
| Network | `train/features.py`, plus `POLICY_KWARGS` in `trpo/config.py` |
| Selected checkpoint | `runs/study-v3/T-1/model.zip` |
| Per-attack results | `artifacts/version3-final-evaluation/episodes.csv` |
| Headline comparison | `artifacts/version3-final-evaluation/comparisons.csv` |
| Evaluation layouts | `final_pool.json`, reproducible via `generate_base(seed, motif)` |
| Training curves | `runs/study-v3/T-*/development.json` |
| Interface integration | `ui/v3_models.py`, `ui/app_v3.py` |

## Units

`episodes.csv` reports `destruction` as a **fraction in 0 to 1**. The summary CSVs
report **percent**. The two differ by a factor of 100; please do not mix them.

## Two things to read before judging the numbers

**The DS result understates TRPO.** The frozen protocol uses DS, deterministic
purchasing with sampled deployment. Under it TRPO scores 56.1% destruction against
PPO's 92.9%. On the same checkpoints and the same layouts under SS, sampled
purchasing, TRPO returns to roughly 92%. The learned distribution matches PPO;
the greedy readout does not. Section 8 has the evidence.

**V3 is not comparable with the published V2 figures.** V3 generated a fresh pool
on seeds 21200 to 21399, disjoint from the V2 final pool, because the V2 pool is
spent and its freeze marker stands. Only within-V3 comparisons are controlled. All
three conditions and the heuristic baseline faced identical layouts, views and
action seeds.

## Verifying nothing was changed

```bash
python -m ui.v2_models check              # 20 V2 models, "status": "ready"
python source_check.py                    # identical algorithm across all 10 learners
python -m pytest trpo/tests v3_final/tests v3_probe/tests -q
```

The first re-hashes 28 scientific source files against `ui/v2_checkpoints.json`.
It passes, which is the machine-checkable form of "no environment files changed".

## Not included

Virtual environment, package caches, intermediate checkpoints and milestones, the
raw per-episode training logs (`episodes.jsonl`), and the evaluation journal
(`evaluation.progress.sqlite3`, 108 MB). `episodes.csv` is the artifact of record
and its hash is in `complete.json`. Ask if you want any of it.

# V3 final evaluation

One frozen protocol covering three algorithms on one fresh layout pool:

| Condition | Algorithm |
| --- | --- |
| P | MaskablePPO, no symmetry augmentation |
| PA | MaskablePPO, symmetry augmentation |
| T | MaskableTRPO, no symmetry augmentation |

plus the heuristic baseline. Primary comparison is **T minus P** across ten
matched learner seeds.

## Why a new pool

The V2 final evaluation is frozen and spent: its protocol was fixed before its
pool was generated, and the completion marker stands. Re-running it with a third
algorithm added would destroy the property that makes the V2 numbers credible.

V3 therefore generates a fresh pool on seeds **21200-21399**, disjoint from V2's
11200-11399, under the same structural and heuristic difficulty filter. The
acting-seed namespace is `v3-final-ds-v1`, so the two studies never share a
random stream. `test_v3_pool_does_not_reuse_the_spent_v2_seeds` enforces this.

Because the pools differ, V3 numbers are **not** directly comparable with the
published V2 headline figures. Only within-V3 comparisons are controlled.

## Nothing frozen was modified

`env/`, `agents/`, `experiments/` and `train/` are imported, never edited. All 28
files hashed by `ui/v2_checkpoints.json` stay byte-identical, so
`python -m ui.v2_models check` keeps reporting twenty verified V2 models.

The V2 PPO checkpoints are read from the existing V2 registry rather than copied,
so their hashes live in exactly one place.

## Running

```bash
python -m v3_final freeze     # record protocol + approved checkpoints; generates nothing
python -m v3_final pool       # generate the fresh pool (resumable: --resume)
python -m v3_final run        # evaluate every condition (resumable: --resume)
python -m v3_final export     # episode, model, condition and comparison tables
```

`freeze` refuses until every condition has all ten approved learners registered,
naming the ones that are missing. That is deliberate: a partial freeze would let
a study start before its models exist.

Both `pool` and `run` journal each completed unit to SQLite with a row checksum
and an immutable contract, so an interrupted run resumes without repeating work
and cannot silently change settings mid-study. Re-running `run` after completion
is a no-op.

## Outputs

Written to `artifacts/version3-final-evaluation/`:

| File | Contents |
| --- | --- |
| `freeze.json` | The frozen protocol and approved checkpoint list |
| `final_pool.json` | Every candidate's accept/reject record, plus accepted layouts |
| `episodes.csv` | One row per attack |
| `model_summary.csv` | Per learner, per scope |
| `condition_summary.csv` | Per condition, with bootstrap intervals |
| `comparisons.csv` | Paired differences and intervals |
| `timing_summary.json` | Measured execution, accumulated across resumes |
| `complete.json` | Completion record and output hashes |

Scopes are `all_final`, `familiar`, `transfer` (layered_core) and one per motif.
Intervals are a percentile bootstrap over **learner seeds**, not attacks, and are
omitted with fewer than five learners rather than being reported misleadingly.

## Reading the comparison honestly

The question is how the algorithms learn, not whether TRPO wins. Two asymmetries
are recorded in `protocol.json` and belong in the write-up:

- SB3's TRPO hard-codes `ent_coef=0.0`; both PPO conditions trained at `0.01`.
  TRPO trains without the exploration pressure PPO had.
- TRPO's critic update zeroes gradients on actor parameters, so the shared
  `SiegeFeatures` trunk is updated only by the trust-region step. Under PPO both
  losses reach it.

TRPO is closer to PPO than DDQN is, and that is a strength: both are on-policy
actor-critic methods sharing GAE, rollout structure and network, so the
comparison isolates trust region versus clipped surrogate with nearly everything
else held constant.

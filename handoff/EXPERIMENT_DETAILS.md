# TRPO experiment details

Condition **T** in the Base Siege project. Everything below is taken from the run
contracts and published artifacts in this ZIP, not from recollection. Every claim
can be checked against `runs/study-v3/T-*/contract.json` or
`artifacts/version3-final-evaluation/`.

---

## 1. Which TRPO implementation

**sb3-contrib's `TRPO`, subclassed to add action masking.** sb3-contrib ships
`MaskablePPO` and `TRPO` but no maskable TRPO, and `TRPO` uses the plain
`ActorCriticPolicy`, which knows nothing about masks. With 246 discrete actions
behind a legality mask, unmasked TRPO would not be a usable comparison.

`trpo/maskable_trpo.py` defines `MaskableTRPO(TRPO)`. It reuses the maskable
policy, the maskable dict rollout buffer and `collect_rollouts` from the
MaskablePPO side. The update is rewritten, because TRPO builds an action
distribution at four separate points and each must be built under the same mask
the action was sampled with.

Package versions (from the contracts): `stable-baselines3 2.9.0`,
`sb3-contrib 2.9.0`, `torch 2.13.0`, `numpy 2.5.2`, `gymnasium 1.3.0`,
Python 3.12.14.

## 2. Learner seeds and number of models

**Ten independent models, learner seeds 0 through 9**, in `runs/study-v3/T-0`
through `T-9`. Plus one pilot at seed 100 (`runs/trpo-pilot-100`), used only to
validate the implementation and measure runtime before committing to the study.

All ten are included in the evaluation results. One checkpoint, **T-1**, is
provided for interface integration.

## 3. Transitions completed

**4,194,304 per model, all ten.** Not a target but the recorded count: each
`contract.json` carries `settings.total_steps = 4194304`, each `manifest.json`
reports `status: completed`, and `model.num_timesteps` on each checkpoint equals
that figure. The V3 evaluation re-checks this at load time and refuses a
checkpoint whose counter disagrees.

## 4. Training from scratch

**Yes.** Every learner was initialised fresh. No pretrained weights, no warm start
from a PPO checkpoint, no fine-tuning. Each run began with
`python -m trpo.runner pilot`, which refuses to write into a non-empty directory.

Some learners were interrupted (a disk-exhaustion failure, and one stalled run)
and resumed from their last committed checkpoint. Resumption continues the same
run, verified by a contract hash check on the checkpoint; it is not a restart.
The evidence is in `runs/study-v3/T-*/recovery/`.

## 5. Rotation and reflection augmentation

**No.** Condition T trains without symmetry augmentation, matching condition P.
This is deliberate: P is the matched control, so `T minus P` isolates the
algorithm change. PA is the augmented PPO arm and is a secondary comparison only,
because `T minus PA` would confound algorithm with augmentation.

The D4 views still appear at **evaluation** time, where all three conditions are
evaluated on all 8 views equally.

## 6. Hyperparameters, architecture and hardware

Matched to V2 PPO wherever the two algorithms share a parameter:

| Parameter | Value | Note |
| --- | --- | --- |
| `gamma` | 1.0 | undiscounted, episodic; same as PPO |
| `gae_lambda` | 0.95 | same as PPO |
| `n_steps` | 2048 per worker | same as PPO |
| workers | 4 (`SubprocVecEnv`) | same as PPO |
| `learning_rate` | 3e-4 | **critic only**; TRPO's actor step is not gradient descent |
| `batch_size` | 256 | value-function minibatch only |
| `normalize_advantage` | True | |

TRPO-specific, left at **SB3 defaults and recorded rather than tuned**. A tuned
TRPO against an untuned PPO would not be a controlled comparison:

| Parameter | Value |
| --- | --- |
| `target_kl` | 0.01 |
| `cg_max_steps` | 15 |
| `cg_damping` | 0.1 |
| `line_search_shrinking_factor` | 0.8 |
| `line_search_max_iter` | 10 |
| `n_critic_updates` | 10 |
| `sub_sampling_factor` | 1 |

**Two asymmetries that cannot be removed and should be weighed in any comparison:**

- SB3's TRPO hard-codes `ent_coef = 0.0`. Both PPO arms trained at `0.01`. TRPO
  trained without the exploration pressure PPO had.
- TRPO's critic update zeroes gradients on actor parameters, so the shared
  `SiegeFeatures` trunk is updated only by the trust-region step. Under PPO both
  losses reach it.

**Architecture**, identical to PPO. Custom extractor `train.features.SiegeFeatures`:
two conv layers over the 8-layer 16x16 grid (8 to 32 to 64, stride 2, ReLU), a
9-dimensional vector branch through Linear(9, 32) with ReLU, concatenated and
fused through Linear(1056, 256) with ReLU. Heads are `pi: [128]`, `vf: [128]`.
`normalize_images = False`. **No observation normalisation anywhere**: no
`VecNormalize`, no saved statistics, nothing to ship alongside the checkpoint.

**Hardware:** AWS SageMaker Studio, `ml.g4dn.xlarge`, one Tesla T4, 4 vCPU,
torch CUDA build 13.0, Linux 6.12.103 (Amazon Linux 2023). Training on `cuda`;
the final evaluation ran on **CPU** by protocol.

## 7. Checkpoint selection and the evidence for it

**T-1 selected**, on per-learner development metrics at the final budget:

| Learner | Destruction | Full clear | Town hall | Return |
| --- | --- | --- | --- | --- |
| **T-1** | **95.0%** | **79.9%** | **99.3%** | **1.35** |
| T-5 | 94.5% | 76.4% | 97.2% | 1.33 |
| T-4 | 93.6% | 72.2% | 97.2% | 1.31 |
| T-2 | 93.3% | 72.9% | 93.1% | 1.30 |
| T-8 | 92.9% | 69.4% | 96.5% | 1.30 |
| T-6 | 92.7% | 72.9% | 94.4% | 1.29 |
| T-0 | 91.8% | 66.7% | 93.1% | 1.27 |
| T-3 | 91.2% | 71.5% | 91.7% | 1.27 |
| T-7 | 90.5% | 61.8% | 88.9% | 1.23 |
| T-9 | 89.5% | 69.4% | 91.7% | 1.23 |

T-1 leads on all four. Two caveats stated plainly:

- These are **development-pool** figures, under **sampled** acting. The pool was
  used during training, so selecting on it is selection on seen data. T-1 is the
  demonstration model, not the result.
- The reported result comes from all ten learners on the held-out V3 pool.

**There is no per-learner "best checkpoint within a run".** Each run's final
checkpoint at 4,194,304 transitions is used. No early stopping, no picking the
best intermediate milestone.

## 8. Deterministic or sampled, per phase

This matters more for TRPO than for PPO, so it is set out in full.

An attack has two phases: **purchasing**, then **deployment**. There is no separate
waiting policy; waiting is an action within the deployment phase, so it follows
the deployment setting.

| Mode | Purchasing | Deployment |
| --- | --- | --- |
| DD | deterministic | deterministic |
| SS | sampled | sampled |
| **DS** | **deterministic** | **sampled** |
| SD | sampled | deterministic |

**The frozen V3 evaluation uses DS.** Development checks recorded DD and SS.

**TRPO is strongly sensitive to the purchasing setting and PPO is not.** On the
frozen pool under DS, TRPO reaches 56.1% destruction against PPO's 92.9%, while
losing its army in 72% of attacks against PPO's 9.5%, despite spending 199.7 of
200 resources. Under SS, on the same checkpoints and same layouts, TRPO returns
to roughly 92%.

The same split appears in the development logs for all ten learners throughout
training: at 4,194,304 transitions, mean deterministic destruction is 0.365 while
mean sampled destruction is 0.925. Sampled saturates by about 1.3M transitions;
deterministic climbs slowly and never plateaus.

The reading we take from this: **TRPO's learned distribution is as good as PPO's,
but its mode is not.** A trust region capped at KL 0.01 per update sharpens a
246-way categorical too slowly for the greedy readout to become reliable within
4.19M transitions, even though the distribution itself is correct by 1.3M.

A DS-only comparison therefore understates TRPO considerably. Please read the
56.1% alongside the SS figure rather than on its own.

## 9. Base pools, views and action seeds

**Training:** `bases/train-v2.json`, the same pool the V2 PPO conditions used.
SHA-256 `25ee1296...`. Development checks used `bases/dev-v2.json`, SHA-256
`79b61544...`, at every 262,144 transitions with sample seeds 31001, 31002, 31003.

**Final evaluation:** a fresh pool on candidate seeds **21200 to 21399**, disjoint
from the V2 final pool's 11200 to 11399. 800 candidates screened by the same
structural and heuristic difficulty filter, **241 accepted** (179 familiar,
62 transfer on `layered_core`). Recorded in `final_pool.json`, SHA-256
`3afa8ba7...`.

Layouts are **generated from seeds**, so they are fully reproducible rather than
merely listed: `generate_base(seed, motif)`.

**Views:** all 8 D4 transforms. **Action seeds:** 41001, 41002, 41003. That gives
24 attacks per accepted layout per learner, and 175,448 attacks in total.

Per-attack action seeds are derived as
`sha256("v3-final-ds-v1:{label}:{motif}:{base_seed}:{view}")`, truncated to 32
bits. The `v3-final-ds-v1` namespace differs from V2's, so the two studies never
share a random stream.

**Comparability caveat, stated upfront:** because V3 uses a different pool from
the published V2 evaluation, **V3 numbers are not directly comparable with the V2
headline figures.** Only within-V3 comparisons are controlled. All three
conditions and the heuristic baseline faced the identical 241 layouts, the
identical 8 views and the identical action seeds.

## 10. Changes to the environment

**None.** No reward, observation, action-space, movement, termination or
deployment-timing change. Nothing under `env/`, `agents/`, `experiments/` or
`train/` was modified.

This is verifiable rather than merely asserted. `ui/v2_checkpoints.json` records
SHA-256 hashes of 28 scientific source files, including `env/config.py`,
`env/combat.py`, `train/features.py` and `experiments/rollout.py`. Run:

```bash
python -m ui.v2_models check     # expect "status": "ready", 20 V2 models
```

A single byte changed in any hashed file fails that check and stops the
application loading the approved V2 checkpoints. It passes.

The TRPO work is **additive only**: new packages `trpo/`, `v3_final/`, `v3_probe/`,
and two new UI files. TRPO hyperparameters live in `trpo/config.py`, deliberately
not in `env/config.py`, which is hash-guarded. The V3 UI entry point
`ui/app_v3.py` reaches the interface by rebinding a catalog function through
`runpy` rather than editing `ui/app.py`.

All ten learners also carry an identical SHA-256 for `trpo/maskable_trpo.py` in
their contracts, so the algorithm did not change mid-study. Check with
`python source_check.py`. `trpo/runner.py` does differ between early and late
learners, because the checkpoint-retention fix landed partway through the study.
That file schedules checkpoints and parses manifests; it does not touch the
update rule.

## 11. Measured durations and what the timers include

| Measurement | Value |
| --- | --- |
| TRPO throughput | 2.01 ms per transition |
| PPO throughput | 1.70 ms per transition |
| Overhead | about 18% more per transition |
| Per learner | about 1.87 hours |
| Ten learners | about 18.7 hours of training |
| V3 final evaluation | 14,886.5 s (4.13 h) for 175,448 attacks, CPU |

All figures are **measured on this hardware**, not estimated and not borrowed
from the DDQN arm.

Timer scope, from `timing_summary.json`: **execution only.** Excludes pauses,
preflight checks, model loading between conditions, and pool generation. Timing
accumulates across resumes, so an interrupted run's work still counts toward the
measured total.

Training was run **one learner at a time.** Each learner spawns 4 simulator
workers and the workload is CPU-bound on simulation, so concurrent learners on a
4-vCPU instance oversubscribe rather than parallelise.

---

# Legal-action masking

Flagged for particular attention, so stated directly.

**Masking is part of the algorithm. Invalid actions are never sampled and then
replaced.**

## During training

`MaskableTRPO` builds a `MaskableCategorical` at every point the update needs an
action distribution, always under the same mask the action was sampled with. The
mask travels through `MaskableDictRolloutBuffer` alongside observations and
actions, so the mask applied during the update is the mask that was in force when
the action was taken, not a recomputed approximation.

Two implementation details that matter for review:

**Distribution build order is load-bearing.** The policy owns a single
`action_dist` object that `proba_distribution()` mutates in place. The `no_grad`
reference distribution must be built first and shallow-copied; building it second
silently detaches the live distribution and every gradient below comes back
empty. The code comments this at the point it happens.

**Masking uses a large negative logit, not negative infinity.** `MaskableCategorical`
substitutes `-1e8`. This keeps Fisher-vector products finite under heavy masking,
which matters here because TRPO's conjugate-gradient step differentiates through
the KL divergence. With `-inf`, that product is `NaN` and the trust-region step
fails silently.

## During evaluation

`PhasePolicy` (`v2_phases/runner.py`) passes the mask into `predict()` as
`action_masks`, then asserts that the chosen action is legal and **raises** if it
is not. It does not substitute a legal action. A masking bug surfaces as a crash
rather than as quietly degraded numbers.

The identical object is used for conditions P, PA and T, so all three are driven
through the same acting code. The only difference between them is the checkpoint
loaded.

## Where to look

| File | What it shows |
| --- | --- |
| `trpo/maskable_trpo.py` | masked distributions in the update, build order, finite masking |
| `v2_phases/runner.py` | `PhasePolicy`, the evaluation-time acting procedure |
| `trpo/tests/test_masked_trpo.py` | legality, mask alignment, zero probability on masked actions, finite Fisher-vector products, KL binding, line-search restore |
| `handoff/predict_example.py` | the whole path end to end on one attack |

```bash
python -m pytest trpo/tests v3_final/tests v3_probe/tests -q
```

# Masked TRPO

A third algorithm arm alongside the V2 PPO conditions (P, PA). Adds code; changes
nothing that already exists.

## Why this package exists

sb3-contrib ships `MaskablePPO` and `TRPO`, but no maskable TRPO. `TRPO` uses the
plain `ActorCriticPolicy` and knows nothing about action masks. With 246 discrete
actions and a mask that keeps purchases and deployments legal, unmasked TRPO is not
a usable comparison.

The maskable policy, the maskable dict rollout buffer and `collect_rollouts` are
reusable from the MaskablePPO side. The update is not: TRPO builds an action
distribution at four separate points, and each must be built under the same mask the
action was sampled with.

## Frozen source

`ui/v2_checkpoints.json` hashes 28 files, including `env/config.py`,
`train/features.py`, `train/checkpoints.py` and `train/train_ppo.py`. Editing any of
them fails `python -m ui.v2_models check` and the application stops loading the
twenty approved V2 checkpoints. This package imports them and modifies none, so

```bash
python -m ui.v2_models check     # still "status": "ready", 20 models
```

TRPO hyperparameters live in `trpo/config.py`, not `env/config.py`.

## Files

| File | Contents |
| --- | --- |
| `config.py` | Hyperparameters; shared constants imported read-only |
| `maskable_trpo.py` | `MaskableTRPO` and `RecoverableTRPO` |
| `runner.py` | Training execution, pause/resume, contract and manifest |
| `tests/test_masked_trpo.py` | Correctness checks; run before any training |

## Running

```bash
# correctness first -- a masking bug does not crash, it misleads
python -m pytest trpo/tests -q

# pilot: seed 100, no augmentation, 262144 transitions
python -m trpo.runner pilot --out runs/trpo-pilot-100 --device cuda

# after an interruption
python -m trpo.runner resume --out runs/trpo-pilot-100
```

`pilot` refuses to overwrite a non-empty directory. `--device cuda` raises if CUDA is
absent rather than training on CPU silently, matching the PPO trainer. It also
refuses a `--steps` budget smaller than one rollout, which would otherwise commit
zero transitions and report success.

Checkpoints commit every 8 rollouts and only the 3 newest generations are kept
(`--checkpoint-every-rollouts`, `--keep-checkpoints`), matching the V2 defaults.
Both matter: each generation copies the whole `episodes.jsonl`, which grows as
training proceeds, so committing every rollout and keeping every generation fills
the disk quadratically. An abrupt stop costs at most 8 rollouts.

Run **one learner at a time.** Each spawns `N_TRAINING_ENVS = 4` simulator
workers and the workload is CPU-bound on simulation, so two concurrent learners
on a 4-vCPU instance do not run twice as fast — they oversubscribe, and a worker
dying takes the run down with `BrokenPipeError`.

## Two traps worth knowing

**Distribution build order is load-bearing.** The policy owns one `action_dist`
object that `proba_distribution()` mutates in place. The `no_grad` reference
distribution must be built *before* the live one; reverse them and the live
distribution is silently detached, every gradient comes back empty, and nothing
raises. `train()` and the tests both build reference-first, and
`test_fisher_vector_product_is_finite_under_heavy_masking` asserts `kl.grad_fn is
not None` to catch a regression.

**Invalid step lengths.** If the curvature denominator is non-positive the step
length is `nan`. Upstream TRPO writes that into the weights. Here an invalid step
counts as a failed line search and the actor is restored;
`test_invalid_step_length_does_not_poison_the_policy` covers it.

The `NaN`-from-masked-logits risk did *not* materialise: sb3-contrib masks with
`-1e8` rather than `-inf`, and Fisher-vector products stay finite with 241 of 246
actions masked. The test keeps it that way.

## Differences from PPO that the write-up must state

Matched: `gamma` 1.0, `gae_lambda` 0.95, `n_steps` 2048, 4 workers, `SiegeFeatures`,
`net_arch` pi/vf 128, curriculum split, pools.

Not matchable:

- `clip_range` — replaced by the trust region.
- `n_epochs` — TRPO takes one policy step per rollout; `n_critic_updates` governs
  only the value function.
- `ent_coef` — SB3's TRPO hard-codes `0.0`. PPO trained at `0.01`, so **TRPO trains
  without the exploration pressure PPO had.** A real asymmetry; report it rather
  than engineering around it.
- `batch_size` — same name, different meaning: value-function minibatch only.
- `learning_rate` — critic only; the policy step comes from the KL constraint.

One behavioural note: `SiegeFeatures` is shared between actor and critic, and TRPO's
critic update zeroes gradients on actor parameters. The shared trunk is therefore
updated only by the trust-region step, whereas under PPO both losses reach it.

## Diagnostics

`development.json` records per-update KL, line-search success rate, policy objective
and value loss. Without those there is no way to tell from the logs whether the trust
region is binding. Watch KL stays at or below `target_kl`, and that the line search
backtracks sometimes but does not reject constantly.

## Scope

Development runs on the development pool. The V2 final evaluation is frozen and
already spent; re-running its protocol with a third algorithm would destroy what
makes the V2 numbers credible. Application integration (`ui/v2_checkpoints.json`,
`ui/policy_playback.py`) assumes a two-condition registry and is deliberately
untouched.

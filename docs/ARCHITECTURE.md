# Architecture

## One authoritative environment

The app, baselines, notebooks and PPO all use `SiegeEnv`. Only that environment advances the battle. A replay displays saved decision snapshots; it does not recalculate combat in JavaScript.

```text
seed + motif → BaseSpec → BattleState
                              ↓
policy observation ← SiegeEnv.step(action) ← legal masked action
                              ↓
                     up to 10 combat ticks
                              ↓
               observation + reward + ending flags
                     ↙                      ↘
              PPO rollout               snapshot / UI
```

`BaseSpec` is immutable. `BattleState` holds mutable HP, reserve/deployed troops, time, destroyed value and milestones. `GridState` provides spatial lookup and pathfinding access. `env/config.py` supplies numerical rules.

## Decision contract

| Action IDs | Meaning | Legal when |
|---|---|---|
| 0–3 | Buy tank, melee, ranged or breaker | Shopping, enough resources, army below 25 |
| 4 | Finish shopping | Shopping |
| 5–244 | Deploy one troop: four types × 60 perimeter positions | Battle, that type remains in reserve |
| 245 | Wait | Battle |

The perimeter starts at the northwest corner and proceeds clockwise. Deployment ID is `5 + troop_index * 60 + perimeter_index`. Shopping takes zero battle ticks. Deployment and wait advance up to ten, stopping early if the battle ends.

Illegal actions raise before changing state. Masks remove impossible choices, not strategically weak choices. Finishing with an empty army is legal; a learner may discover that doing nothing avoids spending, so this failure mode must be measured.

Full destruction, the 300-tick deadline, or no deployed/reserve troops set `terminated`. A separate 70-decision safety guard sets `truncated`. Any ended episode requires a reset before another step.

## Observation

The grid is `(8, 16, 16)` float32. Five channels indicate Town Hall, cannon, mortar, storage and wall locations. Three encode remaining building health, active defence coverage and troop density. Health is divided by maximum HP, coverage by five and density by eight, clipped at one.

The nine-value vector contains four reserve counts, remaining budget, remaining time, phase, destroyed-value fraction and live troop count. Counts use the army cap of 25. All values are in `[0, 1]`.

Individual troop type, HP, target and movement credit are hidden. This is a partial observation. The renderer and event log expose diagnostic state that the policy does not receive.

## Combat and pathfinding

Each tick removes existing dead entities, invalidates stale paths, chooses troop targets, moves or attacks, lets defences choose targets, applies buffered damage, removes new deaths and settles score/end conditions. Attackers alive when damage was scheduled can still deal that damage even if they die in the tick.

A* costs are in estimated ticks: movement costs `1 / speed`; a wall adds `wall_hp / damage_per_tick`. A troop cannot enter a live wall. It attacks the obstruction and moves only after removal. Integer movement credits avoid floating-point drift at slow speeds.

Range uses Euclidean distance. Walls do not provide line-of-sight cover. Mortars have a minimum range and record both the primary impact and every splash victim. Destruction counts non-wall building value, not the number of occupied cells or damage dealt. Terminal kills are scored in the same tick.

## Reward

`reward = newly destroyed value / initial non-wall value + new milestones - one-time cost penalty`

Each of 50% destruction, Town Hall destruction and full clear pays 0.25 once. Spending costs `0.30 * spent / 200`, once when shopping ends. Fixed-army episodes apply the cost on their first battle action. Optional wall shaping is zero in the main configuration.

The heuristic army costs 199, so its cost term is -0.2985. Destroying all value adds 1.0, and all milestones add 0.75: total return 1.4515. Return and destruction are different metrics.

## Learning architecture

The spatial branch has two 3 × 3 convolutions (32 then 64 channels, stride 2). A 9 → 32 MLP handles the vector. Concatenated features feed a 256-unit fusion layer, then separate 128-unit policy and value heads. `normalize_images=False` preserves already-normalised floats.

Maskable PPO learns a distribution over legal actions and an estimate of future return. SB3 Contrib provides rollout collection, PPO loss and optimisation. Our code defines the environment, features, masks, curriculum and experiments.

The curriculum first learns deployment with the fixed 199-resource army, then learns shopping and deployment jointly. The same model continues; workers reset into phase two. Gamma is 1.0 because episodes are finite and shopping does not advance game time. Gamma does not fix partial observability.

## D4 augmentation

PA samples four rotations, optionally reflected, at each reset. Grid coordinates and deployment actions transform together; scalars, purchases and wait remain unchanged. Masks are permuted into visible coordinates; chosen actions map back before `SiegeEnv.step`.

View and layout RNGs are separate. P and PA receive the same base-sample sequence for the same reset seed and calls. Different policies can still have different episode lengths. The physical simulator is not rotated, so its coordinate tie-breaks remain unchanged.

## Recovery architecture

`RecoverablePPO` uses upstream Maskable PPO without changing its learning algorithm. It invokes a persistence hook only after an optimiser update completes. `SnapshotMonitor` serialises each worker's environment, wrapper, RNGs and in-progress episode. The checkpoint also includes last observations, episode-start flags, global Python/NumPy/Torch/CUDA RNG state, and committed history. A partial rollout is discarded on interruption and collected again after restore.

New checkpoint generations are checksummed, flushed and published by atomic directory rename, followed by an atomic latest-pointer replacement. Recovery verifies generations newest-first and can fall back from a damaged one. The immutable run contract binds settings, source, packages and copied input pools. A kernel-owned file lock prevents two writers; it is released if the process dies.

Run paths are local and portable as complete folders. The Studio workflow uses persistent project storage, with no S3 integration. Local CPU equality is tested; AWS GPU execution and cross-hardware bitwise equality are not claimed. See `SAGEMAKER_STUDIO.md` for the recovery boundary.

## Experiment recovery and orchestration

The current workflow has three recovery units: completed PPO updates, validated pool candidates
and completed evaluation episodes. SQLite journals use full synchronous transactions, explicit
single-writer locks, row checksums and an immutable input/settings contract. Exports can be
completed after interruption without replacing different existing evidence.

The study controller binds a matrix to one registration and device, alternates condition order,
checks free storage and resumes or verifies each learner. It never opens final layouts.
A separate pilot-review gate verifies the pilot models, development evaluation and team note
before full registration.

Final freezes use relative model paths. The final marker binds one freeze to one output;
compatible restarts reuse committed pool candidates and episodes. A changed model, source or
destination is rejected. These guards are not protection against intentional source tampering.

Schema-3 worker checkpoints externalise the immutable pool through a content digest. Their
dynamic game state and random generators remain serialised. The complete run-input folder is
needed for recovery.

## Evaluation and provenance

Pools retain candidate counts, rejection reasons, accepted layouts, configuration and generator/validator hashes. Training manifests record package versions, source hashes, settings, pool checksums, requested/actual steps, runtime and model checksum.

P and PA must match learner seeds, configuration, pools and actual curriculum budgets. Each learner plays identical layout/structure/view units. PA minus P is averaged within each learner; a t-based 95% interval is then calculated across independent learner seeds. Results are separated by structure and training-family versus transfer-family layouts. The default is ten independent seeds per condition; games are not independent learners. The interval is conditional on the frozen evaluation pool.

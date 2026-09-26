# Initial Assignment 3 implementation plan

Historical build plan. The current study decisions and execution sequence are in
[TEAM_MASTER.md](TEAM_MASTER.md), [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) and
[RUNBOOK.md](RUNBOOK.md). The later revision uses an explicit 262,144-step budget,
a separate pilot-review gate and resumable pool/evaluation journals.

Project 12, Dihedrals. This folder is independent of the older `base-siege` demo.

## Scope

Use Vaibhav's Stage 5 archive as the simulator and visual-design source. Preserve the received ZIP in `reference/`. Implement the Part A/B scope: 16 by 16 grid, 200 resources, four troop types, army selection then deployment, Maskable PPO with and without D4 view augmentation, valid-random and heuristic baselines, separate seed partitions, and a local Streamlit application.

The referenced `BASE_SIEGE_PROJECT_PLAN_v3.md` was not supplied. `BUILD_ORDER_v2.md`, executable configuration, and the submitted Part A/B scope are the available design evidence. No trained-policy result is assumed.

## Build sequence

1. Regression tests and simulator corrections: live-wall collisions, terminal scoring, simultaneous endings, complete attack events.
2. Gymnasium reset/step, 246 actions, masks, normalized observations and decomposed rewards.
3. Baselines, structural/difficulty validation, guarded pools and replay records.
4. D4 observation/action transformations and transition-invariance tests.
5. Custom CNN/summary feature extractor, Maskable PPO curriculum, manifests, development callbacks and fixed-budget evaluation.
6. Streamlit app using Vaibhav's HTML palette, board glyphs and layout with Python as the sole simulator.
7. Executable notebooks for environment exploration, baselines, learning, evaluation and replay. Shared code remains in modules rather than diverging notebook copies.
8. Local tests, notebook execution, browser interaction and short training smoke checks only.

## Architecture

```text
env/config + generator + entities + pathfinding + combat
                           |
                       SiegeEnv
          +----------------+-----------------+
          |                |                 |
       baselines       D4 wrapper       Streamlit session
          |                |                 |
          +---------- Maskable PPO     HTML canvas component
                           |
               checkpoints / run manifests
                           |
                 paired evaluation / CSV
                           |
                   analysis notebooks
```

## Decisions to record

- HP is applied simultaneously; movement precedes defence targeting. Final scoring is settled in the same tick. Ending priority: full clear, deadline, exhausted army, matching the source order.
- A troop damages an encountered wall each attack tick and cannot enter it while alive. Movement does not accumulate while attacking.
- D4 changes the agent's view, not the physical simulator. Coordinate tie-breaks in the simulator are not claimed to be physically rotation-equivariant.
- Gymnasium receives partial observations. No recurrent policy or claim of a fully observed MDP.
- Full experiment settings retain source values. PPO rollouts can overshoot requested budgets; record actual placement/joint budgets and require matching before comparing runs.
- Final seed ranges are guarded even before final pools exist. Do not generate, view or evaluate final study layouts during this build.
- The build-order difficulty filter is implemented explicitly; unfiltered demonstration layouts must never be described as the registered filtered experimental pool.

## Verification boundary

No full 250,000-step experiments, final-test access, Canvas submissions, commits or pushes are part of this build. Development smoke artifacts must be labelled as software checks, not research findings.

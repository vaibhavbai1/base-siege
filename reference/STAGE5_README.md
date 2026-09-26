# Base Siege

Learning army selection and troop deployment with Masked PPO.

A custom Gymnasium environment in which an agent attacks a defended base on a
16×16 grid. It makes two linked decisions: buy an army under a 200-resource
budget, then choose where and when to deploy troops during a battle.

**Research question.** At a matched 250,000-timestep training budget, does
rotation and reflection augmentation improve MaskablePPO's generalisation to
unseen layouts?

Design authority: `BASE_SIEGE_PROJECT_PLAN_v3.md` (not in this package — keep it
alongside).
Build authority: `BUILD_ORDER_v2.md`.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip freeze > requirements.txt      # pin versions, then commit
```

Python 3.10 or newer (required by Gymnasium 1.3.0).

## Running the tests

From anywhere in the repository:

```bash
pytest -q
```

**160 tests pass as of Stage 5.**

## Changing a stat

`env/config.py` is the only file to edit. Then:

```bash
python tools/sync_prototype.py     # push the numbers into the prototype
pytest -q                          # 160 tests: parity AND design invariants
```

`tools/sync_prototype.py --check` reports drift without writing.

The prototype's constants live in a BEGIN/END GENERATED block — never hand-edit
it, or the next sync overwrites you. Colours and description flavour text are
prototype-only and are preserved across syncs. Per-stage gate counts in BUILD_ORDER §6 predate D8
and under-count; the running total in BUILD_ORDER §1 is authoritative.

---

## Build state

| Stage | Module | Status |
|---|---|---|
| 0 | repository, tooling | done |
| 1 | `env/config.py` | done |
| 2 | `env/entities.py` | **done** (18 tests) |
| 3 | `env/generator.py` (structural) | **done** (24 tests) |
| 4 | `env/pathfinding.py` | **done** (18 tests) |
| 5 | `env/render.py` | **done** (13 tests) |
| 5 | `env/combat.py` | **done** (30 tests) |
| 6 | **Checkpoint A** — random agent, speed profile | not started |
| 7 | `env/siege_env.py` | not started |
| 8 | `agents/heuristic.py` | not started |
| 9 | generator validation, base pools | not started |
| 10 | `train/train_ppo.py` | not started |
| 11 | **Checkpoint B** — 50k smoke run | not started |
| 12 | `env/symmetry.py` | not started |
| 13 | `train/evaluate.py` | not started |
| 14 | `ui/app.py` | not started |
| 15 | protocol runs | not started |

---

## Rules that are easy to break

These are settled decisions. Changing one silently invalidates results.

1. **`action_masks()` is a method on the environment class**, never the
   `ActionMasker` wrapper. `ActionMasker` is incompatible with `SubprocVecEnv`.
2. **`gamma = 1.0`.** SB3 discounts per timestep and shopping does not advance
   game time, so `gamma < 1` penalises larger armies for taking more decisions
   to buy.
3. **Path costs are in tick units.** `open_tile_cost = 1 / speed` and
   `wall_cost = open_tile_cost + hp / damage_per_tick`. Costing an open tile at
   1 mixes grid steps with ticks.
4. **Walls score 0.** Compute the reward divisor from live buildings; never
   hardcode 155.
5. **The 300-tick deadline terminates, it does not truncate.** Only the
   70-decision safety guard truncates.
6. **The observation is partial, not Markov.** Troop density only — no
   per-troop health or targets.
7. **Damage is buffered within a tick** and applied simultaneously.
8. **Use `MaskableEvalCallback` and the maskable `evaluate_policy`**, not the
   standard Stable-Baselines3 versions.
9. **The layered core holds a mortar, not a cannon** (D25). It is blind to
   its own 3×3 interior and covers the entire wall ring plus 52 of 56 outer
   tiles. Breaching it is how you stop the shelling.
10. **Movement credit is an integer**, in units of `1/SPEED_SCALE` of a tile.
   A float accumulator drifts: `0.10` added ten times is `0.9999999999999999`,
   and a tank never moves.
11. **Bases are seeded from `seed` alone**, not `(seed, motif)`. The three ring
    motifs therefore share building placement at a given seed. Deliberate — D16.
12. **Nothing blocks a shot.** Range alone decides what can be hit. The
   ranged troop is the only troop with range above 1, so it is the only one
   that can attack across a wall; everything else must get through.

## Seed protection

| Purpose | Base seeds | Access |
|---|---|---|
| Training | 0 – 9,999 | free |
| Development | 10,200 – 10,399 | free |
| Protected final | 11,000 – 11,999 | **none before freeze** |

`bases/pool_final.json` is generated once and not opened until freeze. Every
development tool must refuse to load it, and the GUI must refuse final-range
seeds.

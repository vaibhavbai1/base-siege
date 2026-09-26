# Base Siege — Build Order v2

**Supersedes** `BUILD_ORDER.md`. Companion to `BASE_SIEGE_PROJECT_PLAN_v3.md`.

The v3 plan is the *research* document — what we are testing and how we report it.
This is the *implementation* document — what to write, in what order, how to know each
piece is finished, and what to do the moment it is.

| Item | Value |
|---|---|
| Version | v2 |
| Date | 10 August 2026 |
| Supersedes | BUILD_ORDER.md (v1, 9 August) |
| Status of Stages 0–1 | Complete, 43 tests passing |
| Reason for v2 | Eight design decisions taken during Stage 1 that v1 does not contain |

---

## 0. How to read this

Build in stage order. Do not skip. Each stage has seven parts:

| Part | What it is |
|---|---|
| **Owner** | Which team member leads it |
| **Blocked by** | What must be finished first |
| **Why** | The design reasoning — this is your viva material, not decoration |
| **Contract** | Exact signatures the module exposes |
| **Acceptance** | Numbered tests, each with what it proves |
| **Pitfalls** | The specific things that go wrong here |
| **After** | What to do the moment the tests pass |

Three stages are **checkpoints**. If a checkpoint fails, stop. Building past a failed
checkpoint is how projects die in week 9.

**Every module gets its tests written in the same session as its code.** Not later.
Later never comes.

---

## 1. Current status

### Done

| Stage | Module | Tests | Committed |
|---|---|---:|---|
| 0 | Repository, venv, pytest, `.gitignore`, README | — | yes |
| 1 | `env/config.py` | 43 | yes |
| 2 | `env/entities.py` | 18 | yes |
| 3 | `env/generator.py` (structural half) | 24 | yes |
| 4 | `env/pathfinding.py` (+ `GridState`) | 18 | yes |
| 5 | `env/combat.py` | 30 | yes |
| — | defence rebalance: fire rate, splash falloff, ranges, core mortar | 6 | yes |
| — | prototype parity with `config.py` + `tools/sync_prototype.py` | 9 | yes |
| 5a | `env/render.py` (the renderer half of Stage 5) | 13 | yes |
| — | `tools/watch_episode.py` + episode walkthrough figure | — | yes |
| — | `tools/preview_motifs.py` + four motif figures | — | yes |
| — | `tools/render_demo.py` + three render figures | — | yes |

`render.py` was written out of order, deliberately. It consumes a **snapshot dict**
rather than live objects, so it has no dependency on `entities.py` or `combat.py`.
Writing it first means the renderer *defines* the snapshot schema that
`combat.record_snapshot()` must satisfy, instead of being retrofitted to whatever
that function happens to emit. The schema is documented at the top of the module and
asserted by `test_example_snapshot_matches_the_config_it_claims_to_use`.

### Next

Stage 6, CHECKPOINT A. The simulator is complete and playable end to end.

### Test count

**160 passing as of Stage 5** (including the defence-rebalance tests). Per-stage gate counts printed in §6 were written before
D8 moved the renderer forward to Stage 5, so they under-count from Stage 2 onward; the
running total in this section is authoritative. v1 targeted 63 tests for the core. We are
on track for well over **139**, because `config.py`
carries 43 rather than the 4 v1 anticipated. That is not padding: most of those 43 assert
design invariants that would otherwise break silently — a stat change that kills the
breaker's purpose, a seed range that starts overlapping, an action-encoding drift. See §7.

---

## 2. Decision register

Every non-obvious choice, why it was made, what was rejected, and where it lives in code.
**Bring this to the viva.** Question 20 on the v3 checklist is "what is the strongest
limitation of the final system", and the difference between a good and a weak answer is
whether you can show the alternatives were considered rather than missed.

### Settled

| # | Decision | Rejected alternative | Why | Recorded in |
|---|---|---|---|---|
| **D1** | Deployment tiles ordered **clockwise from (0,0)** | Row-major; four separate per-edge lists | Makes a 90° rotation exactly a cyclic shift of +15 in tile index, which turns half of Stage 12 into arithmetic. Reflections are *not* cyclic shifts — that is why Stage 12 still needs an explicit bijection test | `config.DEPLOY_TILES`, `config.ROTATION_INDEX_SHIFT`, `test_90_degree_rotation_is_a_cyclic_shift_of_the_tile_index` |
| **D2** | Wall ring is a **10×10 box, rows/cols 3–12, exactly 36 tiles** | The plan's "~36" | 4×10−4 = 36 exactly. Removes the tilde and makes wall count a testable property rather than an approximation | `config.CENTRED_RING_BOX` |
| **D3** | `BUILDABLE_MIN/MAX = 1/14` is the **hard** bound; the tighter motif interior is derived in `generator.py` | One fixed buildable box in config | The motif interior depends on which motif is being built, so it is a rule, not a constant. Config holds geometry; the generator holds the rule that turns geometry into a placement region | `config.BUILDABLE_MIN/MAX` |
| **D4** | Gates are **centred on their edge, 2 tiles wide**; corners are never removed | Random gate position per seed | Templates must be deterministic. A centred gate has a radius of influence of roughly 3 tiles (see D9), so it creates a spatial gradient rather than funnelling the whole army to one tile | `config.GATE_WIDTH` |
| **D5** | **Combat range is Euclidean. Movement is 4-neighbour A\*.** Two metrics, deliberately | A single metric for both | Splash radius 1.5 only makes sense as a Euclidean radius (it captures the full 3×3 block, since diagonals sit at 1.41). Movement must be 4-neighbour because the grid is a lattice. Mixing them turns the mortar's range-6 circle into a diamond and shifts every base's difficulty | `config.DISTANCE_METRIC_COMBAT`, `config.PATHFINDING_NEIGHBOURS` |
| **D6** | **`layered_core` replaces `offset_ring`** as the transfer motif | Offset ring (shifted 10×10) | A shifted ring is the same topology moved, so a policy trained on the closed ring transfers to it almost free — a weak structural test. Layered core is a genuinely different topology: one breaker no longer opens the base, all eight orientations are distinct, and it restores meaning to the two §6.7 structural checks that are otherwise vacuous | `config.TRANSFER_MOTIF`, `config.LAYERED_*` |
| **D7** | **Nothing blocks a shot.** Range alone decides what can be hit | Solid tiles block line of sight | Blocking seals every defence inside a closed ring, unable to fire outward at all — the approach phase becomes free and motif difficulty inverts. The asymmetry we wanted comes from the range table instead: ranged is the only troop with range above 1, so it is the only one that can attack across a wall | `config.WALLS_BLOCK_LINE_OF_SIGHT`, `config.WALL_CROSSING_TROOPS` |
| **D8** | The **renderer moves to Stage 5**, alongside `combat.py` | Renderer at Stage 14 only | Graphics are a debugging tool before they are a demo. Stage 5 has eleven acceptance tests about targeting, splash, minimum range and buffering — the exact class of bug that costs a day in a debugger and a minute in a picture. Stage 14 needs the same snapshot-rendering function anyway, so this is reordering, not scope creep | This document, §6 Stage 5 |
| **D16** | Base RNG seeded from **`seed` alone**, not `(seed, motif)` | Hashing the seed with the motif name | The three ring motifs share a 64-tile placement region, so one seed gives all three the same buildings. Kept because: the augmented-vs-baseline comparison is paired on identical bases either way and is unaffected; ~5,000 training episodes against 30,000 available bases means diversity is not the binding constraint; and constant placement across motifs removes a variance component from motif-level secondary analysis, which matters at five learner seeds. Revisit only if the training budget rises far enough that episode count approaches pool size | `generator.generate_base`, `test_ring_motifs_share_a_layout_at_a_given_seed` |
| **D17** | Movement credit accumulates as an **integer** over `config.SPEED_SCALE`, not as a float | The float accumulator prescribed by Stage 2 | The float version fails Stage 2's own acceptance test 1: `0.10` added ten times is `0.9999999999999999`, so a tank never takes its first step. Fixing position drift while leaving credit drift in place is half a fix. Integer arithmetic removes floating point from the movement path entirely, which is what bit-identical episodes across machines actually require | `config.SPEED_SCALE`, `entities.Troop.accumulate_movement`, `test_slow_troop_advances_exactly_one_tile_after_ten_accumulations` |
| **D18** | `GridState` is **defined in `pathfinding.py`** | Leaving it undefined, as the contract does | The Stage 4 contract names `GridState` in `find_path`'s signature and no stage ever creates it — it appears exactly once in this document. Defining it in its first real consumer means the interface is shaped by an actual caller rather than guessed at, the same argument that put `render.py` ahead of `combat.py`. Stage 5 takes ownership and mutates it | `pathfinding.GridState` |
| **D19** | Cache version bumps on **any building death**, not only wall death | The contract's `wall_state_version` | A destroyed cannon opens its tile, so a wall-only counter leaves every cached path routing around a building that is no longer there. Bumping on any death costs at most ten extra invalidations per episode against thirty-six for walls. Renamed `topology_version` so the wider meaning is visible at the call site. Accepted staleness: partial wall damage lowers cost without bumping the version, which is deliberate — re-keying on every HP change would destroy the hit rate, and the error vanishes when the wall dies | `pathfinding.GridState.topology_version`, `test_a_building_death_also_invalidates` |
| **D20** | `Troop.target` holds a **tile**; `Building.target` holds an **entity id** | One identifier kind for both, as the Stage 2 contract assumed | Each entity is identified by whatever is actually unique about it. Two buildings can never share a tile, so position identifies one; troops CAN stack on a deployment tile, so only an id does. Forcing a single scheme would mean either giving buildings ids they do not need or encoding tiles as ints and decoding at every call site | `entities.Troop.target`, `combat.acquire_target` |
| **D21** | Troop→building ties break by **ascending (row, col)**; defence→troop ties by **ascending entity_id**. Closes O1 and O2 | One tie-break rule for both directions | Follows directly from D20 — a tie-break key has to be a total order over the things being compared. Both directions are pinned by tests, because a weak tie-break fails the shuffle test one run in twenty rather than every time, which is far harder to diagnose than a clean failure | `combat.acquire_target`, `combat._acquire_defence_target`, `test_ties_break_by_ascending_*` |
| **D22** | Defences have a **fire rate**: `damage_per_shot` + `attack_cooldown`. Mortar 45 every 5 ticks; cannon 5 every tick | Every defence firing every tick | A mortar firing every tick is a beam, not artillery: there is no moment of safety after a shell lands, so timing an approach is pointless and the mortar teaches the agent nothing spatial. 45/5 = 9, so sustained damage is UNCHANGED — the burst pattern is the only variable that moved, which keeps the change interpretable. Stat renamed from `damage_per_tick` so the mortar's number cannot be misread as a rate | `config.BUILDING_STATS`, `entities.Building.attack_cooldown`, `test_mortar_fires_once_every_cooldown_ticks` |
| **D23** | Splash **falls off** outside the impact tile: full damage at the centre, `SPLASH_FALLOFF` × damage on the surrounding eight | Uniform damage across the 3×3 | Uniform splash made the mortar's aim irrelevant — landing a shell one tile off cost nothing, so a well-placed shell and a lucky one were the same event. Falloff punishes a clustered army harder at the point of impact, which is a spatial property an agent can learn to avoid. Splash still radiates from where the shell lands, not from the gun, so it reaches past the mortar's own range | `config.SPLASH_FALLOFF`, `combat._step_5_defences_act`, `test_mortar_splash_falls_off_outside_the_impact_tile` |
| **D24** | Mortar range **6 → 4.5**; cannon range unchanged at 3 | Cutting the cannon too, or raising it to 4 | Raising the cannon to 4 would break the standoff annulus — `ranged` range 4 must strictly exceed cannon range, or the ranged troop is a strictly worse melee troop and rational shopping never buys one. 4.5 is the lowest the mortar can go while still clearing both locked invariants: it must exceed ranged range 4, and it must reach the deployment ring from the interior edge at distance 4 | `config.BUILDING_STATS`, `test_mortar_outranges_the_standoff_band`, `test_the_cannon_cannot_answer_a_troop_on_the_deployment_ring` |
| **D25** | The layered core holds a **mortar**, not a cannon. Supersedes D14 | Cannon, per D14 | Measured from the core centre: a cannon covers 9/9 interior, 16/16 wall ring, 4/56 outer. A mortar at range 5 covers 0/9 interior, **16/16 wall ring**, **52/56 outer**. Every breach must happen at the wall ring — a troop has to stand adjacent to a wall to break it — so the mortar covers the breach and additionally shells the whole outer fight, which a cannon cannot. The core becomes an artillery emplacement the attacker must silence rather than a prize it can ignore, giving the transfer motif a tactical problem no training motif poses. Cost, stated plainly: once inside the 3×3 nothing fires back. Locked value rises 55 → 60, outer-only ceiling falls 64.5% → 61.3% | `config.LAYERED_CORE_CONTENTS`, `test_the_core_mortar_covers_the_breach_but_not_the_interior`, `test_the_core_mortar_shells_the_outer_region` |
| **D26** | Mortar range settled at **5.0**, not 4.5 | 4.5, and the original 6 | 4.5 was measured to be on the wrong side of a tipping point: an all-ranged perimeter camper scored 76.3% against the heuristic's 73.9%, because only 64% of bases had any mortar able to reach the deployment ring. At 5.0 that rises to 92.5% of bases and the camper loses (65.4% vs 67.9%). Band-crossing survival is identical at 4.5, 5.0, 5.5 and 6.0 — only the tank lives — so nothing is given up by choosing 5.0 | `config.BUILDING_STATS`, measured sweep in the Stage 5 handover |
| **D27** | The prototype's constants are **pinned to `config.py` by a test** | Keeping the two in sync by hand | The prototype is a second implementation of the same game, so it holds a second copy of every number, and that copy silently drifted during the Stage 5 rebalance — mortar range, fire rate, splash falloff and the core contents all moved in Python and not in JavaScript. A balance tool that plays a different game from the simulator is worse than no balance tool. `tools/sync_prototype.py` regenerates the prototype's constants from `config.py`, so there is one source of truth and one command; `tests/test_prototype_parity.py` reads the constants out of the HTML and fails if they disagree; it also checks the prototype actually *implements* the cooldown and falloff rules, since matching numbers mean nothing if the loop ignores them | `tests/test_prototype_parity.py` |

### Derived facts worth knowing (not decisions, but consequences)

| # | Fact | Why it matters |
|---|---|---|
| **D9** | A troop will accept a detour of **3.2 tiles (melee) to 3.8 (tank)** rather than break one wall; a breaker accepts **0.2** | Gates attract troops arriving near the middle of their edge and not those arriving near a corner. Breakers ignore walls entirely and walk straight through. Verified by Stage 4 test 3 |
| **D10** | A ranged troop on the deployment ring can already hit **28 of the 64 interior tiles** without moving, because the interior edge sits at exactly its range of 4 | This is the same number 4 that gives the standoff against cannons. You cannot drop one without the other. If composition collapses to all-ranged, the fix is geometric (shrink the placement region), not statistical |
| **D11** | Even with optimal placement, two mortars cover only **19 of 60 perimeter tiles** | Two-thirds of the deployment ring is always safe. Finding the safe arc *is* the spatial task, and it is orientation-dependent — which is precisely what hypothesis 3 predicts augmentation should help with |
| **D12** | `MAX_ARMY_SIZE = 25` equals `RESOURCE_BUDGET / CHEAPEST_TROOP_COST` exactly | The army cap binds only for an all-melee army. The budget is the real constraint |
| **D15** | A tank whose target rule is `nearest_defence` **falls back to the nearest non-wall building** once every defence is destroyed. A breaker does not fall back | Leaving idle | Found by playing the prototype: with no fallback a tank stands idle for the remainder of the battle once the five defences die, wasting 900 HP of army. The breaker is `walls_only`, so it physically cannot attack a building; idling is the honest cost of overbuying breakers and is information the agent should learn. Recorded in `config.TARGET_RULE_FALLBACK` |
| **D14** | The layered core holds a **Town Hall, a cannon and a storage** — not the Town Hall alone. Closes O3 | Locks 55 of 155 behind the second wall, so stripping the outer region alone caps at 64.5%, below the 85% difficulty ceiling. Scoring well therefore requires breaching both rings, which is the problem the motif exists to pose. **SUPERSEDED 24 Aug by D25 — the core now holds a MORTAR and locks 60, capping outer-only destruction at 61.3%.** The original argument was that a mortar cannot defend the 3×3 compartment it sits in, which is true of the interior and misses the wall ring. Recorded in `config.LAYERED_CORE_CONTENTS` |
| **D13** | §6.5's balance arithmetic is **wrong**: it divides pooled army HP by pooled defence DPS | Defences focus-fire on one target each, so five defences are five independent kill-slots (~85 fragile-troop kills over 300 ticks against an army of at most 25). The binding constraint is time-under-fire during the approach, not damage capacity. Must be footnoted in §6.5 — see §9 |

### Open — must be closed at the stage named

| # | Question | Close at | Default if undecided |
|---|---|---|---|
| **O1** | What do defences target? Nearest troop? Lowest HP? Most splash value? | Stage 5 | Nearest troop, ties broken by ascending entity id (deterministic) |
| **O2** | How are ties broken in troop targeting ("nearest defence", "nearest building")? | Stage 5 | Ascending entity id. Must be deterministic or the shuffle-invariance test fails |
| ~~O3~~ | ~~What goes inside the layered core?~~ | **Closed 12 Aug** | See D14 |
| **O4** | Learner seed count: 5 or 10 | Before Week 7 freeze | Gated on timing one full 250k run. Under ~90 min → 10 seeds |
| ~~**O5**~~ | ~~Shrink the building placement region to the inner 6×6 to kill perimeter sniping?~~ | **Closed 24 Aug** | **No.** No longer a prediction: measured directly. At mortar range 5.0 a 20-troop all-ranged camper scores 65.4% against the heuristic's 67.9%, so camping is already dominated and the placement region does not need shrinking. Would have been the wrong call at range 4.5, where the camper wins — the fix was the mortar's range, not the geometry |

---

## 3. Stack and environment

| Component | Choice | Note |
|---|---|---|
| Python | 3.10+ | Required by Gymnasium 1.3.0 |
| Environment API | `gymnasium` | Not the deprecated `gym` |
| RL | `stable-baselines3` + `sb3-contrib` | MaskablePPO lives in contrib |
| Arrays | `numpy` | The main tool for making the simulator fast |
| Testing | `pytest` | |
| GUI | `streamlit` | Browser-based; no display-driver risk on a presentation machine |
| Plotting | `matplotlib` | Figures, debug renderer, and GUI board rendering — one library, three jobs |
| Logging | `tensorboard` | |

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip freeze > requirements.txt      # pin, then commit immediately
```

Run manifests must record exact versions (v3 §13.2). An unpinned `requirements.txt`
cannot support the clean-start verification in §13.3.

### Why Streamlit rather than pygame

Streamlit runs in a browser: no display driver, no second-monitor surprise, and screen
sharing works. Replay steps through recorded snapshots rendered as matplotlib figures.
Less smooth than pygame, far more robust — and robustness is what a 15-point live demo
needs. The cost is that the Tier 2 overlay toggles become checkboxes rather than keypresses.

---

## 4. Repository layout

```
base_siege/
  env/
    __init__.py
    config.py          all constants, no logic                    [DONE]
    entities.py        Building, Troop                            Stage 2
    generator.py       motifs, placement, validation, pools       Stages 3, 9
    pathfinding.py     A* with tick-unit costs and caching        Stage 4
    combat.py          the seven-step tick loop                   Stage 5
    render.py          snapshot -> matplotlib axes                Stage 5
    base_spec.py       base save/load as JSON                     Stage 3
    siege_env.py       Gymnasium environment                      Stage 7
    symmetry.py        rotation and reflection wrapper            Stage 12
  agents/
    __init__.py
    random_agent.py    valid-random from the mask                 Stage 6
    heuristic.py       hand-written commander                     Stage 8
  train/
    train_ppo.py       MaskablePPO training                       Stage 10
    evaluate.py        paired evaluation and statistics           Stage 13
  ui/
    app.py             Streamlit interface                        Stage 14
  tools/
    preview_motifs.py  motif figures from config                  [DONE]
  tests/
    __init__.py
    test_config.py     [DONE, 43]
    test_entities.py test_pathfinding.py test_combat.py
    test_generator.py test_render.py test_env.py test_mask.py
    test_reward.py test_termination.py test_symmetry.py
    profile_env.py     speed measurement, not a test
  bases/
    pool_train.json  pool_dev.json
    pool_final.json    generated but NOT opened until freeze
  figures/             design and report figures (gitignored, regenerable)
  runs/                checkpoints, manifests, CSVs
  conftest.py  requirements.txt  README.md
  BUILD_ORDER_v2.md  BASE_SIEGE_PROJECT_PLAN_v3.md
```

**Every tunable number lives in `config.py`.** A magic number anywhere else is a bug.

---

## 5. The stage ritual

Do all seven, every stage, in order. It takes about twenty minutes and it is the
difference between a project you can defend and a pile of code you cannot.

### 1. Run the whole suite, not just the new file

```bash
pytest -q
```

Not `pytest tests/test_entities.py`. The whole thing. Stage 7 will break a Stage 2
assumption at some point and you want to find out the same day.

### 2. Check the stage gate

Each stage lists an explicit gate under **After**. Checkpoints A and B are hard stops.

### 3. Update the build-state table in `README.md`

One line. It is how a teammate — or a marker — knows where the project actually is.

### 4. Record any decision

If the stage forced a choice that is not already in §2, add a row: what you chose, what
you rejected, why, and where it lives in code. Do this *now*, while you remember the
reasoning. Reconstructing it in November is guesswork.

### 5. Commit with the stage number

```bash
git add -A
git commit -m "Stage 4: pathfinding.py — 7 tests, 57 passing total"
```

Checkpoints get a tag:

```bash
git tag checkpoint-a && git push --tags
```

The commit history is contribution evidence. v3 §15 requires weekly records per member;
a clean per-stage history produces them for free.

### 6. Teach it to the other two — fifteen minutes

The person who owns a module explains it to the other two before the next stage starts.
Use the **viva questions** listed for that stage as the agenda.

This is not ceremony. Assignment 3 is *individually assessed* with an oral defence
(Part G, 25 points). A teammate who cannot explain `combat.py` loses those marks
personally, no matter how good the code is. Fifteen minutes a stage is the cheapest
insurance available.

Log the date and who attended.

### 7. Update the contribution log

One line per person per week: what you wrote, what you reviewed, what you reproduced.
v3 §15 requires that every member makes technical contributions *and* reviews another
member's work. Backfilling this in week 12 is both painful and obvious.

---

## 6. Stages

### Stage order at a glance

```
Stage  0   setup, repo, pytest running                    [DONE]
Stage  1   config.py                                      [DONE — 43 tests]
Stage  2   entities.py                                    [DONE — 18 tests]
Stage  3   generator.py (motifs and placement)             [DONE — 24 tests]
Stage  4   pathfinding.py                                  [DONE — 18 tests]
Stage  5   combat.py + render.py                          [DONE — 26 tests]
Stage  6   CHECKPOINT A — random agent, speed, balance
Stage  7   siege_env.py                          B
Stage  8   heuristic.py                          C
Stage  9   generator validation + pools          C
Stage 10   train_ppo.py                          B
Stage 11   CHECKPOINT B — 50k smoke run
Stage 12   symmetry.py                           B
Stage 13   evaluate.py                           C
Stage 14   ui/app.py                             C
Stage 15   protocol runs                         all
Stage 16   freeze, final evaluation, report      all
```

C is blocked until Stage 6 clears. Use that window for the Part A and B written
deliverables, the plan edits in §9, and the statistics section of the report.

---

### Stage 0 — setup [DONE]

**Gate met.** Repository skeleton, virtual environment, `pytest` running, `conftest.py`
resolving imports from any working directory, `.gitignore` protecting `bases/pool_final.json`
and `runs/`.

---

### Stage 1 — `config.py` [DONE]

**Gate met.** 43 tests. All constants from v3, plus decisions D1–D7 recorded as constants
with tests asserting them.

The four v1 acceptance criteria all pass: `TOTAL_BASE_VALUE == 155`, wall value exactly 0,
60 unique perimeter tiles, no buildable coordinate on a deployment tile.

---

### Stage 2 — `env/entities.py`

**Owner** A · **Blocked by** Stage 1

#### Why

Two classes with mutable state and immutable stats. The whole stage exists for one
problem: **fractional speed without floating-point position drift.**

A tank moves 0.10 tiles per tick. The naive implementation stores position as a float and
adds 0.10 each tick. After 300 ticks of binary floating-point addition the position is not
30.0, and worse, it is not *reproducibly* not-30.0 across machines. Reproducibility is a
hard requirement — the entire seed-split protocol depends on identical bases producing
identical episodes.

The fix is a **movement credit accumulator**: keep position as integers, accumulate speed
as a float, and move exactly one tile when the accumulator crosses 1.0, subtracting 1.0
rather than resetting to 0.0. Error stays bounded and position is always exact.

#### Contract

```python
@dataclass
class Building:
    building_type: str          # key into config.BUILDING_STATS
    position: tuple[int, int]   # (row, col), immutable after construction
    hp: int                     # current, initialised from stats
    max_hp: int
    target: int | None = None   # entity id of the troop it is firing at

    @property
    def is_alive(self) -> bool
    @property
    def is_wall(self) -> bool
    @property
    def value(self) -> int              # 0 for walls
    @property
    def damage_per_tick(self) -> int
    @property
    def attack_range(self) -> int
    @property
    def min_range(self) -> int
    @property
    def splash_radius(self) -> float
    @property
    def health_fraction(self) -> float  # for observation layer 5
    def take_damage(self, amount: int) -> None

@dataclass
class Troop:
    troop_type: str
    position: tuple[int, int]
    hp: int
    max_hp: int
    entity_id: int              # deterministic tie-break key — see O1, O2
    movement_credit: float = 0.0
    target: int | None = None
    cached_path: list[tuple[int, int]] | None = None

    @property
    def is_alive(self) -> bool
    @property
    def speed(self) -> float
    @property
    def damage_per_tick(self) -> int
    @property
    def attack_range(self) -> int
    @property
    def walls_only(self) -> bool
    @property
    def dies_after_attack(self) -> bool
    def accumulate_movement(self) -> bool   # True if a tile step is now due
    def take_damage(self, amount: int) -> None
```

`entity_id` is not decoration. Every "nearest X" rule needs a deterministic tie-break or
the shuffle-invariance test at Stage 5 fails intermittently — the worst kind of failure.

#### Design decisions inside this stage

**Dataclass, not a plain class.** Free `__repr__` makes debug output readable, and free
`__eq__` makes test assertions short.

**Stats read from config by type, never copied onto the instance.** Copying means a stat
table change silently fails to propagate to entities constructed before it. Properties
that look up `config.TROOP_STATS[self.troop_type]` cost a dict lookup and remove that
whole failure mode. If profiling at Checkpoint A shows the lookups matter, cache them in
`__post_init__` — but measure first.

**`take_damage` clamps at zero.** Negative HP has no meaning and makes health-fraction
arithmetic in the observation go out of the `Box(0, 1)` range, which Gymnasium will
reject at Stage 7.

#### Acceptance

| # | Test | Proves |
|---|---|---|
| 1 | A troop with speed 0.10 advances exactly one tile after 10 accumulations, and its position is an integer tuple | No float drift (the whole point of the stage) |
| 2 | After 100 accumulations at 0.16, the troop has moved exactly 16 tiles | The accumulator subtracts 1.0 rather than resetting — a reset loses the remainder and slow troops arrive late |
| 3 | Damage below current HP leaves the entity alive; damage equal to or above it kills | Boundary at exactly `hp == damage` is death, not survival |
| 4 | `take_damage` never drives HP below zero | Observation stays inside `Box(0, 1)` |
| 5 | Every troop type and every building type constructs with the correct stats from config | Table wiring, all nine types |
| 6 | A wall reports `value == 0` and `is_wall is True` | The scoring rule is a property of the entity, not of the caller |

#### Pitfalls

- **Storing position as a float.** The bug this stage exists to prevent.
- **Resetting the accumulator to 0.0 instead of subtracting 1.0.** Test 2 catches it.
  A troop at 0.16 accumulates to 1.12, and resetting throws away 0.12 every tile — the
  troop arrives roughly 10% late, which is invisible until you wonder why melee troops
  underperform.
- **Copying stats onto instances.** See above.

#### After

- Gate: 18 tests pass, 79 total. (Spec said 6/49; written before D8.)
- Commit `Stage 2: entities.py — 18 tests, 79 passing`.
- Teach it. Fifteen minutes.

#### Viva questions this arms you for

- Why is troop position an integer and speed a float?
- What is a movement credit and what breaks without one?
- Why does every entity carry an id?

---

### Stage 3 — `env/generator.py` (structural half)

**Owner** A · **Blocked by** Stage 2

#### Why

Deterministic base construction. Same seed and motif must always produce a byte-identical
base, because the training/development/final seed split is the backbone of every claim in
the report. If generation is not reproducible, "held-out layout" means nothing.

Motifs are **templates drawn by a loop**, never random scatter. A ring drawn by a loop is
always a coherent ring; scattered walls produce incoherent bases and an incoherent
difficulty distribution.

#### Contract

```python
@dataclass(frozen=True)
class BaseSpec:
    seed: int
    motif: str
    buildings: tuple[tuple[str, int, int], ...]   # (type, row, col), sorted
    walls: tuple[tuple[int, int], ...]            # sorted

    def to_json(self) -> str
    @classmethod
    def from_json(cls, text: str) -> "BaseSpec"
    @property
    def total_value(self) -> int      # sum of non-wall values; MUST equal 155

def generate_base(seed: int, motif: str) -> BaseSpec
def wall_tiles(motif: str) -> set[tuple[int, int]]
def placement_region(motif: str) -> set[tuple[int, int]]
```

`BaseSpec` is frozen and its collections are sorted tuples. Frozen means a base cannot be
mutated mid-episode by accident; sorted means JSON round-trips are byte-identical, which
is what test 1 checks.

#### Motif construction

| Motif | Walls | Construction |
|---|---:|---|
| `closed_ring` | 36 | Boundary of `CENTRED_RING_BOX` (3,3,12,12) |
| `ns_gates` | 32 | Closed ring, minus 2 centred tiles from the north edge and 2 from the south |
| `ew_gates` | 32 | Closed ring, minus 2 centred tiles from the west edge and 2 from the east |
| `layered_core` | 35 | North and west edges of `LAYERED_OUTER_BOX` (19 tiles, sharing one corner) plus the full boundary of `LAYERED_CORE_BOX` (16 tiles) |

`placement_region` differs per motif:

- Ring motifs: the 8×8 interior, rows 4–11, cols 4–11 → **64 tiles**
- Layered core: the 3×3 core interior plus the outer region inside the L, minus the core
  footprint → **65 tiles**

Those two numbers are close on purpose. Building **density** is held constant so that only
topology varies between motifs. A motif that also changed how tightly ten buildings were
packed would confound the transfer result with a crowding effect. There is already a
config test asserting they stay within 8 tiles of each other.

#### Placement rules

1. Use `numpy.random.default_rng(seed)`. **Never the global `random` module** — it is
   process-global state and another import can reseed it.
2. For `layered_core`, place `LAYERED_CORE_CONTENTS` — Town Hall, one cannon, one
   storage — into the core interior first (D14). For every other motif, place the Town
   Hall into the single interior. This satisfies v3 §6.7's "innermost enclosed region"
   check and gives the core something to defend itself with.
3. Then the remaining buildings, sampled without replacement from the outer placement
   region: for `layered_core` that is 2 cannons, 2 mortars and 3 storages.
4. No building on a wall tile. No two buildings on one tile.
5. Sort everything before constructing the `BaseSpec`.

**Open question O3 was already closed on 12 August — see D14.** The paragraph below is
retained as the reasoning behind that decision, not as a live question: A mortar at the
core's centre cannot fire at anything inside the 3×3 core — maximum internal distance is
1.41 and its minimum range is 2. Placed in a core corner it just reaches the opposite
corner at 2.83. Decide deliberately and record it.

#### Acceptance

| # | Test | Proves |
|---|---|---|
| 1 | `generate_base(42, "closed_ring")` twice produces byte-identical JSON | Determinism — the foundation of the seed split |
| 2 | Different seeds produce different building positions | The seed actually does something |
| 3 | Exactly 10 non-wall buildings, with counts 1/3/2/4 | Firepower is constant across bases, so reward is not dominated by which base you drew |
| 4 | No building on a wall tile; no two buildings on one tile | Placement sampling is without replacement and excludes walls |
| 5 | All buildings within `placement_region(motif)` | The region rule is enforced, not merely documented |
| 6 | Wall counts are exactly 36 / 32 / 32 / 35 by motif | Templates are drawn correctly; gates actually remove tiles |
| 7 | Gate motifs contain gaps; the closed ring does not | The gate carve works |
| 8 | `layered_core` has two disconnected enclosed regions; ring motifs have one | The transfer motif is structurally what we claim |
| 9 | `spec.total_value == 155` for every motif and 20 sampled seeds | The objective scale never drifts |
| 10 | JSON round-trip returns an equal `BaseSpec` | Pools will reload identically |

#### Pitfalls

- **The global `random` module.** Reproducibility is a hard requirement and this is the
  single most common way to lose it.
- **Iterating a `set` and assuming order.** Python set iteration order depends on insertion
  and hash values. Sort before sampling, or the same seed gives different bases across runs.
- **Building the pools here.** Pools come at Stage 9, after the stat table is frozen.
  Difficulty is defined by the heuristic's score, so any stat change invalidates every pool.

#### After

- Gate: 24 tests, 103 total. (Spec said 10/59; written before D8.)
- Regenerate the motif figures: `python tools/preview_motifs.py`. They should now be
  produced from `generator.wall_tiles()` rather than the tool's own copy of the geometry —
  **delete the duplicate implementation in `tools/preview_motifs.py` and import instead.**
- Commit `Stage 3: generator.py structural — 24 tests, 103 passing`.
- Teach it.

#### Viva questions

- Why are wall layouts templates rather than random placement?
- Why must the same seed always produce the same base?
- Why is building density held constant across motifs?
- Which region does the Town Hall go in, and why?

---

### Stage 4 — `env/pathfinding.py`

**Owner** A · **Blocked by** Stage 3

#### Why

Four-neighbour A\*. Non-wall buildings are blocked cells. Walls are **passable at a cost**,
which is what makes "break through or go around" a decision rather than a rule.

The stage exists for one correction, and it is v3 pre-freeze item 3:

```python
open_tile_cost = 1.0 / troop.speed                                   # ticks per tile
wall_tile_cost = open_tile_cost + (wall.hp / troop.damage_per_tick)  # + ticks to break
```

**Both terms are in ticks.** The naive formulation costs an open tile at 1 and a wall at
`hp / damage`, which compares one grid step against several dozen wall-breaking ticks as
though they were the same unit. Troops then detour far too eagerly and breakers quietly
become worthless.

Concretely: a melee troop at speed 0.16 takes 6.25 ticks per tile. A 10-tile detour costs
62.5 ticks, not 10.

#### Contract

```python
class PathCache:
    def __init__(self) -> None
    def find_path(
        self,
        troop_type: str,
        start: tuple[int, int],
        goal: tuple[int, int],
        grid: "GridState",
    ) -> list[tuple[int, int]] | None
    def invalidate(self) -> None      # called when any wall dies
    @property
    def hit_rate(self) -> float       # for the Checkpoint A profile

def open_tile_cost(troop_type: str) -> float
def wall_tile_cost(troop_type: str, wall_hp: int) -> float
```

Cache key: `(troop_type, start, goal, wall_state_version)`. `wall_state_version` is an
integer bumped every time a wall dies. Bumping a counter is cheaper and less error-prone
than working out which paths a particular wall affected.

#### Design decisions inside this stage

**Why four neighbours and not eight.** Eight-neighbour movement makes diagonal moves cost
√2, which interacts awkwardly with the tick-unit costs, and it lets troops slip diagonally
between two wall corners — a rule nobody wants to define or defend. Four neighbours keeps
movement a clean lattice.

**Why walls are passable rather than impassable.** Because A\* treats them as expensive
rather than blocked, **there is no such thing as a structurally unattackable base.** A
perfectly sealed ring is always solvable, just costly. That removes an entire class of
generator validation.

**Heuristic:** Manhattan distance × `open_tile_cost`. This is admissible (never
overestimates) because no tile costs less than `open_tile_cost` and movement is
4-neighbour. Admissibility is what makes A\* optimal — worth being able to state.

#### Acceptance

| # | Test | Proves |
|---|---|---|
| 1 | A path around a wall is chosen when the detour costs fewer ticks than breaking through | The comparison happens in one unit |
| 2 | A path through a wall is chosen when the reverse holds | Both branches work |
| 3 | Changing **only** troop speed changes which route is chosen | **The unit test.** With mismatched units, speed cannot affect the trade-off, so this fails |
| 4 | Break-even detour matches the closed form: 3.2 tiles melee, 3.5 ranged, 3.8 tank, 0.2 breaker | D9, verified rather than asserted |
| 5 | No path enters a non-wall building tile | Buildings block |
| 6 | Cached lookup returns a result identical to an uncached recompute | The cache is not lying |
| 7 | Killing a wall invalidates the cache and the next path differs | Invalidation is wired to wall death, not to time |
| 8 | A fully sealed ring still yields a path | Walls are passable, so no base is unattackable |

#### Pitfalls

- **The unit bug.** Test 3 is the one that catches it. Write test 3 first.
- **Not invalidating on wall death.** Troops keep pathing through a hole that has been
  filled, or ignore one that has opened. Symptom: troops walking into walls.
- **Recomputing every tick.** A\* for 25 troops × 300 ticks with no cache is far too slow
  to train on. This is the main thing Checkpoint A measures.

#### After

- Gate: 18 tests, 121 total. (Spec said 8/67; written before D8.)
- Record the measured cache hit rate in the commit message. You will want it at Checkpoint A.
- Commit `Stage 4: pathfinding.py — 18 tests, 121 passing, cache hit rate 69.5%`.
- Teach it.

#### Viva questions

- Why must path costs be in consistent time units? What breaks otherwise?
- Why are walls expensive rather than impassable?
- What is the A\* heuristic and why is it admissible?
- How does the cache know when to invalidate?

---

### Stage 5 — `env/combat.py` and `env/render.py`

**Owner** A · **Blocked by** Stage 4

The largest stage. Two modules because the renderer is how you debug the tick loop.

#### Why — the seven-step tick order

Every simulator tick does exactly this, in this order, without exception (v3 §6.8):

```
1. remove dead troops and buildings
2. invalidate cached paths if a wall was destroyed
3. retarget troops whose target is gone
4. troops move, or add attacks to a pending-damage buffer
5. defences add attacks to the same buffer
6. apply all pending damage together
7. check whether the episode has ended
```

**Step 6 is the one to understand.** Applying damage immediately as you loop through units
means whichever unit sits earliest in a Python list attacks before it can be killed.
Results would then depend on list ordering, which is meaningless — and worse, it makes
episodes non-reproducible across Python versions where dict and set ordering differ.
Buffering makes the simulation order-independent.

Expect to be asked to recite this in the viva. It is question 10 on the v3 checklist.

#### Contract — `combat.py`

```python
@dataclass
class BattleState:
    buildings: list[Building]
    troops: list[Troop]                # deployed and alive
    reserve: dict[str, int]            # bought but not yet deployed
    tick: int
    wall_state_version: int
    path_cache: PathCache
    destroyed_value: int
    milestones: dict[str, bool]        # half, town_hall, full_clear
    finish_reason: str | None

def step_tick(state: BattleState) -> None
def record_snapshot(state: BattleState) -> dict
def deploy_troop(state: BattleState, troop_type: str, tile: tuple[int, int]) -> None
def in_range(attacker, target, ) -> bool     # Euclidean, honours min_range
def splash_targets(centre, radius, troops) -> list[Troop]
```

`step_tick` mutates in place and returns `None`. Returning a new state each tick would
allocate 300 times per episode for no benefit.

#### Targeting rules

| Attacker | Targets | Notes |
|---|---|---|
| Tank | Nearest **defence** | Cannon or mortar only. Walks past storages |
| Melee | Nearest **building** | Any non-wall building |
| Ranged | Nearest **building** | Range 4 — the only troop that can hit across a wall |
| Breaker | Nearest **wall** | Dies immediately after striking. 350 damage > 300 wall HP |
| Cannon | Nearest **troop** in range 3 | See O1 |
| Mortar | Nearest **troop** in range 2–6, splash 1.5 | Cannot fire at anything closer than 2 |

**Close O1 and O2 here.** All "nearest" rules break ties by ascending `entity_id`. Any
other tie-break — list order, arbitrary `min()` on unsorted input — reintroduces exactly
the ordering dependence that step 6 exists to remove, and test 2 will catch it
intermittently rather than reliably, which is worse than failing every time.

#### Range and line of sight

Range is **Euclidean** (D5). Nothing blocks a shot (D7). A defence fires if
`min_range <= euclidean_distance <= range`.

The blind spot is real: with `min_range = 2` there are **8 tiles** a mortar cannot touch —
its four orthogonal and four diagonal neighbours. Four of those are at distance 1, which
is melee and tank range, so a troop that reaches the blind spot can kill the mortar while
it is helpless.

Crossing the mortar's danger band costs roughly:

| Troop | Ticks to cross the 2.5-tile band | Mortar damage taken | HP | Survives |
|---|---:|---:|---:|---|
| Breaker | 11.4 | 102 | 60 | no |
| Melee | 15.6 | 141 | 120 | no |
| Ranged | 17.9 | 161 | 90 | no |
| Tank | 25.0 | 225 | 900 | **yes** |

Recomputed 24 August 2026 for mortar range 4.5 and a 5-tick cooldown. The band is
narrower than the original range-6 table assumed, so every figure fell — but the
conclusion is unchanged, and by a wider margin than before.

Only the tank can reach the blind spot. That is why it costs 25 and moves at 0.10 — its
900 HP is a delivery mechanism, not a damage source.

#### Contract — `render.py`

```python
def draw_snapshot(
    snapshot: dict,
    ax: "matplotlib.axes.Axes",
    show_ranges: bool = False,
    show_paths: bool = False,
    show_damage: bool = False,
) -> None
```

Pure matplotlib. **No Streamlit import** — Stage 14 imports this, not the other way round.

**Tier 1, always drawn:**

- Coloured squares per entity, one `Rectangle` each. No sprites, no art assets, no
  question about using anyone's imagery.
- Health bars: two rectangles, grey background and coloured foreground scaled by
  `hp / max_hp`. Green above 60%, amber above 25%, red below.
- Attack lines: every troop currently dealing damage draws a line to its target; every
  defence firing draws a line to the troop it hits. Two colours, one per direction.
- HUD text: tick, destruction percentage, resources spent, troops alive, phase.

**Tier 2, behind the three flags:**

- `show_ranges` — a `Circle` per defence at its attack radius, plus a second inner circle
  at the mortar's minimum range. This is the picture that demonstrates the design's most
  important mechanic without narration.
- `show_paths` — each troop's cached A\* path as a dashed line. Your pathfinding debugger.
- `show_damage` — damage dealt this tick annotated at the target tile. One mortar shot
  producing four numbers at once confirms splash works.

Attack lines are the highest-value debugging feature and it is not close. Targeting rules
are the part most likely to be subtly wrong and they are completely invisible in numbers.
A tank drawing a line to a storage instead of a cannon tells you the rule is broken,
instantly. So does a line longer than the weapon's range.

#### Acceptance — `combat.py`

| # | Test | Proves |
|---|---|---|
| 1 | Two entities that would kill each other on the same tick both die | **Damage is buffered.** The single most valuable test in the suite |
| 2 | Shuffling `state.troops` does not change episode outcome for a fixed seed | Order independence, end to end. Run 20 shuffles |
| 3 | A mortar does not fire at a troop at distance 1 or 1.41 | The blind spot exists |
| 4 | A mortar does fire at distance exactly 2 | The boundary is inclusive, not off by one |
| 5 | A mortar's splash damages every troop within 1.5 of its target, and none beyond | Splash uses Euclidean radius, so it hits the 3×3 block |
| 6 | A tank targets a defence when both a defence and a storage are in range | Tank targeting is not "nearest building" |
| 6b | A tank with every defence destroyed attacks the nearest other building; a breaker with no walls left waits | D15 — the empty-target-pool rule, per troop |
| 7 | A breaker dies immediately after striking a wall, and the wall dies too | 350 > 300, one strike one wall |
| 8 | A ranged troop attacks from distance 4; a cannon only from 3 | The standoff annulus |
| 9 | A ranged troop hits a building across a wall; a melee troop at the same position cannot | D7 — the only troop asymmetry that matters |
| 10 | Ties in every "nearest" rule resolve by ascending `entity_id` | O1, O2 — deterministic and testable |
| 11 | A troop with no reachable target waits rather than raising | No crash on a degenerate state |

#### Acceptance — `render.py`

| # | Test | Proves |
|---|---|---|
| 1 | `draw_snapshot` on a mid-battle snapshot produces a figure with no exception | It runs headless (`matplotlib.use("Agg")`) |
| 2 | Every entity in the snapshot appears as a patch on the axes | Nothing is silently dropped |
| 3 | `show_ranges=True` adds exactly `2 × n_mortars + n_cannons` circles | Range and blind-spot circles are both drawn |

#### Pitfalls

- **Applying damage inside the loop instead of buffering.** Test 1 and test 2 catch it.
- **`min_range` off by one.** Decide whether the boundary is `d >= min_range` or
  `d > min_range` and test both sides. Tests 3 and 4 pin it.
- **Splash using Manhattan distance.** Radius 1.5 then hits only the 4 orthogonal
  neighbours instead of the 3×3 block, halving mortar effectiveness.
- **Non-deterministic tie-breaks.** Test 2 fails one run in twenty and you spend a day
  chasing it.
- **Building the renderer against live objects rather than snapshots.** Stage 14 replays
  from recorded snapshots and must not re-simulate. Take the snapshot dict as the only input.

#### After

- Gate: 26 tests, 147 total. (Spec said 14/81; written before D8.)
- **Watch one episode.** Run a scratch script that steps 300 ticks and renders every 10th
  into a grid of subplots. Look for: troops stuck against walls, attack lines longer than
  the weapon's range, tanks walking past defences, a mortar firing into its own blind spot.
  This is the payoff for pulling the renderer forward — spend twenty minutes on it.
- Save one frame to `figures/debug_frame_example.png`. It goes in the report as evidence
  the simulator was verified visually, not only by assertion.
- Commit `Stage 5: combat.py — 26 tests, 147 passing`.
- Teach it. Use the seven-step order as the agenda — every member must be able to recite it.

#### Viva questions

- Recite the seven-step tick order. Why is damage buffered?
- What happens if two troops would kill each other on the same tick?
- Why does the mortar have a minimum range? What does it cost the defender?
- Why can a ranged troop hit across a wall when a melee troop cannot?
- How are ties broken in targeting, and why does it matter?

---

### Stage 6 — CHECKPOINT A

**Owner** A, reviewed by B and C · **Blocked by** Stage 5

A hard stop. Three gates.

#### Gate 1 — correctness

Write `agents/random_agent.py` and a scratch driver that runs full episodes with random
legal actions, calling the simulator directly. **No Gymnasium wrapper yet** — that is
Stage 7, and you want to know the simulator works before wrapping it.

100 random episodes must complete with:

- no exception
- no infinite loop
- destruction between 0% and 100% inclusive
- every episode ending with a recorded `finish_reason`

#### Gate 2 — speed

`tests/profile_env.py` measures milliseconds per episode.

| Result | Action |
|---|---|
| under 50 ms | Proceed |
| 50–200 ms | Proceed, but plan to optimise before the protocol runs |
| over 200 ms | **Stop and optimise now** |

Optimise in this order:

1. Cache paths harder — check the hit rate from Stage 4
2. Replace Python loops over troops with NumPy array operations
3. Reduce max troops from 25 to 15
4. Reduce battle ticks from 300 to 200

Why this matters, in numbers: 250,000 timesteps is roughly 5,000 episodes per learner, and
the protocol needs 10 to 20 learners. At 1 second per episode that is 14 to 28 hours of
pure simulation. At 50 ms it is under an hour. **This single number decides whether you
can afford 10 learner seeds, which decides your statistical power** (O4, v3 §10.2).

#### Gate 3 — the balance investigation

This is v3 pre-freeze item 10 and it is new to this build order.

Instrument one hundred heuristic episodes and log, per troop:

- ticks spent walking vs ticks spent attacking
- ticks spent alive
- damage dealt
- what killed it

**Prediction to test (D13):** the binding constraint is time-under-fire during the
approach, not damage capacity. §6.5's arithmetic divides pooled army HP by pooled defence
DPS and concludes the attacker has a huge margin. That model is wrong because defences
focus-fire — five defences are five independent kill-slots, capable of roughly 85
fragile-troop kills over 300 ticks against an army of at most 25.

If the instrumentation shows troops spend most of the clock walking, the prediction holds
and you have a genuinely good paragraph for the report plus a real answer to "why does the
heuristic cap near 29%".

If it shows something else, that is more interesting still. Either way, **record the
result** — it is the difference between a report that explains its own numbers and one
that presents them.

Also log, per base: how many of the 60 perimeter tiles sit under defence fire. Two mortars
cover at most 19 (D11), so this varies mainly with placement. It is likely to be the
single strongest predictor of base difficulty, and it makes a cheap, good figure.

#### After

- Tag: `git tag checkpoint-a`
- Record milliseconds per episode in `README.md`. **This number gates O4.**
- Write the balance-investigation result into a short note in `runs/` and cite it in the
  report.
- **Unblock C.** Stages 8, 9, 13, 14 can now start.
- All three members run the profile themselves. v3 §13.3 requires a member who did not
  build a module to reproduce it.

---

### Stage 7 — `env/siege_env.py`

**Owner** B · **Blocked by** Stage 6

#### Why

The Gymnasium wrapper. This is where the simulator becomes an RL problem.

#### Contract

```python
class SiegeEnv(gymnasium.Env):
    observation_space = Dict({
        "grid":   Box(0.0, 1.0, (8, 16, 16), dtype=np.float32),
        "vector": Box(0.0, 1.0, (9,),        dtype=np.float32),
    })
    action_space = Discrete(246)

    def reset(self, seed=None, options=None) -> tuple[dict, dict]
    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]
    def action_masks(self) -> np.ndarray          # METHOD ON THE CLASS
    def render(self) -> None
```

**`action_masks()` is a method on the environment class, not the `ActionMasker` wrapper.**
`ActionMasker` cannot be used with `SubprocVecEnv`, and parallel environments are how
training time stays manageable. This is non-negotiable and it is the single most common
way this project could waste a week.

#### Observation

Eight 16×16 layers, every value normalised into [0, 1]:

| Layer | Contents | Normaliser |
|---:|---|---|
| 0 | Town Hall locations | binary |
| 1 | Cannon locations | binary |
| 2 | Mortar locations | binary |
| 3 | Storage locations | binary |
| 4 | Wall locations | binary |
| 5 | Building health fraction | `hp / max_hp` |
| 6 | Defence coverage | `count / MAX_DEFENCE_COVERAGE` (5) |
| 7 | Friendly troop density | `count / MAX_TROOP_DENSITY_PER_TILE`, clipped |

Nine summary values: four remaining troop counts (÷25), remaining budget (÷200), remaining
battle time (÷300), phase (0 or 1), destroyed-value fraction, live troop count (÷25).

**Layer 6 is a design decision worth naming in the report.** We pre-compute "how dangerous
is this tile" rather than hoping the network infers it from cannon and mortar positions.
Given D11 — two mortars cover at most 19 of 60 perimeter tiles, so finding the safe arc is
the opening decision — layer 6 is arguably the single most important input the policy gets.

**The observation is partial, not Markov** (v3 §7.3, pre-freeze item 6). Layer 7 gives
troop *density* only: no per-troop health, type, target, movement credit, or cached path.
Two full-health melee troops on a tile and two nearly-dead tanks look identical to the
policy and have completely different futures.

Do not confuse two separate claims:

| Claim | True? |
|---|---|
| The simulator has no hidden information | Yes |
| The observation is sufficient to act optimally | **No** |

This is a deliberate compactness trade-off, not an oversight. The alternatives are
per-troop-type health layers (more channels, slower, more parameters) or a recurrent policy
(out of scope). State the trade-off and be ready to defend it — it is viva question 3.

#### Action space

```
  0 – 3     buy tank / melee / ranged / breaker
  4         finish shopping
  5 – 244   deploy: 5 + (troop_index × 60) + tile_index
  245       wait
```

#### Masking rules

- Unaffordable purchases masked
- Purchases masked once army size reaches 25
- All shopping actions masked during battle
- All deployment actions masked during shopping
- A troop type with zero in reserve cannot be deployed
- Wait is always legal during battle
- Finish-shopping is always legal during shopping

The mask removes illegal actions only. It does **not** indicate which legal action is good.
That distinction is viva question 5.

#### Reward

```
  incremental non-wall building value destroyed / total_base_value
+ 0.25 when destruction first reaches 50%
+ 0.25 when the Town Hall is destroyed
+ 0.25 when all non-wall buildings are destroyed
− 0.30 × (resources spent / 200)   applied once, at the end of shopping
```

`total_base_value` is **computed at reset from live non-wall buildings**, never hardcoded
as 155:

```python
self.total_base_value = sum(b.value for b in self.buildings if not b.is_wall)
```

The two agree for a valid base. Computing it means a base with a different composition
cannot silently mis-scale the reward — and it makes "full clear equals exactly 1.0" true
by construction rather than by coincidence.

`info` must carry the components separately: `destruction`, `milestone`, `cost_penalty`,
`wall_shaping`. A flat total with rising destruction and rising penalty is a completely
different situation from a flat total with nothing happening.

#### Termination and truncation

**Terminates** when all non-wall buildings are destroyed, or the army is exhausted
(none deployed alive and none in reserve), or 300 ticks are reached.

**Truncates** only at the 70-decision safety guard.

The 300-tick deadline is visible to the policy through the observation, so it is part of
the task and must terminate. Truncating would make the value function bootstrap past the
game's end, estimating reward that cannot exist. That is viva question 9.

The 70-decision guard exists only to catch an unexpected control-loop fault and should
never fire in normal operation — worst legitimate case is 25 purchases + 1 finish +
30 battle decisions = 56. Log it if it fires.

#### Acceptance

| # | Test | Proves |
|---|---|---|
| 1 | Mask is False for unaffordable purchases | Budget rule |
| 2 | Mask is False for shopping actions during battle and deployment during shopping | Phase separation |
| 3 | Mask is False for troop types with zero in reserve | Inventory rule |
| 4 | Wait is always legal during battle; finish-shopping always legal during shopping | No dead ends |
| 5 | The agent can never execute a masked action | Fuzz 10,000 random masked actions |
| 6 | Mask is False for purchases once army size reaches 25 | Army cap |
| 7 | Destroying every non-wall building yields destruction reward summing to exactly 1.0 | Normalisation by live value |
| 8 | Walls contribute exactly 0 to reward | No wall farming |
| 9 | Reward components in `info` sum to the returned reward | Logging is trustworthy |
| 10 | Cost penalty is applied once, at the end of shopping, not per purchase | Timing |
| 11 | Reaching 300 ticks sets `terminated=True, truncated=False` | Termination semantics |
| 12 | Hitting the 70-decision guard sets `truncated=True` | Truncation semantics |
| 13 | Army exhaustion terminates | Third ending |
| 14 | `reset(seed=n)` twice produces identical observations | Reproducibility survives the wrapper |
| 15 | Observation values all lie in [0, 1] and match the declared dtype | Gymnasium will reject otherwise |

#### Pitfalls

- **Using `ActionMasker`.** It cannot be used with `SubprocVecEnv`. The mask must be a
  method on the class.
- **`check_env` false alarms.** It feeds random actions ignoring the mask, so it will
  report failures that are not real. Read its output; do not obey it blindly.
- **Hardcoding 155 as the divisor.** Compute it from live buildings.
- **Truncating on the tick deadline.** Silently corrupts value learning; nothing crashes.
- **Applying the cost penalty per purchase.** Interacts with `gamma` and reintroduces
  exactly the artefact `gamma = 1.0` exists to remove.

#### After

- Gate: 15 tests, 96 total.
- Run `check_env` and record which warnings are spurious and why. That note belongs in the
  report's verification section.
- Commit `Stage 7: siege_env.py — 15 tests, 96 passing`.
- Teach it. This is the module a marker is most likely to ask any member about.

#### Viva questions

- What are the 246 actions and how is a deployment action encoded?
- Why are illegal actions masked rather than penalised?
- Why is the observation partial rather than Markov?
- Why does the 300-tick deadline terminate rather than truncate?
- Why is the reward divisor computed rather than hardcoded?

---

### Stage 8 — `agents/heuristic.py`

**Owner** C · **Blocked by** Stage 7

#### Why

The strongest non-learning reference. PPO is **not assumed to beat it** — v3 hypothesis 4
says explicitly that if the heuristic stays ahead, that is a valid result reported honestly.

The critical requirement: the heuristic must run through **the same `siege_env.step()`
path as every other policy**, so returns, reward components and finish reasons are
directly comparable. A heuristic that calls the simulator directly produces numbers that
cannot be placed in the same table as PPO's.

#### Contract

```python
class HeuristicCommander:
    def __init__(self, side: str | None = None) -> None
    def act(self, obs: dict, mask: np.ndarray) -> int
    def reset(self) -> None

def evaluate_from_all_sides(base: BaseSpec) -> float   # best of four; used by Stage 9
```

#### Policy

1. **Army:** fixed. 3 tanks, 6 melee, 4 ranged, 3 breakers = **199 resources**, 16 troops.
2. **Side selection:** pick the perimeter arc with the lowest summed defence coverage
   (observation layer 6). Given D11 — two mortars cover at most 19 of 60 tiles — this
   choice is most of the heuristic's strength.
3. **Deployment order:** breakers, then tanks, then ranged, then melee.

The order is not arbitrary. Breakers first because they open the route everything else
uses, and their 0.22 speed means they arrive first anyway. Tanks second because they are
slowest (0.10) and need the head start, and because they draw fire while fragile troops
close. Ranged before melee because ranged can begin working from range 4 while melee is
still walking.

#### Acceptance

| # | Test | Proves |
|---|---|---|
| 1 | Fully deterministic for a fixed base and side | Reproducible baseline |
| 2 | Spends exactly 199 resources and buys exactly 16 troops | Army matches config |
| 3 | Beats the random agent by a clear margin over 100 bases | It is actually a strong baseline |
| 4 | Produces the same reward accounting as a PPO policy on an identical trajectory | Same code path |
| 5 | `evaluate_from_all_sides` returns the maximum of four runs, and is deterministic | Stage 9 depends on it |
| 6 | Never selects a masked action | Same guarantee as PPO |

#### After

- Gate: 6 tests, 102 total.
- Record the heuristic's mean destruction on 50 development bases. **This is the number
  every performance gate in the protocol is calibrated against** — v3 pre-freeze item 2
  requires all three baselines regenerated on the 155-point scale.
- Commit, teach.

---

### Stage 9 — generator validation and base pools

**Owner** C · **Blocked by** Stage 8

#### Why

v3 §6.7. Layouts currently vary by seed with no check on whether the result is reasonable.
Some seeds produce near-trivial bases and some near-impossible ones, and that variance
lands directly in the primary outcome — which, given the power concern in §10.2, the study
cannot afford.

#### Contract

```python
def validate_structure(spec: BaseSpec) -> tuple[bool, str]
def validate_base(spec: BaseSpec) -> tuple[bool, float]
def build_pool(seed_range, motifs, out_path) -> dict   # returns statistics
def load_pool(path: str) -> list[BaseSpec]             # RAISES on the final pool
```

#### Stage 1 — structural checks (milliseconds, no simulation)

1. No building on a wall tile
2. All buildings inside `placement_region(motif)`
3. Town Hall inside the innermost enclosed region
4. Every enclosed compartment contains at least one building
5. At least two defences whose range covers the Town Hall

Checks 3 and 4 were **vacuous under the old motif set** — one region, all ten buildings
inside it. D6 restores their meaning: `layered_core` has two nested regions, so the Town
Hall must be in the core and the core must not be empty.

Check 5 needs a wording fix in v3 §6.7 to say "range" and nothing about line of sight,
now that D7 settles that nothing blocks (see §9).

#### Stage 2 — heuristic difficulty gate (~30 ms)

Run `evaluate_from_all_sides` and take the best score.

| Best heuristic score | Verdict |
|---|---|
| Above 85% | Too easy — no headroom to show improvement. Discard |
| 25% to 85% | Keep |
| Below 25% | Too hard — reward too sparse to learn from. Discard |

Apply identical filtering to all three pools using their own seed ranges. **Do not
validate inside `reset()`** — that would pay 30 ms on every episode and destroy the
Checkpoint A speed budget.

#### Pools

| Pool | Seeds | Motifs | File |
|---|---|---|---|
| Train | 0 – 9,999 | closed_ring, ns_gates, ew_gates | `bases/pool_train.json` |
| Development | 10,200 – 10,399 | all four | `bases/pool_dev.json` |
| **Final** | 11,000 – 11,999 | all four | `bases/pool_final.json` |

**The final pool is generated and then not opened until freeze.** `load_pool` must raise
on any path resolving to the final pool unless an explicit override is passed. Put the
guard in `load_pool` itself, not in each caller — one guard that cannot be forgotten beats
five that can.

#### Acceptance

| # | Test | Proves |
|---|---|---|
| 1 | A base with a building on a wall tile is rejected | Structural check 1 |
| 2 | A `layered_core` base with an empty core is rejected | Structural check 4, now non-vacuous |
| 3 | A base with the Town Hall outside the core is rejected for `layered_core` | Structural check 3 |
| 4 | Every accepted base scores between 25% and 85% for the heuristic | The gate works |
| 5 | Pool files reload to identical `BaseSpec` objects | JSON round-trip at scale |
| 6 | `load_pool` on the final pool raises | Contamination guard |
| 7 | `load_pool` on the final pool with the explicit override succeeds | The escape hatch exists and is deliberate |
| 8 | Acceptance rate is recorded in the pool statistics | Evidence |

#### Pitfalls

- **Building the pool before the stat table is frozen.** Difficulty is *defined by* the
  heuristic's score, so any troop or building stat change invalidates every pool. Freeze
  the table at the end of Stage 8 and record the freeze in the decision register.
- **Forgetting that D7 changed difficulty.** Nothing blocks shots, so ranged troops can
  strip the interior edge from the perimeter. Acceptance rates may be lower than expected
  because more bases score above 85%. If so, the lever is defence damage, not the LOS rule.

#### After

- Gate: 8 tests, 110 total.
- **Freeze the stat table.** Add a decision-register row with the date.
- Plot the difficulty histogram of the accepted training pool → `figures/difficulty_hist.png`.
  That figure demonstrates you thought about the training distribution and it is cheap.
- Record the acceptance rate per motif. A motif accepting far fewer bases than the others
  is telling you something about balance.
- Commit `Stage 9: base validation and pools — 8 tests, 110 passing`.

---

### Stage 10 — `train/train_ppo.py`

**Owner** B · **Blocked by** Stage 9

#### Contract

```python
def make_env(pool_path, motifs, seed, symmetry=False) -> Callable[[], gym.Env]
def build_model(venv, seed, tb_log_dir) -> MaskablePPO
def train(condition: str, learner_seed: int, out_dir: Path) -> Path
def write_manifest(out_dir: Path, **fields) -> None
```

#### Hyperparameters (v3 §8.1 — fixed, do not tune after freeze)

| Parameter | Value |
|---|---:|
| Learning rate | 0.0003 |
| Rollout steps | 2048 |
| Batch size | 256 |
| PPO epochs | 10 |
| GAE lambda | 0.95 |
| Entropy coefficient | 0.01 |
| **Gamma** | **1.0** |

**`gamma = 1.0`, not 0.99.** SB3 discounts per *timestep*, and shopping decisions do not
advance game time. At 0.99, buying 20 troops discounts all subsequent battle reward by
0.99²⁰ ≈ 0.82 while buying 8 discounts by 0.99⁸ ≈ 0.92 — a larger army is penalised purely
for taking more decisions to assemble, an artefact stacked on top of the intended cost
penalty. Episodes are finite and bounded, so 1.0 is well defined. Viva question 8.

#### Architecture

Two-layer CNN over the eight spatial layers, small MLP over the nine summary values,
256-dimensional fused representation, separate 128-unit policy and value heads.

#### Curriculum (v3 §8.2)

1. **Placement stage** — 25,000 timesteps with the fixed 199-resource heuristic army
2. **Joint stage** — 225,000 timesteps with learned shopping and deployment

Shopping decisions sit furthest from their consequences, so they are the most prone to
premature convergence: if the agent stumbles onto an adequate army early it stops sampling
alternatives. Training placement first against a fixed army means the deployment policy is
already competent before composition is unlocked.

#### The run manifest — required fields

Git revision; dirty-worktree status; Python and package versions; complete config dump;
learner seed; base pool paths and their checksums; requested and actual timesteps; start
time, end time, wall-clock duration; a checksum over every output file.

v3 §13.2 requires this for every final learner, and §13.3 requires a member who did not
run it to reproduce one. A manifest missing the dirty flag makes that impossible.

#### Acceptance

| # | Test | Proves |
|---|---|---|
| 1 | A 5,000-step run completes and saves a checkpoint | End to end |
| 2 | The checkpoint reloads and predicts with masks applied | Save/load round-trip |
| 3 | `SubprocVecEnv` with 4 workers runs without mask errors | D-day for the `action_masks()` method decision |
| 4 | The manifest contains every required field | Reproducibility |
| 5 | `MaskableEvalCallback` is used, not the base `EvalCallback` | The standard one silently ignores masks |
| 6 | Reward components appear as separate TensorBoard scalars | Diagnosability |
| 7 | Army composition is logged per rollout | The monitor for O5 |

#### Pitfalls

- **Using SB3's standard `evaluate_policy` or `EvalCallback`.** The maskable versions are
  required. The standard ones do not error — they silently ignore the mask, so evaluation
  measures a different policy from the one you trained. This is the quietest and most
  damaging bug available in this project.
- **Not logging composition.** O5 is monitored here or not at all.

#### After

- Gate: 7 tests, 117 total.
- **Time one full 250,000-step run, end to end, and write the number down.** This closes
  O4: under ~90 minutes → 10 learner seeds per condition; longer → 5 seeds and state the
  detection limit explicitly in the report.
- **Record the seed-count decision and its justification before any final training.**
  Choosing the seed count after seeing results is a protocol violation.
- Commit, teach.

---

### Stage 11 — CHECKPOINT B

**Owner** B, reviewed by A and C · **Blocked by** Stage 10

A hard stop. Run 50,000 timesteps on the training pool, evaluating on dev.

#### Gate

`rollout/ep_rew_mean` must be **visibly rising**. Not good. Rising.

#### If it is flat

Work in this order:

1. **Check the mask is applied.** Log the fraction of legal actions per step. If it is
   1.0, the mask is not reaching the policy.
2. **Check reward scale.** Episode returns should sit roughly between −1 and +2. Numbers
   in the hundreds mean the divisor is wrong.
3. **Check the components separately.** Rising destruction with a rising penalty looks
   identical to nothing happening if you only watch the total.
4. **Simplify until something learns.** One motif, two troop types, no walls. Then add
   complexity back one piece at a time.

**Do not proceed to symmetry work until a plain learner shows a rising curve.** A broken
transform on top of a broken learner produces a comparison that measures nothing, twice.

#### Also check here

- **Army composition.** If the ranged share exceeds ~70%, O5 has fired: perimeter sniping
  is dominating and army selection has collapsed to one option. The response is *not* to
  change the LOS rule; it is to shrink the building placement region to the inner 6×6 so
  the interior edge sits 5 tiles from the perimeter, out of ranged reach. That preserves
  the standoff against cannons, which is about distance to a cannon rather than to the ring.
- **Breaker share.** If breakers vanish entirely, v3 §6.4's contingency applies: a small
  wall-destruction shaping bonus (~0.005 each), **excluded from the reported score**, with
  a written explanation. `config.WALL_SHAPING_BONUS` exists for this and defaults to 0.0.

#### After

- Tag `checkpoint-b`.
- Save the curve to `figures/checkpoint_b_curve.png`.
- Record composition statistics and the O5 decision.

---

### Stage 12 — `env/symmetry.py`

**Owner** B · **Blocked by** Stage 11

#### Why

**The variable under test.** Everything else in the project exists to make this measurable.

At reset, sample one of the eight dihedral transforms of the square. Transform the
observation grid, and map the selected deployment action back to simulator coordinates
before executing it. Shopping, finish-shopping and wait pass through unchanged.

#### Contract

```python
TRANSFORMS = ("identity", "rot90", "rot180", "rot270",
              "flip_h", "flip_v", "transpose", "anti_transpose")

def transform_grid(grid: np.ndarray, name: str) -> np.ndarray
def transform_tile(tile: tuple[int, int], name: str) -> tuple[int, int]
def inverse_tile(tile: tuple[int, int], name: str) -> tuple[int, int]
def transform_action(action: int, name: str) -> int
def inverse_action(action: int, name: str) -> int

class SymmetryWrapper(gymnasium.Wrapper):
    def reset(self, seed=None, options=None)
    def step(self, action)
    def action_masks(self) -> np.ndarray     # must transform consistently
```

D1 pays off here: with tiles ordered clockwise from (0,0), `rot90` is a cyclic shift of
+15 in tile index. **Reflections are not cyclic shifts** — they are index reversals about
an axis — which is exactly why the bijection test below is mandatory rather than optional.

#### Acceptance — all six must pass before any PA training run

| # | Property | Test |
|---|---|---|
| 1 | Every transform has a correct inverse | `inverse(transform(x)) == x` for all eight, on 100 random grids |
| 2 | Every perimeter tile maps to exactly one valid perimeter tile, and the mapping is a bijection over all 60 | Build the full map for each of the eight and assert it is a permutation |
| 3 | Action masks transform consistently with observations | Mask a random legal set, transform both, assert they agree |
| 4 | A transformed action produces the transformed equivalent transition | Step both, compare resulting states |
| 5 | Reward and termination are invariant under every transform | 50 episodes, 8 transforms, identical returns |
| 6 | Reset is deterministic for a fixed (seed, transform) pair | Reproducibility survives the wrapper |

**The governing property, worth its own test:** for a fixed physical state and action
sequence, transforming the view and mapping actions back must leave the physical
trajectory unchanged.

#### Pitfalls

- **Perimeter tile indexing.** Write the bijection test (property 2) *before* writing the
  mapping. It is far easier to make the mapping satisfy a test than to debug a mapping you
  already believe in.
- **Transforming the observation but not the mask.** The policy then chooses a legal-looking
  action that is illegal in real coordinates.
- **A broken transform produces a comparison that measures nothing, and you will not notice
  from the training curve.** Both conditions will train happily. This is the highest-risk
  silent failure in the project.

#### After

- Gate: 6 tests, 123 total. **All six pass, or PA does not train.**
- Commit `Stage 12: symmetry.py — 6 tests, all invariance properties verified`.
- Teach it. Every member must be able to explain the action mapping — viva question 13.

---

### Stage 13 — `train/evaluate.py`

**Owner** C · **Blocked by** Stage 12

#### Contract

```python
def evaluate_policy_on_pool(policy, pool, views, out_csv) -> pd.DataFrame
def paired_difference(csv_a, csv_b) -> dict
def bootstrap_within_learner(df, n=10_000) -> tuple[float, float]
```

Per-unit CSV rows keyed by `(learner_seed, base_seed, motif, view)`, plus aggregate JSON.

#### Statistics (v3 §10.8) — the order matters

1. For each learner seed, **first average over its registered final units**
2. Compute PA − P for each **matched learner seed pair**
3. Report the mean paired difference with a Student-t 95% interval **across learner seeds**
4. Paired bootstrap intervals *within* a learner are supporting evidence only

**Learner seeds are the independent unit, not evaluation rows.** 6,400 evaluation units per
learner is not N = 6,400 when you trained 5 or 10 learners. Getting this wrong inflates
significance by roughly a factor of 25 and it is the error a statistically literate marker
will look for first. Viva question 15.

#### Acceptance

| # | Test | Proves |
|---|---|---|
| 1 | Every policy type — random, heuristic, P, PA — evaluates through the identical code path | Comparability |
| 2 | Paired differences are computed on matched `(motif, base_seed)` units | Pairing is real |
| 3 | The confidence interval is computed across learner seeds, with n reported | The right independent unit |
| 4 | Refuses to run on the final pool without `--final --confirm-freeze` | Contamination guard |
| 5 | Every reported number regenerates from the saved CSV | v3 §10.9 gate 7 |
| 6 | The random baseline averages 10 action seeds per base before comparison | v3 §9.1 |

---

### Stage 14 — `ui/app.py`

**Owner** C · **Blocked by** Stage 13

Streamlit, importing `env/render.py` from Stage 5. The renderer is already written and
already debugged; this stage is the interface around it.

#### Required views (v3 §14)

Base seed and motif selection · manual shopping and deployment · replay for heuristic,
standard PPO and augmented PPO · board with buildings, walls, troops, health, phase, tick,
budget · current legal actions · last action with its reward components · destruction and
Town Hall status · chronological event history.

The three Tier 2 overlays from Stage 5 become checkboxes: range circles, path overlay,
damage numbers.

**Replay works from recorded snapshots, not by re-simulating.** Re-simulating in the GUI
means the demo can diverge from the evidence, live, in front of a marker.

**A guard must block final-range seeds** with a clear message.

#### Acceptance

| # | Test | Proves |
|---|---|---|
| 1 | Manual play completes an episode | The interface is operable |
| 2 | Replay steps through a saved episode and matches the recorded destruction | No divergence |
| 3 | Entering a final-range seed is refused with a clear message | Contamination guard |
| 4 | Someone who did not build it can run it from the README | v3 §13.3 |

---

### Stage 15 — protocol runs

**Owner** all · **Blocked by** Stages 12, 13, 14

Train the registered number of learner seeds (O4) for each condition, P and PA, at matched
budgets and identical seed IDs.

Rules that are not negotiable once this starts:

- **Matched budgets.** Both conditions get 25,000 + 225,000. Symmetry augmentation is the
  only intended difference.
- **Use the checkpoint at the fixed end of the registered budget.** Do not select the
  best-looking checkpoint using final-test results.
- **No final-pool access.** Development evaluation only.
- Plot the full learning curve for every learner, with dev evaluations every 25,000 steps.
  If curves are still rising at budget end, say so plainly and frame the result as a
  matched-budget comparison rather than an asymptotic one. That framing is entirely
  defensible; silently implying convergence is not.

---

### Stage 16 — freeze and final evaluation

**Owner** all · **Blocked by** Stage 15

The ceremony matters. Do it in this order and record each step.

1. **Freeze the code.** Tag `freeze-final`. No commits to `env/` or `train/` after this.
2. **Confirm the final pool has never been read.** Check the git history and every run
   manifest for a `pool_final.json` path.
3. **Run the final evaluation once**, on seeds 11,000–11,199, all four motifs, all eight
   views. Seeds 11,200–11,999 remain unused reserve.
4. **Regenerate every table and figure from the saved CSV**, not from memory.
5. **Have a member who did not run it reproduce one learner** from a fresh environment,
   then regenerate one report table from raw evidence (v3 §13.3).
6. Update the truth-status register (v3 §20).

**Once.** If the result is disappointing, that is the result. v3 §10.9 is explicit: no gate
requires a convenient positive outcome.

---

## 7. Test inventory

| File | Tests | Cumulative | Covers |
|---|---:|---:|---|
| `test_config.py` | **48** | 48 | Value totals, tile counts, action encoding, rotation shift, LOS decision, motif geometry, seed splits |
| `test_entities.py` | 6 | 49 | Movement credit, damage, death, stat loading |
| `test_generator.py` | 10 | 59 | Determinism, counts, placement, motifs, JSON round-trip |
| `test_pathfinding.py` | 8 | 67 | Route choice, unit consistency, break-even, caching, invalidation |
| `test_combat.py` | 12 | 79 | Buffering, shuffle invariance, targeting, splash, min range, wall-crossing |
| `test_render.py` | 13 | 97 | Headless render, entity coverage, overlay counts |
| `test_env.py` + `test_mask.py` + `test_reward.py` + `test_termination.py` | 15 | 96 | Spaces, phases, every masking rule, component sums, termination vs truncation |
| `test_heuristic.py` | 6 | 102 | Determinism, army cost, margin over random, shared code path |
| `test_generator_validation.py` | 8 | 110 | Structural checks, difficulty gate, pool round-trip, final-pool guard |
| `test_train.py` | 7 | 117 | Save/reload, SubprocVecEnv, manifest, maskable callbacks |
| `test_symmetry.py` | 6 | 123 | The six invariance properties |
| **Total** | **139** | | |

v1 targeted 63. The increase is almost entirely `config.py` (43 rather than 4) plus the
renderer and the LOS and layered-core additions. Add more freely; these are minimums.

---

## 8. Definition of done for the core

- [ ] All 139 tests pass
- [ ] Checkpoint A: under 200 ms per episode, ideally under 50
- [ ] Checkpoint A: the balance investigation is written up
- [ ] Checkpoint B: a rising reward curve on a plain learner
- [ ] All six symmetry properties verified **before** any PA run
- [ ] Three pools built; the final pool untouched and guarded
- [ ] One learner trained end to end with a complete manifest
- [ ] The seed-count decision (O4) recorded with its justification, before final training
- [ ] A teammate who did not build a module can run it from the README and explain it

Only then start the protocol runs.

---

## 9. Outstanding edits to `BASE_SIEGE_PROJECT_PLAN_v3.md`

All six before the Part B freeze in Week 7. Each is a consequence of a decision in §2.

| # | Section | Edit | Because |
|---|---|---|---|
| 1 | §6.6 | Replace `offset ring` with `layered core` in the motif list. Describe it as an L-shaped outer wall on two adjacent sides plus a 5×5 ring enclosing the Town Hall, 35 walls | D6 |
| 2 | §6.6 | Clarify that wall counts are fixed **across seeds within a motif**, not across motifs — gated motifs have 32, the closed ring 36, the layered core 35. Walls score 0, so this does not touch the objective scale | The current wording contradicts the gate motifs |
| 3 | §10.5, §10.9 gate 5 | Rename the held-out motif to `layered core` | D6 |
| 4 | §6.7 | Keep both the "innermost enclosed region" and "every enclosed compartment" checks — they are now non-vacuous. Amend check 5 to say defences whose **range** covers the Town Hall, with no line-of-sight qualifier | D6, D7 |
| 5 | §6.8 | Add a line-of-sight statement: nothing blocks a shot; range alone determines eligible targets. Name the consequence — the ranged troop is the only troop that can attack across a wall, because it is the only one with range above 1 | D7 |
| 6 | §6.5 | Footnote the balance arithmetic. The pooled-HP-over-pooled-DPS model is wrong because defences focus-fire; five defences are five independent kill-slots. State the corrected prediction and cite the Checkpoint A instrumentation | D13 |

Two further items carried over from v3 §12 and still open:

- **Item 8** — reconcile the 12.7% gap-closure figure. The table gives
  (20.22 − 11.92) / (29.17 − 11.92) = 48.1%. Identify which definition produced 12.7% and
  footnote it wherever it appears.
- **Item 9** — record the learner seed count decision (O4) before any final training.

---

## 10. Build risk register

| Risk | Early warning | Response |
|---|---|---|
| Episode too slow for 10 learner seeds | Checkpoint A over 200 ms | Optimise in the Stage 6 order. If still slow, 5 seeds and state the detection limit |
| Symmetry mapping wrong | Any of the six properties fails | **Do not train PA.** A broken transform measures nothing and the curve looks fine |
| Composition collapses to all-ranged | Ranged share over ~70% at Checkpoint B | Shrink the placement region to the inner 6×6 (O5). Do not change the LOS rule |
| Agent refuses to buy breakers | Zero breakers in composition | Wall shaping bonus ~0.005, excluded from the reported score, written up |
| Pools invalidated by a late stat change | Any edit to `TROOP_STATS` or `BUILDING_STATS` after Stage 9 | Regenerate all three pools. Freeze the table at the end of Stage 8 |
| Maskable callbacks not used | Evaluation results inconsistent with training curves | Stage 10 test 5. The standard versions fail silently |
| Final pool touched early | A run manifest references `pool_final.json` | Guard in `load_pool`; audit manifests at Stage 16 |
| Curves still rising at 250k | Dev evaluation still climbing at budget end | Reframe as a matched-budget comparison, state it plainly |
| Study underpowered | Paired interval spans zero and is wide | Report the detection limit from v3 §10.2 as a table |
| Contribution unequal | Gaps in the weekly log | The stage ritual step 7 exists to prevent this. Rebalance early |

---

## 11. Viva preparation

Individual oral defence is Part G, 25 points, and it is assessed per member. Every member
must be able to answer every question below regardless of who wrote the module.

### The v3 checklist, mapped to the stage that arms you

| # | Question | Armed by |
|---|---|---|
| 1 | Why is this sequential RL rather than prediction? | Stages 7, 10 |
| 2 | Simulator state vs policy observation? | Stage 7 |
| 3 | Why partial rather than Markov? | Stage 7 |
| 4 | What are the 246 actions? How is deployment encoded? | Stages 1, 7 |
| 5 | Why mask illegal actions rather than penalise? | Stage 7 |
| 6 | What does PPO actually do? | Stage 10 |
| 7 | Why is shopping hard credit assignment? | Stage 10 |
| 8 | Why `gamma = 1.0`? | Stage 10 |
| 9 | Why does the deadline terminate, not truncate? | Stage 7 |
| 10 | Recite the seven-step tick order. Why buffer damage? | Stage 5 |
| 11 | Why consistent time units in path costs? | Stage 4 |
| 12 | Why matched budgets? | Stage 15 |
| 13 | How do rotations map back to deployment actions? | Stages 1, 12 |
| 14 | Why separate training, dev and final seeds? | Stage 9 |
| 15 | Why are repeated layouts not independent learner seeds? | Stage 13 |
| 16 | What is your minimum detectable difference? | Stage 10 (O4) |
| 17 | What did protocol v1 prove and fail to prove? | v3 §11 |
| 18 | Why do walls score zero, and what did it cost? | Stages 1, 5 |
| 19 | Why is MCTS not a ceiling? Why learn rather than search? | v3 §5 |
| 20 | Strongest limitation of the final system? | §2 of this document |

### Six questions this build adds

| Question | Answer lives in |
|---|---|
| Why does the ranged troop outrange the cannon? | D10 — it is the only thing that justifies buying one, and it is the same 4 that enables perimeter sniping |
| Why does the mortar have a minimum range? | Stage 5 — it makes danger non-monotone in distance, so placement is a spatial problem rather than a scalar sort |
| Why does nothing block line of sight? | D7 — blocking seals defences inside a closed ring; the asymmetry comes from the range table instead |
| Why is the held-out motif a layered core rather than a shifted ring? | D6 — a shifted ring is the same topology moved |
| Why is the deployment ring ordered clockwise from a corner? | D1 — it makes a 90° rotation a cyclic shift |
| What is wrong with §6.5's balance arithmetic? | D13 — it pools HP against DPS; defences focus-fire |

---

## 12. Schedule

Aligned to v3 §17. Dates nominal until each Canvas page is checked.

| Period | Stages | Milestone |
|---|---|---|
| Week 3 (10–14 Aug) | 2, 3 | Part A prep |
| Week 4 (from 17 Aug) | 4, 5 | Part A due. Plan edits §9 items 1–5 |
| Week 5 | 6 (Checkpoint A), 7 | Speed number known → O4 decidable |
| Week 6 (to 6 Sep) | 8, 9, 10 | **Time one full 250k run.** Freeze the stat table |
| Week 7 (from 7 Sep) | 11 (Checkpoint B) | Part B due. **Protocol freeze including seed count** |
| Weeks 8 + Stuvac | 12, 13 | Symmetry verified. No final access |
| Week 9 (from 28 Sep) | 15 | Protocol runs |
| Week 10 (from 5 Oct) | 13 | Part C due — development results |
| Week 11 (from 12 Oct) | 14 | Part D due — GUI and recorded demo |
| Week 12 (19–23 Oct) | 16 | **Freeze. Final evaluation once.** Part E due 23 Oct |
| Final Stuvac | — | Presentation and viva rehearsal |
| Week 14 (from 2 Nov) | — | Parts F and G |

**Two hard dependencies.** Timing a full run gates the seed-count decision, which gates
protocol freeze in Week 7. Baseline regeneration on the 155-point scale gates every
performance gate in the protocol. Neither can slip to Week 9.

---

## Appendix A — every contract in one place

```python
# entities.py
Building(building_type, position, hp, max_hp, target)
Troop(troop_type, position, hp, max_hp, entity_id, movement_credit, target, cached_path)

# generator.py
BaseSpec(seed, motif, buildings, walls)
generate_base(seed, motif) -> BaseSpec
wall_tiles(motif) -> set
placement_region(motif) -> set
validate_structure(spec) -> (bool, str)
validate_base(spec) -> (bool, float)
build_pool(seed_range, motifs, out_path) -> dict
load_pool(path) -> list[BaseSpec]                     # raises on the final pool

# pathfinding.py
PathCache.find_path(troop_type, start, goal, grid) -> list | None
PathCache.invalidate() -> None
open_tile_cost(troop_type) -> float
wall_tile_cost(troop_type, wall_hp) -> float

# combat.py
BattleState(buildings, troops, reserve, tick, wall_state_version, path_cache,
            destroyed_value, milestones, finish_reason)
step_tick(state) -> None
record_snapshot(state) -> dict
deploy_troop(state, troop_type, tile) -> None

# render.py
draw_snapshot(snapshot, ax, show_ranges, show_paths, show_damage) -> None

# siege_env.py
SiegeEnv.reset(seed, options) -> (obs, info)
SiegeEnv.step(action) -> (obs, reward, terminated, truncated, info)
SiegeEnv.action_masks() -> np.ndarray                 # METHOD, NOT WRAPPER

# symmetry.py
transform_grid / transform_tile / inverse_tile / transform_action / inverse_action
SymmetryWrapper(env)

# agents
RandomAgent.act(obs, mask) -> int
HeuristicCommander.act(obs, mask) -> int
evaluate_from_all_sides(base) -> float

# train
train(condition, learner_seed, out_dir) -> Path
evaluate_policy_on_pool(policy, pool, views, out_csv) -> DataFrame
paired_difference(csv_a, csv_b) -> dict
```

---

## Appendix B — the numbers, in one place

| Quantity | Value |
|---|---:|
| Grid | 16 × 16 |
| Deployment tiles | 60 (full perimeter) |
| Placement tiles | 64 ring motifs, 65 layered core |
| Budget | 200 |
| Max army | 25 (= 200 / 8, the melee cost) |
| Battle ticks | 300 |
| Ticks per decision | 10 → 30 battle decisions |
| Safety guard | 70 decisions (worst legitimate case is 56) |
| Total base value | 155 (walls score 0) |
| Wall counts | 36 / 32 / 32 / 35 |
| Heuristic army | 3 tanks, 6 melee, 4 ranged, 3 breakers = 199 |
| Difficulty gate | 25% – 85% |
| Training timesteps | 25,000 + 225,000 = 250,000 |
| Gamma | **1.0** |
| Train / dev / final seeds | 0–9,999 / 10,200–10,399 / 11,000–11,999 |
| Final study seeds | 11,000–11,199, used once |
| Mortar blind spot | 8 tiles |
| Standoff annulus vs a cannon | 20 tiles |
| Interior tiles snipeable from the ring | 28 of 64 |
| Perimeter tiles under mortar fire, best case | 19 of 60 |
| Break-even detour vs one wall | 3.2 melee / 3.5 ranged / 3.8 tank / 0.2 breaker |

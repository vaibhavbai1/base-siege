# Smooth playback verification

This revision preserves the prototype's artwork, palette, board and army cards,
and adds a player for actual Python-recorded simulation ticks. The included
`demo_preview.html` is a heuristic baseline, not evidence of trained PPO quality.

## Changes

- `env/siege_env.py`: opt-in tick snapshots and movement credit for presentation.
  Disabled by default; normal training does not retain these frames.
- `experiments/rollout.py`: optional visual frames alongside existing actions,
  decision snapshots and summary. Legacy replay schema remains readable.
- `ui/app.py`: sends complete replay timelines and manual action tick sequences.
- `ui/frontend/`: prototype artwork with animated troop movement and decorative
  attack effects; pause/play, seek, restart, decision step, speed and fullscreen.
- `tools/export_replay_html.py`: self-contained offline recorded demo export.
- `tools/build_deploy_bundle.py`: includes the additional player stylesheet.
- `tests/test_visual_replay.py`: recording equivalence and replay round trips.

No combat/pathfinding fixes, reward changes, training jobs or seed exclusions
were introduced. This is a presentation revision, not completion of the training
or evaluation study. Original planning documents retain their historical status.
The new handoff manifest supersedes the original root manifest, retained in
`reference/HANDOFF_MANIFEST-2026-09-08.json` for provenance.

## Automated checks

**204 passed, 1 deselected** across configuration, entities, generation,
pathfinding, combat, rendering, prototype parity, RL contracts, visual replay,
UI and public application tests. The deselected policy-cache test needs Torch,
which was unavailable in this verification environment. The original full suite
was not run. The command, after installing dependencies, was:

```bash
python -m pytest -q tests/test_config.py tests/test_entities.py \
  tests/test_generator.py tests/test_pathfinding.py tests/test_combat.py \
  tests/test_render.py tests/test_prototype_parity.py tests/test_rl_contract.py \
  tests/test_visual_replay.py tests/test_ui.py tests/test_public_app.py \
  -k 'not policy_cache'
```

The recording equivalence checks compare every existing result field against
ordinary recording for three seed/motif pairs: actions, rewards, decision
snapshots and final outcomes match exactly. They also check tick coverage,
JSON round trips and no recorded troop placement inside live walls. These
checks support the recording change, not universal correctness of the simulator.

JavaScript syntax checking passed. The standalone deployment bundle built with
31 listed source files, including the new stylesheet.

## Browser checks

Headless Chromium passed all eight checks in `verification/browser-results.json`,
with no JavaScript page errors:

- Play/pause, restart and next-decision controls.
- 5x speed advances playback more quickly.
- Actual recorded movement produces fractional visual positions.
- Final HUD matches Python: 67.7% destruction, tick 249, 199 resources spent.
- 390-pixel viewport has no horizontal overflow.
- Streamlit manual buy/start/deploy animates correctly.
- Manual reset restores the budget.
- Streamlit policy replay generates and plays.

The included episode has 308 visual frames and 42 decisions, destroys the Town
Hall, and ends with the army eliminated. Default playback duration is about
17 seconds; the speed control can slow it for a presentation. Desktop and mobile
screenshots were visually inspected. An example is included as
`verification/smooth-playback-desktop.png`.

## Limits

Verification used Python 3.12 with a lightweight dependency environment including
NumPy 2.5.3, Gymnasium 1.3.0 and Streamlit 1.64.0, not a full frozen training
installation. The repository dependency lock remains unchanged. Torch, PPO
training/checkpoint inference, notebook execution, AWS routing and GPU operation
were not tested. Only Chromium was exercised; fullscreen permission depends on
the browser and embedding host. No frame-rate benchmark or 60-fps guarantee is
claimed. Browser rendering uses requestAnimationFrame independently of the
recorded simulator tick schedule.

Smooth paths use movement credit and adjacent cells observed in the recorded
trajectory. Projectiles, arcs, damage text, recoil and health easing are visual
effects; Python states determine game outcomes. Older decision-only replay files
cannot recover missing tick-level events and display a notice.

For source-hash/reproducibility guidance and demo commands, read
`../DEMO_START_HERE.md` before generating new experiment pools or registrations.

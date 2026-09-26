# Smooth Base Siege demo

## Watch immediately

Open **demo_preview.html** in Chrome, Edge or Firefox. It is self-contained: no
server, Python, internet connection or SageMaker session is required. Press
**Play**. This included example is a **heuristic baseline**, not a trained PPO
agent. It shows an actual Python-simulated attack on seed 42, closed ring.

The player keeps the prototype's colours, board, building/troop drawings, army
cards and overlays. It adds smooth movement, projectiles, mortar arcs, damage
numbers, destruction debris, play/pause, restart, next-decision stepping,
0.2×–5× speed, seeking and a final-result banner. Fullscreen depends on browser
permissions; the standalone file is the simplest presentation option.

## Play manually or watch your trained agent

From the project folder, with the project environment installed:

```bash
uv sync --frozen
uv run streamlit run ui/app.py
```

Use the URL Streamlit prints. On SageMaker, use your institution's supported
application/port access method; its access routing has not been tested here.

- **Manual play:** buy your army, start battle, select a troop and click a green
  perimeter tile. Each legal decision animates its actual simulation ticks.
  Leave **Pause every decision** checked to decide after each action. Uncheck
  it to send successive wait actions when an animation finishes. Deployment
  remains on the same 10-tick decision schedule used for RL.
- **Policy replay:** choose Heuristic, Valid random or Saved Maskable PPO, then
  Generate replay. Press Play in the board controls. Speed affects playback,
  not the simulated outcome. Saved PPO requires your own compatible model.
- **Saved replay:** older decision-only files still work, with an explicit
  notice. Regenerate an episode in this version for full tick-level animation.
- **Download recorded episode:** saves the actions, ordinary decision snapshots,
  summary and optional tick-level visual frames together.

The prototype HTML remains available under `prototype/`. The demo uses Python
for game outcomes; JavaScript only draws recorded states and decorative effects.
Projectile travel and health-bar easing are visual, not new combat mechanics.
Smooth troop positions use movement credit and only routes that occurred in the
recorded trajectory. Exact grid positions remain in the replay data.

## Export your own standalone presentation replay

Download an episode from the app, then run:

```bash
uv run python -m tools.export_replay_html --replay your-replay.json --out my-demo.html
```

The output contains the recorded episode and player. It does not train a model
or contain interactive battle simulation. Export refuses to overwrite a file.
Use a new filename for subsequent exports. Opening the file locally is enough.

## Training and reproducibility

`record_ticks` defaults to false and is enabled only for demo environments.
The normal action/reward interface and decision snapshots are preserved.
Do not render or enable tick recording for the whole training matrix.

This revision adds recording to `env/siege_env.py` and replay export to
`experiments/rollout.py`, so the project's source hashes change. Adopt this
version **before** generating the shared pools, pilots and study registration.
Existing registered runs/checkpoints should continue in their original source
version; do not remove their provenance guards to force a resume. Exported model
inference in the demo does not itself resume training.

All three members should use the same revised package. Keep actual run folders
and credentials out of demo archives.

## Verification

See `docs/SMOOTH_PLAYBACK_VERIFICATION.md` for checks and limitations. The original
handoff manifest is retained in `reference/HANDOFF_MANIFEST-2026-09-08.json`; the
root manifest describes this updated source package.

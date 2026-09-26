# Verification · 24 September 2026

## Result

**59 focused Python/Streamlit/Node checks passed. Two existing Torch-dependent model checks were excluded because trained checkpoints and Torch were not available locally.**

The new Node tests executed the actual browser engine and compared every command/tick with Python in **19 complete battle scenarios**: all four structures at three seeds, five army compositions (including single-troop-type armies), and two custom layouts with no walls or dense walls. Comparisons included positions, HP, movement credit, cached paths, reserve, damage events, destruction, milestones, ending conditions and legal actions.

The checks do not establish mathematical equivalence for every possible layout or future configuration. They provide regression coverage for the mechanics introduced in this browser port. Keep them passing when combat changes.

## Browser controls and motion

Chromium opened the shipped frontend files with the Python-generated initial payload. Manual control checks passed for buying/refunding, starting paused, deploying one/five, deploying while running, ten-tick automatic pauses, continuous mode, keyboard selection and Space. These interactions sent **zero per-tick/action requests** to Streamlit.

A normal-speed movement test recorded **37 distinct display positions across 10 ticks** (38 sampled frames). The largest observed frame-to-frame movement was **0.028 tiles**. This verifies intermediate movement along a tile rather than waiting for an integer move and jumping. Exact frame rate depends on the user's browser/device; these are observed local measurements.

Pause froze the fractional display position; resume continued it. Reinjecting the initial server payload with the same game key did not rewind the locally played game. The separate `requestAnimationFrame` rendering loop interpolates movement credit and lets effects fade between ticks. Manual interpolation remains active when the OS requests reduced decorative motion; particles/bobbing retain the reduced-motion handling.

Editor clear/undo, movement, wall painting, perimeter rejection and custom-base submission passed. The self-contained custom-agent replay still played, paused, sought to the end and displayed its outcome without JavaScript errors. Screenshots were inspected.

## Installer and existing work

Installation, idempotence, exact restore, reinstall and unknown-edit protection were checked on the original Smooth Demo, the first App Update and v2. Model/pool/evaluation sentinels remained unchanged. Python source files under `env`, `agents`, `train` and `experiments` are unchanged.

The actual trained checkpoint files remain on SageMaker, so PPO inference was not rerun here. Its Python inference/recording path is retained. Full browser-to-Streamlit networking was not exercised in this workspace; Streamlit pages were checked with AppTest and the actual browser component was tested independently. Removing requests from the manual tick/action loop removes that network delay from ongoing manual play.

## Optional browser checks

From the installed project root:

```bash
uv run --with playwright==1.51.0 python -m playwright install chromium
uv run --with playwright==1.51.0 python base_siege_app_update/checks/browser_controls.py
uv run --with playwright==1.51.0 python base_siege_app_update/checks/browser_motion.py
uv run --with playwright==1.51.0 python base_siege_app_update/checks/browser_editor_replay.py
```

Checks write screenshots and a test replay under `artifacts/ui-v3-checks`. Temporary browser-test dependencies do not change the project dependency file.

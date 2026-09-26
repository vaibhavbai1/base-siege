# Base Siege application update · v3 · Smooth manual play

This fixes the lag in v2 manual play. The earlier update restored the controls but still waited for Streamlit between ticks. Its two-frame buffer also left slow troops stationary until an integer tile move, then animated that move too quickly.

Manual shopping, deployment, clock ticks and combat now run locally in the browser. Integer movement credit is drawn as fractional progress along open paths, interpolated on animation frames. A running battle sends no per-tick/action requests to Streamlit. The browser engine uses the base geometry and stat tables supplied by Python and is tested against the Python combat engine tick for tick. Trained-policy replay and evaluation continue to use Python.

## Install on SageMaker

Download this new ZIP even if an older file has the same name. Stop Streamlit with **Ctrl+C**, replace the old ZIP in your project folder, then run:

```bash
cd ~/Base_Siege_Smooth_Demo/base-siege-assignment3
unzip -o Base_Siege_App_Update.zip
uv run python base_siege_app_update/apply_update.py
uv run streamlit run ui/app.py --server.address 0.0.0.0 --server.port 8501
```

Keep any SageMaker proxy/base-URL launch options you already use. Refresh the app; use **Ctrl+Shift+R** if the browser retains the old component. The game should show **“Smooth manual play · Runs in your browser”** beneath its view controls.

This cumulative update supports the original Smooth Demo project and both previous App Updates. Keep the main project, `runs`, `bases`, and `artifacts`. The installer preserves models, pool files, experiment results and the Python sources under `env`, `agents`, `train`, and `experiments`.

## Manual controls

- Click a troop row to buy; Shift-click it to refund one.
- Start battle leaves the clock paused. Click a green perimeter tile to deploy one, or Shift-click to deploy up to five.
- Keys **1–4** select troops; **Space** runs or pauses. Click inside the game first to give it keyboard focus.
- You can deploy while the clock runs.
- **Pause every decision** runs ten ticks before pausing. Uncheck it for continuous play.
- The speed selector changes the local clock rate. **New base** restarts the selected geometry and seed.
- Switching away from the manual game can recreate its browser component and start a fresh battle when you return. A normal rerender of the same component does not rewind it. Changing the base starts a new game.

All four layouts, including **`layered_core`**, remain available. The base editor still supports moving buildings, drawing walls, undo, JSON import/export, and **Play manually / Let agent play**. Keep the standard inventory of 1 Town Hall, 3 cannons, 2 mortars and 4 storages. Custom geometry stays separate from experimental pools.

For a trained agent on a custom base, choose **Let agent play**, select the saved Maskable PPO model and Sampled/Deterministic settings, then **Generate replay → Play**. Custom and generated agent attacks retain the existing Python-recorded smooth replay path. A downloaded HTML replay is a recorded viewer; manual play is in the Streamlit app.

## Why there are now two combat implementations

Python remains the training/evaluation implementation. JavaScript runs only interactive human play, avoiding network delays. Browser constants come from Python configuration, and `tests/test_browser_manual_parity.py` compares every recorded tick across complete battles. Manual timing is different from agent decision timing: humans can deploy while paused or between ticks, whereas Gym actions retain their ten-tick decisions. Manual results are illustrative.

If combat mechanics change in a later project version, update the browser implementation and run the parity checks along with the Python tests. Changing only reward weights does not require rewriting the browser combat loop. New compatible models can still be selected as before.

## Checks and restore

```bash
uv run pytest -q tests/test_browser_manual_parity.py tests/test_manual_custom_ui.py tests/test_policy_playback.py tests/test_ui.py tests/test_public_app.py tests/test_visual_replay.py tests/test_deployment.py
```

The browser-engine parity tests need Node; they explicitly skip if it is unavailable. The included verification notes record the local results and optional Chromium checks.

To restore the interface present immediately before this update, stop Streamlit and run:

```bash
uv run python base_siege_app_update/apply_update.py --restore
```

Restart the app and refresh the browser. Backups are in `artifacts/app-backups/manual-smooth-ui-v3`. This restores interface files, including removing the new browser engine when appropriate; models and experiment results are preserved.

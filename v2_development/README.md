# Version 2 full development evaluation

This package evaluates the completed `version2-longer-control` models. It performs **no training**, makes no simulator changes and does not access the protected final set. Keep the original `version2/` folder unchanged. This separate package adds `v2_development/` only.

## What is being compared

- Six independent learners: P-0, P-1, P-2, PA-0, PA-1 and PA-2.
- Four saved budgets per learner: 262,144; 1,048,576; 2,097,152; 4,194,304 transitions.
- All **156 frozen development layouts**, in all **eight coordinate views**.
- Sampled actions using three fixed action-seed labels, plus deterministic actions.
- The original heuristic, evaluated once on each layout/view.
- Primary metric: percentage of building **value** destroyed. Full-clear and Town Hall rates are separate metrics.

There are 24 model checkpoints. Each receives 3,744 sampled attacks and 1,248 deterministic attacks. The heuristic receives 1,248 attacks. Total: **121,056 evaluation episodes**. These do not increase the training-episode count.

The full evaluation includes the twelve layouts already monitored during training. Summaries also report the remaining **144 layouts**, labelled `outside_monitor_panel`. Those layouts were absent from the V2 monitoring panel, but they are still development data, not a new final test set.

This evaluation plan was specified **after reviewing the four-million diagnostics**. Do not describe it as preregistered before training. The same coordinate views and action seeds are used across checkpoints, conditions and learner seeds. Seed derivation matches `version2.evaluation.episode_seed`.

## 1. Install the additional package

Upload `Base_Siege_V2_Development_Evaluation.zip` into the existing project folder. Open a **SageMaker terminal**, not a Python notebook cell:

```bash
cd ~/Base_Siege_Smooth_Demo/base-siege-assignment3
unzip -n Base_Siege_V2_Development_Evaluation.zip
```

The correct folder contains `env/`, `train/`, `version2/`, `runs/`, `bases/`, `pyproject.toml` and `uv.lock`. If you moved the project, adjust the `cd` path. `unzip -n` preserves existing files; it does not upgrade an older edited copy of this evaluation package.

Use the existing project environment. There is no dependency update or new training registration. Keep the installed scientific versions unchanged.

## 2. Check the inputs and software

```bash
uv run --frozen python -m v2_development preflight
uv run --frozen python -m pytest v2_development/tests -q
```

Preflight checks the original scientific source, registered V2 source, training registration, six completed run manifests/contracts, frozen development pool, and the **actual bytes of all 24 milestone model ZIPs**. Their hashes must match the metadata supplied in your review archive. It also checks the milestone sidecars and recorded scientific Python/package versions. Stop and share the error if a check fails; do not edit the expected hashes.

Expected preflight output includes:

```text
model_milestones_verified: 24
development_layouts: 156
expected_episodes: 121056
device: cpu
```

The tests evaluate a few attacks with the existing original `runs/study-v2/P-0/model.zip`. They do not train a model. Keep that original checkpoint available for the tests. Production evaluation reads only the selected V2 checkpoint paths recorded in `expected.json`.

CPU inference is intentional and uses one PyTorch thread for reproducibility. Low GPU utilisation during evaluation is expected. No change to your current SageMaker instance is required to launch it.

## 3. First run: 1,000 attacks

```bash
uv run --frozen python -m v2_development run --max-new-episodes 1000
```

This is the beginning of the real development evaluation, not a separate disposable test. It commits 1,000 attacks and then pauses. Its results are reused when you continue.

Progress prints every 100 new attacks, including a rough remaining-time estimate based on this attempt's measured throughput. That estimate will change with the model and layout; it is not a guaranteed finish time. This evaluation has many more attacks than the small training diagnostics, so do not estimate its duration from a single diagnostic check.

After the initial batch, inspect:

```bash
uv run --frozen python -m v2_development status
```

The expected state is `paused`, with 1,000 completed episodes out of 121,056. Continue with:

```bash
uv run --frozen python -m v2_development run
```

The same command starts a new evaluation when none exists and resumes one when it exists. A completed evaluation is checked/exported again without repeating attacks. Do not launch two copies at the same time.

## 4. Pause, renew your session and resume

In a **second terminal**, from the same project folder:

```bash
uv run --frozen python -m v2_development pause
```

The current attack finishes, commits to the progress database, and evaluation pauses before the next attack. Then check:

```bash
uv run --frozen python -m v2_development status
```

Wait for `paused` and the original terminal prompt before stopping the SageMaker space. Renewing login and restarting a stopped process are different operations. If you sign back in and the original evaluation is still printing progress, let it continue; do not launch a second copy.

When the previous process has stopped, resume with:

```bash
cd ~/Base_Siege_Smooth_Demo/base-siege-assignment3
uv run --frozen python -m v2_development run
```

Each finished attack is saved in a checksummed SQLite transaction. An abrupt interruption may repeat the attack that was in progress, but committed attacks are reused. `status` is a last-written progress snapshot, not proof that a process is alive. The output lock prevents two evaluation writers.

Do not delete the progress database, change evaluation code or alter inputs while an evaluation is paused. Resume checks the frozen evaluation contract. Code or input changes require a separately documented evaluation version.

## 5. Results and upload

Results are saved separately under:

```text
artifacts/version2-full-development/
```

The main files are:

| File | Purpose |
|---|---|
| `evaluation_contract.json` | Recorded inputs, hashes, layout/view/seed design and software |
| `progress.sqlite3` | Durable per-attack progress; retain for recovery |
| `progress.json` | Latest status and approximate time remaining |
| `attempts.jsonl` | Attempt start/end events and measured attempt durations |
| `episodes.csv` | Every completed evaluation attack, exported when complete |
| `model_summary.csv` | Each learner/checkpoint/mode, for both reporting scopes |
| `condition_summary.csv` | Equal-weight averages across the three learner runs |
| `budget_comparison.csv` | Each learner's change relative to its 262,144 checkpoint |
| `complete.json` | Completion receipt and exported result hashes |

An abrupt kill can leave an unmatched attempt-start event. Do not invent its missing duration or treat the gap until the next session as active computation time.

Once status says `completed`, create a compact results ZIP:

```bash
uv run --frozen python - <<'PY'
from pathlib import Path
from datetime import datetime, timezone
import json
import zipfile

root = Path.cwd()
folder = root / 'artifacts/version2-full-development'
receipt = json.loads((folder / 'complete.json').read_text())
assert receipt['status'] == 'completed'
names = ['evaluation_contract.json', 'complete.json', 'progress.json',
         'attempts.jsonl', 'episodes.csv', 'model_summary.csv',
         'condition_summary.csv', 'budget_comparison.csv']
for name in names:
    if not (folder / name).is_file():
        raise SystemExit(f'Missing output: {name}')
stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
output = root / f'Base_Siege_V2_Full_Dev_Results_{stamp}.zip'
with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as archive:
    for name in names:
        archive.write(folder / name, f'version2-full-development/{name}')
print(output)
PY
```

Download the resulting ZIP and upload it for analysis. Keep the SQLite database in your project backup; it is intentionally omitted from the compact analysis ZIP. Completed CSVs are evidence exports, not a replacement for the recovery database during an unfinished evaluation.

## How we will interpret the results

Compare budgets within each learner first, then average across the three independent learners. Hundreds of layouts/views/action samples do not become hundreds of independent trained models. The condition summary reports standard deviation across learners; it does not claim statistical significance from three seeds.

Use full development results to assess whether longer training improves performance beyond the small monitored panel. Examine full clears, low-destruction failures, P versus PA and sampled versus deterministic actions. A higher isolated checkpoint score on the small panel is not a retained model unless its ZIP exists; the evaluation deliberately uses only the four selected saved milestones.

Preserve the original scientific results. A new final evaluation and any confirmatory training require a separately fixed protocol. This package does not access or update old final results, add reachable-target fallback, change movement, or merge anything into GitHub.

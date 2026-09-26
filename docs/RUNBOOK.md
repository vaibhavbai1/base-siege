# Runbook: local work and the revision 2 study

Run from the complete base-siege-assignment3 folder. The [team master](TEAM_MASTER.md) explains
the project; [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) defines the study. These are instructions,
not a record that the long experiments have already run.

Use Python 3.12 and the locked environment. Keep inputs, journals and run folders on persistent
storage. No S3, cloud provisioning or deployment is performed by these commands.

## 1. Install and run bounded checks

~~~bash
uv sync --frozen
uv run pytest -q
uv run python tools/execute_notebooks.py --out artifacts/my-notebook-check
uv run python tools/smoke_pipeline.py --out artifacts/my-smoke-check
uv run python -m tools.check_recovery --out artifacts/my-recovery-check --kill-process
~~~

Use a new output folder for each check. The smoke pipeline trains two 256-step CPU learners;
it is not the learning-quality pilot. The recovery drill kills only its own test child.
Notebook training is disabled by default. Notebook 03b runs a short CPU recovery exercise.

For the app and notebooks:

~~~bash
uv run streamlit run ui/app.py --server.address 127.0.0.1 --server.port 8512
uv run jupyter lab notebooks
~~~

For a public demo, use ui/streamlit_app.py and [DEPLOYMENT.md](DEPLOYMENT.md). The public source
bundle is not the complete project to upload to SageMaker.

## 2. Prepare the declared pools

This section can take longer than a smoke check. It does not access protected final seeds.

~~~bash
uv run python -m experiments.pools --split train --count 10000 --out bases/train-v2.json
uv run python -m experiments.pools --split dev --count 200 --out bases/dev-v2.json
~~~

There are 30,000 training candidates and 600 development candidates, not guaranteed accepted
counts. Both partitions use the three training structures. Layered-core is excluded from current
development; its historical use in software checks is disclosed in the experiment plan.

If a pool command is interrupted, repeat that exact command with --resume. The companion
.progress.sqlite3 database saves one committed candidate at a time. Keep it beside the pool.
Do not change source, seeds, structures or filter settings while resuming.

## 3. Run the development pilot pair

First complete the Studio kernel, disk and recovery checks in [SAGEMAKER_STUDIO.md](SAGEMAKER_STUDIO.md).
Use --device cpu instead of cuda only if that is the device being assessed for the main study.

~~~bash
uv run python -m train.train_ppo --pilot --confirm-long-run --device cuda --condition P --seed 100 --train-pool bases/train-v2.json --dev-pool bases/dev-v2.json --out runs/pilot-v2/P-100
uv run python -m train.train_ppo --pilot --confirm-long-run --device cuda --condition PA --seed 100 --train-pool bases/train-v2.json --dev-pool bases/dev-v2.json --out runs/pilot-v2/PA-100
~~~

Each pilot has 57,344 transitions. It tests the planned network, four workers, optimiser settings
and phase transition with a shortened joint phase. Neither pilot counts as a full learner.

Individual pause/resume:

~~~bash
uv run python -m train.train_ppo --request-pause --out runs/pilot-v2/P-100
uv run python -m train.train_ppo --resume --out runs/pilot-v2/P-100
~~~

Wait for status paused before a planned shutdown. Resume reads saved settings; do not add a
new seed, condition, device or budget. A valid completed run is not trained again.

## 4. Evaluate and review the pilots

~~~bash
uv run python -m train.evaluate --pool bases/dev-v2.json --runs runs/pilot-v2/P-100 runs/pilot-v2/PA-100 --all-views --out artifacts/pilot-dev-v2
uv run python -m experiments.pilot --runs runs/pilot-v2/P-100 runs/pilot-v2/PA-100 --evaluation artifacts/pilot-dev-v2
~~~

The first command journals completed episodes. If interrupted, repeat it with --resume.
The second is a read-only report, including pilot-based training-time estimates. It does not
automatically approve the policy or prove augmentation works.

Inspect curves, baselines, empty armies, failures, throughput and free disk space. If changes
are needed, make them now and repeat matching pilots with new folders. After the team genuinely
agrees to proceed, record its decision:

~~~bash
uv run python -m experiments.pilot --runs runs/pilot-v2/P-100 runs/pilot-v2/PA-100 --evaluation artifacts/pilot-dev-v2 --approve-study --note "<replace with the team's actual evidence-based review>" --out experiments/pilot-review-v2.json
~~~

Do not keep the placeholder as a real review. Registration checks the review, pilot files,
evaluation receipt and pool identities. A software-smoke run cannot satisfy this gate.

## 5. Register and inspect the full matrix

The default is ten learner seeds per condition: 20 learners and 5,242,880 transitions.

~~~bash
uv run python -m experiments.protocol register --train-pool bases/train-v2.json --dev-pool bases/dev-v2.json --pilot-review experiments/pilot-review-v2.json --out experiments/registration-v2.json
uv run python -m experiments.study plan --registration experiments/registration-v2.json --device cuda
~~~

The plan command does not train. An explicit five-seed registration is supported if the team
changes the resource plan before main training; do not choose it after inspecting significance.
The commands below assume the default ten seeds.

## 6. Execute, pause and resume the study

~~~bash
uv run python -m experiments.study run --registration experiments/registration-v2.json --device cuda --out runs/study-v2 --confirm-long-run
~~~

The controller alternates P/PA order between seed pairs. It checks free storage before each
learner and records progress. Do not launch separate copies of the same controller.

From another terminal, request a clean pause:

~~~bash
uv run python -m experiments.study pause --out runs/study-v2
~~~

Wait for the controller to report paused. After restarting the same environment:

~~~bash
uv run python -m experiments.study run --registration experiments/registration-v2.json --device cuda --out runs/study-v2 --confirm-long-run --resume
~~~

The controller verifies and skips completed learners. It resumes the interrupted learner using
its own checkpoint. The default free-space threshold is 8 GiB; no checkpoint cleanup happens
automatically. A failed attempt remains recorded.

## 7. Complete development evaluation

The brace expansion below works in Bash and zsh. Include every registered learner.

~~~bash
uv run python -m train.evaluate --pool bases/dev-v2.json --runs runs/study-v2/P-{0..9} runs/study-v2/PA-{0..9} --all-views --out artifacts/study-dev-v2
~~~

Repeat with --resume after interruption. Outputs are episodes.csv, paired_summary.json,
study_summary.json, evaluation.json and evaluation.progress.sqlite3. Use notebook 04 to analyse
these files. Do not promote development scores to final findings.

## 8. Freeze and run final evaluation later

Only proceed when development is complete and all registered learners are verified. These
commands are never invoked by the default notebooks or the study controller.

~~~bash
uv run python -m experiments.protocol freeze --registration experiments/registration-v2.json --runs runs/study-v2/P-{0..9} runs/study-v2/PA-{0..9} --out experiments/freeze-v2.json
uv run python -m experiments.protocol final --freeze experiments/freeze-v2.json --out artifacts/final-study-v2 --confirm-final
~~~

The final marker binds this freeze to this output. A crash is resumed using:

~~~bash
uv run python -m experiments.protocol final --freeze experiments/freeze-v2.json --out artifacts/final-study-v2 --confirm-final --resume
~~~

Pool candidates and evaluation episodes already committed are reused. Different code, weights
or destinations are rejected. Do not delete the marker or use reserve seeds to retry a hypothesis.
Final completion is recorded in complete.json.

## 9. Share the right files

Preserve source, uv.lock, approved review, registration/freeze, accepted pools, run contracts,
models, journals and result receipts together. Exclude machine-specific environments and caches
from a submission ZIP. Freeze model paths are relative, so move the complete tree together.
Registration contains original pool paths: keep that location for the live matrix; individual
learner folders contain portable copies of their inputs.

Only load trusted PPO ZIPs and recovery files. For public hosting, export reviewed inference
assets rather than entire research run folders. No Git commit, push, Canvas upload or cloud
deployment is performed by this runbook automatically.

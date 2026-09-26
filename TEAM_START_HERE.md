# Base Siege: start here

UTS 43008 Reinforcement Learning · Assignment 3 · Project 12 · Team handoff, 8 September 2026

This is the complete research-project source, not the smaller public-demo bundle.
Keep the whole base-siege-assignment3 folder together after extracting the ZIP.

## Read first

1. [Team master document](docs/TEAM_MASTER.md): our question, game, study and remaining work.
2. [README](README.md): project map and notebook order.
3. [Runbook](docs/RUNBOOK.md): exact setup, pilot, training, recovery and evaluation commands.

The separate master-document file sent alongside this ZIP is an identical reading copy.
Use the copy inside docs/ when following its relative links.

## Run locally

Use Python 3.12 on macOS or Linux. On Windows, use WSL2: the research recovery code
uses Unix file locking. The complete Linux/SageMaker workflow has not yet been run.

With uv installed, open a terminal inside the extracted project folder:

```bash
uv sync --frozen
uv run pytest -q
uv run streamlit run ui/app.py --server.address 127.0.0.1 --server.port 8512
```

Open http://127.0.0.1:8512 to use the local app. To open the notebooks, use another terminal:

```bash
uv run jupyter lab notebooks
```

Follow the seven notebooks in the order listed in README.md. Shared modules contain the
simulator and learning implementation, so sending the notebooks alone is not sufficient.
Installing dependencies needs internet access and disk space; a virtual environment is not
included. The ZIP's small size does not represent the installed environment or training storage.

## What the team can do

- Read and analyse the environment, policies, experiment design and notebooks.
- Play manually and run the random and heuristic agents without a trained model.
- Run the bounded software checks, including the short checkpoint-recovery exercise.
- Generate procedural layout pools, run the pilots, and then register and execute the study.
- Resume saved work and analyse the resulting CSVs and replay files.
- Prepare a separate public-demo bundle using the included deployment tooling.

Use docs/SAGEMAKER_STUDIO.md before any GPU work. No S3 is required. Keep the project, pools,
journals and run folders on persistent storage and back up important evidence separately.
Do not turn on long-training controls until the team has reviewed the runbook and pilot gates.

## What is not included or claimed

This package has no full-study trained models, final research results, completed pilot review,
or registered experiment. Those steps have not been completed. Long notebook training is off
by default. Small recovery and smoke checks do not prove that a policy learned a useful strategy.

Machine-specific environments, Git history, local caches, run folders and local verification
outputs are excluded. docs/VERIFICATION.md records earlier local software checks; its
artifacts/ paths refer to outputs on the originating machine, not files supplied in this ZIP.
The team can generate fresh outputs using the bounded commands in the runbook.

The original Stage 5 ZIP is preserved unchanged in reference/ for provenance. Work from the
current project files, not by replacing them with that earlier archive. No cloud resources,
credentials, deployment or automatic experiment execution are provided by this handoff.

HANDOFF_MANIFEST.json records the source revision and checksums. Packaging checks confirm
archive integrity, required files and Python/notebook syntax; they do not execute the project.

# TRPO in the Base Siege interface

The standard local/SageMaker application now includes `Saved Maskable TRPO` in
Policy replay, alongside the existing PPO and DDQN choices. Use the same Base
editor to create and save a custom layout, then choose `My custom base` in Policy
replay. The existing smooth viewer, JSON download and HTML download are reused.

The included model is T-1 (learner seed 1), trained for 4,194,304 transitions
without symmetry augmentation. It was selected on sampled development results.
It is a demonstration checkpoint, not an ensemble or an aggregate benchmark.

## Start the application

In a SageMaker terminal, change to your existing project root (the folder that
contains `pyproject.toml` and `ui/app.py`). Run:

```bash
uv run --frozen python -m ui.trpo_models check
uv run --frozen streamlit run ui/app.py --server.address 0.0.0.0 --server.port 8501
```

Open the usual SageMaker port preview. Choose **Policy replay → Saved Maskable
TRPO → Generate replay**. SS is selected initially. If the application is already
running, stop it with Ctrl+C and start it again after installing the update.

- SS samples both purchases and deployment/wait actions.
- DS chooses purchases deterministically and samples deployment/wait actions.
- DD chooses both phases deterministically; the action seed is disabled.

The action seed and agent-view controls work as before. Changing the model,
acting mode, seed, view or base clears an outdated replay. The V3 random-seed
namespace matches the V3 evaluator for generated bases; custom bases also include
their geometry in the seed identity. Custom replays are illustrations, not final
benchmark attacks.

## Integrity and scope

The adapter loads with `MaskableTRPO.load`, verifies the selected checkpoint,
manifest and training contract, checks frozen inference source and configuration,
and verifies the loaded transition counter and learner seed. Inference uses CPU
and legal-action masks. No training or full evaluation is launched.

Files under env/, agents/, experiments/ and train/ are unchanged. Frozen result
files, the V2 model registry, PPO defaults and the DDQN adapter are unchanged.
The training algorithm in trpo/maskable_trpo.py is copied byte-for-byte from the
reviewed handoff. Existing public-only `ui/streamlit_app.py` remains independent
of local TRPO checkpoints, just as it is independent of local DDQN checkpoints.
This update targets the normal application launched with `ui/app.py`.

## Verification

From the project root, run the short inference and UI checks:

```bash
uv run --frozen python -m pytest tests/test_trpo_application.py -q
```

These checks load the existing model; they do not train it. They cover acting-mode
parity, custom-base geometry, downloads, invalid checkpoints, controls and stale
replay removal. An example replay score is not reported as a new research result.

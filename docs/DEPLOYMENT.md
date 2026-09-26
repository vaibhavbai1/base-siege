# Public deployment: Streamlit Community Cloud

Prepared for the free Community Cloud service. The app is public; training and checkpoint
recovery stay in SageMaker Studio. No Dockerfile, S3 bucket, paid API or application secret
is needed. This guide does not mean the app has already been published.

## Deployment settings

| Setting | Value |
|---|---|
| Main file | `ui/streamlit_app.py` |
| Python | **3.12**, selected in Advanced settings |
| Python dependencies | `ui/requirements.txt` |
| Visibility | Public |
| Secrets | Leave empty |
| Compute | CPU; no GPU required for this demo |

The hosted entrypoint uses the same `ui/app.py`, Python environment and canvas artwork as
the local app. It enables a public mode that reads only `demo_assets/`. Local runs and
checkpoints are not offered to visitors. Each visitor has a separate game in session state.
At most two saved models are cached per process, with one CPU thread and serial prediction
per model. There is no training button, arbitrary file upload or cloud write operation.

## 1. Test the public version locally

Run from the project root using the existing environment:

```bash
.venv/bin/python -m streamlit run ui/streamlit_app.py --server.address 127.0.0.1 --server.port 8513
```

Open `http://127.0.0.1:8513`. This is a different port from the local research app.
The double-click launcher still opens the research app on port 8512.

For a clean deployment environment on Linux x86_64 or an Apple Silicon Mac:

```bash
uv venv --python 3.12 .venv-demo
uv pip install --python .venv-demo/bin/python -r ui/requirements.txt
.venv-demo/bin/python tools/check_public_demo.py
```

Do not install the root training `requirements.txt` for hosting. The UI profile uses an
explicit, hash-pinned Linux CPU PyTorch wheel. It does not request CUDA, JupyterLab or
TensorBoard. Direct runtime packages are pinned; indirect dependencies are resolved at
install time, and CI runs a dependency-consistency check. Keep the original `pyproject.toml`,
`uv.lock` and training environment for SageMaker unchanged.

## 2. Build the standalone deployment folder

In the full project, run:

```bash
.venv/bin/python tools/build_deploy_bundle.py --out artifacts/streamlit-deploy
```

The builder creates a folder and `artifacts/streamlit-deploy.zip`. It refuses to overwrite
an existing bundle; choose a new output name for a later build. This is a copy operation,
not a Git commit or upload.

The copy contains the shared simulator, UI, CPU requirements, selected demo assets, a
checksum inventory and its own small CI workflow. It excludes the training runner,
recovery state, notebooks, original ZIP, local caches, credentials and experimental output
folders. `train/features.py` is included because saved policies need its feature extractor.

Test the copy from its own root, using the full project's absolute Python path or a fresh
environment installed from the copy's `ui/requirements.txt`:

```bash
python tools/check_public_demo.py
```

Do not set `PYTHONPATH` to the full project. The copy must work by itself.

## 3. Publish later

These are manual setup steps, not actions performed by the local build:

1. Put the **contents** of the deployment folder in a dedicated GitHub repository. Keeping
   that repository private does not prevent making the running demo public. Do not publish
   the whole coursework archive just to host this app.
2. Connect that repository in Streamlit Community Cloud and choose its actual branch,
   `ui/streamlit_app.py`, and Python 3.12. No server address or port override is needed.
3. In the app's Sharing settings, choose public access. A private source repository initially
   creates a private app, and Community Cloud permits only one private app at a time. If
   that slot is already occupied, resolve it deliberately; do not delete another assignment.
4. After the build, test the public URL in a signed-out browser: shopping, deployment,
   baseline replay, restart/reconnect, and any published policy.

The repository-root `.streamlit/config.toml` keeps the existing theme and uses the host's
default address/port. XSRF and CORS protections stay enabled. Error details are hidden from
visitors; inspect deployment logs to diagnose failures. Community Cloud may override some
configuration, including its own usage telemetry setting.

## CI and deployment updates

The full project's `.github/workflows/ci.yml` runs the regression suite and the public
preflight on Linux with Python 3.12 and CPU PyTorch, then builds a downloadable bundle.
The standalone bundle has a smaller workflow with app tests and preflight checks only.
Neither workflow needs deployment credentials or pushes to another repository. No full
training study runs in CI; the main regression suite includes only tiny recovery checks.

Community Cloud redeploys when its connected GitHub branch changes. It does **not** wait
for these tests automatically. To gate releases, make the CI check required on that branch
and merge reviewed pull requests only after it passes. Those repository rules and the
Streamlit connection must be configured later; they have not been created locally.
GitHub Actions uses the repository's included minutes/storage, so check its allowance.

## Adding a trained policy later

Start with manual play, the heuristic and valid-random agent. These are working baselines,
not evidence that PPO has learned a strong strategy.

After reviewing a trusted completed training run in the full project:

```bash
.venv/bin/python tools/export_demo_model.py --run runs/P-0 --name P-0
```

The command verifies the run/model checksum and step count, then builds a fresh CPU model
with the same policy weights. It writes `demo_assets/models/P-0/model.zip` and a small
manifest. Source machine paths, TensorBoard settings, worker/RNG recovery state and learned
optimiser state are not carried over. SB3 retains an empty optimiser configuration for
loading compatibility. This export is for inference, **not for resuming training**.

Only load trusted team models: Python serialisation in an SB3 ZIP can execute code. Smoke
runs retain an explicit software-check warning. Full training alone is not a performance
claim; report held-out evaluation separately. Exports never overwrite an existing folder.
Rebuild the deployment bundle after adding or changing an approved export.

Reviewed development replay JSON and evaluation CSV files can be placed in `demo_assets/`
as described in its README. Nothing is copied automatically from `artifacts/` or `runs/`.
Do not publish protected final layouts or claim development results as final findings.

## Free-host limits and presentation checklist

- Treat the hosted filesystem and browser sessions as temporary. A restart can reset the
  current game; use the replay download button before leaving. Keep important files in the
  source repository and long-run recovery checkpoints in Studio, never on the demo host.
- Free Community Cloud apps sleep after 12 hours without traffic. Open the link before a
  presentation and keep the local launcher as a fallback.
- Resource limits can change. The local memory preflight measures one process, not a
  concurrency guarantee or a measurement of Streamlit's Linux host. Check hosted memory,
  startup and a few simultaneous visitors after the first real deployment.
- If the host cannot install PyTorch, check Python 3.12 and `ui/streamlit_app.py` first.
  Selecting the root training lockfile by mistake can install unnecessary GPU packages.
- No paid service is required by this code. Creating or managing cloud accounts is outside
  the local preparation step.

## Provider references

Checked 8 September 2026: [Community Cloud overview](https://docs.streamlit.io/deploy/streamlit-community-cloud),
[dependency selection](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies),
[file organisation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization),
[sharing](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app),
[configuration limitations](https://docs.streamlit.io/deploy/streamlit-community-cloud/status),
[sleep and resource limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app),
and the [official PyTorch CPU wheel index](https://download.pytorch.org/whl/cpu/torch/).

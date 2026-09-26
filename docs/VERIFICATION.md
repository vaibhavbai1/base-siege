# Local verification · 8 September 2026

Software checks on this Mac, using the project's Python 3.12 environment. These are not trained-policy research results.

## Current: study revision 2

| Check | Observed result | Evidence |
|---|---|---|
| Full regression suite | 245 passed in 30.77 seconds; no failures or skips | artifacts/study-v2/tests.xml |
| Updated notebooks | Seven notebooks, 119 cells and 55 code cells; every notebook executed successfully | artifacts/study-v2/notebooks-verified/execution.json |
| Markdown rendering regression | Prose after headings remains prose; notebook syntax, schemas and disabled long-run controls pass | tests/test_notebooks.py |
| Short P/PA integration | Two workers, 256 transitions each; save/reload and journalled evaluation passed | artifacts/study-v2/integration/verification.json |
| Integration evaluation | Four accepted development layouts, eight views, P/PA, heuristic and ten random seeds: 416 rows; compatible completed resume is idempotent | artifacts/study-v2/integration/evaluation/ |
| Training recovery | Clean pause and SIGKILL recovery matched uninterrupted weights, optimiser, episode history and counters | artifacts/study-v2/process-recovery/recovery_report.json |
| Compact pool snapshots | P and PA recovery with two workers and real small training pools matched uninterrupted runs | tests/test_checkpoints.py |
| Pool and episode recovery | Completed units reused; changed contracts, overwritten evidence, corrupt rows and competing writers rejected | tests/test_pipeline_recovery.py |
| Abrupt journal interruption | Only the test's child writer was killed during a transaction; committed rows survived and the uncommitted row rolled back | tests/test_pipeline_recovery.py |
| Final recovery contract | Same-freeze/output resume tested with training-seed fixtures; no actual protected final layout generated or evaluated | tests/test_pipeline_recovery.py |
| Pilot and study gates | Pending review, changed evidence and smoke-as-pilot rejected; matrix sequencing, completed-run reuse, source/device guards and relative freeze paths tested | tests/test_pilot_review.py and tests/test_pipeline_recovery.py |
| Public source bundle | 30 selected source/assets files rebuilt; no trained research policy included | artifacts/study-v2/streamlit-deploy.zip |
| Standalone demo preflight | Ran from the bundle's own root in the demo-only environment; heuristic replay completed; 273.7 MiB peak RSS in one local process | artifacts/study-v2/public-preflight.json |
| Source provenance | Current scientific source hashes match the verified integration; original download and reference ZIP retain the same SHA-256 | artifacts/study-v2/integration/verification.json and source comparison |
| Static checks | Python compilation, CLI entrypoints and whitespace checks passed | Local command output |

The first notebook attempt was blocked by the desktop sandbox's loopback-port restriction.
The retry used permission for local Jupyter kernels and a new output directory; all seven
notebooks passed. The failed initial check was stopped. The notebook kernels emitted their
unencrypted-local-TCP warning and were shut down after execution. Streamlit's command-line
preflight emitted its usual bare-mode ScriptRunContext warning.

The public demo retains its existing theme. Its application regression tests and standalone
preflight passed; a new browser interaction session was not part of this revision. Earlier
browser screenshots below describe the preceding local UI verification. No Linux or hosted
runtime execution, GitHub Actions run, SageMaker/CUDA test or public deployment is claimed.

No 57,344-step pilot, full learner matrix, real final evaluation or assignment submission was
run. Synthetic fixtures test protocol behaviour, not policy performance. The pilot review gate
is implemented and tested, but no actual pilot approval was created.

Previous source/configuration artifacts remain untouched below. They are historical software
evidence and must not be presented as results of revision 2.


## Historical: public Streamlit deployment preparation

| Check | Observed result | Evidence |
|---|---|---|
| Complete regression suite | 228 passed in 23.46 seconds | `artifacts/streamlit-deployment-tests.xml` |
| Standalone public source | 30 explicitly selected files, plus generated README/inventory; approximately 62 KB zipped; hashes match the project source | `artifacts/streamlit-deploy/`, `artifacts/streamlit-deploy.zip` |
| Clean demo-only environment | 53 packages installed; dependency check passed; no JupyterLab, TensorBoard, nbformat or pytest installed | `.venv-demo/`, `ui/requirements.txt` |
| Isolated public preflight | Copy ran from its own root using the demo-only environment; heuristic episode completed; 301.5 MiB peak process RSS on this Mac | `artifacts/streamlit-public-preflight.json` |
| Real checkpoint export | Existing 256-step software-check run exported into a separate validation copy; 1.5 MB inference ZIP loaded without the training modules; completed a 56-decision episode | `artifacts/streamlit-export-preflight.json`, `artifacts/streamlit-policy-validation/` |
| Export integrity | Weight equality, prediction parity, empty learned optimiser state, private metadata removal, checksum rejection and overwrite refusal tested | `tests/test_deployment.py` |
| Public session and asset boundaries | Independent games; public mode excludes local runs/artifacts; reserved seeds and escaped asset symlinks rejected | `tests/test_public_app.py` |
| Linux dependency resolution | All-binary dry run resolved 54 packages for Python 3.12, Linux x86_64/manylinux 2.28; selected the official CPU PyTorch wheel, not CUDA dependencies | `ui/requirements.txt` and local resolver output |
| Browser interactions | Buying tank left 175 resources; shopping ended at 300 ticks; deployment advanced to 290; heuristic replay reached its last frame; JSON download contained 43 frames | `artifacts/streamlit-public-downloaded-replay.json` and screenshots below |
| Responsive display | Page width/scroll width 390/390; component width/scroll width 358/358; no horizontal overflow | `artifacts/streamlit-public-mobile.png` |
| Browser errors | No page errors or console messages observed during the standalone checks | Isolated Chrome verification on local port 8513 |
| Configuration and CI structure | Python compilation, launcher syntax, workflow parsing and `git diff --check` passed | `tests/test_deployment.py`, `.github/workflows/ci.yml` |

Browser screenshots: `artifacts/streamlit-public-initial.png`,
`artifacts/streamlit-public-replay.png` and `artifacts/streamlit-public-mobile.png`.
The verification browser and temporary port-8513 server were closed after testing;
the existing local research service was not stopped.

The deliverable bundle contains **no policy weights or experimental result files**. Manual
play and the two baselines work immediately. The real smoke export was used only in the
separate validation copy; its public manifest clearly labels it as a software check.
The original model and recovery artifacts were not changed.

The model-export preflight reached 384.2 MiB peak RSS in one local process. Neither memory
measurement establishes the Linux host's actual memory use or concurrent-visitor capacity.
Linux wheels were resolved, not executed locally; Docker was not running and no cloud
compute was started. The GitHub Actions workflows are prepared but have not run on GitHub.
No Streamlit account, repository remote, public app or branch-protection rule was created.

The Streamlit test helper emitted its normal bare-mode `ScriptRunContext` warning; the
actual browser application did not show an error. Training code, notebook source, source
ZIP, training dependency lock and registered evaluation rules are unchanged by this work.

## Checkpoint and notebook upgrade: earlier checks

| Check | Observed result | Evidence |
|---|---|---|
| Complete regression suite | 211 passed in 12.20 seconds | `artifacts/test-results-checkpoint-upgrade.xml` |
| Expanded notebooks | Seven notebooks, 115 cells, 53 executable cells; all executed successfully | `artifacts/notebooks/execution.json` |
| Markdown regression | Paragraphs after first-line headings render as prose, not indented code | `tests/test_notebooks.py` |
| Visual notebook review | Baseline prose/plots and recovery table inspected in isolated Chrome; no page/console errors observed | `artifacts/notebook-preview/baselines-top.png`, `recovery-table.png` and HTML previews |
| Controlled pause/resume | Identical weights, optimiser, episode history and counters to uninterrupted CPU training | `artifacts/recovery-2026-09-08/recovery_report.json` |
| Abrupt shutdown | SIGKILL of the drill's own child process; restored from 32 committed transitions and matched uninterrupted final state | `artifacts/recovery-2026-09-08/interruption.json` and recovery report |
| Recovery edge cases | P/PA, two workers, both phases, rounded budgets, endpoint interruption, corrupt checkpoint/pointer, moved folder, source mismatch, history rollback, locking | `tests/test_checkpoints.py` |
| Updated integration pipeline | P/PA: 256 transitions each, two workers, save/reload and eight-view development evaluation; 728 rows | `artifacts/checkpoint-upgrade-smoke-2026-09-08/verification.json` |
| Source preservation | Original download and reference ZIP both hash to `650acdfd85271c1fe55c4107d8d31488eedca0410413416ea526757b56b50ece` | Live SHA-256 comparison |

The exact UTSWork Semester 3 Assignment 3 notebooks and shared training helper were read without
editing that repository. Their local, shared-helper workflow informed this revision. No S3
integration, AWS resources, long study, final-layout access, submission or commit was performed.

Full training is disabled by default in the notebooks. Notebook 03b deliberately runs a tiny CPU
recovery exercise; its output is software evidence. Optional real-result plotting branches require
an explicitly selected CSV/run folder and were not populated with invented research results.

The development diagnostic now runs at completed updates and its history is checkpointed. A
shortened interval is exercised in recovery tests. The previous mixed-vector callback warning
does not describe the new implementation. GPU support is implemented but not verified in Studio.

Notebook kernels emitted a local unencrypted-TCP warning and were explicitly shut down. HTML
conversion warned that default Matplotlib output images lack alternative text; surrounding
notebook explanations remain present. Visual review used local HTML files, not a deployed site.

## Initial build: earlier verification snapshot

The following records the build before this checkpoint/notebook upgrade. Counts and paths below
are historical evidence, not the current notebook size or test count.

| Check | Observed result | Evidence |
|---|---|---|
| Complete regression suite | 197 passed in 14.37 seconds | `artifacts/test-results.xml` |
| Notebook execution | All five passed, 17 code cells total | `artifacts/notebooks/execution.json` and executed notebooks |
| P integration | 256 actual steps: 64 placement, 192 joint; saved/reloaded | `artifacts/verification-2026-09-08/P/manifest.json` |
| PA integration | Same budgets; saved/reloaded | `artifacts/verification-2026-09-08/PA/manifest.json` |
| Multiprocessing | Both conditions trained with two spawned workers and environment action masks | Run settings and `verification.json` |
| Development evaluation | Seven accepted layouts × eight views; P, PA, heuristic and ten random action seeds; 728 rows | `artifacts/verification-2026-09-08/episodes.csv` |
| Independent learners | One P/PA pair only; confidence interval deliberately unavailable | `paired_summary.json` |
| Command-line evaluation | Separate invocation completed on seven development layouts | `artifacts/cli-evaluation-check/` |
| Prototype configuration parity | Source prototype remains in sync with shared constants | `tools/sync_prototype.py --check` and parity tests |
| Syntax checks | Python modules compile; both frontend scripts and launcher parse | `compileall`, Node `--check`, zsh `-n` |
| Original ZIP | Download and preserved reference copy have identical SHA-256 | Hash in `DECISIONS.md` |

The small filtered pools accepted 14 of 60 training candidates and seven of 20 development candidates. These are integration pools, not the full registered study. The final range was not generated, viewed or evaluated.

## Browser checks

An isolated Chrome session opened `http://127.0.0.1:8512`. The app rendered its board, troop cards, overlays, resource/time counters and replay controls without page errors. Initial gameplay and replay checks had no console messages. The browser-verification workflow required interaction checks, not only a running server process.

Verified buying one tank, ending shopping, deploying at perimeter tile 7, waiting, resetting, displaying ranges, generating a heuristic replay, moving to its last frame, and generating a saved PPO smoke-checkpoint replay. Buying a tank left 175 resources; finishing shopping used zero ticks; deployment advanced the clock to 290 ticks remaining.

The heuristic replay's final frame matched its Python record. That is one demonstration episode, not a general performance estimate. Desktop and 390-pixel-wide views were inspected; the component's scroll width matched its 358-pixel viewport width with no horizontal overflow. The mobile sidebar can be collapsed.

Screenshots are saved in `artifacts/browser-*.png`. Server-side app tests also cover manual initial state, labelled heuristic replay, experiment results and saved replay modes. The app remains a local development service, not a deployed or hardened public application.

## Limits of this verification

- No 50,000-step learning-quality gate, 250,000-requested-step curriculum study, five/ten-seed experiment or authorised final evaluation was run.
- Full training and final-protocol code exists, but the long execution paths have not been empirically completed. Their access and matching guards have unit coverage.
- Short training cannot establish convergence, good strategy, stable reward learning or augmentation benefit. The single-pair development CSV must not become a final-results table.
- SB3 emitted a wrapper-type warning because training used `SubprocVecEnv` and the development callback used `DummyVecEnv`. Both use the same `SiegeEnv` observation/action contract. The short run did not reach the 25,000-step callback interval; separate development evaluation did execute.
- Notebook kernels emitted a local TCP transport warning. They were launched for local verification and explicitly shut down after execution.
- Reloading after the results chart emitted Vega empty-extent warnings while the old chart was removed. The populated chart rendered correctly and the final manual screen had no page errors.

Re-run checks after changing the environment or experiment code. Pool source checksums force regeneration if mechanics or validation change. Registration freezes source/pool choices before full training; it is not evidence that a study has already been performed.

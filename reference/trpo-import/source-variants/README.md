# Base Siege

**A reinforcement-learning commander for army selection and troop deployment.**

Developed as a capstone project for **43008 Reinforcement Learning**, Master of Artificial Intelligence, **University of Technology Sydney (UTS)**. Project contact: **Vaibhav Bairathi**.

Base Siege is a custom base-attack simulator. A trained agent spends a resource budget on troops, chooses their deployment positions and timing, and tries to destroy a defended base. Troop movement, targeting and defensive combat follow programmed rules.

The project includes the simulator, training and evaluation workflows, trained checkpoints, experiment records, and an interactive Streamlit application.

## Project status

- Version 1 (V1): pilot, twenty main training runs, development diagnostics, and final evaluations completed.
- Version 2 (V2): twenty independent training runs at **4,194,304 transitions per model**, followed by development analysis and a fresh final evaluation, completed.
- Application: V2 replay integration, smooth manual combat, custom-base editing, and interface cleanup implemented.
- Current application modes: **Manual play**, **Policy replay**, **Base editor**, and **Saved replay**.
- Experiment results are stored in files; the application no longer has an Experiment Results mode.

The setup commands below load existing checkpoints. They do not start training or rerun the final evaluation.

## V2 final results

The final evaluation used **235 accepted layouts** and **114,680 attacks**, covering all twenty trained models and the heuristic baseline. Learned agents used **DS: deterministic purchasing and sampled deployment**.

| Agent | Mean destruction | Full-clear rate | Town Hall destroyed |
|---|---:|---:|---:|
| P — no symmetry augmentation | 92.96% | 74.26% | 89.68% |
| PA — symmetry augmentation | 93.53% | 76.72% | 90.54% |
| Heuristic baseline | 56.90% | 2.29% | 60.00% |

**Destruction is the fraction of total building value destroyed.** It is not classification accuracy, a win rate, or a simple count of buildings. Full clear means every scoring building was destroyed; walls have no score.

The primary paired comparison was **PA minus P: +0.57 percentage points**, with a **95% bootstrap interval of +0.11 to +1.06 points**. This supports a small average augmentation benefit under this evaluation protocol; PA performed better in six of the ten matched learner pairs.

Intervals resample independent learner seeds and are conditional on the fixed layouts, views and action labels. They do not measure uncertainty over every possible custom base.

### Final evaluation design

- 800 candidate layouts: seeds 11200–11399 across four structures.
- Original structural checks and heuristic difficulty filter retained; 235 candidates accepted, with no replacement candidates.
- 177 accepted layouts from the three training structures and 58 `layered_core` transfer layouts.
- Eight rotated/reflected coordinate views per layout.
- Three action-seed labels per learned model and view: 41001, 41002 and 41003.
- 56,400 attacks for P, 56,400 for PA, and 1,880 heuristic attacks.
- All twenty final checkpoints included; no selection of the best learner using final results.
- Protocol frozen after development analysis and before generating/evaluating the fresh final pool. It was **not preregistered before training**.

On the transfer layouts, mean destruction was **91.63% for P**, **91.99% for PA**, and **51.12% for the heuristic**. The transfer-only PA–P interval included zero, so that subset did not demonstrate a clear additional augmentation benefit.

Measured final-evaluation execution time was approximately **2 hours 21 minutes 37 seconds**, including pool construction, attack execution and export; it excludes pauses and preflight checks.

### Result files

The values above come from the completed evaluation records:

- [Condition summaries](artifacts/version2-final-evaluation/condition_summary.csv)
- [Paired comparisons and intervals](artifacts/version2-final-evaluation/comparisons.csv)
- [Individual model summaries](artifacts/version2-final-evaluation/model_summary.csv)
- [Individual attacks](artifacts/version2-final-evaluation/episodes.csv)
- [Frozen evaluation decision](artifacts/version2-final-evaluation/freeze.json)
- [Completion record and output hashes](artifacts/version2-final-evaluation/complete.json)
- [Measured execution timing](artifacts/version2-final-evaluation/timing_summary.json)
- [Final evaluation protocol](v2_final/protocol.json)

## Algorithm and training

We use **Stable-Baselines3 Contrib MaskablePPO**, with a custom convolutional feature extractor and legal-action masks. The network receives a spatial grid and a short summary vector. The mask prevents invalid purchases and deployments.

| Condition | Meaning |
|---|---|
| **P** | PPO trained without symmetry augmentation |
| **PA** | The same PPO setup, trained with randomly rotated/reflected coordinate views |

Each condition contains **ten independently trained models**, labelled **P-0 through P-9** and **PA-0 through PA-9**. Learner seeds are 0–9. These are repeated training runs of two experimental conditions, not ten algorithms or an ensemble. Each replay uses one model.

| Study stage | Models | Transitions per model | Placement-only phase | Joint purchasing/deployment phase |
|---|---:|---:|---:|---:|
| V1 pilot, learner seed 100 | 2 | 57,344 | Separate pilot schedule | Separate pilot schedule |
| V1 main, learner seeds 0–9 | 20 | 262,144 | 32,768 | 229,376 |
| V2, learner seeds 0–9 | 20 | 4,194,304 | 32,768 | 4,161,536 |

V2 contains **83,886,080 training transitions** across its twenty models. A transition is one agent decision; an episode is one complete attack and contains multiple decisions.

V2 models started independently from scratch. Seeds 0–2 formed the six-model screening cohort; seeds 3–9 formed the fourteen-model replication cohort. V2 retained the frozen simulator, reward, observation, action masks, network and PPO hyperparameters, while extending training and investigating action selection.

### Action-selection modes

| Mode | Purchasing | Deployment |
|---|---|---|
| **DS** — V2 application default | Highest-probability legal action | Sample from legal-action probabilities |
| **Sampled / SS** | Sample | Sample |
| **Deterministic / DD** | Highest-probability legal action | Highest-probability legal action |

DS was selected using development evidence. The final evaluation then used the frozen DS procedure.

### V1 results are a separate experiment

V1 final mean destruction was **85.95% for sampled P**, **85.45% for sampled PA**, **55.10% for the heuristic**, **43.53% for deterministic P**, and **44.59% for deterministic PA**.

V1's sampled PA–P difference was approximately **−0.50 percentage points**, with a 95% interval of **−2.06 to +1.06**; that study did not establish an additional mean augmentation benefit. The sampled extension used existing checkpoints and was documented after development diagnostics, before its final evaluation.

V1 and V2 used different final layouts and acting procedures. Their headline scores are not a controlled estimate of the effect of longer training alone.

## Environment

| Setting | Value |
|---|---|
| Grid | 16 × 16 |
| Starting resource budget | 200 |
| Maximum purchased army | 25 troops |
| Troop types | Tank, melee, ranged, breaker |
| Perimeter deployment positions | 60 |
| Discrete actions | 246 |
| Observation | Eight-channel grid plus nine summary values |
| RL battle decision | Advances up to ten simulator ticks |
| Battle limit | 300 ticks |
| Base structures | `closed_ring`, `ns_gates`, `ew_gates`, `layered_core` |

Purchasing decisions do not advance battle time. The commander learns army composition and deployment; individual troops do not have their own learned policies.

## Installation and launch

### Prerequisites

Install **Git**, **Git LFS**, and **uv**. The project requires **Python 3.12**. Dependency versions are recorded in [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock).

Official installation instructions: [Git](https://git-scm.com/downloads), [Git LFS](https://git-lfs.com/), and [uv](https://docs.astral.sh/uv/getting-started/installation/).

Run the following commands in a terminal. Start in the directory where you want to download the project:

```bash
git lfs install
git clone https://github.com/vaibhavbai1/base-siege.git
cd base-siege
git lfs pull
uv python install 3.12
uv sync --frozen --python 3.12
```

Git LFS downloads the actual checkpoints and other LFS-managed files. A small text pointer in place of a checkpoint ZIP is not a usable model. Use the clone/LFS workflow for the team handover.

### Verify the twenty V2 checkpoints

From the project root, where `pyproject.toml` is located:

```bash
uv run --frozen python -m ui.v2_models check
```

Expected output includes `"status": "ready"` and `"verified_v2_models": 20`. This checks checkpoint hashes, manifests and frozen scientific source compatibility; it does not train a model.

If files are missing, run `git lfs pull` again and repeat the check. For a checksum or source mismatch, keep the error and investigate the changed file rather than changing the expected hashes.

### Launch on your computer

```bash
uv run --frozen streamlit run ui/app.py --server.address 127.0.0.1 --server.port 8501
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501). Keep the terminal running while using the app; press **Ctrl+C** to stop it.

### Launch on SageMaker

In a SageMaker Terminal, first change into your project folder. For the existing project workspace:

```bash
cd ~/Base_Siege_Smooth_Demo/base-siege-assignment3
uv run --frozen streamlit run ui/app.py --server.address 0.0.0.0 --server.port 8501
```

Open port 8501 through your SageMaker environment's application preview or proxy. The exact URL depends on the SageMaker workspace. If you used a fresh Git clone, change into that clone's folder instead.

## Using the application

- **Manual play:** purchase troops, start the battle, select a troop type and deploy around the green perimeter. Use the pause, speed and view controls during combat.
- **Policy replay:** select `Saved Maskable PPO`, model version `V2`, condition `P` or `PA`, and learner seed 0–9. Leave action selection on DS for the evaluated V2 procedure. Click **Generate replay**, then **Play**.
- **Base editor:** edit a layout, save it, and use it for manual play or policy replay. Custom bases can be exported/imported as JSON.
- **Saved replay:** inspect saved attacks. Generated policy replays can also be downloaded as JSON or standalone HTML.

### The different seeds and views

| Control | What it changes |
|---|---|
| Learner seed | Which independently trained model is loaded |
| Base seed | Generated base geometry, together with the selected structure |
| Action seed | Random choices during sampled action selection; it does not retrain the agent |
| Agent view | One of eight rotated/reflected coordinate views presented to the policy |

The V2 action-seed default is 41001. The same model, base, view, action seed and software reproduce the same attack. Playback shows the original base orientation. Because the view is included in the acting-seed calculation, changing the view in a sampled replay changes both the observed orientation and the random sequence.

Manual combat runs smoothly in the browser. Training and learned-policy evaluation use the Python simulator. Manual deployment timing differs from the RL interface, so manual play is not a controlled human-versus-agent benchmark.

## Repository guide

| Path | Contents |
|---|---|
| `env/` | Simulator, configuration, observations, actions and symmetry transforms |
| `agents/` | Programmed heuristic and random baselines |
| `train/` | Feature extractor, PPO training, evaluation and recovery |
| `experiments/` | Original experiment workflow, pools and provenance |
| `version2/` | Six-model V2 screening workflow, seeds 0–2 |
| `v2_replication/` | Fourteen-model replication workflow, seeds 3–9 |
| `v2_development/`, `v2_phases/`, `v2_replays/` | Development evaluation and action/behaviour diagnostics |
| `v2_cohort_evaluation/` | Combined twenty-model development evaluation |
| `v2_final/` | Fresh final evaluation code and frozen protocol |
| `version2-records/` | V2 workflow records |
| `runs/version2-longer-control/` | Screening checkpoints, manifests and training logs |
| `runs/version2-seed-replication/` | Replication checkpoints, manifests and training logs |
| `artifacts/version2-*/` | Saved development and final evaluation evidence |
| `ui/` | Streamlit application, manual combat, editor and policy replay |
| `ui/v2_checkpoints.json` | Exact approved V2 checkpoint paths and hashes |
| `notebooks/` | Original teaching and experiment notebooks |
| `tests/` | Simulator, workflow and application checks |
| `tools/` | Validation and deployment utilities |
| `docs/` | Design documents, runbooks and historical study documentation |

Approved V2 application checkpoints are the 4,194,304-step milestone ZIPs for screening seeds 0–2 and the final `model.zip` files for replication seeds 3–9. The registry supplies their exact paths; do not rename them or substitute a different milestone.

Some older documents and package READMEs describe plans at the time they were written. For completed V2 outcomes, use the saved completion records, frozen protocols and result tables linked above.

## Limitations and future work

- The known inaccessible-target movement limitation remains in the frozen simulator. No reachable-target fallback was added.
- The compact observation does not expose every internal simulator variable.
- Final results apply to the accepted, heuristic-filtered layouts and the recorded evaluation procedure; arbitrary custom bases may be harder.
- `layered_core` was excluded from V2 training and development, but had been examined in the earlier project.
- An individual replay is an illustration, not an aggregate performance estimate.
- The application and checkpoints support demonstration without new training. Further training or changes to rewards, observations, movement or network design should be recorded as a new experiment version, preserving the completed V1/V2 evidence.

## Acknowledgements

The PPO implementation is provided by [Stable-Baselines3 Contrib MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html). The project also uses [Gymnasium](https://gymnasium.farama.org/), [PyTorch](https://pytorch.org/) and [Streamlit](https://streamlit.io/).

Our contribution is the custom task and simulator integration, representations, symmetry augmentation, experiment and recovery workflows, analysis, and interactive application. We do not claim to have invented PPO.

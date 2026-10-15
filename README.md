# Base Siege

**A custom reinforcement-learning environment for army selection and troop deployment, with MaskablePPO and masked Double DQN agents.**

Developed as a capstone project for **43008 Reinforcement Learning**, Master of Artificial Intelligence, **University of Technology Sydney (UTS)**. Project contact: **Vaibhav Bairathi**.

Base Siege is a custom base-attack simulator. A trained agent spends a resource budget on troops, chooses their deployment positions and timing, and tries to destroy a defended base. Troop movement, targeting and defensive combat follow programmed rules.

The project includes the simulator, training and evaluation workflows, trained checkpoints, experiment records, and an interactive Streamlit application.

## Project status

- Version 1 (V1): pilot, twenty main training runs, development diagnostics, and final evaluations completed.
- Version 2 (V2): twenty independent training runs at **4,194,304 transitions per model**, followed by development analysis and a fresh final evaluation, completed.
- Masked Double DQN (DDQN): two seed-100 pilot runs, development checkpoint selection, and evaluation on the existing V2 benchmark completed. No multi-seed DDQN cohort was run.
- Application: V1/V2 PPO and selected DDQN replay, smooth manual combat, custom-base editing, and saved replay supported in the local/SageMaker app.
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
- [Measured execution attempt records](artifacts/version2-final-evaluation/attempts.jsonl)
- [Final evaluation protocol](v2_final/protocol.json)

## PPO algorithm and training

The V1 and V2 studies use **Stable-Baselines3 Contrib MaskablePPO**, with a custom convolutional feature extractor and legal-action masks. The network receives a spatial grid and a short summary vector. The mask prevents invalid purchases and deployments.

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

### PPO action-selection modes

| Mode | Purchasing | Deployment and waiting |
|---|---|---|
| **DS** — V2 application default | Highest-probability legal action | Sample from legal-action probabilities |
| **Sampled / SS** | Sample | Sample |
| **Deterministic / DD** | Highest-probability legal action | Highest-probability legal action |

DS was selected using development evidence. The final evaluation then used the frozen DS procedure. The four-way development diagnostic also tested **SD** (sampled purchasing, deterministic deployment/waiting). DD, DS, SD and SS are ways of using an existing PPO checkpoint, not separately trained algorithms; the application offers DD, DS and SS.

### V1 results are a separate experiment

V1 final mean destruction was **85.95% for sampled P**, **85.45% for sampled PA**, **55.10% for the heuristic**, **43.53% for deterministic P**, and **44.59% for deterministic PA**.

V1's sampled PA–P difference was approximately **−0.50 percentage points**, with a 95% interval of **−2.06 to +1.06**; that study did not establish an additional mean augmentation benefit. The sampled extension used existing checkpoints and was documented after development diagnostics, before its final evaluation.

V1 and V2 used different final layouts and acting procedures. Their headline scores are not a controlled estimate of the effect of longer training alone.

## Masked Double DQN experiment

**DDQN means Double Deep Q-Network.** This is a custom PyTorch implementation, not an SB3 DQN or SB3 Contrib MaskableDQN class. It reuses the existing simulator, observation, reward, legal-action masks and CNN feature extractor, and trains new network weights from scratch. It does not convert or continue a PPO checkpoint.

PPO learns probabilities for choosing actions. DDQN learns a value for each action: how much future reward it expects from taking that action. Its online network chooses the next legal action for the learning target, and a separate target-network copy evaluates that choice. These two networks belong to one learner; they are not two independent training runs.

Masks are applied to random exploration, greedy action selection and next-state selection in the Double DQN target. The DDQN condition has **no symmetry augmentation**. Selecting a rotated/reflected view during replay does not mean the model was trained with augmentation.

### Development and checkpoint selection

Two pilot runs used **the same learner seed, 100**, with different learning rates. Both trained for **4,194,304 transitions**, consuming **8,388,608 transitions in total**. Full development checks used 156 layouts and four views (0, 1, 4, 5): 624 attacks per checkpoint.

| Checkpoint transitions | Learning rate 0.0001: mean destruction | Learning rate 0.00003: mean destruction |
|---|---:|---:|
| 1,048,576 | 61.87% | 58.48% |
| 2,097,152 | 59.33% | **68.41% — selected** |
| 4,194,304 | 48.14% | 51.85% |

Development evidence selected the **0.00003 learning-rate checkpoint at 2,097,152 transitions**. Longer training did not improve these recorded development scores. The selected checkpoint contains about two million transitions of learning, but finding it involved both complete pilot runs; its checkpoint budget is not the total tuning cost.

Sources: original-rate full-development summaries at [1,048,576](artifacts/ddqn/pilot-seed100-full-dev-1048576/summary.json), [2,097,152](artifacts/ddqn/pilot-seed100-full-dev-2097152/summary.json) and [4,194,304](artifacts/ddqn/pilot-seed100-full-dev-4194304/summary.json); [lower-rate development records](artifacts/ddqn/pilot-lr3e5-seed100/).

### Selected DDQN on the existing V2 benchmark

The selected checkpoint was frozen before this evaluation. It completed **1,880 deterministic attacks: 235 layouts × eight views**, using the highest-valued legal action for purchasing, deployment and waiting. Equal Q values are resolved by the lowest action ID.

| Agent | Independent learners represented | Evaluated checkpoint transitions | Acting procedure | Mean destruction | Full-clear rate |
|---|---:|---:|---|---:|---:|
| V2 P | 10 | 4,194,304 each | PPO DS | 92.96% | 74.26% |
| V2 PA | 10 | 4,194,304 each | PPO DS | 93.53% | 76.72% |
| Selected DDQN pilot | 1 | 2,097,152 | Greedy legal actions | 54.41% | 21.70% |
| Heuristic | No trained learner | Not applicable | Programmed rules | 56.90% | 2.29% |

PPO and heuristic values are reused from the completed V2 evaluation on the same layouts and views. PPO includes three action-seed repetitions for each learner/layout/view; deterministic DDQN needs one attack per layout/view.

This selected DDQN policy had lower mean destruction than the heuristic but more full clears. Its performance differed sharply by layout group:

| Layout group | Layouts | DDQN attacks | Mean destruction | Full-clear rate |
|---|---:|---:|---:|---:|
| Familiar structure families | 177 | 1,416 | 68.57% | 28.74% |
| `layered_core` transfer family | 58 | 464 | 11.23% | 0.22% |

**Comparison limits:** this is one development-selected pilot, not a ten-learner DDQN study. No across-DDQN-learner confidence interval can be estimated from it. The V2 benchmark and PPO outcomes were already known before the DDQN comparison, so this is an extension on an existing benchmark, not a newly unseen final test. Different checkpoint budgets, inference procedures, tuning histories and learner counts prevent attributing the entire performance gap to the algorithm alone. The checkpoint was not replaced based on these benchmark outcomes.

### DDQN evidence

- [Benchmark summary and limitations](artifacts/ddqn/selected-lr3e5-seed100-v2-benchmark/summary.json)
- [Condition and layout-group results](artifacts/ddqn/selected-lr3e5-seed100-v2-benchmark/condition_summary.csv)
- [Individual attacks](artifacts/ddqn/selected-lr3e5-seed100-v2-benchmark/episodes.csv)
- [Comparisons with existing PPO and heuristic records](artifacts/ddqn/selected-lr3e5-seed100-v2-benchmark/comparisons.csv)
- [Frozen checkpoint and evaluation decision](artifacts/ddqn/selected-lr3e5-seed100-v2-benchmark/freeze.json)
- [Completion receipt and output hashes](artifacts/ddqn/selected-lr3e5-seed100-v2-benchmark/complete.json)
- [Selected model, training contract and checksum sidecar](artifacts/ddqn/selected-lr3e5-seed100-v2-benchmark/selected_model/)
- [Original pilot settings and progress](runs/ddqn/pilot-seed100/) and [lower-rate pilot settings and progress](runs/ddqn/pilot-lr3e5-seed100/)

The benchmark's measured episode loops total **97.84 seconds**. This excludes preflight checks, checkpoint loading, comparison and export; it is not an end-to-end evaluation time or a controlled speed comparison with PPO.

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

Purchasing decisions do not advance battle time. The 246 actions comprise four purchases, one finish-shopping action, 240 deployment choices (four troop types × 60 positions), and one wait action. Each deployment action places **one troop**, then advances combat by up to ten ticks; already deployed troops act together during those ticks. The commander learns army composition and deployment; individual troops do not have their own learned policies.

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

### Verify the selected DDQN checkpoint

From the same project root:

```bash
uv run --frozen python -m ui.ddqn_models check
```

Expected output begins with `DDQN ready: seed 100, 2,097,152 transitions, greedy legal acting.` The check verifies the selected policy, training contract, checksum sidecar, scientific source and runtime compatibility. It does not train or evaluate a batch of attacks.

The app prefers the frozen copy under `artifacts/ddqn/selected-lr3e5-seed100-v2-benchmark/selected_model/`. It can also use the identical selected checkpoint in the original lower-rate run. The original run's other checkpoints are not required to demonstrate DDQN.

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
- **PPO policy replay:** select `Saved Maskable PPO`, model version `V2`, condition `P` or `PA`, and learner seed 0–9. Leave action selection on DS for the evaluated V2 procedure. Click **Generate replay**, then **Play**. Earlier PPO models remain available through the model-version selector.
- **DDQN policy replay:** select `Saved Masked DDQN`, choose a base and agent view, then click **Generate replay** and **Play**. The app loads the selected seed-100, 2,097,152-transition checkpoint automatically. It always uses greedy legal actions, so there is no action-sampling seed or learner selection.
- **Base editor:** edit a layout, save it, and use it for manual play or policy replay. Custom bases can be exported/imported as JSON.
- **Saved replay:** inspect saved attacks. Generated policy replays can also be downloaded as JSON or standalone HTML.

DDQN supports custom bases, smooth playback, and JSON/standalone HTML replay downloads through the existing replay interface. This integration is for the local/SageMaker application; the separate public-demo configuration does not offer DDQN.

### The different seeds and views

| Control | What it changes |
|---|---|
| Learner seed | Which independently trained model is loaded |
| Base seed | Generated base geometry, together with the selected structure |
| Action seed | Random choices during sampled action selection; it does not retrain the agent |
| Agent view | One of eight rotated/reflected coordinate views presented to the policy |

For PPO, the V2 action-seed default is 41001. The same model, base, view, action seed and software reproduce the same attack. Playback shows the original base orientation. Because the view is included in the acting-seed calculation, changing the view in a sampled replay changes both the observed orientation and the random sequence.

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
| `v2_final/` | Fresh final PPO evaluation code and frozen protocol |
| `ddqn_experiment/` | Custom masked Double DQN training, development evaluation and replay |
| `ddqn_final_benchmark/` | Selected DDQN evaluation on the existing V2 benchmark |
| `runs/ddqn/` | Compact pilot training contracts, status, progress and attempt records |
| `artifacts/ddqn/` | Development results, completed benchmark evidence and selected inference checkpoint |
| `version2-records/` | V2 workflow records |
| `runs/version2-longer-control/` | Screening checkpoints, manifests and training logs |
| `runs/version2-seed-replication/` | Replication checkpoints, manifests and training logs |
| `artifacts/version2-*/` | Saved development and final evaluation evidence |
| `ui/` | Streamlit application, manual combat, editor and policy replay |
| `ui/v2_checkpoints.json` | Exact approved V2 checkpoint paths and hashes |
| `ui/ddqn_models.py` | Selected DDQN checkpoint identity, validation and replay adapter |
| `notebooks/` | Original teaching and experiment notebooks |
| `tests/` | Simulator, workflow and application checks |
| `tools/` | Validation and deployment utilities |
| `docs/` | Design documents, runbooks and historical study documentation |

Approved V2 application checkpoints are the 4,194,304-step milestone ZIPs for screening seeds 0–2 and the final `model.zip` files for replication seeds 3–9. The registry supplies their exact paths; do not rename them or substitute a different milestone.

### What the DDQN upload contains

GitHub contains the DDQN source, compact training/evaluation evidence, and the selected inference checkpoint managed through Git LFS. Full DDQN recovery buffers and raw training episode logs remain on SageMaker. This compact GitHub subset supports replay and result inspection; it is **not sufficient to resume those training runs**. Preserve the complete SageMaker run folders for recovery.

Some older documents and package READMEs describe plans at the time they were written. In particular, the [original DDQN runbook](ddqn_experiment/README.md) predates the completed benchmark and UI selector: its proposed ten-learner study was not performed, and its statement that no DDQN selector exists is historical. Use this README and the linked completion records for current status. Training commands in historical runbooks are not required to launch the application.

## Limitations and future work

- The known inaccessible-target movement limitation remains in the frozen simulator. No reachable-target fallback was added.
- The compact observation does not expose every internal simulator variable.
- Final results apply to the accepted, heuristic-filtered layouts and the recorded evaluation procedure; arbitrary custom bases may be harder.
- `layered_core` was excluded from V2 training and development, but had been examined in the earlier project.
- DDQN results represent one selected learner and showed weak transfer to `layered_core`; more attacks from that same checkpoint do not replace independent training seeds.
- An individual replay is an illustration, not an aggregate performance estimate.
- The application and checkpoints support demonstration without new training. Further training or changes to rewards, observations, movement or network design should be recorded as a new experiment version, preserving the completed V1/V2 evidence.

## Acknowledgements

The PPO implementation is provided by [Stable-Baselines3 Contrib MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html). The project also uses [Gymnasium](https://gymnasium.farama.org/), [PyTorch](https://pytorch.org/) and [Streamlit](https://streamlit.io/).

Our contribution is the custom task and simulator integration, representations, symmetry augmentation, masked Double DQN implementation, experiment and recovery workflows, analysis, and interactive application. PPO and Double DQN are established algorithms; we do not claim to have invented them.

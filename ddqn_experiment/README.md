# Masked Double DQN for Base Siege

This is a new algorithm experiment alongside V1 and V2 MaskablePPO. It trains a
new network from scratch. It does not load a PPO model or change the simulator.
It is a no-augmentation condition, intended initially for comparison with P.

Start with the pilot below. A successful software test is not evidence that the
agent has learned a strong strategy. Review pilot learning and development
results before committing to ten long main runs.

## 1 Install the add-on

In SageMaker JupyterLab, open your existing `base-siege` project folder in the
file browser. Upload `Base_Siege_Masked_Double_DQN_Addon.zip` into that folder.
Right-click the project folder and choose **Open in Terminal**. All commands
below run in that terminal, at the repository root containing `pyproject.toml`,
`env`, `train`, `bases`, and `version2`.

Check your location, extract, and install:

```bash
pwd
ls pyproject.toml env/siege_env.py bases/train-v2.json bases/dev-v2.json
python -m zipfile -e Base_Siege_Masked_Double_DQN_Addon.zip .
python base_siege_ddqn_addon/install.py --repo .
uv run --frozen python -m ddqn_experiment check
uv run --frozen python -m pytest ddqn_experiment/tests -q
```

The installer checks the scientific source files against the inspected version
before adding the `ddqn_experiment` folder. It does not overwrite existing
source, results, checkpoints, dependencies or the handbook. An identical
installation is a no-op. If it reports a source mismatch, keep the message and
resolve the difference; do not bypass it or regenerate the frozen pools.

The runtime check should report 8,118 training layouts and 156 development
layouts. The existing `uv.lock` supplies the dependencies; no extra library is
required. The tests include very short training and development checks in
temporary folders. They are not research results.

## 2 Run the pilot

Run this in the **same SageMaker terminal at the repository root**:

```bash
uv run --frozen python -m ddqn_experiment train --out runs/ddqn/pilot-seed100 --seed 100 --mode pilot --until 262144
```

This runs **262,144 transitions**, not 262,144 episodes. It includes 32,768
placement-only transitions, followed by 229,376 joint purchasing/deployment
transitions. Pilot seed 100 is separate from main learner seeds 0–9.

The declared maximum schedule is 4,194,304 transitions. `--until 262144` is an
absolute pause point within that schedule. It does not shorten or reset the
exploration schedule. This makes an extension of the same pilot consistent.
The pilot ends with status `paused`; that is expected, not an error.

Progress appears every 4,096 transitions. `Saved` confirms a recovery checkpoint
has been committed. CPU is the default. The small CNN does not guarantee that a
GPU will make the complete simulator loop faster. No full-training duration is
claimed before measuring it on your SageMaker instance.

## 3 Pause, inspect and resume

Open a **second terminal in the same repository root** to inspect saved progress:

```bash
uv run --frozen python -m ddqn_experiment status --run runs/ddqn/pilot-seed100
```

`last_committed` is the saved transition count. It can lag the running process
between checkpoints. `last_exit` describes the previous clean exit, if any;
it is not a live running-status indicator.

For a controlled pause, run in the second terminal:

```bash
uv run --frozen python -m ddqn_experiment pause --run runs/ddqn/pilot-seed100
```

Wait for the training terminal to print `Saved` and exit before stopping the
SageMaker application. Ctrl+C also requests a controlled pause. A forced process
kill or interrupted instance can lose work after the last committed checkpoint.

To resume toward the original pilot pause point, run at the repository root:

```bash
uv run --frozen python -m ddqn_experiment train --out runs/ddqn/pilot-seed100 --seed 100 --mode pilot --until 262144 --resume
```

After reviewing the pilot, this command extends the **same pilot** to 1,048,576
total transitions, rather than adding 1,048,576 more:

```bash
uv run --frozen python -m ddqn_experiment train --out runs/ddqn/pilot-seed100 --seed 100 --mode pilot --until 1048576 --resume
```

Keep the same code, dependencies, device, seed, settings and pool contents when
resuming. Changes are rejected rather than silently creating a different run.
Keep the entire run directory on persistent SageMaker storage. Checkpoints do
not survive deliberate deletion of that storage. A policy-only `.pt` file is
enough for inference but is not a full training recovery file.

## 4 Evaluate the pilot on development layouts

After the pilot stops, run these commands at the **repository root**:

```bash
uv run --frozen python -m ddqn_experiment evaluate --run runs/ddqn/pilot-seed100 --steps 262144 --out artifacts/ddqn/pilot-seed100-dev-262144 --panel --views 0 1 4 5
```

This uses the existing, fixed 12-layout development panel and four coordinate
views: 48 episodes in total. It does not inspect final evaluation pools. The
policy selects the highest-valued legal action. Ties use the lowest action ID.
No action-sampling repetitions are needed for this deterministic policy.

The output folder contains:

- `summary.json`: destruction, full-clear rate, Town Hall rate, return and spending.
- `episodes.csv`: each layout/view result; destruction is stored as a fraction.
- `contract.json`: exact policy identity, pool, views, source and runtime.
- `progress.sqlite3`: committed progress for interrupted evaluation.

To resume an interrupted evaluation, repeat the identical command with `--resume`.
For example:

```bash
uv run --frozen python -m ddqn_experiment evaluate --run runs/ddqn/pilot-seed100 --steps 262144 --out artifacts/ddqn/pilot-seed100-dev-262144 --panel --views 0 1 4 5 --resume
```

For a later full development check, omit `--panel` and use a **new output folder**.
That uses all 156 development layouts, or 624 episodes with four views. This is
still development evidence, not a final test result.

## 5 Record one attack for inspection

Run at the **repository root**, after the pilot has reached 262,144 transitions:

```bash
uv run --frozen python -m ddqn_experiment replay --run runs/ddqn/pilot-seed100 --steps 262144 --out artifacts/ddqn/pilot_seed100_replay.json --index 0 --view 0
```

This writes the existing replay schema, including tick frames, and records the
DDQN identity in the JSON. `--index` selects a development layout from the
motif/seed-sorted pool. In the local application, **Saved replay** discovers
files under `artifacts` with `replay` in the filename. The public deployment
uses its published asset list instead. No DDQN model selector has been added
to the application's Policy replay mode. The exporter and replay loader are
tested; browser display was not exercised in this implementation check.

## 6 Settings and what they mean

These are initial settings for a pilot, not tuned or proven optimal settings.

| Setting | Value | Meaning |
|---|---:|---|
| Algorithm | Masked Double DQN | Online network selects; target network evaluates |
| Training condition | No augmentation | Suitable first comparison with P |
| Maximum transitions | 4,194,304 | Same transition budget as one V2 PPO learner |
| Placement phase | 32,768 | Fixed heuristic army; learns deployment first |
| Environments | 1 | Sequential interaction; PPO used four environments |
| Features | Existing SiegeFeatures CNN | Same grid and nine summary inputs |
| Q head | 128 hidden units, 246 outputs | One value per discrete action |
| Learning rate | 0.0001 | Adam update size |
| Discount | 1.0 | Retains the existing undiscounted reward objective |
| Replay capacity | 50,000 transitions | Uniform replay sampling, with replacement |
| Batch size | 128 | Experiences per gradient update |
| Learning warm-up | 2,048 transitions per phase | Collect experience before updating |
| Update frequency | Every 4 transitions | One gradient update per update event |
| Target copy interval | 4,096 transitions | Hard copy from online to target network |
| Loss | Huber | Smooth L1 loss on predicted and target values |
| Gradient norm limit | 10 | Clips excessively large gradients |
| Placement exploration | 1.0 toward 0.1 | Chance of a random legal action |
| Joint exploration | 1.0 to 0.05 over 524,288 joint transitions | Restarts at shopping introduction |
| Recovery save interval | 65,536 transitions | Also saves at phase change and controlled exit |
| Full recovery retention | Two generations | Includes replay buffer, optimizer and simulator |

At the curriculum boundary, the episode resets as in the existing PPO trainer.
The DDQN replay buffer is cleared so fixed-army experiences do not dominate
the new joint phase. Learned weights and optimizer state are retained. Any
unfinished episode at this boundary is counted as a phase cutoff, not a
completed episode. The exact transition count remains recorded.

The two networks in Double DQN are an online network and a lagged target copy.
They are not an ensemble or two independent learners. Each seed trains one
independent agent. The network head differs from PPO's actor and critic heads;
the total parameter count is therefore not claimed to be identical.

Masks apply during random exploration, greedy selection and next-state action
selection in the learning target. Genuine terminal states do not bootstrap.
The 70-decision safety guard is treated as truncation and bootstraps from the
underlying continuation mask. The 300-tick game deadline remains a true terminal
outcome. No movement, targeting, reward or termination rule is changed.

Observations remain float32 in replay. At full capacity the array payload is
about 0.78 GiB, before networks, batches and checkpoint staging. Checkpoint
creation temporarily copies the buffer. Allow several GiB of free RAM and
several GiB of disk per active run. Ten retained run folders require much more
disk than one pilot. Do not push large recovery buffers to ordinary GitHub.

## 7 Files to keep and send for review

Keep the complete `runs/ddqn/pilot-seed100` folder for recovery. Its key files are:

- `contract.json`: immutable settings, source hashes, pool hashes and runtime.
- `LATEST.json`: authoritative committed checkpoint pointer.
- `checkpoints/`: two checksummed generations, including complete training state.
- `policies/`: compact policies at saved transition counts, with checksum sidecars.
- `episodes.jsonl`: completed training episodes and actual outcomes.
- `progress.jsonl`: transitions, optimizer updates, exploration and recent training scores.
- `status.json`: progress at the most recent clean exit.
- `attempts.jsonl`: start/resume attempts; `last_error.json` if an exception occurred.

For a compact review ZIP, run at the **repository root** after evaluation:

```bash
uv run --frozen python -m zipfile -c ddqn_pilot_review.zip runs/ddqn/pilot-seed100/contract.json runs/ddqn/pilot-seed100/status.json runs/ddqn/pilot-seed100/episodes.jsonl runs/ddqn/pilot-seed100/progress.jsonl artifacts/ddqn/pilot-seed100-dev-262144/summary.json artifacts/ddqn/pilot-seed100-dev-262144/episodes.csv
```

Share that small ZIP to review learning before scaling up. It deliberately
excludes large recovery files and is not a backup from which training can resume.
Training `loop_seconds` measures interaction, updates and loop bookkeeping; it
excludes checkpoint writes and process setup. Evaluation timing measures each
episode loop. These are measured durations, not archive-timestamp estimates,
and neither should be labelled total end-to-end wall time.

## 8 Main study after the pilot is reviewed

Freeze the chosen settings and evaluation design before main DDQN training.
The per-run contract is an implementation record, not evidence that this new
experiment was preregistered before the earlier PPO studies. Keep the original
PPO results frozen.

The proposed main study is ten fresh DDQN learners, seeds 0–9, each at 4,194,304
transitions. Do not promote the seed-100 pilot into the main cohort. Example
command for one main learner, to use only after the settings are selected:

```bash
uv run --frozen python -m ddqn_experiment train --out runs/ddqn/main-seed0 --seed 0 --mode main
```

Run from the repository root. Change both the seed and output folder for each
independent learner. Do not launch ten runs in parallel without checking the
available memory, CPU and disk. The package does not start them automatically.

If changing a pilot hyperparameter, create a JSON file with Settings overrides
and use `--settings path/to/file.json` with a **new run folder**. Supply the same
file on resume. Changes to exploration duration, discount or the curriculum
change the experimental setup and must be reported.

An equal transition budget is not an equal wall-time or gradient-update budget.
DDQN uses one environment and experience replay; PPO used vectorized rollouts.
Compare algorithms on the same evaluation layouts/views with explicitly stated
acting rules. PPO DS samples deployments; DDQN greedy acting does not. A later
comparison can include PPO DD as an additional diagnostic, but should retain
the declared primary PPO recipe. Do not compare this development panel directly
with the published V2 final average as if the scenarios were identical.

No final evaluation command is included. A new comparison protocol and suitable
held-out evaluation must be settled after development choices are frozen.
Repeated learner runs, not the number of attacks from one learner, support
claims about training variability. Results need not beat PPO to be informative.

## References

- Double DQN paper: https://arxiv.org/abs/1509.06461
- Gymnasium termination/truncation guidance: https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
- Existing project: https://github.com/vaibhavbai1/base-siege

The masked Double DQN update is implemented in this add-on. It is not an
unmodified SB3 DQN or a separate SB3 Contrib MaskableDQN class.

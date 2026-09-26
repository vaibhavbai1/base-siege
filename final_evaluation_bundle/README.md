# Base Siege: final evaluation package

Prepared 22 September 2026. Run from your existing SageMaker project root.
This package evaluates the twenty trained models. It does not train new models.

## 1. Upload and extract

Upload `Base_Siege_Final_Evaluation.zip` into the folder containing `pyproject.toml`,
`runs`, `bases`, and `artifacts`. Then open a terminal:

```bash
cd ~/Base_Siege_Smooth_Demo/base-siege-assignment3
unzip Base_Siege_Final_Evaluation.zip
```

The ZIP adds only `final_evaluation_bundle/`. Keep its three Python files together.
It does not replace `env`, `agents`, `train`, or `experiments` source files.
Use the existing environment; no new dependencies are required.

## 2. Prepare the amendment and freeze

```bash
uv run python final_evaluation_bundle/run_final_evaluation.py prepare
```

Wait for the `Prepared ...` message, or `Already prepared with identical inputs`.
Do not proceed after an error.

This command checks all twenty models, the original registration, both training
and development pools, the original development evaluation, the sampling
diagnostic, and the completed sampled development evaluation. It creates:

- `experiments/final-freeze-v2.json`: the fixed set of model checkpoints.
- `experiments/final-sampling-amendment-v1.json`: the actual UTC preparation time,
  development evidence, inference settings, comparisons, and reporting rules.

Preparation does not generate or evaluate final bases. If a final run has already
started without this amendment, the command stops: do not remove its marker or
rename its results to get past the check. Share the error for review.

## 3. Run both final evaluations

The following runs in the background and writes a log. It can survive a terminal
disconnect, but not shutdown or termination of the underlying instance.

```bash
mkdir -p artifacts
nohup uv run python final_evaluation_bundle/run_final_evaluation.py run --confirm-final > artifacts/final-evaluation-v1.log 2>&1 < /dev/null &
```

Run this once. The `--confirm-final` flag authorizes use of the reserved final
partition under the prepared freeze and amendment.

The script performs these phases in sequence:

1. Call the unchanged original final protocol. Generate the registered candidate
   window, filter bases, then evaluate deterministic P, deterministic PA, the
   heuristic, and the ten random-policy action seeds.
2. Evaluate sampled P and PA on exactly the same accepted bases. Use all twenty
   fixed-budget models, all eight orientations, and action seeds 0, 1, and 2.
3. Export separate results and comparisons, including familiar and transfer
   structures. The original deterministic results remain separate.

The candidate window is seeds **11000–11199**, across four structures:
`closed_ring`, `ns_gates`, `ew_gates`, and `layered_core`. There are 800 candidates;
the unchanged structure-and-difficulty filter determines the accepted count.
Reserve seeds are not used, including if a structure has no accepted bases.

## 4. Check progress

```bash
tail -n 25 artifacts/final-evaluation-v1.log
uv run python final_evaluation_bundle/run_final_evaluation.py status
```

During the original protocol, the log prints a heartbeat approximately every
30 seconds. This reports elapsed time, not episode completion or an ETA, because
the original protocol does not expose that progress through its API.

During the sampled phase, the log prints completed episodes, percentage and an
estimated remaining time every 100 episodes. That ETA covers the sampled phase.
The status command reads a small JSON file without opening the live database.
Check its timestamp: an abruptly stopped process can leave a stale `running` state.

For A accepted bases, the original evaluation has **248 × A** games and the
sampled evaluation has **480 × A** games, for **728 × A** total. Pool generation
also takes time. A reliable whole-run duration is not known in advance.

CPU inference with one PyTorch thread is deliberate: it matches the validated
development sampling procedure. A CUDA training instance can run this evaluation.

## 5. Resume after an interruption

First check that the old process is no longer active:

```bash
pgrep -af '[r]un_final_evaluation.py run'
```

If it is still running, monitor it instead of starting another copy. If it has
stopped, resume with:

```bash
cd ~/Base_Siege_Smooth_Demo/base-siege-assignment3
nohup uv run python final_evaluation_bundle/run_final_evaluation.py run --confirm-final --resume >> artifacts/final-evaluation-v1.log 2>&1 < /dev/null &
```

Completed pool candidates and evaluation episodes are retained. The same source,
runtime, models, evidence, amendment and output directories must be present.
Locks prevent concurrent writers. A hard interruption during a transaction may
require repeating the single uncommitted episode.

Do not delete progress databases, freezes or authorization markers. Preserve the
log if a guard rejects a run. Do not use a new output directory to rerun final
evaluation after looking at results.

## 6. Download the results for analysis

Wait for `Both final evaluations completed` in the log. Then run:

```bash
uv run python final_evaluation_bundle/run_final_evaluation.py collect
```

Download **`Base_Siege_Final_Results.zip`** from the project root and upload it in
the chat. The collection command checks completion receipts and includes:

- The registration, freeze, amendment, and final authorization marker.
- Original results in `artifacts/study-final-v2/`, including the accepted pool.
- Sampled results in `artifacts/sampled-final-v1/`, including episode records,
  summaries, the contract and completion receipt.
- The evaluation scripts and these instructions.

It excludes large model checkpoints and live progress databases. Keep those in
your project backup; they are still needed for reproducibility and resuming.

## What this amendment means for the report

The original registered primary comparison remains **deterministic PA minus P**.
Sampling is an additional evaluation selected after development analysis and
specified before final access. Do not describe it as part of the original
pre-training registration.

All twenty endpoints are evaluated without selecting a winning checkpoint.
The sampled extension compares PA with P, sampled with deterministic inference,
and sampled with the heuristic. Destruction is the main extension metric;
Town Hall destruction, full clears, return, spending and failures provide context.

Average games and action repeats within each learner, then compare the ten paired
learner seeds. The 95% intervals describe learner variation on this fixed pool;
individual games and action repeats are not independent trained agents.
Secondary comparisons and motif analyses are descriptive, without adjustment for
multiple comparisons.

Report familiar structures and `layered_core` separately. `layered_core` was held
out from training and current development, but appeared in earlier software
checks. Conclusions apply to this simplified simulator and its filtered base
distribution. An unclear P/PA difference is a valid result; do not tune or retrain
against the final results to obtain a preferred conclusion.

After final analysis, the remaining work is the trained-policy demo, report,
presentation, and submission package. The smooth playback demo should use the
same sampled inference rule if it is presented as a demonstration of these results.

## Local validation and limits

The extension was tested with the real simulator and transactional episode journal
using lightweight test policies. Checks cover identical interrupted/resumed
results, no repeated completed inference, changed-contract rejection, preserved
sampling semantics, explicit final authorization, prevention of backdated
amendments, exact final-pool binding, and learner-level aggregation.

The full trained PPO checkpoints are on SageMaker. Local verification did not run
those checkpoints or consume the protected final bases. Runtime and artifact
checks execute again on SageMaker before evaluation.

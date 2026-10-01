# Base Siege Version 2 — fresh final evaluation

This add-on evaluates the existing twenty Version 2 models. It does not train,
change simulator rules, or replace Version 1 results. The final test has not been
run by the package author. Run it in your existing SageMaker project.

## What is frozen

- All P-0–P-9 and PA-0–PA-9 final checkpoints at 4,194,304 transitions each.
- DS: deterministic purchasing, then sampled deployment. Phase detection uses
  the existing nine-value observation and legal-action mask.
- Candidate layout seeds **11,200–11,399 inclusive**, all four existing motifs:
  800 candidates. Version 1 used 11,000–11,199. Training uses 0–9,999 and the
  frozen development pool uses 10,200–10,399.
- The original structural checks and difficulty filter are unchanged: the
  best of four side-specific heuristic attacks must score between 25% and 85%.
  All accepted candidates are retained. Rejections are recorded. No replacement
  seeds are added to obtain a desired result or accepted count.
- Every accepted layout receives eight coordinate views. Each learned model
  receives three sampled attacks per view, labels 41001, 41002 and 41003.
- The ordinary observation-based heuristic is evaluated once per layout/view.
  It does not receive the filter's best-of-four-side outcome selection.
- 488 evaluation attacks per accepted layout: 20 × 8 × 3 + 8. Filtering attacks
  are separate. If 200 layouts pass, evaluation contains 97,600 attacks.
- All models use CPU inference, one Torch thread and deterministic algorithms,
  as in the development evaluator. Python, NumPy and Torch are seeded once per
  attack. There is no reseeding between purchasing and deployment.

`protocol.json` specifies the complete design. `reference.json` binds the
verified development evidence, exact checkpoints and training records.
`freeze.json` records this decision before this runner generates final layouts.
This is a development-informed final protocol, not a preregistration before
training. Layered core is excluded from V2 training and development, but it was
examined in the earlier project; it is not an entirely new, never-seen concept.

## Run on SageMaker

Use a **Terminal in SageMaker JupyterLab**, not a Python notebook cell. Keep the
existing environment and files. Do not upgrade packages or rerun training.

1. Download `Base_Siege_V2_Final_Evaluation.zip` and upload it into the project
   folder that contains `pyproject.toml`, `uv.lock`, `env`, `runs` and `artifacts`.
   Keep that exact ZIP filename.
2. In the terminal:

```bash
cd ~/Base_Siege_Smooth_Demo/base-siege-assignment3
uv run --frozen python -m zipfile -e Base_Siege_V2_Final_Evaluation.zip .
uv run --frozen python -m v2_final preflight
```

Preflight must report `"status": "ready"`, `"models": 20`, and
`"candidate_layouts": 800`. It reads provenance; it does not generate final
layouts. It checks recorded pools/contracts for reuse of the new seed range.
These checks cannot establish whether somebody tested that range outside the
saved project. If such testing happened, stop and document it before choosing a
new protocol. If preflight reports a mismatch, retain the error and restore the
referenced file/environment. Do not edit expected hashes to get past a failure.

3. Freeze, then run an initial batch:

```bash
uv run --frozen python -m v2_final freeze
uv run --frozen python -m v2_final run --max-new-episodes 1000
```

The first run checks all 800 candidates, freezes the accepted pool, then runs
1,000 evaluation attacks and pauses. This is the beginning of the final test,
not a new development diagnostic. Do not change settings based on its scores.
The terminal shows the accepted layout count and exact total attacks. The
1,000-attack limit applies to newly completed evaluation attacks in this command,
not the heuristic attacks used to filter candidates.

4. After the batch returns `"status": "paused"`, continue:

```bash
uv run --frozen python -m v2_final run
```

The runner resumes after committed attacks. It does not redo the first 1,000.
Keep the SageMaker app/instance running until completion or request a pause.

## Status, pause and restart

In a second terminal, from the same project folder:

```bash
uv run --frozen python -m v2_final status
```

During pool generation, units mean candidate layouts. During evaluation, units
mean attacks. The rough time estimate appears after at least 100 new attacks
in an attempt; different models/layouts may run at different speeds.

To stop cleanly:

```bash
uv run --frozen python -m v2_final pause
```

Wait for `"status": "paused"` and for the running terminal to return its prompt.
Then stop SageMaker if needed. Keep the entire project on its persistent volume,
including `artifacts/version2-final-evaluation/` and both SQLite progress files.
After restarting, open a terminal and run:

```bash
cd ~/Base_Siege_Smooth_Demo/base-siege-assignment3
uv run --frozen python -m v2_final run
```

An interrupted attack can be repeated from its recorded seed; every committed
attack is retained exactly once. A second writer is rejected. Changed settings,
weights, scientific source or relevant package versions are rejected. Repeating
`run` after completion verifies and reproduces the same exports with zero new
attacks. Never delete a final output folder to try for a better result.

## Completion and results to share

Completion means `"status": "completed"`, with completed and expected units
equal and a valid `complete.json` receipt. Then run:

```bash
uv run --frozen python -m v2_final pack
```

The terminal prints a timestamped ZIP path under `artifacts`. Download that
`Base_Siege_V2_Final_Results_....zip` and upload it for review. The small result ZIP
contains results and provenance; it is **not** a full model/project backup.
Keep the original run folders and SQLite journals on SageMaker.

Output folder: `artifacts/version2-final-evaluation/`.

- `freeze.json`: timestamped protocol, exact model identities and source identity.
- `final_pool.json`: every candidate decision and accepted base specification.
- `evaluation_contract.json`: pool hash and expected attack count.
- `episodes.csv`: one row per model/layout/view/action-repeat attack, plus baseline.
- `model_summary.csv`: results per learner and reporting scope.
- `condition_summary.csv`: equally weighted learner averages for P and PA;
  heuristic listed separately, without invented independent learner repetitions.
- `comparisons_by_learner.csv`, `comparisons.csv`: paired PA−P and learned−heuristic
  differences; percentage-point units for percentage metrics.
- `complete.json`: counts and hashes of scientific outputs.
- `attempts.jsonl`, `progress.json`: measured execution records and status.
- `timing_summary.json` (inside packed ZIP): sum of finished attempt durations and
  any attempts missing an end record, such as a hard shutdown. Missing durations
  are flagged rather than inferred from archive timestamps.

## Interpretation agreed before seeing results

Mean destruction is the fraction of **building value** destroyed. It is not
classification accuracy or a full-clear probability. Full-clear percentage,
Town Hall destruction, zero-destruction and below-50% rates answer other questions.

Primary comparison: PA minus P, averaged over all accepted layouts, views and
action labels within each learner, then across all ten paired learner seeds.
All twenty models contribute; no best seed is selected. Secondary comparisons
include P/PA against the heuristic and results for familiar motifs, layered-core
transfer, and each motif separately. Overall means weight accepted layouts equally,
so the accepted motif mixture determines their weighting.

The 95% percentile bootstrap intervals use 20,000 resamples of learner observations
(seed 53001), pairing P and PA by learner seed. These intervals are conditional
on the fixed final layout set and acting labels. They do not treat thousands of
attacks or eight transformed views as independent trained agents. Baseline
comparisons use the same deterministic baseline average for each learner's
contrast. Secondary intervals are unadjusted for multiple comparisons.

This simulator retains the known inaccessible-target movement limitation.
Reachable-target fallback is not added. Manual play is not a controlled human
benchmark. A difference from Version 1's old final score mixes training budget,
acting procedure and different test layouts; do not call it a clean causal
measurement of longer training alone.

Do not tune on these final scores. If another training experiment is later
requested, document a new version and a suitable evaluation plan.

## Time estimate

The previous cohort evaluation measured 122,304 new attacks in 10,704.79 seconds,
about 0.088 seconds per attack on that SageMaker setup. At the same average speed,
200 accepted layouts would take about 2 hours 22 minutes for 97,600 attacks, plus
filtering, checks and export. This is an estimate, not a promise: DS-only attacks
and the new layouts may run differently. Use the live estimate after the first
batch. The theoretical upper bound is 390,400 attacks if all 800 candidates pass;
the acceptance count is deliberately not known before the final pool is generated.

## Software verification

See `VALIDATION.md`. Optional tests use development seeds and an already-existing
checkpoint; they never train or generate the new final layouts:

```bash
uv run --frozen python -m pytest v2_final/tests -q
```

The add-on uses the project's existing dependencies and helper modules. It adds
only `v2_final/`. Do not copy test virtual environments or replace `env/`, `train/`,
`agents/`, `experiments/`, `uv.lock` or existing result folders.

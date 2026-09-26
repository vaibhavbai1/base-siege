# Experiment plan: revision 2

Project 12 · Base Siege · 8 September 2026

This is the current prospective design. Draft A/B documents and historical software artifacts
remain unchanged. [env/config.py](../env/config.py) is the executable source of constants.

## Question and comparison

Does episode-level D4 view augmentation improve Maskable PPO on unseen procedural layouts?
P and PA share architecture, reward, PPO settings, training partitions and transition budgets.
Augmentation is their only planned learning-method difference.

PPO hyperparameters are gamma 1.0, learning rate 0.0003, GAE lambda 0.95, entropy coefficient 0.01,
batch size 256 and ten epochs per rollout. Four workers collect 8,192 transitions per rollout.
The main curriculum has four placement rollouts and twenty-eight joint rollouts.

Use ten matched learner seeds, 0–9, for 20 full learners. Alternate P/PA execution order between
seed pairs. No best-seed selection, early stopping based on final scores or favourable-run exclusion.
The registered fixed-budget endpoint is the evaluated model.

## Preliminary development gate

Run P and PA at learner seed 100 for 57,344 transitions each: four placement rollouts and three
joint rollouts. These are development pilots, not independent main-study results.

Use the same filtered pools as the intended study. Run complete eight-view development evaluation.
The pilot-review tool checks provenance and reports measured throughput and diagnostics. Before
recording approval, the team must examine:

- whether rewards, actions and masks remain finite and valid;
- whether destruction/return curves show useful learning rather than only empty-army behaviour;
- performance and failures relative to random and the heuristic;
- balance across structures, and whether the curriculum transition causes collapse;
- checkpoint size, free storage, pause/resume behaviour and measured runtime.

There is no automatic “good policy” threshold. A single pilot pair cannot establish statistical
superiority. If the task does not support useful learning at this stage, review the observation,
reward or training budget before spending on the full matrix. Changes require new matching
pilots and provenance; do not silently pool incompatible runs.

## Layout partitions and accounting

| Partition | Candidate seeds | Structures | Candidate layouts |
|---|---|---|---:|
| Training | 0–9,999 | Three training structures | 30,000 |
| Development | 10,200–10,399 | Same three structures | 600 |
| Final | 11,000–11,199 | Three training structures plus layered core | 800 |
| Reserve | 11,200–11,999 | None generated for this study | 0 |

The fourth structure appeared in earlier development software checks. Current evaluation excludes
it until the frozen final stage; this does not make historical exposure disappear. Describe it as
held out from training, not as an entirely undiscovered structure.

Structural validation checks geometry, placement, composition and defensive coverage. Difficulty
validation runs the fixed heuristic from four sides and accepts best-side destruction between
0.25 and 0.85 inclusive. The reported heuristic chooses using its observation, not an oracle
that knows those four outcomes.

Sampling is uniform over the accepted pool. Accepted counts may differ by structure. Report
candidate counts, acceptance rates and the induced distribution; do not imply equal weighting
by structure. The primary pooled estimate follows that distribution. Per-structure comparisons
are additional analyses.

Registration requires complete candidate accounting and representation of every declared structure.
Do not substitute smoke pools. Do not backfill rejected final layouts using reserve seeds.

## Evaluation workload

Every accepted layout is evaluated in all eight coordinate views.

- Ten P learners and ten PA learners.
- One deterministic heuristic.
- Ten random-action seeds, reset for each episode.

That is 248 evaluation episodes per accepted layout: at most 148,800 for the development pool
and 198,400 for the final pool. Actual counts depend on filtering. The pilot pair with baselines
uses 104 episodes per accepted development layout.

During training, the diagnostic panel takes up to four deterministically spread seed positions
from each development structure, across views 0, 1, 4 and 5. With full representation it uses
48 episodes per check. It runs before learning and at completed updates after each 25,000-step
threshold. It is not a replacement for full development evaluation.

## Metrics and uncertainty

The primary metric is destruction: the fraction of non-wall building value destroyed.
The primary contrast is PA minus P. Match each learner pair on layout, structure and view,
average the difference within each independent learner seed, then form a two-sided t-based
95% confidence interval across learner seeds.

The independent learner count is ten, not the number of evaluation episodes. This interval
describes seed variation conditional on the fixed evaluation pool. It does not include all
uncertainty over possible layout generators or real-world tasks.

Secondary measures are Town Hall destruction, full clears, return, spending, per-structure
performance and max-minus-min mean destruction across views. Treat multiple secondary comparisons
as exploratory; do not select whichever happens to look significant as the primary finding.

Report raw learner values, effect sizes and uncertainty. The held-out seeds, model hashes and
configuration must be traceable from every research result.

## Recovery does not permit retesting a new hypothesis

Pool candidates and evaluation episodes commit to checksummed transactional journals. Resuming
requires identical settings, inputs, model identities and relevant source/runtime versions.
The interrupted unit may repeat; completed units are reused.

Final evaluation creates a binding to one freeze and output before generating layouts. A crash
can resume that same experiment, not open a new output with different settings. Never remove
the binding just to obtain another final attempt. Final guards prevent accidental misuse, not
deliberate editing of source or manifests.

## Remaining scientific limits

The observation is partial, the policy is feed-forward, combat is simplified and the layout filter
is tied to one heuristic. D4 changes coordinate views; physical combat tie-breaks are not asserted
to be equivariant. No claim is made about a commercial game, an optimal policy, or augmentation
improvement until measured evidence exists.

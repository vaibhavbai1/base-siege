# Base Siege: team master document

UTS 43008 Reinforcement Learning · Assignment 3 · Project 12 · Dihedrals  
Current design: study revision 2, 8 September 2026

## The project in one minute

We are building a small strategy-game environment in which an agent learns what army to buy,
where to deploy it and when to wait. It has 200 resources and attacks a defended 16 × 16 base.
Once troops are deployed, fixed movement and combat rules control them.

Our research question is:

> Does training Maskable PPO with rotated and reflected views improve its performance on new
> base layouts and reduce its sensitivity to the orientation of the board?

We compare standard Maskable PPO, called **P**, with the same learner using view augmentation,
called **PA**. Random and rule-based agents provide reference points. The environment, network,
reward, training budget and learner seeds are matched between P and PA.

The software exists. Useful learned strategies, the full experiment and final findings do not
exist yet. We must not present short software checks as evidence that augmentation works.

## 1. What came from Vaibhav?

Yes: this project is based on the Stage 5 ZIP Vaibhav shared, not a separate replacement idea.
The received archive is kept unchanged at [reference/base_siege_stage5_sync.zip](../reference/base_siege_stage5_sync.zip).

Its SHA-256 is:

650acdfd85271c1fe55c4107d8d31488eedca0410413416ea526757b56b50ece

| Source | What it contributed |
|---|---|
| Vaibhav's Stage 5 archive | Game constants, entities, procedural structures, A* pathfinding, combat scaffold, renderers, HTML theme and original tests |
| This project's extensions | Simulator corrections, Gymnasium contract, baselines, PPO integration, augmentation, experiment design, recovery, notebooks and Streamlit app |
| External libraries | Gymnasium API, PyTorch networks, SB3 Contrib's Maskable PPO and Streamlit interface |
| Draft Parts A and B | Initial question and proposed scope; they are historical drafts, not frozen implementation specifications |

We did not invent PPO. We should credit Vaibhav's foundation and the libraries, explain our
custom integration and controlled experiment, and record each student's real contribution.
The separately mentioned v3 plan was not in the supplied ZIP; we have not assumed its contents.

## 2. How the game works

1. A seed and a structure generate a base containing a Town Hall, defences, storages and walls.
2. In the shopping phase, the agent buys tanks, melee troops, ranged troops and wall breakers.
3. It finishes shopping and starts deployment.
4. It chooses a troop and perimeter tile, or waits. A deployment or wait advances up to ten
   simulation ticks. Buying troops does not advance battle time.
5. The simulator moves troops, selects targets, applies damage and checks whether the battle ends.
6. The episode ends after a full clear, the deadline, or the army is exhausted.

The agent is a commander, not a joystick controller. It does not choose every movement step,
attack target or pathfinding decision. Those mechanics are part of the environment.

### What the agent sees

The observation has eight 16 × 16 spatial channels and nine summary values. It includes building
locations and health, defence coverage, troop density and army/resource/phase information.

We are deliberately keeping this compact observation for the main comparison. It is **partially
observable**: it does not expose every troop's internal state. The feed-forward policy has no
memory. This is a limitation to analyse, not something gamma or action masks can solve.

### What the agent can do

There are 246 action IDs: four purchases, finish shopping, 240 troop/tile deployments and wait.
A legal-action mask disables actions that cannot currently be taken. It does not choose the
best move for the agent.

### What it is rewarded for

Reward includes destroyed building value, milestone bonuses and a resource-use penalty.
Walls are obstacles, not score targets. Optional wall shaping remains off.

The main reported metric is the fraction of non-wall building value destroyed, not the shaped
training return. We also report Town Hall destruction, full clears, return, spending and failures.

## 3. What actually learns?

The network processes the spatial observation with two convolutional layers. A small separate
branch processes the summary vector. Their combined features feed:

- a policy head, which scores legal actions; and
- a value head, which estimates future return.

PPO collects experience, calculates advantages and updates these heads and their shared features.
Its clipped objective limits how abruptly the policy changes during an update.

P uses the normal coordinate view. PA samples one of eight square rotation/reflection views at
episode reset. Observations, legal masks and deployment actions are mapped consistently.

This is view augmentation of one physical simulator. We are not claiming that all physical
combat tie-breaks are rotation-equivariant.

## 4. Our current working experiment

| Item | Revision 2 decision |
|---|---|
| Main learner count | Ten seeds per condition: 0–9, giving 20 full learners |
| Preliminary pilot | One P/PA pair at learner seed 100, separate from the main learners |
| Pilot budget | 57,344 transitions per learner; development evidence only |
| Main budget | Exactly 262,144 transitions per learner |
| Curriculum | 32,768 fixed-army placement transitions, then 229,376 joint transitions |
| Parallel environments | Four workers |
| PPO rollout | 2,048 transitions per worker, or 8,192 combined |
| Main total | 5,242,880 transitions, excluding pilots and evaluation |
| Model selection | The fixed-budget endpoint of every registered learner |
| Training structures | Closed ring, north/south gates, east/west gates |
| Current development structures | The same three families, with separate layout seeds |
| Transfer structure | Layered core; absent from training and current development evaluation |
| Evaluation views | All eight, for every accepted layout and compared policy |
| Primary comparison | PA minus P destruction, with uncertainty across independent learner seeds |

Earlier software checks included layered-core development layouts. Removing it now does not
erase that history. Our accurate claim is held out from training and current tuning, not a
structure nobody has ever inspected. Protected final-layout seeds have not been used.

Ten seeds are our default, not an assignment-wide rule. The registration tool can support an
explicit five-seed design if resources require it, but that decision must precede the main
experiment. Do not choose the sample size after looking for a favourable result.

The [experiment plan](EXPERIMENT_PLAN.md) defines sampling, metrics and the reporting rules.

## 5. What we run, in order

Use the commands in the [runbook](RUNBOOK.md). Do not start by running every notebook cell that
mentions training.

1. **Verify the machine.** Install the locked project environment, check CUDA and persistent
   storage, and run the short recovery check.
2. **Prepare the pools.** Validate the declared candidate windows. Interrupted generation resumes
   completed candidates instead of starting again.
3. **Run the pilots.** Train P and PA with the same pilot settings. Run eight-view development
   evaluation. Review learning curves, the heuristic/random comparison, empty armies, failures,
   throughput and checkpoint size.
4. **Make a team decision.** The pilot tool can summarise evidence but cannot decide whether
   learning is scientifically convincing. Record an honest review before registering the study.
5. **Register and train.** Freeze the pool/code identities and learner list. The study controller
   runs the matrix, skips verified completed learners and resumes interrupted learners.
6. **Evaluate development.** Inspect all registered learners, not just the most attractive run.
7. **Freeze and test once.** Final evaluation requires a separate explicit command. It can resume
   the same frozen experiment but cannot silently switch models, outputs or settings.
8. **Analyse and present.** Generate tables and plots from saved CSVs, explain negative cases,
   export reviewed demo assets, write the report and practise the defence.

If development changes the reward, observation, model or other scientific settings, create new
matching pilots and a new registration. Old runs remain evidence of the old design.

## 6. Recovery and storage

| Work | Saved recovery unit | What may repeat after interruption |
|---|---|---|
| PPO training | Completed update, model/optimiser, worker state, RNGs and history | The uncommitted rollout/update; more if falling back from damaged files |
| Pool construction | One validated candidate in a checksummed SQLite journal | The candidate interrupted before its transaction committed |
| Policy evaluation | One completed episode in a checksummed SQLite journal | The episode interrupted before its transaction committed |
| Full learner matrix | Immutable plan plus each learner's own checkpoint | The interrupted learner's uncommitted work, not verified completed learners |

Checkpoint schema 3 references the immutable pool saved with the run; it does not copy that pool
into every worker snapshot. Dynamic simulation state is still fully saved. Older checkpoints
belong to their original code version and must not be relabelled as new-study runs.

No S3 is used. Keep the complete project, inputs, journals and run folders on persistent Studio
storage. Make a separate backup of important evidence. Local checkpoints are not protection
against deleting the volume or losing the only copy.

## 7. SageMaker and realistic timing

The user's screenshots show ml.g4dn.xlarge and ml.g4dn.2xlarge, a 50 GB storage allocation and
a 60-minute administrator-controlled idle setting. Both GPU options have one T4. The larger
option adds CPU and RAM, which may help the four simulator workers.

Begin with the xlarge option and measure it. We have not executed this project on SageMaker or
verified its CUDA recovery behaviour. An available GPU does not automatically make all stages
faster: pool filtering and the current evaluator use the CPU.

The pilot report estimates main-training hours from actual pilot wall time. It excludes full
pool construction, separate evaluation, installation and reruns. Use that measurement before
promising a completion time. The [Studio guide](SAGEMAKER_STUDIO.md) covers storage and shutdown.

## 8. Assignment deliverables and team work

Our saved specification allocates 5 points each to A, B, C and D; 40 to E; 15 to F; and 25 to G.
The detailed current Canvas pages remain authoritative. This document is not a marking guarantee.

| Deliverable | Evidence we need |
|---|---|
| C: initial results | Real baseline/pilot measurements, exact settings and a clear preliminary-results table |
| D: environment/interface | Explanation of rules, observation/action/reward design and how the GUI uses the same environment |
| E: final report | Related work, reproducible methods, matched results, uncertainty, failure analysis, limitations and contribution agreement |
| F: presentation/demo | A reliable demonstration of decisions and replays, with truthful trained-policy labels |
| G: oral defence | Each member can explain the whole pipeline, not only their own notebook |

Suggested ownership areas are environment/tests, learning/recovery, and evaluation/app/reporting.
The team should allocate names and record actual contributions; no allocation here claims that
someone has already completed work. Every member should review another member's area.

The Streamlit public deployment files are prepared locally. Deployment is separate from the
research run and has not been performed. Do not upload recovery checkpoints or arbitrary local
artifacts; export only reviewed inference weights, replays and result files.

## 9. Questions everyone should answer

| Professor's question | Short answer |
|---|---|
| Why is this RL? | Purchases and deployment affect later outcomes; the policy learns from reward over episodes. |
| What is custom? | The task, simulator integration, representations, reward, augmentation and experiment/recovery design. PPO itself is a library algorithm. |
| Why action masks? | Many actions are impossible in a given phase or with the current resources/reserves. |
| Why gamma 1? | Episodes are finite, and zero-time shopping decisions should not discount later reward merely because more purchases were made. |
| Why two phases? | The fixed-army phase introduces deployment before the learner also has to discover army composition. Its usefulness still needs measured evidence. |
| What does one seed mean? | One independent learner initialisation/random stream. Thousands of test games do not replace independent learner seeds. |
| Why might PA help? | A tactic may transfer between equivalent coordinate views. It is a hypothesis, not a finding yet. |
| Is the observation Markov? | Not fully. Aggregated troop information hides internal state. |
| Can the experiment survive shutdown? | Completed updates, candidates and episodes have recovery records, as long as the storage survives. |
| What if PA is worse? | Report it, quantify uncertainty and examine failures. A fair negative result is still informative. |

For code-level answers, use the [architecture](ARCHITECTURE.md) and [viva guide](VIVA_GUIDE.md).
For measured software checks and their limits, use [verification](VERIFICATION.md).

## 10. Completion gate

The project is ready for final submission only when the pilot review, registered learners,
development analysis, frozen final evaluation, report, demo and individual preparation are
complete. Passing regression tests alone does not complete that gate.

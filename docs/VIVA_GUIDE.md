# Explain the project

Current design and responsibilities: [TEAM_MASTER.md](TEAM_MASTER.md).

## How many runs and why?

Ten independent learner seeds for P and ten for PA, giving 20 full learners at 262,144
transitions each. A separate pilot pair uses learner seed 100 and 57,344 transitions per learner.
Random-action seeds and eight orientation views are evaluation repetitions, not extra PPO learners.

## What happens if the machine stops?

Training resumes from a committed update with optimiser, worker, RNG and history state.
Pool generation resumes committed candidates; evaluation resumes committed episodes. Final
recovery is bound to the same freeze and output, so it does not permit testing a changed policy.
This works only while the underlying storage and compatible code/environment survive.

## Thirty seconds

“We built a small base-attack environment. An agent has 200 resources to buy an army, then chooses where and when to deploy troops. We compare standard Maskable PPO with the same algorithm trained on rotated and reflected views. We ask whether augmentation helps on unseen layouts at the same training budget. The software is implemented and locally tested; longer experiments still need to run.”

## What is learned and what is programmed?

The policy learns purchases, deployment type/location and waiting. A* movement, target priorities, attack range, damage and defence behaviour are programmed. We are not learning low-level movement or an opposing defender.

## Why reinforcement learning?

A purchase affects later deployment choices; deployment affects survival and destruction. There is no supplied label for the correct army. The agent learns from rewards over a sequence, including delayed consequences.

## Explain PPO simply, then technically

Simply: try actions, observe outcomes and make useful choices more likely. PPO limits policy changes per update to reduce erratic learning.

Technically: an on-policy actor-critic method collects rollouts and estimates advantages relative to expected returns. It uses a clipped policy-ratio objective, value loss and entropy term. SB3 Contrib implements those updates; our masks remove invalid actions.

## Why masks and a CNN?

Buying without money and deploying absent troops are impossible. Masks prevent those choices, without revealing which legal move is best. All policies receive legality information. The CNN learns spatial patterns in walls, openings and defences; a small separate network processes army counts, resources and time.

## Why rotations and reflections?

A tactic should not only work when an opening appears on one screen edge. PA sees transformed views and must map actions consistently. Architecture and learning settings remain matched. Improvement is a hypothesis, not a result we can claim yet.

## Is this fully observed?

No. The internal simulator has full state, but the policy sees density instead of each troop's type, health and target. Identical observations can therefore have different futures. A feed-forward policy is a compact baseline under partial observability.

## How do you know it works?

Explain three levels: simulator regressions for walls/damage/scoring; RL contracts for spaces/masks/rewards and D4 mapping; integration checks for two-worker training, checkpoint reload, notebooks and browser controls. These establish software behaviour, not that a 256-step model has learned to win.

## How is evaluation fair?

Match training seeds, pools, architecture, configuration and actual interaction budgets. Evaluate paired layouts and views. Average each learner's PA-minus-P difference, then calculate uncertainty across independent learners. Keep final layouts out of tuning and report the transfer structure separately.

## What could go wrong?

The agent may buy nothing, spend poorly or fail to breach walls. Rewards may encourage unintended behaviour. Heuristic filtering narrows the population. Even ten learners can give wide uncertainty. Report these outcomes instead of selecting only attractive replays.

## What is the team's contribution?

Vaibhav supplied the Stage 5 simulator and visual foundation. This build extends it into the RL interface, baselines, PPO training, D4 comparison, evaluation protocol, notebooks and app. PPO, Gymnasium, PyTorch and Streamlit are third-party tools. The research question concerns a controlled comparison in this custom task, not a new PPO algorithm.

## Live explanation sequence

Show notebook 01's observation layers. Buy one tank and finish shopping: resources become 175 while the clock remains 300. Deploy it: reserve decreases and up to ten ticks pass. Show a wall interaction and event log. Generate a heuristic replay, then show the software-smoke label on a PPO checkpoint. Finish with notebook 04's evaluation procedure, without claiming final results.

"""Load a TRPO checkpoint and run one complete attack.

This is the minimal reference for loading condition T into the interface. It does
the five steps end to end: load the checkpoint, build the environment, read the
observation and the legal-action mask, choose an action by the same procedure the
frozen evaluation used, and play one attack to termination.

    python handoff/predict_example.py
    python handoff/predict_example.py --checkpoint runs/study-v3/T-1/model.zip \
        --base-seed 21200 --motif closed_ring --view 0 --mode DS --action-seed 41001
    python handoff/predict_example.py --mode SS      # sampled purchasing

Works from any directory: it puts the project root on sys.path itself, and a
relative --checkpoint is resolved against the project root.

No preprocessing or saved statistics are needed. There is no VecNormalize and no
observation normalisation anywhere in the pipeline: the feature extractor consumes
the raw Dict observation. The one thing that is not a stock SB3 object is
`train.features.SiegeFeatures`, and SB3 rebuilds it from the checkpoint's own
policy_kwargs, so `MaskableTRPO.load` is all that is required.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Running a file inside handoff/ puts handoff/ on sys.path, not the project root,
# so `import env` fails. Put the root first, before any project import below.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from env import config as C
from env.generator import generate_base
from env.siege_env import SiegeEnv
from env.symmetry import SymmetryWrapper
from trpo.maskable_trpo import MaskableTRPO

# MODES[name] = (purchasing_deterministic, deployment_deterministic)
#   DS is the frozen V3 evaluation mode: greedy purchasing, sampled deployment.
#   SS samples both. For TRPO the difference is large; see EXPERIMENT_DETAILS.md.
MODES = {"SS": (False, False), "DD": (True, True), "DS": (True, False), "SD": (False, True)}


def load_checkpoint(path, device="cpu"):
    """Step 1. MaskableTRPO.load restores the policy, the custom feature
    extractor and the action distribution class from the saved policy_kwargs."""
    model = MaskableTRPO.load(path, device=device)
    model.policy.set_training_mode(False)
    return model


def build_environment(base_seed, motif, view):
    """Step 2. allow_final=True permits evaluation seeds that training refuses.
    SymmetryWrapper applies D4 view `view` to coordinates and permutes the
    deployment actions to match; the combat itself is not rotated."""
    base = generate_base(base_seed, motif)
    # allow_final=True is required: the V3 pool uses seeds 21200-21399, which
    # SiegeEnv otherwise refuses as protected evaluation seeds. This matches
    # v3_final/runner.py exactly.
    env = SymmetryWrapper(SiegeEnv(motifs=C.MOTIFS, allow_final=True), transform=view)
    return env, base


def choose_action(model, observation, mask, mode):
    """Steps 3 and 4. The phase flag decides which determinism setting applies.

    The mask is passed into predict() as action_masks, so masking happens inside
    the policy's distribution. No action is ever sampled and then corrected.
    """
    phase = float(observation["vector"][6])
    if phase not in (0.0, 1.0):
        raise ValueError(f"Unexpected phase observation: {phase}")
    shopping = phase == 0.0
    if bool(mask[C.ACTION_FINISH_SHOPPING]) != shopping:
        raise ValueError("Phase flag and legal actions disagree")

    deterministic = MODES[mode][0 if shopping else 1]
    action, _state = model.policy.predict(observation, action_masks=mask, deterministic=deterministic)
    action = int(action)

    # If this ever fires, masking is broken. It is an assertion, not a repair:
    # silently substituting a legal action would hide the bug and corrupt results.
    if not 0 <= action < len(mask) or not mask[action]:
        raise ValueError(f"Policy selected an illegal action: {action}")
    return action, "purchasing" if shopping else "deployment", deterministic


def run_attack(model, base_seed, motif, view, mode, action_seed, verbose=True):
    """Step 5. One complete attack, with the same RNG discipline the evaluation
    used: seed Python, NumPy and Torch from the acting seed before the episode."""
    random.seed(action_seed)
    np.random.seed(action_seed)
    torch.manual_seed(action_seed)

    env, base = build_environment(base_seed, motif, view)
    try:
        observation, _info = env.reset(seed=0, options={"spec": base})
        steps = 0
        phase_counts = {"purchasing": 0, "deployment": 0}
        while True:
            mask = np.asarray(env.action_masks())
            action, phase, deterministic = choose_action(model, observation, mask, mode)
            phase_counts[phase] += 1
            if verbose and steps < 5:
                legal = int(mask.sum())
                print(f"  step {steps:>3}  {phase:<11} "
                      f"{'greedy' if deterministic else 'sampled':<8} "
                      f"action {action:>3}  ({legal} of {len(mask)} legal)")
            observation, _reward, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                break
    finally:
        env.close()
    return info, steps, phase_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="runs/study-v3/T-1/model.zip")
    parser.add_argument("--base-seed", type=int, default=21200)
    parser.add_argument("--motif", default="closed_ring", choices=list(C.MOTIFS))
    parser.add_argument("--view", type=int, default=0, choices=range(8))
    parser.add_argument("--mode", default="DS", choices=sorted(MODES))
    parser.add_argument("--action-seed", type=int, default=41001)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    path = Path(args.checkpoint)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise SystemExit(f"No checkpoint at {path}")

    print(f"checkpoint   {path}")
    model = load_checkpoint(path, device=args.device)
    print(f"algorithm    {type(model).__name__}")
    print(f"transitions  {model.num_timesteps:,}")
    print(f"extractor    {type(model.policy.features_extractor).__name__}")
    purchasing, deployment = MODES[args.mode]
    print(f"mode         {args.mode}  "
          f"(purchasing {'deterministic' if purchasing else 'sampled'}, "
          f"deployment {'deterministic' if deployment else 'sampled'})")
    print(f"layout       {args.motif} seed {args.base_seed}, view {args.view}, "
          f"action seed {args.action_seed}")
    print()

    summary, steps, phases = run_attack(
        model, args.base_seed, args.motif, args.view, args.mode, args.action_seed)

    print(f"  ... {steps} decisions "
          f"({phases['purchasing']} purchasing, {phases['deployment']} deployment)")
    print()
    print("result")
    # destruction is a FRACTION in 0..1 here. The summary CSVs report percent.
    for key in ("destruction", "full_clear", "town_hall", "episode_return",
                "spent", "tick", "decisions", "finish_reason"):
        if key in summary:
            value = summary[key]
            shown = f"{value:.4f}" if isinstance(value, float) else value
            print(f"  {key:<16} {shown}")
    print()
    print(f"  destruction as percent: {100 * summary['destruction']:.1f}%")


if __name__ == "__main__":
    main()

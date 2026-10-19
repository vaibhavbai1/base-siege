"""TRPO hyperparameters.

These live here rather than in env/config.py on purpose. env/config.py is one
of the 28 files hashed in ui/v2_checkpoints.json; editing it fails
`python -m ui.v2_models check` and the application stops loading the twenty
approved V2 checkpoints. Shared constants are imported read-only below.
"""
from env import config as C

# Matched to V2 PPO so the comparison is controlled.
TRPO_GAMMA = C.PPO_GAMMA                      # 1.0 -- undiscounted, episodic
TRPO_GAE_LAMBDA = C.PPO_GAE_LAMBDA            # 0.95
TRPO_N_STEPS = C.PPO_N_STEPS                  # 2048 per worker
N_TRAINING_ENVS = C.N_TRAINING_ENVS           # 4 workers

# Curriculum: V2 placement budget, pilot joint budget.
CURRICULUM_PLACEMENT_TIMESTEPS = C.CURRICULUM_PLACEMENT_TIMESTEPS   # 32768
PILOT_TOTAL_TIMESTEPS = 262144
PILOT_JOINT_TIMESTEPS = PILOT_TOTAL_TIMESTEPS - CURRICULUM_PLACEMENT_TIMESTEPS

# TRPO-specific. SB3 defaults, recorded rather than tuned: a tuned TRPO against
# an untuned PPO is not a controlled comparison.
TRPO_TARGET_KL = 0.01
TRPO_CG_MAX_STEPS = 15
TRPO_CG_DAMPING = 0.1
TRPO_LINE_SEARCH_SHRINKING_FACTOR = 0.8
TRPO_LINE_SEARCH_MAX_ITER = 10
TRPO_N_CRITIC_UPDATES = 10
TRPO_CRITIC_LEARNING_RATE = C.PPO_LEARNING_RATE   # 3e-4, critic only
TRPO_BATCH_SIZE = C.PPO_BATCH_SIZE               # 256, value-function minibatch only
TRPO_SUB_SAMPLING_FACTOR = 1
TRPO_NORMALIZE_ADVANTAGE = True

# No PPO counterpart:
#   clip_range   -- replaced by the trust region
#   n_epochs     -- TRPO takes one policy step per rollout
#   ent_coef     -- SB3's TRPO hard-codes 0.0, so TRPO trains without the
#                   exploration pressure PPO had at C.PPO_ENT_COEF = 0.01.
#                   A real asymmetry; report it rather than engineering around it.

PILOT_SEED = 100

POLICY_KWARGS = {
    "normalize_images": False,
    "net_arch": {"pi": [C.POLICY_HEAD_DIM], "vf": [C.VALUE_HEAD_DIM]},
}


def model_kwargs(*, device="cpu", seed=PILOT_SEED, n_steps=TRPO_N_STEPS):
    """Constructor arguments for MaskableTRPO/RecoverableTRPO."""
    return {
        "seed": seed,
        "learning_rate": TRPO_CRITIC_LEARNING_RATE,
        "n_steps": n_steps,
        "batch_size": TRPO_BATCH_SIZE,
        "gamma": TRPO_GAMMA,
        "gae_lambda": TRPO_GAE_LAMBDA,
        "target_kl": TRPO_TARGET_KL,
        "cg_max_steps": TRPO_CG_MAX_STEPS,
        "cg_damping": TRPO_CG_DAMPING,
        "line_search_shrinking_factor": TRPO_LINE_SEARCH_SHRINKING_FACTOR,
        "line_search_max_iter": TRPO_LINE_SEARCH_MAX_ITER,
        "n_critic_updates": TRPO_N_CRITIC_UPDATES,
        "sub_sampling_factor": TRPO_SUB_SAMPLING_FACTOR,
        "normalize_advantage": TRPO_NORMALIZE_ADVANTAGE,
        "device": device,
        "verbose": 0,
    }

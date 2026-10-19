"""Correctness checks for masked TRPO.

A masking bug here does not crash. It produces a plausible learning curve for a
constraint that was never enforced, so these run before any training does.
"""
import copy
from functools import partial

import numpy as np
import pytest
import torch as th
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.common.maskable.utils import get_action_masks

from env import config as C
from train.features import SiegeFeatures
from train.train_ppo import make_env
from trpo import config as T
from trpo.maskable_trpo import MaskableTRPO, RecoverableTRPO


def build(n_steps=64, n_envs=2, cls=MaskableTRPO):
    ctors = [partial(make_env, seed=0, rank=i, symmetry=False, fixed_army=True) for i in range(n_envs)]
    env = DummyVecEnv(ctors)
    kwargs = T.model_kwargs(device="cpu", seed=T.PILOT_SEED, n_steps=n_steps)
    kwargs["batch_size"] = 64
    policy_kwargs = dict(T.POLICY_KWARGS)
    policy_kwargs["features_extractor_class"] = SiegeFeatures
    return cls("MultiInputPolicy", env, policy_kwargs=policy_kwargs, **kwargs), env


@pytest.fixture(scope="module")
def trained():
    model, env = build()
    model.learn(256, reset_num_timesteps=False)
    return model, env


def test_uses_maskable_policy_and_buffer():
    model, _ = build()
    from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

    assert isinstance(model.policy, MaskableActorCriticPolicy)
    assert isinstance(model.rollout_buffer, MaskableDictRolloutBuffer)


def test_sampled_actions_are_always_legal(trained):
    model, env = trained
    obs = env.reset()
    for _ in range(150):
        masks = get_action_masks(env)
        actions, _ = model.predict(obs, action_masks=masks, deterministic=False)
        for i, action in enumerate(np.atleast_1d(actions)):
            assert masks[i][int(action)], f"illegal action {action} chosen for env {i}"
        obs, _, _, _ = env.step(actions)


def test_deterministic_actions_are_always_legal(trained):
    model, env = trained
    obs = env.reset()
    for _ in range(150):
        masks = get_action_masks(env)
        actions, _ = model.predict(obs, action_masks=masks, deterministic=True)
        for i, action in enumerate(np.atleast_1d(actions)):
            assert masks[i][int(action)], f"illegal deterministic action {action} for env {i}"
        obs, _, _, _ = env.step(actions)


def test_buffer_masks_match_the_actions_taken(trained):
    """An off-by-one here misaligns the whole update without raising."""
    model, _ = trained
    data = next(model.rollout_buffer.get(batch_size=None))
    masks = data.action_masks.reshape(-1, int(C.ACTION_WAIT) + 1)
    actions = data.actions.long().flatten()
    legal = masks[th.arange(actions.shape[0]), actions]
    assert bool(legal.all()), "a stored action is illegal under its own stored mask"


def test_illegal_actions_have_exactly_zero_probability(trained):
    model, _ = trained
    data = next(model.rollout_buffer.get(batch_size=None))
    dist = model._masked_distribution(data.observations, data.action_masks)
    probs = dist.distribution.probs
    masks = data.action_masks.reshape(probs.shape).bool()
    assert th.all(probs[~masks] == 0), "masked-out actions carry probability"
    assert th.allclose(probs.sum(dim=-1), th.ones(probs.shape[0]), atol=1e-5)


def test_fisher_vector_product_is_finite_under_heavy_masking(trained):
    """Shopping steps leave very few of the 246 actions legal."""
    model, _ = trained
    data = next(model.rollout_buffer.get(batch_size=None))
    # Same order as train(): the policy mutates one shared action_dist, so the
    # no_grad reference must be built FIRST or it detaches the live one.
    with th.no_grad():
        old = copy.copy(model._masked_distribution(data.observations, data.action_masks))
    dist = model._masked_distribution(data.observations, data.action_masks)
    kl = kl_divergence(dist, old).mean()
    assert kl.grad_fn is not None, "KL is detached; the distributions were built in the wrong order"

    params = [p for name, p in model.policy.named_parameters() if "value" not in name]
    grad_kl = th.cat([
        g.reshape(-1)
        for g in th.autograd.grad(kl, params, create_graph=True, allow_unused=True, retain_graph=True)
        if g is not None
    ])
    assert bool(th.isfinite(grad_kl).all())

    vector = th.randn_like(grad_kl)
    jvp = (grad_kl * vector).sum()
    used = [p for p in params if p.requires_grad]
    hvp = th.cat([
        g.reshape(-1)
        for g in th.autograd.grad(jvp, used, retain_graph=True, allow_unused=True)
        if g is not None
    ])
    assert bool(th.isfinite(hvp).all()), "Fisher-vector product produced NaN or inf"


def test_kl_constraint_binds_across_updates():
    """If KL wanders above target_kl, it is being measured unmasked."""
    model, _ = build()
    model.learn(512, reset_num_timesteps=False)
    recorded = model.logger.name_to_value.get("train/kl_divergence_loss")
    assert recorded is not None
    assert recorded <= model.target_kl + 1e-6, f"KL {recorded} exceeded target {model.target_kl}"


def test_failed_line_search_restores_parameters():
    """A rejected step must leave the actor exactly as it was.

    Uses its own model: a corrupted policy would leak into the shared fixture.
    target_kl is driven to a value no candidate can satisfy, which is what the
    line search is for.
    """
    model, _ = build()
    model.learn(128, reset_num_timesteps=False)
    model.target_kl = 1e-30  # positive, so the step length stays well defined
    before = [p.detach().clone() for name, p in model.policy.named_parameters() if "value" not in name]
    model.train()
    after = [p.detach().clone() for name, p in model.policy.named_parameters() if "value" not in name]

    assert model.logger.name_to_value["train/is_line_search_success"] == 0.0
    assert all(bool(th.isfinite(p).all()) for p in after), "line search wrote non-finite parameters"
    for a, b in zip(before, after):
        assert th.allclose(a, b), "actor parameters moved despite the line search failing"


def test_invalid_step_length_does_not_poison_the_policy():
    """Non-positive curvature must restore the actor, not write nan into it."""
    model, _ = build()
    model.learn(128, reset_num_timesteps=False)
    model.target_kl = -1.0  # forces a nan/imaginary step length
    before = [p.detach().clone() for name, p in model.policy.named_parameters() if "value" not in name]
    model.train()
    after = [p.detach().clone() for name, p in model.policy.named_parameters() if "value" not in name]

    assert all(bool(th.isfinite(p).all()) for p in after), "nan step length reached the weights"
    for a, b in zip(before, after):
        assert th.allclose(a, b)


def test_save_reload_gives_identical_actions(tmp_path, trained):
    model, env = trained
    path = tmp_path / "model.zip"
    model.save(path)
    reloaded = MaskableTRPO.load(path, device="cpu")

    obs = env.reset()
    masks = get_action_masks(env)
    a1, _ = model.predict(obs, action_masks=masks, deterministic=True)
    a2, _ = reloaded.predict(obs, action_masks=masks, deterministic=True)
    assert np.array_equal(a1, a2)


def test_recoverable_hook_fires_once_per_update():
    """The contract train/checkpoints.py relies on: one call per completed rollout."""
    model, _ = build(cls=RecoverableTRPO)
    seen = []
    model.after_update = lambda updated: seen.append(updated.num_timesteps)
    model.learn(256, reset_num_timesteps=False)
    assert len(seen) == model._n_updates == 2
    assert seen == sorted(seen)


def test_after_update_is_not_pickled(tmp_path):
    model, _ = build(cls=RecoverableTRPO)
    model.after_update = lambda updated: None
    path = tmp_path / "model.zip"
    model.save(path)  # a bare lambda would fail here if it were being saved
    assert RecoverableTRPO.load(path, device="cpu").after_update is None

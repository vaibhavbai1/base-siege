"""Masked TRPO for the Base Siege action space.

sb3-contrib ships MaskablePPO and TRPO, but no maskable TRPO. The policy,
rollout buffer and rollout collection are reusable from the MaskablePPO side;
the update is not. TRPO evaluates the policy at four separate points -- the
reference distribution, the surrogate gradient, the Fisher-vector products
driving conjugate gradient, and every candidate inside the backtracking line
search -- and each one has to be built under the same action mask the action
was sampled with. A missed site does not raise; it silently enforces the KL
constraint against a distribution over illegal actions.

`train` below is TRPO.train with masks threaded through those sites and
nothing else changed.
"""
from __future__ import annotations

import copy
from functools import partial
from typing import Any, ClassVar, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from sb3_contrib import TRPO
from sb3_contrib.common.maskable.buffers import (
    MaskableDictRolloutBuffer,
    MaskableRolloutBuffer,
)
from sb3_contrib.common.maskable.policies import (
    MaskableActorCriticCnnPolicy,
    MaskableActorCriticPolicy,
    MaskableMultiInputActorCriticPolicy,
)
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from sb3_contrib.common.utils import conjugate_gradient_solver, flat_grad

SelfMaskableTRPO = TypeVar("SelfMaskableTRPO", bound="MaskableTRPO")


class MaskableTRPO(TRPO):
    """TRPO restricted to legal actions by an environment-supplied mask."""

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MaskableActorCriticPolicy,
        "CnnPolicy": MaskableActorCriticCnnPolicy,
        "MultiInputPolicy": MaskableMultiInputActorCriticPolicy,
    }

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        if not isinstance(self.policy, MaskableActorCriticPolicy):
            raise ValueError("Policy must subclass MaskableActorCriticPolicy")

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = MaskableDictRolloutBuffer
            else:
                self.rollout_buffer_class = MaskableRolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )

    def _masked_distribution(self, observations, action_masks):
        """Every distribution in the update is built here, so no site is missed."""
        return self.policy.get_distribution(observations, action_masks=action_masks)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        """As MaskablePPO: fetch the mask each step and store it with the transition."""
        assert isinstance(
            rollout_buffer, (MaskableRolloutBuffer, MaskableDictRolloutBuffer)
        ), "RolloutBuffer doesn't support action masking"
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)
        n_steps = 0
        action_masks = None
        rollout_buffer.reset()

        if use_masking and not is_masking_supported(env):
            raise ValueError("Environment does not support action masking. Consider using ActionMasker wrapper")

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                if use_masking:
                    action_masks = get_action_masks(env)
                actions, values, log_probs = self.policy(obs_tensor, action_masks=action_masks)

            actions = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                action_masks=action_masks,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.on_rollout_end()
        return True

    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ):
        return self.policy.predict(observation, state, episode_start, deterministic, action_masks=action_masks)

    def train(self) -> None:
        """TRPO's update with the action mask applied at every distribution."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        policy_objective_values = []
        kl_divergences = []
        line_search_results = []
        value_losses = []

        # One pass: TRPO takes a single policy step per rollout.
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            action_masks = rollout_data.action_masks
            observations = rollout_data.observations
            actions = rollout_data.actions
            old_log_prob = rollout_data.old_log_prob
            advantages = rollout_data.advantages

            if self.sub_sampling_factor > 1:
                # Masks must be subsampled with everything else or they stop
                # lining up with the actions they were sampled under.
                step = self.sub_sampling_factor
                if isinstance(observations, dict):
                    observations = {key: value[::step] for key, value in observations.items()}
                else:
                    observations = observations[::step]
                actions = actions[::step]
                old_log_prob = old_log_prob[::step]
                advantages = advantages[::step]
                action_masks = action_masks[::step]

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.long().flatten()

            # Order matters and is not interchangeable. The policy owns ONE
            # action_dist object that proba_distribution() mutates in place, so
            # whichever distribution is built last is the one holding a live
            # graph. Build the no_grad reference first and shallow-copy it;
            # building it second silently detaches `distribution` and every
            # gradient below comes back empty.
            with th.no_grad():
                old_distribution = copy.copy(self._masked_distribution(observations, action_masks))

            distribution = self._masked_distribution(observations, action_masks)
            log_prob = distribution.log_prob(actions)

            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratio = th.exp(log_prob - old_log_prob)
            policy_objective = (advantages * ratio).mean()

            # Masked KL -- this is the one whose curvature drives conjugate gradient.
            kl_div = kl_divergence(distribution, old_distribution).mean()

            self.policy.optimizer.zero_grad()
            actor_params, policy_objective_gradients, grad_kl, grad_shape = self._compute_actor_grad(
                kl_div, policy_objective
            )

            hessian_vector_product_fn = partial(self.hessian_vector_product, actor_params, grad_kl)

            search_direction = conjugate_gradient_solver(
                hessian_vector_product_fn,
                policy_objective_gradients,
                max_iter=self.cg_max_steps,
            )

            denominator = float(
                th.matmul(search_direction, hessian_vector_product_fn(search_direction, retain_graph=False))
            )
            squared_step = 2 * self.target_kl / denominator if denominator > 0 else -1.0
            line_search_max_step_size = np.sqrt(squared_step) if squared_step > 0 else np.nan

            line_search_backtrack_coeff = 1.0
            original_actor_params = [param.detach().clone() for param in actor_params]

            is_line_search_success = False
            # A non-positive curvature denominator makes the step length nan or
            # imaginary. Upstream would write that straight into the weights;
            # here an invalid step counts as a failed line search, so the
            # policy is restored instead of being poisoned with nan.
            step_is_valid = bool(np.isfinite(line_search_max_step_size))
            with th.no_grad():
                for _ in range(self.line_search_max_iter if step_is_valid else 0):
                    start_idx = 0
                    for param, original_param, shape in zip(
                        actor_params, original_actor_params, grad_shape, strict=True
                    ):
                        n_params = param.numel()
                        param.data = (
                            original_param.data
                            + line_search_backtrack_coeff
                            * line_search_max_step_size
                            * search_direction[start_idx : (start_idx + n_params)].view(shape)
                        )
                        start_idx += n_params

                    # Candidate policy, masked. The acceptance test below compares
                    # masked against masked; an unmasked rebuild here would make the
                    # KL constraint meaningless while still looking like it works.
                    distribution = self._masked_distribution(observations, action_masks)
                    log_prob = distribution.log_prob(actions)

                    ratio = th.exp(log_prob - old_log_prob)
                    new_policy_objective = (advantages * ratio).mean()
                    kl_div = kl_divergence(distribution, old_distribution).mean()

                    if (kl_div < self.target_kl) and (new_policy_objective > policy_objective):
                        is_line_search_success = True
                        break

                    line_search_backtrack_coeff *= self.line_search_shrinking_factor

                line_search_results.append(is_line_search_success)

                if not is_line_search_success:
                    for param, original_param in zip(actor_params, original_actor_params, strict=True):
                        param.data = original_param.data.clone()
                    policy_objective_values.append(policy_objective.item())
                    kl_divergences.append(0.0)
                else:
                    policy_objective_values.append(new_policy_objective.item())
                    kl_divergences.append(kl_div.item())

        # Critic update. No masks: predict_values returns a scalar and builds
        # no action distribution.
        for _ in range(self.n_critic_updates):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                values_pred = self.policy.predict_values(rollout_data.observations)
                value_loss = th.nn.functional.mse_loss(rollout_data.returns, values_pred.flatten())
                value_losses.append(value_loss.item())

                self.policy.optimizer.zero_grad()
                value_loss.backward()
                # Shared trunk stays under the trust region only.
                for param in actor_params:
                    param.grad = None
                self.policy.optimizer.step()

        self._n_updates += 1
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        self.logger.record("train/policy_objective", np.mean(policy_objective_values))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/kl_divergence_loss", np.mean(kl_divergences))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/is_line_search_success", np.mean(line_search_results))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    def learn(
        self: SelfMaskableTRPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "MaskableTRPO",
        reset_num_timesteps: bool = True,
        use_masking: bool = True,
        progress_bar: bool = False,
    ) -> SelfMaskableTRPO:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar
        )
        callback.on_training_start(locals(), globals())
        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(
                self.env, callback, self.rollout_buffer, self.n_steps, use_masking
            )
            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            if log_interval is not None and iteration % log_interval == 0:
                self.dump_logs(iteration)

            self.train()

        callback.on_training_end()
        return self


class RecoverableTRPO(MaskableTRPO):
    """Same contract as RecoverablePPO: notify persistence after the update ends.

    MaskableTRPO and MaskablePPO both descend from OnPolicyAlgorithm, where
    train() runs exactly once per completed rollout, so the existing recovery,
    pause and atomic-publication machinery applies unchanged.
    """

    after_update = None

    def train(self) -> None:
        super().train()
        if self.after_update is not None:
            self.after_update(self)

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["after_update"]

# train_ppo_lagrangian.py

# ======================================================================================================================
#
#                                           SECTION 1: IMPORTS AND SETUP
#
# ======================================================================================================================

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
from flax import serialization
import distrax
import yaml
import os
import scipy.stats
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

from wrappers import LogWrapper, VecEnv, FrameStackWrapper
from jax_jetbot_env_replicated import JAXJetBotEnvReplicated
from env_configs import EnvParams

def print_pytree_summary(pytree, name="Pytree"):
    """Helper function to print summary statistics of a JAX PyTree."""
    leaves = jax.tree_util.tree_leaves(pytree)
    if not leaves:
        jax.debug.print("{name} is empty.", name=name)
        return

    flat_params = jnp.concatenate([jnp.ravel(x) for x in leaves])
    jax.debug.print(
        "--- {name} Summary ---\n"
        "  Mean: {mean:.6f}, Std: {std:.6f}\n"
        "  Min: {min:.6f}, Max: {max:.6f}\n"
        "  Max Abs: {max_abs:.6f}, Count: {count}",
        name=name,
        mean=jnp.mean(flat_params),
        std=jnp.std(flat_params),
        min=jnp.min(flat_params),
        max=jnp.max(flat_params),
        max_abs=jnp.max(jnp.abs(flat_params)),
        count=len(flat_params)
    )

# ======================================================================================================================
#
#                                     SECTION 2: THE ACTOR-CRITIC NETWORK DEFINITION
#
# ======================================================================================================================

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x, forced_log_std: float = None):
        if self.activation == "tanh": activation = nn.tanh
        else: activation = nn.tanh

        # --- Actor ---
        actor_mean = nn.Dense(192, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        # --- Critic ---
        critic = nn.Dense(192, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)

# ======================================================================================================================
#
#                                     SECTION 3: DATA STRUCTURES AND CONFIGURATION
#
# ======================================================================================================================

class Transition(NamedTuple):
    done: jnp.ndarray; action: jnp.ndarray; value: jnp.ndarray; reward: jnp.ndarray
    cost: jnp.ndarray; log_prob: jnp.ndarray; obs: jnp.ndarray; info: dict
    action_mean: jnp.ndarray
    action_stddev: jnp.ndarray 

config = {
    # --- PPO Algorithm Hyperparameters ---
    "LR": 3e-4,
    "LR_DECAY_START_FRAC": 0.9999,
    "LR_END_FACTOR": 3e-4,
    "ALPHA_LAMBDA_FINAL": 0.5,
    "ALPHA_LAMBDA_DECAY_START_FRAC": 0.9999,
    "UPDATE_EPOCHS": 4,
    "MINIBATCH_SIZE": 65536,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.05,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 1.0,
    "CLIP_VF": True,
    "KL_THRESHOLD": 0.015,

    # --- Environment & Rollout Hyperparameters ---
    "NUM_ENVS": 4096,
    "NUM_STEPS": 32,

    # --- Primal-Dual Loop Hyperparameters ---
    "H_CHUNK_SIZE": 200,
    "NUM_CHUNKS": 110,

    # --- Network Hyperparameters ---
    "ACTIVATION": "tanh",

    # --- Constraint & Lagrangian Hyperparameters ---
    "CONSTRAINT_MODE": "hybrid",
    "INITIAL_LAMBDA": 0.0,
    "ALPHA_LAMBDA_INITIAL": 0.5,
    "COST_LIMIT": 0.02,
    "SAFETY_PROB_LIMIT": 0.98,

    # --- Miscellaneous ---
    "SEED": 42,
}
config["TOTAL_PPO_UPDATES"] = config["NUM_CHUNKS"] * config["H_CHUNK_SIZE"]

# ======================================================================================================================
#
#                                     SECTION 4: THE MAIN TRAINING SETUP FUNCTION (`make_train`)
#
# ======================================================================================================================

def make_train(config):
    config["NUM_MINIBATCHES"] = (config["NUM_ENVS"] * config["NUM_STEPS"]) // config["MINIBATCH_SIZE"]
    assert (config["NUM_ENVS"] * config["NUM_STEPS"]) % config["MINIBATCH_SIZE"] == 0, "Total samples must be divisible by minibatch size"

    env = JAXJetBotEnvReplicated()
    env_params = EnvParams()
    env = LogWrapper(env)

    # --- Apply Frame Stacking ---
    env = FrameStackWrapper(env, num_stack=3)
    
    env = VecEnv(env)

    def train(rng):
        network = ActorCritic(env.action_space(env_params).shape[0], activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)["params"]

        print("--- DEBUG: MODEL INPUT SHAPE VERIFICATION ---")
        input_layer_shape = network_params['Dense_0']['kernel'].shape
        print(f"    Shape of the first layer kernel: {input_layer_shape}")
        print(f"    This means the model expects an input of size: {input_layer_shape[0]}")
        print("-------------------------------------------")
        
        # Setup Learning Rate Schedule
        total_updates = config["TOTAL_PPO_UPDATES"]
        decay_start_step = int(total_updates * config["LR_DECAY_START_FRAC"])

        constant_phase = optax.constant_schedule(config["LR"])
        decay_phase = optax.linear_schedule(
            init_value=config["LR"],
            end_value=config["LR"] * config["LR_END_FACTOR"],
            transition_steps=total_updates - decay_start_step
        )
        
        learning_rate_schedule = optax.join_schedules(
            schedules=[constant_phase, decay_phase],
            boundaries=[decay_start_step]
        )

        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=learning_rate_schedule, eps=1e-5)
        )

        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)
        lagrangian_state = {"lambda_val": config["INITIAL_LAMBDA"]}

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)

        @jax.jit
        def _primal_update_step(runner_state, update_count):
            train_state, env_state, last_obs, rng, lagrangian_state = runner_state

            # --- Rollout Phase ---
            def _env_step(runner_state, step_idx):
                train_state, env_state, last_obs, rng = runner_state

                rng, _rng = jax.random.split(rng)
                pi, value = network.apply({"params": train_state.params}, last_obs)
                
                action = pi.sample(seed=_rng) 
                log_prob = pi.log_prob(action)
                action_mean = pi.mean()
                
                # Extract standard deviation and broadcast to match action_mean shape
                scale = pi.scale_diag
                action_stddev = jnp.broadcast_to(scale, action_mean.shape)
                
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, cost, done, info = env.step(rng_step, env_state, action, env_params)

                transition = Transition(done, action, value, reward, cost, log_prob, last_obs, info, action_mean, action_stddev)

                return (train_state, env_state, obsv, rng), transition

            rollout_runner_state = (train_state, env_state, last_obs, rng)
            rollout_runner_state, traj_batch = jax.lax.scan(_env_step, rollout_runner_state, jnp.arange(config["NUM_STEPS"]))

            # --- Advantage Calculation Phase ---
            train_state, env_state, last_obs, rng = rollout_runner_state
            _, last_val = network.apply({"params": train_state.params}, last_obs)

            lambda_val = lagrangian_state["lambda_val"]

            # Conditional reward shaping
            if config["CONSTRAINT_MODE"] == "prob_only":
                safety_prob = jnp.mean(traj_batch.info["safety_indicator"])
                unsafety_prob = 1.0 - safety_prob
                lagrangian_reward = traj_batch.reward - lambda_val * unsafety_prob
            else:
                lagrangian_reward = traj_batch.reward - lambda_val * traj_batch.cost

            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                delta = transition.reward + config["GAMMA"] * next_value * (1 - transition.done) - transition.value
                gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - transition.done) * gae

                return (gae, transition.value), gae

            lagrangian_traj_batch = traj_batch._replace(reward=lagrangian_reward)
            _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val), lagrangian_traj_batch, reverse=True, unroll=16)

            targets = advantages + traj_batch.value

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            norm_adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # --- PPO Update Phase ---
            def _update_epoch(update_state, epoch_num):
                train_state, rng = update_state
                rng, _rng = jax.random.split(rng)

                batch_size = config["NUM_ENVS"] * config["NUM_STEPS"]
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)

                minibatches = jax.tree_util.tree_map(lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])), shuffled_batch)

                def _loss_fn(params, traj, gae, targ):

                    pi, value = network.apply({"params": params}, traj.obs)
                    log_prob = pi.log_prob(traj.action)

                    value_predicted = value
                    if config["CLIP_VF"]:
                        value_predicted = traj.value + (value_predicted - traj.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    value_loss = 0.5 * jnp.square(targ - value_predicted).mean()

                    ratio = jnp.exp(log_prob - traj.log_prob)
                    loss_actor1 = ratio * gae
                    loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                    entropy = pi.entropy().mean()
                    total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                    approx_kl = ((ratio - 1) - jnp.log(ratio)).mean()

                    return total_loss, {
                        "total_loss": total_loss, "actor_loss": loss_actor,
                        "value_loss": value_loss, "entropy": entropy,
                        "approx_kl": approx_kl, "value_minus_targ_mean": jnp.mean(value - targ),
                        "value_mean": jnp.mean(value), "targ_mean": jnp.mean(targ),
                        "ratio_mean": jnp.mean(ratio),
                    }

                def _update_minbatch(carry, i):
                    train_state, stop_early = carry
                    traj, adv, targ = jax.tree_util.tree_map(lambda x: x[i], minibatches)

                    def perform_update():
                        (total_loss, metrics), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params, traj, adv, targ)
                        new_train_state = train_state.apply_gradients(grads=grads)
                        kl_exceeded = metrics["approx_kl"] > config["KL_THRESHOLD"]
                        return new_train_state, grads, metrics, kl_exceeded

                    def skip_update():
                        dummy_traj, dummy_adv, dummy_targ = jax.tree_util.tree_map(lambda x: x[0], minibatches)
                        value_and_grad_output = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params, dummy_traj, dummy_adv, dummy_targ)
                        (dummy_loss_and_metrics, dummy_grads) = value_and_grad_output
                        _dummy_loss, dummy_metrics = dummy_loss_and_metrics
                        return train_state, dummy_grads, dummy_metrics, True

                    new_train_state, grads, metrics, kl_exceeded = jax.lax.cond(
                        stop_early, skip_update, perform_update
                    )

                    new_carry = (new_train_state, jnp.logical_or(stop_early, kl_exceeded))
                    return new_carry, None

                initial_carry = (train_state, False)
                (train_state, _), _ = jax.lax.scan(
                    _update_minbatch, initial_carry, jnp.arange(config["NUM_MINIBATCHES"]))

                return (train_state, rng), None

            (train_state, rng), _ = jax.lax.scan(_update_epoch, (train_state, rng), jnp.arange(config["UPDATE_EPOCHS"]))

            return (train_state, env_state, last_obs, rng, lagrangian_state), traj_batch

        runner_state = (train_state, env_state, obsv, rng, lagrangian_state)
        all_metrics = []

        jitted_primal_update = jax.jit(_primal_update_step)

        PRINT_INTERVAL = 300 

        for update_step in tqdm(range(config["TOTAL_PPO_UPDATES"]), desc="PPO Updates"):
            runner_state, traj_batch = jitted_primal_update(runner_state, update_step)

            if (update_step + 1) % PRINT_INTERVAL == 0:
                traj_batch_cpu = jax.device_get(traj_batch)
                env_to_inspect = 0
                step_to_inspect = -1
                
                sampled_action = traj_batch_cpu.action[step_to_inspect, env_to_inspect]
                mean_action = traj_batch_cpu.action_mean[step_to_inspect, env_to_inspect]
                stddev_action = traj_batch_cpu.action_stddev[step_to_inspect, env_to_inspect] 
            
                print(f"\n--- [DEBUG REPORT @ Update #{update_step + 1}] ---")
                print(f"    Inspecting Env #{env_to_inspect}, Step #{step_to_inspect} of rollout:")
                print(f"    Policy Mean (post-tanh): {mean_action}")
                print(f"    Policy Std Dev (exp(log_std)): {stddev_action}") 
                print(f"    Sampled Action (mean + N(0,1)*std): {sampled_action}")
                print(f"--------------------------------------------------")

            if (update_step + 1) % config["H_CHUNK_SIZE"] == 0:
                total_dual_updates = config["NUM_CHUNKS"]
                current_dual_update = (update_step + 1) // config["H_CHUNK_SIZE"]
                start_frac = config["ALPHA_LAMBDA_DECAY_START_FRAC"]

                if current_dual_update < total_dual_updates * start_frac:
                    current_alpha_lambda = config["ALPHA_LAMBDA_INITIAL"]
                else:
                    decay_start_update = total_dual_updates * start_frac
                    decay_duration = total_dual_updates - decay_start_update
                    decay_progress = (current_dual_update - decay_start_update) / decay_duration
                    decay_progress = np.clip(decay_progress, 0.0, 1.0) 

                    start_alpha = config["ALPHA_LAMBDA_INITIAL"]
                    end_alpha = config["ALPHA_LAMBDA_FINAL"]
                    current_alpha_lambda = start_alpha + decay_progress * (end_alpha - start_alpha)

                old_lambda_val = runner_state[-1]["lambda_val"]

                train_state, env_state, last_obs, rng, lagrangian_state = runner_state
                traj_batch_cpu = jax.device_get(traj_batch)
                info_cpu = jax.device_get(traj_batch.info)

                update_term = 0.0
                if config["CONSTRAINT_MODE"] == "cost_only":
                    avg_cost_rollout = traj_batch_cpu.cost.mean()
                    update_term = avg_cost_rollout - config["COST_LIMIT"]
                elif config["CONSTRAINT_MODE"] in ["hybrid", "prob_only"]:
                    current_safety_prob = info_cpu.get("safety_indicator", np.zeros(1)).mean()
                    update_term = config["SAFETY_PROB_LIMIT"] - current_safety_prob
                else:
                    raise ValueError(f"Unknown CONSTRAINT_MODE: {config['CONSTRAINT_MODE']}")

                new_lambda_val = np.maximum(0.0, old_lambda_val + current_alpha_lambda * update_term)

                lagrangian_state = {"lambda_val": new_lambda_val}
                runner_state = (train_state, env_state, last_obs, rng, lagrangian_state)

            info_cpu = jax.device_get(traj_batch.info)
            all_returns = info_cpu["returned_episode_returns"][info_cpu["returned_episode"]]
            avg_return_metric = all_returns.mean() if len(all_returns) > 0 else 0.0

            metrics = {
                "update_step": update_step, "avg_return": avg_return_metric,
                "avg_cost": traj_batch.cost.mean(), "std_cost": traj_batch.cost.std(),
                "safety_prob": info_cpu.get("safety_indicator", np.zeros(1)).mean(),
                "std_safety_prob": info_cpu.get("safety_indicator", np.zeros(1)).std(),
                "lambda": runner_state[-1]["lambda_val"],
            }
            for k, v in info_cpu.items():
                if "reward" in k:
                    metrics[k] = v.mean()
                    metrics[f"{k}_std"] = v.std()
            all_metrics.append(metrics)

        return {"runner_state": runner_state, "metrics": all_metrics}

    return train

# ======================================================================================================================
#
#                                     SECTION 5: SCRIPT EXECUTION AND PLOTTING
#
# ======================================================================================================================

if __name__ == "__main__":
    from play_and_plot_trajectories import generate_trajectory_plot
    from env_configs import EnvParams 

    rng = jax.random.PRNGKey(config["SEED"])

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = (
        f"PPO_{config['ACTIVATION']}"
        f"_lr{config['LR']}"
        f"_envs{config['NUM_ENVS']}"
        f"_steps{config['NUM_STEPS']}"
        f"_mb{config['MINIBATCH_SIZE']}"
        f"_epochs{config['UPDATE_EPOCHS']}"
        f"_H{config['H_CHUNK_SIZE']}"
        f"_chunks{config['NUM_CHUNKS']}"
        f"_{timestamp}"
    )

    save_dir = os.path.join("plots", run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved in: {save_dir}")

    env_params = EnvParams()

    env_hyperparams_to_save = {
        "dt": env_params.dt,
        "max_steps_in_episode": env_params.max_steps_in_episode,
        "reward_scale_goal": env_params.reward_scale_goal,
        "reward_scale_distance_delta": env_params.reward_scale_distance_delta,
        "reward_scale_terminated": env_params.reward_scale_terminated,
        "cost_obstacle": env_params.cost_obstacle,
        "min_distance_to_obstacle": env_params.min_distance_to_obstacle,
        "exploration_reward": env_params.exploration_reward,
        "exploration_penalty": env_params.exploration_penalty,
        "grid_size": env_params.grid_cell_size,
    }

    config.update(env_hyperparams_to_save)

    config_save_path = os.path.join(save_dir, "hyperparameters.txt")
    with open(config_save_path, 'w') as f:
        agent_params = {k: v for k, v in config.items() if k not in env_hyperparams_to_save}

        f.write("--- Agent Hyperparameters ---\n")
        for key, value in agent_params.items():
            f.write(f"{key}: {value}\n")

        f.write("\n--- Environment Hyperparameters ---\n")
        for key, value in env_hyperparams_to_save.items():
            f.write(f"{key}: {value}\n")

    print(f"Hyperparameters successfully saved to: {config_save_path}")

    train_fn = make_train(config)

    print("Starting training...")
    out = train_fn(rng)

    print("\nExtracting and saving trained parameters...")
    final_runner_state = jax.device_get(out["runner_state"])
    trained_params = final_runner_state[0].params
    bytes_output = serialization.to_bytes(trained_params)
    with open("trained_params.pkl", "wb") as f: f.write(bytes_output)
    print("Training complete. Trained parameters saved to trained_params.pkl")

    print("\nGenerating trajectory plot...")
    trajectory_save_path = os.path.join(save_dir, "trajectories.png")
    generate_trajectory_plot(trained_params, config, trajectory_save_path)

    metrics = out["metrics"]
    updates = [m["update_step"] for m in metrics]

    def plot_metric_with_std(ax, x, mean_data, label, color, std_data=None, std_alpha=0.15):
        mean = np.array(mean_data)
        ax.plot(x, mean, label=label, color=color, linewidth=2)
        if std_data is not None:
            std = np.array(std_data)
            if len(mean) == len(std):
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=std_alpha)
        ax.grid(True)
        ax.legend()

    print("Generating plot for Reward Components...")
    fig_rewards, axs_rewards = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    fig_rewards.suptitle("Evolution of Reward Components", fontsize=16)
    plot_metric_with_std(axs_rewards[0, 0], updates, [m["goal_reward"] for m in metrics], "Goal Reward", "darkorange", std_data=[m.get("goal_reward_std") for m in metrics])
    axs_rewards[0, 0].set_ylabel("Goal Reward")
    plot_metric_with_std(axs_rewards[0, 1], updates, [m["distance_reward"] for m in metrics], "Distance Reward", "royalblue", std_data=[m.get("distance_reward_std") for m in metrics])
    axs_rewards[0, 1].set_ylabel("Distance Reward")
    plot_metric_with_std(axs_rewards[1, 0], updates, [m["termination_reward"] for m in metrics], "Termination Penalty", "darkviolet", std_data=[m.get("termination_reward_std") for m in metrics])
    axs_rewards[1, 0].set_ylabel("Termination Penalty"); axs_rewards[1, 0].set_xlabel("PPO Update Step")
    plot_metric_with_std(axs_rewards[1, 1], updates, [m["total_primal_reward"] for m in metrics], "Total Primal Reward", "forestgreen", std_data=[m.get("total_primal_reward_std") for m in metrics])
    axs_rewards[1, 1].set_ylabel("Total Primal Reward"); axs_rewards[1, 1].set_xlabel("PPO Update Step")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, "reward_components.png"))

    print("Generating plot for Main Training Metrics...")
    fig_main, axs_main = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
    fig_main.suptitle(f"Training Metrics ({config['CONSTRAINT_MODE']} mode)", fontsize=16)
    plot_metric_with_std(axs_main[0], updates, [m["lambda"] for m in metrics], "Lagrange Multiplier (λ)", "black")
    axs_main[0].set_ylabel("Lagrange Multiplier (λ)")
    plot_metric_with_std(axs_main[1], updates, [m["avg_cost"] for m in metrics], "Average Cost", "r", std_data=[m.get("std_cost") for m in metrics])
    axs_main[1].axhline(y=config["COST_LIMIT"], color='r', linestyle='--', label=f"Cost Limit ({config['COST_LIMIT']})")
    axs_main[1].set_ylabel("Average Cost"); axs_main[1].legend()
    plot_metric_with_std(axs_main[2], updates, [m["safety_prob"] for m in metrics], "Safety Probability", "magenta", std_data=[m.get("std_safety_prob") for m in metrics])
    axs_main[2].axhline(y=config["SAFETY_PROB_LIMIT"], color='magenta', linestyle='--', label=f"Safety Limit ({config['SAFETY_PROB_LIMIT']})")
    axs_main[2].set_ylabel("Safety Probability"); axs_main[2].set_xlabel("PPO Update Step"); axs_main[2].legend(); axs_main[2].set_ylim(0, 1.05)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, "main_metrics.png"))

    print("Generating plot for Correlation Plots...")
    fig_corr, (ax_scatter, ax_timeseries) = plt.subplots(1, 2, figsize=(18, 6))
    fig_corr.suptitle("Cost vs. Safety Probability")
    avg_costs_flat = np.array([m["avg_cost"] for m in metrics])
    safety_probs_flat = np.array([m["safety_prob"] for m in metrics])
    if len(avg_costs_flat) > 1 and len(safety_probs_flat) > 1:
        pearson_r, _ = scipy.stats.pearsonr(avg_costs_flat, safety_probs_flat)
        ax_scatter.set_title(f"Cost vs. Safety Probability (Pearson r = {pearson_r:.4f})")
    ax_scatter.scatter(avg_costs_flat, safety_probs_flat, alpha=0.5, s=10)
    ax_scatter.set_xlabel("Mean Cost"); ax_scatter.set_ylabel("Safety Probability"); ax_scatter.grid(True)
    ax_cost = ax_timeseries; ax_prob = ax_timeseries.twinx()
    ax_cost.set_title("Time-series: Mean Cost (red) vs. Safety Probability (green)")
    ax_cost.set_xlabel("PPO Update Step"); ax_cost.set_ylabel("Mean Cost", color='r')
    ax_cost.plot(updates, avg_costs_flat, color='r', label="Mean Cost"); ax_cost.tick_params(axis='y', labelcolor='r')
    ax_prob.axhline(y=config["SAFETY_PROB_LIMIT"], color='magenta', linestyle='--', label=f"Safety Limit ({config['SAFETY_PROB_LIMIT']})")
    ax_cost.axhline(y=config["COST_LIMIT"], color='r', linestyle='--', label=f"Cost Limit ({config['COST_LIMIT']})")
    ax_prob.set_ylabel("Safety Probability", color='g'); ax_prob.plot(updates, safety_probs_flat, color='g', label="Safety Probability")
    ax_prob.tick_params(axis='y', labelcolor='g'); ax_prob.set_ylim(0, 1.05)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, "correlation_plots.png"))

    print(f"All plots and configs saved in {save_dir}")
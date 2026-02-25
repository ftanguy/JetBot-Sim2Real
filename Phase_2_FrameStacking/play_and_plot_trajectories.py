# play_and_plot_trajectories.py

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
from flax import serialization
import os

from jax_jetbot_env_replicated import JAXJetBotEnvReplicated
from train_ppo_lagrangian import ActorCritic
from env_configs import EnvParams
from wrappers import FrameStackWrapper, LogWrapper 

def plot_maze(ax, maze_config_path="maze_cfg.yaml", env_params=None):
    """Parses the maze configuration and draws the walls, goals, and grid on the given axis."""
    with open(maze_config_path, "r") as f:
        maze_config = yaml.safe_load(f)

    # Plot the background grid if environment parameters are provided
    if env_params is not None:
        x_min, x_max, y_min, y_max = env_params.grid_boundaries
        cell_size = env_params.grid_cell_size
        
        v_lines = np.arange(x_min, x_max + 0.001, cell_size)
        for x in v_lines:
            ax.axvline(x, ymin=0, ymax=1, color='darkgray', linestyle=':', linewidth=0.5, zorder=0)
            
        h_lines = np.arange(y_min, y_max + 0.001, cell_size)
        for y in h_lines:
            ax.axhline(y, xmin=0, xmax=1, color='darkgray', linestyle=':', linewidth=0.5, zorder=0)
    
    # Plot the walls
    for wall in maze_config["maze"]["walls"]:
        ax.plot([wall["start"][0], wall["end"][0]], [wall["start"][1], wall["end"][1]], color='black', linewidth=3)
        
    # Plot the goals
    goals = np.array(maze_config["maze"]["goals"])
    ax.scatter(goals[:, 0], goals[:, 1], c='magenta', marker='*', s=200, label='Goals', zorder=10)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Trajectories at Test Time")
    ax.set_xlabel("X coordinate (m)")
    ax.set_ylabel("Y coordinate (m)")
    ax.legend()

def run_evaluation(trained_params, config, num_episodes=100):
    """Runs a batch of episodes using the trained policy and returns the recorded positions."""
    env = JAXJetBotEnvReplicated()
    env = LogWrapper(env)
    
    # Apply Frame Stacking (k=3) as used during training
    env = FrameStackWrapper(env, num_stack=3)
    
    env_params = EnvParams()
    network = ActorCritic(env.action_space(env_params).shape[0], activation=config["ACTIVATION"])

    @jax.jit
    def run_one_episode(rng_key):
        def _step_fn(carry, unused):
            obs, state, rng_key = carry
            pi, _ = network.apply({"params": trained_params}, obs)
            action = pi.mode()
            rng_key, step_key = jax.random.split(rng_key)

            # Call the wrapped env.step which returns 6 values
            new_obs, new_state, _, _, done, _ = env.step(step_key, state, action, env_params)
            
            # Access position through the nested wrapper states:
            # StackEnvState(env_state=LogEnvState(env_state=EnvState(...)))
            real_pos = new_state.env_state.env_state.pos
            
            return (new_obs, new_state, rng_key), (real_pos, done)

        rng_key, reset_key = jax.random.split(rng_key)
        obs, state = env.reset(reset_key, env_params)
        
        initial_carry = (obs, state, rng_key)
        _, (positions, dones) = jax.lax.scan(_step_fn, initial_carry, None, length=env_params.max_steps_in_episode)

        first_done_idx = jnp.argmax(dones)
        episode_length = jax.lax.cond(
            jnp.any(dones),
            lambda: first_done_idx,
            lambda: env_params.max_steps_in_episode
        )
        return positions, episode_length

    rng = jax.random.PRNGKey(config["SEED"] + 100)
    episode_keys = jax.random.split(rng, num_episodes)
    
    print("Running evaluation to generate trajectories...")
    vmapped_run = jax.vmap(run_one_episode)
    all_positions, all_lengths = vmapped_run(episode_keys)
    print("Evaluation complete.")
    
    all_trajectories = []
    all_positions_np = np.array(all_positions)
    all_lengths_np = np.array(all_lengths)
    for i in range(num_episodes):
        length = all_lengths_np[i]
        all_trajectories.append(all_positions_np[i, :length])
    
    return all_trajectories

def generate_trajectory_plot(trained_params, train_config, save_path):
    """Main function to generate and save the trajectory plot."""
    trajectories = run_evaluation(trained_params, train_config, num_episodes=500)
    
    fig, ax = plt.subplots(figsize=(8, 12))
    env_params = EnvParams()
    plot_maze(ax, env_params=env_params)
    
    trajectories_on_cpu = jax.device_get(trajectories)
    for traj_np in tqdm(trajectories_on_cpu, desc="Plotting trajectories"):
        if traj_np.shape[0] > 1:
            ax.plot(traj_np[:, 0], traj_np[:, 1], alpha=0.5, linewidth=1)

    plt.savefig(save_path)
    print(f"Trajectory plot saved to {save_path}")
    plt.close(fig)

if __name__ == "__main__":
    from train_ppo_lagrangian import config as train_config

    try:
        env = JAXJetBotEnvReplicated()
        env = LogWrapper(env)
        env = FrameStackWrapper(env, num_stack=3)
        
        env_params = EnvParams()
        network = ActorCritic(env.action_space(env_params).shape[0], activation=train_config["ACTIVATION"])
        rng = jax.random.PRNGKey(0)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        empty_params = network.init(rng, init_x)["params"]

        with open("trained_params.pkl", "rb") as f:
            bytes_input = f.read()
        
        trained_policy_params = serialization.from_bytes(empty_params, bytes_input)
        print("Successfully loaded trained parameters from trained_params.pkl")
    
    except FileNotFoundError:
        print("Error: trained_params.pkl not found. Please run training first to generate it.")
        exit()

    generate_trajectory_plot(trained_policy_params, train_config, "trajectories_replica.png")
    plt.show()
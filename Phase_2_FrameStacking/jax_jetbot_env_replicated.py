# jax_jetbot_env_replicated.py

import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct
import yaml
import os

from env_configs import EnvParams

@struct.dataclass
class EnvState:
    """State of a single JetBot environment instance."""
    pos: chex.Array          # (x, y) position
    theta: float             # yaw angle
    previous_distance_to_goal: float
    safety_indicator: float  # 1.0 if safe, 0.0 if unsafe since last reset
    grid_visited: chex.Array # Flattened boolean grid for exploration
    time: int
    
    # Track which goal was selected in the previous step
    last_goal_idx: int

    # GMM Physics parameters specific to this environment instance
    alpha: float
    d_plus: float
    d_minus: float

class JAXJetBotEnvReplicated(environment.Environment):
    """
    A high-fidelity JAX replica of the IsaacLab JetBot Maze Environment.
    Phase 2: Frame Stacking + Gaussian Mixture Model (GMM) Physics.
    """

    def __init__(self):
        super().__init__()
        base_path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_path, "maze_cfg.yaml")
        if not os.path.exists(config_path):
             raise FileNotFoundError(f"maze_cfg.yaml not found at {config_path}.")
        with open(config_path, "r") as f:
            maze_config = yaml.safe_load(f)

        self.walls_start = jnp.array([wall["start"] for wall in maze_config["maze"]["walls"]], dtype=jnp.float32)
        self.walls_end = jnp.array([wall["end"] for wall in maze_config["maze"]["walls"]], dtype=jnp.float32)
        self.goals = jnp.array(maze_config["maze"]["goals"], dtype=jnp.float32)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()
    
    def deadzone_response(self, u, d_minus, d_plus, alpha_minus, alpha_plus):
        """Simulates motor deadzone."""
        positive_speed = alpha_plus * (u - d_plus)
        negative_speed = alpha_minus * (u - d_minus)
        speed = jnp.where(u > d_plus, positive_speed, 0.0)
        speed = jnp.where(u < d_minus, negative_speed, speed)
        return speed

    def _get_best_goal_idx(self, pos, last_goal_idx=None):
        """
        Selects the best goal based on a 'Feasibility Score'.
        Score = Euclidean_Distance + (PENALTY * Is_Blocked) - (BONUS * Is_Last_Target)
        """
        all_dists = jnp.linalg.norm(self.goals - pos, axis=1)

        def check_single_goal_los(goal):
            return self._check_line_of_sight(pos, goal, self.walls_start, self.walls_end)
        
        all_visible = jax.vmap(check_single_goal_los)(self.goals)

        LOS_PENALTY = 0.75
        scores = all_dists + (1.0 - all_visible.astype(jnp.float32)) * LOS_PENALTY

        # Apply hysteresis logic to encourage the robot to stick with its current goal
        HYSTERESIS_BONUS = 0.1 
        
        if last_goal_idx is not None:
             mask = (jnp.arange(self.goals.shape[0]) == last_goal_idx).astype(jnp.float32)
             scores = scores - mask * HYSTERESIS_BONUS

        best_idx = jnp.argmin(scores)
        return best_idx, all_dists, all_visible

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: chex.Array, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        
        # --- 1. PHYSICS (GMM Deadzone Response) ---
        raw_action = action
        u_action = jnp.clip(raw_action, -0.6, 0.6) 
        u_left, u_right = u_action[0], u_action[1]

        # Use the GMM parameters sampled during reset_env
        w_left = self.deadzone_response(u_left, state.d_minus, state.d_plus, state.alpha, state.alpha)
        w_right = self.deadzone_response(u_right, state.d_minus, state.d_plus, state.alpha, state.alpha)

        v = (params.wheel_radius / 2.0) * (w_left + w_right)
        omega = (params.wheel_radius / params.wheelbase) * (w_right - w_left)
        
        new_pos = state.pos + jnp.array([v * jnp.cos(state.theta), v * jnp.sin(state.theta)]) * params.dt
        new_theta = (state.theta + omega * params.dt + jnp.pi) % (2 * jnp.pi) - jnp.pi

        # --- 2. COLLISION & SAFETY ---
        distances_to_obstacles, _ = self._check_collision_ang(new_pos, new_theta, params)
        cost = self._compute_cost(distances_to_obstacles, params)
        is_safe = jnp.all(distances_to_obstacles > params.min_distance_to_obstacle)
        new_safety_indicator = state.safety_indicator * is_safe

        # --- 3. SMART REWARD CALCULATION ---
        best_idx, all_dists, all_visible = self._get_best_goal_idx(new_pos, last_goal_idx=state.last_goal_idx)
        current_target_dist = all_dists[best_idx]
        current_target_visible = all_visible[best_idx]

        goal_has_switched = (best_idx != state.last_goal_idx)

        raw_progress = state.previous_distance_to_goal - current_target_dist
        
        distance_progress = jax.lax.select(
            goal_has_switched,
            0.0, 
            raw_progress 
        )

        los_factor = jax.lax.select(current_target_visible, 1.0, 0.2)

        distance_reward = params.reward_scale_distance_delta * distance_progress * los_factor
        
        reached_goal = current_target_dist <= params.goal_dist_threshold
        goal_reward = params.reward_scale_goal * reached_goal

        out_of_bounds = jnp.linalg.norm(new_pos) > params.termination_radius
        termination_reward = params.reward_scale_terminated * out_of_bounds
        
        cell_idx, is_new_cell = self._check_exploration(new_pos, state.grid_visited, params)
        exploration_reward = jax.lax.cond(
            is_new_cell,
            lambda: params.exploration_reward,
            lambda: params.exploration_penalty
        )
        new_grid_visited = state.grid_visited.at[cell_idx].set(1.0)

        # --- REGULARIZATION PENALTIES ---
        excess_u = jnp.maximum(0.0, jnp.abs(raw_action) - 0.2)
        overpower_penalty = -5.0 * jnp.sum(jnp.square(excess_u))
        centering_penalty = -0.15 * jnp.sum(jnp.square(raw_action)) 

        reward = goal_reward + termination_reward + exploration_reward + distance_reward + overpower_penalty + centering_penalty

        time_out = state.time + 1 >= params.max_steps_in_episode
        done = out_of_bounds | reached_goal | time_out

        # --- 4. STATE UPDATE ---
        new_state = EnvState(
            pos=new_pos,
            theta=new_theta,
            previous_distance_to_goal=current_target_dist, 
            safety_indicator=new_safety_indicator,
            grid_visited=new_grid_visited,
            time=state.time + 1,
            last_goal_idx=best_idx, 
            alpha=state.alpha,
            d_plus=state.d_plus,
            d_minus=state.d_minus
        )
        
        obs = self.get_obs(new_state, params)
        info = {
            "cost": cost,
            "goal_reward": goal_reward,
            "distance_reward": distance_reward,
            "termination_reward": termination_reward,
            "exploration_reward": exploration_reward,
            "total_primal_reward": reward,
        }

        return obs, new_state, reward, done, info

    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
        key, pos_key, theta_key, mode_key, subkey = jax.random.split(key, 5)

        # 1. DECIDE MODE: Puddle or Dry?
        is_puddle = jax.random.uniform(mode_key) < params.prob_puddle

        # 2. HELPER: Sample from the correct Gaussian
        def sample_gmm(is_p, mu_p, std_p, mu_d, std_d, rng):
            mu = jnp.where(is_p, mu_p, mu_d)
            std = jnp.where(is_p, std_p, std_d)
            return mu + std * jax.random.normal(rng)

        # 3. SAMPLE GMM PHYSICS
        keys = jax.random.split(subkey, 4) 

        alpha = sample_gmm(is_puddle, 
                           params.puddle_alpha_mu, params.puddle_alpha_std,
                           params.dry_alpha_mu, params.dry_alpha_std, keys[0])
        alpha = jnp.clip(alpha, 5.0, 100.0)

        d_plus = sample_gmm(is_puddle,
                            params.puddle_dz_mu, params.puddle_dz_std,
                            params.dry_dz_mu, params.dry_dz_std, keys[1])
        d_plus = jnp.maximum(0.0, d_plus) 

        d_minus = sample_gmm(is_puddle,
                             -params.puddle_dz_mu, params.puddle_dz_std, 
                             -params.dry_dz_mu, params.dry_dz_std, keys[2])
        d_minus = jnp.minimum(0.0, d_minus) 

        pos = self._sample_safe_position(pos_key, params)
        theta = jax.random.uniform(theta_key, minval=-jnp.pi, maxval=jnp.pi)
        
        best_idx, all_dists, _ = self._get_best_goal_idx(pos, last_goal_idx=None)
        initial_min_goal_dist = all_dists[best_idx]
        
        grid_visited = jnp.zeros(params.num_grid_cells, dtype=jnp.float32)
        state = EnvState(
            pos=pos,
            theta=theta,
            previous_distance_to_goal=initial_min_goal_dist,
            safety_indicator=1.0,
            grid_visited=grid_visited,
            time=0,
            last_goal_idx=best_idx,
            alpha=alpha,
            d_plus=d_plus,
            d_minus=d_minus
        )
        obs = self.get_obs(state, params)
        return obs, state

    def get_obs(self, state: EnvState, params: EnvParams) -> chex.Array:
        robot_pos = state.pos
        robot_yaw = state.theta
        distances_to_obstacles, obstacle_angle_difference = self._check_collision_ang(robot_pos, robot_yaw, params)
        
        distance_to_goal, yaw_difference = self._get_distance_and_angle_to_specific_goal(
            robot_pos, robot_yaw, state.last_goal_idx
        )
        
        return jnp.concatenate([
            robot_pos,
            distance_to_goal,
            yaw_difference,
            distances_to_obstacles,
            obstacle_angle_difference,
        ])
    
    def _get_distance_and_angle_to_specific_goal(self, robot_pos, robot_yaw, goal_idx):
        distance_to_goal = jnp.linalg.norm(self.goals[goal_idx] - robot_pos)
        goal_pos = self.goals[goal_idx]
        
        vector_to_goal = goal_pos - robot_pos
        angle_to_goal = jnp.arctan2(vector_to_goal[1], vector_to_goal[0])
        yaw_difference = (angle_to_goal - robot_yaw + jnp.pi) % (2 * jnp.pi) - jnp.pi
        
        return jnp.array([distance_to_goal]), jnp.array([yaw_difference])
    
    # --- COLLISION LOGIC ---
    def _check_collision(self, pos: chex.Array, params: EnvParams) -> chex.Array:
        wall_dirs = self.walls_end - self.walls_start
        norm_wall_dirs_sq = jnp.sum(jnp.square(wall_dirs), axis=1)
        norm_wall_dirs_sq = jnp.where(norm_wall_dirs_sq == 0, 1e-6, norm_wall_dirs_sq)
        vec = pos - self.walls_start
        t = jnp.sum(vec * wall_dirs, axis=1) / norm_wall_dirs_sq
        t_clamped = jnp.clip(t, 0.0, 1.0)
        closest_points = self.walls_start + t_clamped[:, jnp.newaxis] * wall_dirs
        distances = jnp.linalg.norm(pos - closest_points, axis=1)
        distances_adjusted = jnp.clip(distances - params.wall_width / 2.0, a_min=0.0)
        return distances_adjusted

    def _check_collision_ang(self, pos: chex.Array, yaw: float, params: EnvParams) -> Tuple[chex.Array, chex.Array]:
        wall_dirs = self.walls_end - self.walls_start
        norm_wall_dirs_sq = jnp.sum(jnp.square(wall_dirs), axis=1)
        norm_wall_dirs_sq = jnp.where(norm_wall_dirs_sq == 0, 1e-6, norm_wall_dirs_sq)
        vec = pos - self.walls_start
        t = jnp.sum(vec * wall_dirs, axis=1) / norm_wall_dirs_sq
        t_clamped = jnp.clip(t, 0.0, 1.0)
        closest_points = self.walls_start + t_clamped[:, jnp.newaxis] * wall_dirs
        distances = jnp.linalg.norm(pos - closest_points, axis=1)
        distances_adjusted = jnp.clip(distances - params.wall_width / 2.0, a_min=0.0)
        _, top_k_indices = jax.lax.top_k(-distances_adjusted, k=3)
        smallest_3_dists = distances_adjusted[top_k_indices]
        closest_3_points = closest_points[top_k_indices]
        vectors_to_obs = closest_3_points - pos
        obs_angles = jnp.arctan2(vectors_to_obs[:, 1], vectors_to_obs[:, 0])
        angle_diffs = (obs_angles - yaw + jnp.pi) % (2 * jnp.pi) - jnp.pi
        return smallest_3_dists, angle_diffs

    def _compute_cost(self, dist_to_obs: chex.Array, params: EnvParams) -> float:
        min_safe_dist = params.min_distance_to_obstacle + params.wheelbase / 2.0
        norm_dist = jnp.maximum(0, 1 - dist_to_obs / min_safe_dist)
        k = 10.0
        exponential_cost = (jnp.exp(k * norm_dist) - 1.0) / (jnp.exp(k) - 1.0)
        return params.cost_obstacle * jnp.sum(exponential_cost)
    
    def _sample_safe_position(self, key: chex.PRNGKey, params: EnvParams, max_attempts: int = 100) -> chex.Array:
        x_min, x_max, y_min, y_max = (-1.4, 1.4, -2.9, 2.9)
        def is_safe(pos):
            dists = self._check_collision(pos, params)
            return jnp.all(dists > params.min_distance_to_obstacle + 0.01)
        def cond_fun(loop_state):
            key, pos, safe, attempts = loop_state
            return jnp.logical_and(jnp.logical_not(safe), attempts < max_attempts)
        def body_fun(loop_state):
            key, pos, safe, attempts = loop_state
            key, subkey = jax.random.split(key)
            new_pos = jax.random.uniform(subkey, shape=(2,), minval=jnp.array([x_min, y_min]), maxval=jnp.array([x_max, y_max]))
            new_safe = is_safe(new_pos)
            return (key, new_pos, new_safe, attempts + 1)
        key, subkey = jax.random.split(key)
        initial_pos = jax.random.uniform(subkey, shape=(2,), minval=jnp.array([x_min, y_min]), maxval=jnp.array([x_max, y_max]))
        initial_safe = is_safe(initial_pos)
        initial_state = (key, initial_pos, initial_safe, 1)
        _final_key, final_pos, _final_safe, _final_attempts = jax.lax.while_loop(cond_fun, body_fun, initial_state)
        return final_pos
    
    def _check_exploration(self, pos: chex.Array, grid: chex.Array, params: EnvParams) -> Tuple[int, bool]:
        x_min, _, y_min, _ = params.grid_boundaries
        cell_size = params.grid_cell_size
        num_cells_x = params.num_grid_cells_x
        num_cells_y = params.num_grid_cells_y
        grid_x = jnp.clip(((pos[0] - x_min) / cell_size), 0, num_cells_x - 1).astype(int)
        grid_y = jnp.clip(((pos[1] - y_min) / cell_size), 0, num_cells_y - 1).astype(int)
        cell_idx = grid_x + num_cells_x * grid_y
        is_new_cell = (grid[cell_idx] == 0)
        return cell_idx, is_new_cell

    def _check_line_of_sight(self, pos, goal_pos, walls_start, walls_end):
        intersects = jax.vmap(self._segments_intersect, in_axes=(None, None, 0, 0))(
            pos, goal_pos, walls_start, walls_end
        )
        is_blocked = jnp.any(intersects)
        return jnp.logical_not(is_blocked)

    def _segments_intersect(self, p1, p2, q1, q2):
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        cond1 = ccw(p1, q1, q2) != ccw(p2, q1, q2)
        cond2 = ccw(p1, p2, q1) != ccw(p1, p2, q2)
        return jnp.logical_and(cond1, cond2)

    @property
    def name(self) -> str: return "JAXJetBot-Replicated-v0"
    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box: return spaces.Box(low=-6.0, high=6.0, shape=(2,), dtype=jnp.float32)
    def observation_space(self, params: EnvParams) -> spaces.Box: return spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(10,), dtype=jnp.float32)
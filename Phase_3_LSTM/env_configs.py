# env_configs.py

# env_configs.py

from flax import struct
import math  # Use standard math for pre-calculation

@struct.dataclass
class EnvParams:
    """
    Static parameters for the JAX JetBot Maze environment.
    """
    # --- Simulation Parameters ---
    dt: float = 1 / 5.0 # 0.2s per step
    max_steps_in_episode: int = 3600

    # --- Reward & Cost Parameters ---
    reward_scale_goal: float = 60.0
    reward_scale_distance_delta: float = 10.0
    reward_scale_terminated: float = -50.0
    cost_obstacle: float = 90.0
    min_distance_to_obstacle: float = 0.15
    exploration_reward: float =  1.0
    exploration_penalty: float = -0.002

    # --- Robot Physical Parameters ---
    wheel_radius: float = 0.032
    wheelbase: float = 0.12
    wall_width: float = 0.05
    termination_radius: float = 3.5
    goal_dist_threshold: float = 0.3
    grid_boundaries: tuple = (-1.5, 1.5, -3.0, 3.0)
    grid_cell_size: float = 0.5

    # --- PHYSICS PARAMETERS (GMM DISTRIBUTIONS) ---
    # Used for sampling the physics values in reset_env
    prob_puddle: float = 0.7  # Probability of encountering a puddle episode

    # Puddle Mode (Stiction)
    puddle_alpha_mu: float = 12.0
    puddle_alpha_std: float = 5.0
    puddle_dz_mu: float = 0.45
    puddle_dz_std: float = 0.1

    # Dry Mode (Grip)
    dry_alpha_mu: float = 50.0
    dry_alpha_std: float = 10.0
    dry_dz_mu: float = 0.05
    dry_dz_std: float = 0.05
    # ----------------------------------------------

    # --- Statically Determined Attributes ---
    num_grid_cells_x: int = struct.field(pytree_node=False, default=None)
    num_grid_cells_y: int = struct.field(pytree_node=False, default=None)
    num_grid_cells: int = struct.field(pytree_node=False, default=None)

    def __post_init__(self):
        if self.num_grid_cells is None:
            x_min, x_max, y_min, y_max = self.grid_boundaries
            num_x = math.ceil((x_max - x_min) / self.grid_cell_size)
            num_y = math.ceil((y_max - y_min) / self.grid_cell_size)
            object.__setattr__(self, "num_grid_cells_x", num_x)
            object.__setattr__(self, "num_grid_cells_y", num_y)
            object.__setattr__(self, "num_grid_cells", num_x * num_y)

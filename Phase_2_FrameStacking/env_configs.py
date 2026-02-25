# env_configs.py

from flax import struct
import math 

@struct.dataclass
class EnvParams:
    """
    Static parameters for the JAX JetBot Maze environment.
    All shape-defining parameters are pre-calculated in Python before JAX compilation.
    """
    # --- Simulation Parameters ---
    dt: float = 1 / 5.0 
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
    wheelbase: float = 0.12
    wheel_radius: float = 0.032

    # --- DOMAIN RANDOMIZATION (GMM) ---
    # Bernoulli prior for selecting the physics regime
    prob_puddle: float = 0.55

    # --- MODE 0: PUDDLE (Low-friction regime) ---
    puddle_alpha_mu: float = 0.3   
    puddle_alpha_std: float = 0.05   
    puddle_dz_mu: float = 0.020      
    puddle_dz_std: float = 0.005     

    # --- MODE 1: DRY (High-friction regime) ---
    dry_alpha_mu: float = 34.7      
    dry_alpha_std: float = 3.0     
    dry_dz_mu: float = 0.011         
    dry_dz_std: float = 0.004

    # --- World & Task Parameters ---
    wall_width: float = 0.05
    termination_radius: float = 3.5
    goal_dist_threshold: float = 0.3 
    grid_boundaries: tuple = (-1.5, 1.5, -3.0, 3.0)
    grid_cell_size: float = 0.5

    # --- Statically Determined Attributes ---
    num_grid_cells_x: int = struct.field(pytree_node=False, default=None)
    num_grid_cells_y: int = struct.field(pytree_node=False, default=None) 
    num_grid_cells: int = struct.field(pytree_node=False, default=None)

    def __post_init__(self):
        """
        Pre-calculate values needed for defining array shapes,
        avoiding JAX's ConcretizationTypeError.
        """
        if self.num_grid_cells is None:
            x_min, x_max, y_min, y_max = self.grid_boundaries
            num_x = math.ceil((x_max - x_min) / self.grid_cell_size)
            num_y = math.ceil((y_max - y_min) / self.grid_cell_size)
            
            object.__setattr__(self, 'num_grid_cells_x', num_x)
            object.__setattr__(self, 'num_grid_cells_y', num_y)
            object.__setattr__(self, 'num_grid_cells', num_x * num_y)
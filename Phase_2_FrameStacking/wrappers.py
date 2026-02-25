# wrappers.py

import jax
import jax.numpy as jnp
import chex
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any
import gymnax
from gymnax.environments import environment, spaces

class GymnaxWrapper(object):
    def __init__(self, env):
        self._env = env
    def __getattr__(self, name):
        return getattr(self._env, name)

@struct.dataclass
class LogEnvState:
    env_state: Any
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int

@struct.dataclass
class StackEnvState:
    env_state: Any
    obs_stack: chex.Array

class LogWrapper(GymnaxWrapper):
    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(
            env_state=env_state,
            episode_returns=0.0,
            episode_lengths=0,
            returned_episode_returns=0.0,
            returned_episode_lengths=0,
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key: chex.PRNGKey, state: LogEnvState, action: Union[int, float], params: Optional[environment.EnvParams] = None) -> Tuple[chex.Array, environment.EnvState, float, float, bool, dict]:
        
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        
        cost = info["cost"]
        
        # Extract safety indicator to expose it to the training loop for Lagrangian updates
        safety_indicator = env_state.safety_indicator

        state = state.replace(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=jnp.where(done, new_episode_return, state.returned_episode_returns),
            returned_episode_lengths=jnp.where(done, new_episode_length, state.returned_episode_lengths),
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.episode_lengths
        info["returned_episode"] = done
        
        info["safety_indicator"] = safety_indicator
        
        return obs, state, reward, cost, done, info
    

class FrameStackWrapper(GymnaxWrapper):
    def __init__(self, env, num_stack=3):
        super().__init__(env)
        self.num_stack = num_stack
        
        # Calculate the expanded observation shape (e.g., 10 * 3 = 30)
        old_shape = self._env.observation_space(self._env.default_params).shape
        self.obs_shape = (old_shape[0] * num_stack,)
        self.obs_dim = old_shape[0]

    def observation_space(self, params):
        return spaces.Box(-jnp.inf, jnp.inf, self.obs_shape)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        obs, env_state = self._env.reset(key, params)
        
        # Initialize the history stack with copies of the first observation
        obs_stack = jnp.tile(obs, self.num_stack)
        
        state = StackEnvState(env_state=env_state, obs_stack=obs_stack)
        return obs_stack, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        obs, env_state, reward, cost, done, info = self._env.step(key, state.env_state, action, params)
        
        # Bypass array concatenation if no stacking is applied
        if self.num_stack == 1:
            new_stack = obs
        else:
            # Shift stack: drop the oldest observation, append the newest one
            current_stack_without_oldest = state.obs_stack[self.obs_dim:]
            new_stack = jnp.concatenate([current_stack_without_oldest, obs])
        
        new_state = StackEnvState(env_state=env_state, obs_stack=new_stack)
        
        return new_stack, new_state, reward, cost, done, info

class VecEnv(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))
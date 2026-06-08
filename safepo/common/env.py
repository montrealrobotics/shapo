# Copyright 2023 OmniSafeAI Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import annotations
try :
    from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.ShadowHandCatchOver2underarm_Safe_finger import ShadowHandCatchOver2Underarm_Safe_finger
    from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.ShadowHandCatchOver2underarm_Safe_joint import ShadowHandCatchOver2Underarm_Safe_joint
    from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.ShadowHandOver_Safe_finger import ShadowHandOver_Safe_finger
    from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.ShadowHandOver_Safe_joint import ShadowHandOver_Safe_joint
    from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.freight_franka_pick_and_place import FreightFrankaPickAndPlace
    from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.freight_franka_close_drawer import FreightFrankaCloseDrawer
    from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.base.multi_vec_task import ShadowHandMultiVecTaskPython, FreightFrankaMultiVecTaskPython
    from safepo.common.wrappers import GymnasiumIsaacEnv
except ImportError:
    pass
from typing import Callable
import safety_gymnasium
from safety_gymnasium.wrappers import SafeAutoResetWrapper, SafeRescaleAction, SafeUnsqueeze
from safety_gymnasium.vector.async_vector_env import SafetyAsyncVectorEnv
from safepo.common.wrappers import ShareSubprocVecEnv, ShareDummyVecEnv, ShareEnv, SafeNormalizeObservation, MultiGoalEnv, SafeGymNormalizeObservation
import gymnasium as gym


import gymnasium as gym 
from gymnasium.vector.async_vector_env import AsyncState, AsyncVectorEnv


def make_sa_gymrobot_env(args,num_envs: int, env_id: str, seed: int|None = None):
    """
    Creates and wraps an environment based on the specified parameters.

    Args:
        num_envs (int): Number of parallel environments.
        env_id (str): ID of the environment to create.
        seed (int or None, optional): Seed for the random number generator. Default is None.

    Returns:
        env: The created and wrapped environment.
        obs_space: The observation space of the environment.
        act_space: The action space of the environment.
        
    Examples:
        >>> from safepo.common.env import make_sa_mujoco_env
        >>> 
        >>> env, obs_space, act_space = make_sa_mujoco_env(
        >>>     num_envs=1, 
        >>>     env_id="SafetyPointGoal1-v0", 
        >>>     seed=0
        >>> )
    """
    if num_envs > 1:
        def create_env() -> Callable:
            """Creates an environment that can enable or disable the environment checker."""
            env = gym.make(env_id)
            env = SafeRescaleAction(env, -1.0, 1.0)
            return env
        env_fns = [create_env for _ in range(num_envs)]
        env = AsyncVectorEnv(env_fns)
        env = SafeGymNormalizeObservation(env)
        try:
            env.reset(seed=seed)
        except:
            pass
        obs_space = env.single_observation_space
        act_space = env.single_action_space
    else:
        env = gym.make(env_id, render_mode="rgb_array")
        try:
            env.reset(seed=seed)
        except:
            pass
        obs_space = env.observation_space
        act_space = env.action_space
        env = SafeGymRobotAutoResetWrapper(env)
        # env = SafeRescaleAction(env, -1.0, 1.0)
        env = SafeGymNormalizeObservation(env)
        # env = SafeUnsqueeze(env)
    
    return env, obs_space, act_space




from gymnasium.wrappers.autoreset import AutoResetWrapper


class SafeGymRobotAutoResetWrapper(AutoResetWrapper):
    """A class for providing an automatic reset functionality for gymnasium environments when calling :meth:`step`.

     - ``new_obs`` is the first observation after calling ``self.env.reset()``
     - ``final_reward`` is the reward after calling ``self.env.step()``, prior to calling ``self.env.reset()``.
     - ``final_terminated`` is the terminated value before calling ``self.env.reset()``.
     - ``final_truncated`` is the truncated value before calling ``self.env.reset()``. Both ``final_terminated`` and ``final_truncated`` cannot be False.
     - ``info`` is a dict containing all the keys from the info dict returned by the call to ``self.env.reset()``,
       with an additional key "final_observation" containing the observation returned by the last call to ``self.env.step()``
       and "final_info" containing the info dict returned by the last call to ``self.env.step()``.

    Warning: When using this wrapper to collect roll-outs, note that when :meth:`Env.step` returns ``terminated`` or ``truncated``, a
        new observation from after calling :meth:`Env.reset` is returned by :meth:`Env.step` alongside the
        final reward, terminated and truncated state from the previous episode.
        If you need the final state from the previous episode, you need to retrieve it via the
        "final_observation" key in the info dict.
        Make sure you know what you're doing if you use this wrapper!
    """  # pylint: disable=line-too-long

    def step(self, action):
        """A class for providing an automatic reset functionality for gymnasium environments when calling :meth:`step`.

        Args:
            env (gym.Env): The environment to apply the wrapper
        """  # pylint: disable=line-too-long
        obs, reward, terminated, truncated, info = self.env.step(action)
        # cost = info["cost"]
        if terminated or truncated:
            new_obs, new_info = self.env.reset()
            assert (
                'final_observation' not in new_info
            ), 'info dict cannot contain key "final_observation" '
            assert 'final_info' not in new_info, 'info dict cannot contain key "final_info" '

            new_info['final_observation'] = obs
            new_info['final_info'] = info

            obs = new_obs
            info = new_info

        return obs, reward, terminated, truncated, info



def make_sa_safetygym_env(cfg, num_envs: int, env_id: str, seed: int|None = None):
    """
    Creates and wraps an environment based on the specified parameters.

    Args:
        num_envs (int): Number of parallel environments.
        env_id (str): ID of the environment to create.
        seed (int or None, optional): Seed for the random number generator. Default is None.

    Returns:
        env: The created and wrapped environment.
        obs_space: The observation space of the environment.
        act_space: The action space of the environment.
        
    Examples:
        >>> from safepo.common.env import make_sa_mujoco_env
        >>> 
        >>> env, obs_space, act_space = make_sa_mujoco_env(
        >>>     num_envs=1, 
        >>>     env_id="SafetyPointGoal1-v0", 
        >>>     seed=0
        >>> )
    """
    if num_envs > 1:
        def create_env() -> Callable:
            """Creates an environment that can enable or disable the environment checker."""
            try:
                env = safety_gymnasium.make(env_id, early_termination=cfg.early_termination, term_cost=cfg.term_cost, failure_penalty=cfg.failure_penalty, reward_goal=cfg.reward_goal, reward_distance=cfg.reward_distance)
            except:
                env = safety_gymnasium.make(env_id)
            env = SafeRescaleAction(env, -1.0, 1.0)
            return env
        env_fns = [create_env for _ in range(num_envs)]
        env = SafetyAsyncVectorEnv(env_fns)
        env = SafeNormalizeObservation(env)
        env.reset(seed=seed)
        obs_space = env.single_observation_space
        act_space = env.single_action_space
    else:
        try:
            env = safety_gymnasium.make(env_id, early_termination=cfg.early_termination, term_cost=cfg.term_cost, failure_penalty=cfg.failure_penalty, reward_goal=cfg.reward_goal, reward_distance=cfg.reward_distance)
        except:
            env = safety_gymnasium.make(env_id)
        env.reset(seed=seed)
        obs_space = env.observation_space
        act_space = env.action_space
        env = SafeAutoResetWrapper(env)
        env = SafeRescaleAction(env, -1.0, 1.0)
        env = SafeNormalizeObservation(env)
        env = SafeUnsqueeze(env)
    
    return env, obs_space, act_space



def make_sa_mujoco_env(cfg, num_envs: int, env_id: str, seed: int|None = None):
    """
    Creates and wraps an environment based on the specified parameters.

    Args:
        num_envs (int): Number of parallel environments.
        env_id (str): ID of the environment to create.
        seed (int or None, optional): Seed for the random number generator. Default is None.

    Returns:
        env: The created and wrapped environment.
        obs_space: The observation space of the environment.
        act_space: The action space of the environment.
        
    Examples:
        >>> from safepo.common.env import make_sa_mujoco_env
        >>> 
        >>> env, obs_space, act_space = make_sa_mujoco_env(
        >>>     num_envs=1, 
        >>>     env_id="SafetyPointGoal1-v0", 
        >>>     seed=0
        >>> )
    """
    if num_envs > 1:
        def create_env() -> Callable:
            """Creates an environment that can enable or disable the environment checker."""
            env = gym.make(env_id)
            env = CostWrapper(env)
            return env
        env_fns = [create_env for _ in range(num_envs)]
        env = SafetyAsyncVectorEnv(env_fns)
        env.action_space.seed(seed)
        obs_space = env.single_observation_space
        act_space = env.single_action_space
    else:
        env = gym.make(env_id)
        env = CostWrapper(env)
        env.reset()
        obs_space = env.observation_space
        act_space = env.action_space
        env.action_space.seed(seed)
        env = SafeAutoResetWrapper(env)
    
    return env, obs_space, act_space


class CostWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.cost_threshold = 1.0
        self.early_termination = False
        self.failure_penalty = 0.0
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, terminated, truncated, info
    
    def reset(self):
        obs, _ = self.env.reset()
        return obs, {}

def make_sa_isaac_env(args, cfg, sim_params):
    """
    Creates and returns a VecTaskPython environment for the single agent Isaac Gym task.

    Args:
        args: Command-line arguments.
        cfg: Configuration for the environment.
        cfg_train: Training configuration.
        sim_params: Parameters for the simulation.

    Returns:
        env: VecTaskPython environment for the single agent Isaac Gym task.

    Warning:
        SafePO's single agent Isaac Gym task is not ready for use yet.
    """
    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.device

    cfg["seed"] = args.seed
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]
    task = eval(args.task)(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        device_type=args.device,
        device_id=device_id,
        headless=args.headless,
        is_multi_agent=False)
    try:
        env = GymnasiumIsaacEnv(task, rl_device)
    except ModuleNotFoundError:
        env = None

    return env

def make_ma_mujoco_env(scenario, agent_conf, seed, cfg_train):
    """
    Creates and returns a multi-agent environment using MuJoCo scenarios.

    Args:
        args: Command-line arguments.
        cfg_train: Training configuration.

    Returns:
        env: A multi-agent environment.
    """
    def get_env_fn(rank):
        def init_env():
            """
            Initializes and returns a ShareEnv instance for the given rank.

            Returns:
                env: Initialized ShareEnv instance.
            """
            env=ShareEnv(
                scenario=scenario,
                agent_conf=agent_conf,
            )
            env.reset(seed=seed + rank * 1000)
            return env

        return init_env

    if cfg_train['n_rollout_threads']== 1:
        return ShareDummyVecEnv([get_env_fn(0)], cfg_train['device'])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(cfg_train['n_rollout_threads'])])

def make_ma_multi_goal_env(task, seed, cfg_train):
    """
    Creates and returns a multi-agent environment using MuJoCo scenarios.

    Args:
        args: Command-line arguments.
        cfg_train: Training configuration.

    Returns:
        env: A multi-agent environment.
    """
    def get_env_fn(rank):
        def init_env():
            """
            Initializes and returns a ShareEnv instance for the given rank.

            Returns:
                env: Initialized ShareEnv instance.
            """
            env=MultiGoalEnv(
                task=task,
                seed=seed,
            )
            return env

        return init_env
    
    if cfg_train['n_rollout_threads']== 1:
        return ShareDummyVecEnv([get_env_fn(0)], cfg_train['device'])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(cfg_train['n_rollout_threads'])])

def make_ma_isaac_env(args, cfg, cfg_train, sim_params, agent_index):
    """
    Creates and returns a multi-agent environment for the Isaac Gym task.

    Args:
        args: Command-line arguments.
        cfg: Configuration for the environment.
        cfg_train: Training configuration.
        sim_params: Parameters for the simulation.
        agent_index: Index of the agent within the multi-agent environment.

    Returns:
        env: A multi-agent environment for the Isaac Gym task.
    """
    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.device

    cfg["seed"] = cfg_train.get("seed", -1)
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]
    task = eval(args.task)(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        device_type=args.device,
        device_id=device_id,
        headless=args.headless,
        agent_index=agent_index,
        is_multi_agent=True)
    task_name = task.__class__.__name__
    if "ShadowHand" in task_name:
        env = ShadowHandMultiVecTaskPython(task, rl_device)
    elif "FreightFranka" in task_name:
        env = FreightFrankaMultiVecTaskPython(task, rl_device)
    else:
        raise NotImplementedError

    return env

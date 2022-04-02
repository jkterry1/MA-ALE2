import importlib
import numpy as np
from supersuit import resize_v0, frame_skip_v0, reshape_v0, max_observation_v0
import supersuit as ss
from pettingzoo.atari.base_atari_env import BaseAtariEnv, ParallelAtariEnv
from itertools import cycle
from all.environments import MultiagentPettingZooEnv, GymVectorEnvironment
from all.environments.vector_env import GymVectorEnvironment
import torch


class MAPZEnvSteps(MultiagentPettingZooEnv):
    """`MultiagentPettingZooEnv` that includes the current num
    steps within current episode in the returned state dict.
    This is necessary for NFSP to determine which policy to use for a
    given episode.
    """
    def __init__(self, zoo_env, name, device='cuda'):
        MultiagentPettingZooEnv.__init__(self, zoo_env, name, device=device)
        self._ep_steps = None

    def _add_env_steps(self, state):
        cur_agent = state['agent']
        state['ep_step'] = self._ep_steps[cur_agent]
        return state

    def reset(self):
        self._ep_steps = {ag: 0 for ag in self.agents}
        self._agent_looper = cycle(self.agents)
        return super(MAPZEnvSteps, self).reset()

    def step(self, action):
        self._ep_steps[next(self._agent_looper)] += 1
        return super(MAPZEnvSteps, self).step(action)

    def last(self):
        state = super(MAPZEnvSteps, self).last()
        return self._add_env_steps(state)


def make_env(env_name, vs_builtin=False, device='cuda'):
    if vs_builtin:
        env = get_base_builtin_env(env_name)
    else:
        env = importlib.import_module('pettingzoo.atari.{}'.format(env_name)).env(obs_type='grayscale_image')
    env = ss.max_observation_v0(env, 2)  # sequential observation: (env, 2)== maximum 2 frames can be observation and then skip
    env = ss.frame_skip_v0(env, 4)  # frame skipping: (env, 4)==skipping 4 or 5 (randomly) frames
    env = ss.resize_v0(env, 84, 84)  # resizing
    env = ss.reshape_v0(env, (1, 84, 84))  # reshaping
    # FIXME: this breaks nfsp rainbow for some reason?
    # env = InvertColorAgentIndicator(env)  # Observation indicator for each agent
    return env

def make_vec_env(env_name, device, vs_builtin=False, num_envs=16):
    if vs_builtin:
        env = get_base_builtin_env(env_name, parallel=True)
    else:
        env = importlib.import_module('pettingzoo.atari.{}'.format(env_name)).parallel_env(obs_type='grayscale_image')
    env = ss.max_observation_v0(env, 2)     # stacking observation: (env, 2)==stacking 2 frames as observation
    env = ss.frame_skip_v0(env, 4)          # frame skipping: (env, 4)==skipping 4 or 5 (randomly) frames
    env = ss.resize_v0(env, 84, 84)         # resizing
    env = ss.reshape_v0(env, (1, 84, 84))   # reshaping (expand dummy channel dimension)
    env = InvertColorAgentIndicator(env)    # reshapes to (3, 84, 84)
    env = ss.pettingzoo_env_to_vec_env_v1(env) # -> (n_agents, 3, 84, 84)
    env = ss.concat_vec_envs_v1(env, num_envs, # -> (n_envs*n_agents, 3, 84, 84)
                                num_cpus=num_envs//4, base_class='stable_baselines3')
    env = GymVectorEnvironment(env, env_name, device=device) # -> (n_envs*n_agents,) shape StateArray
    return env


def recolor_surround(surround_env):
    def obs_fn(observation, obs_space):
        new_obs = np.copy(observation)
        mask1 = (observation == 104)
        mask2 = (observation == 110)
        mask3 = (observation == 179)
        mask4 = (observation == 149)
        new_obs[mask1] = 90
        new_obs[mask2] = 147
        new_obs[mask3] = 64
        new_obs[mask4] = 167
        return new_obs
    return ss.observation_lambda_v0(surround_env, obs_fn)


def get_base_builtin_env(env_name, parallel=False):
    name_no_version = env_name.rsplit("_", 1)[0]
    if parallel:
        env = ParallelAtariEnv(game=name_no_version, num_players=1, obs_type='grayscale_image')
    else:
        env = BaseAtariEnv(game=name_no_version, num_players=1, obs_type='grayscale_image')
    if name_no_version == "surround":
        env = recolor_surround(env)
    return env


def InvertColorAgentIndicator(env):
    """
    Agent indicator for better convergence: Idea from
    Terry, Justin K., et al. "Revisiting parameter sharing in multi-agent deep reinforcement learning." arXiv preprint arXiv:2005.13625 (2020).
    https://arxiv.org/pdf/2005.13625.pdf
    """
    def modify_obs(obs, obs_space, agent):
        num_agents = len(env.possible_agents)
        agent_idx = env.possible_agents.index(agent)
        if num_agents <= 2:
            # Color flipping instead or rotation
            if agent_idx == 1:
                rotated_obs = 255 - obs
            else:
                rotated_obs = obs
        elif num_agents == 4:
            # Color rotation
            rotated_obs = ((255*agent_idx)//4 + obs) % 255
        indicator = np.zeros((2, )+obs.shape[1:],dtype="uint8")
        indicator[0] = 255 * (agent_idx % 2)
        indicator[1] = 255 * (((agent_idx+1) // 2) % 2)
        return np.concatenate([obs, rotated_obs, indicator], axis=0)
    env = ss.observation_lambda_v0(env, modify_obs)
    env = ss.pad_observations_v0(env)
    return env

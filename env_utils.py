import importlib
import gym
from gym.spaces import Box, Discrete
import numpy as np
from itertools import cycle
import supersuit as ss
from supersuit.generic_wrappers.utils.base_modifier import BaseModifier
from supersuit.generic_wrappers.utils.shared_wrapper_util import shared_wrapper
from supersuit.utils.frame_stack import stack_obs_space, stack_init, stack_obs
from supersuit.utils.wrapper_chooser import WrapperChooser
from pettingzoo.atari.base_atari_env import BaseAtariEnv, ParallelAtariEnv
from pettingzoo.utils.wrappers import BaseWrapper, BaseParallelWraper
from all.environments import MultiagentPettingZooEnv, GymVectorEnvironment


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
        # self._episodes_seen += 1
        self._ep_steps = {ag: 0 for ag in self.agents}
        self._agent_looper = cycle(self.agents)
        return super(MAPZEnvSteps, self).reset()
        # return self._add_env_steps(state)

    def step(self, action):
        self._ep_steps[next(self._agent_looper)] += 1
        return super(MAPZEnvSteps, self).step(action)
        # return self._add_env_steps(state)

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
        env = get_base_builtin_env(env_name, parallel=True, full_action_space=False)
    else:
        env = importlib.import_module('pettingzoo.atari.{}'.format(env_name)).parallel_env(
            obs_type='grayscale_image',
            full_action_space=False,
        )
    env = noop_reset_v0(env)                # skip randint # steps beginning of each episode
    env = ss.max_observation_v0(env, 2)     # stacking observation: (env, 2)==stacking 2 frames as observation
    env = ss.frame_skip_v0(env, 4)          # frame skipping: (env, 4)==skipping 4 or 5 (randomly) frames
    env = ss.clip_reward_v0(env)            # rewards in (-1, 1)
    env = ss.resize_v0(env, 84, 84)         # resizing
    env = ss.reshape_v0(env, (1, 84, 84))   # reshaping (expand dummy channel dimension)
    env = frame_stack_v2(env)               # FIXME: why does this reshape to (1,84,336)? Shouldn't it be like (4,84,84)?
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


def get_base_builtin_env(env_name, parallel=False, full_action_space=True):
    name_no_version = env_name.rsplit("_", 1)[0]
    if parallel:
        env = ParallelAtariEnv(game=name_no_version, num_players=1,
                               obs_type='grayscale_image', full_action_space=full_action_space)
    else:
        env = BaseAtariEnv(game=name_no_version, num_players=1,
                           obs_type='grayscale_image', full_action_space=full_action_space)
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


def frame_stack_v2(env, stack_size=4):
    assert isinstance(stack_size, int), "stack size of frame_stack must be an int"

    class FrameStackModifier(BaseModifier):
        def modify_obs_space(self, obs_space):
            if isinstance(obs_space, Box):
                assert 1 <= len(obs_space.shape) <= 3, "frame_stack only works for 1, 2 or 3 dimensional observations"
            elif isinstance(obs_space, Discrete):
                pass
            else:
                assert False, "Stacking is currently only allowed for Box and Discrete observation spaces. The given observation space is {}".format(
                    obs_space)

            self.old_obs_space = obs_space
            self.observation_space = stack_obs_space(obs_space, stack_size)
            return self.observation_space

        def reset(self):
            self.stack = stack_init(self.old_obs_space, stack_size)
            self.reset_flag = True

        def modify_obs(self, obs):
            if self.reset_flag:
                for _ in range(stack_size):
                    self.stack = stack_obs(
                        self.stack,
                        obs,
                        self.old_obs_space,
                        stack_size,
                    )

                self.reset_flag = False
            else:
                self.stack = stack_obs(
                    self.stack,
                    obs,
                    self.old_obs_space,
                    stack_size,
                )

            return self.stack

        def get_last_obs(self):
            return self.stack

    return shared_wrapper(env, FrameStackModifier)

class noop_reset_gym(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0

        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

class noop_reset_par(BaseParallelWraper):
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = {}
        for agent in self.possible_agents: # Check
            self.noop_action[agent] = 0

    def reset(self):
        obs = super().reset()
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = np.random.randint(1, self.noop_max + 1)
        assert noops > 0

        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset()
        return obs


noop_reset_v0 = WrapperChooser(gym_wrapper=noop_reset_gym, parallel_wrapper=noop_reset_par)
import argparse
from posix import environ
import torch
import random
torch.set_num_threads(1)

from algorithms.shared_rainbow import make_rainbow_preset
from algorithms.independent_rainbow import make_indepedent_rainbow
# from ppo_ram import make_ppo_ram_vec
from algorithms.shared_ppo import make_ppo_vec, make_ppo_vec_largenet
from algorithms.nfsp import make_nfsp_rainbow
from env_utils import get_base_builtin_env, InvertColorAgentIndicator

import importlib
from supersuit import resize_v0, frame_skip_v0, reshape_v0, max_observation_v0
import supersuit as ss
from all.environments import MultiagentPettingZooEnv
from all.environments import GymVectorEnvironment

trainer_types = {
    "shared_rainbow": make_rainbow_preset,
    "independent_rainbow": make_indepedent_rainbow,
    "shared_ppo": make_ppo_vec,
    # "shared_ppo_ram": make_ppo_ram_vec,
    "shared_ppo_largenet": make_ppo_vec_largenet,
    "nfsp_rainbow": make_nfsp_rainbow,
}

def make_env_test(env_name, vs_builtin=False):
    if vs_builtin:
        env = get_base_builtin_env(env_name)
    else:
        env = importlib.import_module('pettingzoo.atari.{}'.format(env_name)).env(obs_type='grayscale_image')
    env = ss.max_observation_v0(env, 2) # sequential observation: (env, 2)== maximum 2 frames can be observation and then skip
    env = ss.frame_skip_v0(env, 4) # frame skipping: (env, 4)==skipping 4 frames
    env = ss.resize_v0(env, 84, 84) # resizing
    env = ss.reshape_v0(env, (1, 84, 84)) # reshaping
    env = ss.black_death_v2(env)
    env = InvertColorAgentIndicator(env) # handled by body
    return env

def make_vec_env_test(env_name, device):
    env = importlib.import_module('pettingzoo.atari.{}'.format(env_name)).parallel_env(obs_type='grayscale_image')
    env = ss.max_observation_v0(env, 2) # stacking observation: (env, 2)==stacking 2 frames as observation
    env = ss.frame_skip_v0(env, 4) # frame skipping: (env, 4)==skipping 4 or 5 (randomly) frames
    env = ss.resize_v0(env, 84, 84) # resizing
    env = ss.reshape_v0(env, (1, 84, 84)) # reshaping (expand dummy channel dimension)
    env = ss.black_death_v2(env) # Give black observation (zero array) and zero reward to dead agents
    env = InvertColorAgentIndicator(env)
    env = ss.pettingzoo_env_to_vec_env_v0(env)
    env = ss.concat_vec_envs_v0(env, 4, num_cpus=2, base_class='stable_baselines3')
    env = GymVectorEnvironment(env, env_name, device=device)
    
    return env

class TestRandom:
    def __init__(self,env):
        self._env = env
        
    def act(self, state):
        return random.randint(0,17)
    def act_torch(self, state):
        return torch.randint(0,17, (self._env.num_envs,))

def main():
    parser = argparse.ArgumentParser(description="Run an multiagent Atari benchmark.")
    parser.add_argument("env", help="Name of the Atari game (e.g. Pong).")
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()

    def _debug(env):
        agent = TestRandom(env)
        state = env.reset()
        first_agent = state['agent']
        if args.raw:
            print("reset state:",state)
        else:
            print("reset state:",state['observation'].shape)

        # For non-vector env
        if not isinstance(env, GymVectorEnvironment):
            for _ in env.agent_iter():
                state = env.step(agent.act(state))
                if args.raw:
                    print("step state:", state)
                else:
                    print("step state:",state['observation'].shape)
                if state['agent']==first_agent:
                    break
        # For vector env       
        else:
            state = env.step(agent.act_torch(state))
            if args.raw:
                print("step state:", state)
            else:
                print("step state:",state['observation'].shape)



    print("[Debug] Test1: make_env / original")
    env = make_env_test(args.env, vs_builtin=False)
    env = MultiagentPettingZooEnv(env, args.env, 'cpu')
    _debug(env)
    
    print("[Debug] Test2: make_env / vs_builtin")
    env = make_env_test(args.env, vs_builtin=True)
    env = MultiagentPettingZooEnv(env, args.env, 'cpu')
    _debug(env)

    print("[Debug] Test3: make_vec_env")
    env = make_vec_env_test(args.env, 'cpu')
    _debug(env)


    # for trainer_type in trainer_types.keys():
    #     experiment, preset, env = trainer_types[trainer_type](args.env, 'cpu', 10000,
    #                                                             num_frames=100000)
    #     print("[Debug] Test%d: %s"%(i,trainer_type))
    #     state = env.reset()
    #     # print("reset state:",state)
    #     print("reset state shape:",state['observation'].shape)
    #     state = env.step(agent.act(state))
    #     # print("step state:", state)
    #     print("step state shape:",state['observation'].shape)
    #     i += 1


if __name__ == "__main__":
    main()

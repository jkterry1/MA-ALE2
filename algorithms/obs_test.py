from collections import deque
import os
import supersuit as ss
from all.environments import GymVectorEnvironment
import numpy as np
import torch
from PIL import Image

def make_vec_env(n_env, env_name='boxing_v1', num_cpus=16, device='cpu'):
    import importlib
    env = importlib.import_module(f'pettingzoo.atari.{env_name}').parallel_env(
        obs_type='grayscale_image')
    env = ss.max_observation_v0(env, 2)
    env = ss.frame_skip_v0(env, 4)
    env = ss.resize_v0(env, 84, 84)
    env = ss.reshape_v0(env, (1, 84, 84))
    env = ss.pettingzoo_env_to_vec_env_v0(env)
    env = ss.concat_vec_envs_v0(vec_env=env, num_vec_envs=n_env, 
                                num_cpus=num_cpus, base_class='stable_baselines3')
    env = GymVectorEnvironment(env, env_name, device=device)
    return env

class OBS_BUFFER:
    def __init__(self) -> None:
        self.buffer = deque(maxlen=200)
        self.log_dir = "./img/"

    def store_rgb(self, rgb_array):
        rgb_array = np.transpose(rgb_array, (2, 0, 1))
        self.buffer.append(rgb_array)

    def video_summary(self, step, tag, max_size=300, save_gif=False):
        frame_buffer = []
        for img in self.buffer:
            grey_array = np.transpose(img, (1, 2, 0)).squeeze()
            frame_buffer.append(Image.fromarray(grey_array, 'L'))
        if save_gif:
            fp = os.path.join(self.log_dir, 'playbacks')
            os.makedirs(fp, exist_ok=True)
            frame_buffer[0].save(fp+f'/{step}_{tag}.gif', save_all=True,
                                    append_images=frame_buffer[1:-1],
                                    loop=0,
                                    format='GIF')
        self.buffer = []


if __name__=="__main__":
    env_names = ["boxing_v1", "double_dunk_v2", "ice_hockey_v1",
                 "pong_v2", "surround_v1", "tennis_v2",]

    for env_name in env_names:    
        env = make_vec_env(n_env=3, env_name=env_name)
        out = env.reset()
        obs = out['observation']
        loggers = [OBS_BUFFER() for _ in range(6)]

        for t in range(100):
            for i, logger in enumerate(loggers):
                logger.store_rgb(obs[i].cpu().numpy())
            action = env.action_space.sample()
            actions = torch.tensor([action] * 6)
            out = env.step(actions)
            obs = out['observation']
            print(f'step count:{t}')
            print(f'actions: {actions}')
            print(out['reward'])
            print(out['done'])

        for i, logger in enumerate(loggers):
            logger.video_summary(step=t, tag=env_name+str(i), save_gif=True)

import gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common.callbacks import BaseCallback
from env_utils import make_vec_env_sb3, make_vec_env_gym
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder



env_name = 'pong_v2'
num_envs = 16
env = make_vec_env_sb3(env_name, device='cuda', vs_builtin=True, num_envs=num_envs)
test_env = make_vec_env_sb3(env_name, device='cuda', vs_builtin=True, num_envs=1)
total_eval_freq = 100000
eval_freq = total_eval_freq // num_envs


def record_video_fn(step: int) -> bool:
    # print("video record callback saw step", step)
    if step - record_video_fn.last_video_step > total_eval_freq:
        record_video_fn.last_video_step = step
        return True
    else:
        return False
record_video_fn.last_video_step = 0

model = PPO(
    CnnPolicy,
    VecMonitor(env),
    verbose=3,
    gamma=0.95,
    n_steps=256,
    ent_coef=0.0905168,
    learning_rate=0.00062211,
    vf_coef=0.042202,
    max_grad_norm=0.9,
    gae_lambda=0.99,
    n_epochs=5,
    clip_range=0.3,
    batch_size=256,
    tensorboard_log='sb3_ppo_tb',
    create_eval_env=True,
)
model.learn(
    total_timesteps=int(1e7),
    eval_env=VecVideoRecorder(
        VecMonitor(test_env),
        video_folder="videos",
        record_video_trigger=record_video_fn,
    ),
    eval_freq=eval_freq,
)
model.save("policy")

# Rendering

# model = PPO.load("policy")
#
# env.reset()
# for agent in env.agent_iter():
#     obs, reward, done, info = env.last()
#     act = model.predict(obs, deterministic=True)[0] if not done else None
#     env.step(act)
#     env.render()


if __name__ == '__main__':
    pass
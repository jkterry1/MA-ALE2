from all.environments import GymVectorEnvironment
from all.experiments import ParallelEnvExperiment
from all.presets import atari
from all.agents import Agent
from all.logging import DummyWriter
from all.presets import IndependentMultiagentPreset, Preset
from all.core import State
import torch
from env_utils import make_env, make_vec_env
import supersuit as ss
from models import impala_features, impala_value_head, impala_policy_head, nature_features
from env_utils import InvertColorAgentIndicator
from all.bodies import DeepmindAtariBody
from models import ImpalaCNNLarge
from all import nn

def nat_features():
    return nature_features(16)

def make_ppo_vec(env_name, device, _, **kwargs):
    venv = make_vec_env(env_name, device)
    preset = atari.ppo.env(venv).device(device).hyperparameters(
        n_envs=venv.num_envs,
        n_steps=32,
        minibatches=8,
        epochs=4,
        feature_model_constructor=nat_features,
        # value_model_constructor=impala_value_head,
        # policy_model_constructor=impala_policy_head,
        entropy_loss_scaling=0.001,
        value_loss_scaling=0.1,
        clip_initial=0.5,
        clip_final=0.05,
    ).build()
    # base_agent = preset.agent.agent.agent
    # preset = DeepmindAtariBody(base_agent, lazy_frames=True, episodic_lives=False, clip_rewards=True,)
    # print(base_agent)

    experiment = ParallelEnvExperiment(preset, venv)
    return experiment, preset, venv


def impala_value_head():
    return nn.Linear(256, 1)


def impala_policy_head(env):
    return nn.Linear0(256, env.action_space.n)


def largenet():
    largenet = ImpalaCNNLarge(16, 18, nn.Linear, (84, 84), model_size=2)
    return largenet


def make_ppo_vec_largenet(env_name, device, _, **kwargs):
    venv = make_vec_env(env_name, device)
    n_steps = (128*32*2) // venv.num_envs
    preset = atari.ppo.env(venv).device(device).hyperparameters(
        n_envs=venv.num_envs,
        n_steps=n_steps,
        minibatches=32,
        epochs=2,
        feature_model_constructor=largenet,
        value_model_constructor=impala_value_head,
        policy_model_constructor=impala_policy_head,
        entropy_loss_scaling=0.001,
        value_loss_scaling=0.1,
        clip_initial=0.5,
        clip_final=0.05,
    ).build()
    # base_agent = preset.agent.agent.agent
    # preset = DeepmindAtariBody(base_agent, lazy_frames=True, episodic_lives=False, clip_rewards=True,)
    # print(base_agent)

    experiment = ParallelEnvExperiment(preset, venv)
    return experiment, preset, venv

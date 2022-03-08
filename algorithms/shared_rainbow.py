import copy
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.approximation import QDist, FixedTarget
from all.agents import Rainbow, RainbowTestAgent
from all.bodies import DeepmindAtariBody
from all.logging import DummyWriter
from all.memory import PrioritizedReplayBuffer, NStepReplayBuffer
from all.optim import LinearScheduler
from all.presets.atari.models import nature_rainbow
from all.presets.preset import Preset
from all.presets import PresetBuilder
from all.agents.independent import IndependentMultiagent
from shared_utils import DummyEnv, IndicatorBody, IndicatorState
from env_utils import make_env
import argparse
from all.environments import MultiagentPettingZooEnv
from all.experiments.multiagent_env_experiment import MultiagentEnvExperiment
from all.presets import atari
import os
import torch
from models import nature_features


default_hyperparameters = {
    "discount_factor": 0.99,
    "lr": 6.25e-5,
    "eps": 1.5e-4,
    # Training settings
    "minibatch_size": 128,
    "update_frequency": 32,
    "target_update_frequency": 1000,
    # Replay buffer settings
    "replay_start_size": 20000,
    "replay_buffer_size": 1000000,
    # Explicit exploration
    "initial_exploration": 0.02,
    "final_exploration": 0.,
    "test_exploration": 0.001,
    # Prioritized replay settings
    "alpha": 0.5,
    "beta": 0.5,
    # Multi-step learning
    "n_steps": 3,
    # Distributional RL
    "atoms": 51,
    "v_min": -10,
    "v_max": 10,
    # Noisy Nets
    "sigma": 0.5,
    # Model construction
    "model_constructor": nature_rainbow
}


class RainbowAtariPreset(Preset):
    """
    Rainbow DQN Atari Preset.

    Args:
        env (all.environments.AtariEnvironment): The environment for which to construct the agent.
        device (torch.device, optional): The device on which to load the agent.

    Keyword Args:
        discount_factor (float): Discount factor for future rewards.
        lr (float): Learning rate for the Adam optimizer.
        eps (float): Stability parameters for the Adam optimizer.
        minibatch_size (int): Number of experiences to sample in each training update.
        update_frequency (int): Number of timesteps per training update.
        target_update_frequency (int): Number of timesteps between updates the target network.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        replay_buffer_size (int): Maximum number of experiences to store in the replay buffer.
        initial_exploration (float): Initial probability of choosing a random action,
            decayed over course of training.
        final_exploration (float): Final probability of choosing a random action.
        alpha (float): Amount of prioritization in the prioritized experience replay buffer.
            (0 = no prioritization, 1 = full prioritization)
        beta (float): The strength of the importance sampling correction for prioritized experience replay.
            (0 = no correction, 1 = full correction)
        n_steps (int): The number of steps for n-step Q-learning.
        atoms (int): The number of atoms in the categorical distribution used to represent
            the distributional value function.
        v_min (int): The expected return corresponding to the smallest atom.
        v_max (int): The expected return correspodning to the larget atom.
        sigma (float): Initial noisy network noise.
        model_constructor (function): The function used to construct the neural model.
    """

    def __init__(self, env, name, device="cuda", **hyperparameters):
        hyperparameters = {**default_hyperparameters, **hyperparameters}
        super().__init__(env, name, hyperparameters)
        self.model = hyperparameters['model_constructor'](env, frames=10, atoms=hyperparameters["atoms"], sigma=hyperparameters["sigma"]).to(device)
        self.hyperparameters = hyperparameters
        self.n_actions = env.action_space.n
        self.device = device
        self.name = name
        self.agent_names = env.agents

    def agent(self, writer=DummyWriter(), train_steps=float('inf')):
        n_updates = (train_steps - self.hyperparameters['replay_start_size']) / self.hyperparameters['update_frequency']

        optimizer = Adam(
            self.model.parameters(),
            lr=self.hyperparameters['lr'],
            eps=self.hyperparameters['eps']
        )

        q_dist = QDist(
            self.model,
            optimizer,
            self.n_actions,
            self.hyperparameters['atoms'],
            scheduler=CosineAnnealingLR(optimizer, n_updates),
            v_min=self.hyperparameters['v_min'],
            v_max=self.hyperparameters['v_max'],
            target=FixedTarget(self.hyperparameters['target_update_frequency']),
            writer=writer,
        )

        replay_buffer = NStepReplayBuffer(
            self.hyperparameters['n_steps'],
            self.hyperparameters['discount_factor'],
            PrioritizedReplayBuffer(
                self.hyperparameters['replay_buffer_size'],
                alpha=self.hyperparameters['alpha'],
                beta=self.hyperparameters['beta'],
                device=self.device,
                store_device="cpu"
            )
        )

        def make_agent(agent_id):
            return DeepmindAtariBody(
                IndicatorBody(
                    Rainbow(
                        q_dist,
                        replay_buffer,
                        exploration=LinearScheduler(
                            self.hyperparameters['initial_exploration'],
                            self.hyperparameters['final_exploration'],
                            0,
                            train_steps - self.hyperparameters['replay_start_size'],
                            name="exploration",
                            writer=writer
                        ),
                        discount_factor=self.hyperparameters['discount_factor'] ** self.hyperparameters["n_steps"],
                        minibatch_size=self.hyperparameters['minibatch_size'],
                        replay_start_size=self.hyperparameters['replay_start_size'],
                        update_frequency=self.hyperparameters['update_frequency'],
                        writer=writer,
                    ),
                    self.agent_names.index(agent_id),
                    len(self.agent_names)
                ),
                lazy_frames=True,
                episodic_lives=True
            )

        return IndependentMultiagent({
            agent_id : make_agent(agent_id)
            for agent_id in self.agent_names
        })

    def test_agent(self):
        q_dist = QDist(
            copy.deepcopy(self.model),
            None,
            self.n_actions,
            self.hyperparameters['atoms'],
            v_min=self.hyperparameters['v_min'],
            v_max=self.hyperparameters['v_max'],
        )
        def make_agent(agent_id):
            return DeepmindAtariBody(
                IndicatorBody(
                    RainbowTestAgent(q_dist, self.n_actions, self.hyperparameters["test_exploration"]),
                    self.agent_names.index(agent_id),
                    len(self.agent_names)
                )
            )

        return IndependentMultiagent({
            agent_id : make_agent(agent_id)
            for agent_id in self.agent_names
        })

rainbow = PresetBuilder('rainbow', default_hyperparameters, RainbowAtariPreset)


def nat_features(env, frames=4, **kwargs):
    return nature_features(env, frames=10)


def make_rainbow_preset(env_name, device, replay_buffer_size, **kwargs):
    env = make_env(env_name)
    agent0 = env.possible_agents[0]
    obs_space = env.observation_spaces[agent0]
    act_space = env.action_spaces[agent0]
    for agent in env.possible_agents:
        assert obs_space == env.observation_spaces[agent]
        assert act_space == env.action_spaces[agent]
    env_agents = env.possible_agents
    multi_agent_env = MultiagentPettingZooEnv(env, env_name, device=device)
    preset = rainbow.env(multi_agent_env).hyperparameters(replay_buffer_size=replay_buffer_size).device(device).env(
        DummyEnv(
            obs_space, act_space, env_agents
        )
    ).build()

    experiment = MultiagentEnvExperiment(
        preset,
        multi_agent_env,
        write_loss=False,
    )
    return experiment, preset, multi_agent_env

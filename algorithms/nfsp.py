# Copyright 2019 Matthew Judell. All rights reserved.
# Copyright 2019 DATA Lab at Texas A&M University. All rights reserved.
# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

''' Neural Fictitious Self-Play (NFSP) agent implemented in TensorFlow.

See the paper https://arxiv.org/abs/1603.01121 for more details.
'''

import copy
import random
import collections
import enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.agents import Rainbow, Agent
from all.approximation import QDist, FixedTarget
from all.bodies import DeepmindAtariBody
from all.logging import DummyWriter
from all.memory import PrioritizedReplayBuffer, NStepReplayBuffer
from all.optim import LinearScheduler
from all.presets.atari.models import nature_rainbow, nature_features, nature_policy_head
from all.presets.preset import Preset
from all.presets import PresetBuilder
from all.agents.independent import IndependentMultiagent
from shared_utils import DummyEnv, IndicatorBody, IndicatorState
from all.environments import MultiagentPettingZooEnv
from all.experiments.multiagent_env_experiment import MultiagentEnvExperiment
from all.approximation import Approximation
from env_utils import make_env
from typing import Tuple
from torch import TensorType



default_hyperparameters = {
    "discount_factor": 0.99,
    "lr": 6.25e-5,
    "eps": 1.5e-4,
    # Training settings
    "minibatch_size": 128,
    "update_frequency": 32,
    "target_update_frequency": 1000,
    "gradient_steps": None, # MUST be filled in by hparam search
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
    # NFSP
    "anticipatory": 0.1,
    # Model construction
    "model_constructor": nature_rainbow,
}

def save_name(trainer_type: str, env: str, replay_size: int, num_frames: int, seed: float):
    return f"{trainer_type}/{env}/RB{replay_size}_F{num_frames}/S{seed}"



class NFSPRainbowAgent(Rainbow):
    """Rainbow agent with an additional AveragePolicyNetwork"""
    def __init__(self, q_dist, replay_buffer, avg_policy, reservoir_buffer,
                 anticipatory=0.1, discount_factor=0.99, eps=1e-5,
                 exploration=0.02, minibatch_size=32, replay_start_size=5000,
                 update_frequency=1, writer=DummyWriter()):
        super(NFSPRainbowAgent, self).__init__(
            q_dist, replay_buffer, discount_factor=discount_factor, eps=eps,
            exploration=exploration, minibatch_size=minibatch_size,
            replay_start_size=replay_start_size,
            update_frequency=update_frequency, writer=writer
        )
        self.reservoir_replay_start_size = replay_start_size
        self._reservoir_buffer = reservoir_buffer
        self._avg_policy = avg_policy
        self.anticipatory = anticipatory
        self._device = self.replay_buffer.buffer.device
        self.sample_episode_policy()


    def act(self, state):
        if self._first_ep_step(state):
            self.sample_episode_policy()

        self.replay_buffer.store(self._state, self._action, state)
        if self._mode == 'best_response':
            self._reservoir_buffer.store(self._state, self._action, state)
            self._action = self._choose_action(state).squeeze().to(self._device)
        elif self._mode == 'average_policy':
            with torch.no_grad():
                self._action = self._average_action(state).to(self._device)
        else: raise ValueError
        self._state = state
        self._train()

        return self._action

    def eval(self, state):
        if self.evaluate_with == 'best_response':
            return self._best_actions(self.q_dist.eval(state)).item()
        elif self.evaluate_with == 'average_policy':
            return self._average_action(state)
        else:
            raise ValueError

    def sample_episode_policy(self):
        """Sample average/best_response policy"""
        if np.random.rand() < self.anticipatory:
            self._mode = 'best_response'
        else:
            self._mode = 'average_policy'

    def _first_ep_step(self, state) -> bool:
        """whether current timestep is the beginning of an episode"""
        return state['ep_step'] == 0

    def _choose_action(self, state):
        if self._should_explore():
            return torch.randint(0, self.q_dist.n_actions, size=(1,))
        return self._best_actions(self.q_dist.no_grad(state)) # .item()

    def _average_action(self, state) -> Tuple[TensorType, TensorType]:
        # logits = self._avg_policy(state).sum(dim=-1) # Batch x Actions
        logits = self._avg_policy(state) # batch x actions
        probs = F.softmax(logits, dim=-1)
        actions = probs.multinomial(1).squeeze()

        return actions

    def _should_explore(self):
        return (
            len(self.replay_buffer) < self.replay_start_size
            or np.random.rand() < self.exploration
        )

    def _best_actions(self, probs):
        q_values = (probs * self.q_dist.atoms).sum(dim=-1)
        return torch.argmax(q_values, dim=-1)

    def _train(self):
        if self._should_train():
            # sample transitions from buffer
            states, actions, rewards, next_states, weights = self.replay_buffer.sample(self.minibatch_size)
            # forward pass
            dist = self.q_dist(states, actions.squeeze())
            # compute target distribution
            target_dist = self._compute_target_dist(next_states, rewards)
            # compute loss
            kl = self._kl(dist, target_dist)
            loss = (weights * kl).mean()
            # backward pass
            self.q_dist.reinforce(loss)
            # update replay buffer priorities
            self.replay_buffer.update_priorities(kl.detach())
            # debugging
            self.writer.add_loss(
                "q_mean", (dist.detach() * self.q_dist.atoms).sum(dim=1).mean()
            )
        if self._should_train_avg():
            transitions = self._reservoir_buffer.sample(self.minibatch_size)
            states, actions, rewards, next_states, weights = transitions

            # RLcard repo just one-hots the actions. Likely to get around the
            #   Q/policy origin issue (no action probs for best_action or exploration)
            one_hot = F.one_hot(actions, num_classes=self.q_dist.n_actions)
            action_probs = self._avg_policy(states)
            # cross entropy loss and do optimizer step
            ce_loss = - (one_hot * action_probs).sum(dim=-1).mean()
            self._avg_policy.reinforce(ce_loss)

            # debugging
            self.writer.add_loss(
                "ce_loss", ce_loss.detach().item()
            )

            return ce_loss

    def _should_train(self):
        self._frames_seen += 1
        return self._frames_seen > self.replay_start_size and self._frames_seen % self.update_frequency == 0

    def _should_train_avg(self):
        return self._frames_seen > self.replay_start_size \
               and self._frames_seen % self.update_frequency == 0 \
               and len(self._reservoir_buffer) >= self.minibatch_size

    def _compute_target_dist(self, states, rewards):
        actions = self._best_actions(self.q_dist.no_grad(states))
        dist = self.q_dist.target(states, actions)
        shifted_atoms = (
            rewards.view((-1, 1)) + self.discount_factor * self.q_dist.atoms
        )
        return self.q_dist.project(dist, shifted_atoms)

    def _kl(self, dist, target_dist):
        log_dist = torch.log(torch.clamp(dist, min=self.eps))
        log_target_dist = torch.log(torch.clamp(target_dist, min=self.eps))
        return (target_dist * (log_target_dist - log_dist)).sum(dim=-1)

class NFSPRainbowTestAgent(Agent):
    def __init__(self, qdist, avg_policy, anticipatory=0.1, device='cuda'):
        self.q_dist = qdist
        self._avg_policy = avg_policy
        self.anticipatory = anticipatory
        self._device = device

    def act(self, state):
        return self._choose_action(state).squeeze().to(self._device)


    def _first_ep_step(self, state) -> bool:
        """whether current timestep is the beginning of an episode"""
        return state['ep_step'] == 0

    def _choose_action(self, state):
        if self._should_explore():
            return torch.randint(0, self.q_dist.n_actions, size=(1,))
        return self._best_actions(self.q_dist.no_grad(state)) # .item()

    def _best_actions(self, probs):
        q_values = (probs * self.q_dist.atoms).sum(dim=-1)
        return torch.argmax(q_values, dim=-1)

    def _should_explore(self):
        return False


class NFSPRainbowPreset(Preset):
    def __init__(self, env, name, device='cuda', **hyperparameters):
        hparams = {**default_hyperparameters, **hyperparameters}
        # super(NFSPRainbowPreset, self).__init__(env, name, hparams)
        super(NFSPRainbowPreset, self).__init__(name=name, device=device, hyperparameters=hparams)
        self.n_actions = env.action_space.n
        self.agent_names = env.agents
        self.env = env

        # Build model
        self.model = hparams['model_constructor'](env, frames=10, atoms=hparams["atoms"],
                                                  sigma=hparams["sigma"]).to(device)
        from all.nn import RLNetwork, NoisyFactorizedLinear
        self.avg_model = RLNetwork(nn.Sequential(
            nature_features(frames=10),
            NoisyFactorizedLinear(512, env.action_space.n),
        )).to(device)



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
        sl_optimizer = Adam(
            self.avg_model.parameters(),
            lr=self.hyperparameters['lr'],
            eps=self.hyperparameters['eps']
        )
        avg_policy = Approximation(
            self.avg_model,
            sl_optimizer,
            name='average_policy',
            device=self.device,
            target=FixedTarget(self.hyperparameters['target_update_frequency']),
            writer=writer,
        )
        reservoir_buffer = ReservoirBuffer(
            self.hyperparameters['replay_buffer_size'],
            device=self.device,
            store_device="cpu",
        )

        def make_agent(agent_id):
            agent = DeepmindAtariBody(
                IndicatorBody(
                    NFSPRainbowAgent(
                        q_dist,
                        replay_buffer,
                        avg_policy,
                        reservoir_buffer,
                        anticipatory=self.hyperparameters['anticipatory'],
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

            return agent

        return IndependentMultiagent({
            agent_id: make_agent(agent_id)
            for agent_id in self.agent_names
        })

    def test_agent(self):
        q_dist = QDist(
            copy.deepcopy(self.model),
            optimizer=None,
            n_actions=self.n_actions,
            n_atoms=self.hyperparameters['atoms'],
            v_min=self.hyperparameters['v_min'],
            v_max=self.hyperparameters['v_max'],
        )
        avg_policy = Approximation(
            copy.deepcopy(self.avg_model),
            optimizer=None,
            name='average_policy',
            device=self.device,
        )
        def make_agent(agent_id):
            return DeepmindAtariBody(IndicatorBody(
                NFSPRainbowTestAgent(q_dist, avg_policy, anticipatory=self.hyperparameters['anticipatory']),
                self.agent_names.index(agent_id),
                len(self.agent_names)
            ))

        return IndependentMultiagent({
            agent_id: make_agent(agent_id) for agent_id in self.agent_names
        })


from env_utils import MAPZEnvSteps

nfsp_rainbow_builder = PresetBuilder('nfsp_rainbow', default_hyperparameters, NFSPRainbowPreset)

def make_nfsp_rainbow(env_name, device, replay_buffer_size, **kwargs):
    env = make_env(env_name)
    test_env = make_env(env_name, vs_builtin=True)
    agent0 = env.possible_agents[0]
    obs_space = env.observation_spaces[agent0]
    act_space = env.action_spaces[agent0]
    for agent in env.possible_agents:
        assert obs_space == env.observation_spaces[agent]
        assert act_space == env.action_spaces[agent]
    env_agents = env.possible_agents
    multi_agent_env = MAPZEnvSteps(env, env_name, device=device)
    multi_agent_test_env = MAPZEnvSteps(test_env, env_name, device=device)

    hparams = kwargs.get('hparams', {})
    quiet = kwargs.get('quiet', False)

    preset = nfsp_rainbow_builder.env(multi_agent_env).hyperparameters(replay_buffer_size=replay_buffer_size).hyperparameters(**hparams).device(device).env(
        DummyEnv(
            obs_space, act_space, env_agents
        )
    ).build()

    experiment = MultiagentEnvExperiment(
        preset,
        multi_agent_env,
        logdir="runs/" + save_name('nfsp_rainbow', env_name, replay_buffer_size, kwargs['seed'], kwargs['num_frames']),
        test_env=multi_agent_test_env,
        quiet=quiet,
    )
    return experiment, preset, multi_agent_env

#############################################################################################

Transition = collections.namedtuple('Transition', 'info_state action_probs')

from all.memory.replay_buffer import ExperienceReplayBuffer
class ReservoirBuffer(ExperienceReplayBuffer):
    ''' Allows uniform sampling over a stream of data.

    This class supports the storage of arbitrary elements, such as observation
    tensors, integer actions, etc.

    See https://en.wikipedia.org/wiki/Reservoir_sampling for more details.
    '''

    def __init__(self, size, device='cpu', store_device=None):
        super().__init__(size, device=device, store_device=store_device)
        self._add_calls = 0

    def _add(self, sample):
        """Potentially adds `element` to the reservoir buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            idx = np.random.randint(0, self._add_calls + 1)
            if idx < self.capacity:
                self.buffer[idx] = sample
        self._add_calls += 1

    def store(self, state, action, next_state):
        if state is not None and not state.done:
            state = state.to(self.store_device)
            next_state = next_state.to(self.store_device)
            self._add((state, action, next_state))

    def sample(self, num_samples: int):
        """Returns `num_samples` uniformly sampled from the buffer."""
        if len(self.buffer) < num_samples:
            raise ValueError(f"{num_samples} elements could not be sampled from size {len(self.buffer)}")
        minibatch = random.sample(self.buffer, num_samples)
        return self._reshape(minibatch, torch.ones(num_samples, device=self.device))


import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.approximation import QDist, FixedTarget, Approximation
from all.bodies import DeepmindAtariBody
from all.logging import DummyWriter
from all.optim import LinearScheduler
from all.presets.atari.models import nature_rainbow
from all.presets import ParallelPresetBuilder
from all.memory import PrioritizedReplayBuffer
from all.agents._parallel_agent import ParallelAgent
from all.experiments import ParallelEnvExperiment
from all.nn import RLNetwork, NoisyFactorizedLinear

from env_utils import make_vec_env
from buffers import ParallelNStepBuffer, ParallelReservoirBuffer, CompressedPrioritizedReplayBuffer
from .parallel_rainbow import ParallelRainbow, ParallelRainbowPreset
from models import our_nat_features



default_hyperparameters = {
    "discount_factor": 0.99,
    "lr": 6.25e-5,
    "eps": 1.5e-4,
    # Training settings
    "minibatch_size": 128,
    "update_frequency": 1024,
    "target_update_frequency": 1000,
    # Replay buffer settings
    "replay_start_size": 20000,
    "replay_buffer_size": 50000,
    # Explicit exploration
    "initial_exploration": 0.02,
    "final_exploration": 0.,
    "final_exploration_step": 250000,
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
    "model_constructor": nature_rainbow,
    # vectorization
    "n_envs": 16,
    # NFSP
    "anticipatory": 0.1,
    "reservoir_buffer_size": 200000,
}


class ParallelRainbowNFSP(ParallelRainbow):
    """NFSP version of ParallelRainbow"""

    def __init__(
            self,
            q_dist,
            avg_policy,
            replay_buffer,
            reservoir_buffer,
            discount_factor=0.99,
            eps=1e-5,
            exploration=0.02,
            minibatch_size=32,
            replay_start_size=5000,
            update_frequency=1,
            writer=DummyWriter(),
            n_envs=None,
            anticipatory=0.1,
    ):
        super(ParallelRainbowNFSP, self).__init__(
            q_dist,
            replay_buffer,
            discount_factor=discount_factor,
            eps=eps,
            exploration=exploration,
            minibatch_size=minibatch_size,
            replay_start_size=replay_start_size,
            update_frequency=update_frequency,
            writer=writer,
            n_envs=n_envs,
        )
        self._avg_policy = avg_policy
        self.anticipatory = anticipatory
        self._reservoir_buffer = reservoir_buffer
        self._device = self.replay_buffer.buffer.device

        # initialize best response / avg policy modes
        self._br_modes = torch.rand(self.n_envs).to(self._device) < self.anticipatory

    @property
    def _avg_modes(self) -> torch.tensor:
        return ~self._br_modes

    def _sample_episode_policy(self, dones: torch.Tensor):
        """Sample average/best_response policies"""
        best_response_sample = torch.rand(dones.shape[0]).to(self._device) < self.anticipatory
        return dones*best_response_sample + ~dones*self._br_modes

    def _split_states(self, states):
        if states is None:
            return None, None
        return states[self._br_modes], states[self._avg_modes]

    def act(self, states):
        self._frames_seen += self.n_envs
        self.replay_buffer.store(self._state, self._action, states)

        if self._br_modes.any():
            br_states, _ = self._split_states(self._state)
            br_actions, _ = self._split_states(self._action)
            self._reservoir_buffer.store(br_states, br_actions, None)

        br_actions_next = self._choose_action(states).to(self._device)
        with torch.no_grad():
            avg_actions_next = self._average_action(states)

        self._action = self._br_modes*br_actions_next + self._avg_modes*avg_actions_next
        self._state = states

        self._train()
        self._train_avg()

        if states.done.any():
            self._br_modes = self._sample_episode_policy(states.done)

        return self._action

    def _train_avg(self):
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
            self.writer.add_loss("ce_loss", ce_loss.detach().item())

            return ce_loss

    def _should_train_avg(self):
        return (self._frames_seen > self.replay_start_size
                and self._frames_seen % self.update_frequency < self.n_envs
                and len(self._reservoir_buffer) >= self.minibatch_size)

    def _average_action(self, state):
        logits = self._avg_policy(state) # batch x actions
        probs = F.softmax(logits, dim=-1)
        actions = probs.multinomial(1).squeeze()

        return actions

    def get_buffers(self) -> tuple:
        """return all buffers in a dictionary for checkpointing/loading"""
        return (self.replay_buffer, self._reservoir_buffer,)

    def load_buffers(self, buffers: tuple):
        self.replay_buffer, self._reservoir_buffer = buffers


class ParallelRainbowTestAgent(ParallelAgent):
    def __init__(self, q_dist, avg_policy, n_actions, exploration=0.):
        self.q_dist = q_dist
        self.avg_policy = avg_policy
        self.n_actions = n_actions
        self.exploration = exploration

    def act(self, state):
        q_values = (self.q_dist(state) * self.q_dist.atoms).sum(dim=-1)
        return torch.argmax(q_values, dim=-1)


class ParallelRainbowNFSPPreset(ParallelRainbowPreset):
    """Parallel Rainbow implementation"""

    def __init__(self, env, name, device, **hyperparameters):
        super(ParallelRainbowNFSPPreset, self).__init__(env, name, device, **hyperparameters)

        self.avg_model = RLNetwork(nn.Sequential(
            our_nat_features(),
            NoisyFactorizedLinear(512, self.n_actions),
        )).to(device)

    def agent(self, writer=DummyWriter(), train_steps=float('inf')):
        n_updates = (train_steps - self.hyperparameters['replay_start_size']) / self.hyperparameters['update_frequency']

        optimizer = Adam(
            self.model.parameters(),
            lr=self.hyperparameters['lr'],
            eps=self.hyperparameters['eps']
        )
        sl_optimizer = Adam(self.avg_model.parameters(), lr=self.hyperparameters['lr'], eps=self.hyperparameters['eps'])

        q = QDist(
            self.model,
            optimizer,
            self.n_actions,
            self.hyperparameters['atoms'],
            v_min=self.hyperparameters['v_min'],
            v_max=self.hyperparameters['v_max'],
            target=FixedTarget(self.hyperparameters['target_update_frequency']),
            scheduler=CosineAnnealingLR(optimizer, n_updates),
            writer=writer,
        )

        replay_buffer = ParallelNStepBuffer(
            self.hyperparameters['n_steps'],
            self.hyperparameters['discount_factor'],
            CompressedPrioritizedReplayBuffer(
                self.hyperparameters['replay_buffer_size'],
                alpha=self.hyperparameters['alpha'],
                beta=self.hyperparameters['beta'],
                device=self.device,
                store_device="cpu",
                compress=False
            ),
            n_envs=self.n_envs,
        )

        avg_policy = Approximation(
            self.avg_model,
            sl_optimizer,
            name='average_policy',
            device=self.device,
            target=FixedTarget(self.hyperparameters['target_update_frequency']),
            writer=writer,
        )

        reservoir_buffer = ParallelReservoirBuffer(
            self.hyperparameters['reservoir_buffer_size'],
            device=self.device,
            store_device="cpu",
            compress=True
        )

        return DeepmindAtariBody(
            ParallelRainbowNFSP(
                q,
                avg_policy,
                replay_buffer,
                reservoir_buffer,
                exploration=LinearScheduler(
                    self.hyperparameters['initial_exploration'],
                    self.hyperparameters['final_exploration'],
                    0,
                    self.hyperparameters["final_exploration_step"] - self.hyperparameters["replay_start_size"],
                    name="epsilon",
                    writer=writer,
                ),
                discount_factor=self.hyperparameters["discount_factor"],
                minibatch_size=self.hyperparameters["minibatch_size"],
                replay_start_size=self.hyperparameters["replay_start_size"],
                update_frequency=self.hyperparameters["update_frequency"],
                writer=writer,
                n_envs=self.n_envs,
            ),
            frame_stack=0
        )


    def test_agent(self):
        q_dist = QDist(
            copy.deepcopy(self.model),
            None,
            self.n_actions,
            self.hyperparameters['atoms'],
            v_min=self.hyperparameters['v_min'],
            v_max=self.hyperparameters['v_max'],
        )
        avg_policy = Approximation(
            self.avg_model,
            None,
            name='average_policy',
            device=self.device,
            target=FixedTarget(self.hyperparameters['target_update_frequency']),
        )
        return DeepmindAtariBody(ParallelRainbowTestAgent(q_dist, avg_policy, self.n_actions, self.hyperparameters["test_exploration"]),
                                 frame_stack=0)

    def parallel_test_agent(self):
        return self.test_agent()


parallel_rainbow_nfsp = ParallelPresetBuilder('parallel_rainbow_nfsp', default_hyperparameters, ParallelRainbowNFSPPreset)

def rainbow_model(env, frames=10, hidden=512, atoms=51, sigma=0.5):
    return nature_rainbow(env, frames, hidden, atoms, sigma)

def make_parallel_rainbow_nfsp(env_name, device, replay_buffer_size, **kwargs):
    n_envs = 16
    venv = make_vec_env(env_name, device=device, vs_builtin=False, num_envs=n_envs)
    test_venv = make_vec_env(env_name, device=device, vs_builtin=True, num_envs=n_envs)

    quiet = kwargs.get('quiet', False)
    hparams = kwargs.get('hparams', {})
    hparams['n_envs'] = n_envs * 2  # num agents
    hparams['model_constructor'] = rainbow_model

    preset = parallel_rainbow_nfsp.env(venv).device(device).hyperparameters(**hparams).build()
    experiment = ParallelEnvExperiment(preset, venv, test_env=test_venv, quiet=quiet)

    return experiment, preset, venv

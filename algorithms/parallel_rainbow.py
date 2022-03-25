import copy
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.approximation import QDist, FixedTarget
from all.bodies import DeepmindAtariBody
from all.logging import DummyWriter
from all.optim import LinearScheduler
from all.presets.atari.models import nature_rainbow
from all.presets import ParallelPresetBuilder
from all.presets import ParallelPreset
from all.memory import ExperienceReplayBuffer, PrioritizedReplayBuffer, NStepReplayBuffer, NStepAdvantageBuffer
from all.agents._parallel_agent import ParallelAgent
from all.experiments import ParallelEnvExperiment
from env_utils import make_vec_env



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
}


class ParallelRainbow(ParallelAgent):
    """
    A categorical DQN agent (C51).
    Rather than making a point estimate of the Q-function,
    C51 estimates a categorical distribution over possible values.
    The 51 refers to the number of atoms used in the
    categorical distribution used to estimate the
    value distribution.
    https://arxiv.org/abs/1707.06887

    Args:
        q_dist (QDist): Approximation of the Q distribution.
        replay_buffer (ReplayBuffer): The experience replay buffer.
        discount_factor (float): Discount factor for future rewards.
        eps (float): Stability parameter for computing the loss function.
        exploration (float): The probability of choosing a random action.
        minibatch_size (int): The number of experiences to sample in each training update.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        update_frequency (int): Number of timesteps per training update.
    """

    def __init__(
            self,
            q_dist,
            replay_buffer,
            discount_factor=0.99,
            eps=1e-5,
            exploration=0.02,
            minibatch_size=32,
            replay_start_size=5000,
            update_frequency=1,
            writer=DummyWriter(),
    ):
        # objects
        self.q_dist = q_dist
        self.replay_buffer = replay_buffer
        self.writer = writer
        # hyperparameters
        self.eps = eps
        self.exploration = exploration
        self.replay_start_size = replay_start_size
        self.update_frequency = update_frequency
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        # private
        self._state = None
        self._action = None
        self._frames_seen = 0

    # TODO: these Rainbow features are not present
    # It uses Double Q-Learning to tackle overestimation bias.
    # It uses dueling networks.
    # TODO: NStepReplay and PrioritizedReplay are not Parallel
    #       see NStepAdvantageBuffer for example

    def act(self, states):
        self.replay_buffer.store(self._state, self._action, states)
        self._train()
        self._state = states
        self._action = self._choose_action(states)
        return self._action

    """
    def forward(self, state):
        features = self.conv(state)
        features = features.view(features.size(0), -1)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())
        
        return qvals
    """

    def eval(self, state):
        return self._best_actions(self.q_dist.eval(state)).item()

    def _choose_action(self, state):
        if self._should_explore():
            return np.random.randint(0, self.q_dist.n_actions)
        return self._best_actions(self.q_dist.no_grad(state)).item()

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
            dist = self.q_dist(states, actions)
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

    def _should_train(self):
        self._frames_seen += 1
        return self._frames_seen > self.replay_start_size and self._frames_seen % self.update_frequency == 0

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

class ParallelRainbowTestAgent(ParallelAgent):
    def __init__(self, q_dist, n_actions, exploration=0.):
        self.q_dist = q_dist
        self.n_actions = n_actions
        self.exploration = exploration

    def act(self, state):
        if np.random.rand() < self.exploration:
            return np.random.randint(0, self.n_actions)
        q_values = (self.q_dist(state) * self.q_dist.atoms).sum(dim=-1)
        return torch.argmax(q_values, dim=-1)


class ParallelRainbowPreset(ParallelPreset):
    """Parallel Rainbow implementation"""

    def __init__(self, env, name, device, **hyperparameters):
        super().__init__(name, device, hyperparameters)
        self.model = hyperparameters['model_constructor'](env, atoms=hyperparameters['atoms']).to(device)
        self.n_actions = env.action_space.n

    def agent(self, writer=DummyWriter(), train_steps=float('inf')):
        n_updates = (train_steps - self.hyperparameters['replay_start_size']) / self.hyperparameters['update_frequency']

        optimizer = Adam(
            self.model.parameters(),
            lr=self.hyperparameters['lr'],
            eps=self.hyperparameters['eps']
        )

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


        return DeepmindAtariBody(
            ParallelRainbow(
                q,
                replay_buffer,
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
                writer=writer
            ),
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
        return DeepmindAtariBody(ParallelRainbowTestAgent(q_dist, self.n_actions, self.hyperparameters["test_exploration"]))

    def parallel_test_agent(self):
        return self.test_agent()


parallel_rainbow = ParallelPresetBuilder('parallel_rainbow', default_hyperparameters, ParallelRainbowPreset)

def make_parallel_rainbow(env_name, device, replay_buffer_size, **kwargs):
    n_envs = 16
    venv = make_vec_env(env_name, device=device, vs_builtin=False, num_envs=n_envs)
    test_venv = make_vec_env(env_name, device=device, vs_builtin=True, num_envs=n_envs)

    quiet = kwargs.get('quiet', False)
    hparams = kwargs.get('hparams', {})
    hparams['n_envs'] = n_envs * 2  # num agents

    preset = parallel_rainbow.env(venv).device(device).hyperparameters(**hparams).build()
    experiment = ParallelEnvExperiment(preset, venv, test_env=test_venv, quiet=quiet)

    return experiment, preset, venv

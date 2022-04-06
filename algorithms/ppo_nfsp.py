import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from all.presets.atari.ppo import PPOAtariPreset
from all.nn import RLNetwork, NoisyFactorizedLinear
from all.agents import PPO, PPOTestAgent
from all.bodies import DeepmindAtariBody
from all.approximation import VNetwork, FeatureNetwork
from all.logging import DummyWriter
from all.optim import LinearScheduler
from all.policies import SoftmaxPolicy
from all.presets.atari.models import nature_features
from all.approximation import Approximation
from all.approximation import FixedTarget
from all.experiments import ParallelEnvExperiment
from all.presets import ParallelPresetBuilder

from buffers import ParallelReservoirBuffer
from env_utils import make_vec_env


from all.presets.atari.ppo import default_hyperparameters
default_hyperparameters.update({
    "target_update_frequency": 1000,
    "replay_start_size": 80000,
    "replay_buffer_size": 1000000,
    "anticipatory": 0.1,
})


class PPONFSPAgent(PPO):

    def __init__(
            self,
            features,
            v,
            policy,
            avg_policy,
            reservoir_buffer,
            anticipatory=0.1,
            replay_start_size=5000,
            n_actions=None,
            **kwargs
    ):
        super().__init__(features, v, policy, **kwargs)
        self._reservoir_buffer = reservoir_buffer
        self._avg_policy = avg_policy
        self.anticipatory = anticipatory
        self.replay_start_size = replay_start_size
        self._device = self._reservoir_buffer.device
        self._frames_seen = 0
        self.update_frequency = self._batch_size // self.n_envs
        self.n_actions = n_actions

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

        self._buffer.store(self._states, self._actions, states.reward)

        if self._br_modes.any():
            br_states, _ = self._split_states(self._states)
            br_actions, _ = self._split_states(self._actions)
            br_states_next, _ = self._split_states(states)
            self._reservoir_buffer.store(br_states, br_actions, br_states_next)

        br_actions_next = self.policy.no_grad(self.features.no_grad(states)).sample()
        with torch.no_grad():
            avg_actions_next = self._average_action(states).to(self._device)

        self._actions = self._br_modes*br_actions_next + self._avg_modes*avg_actions_next

        self._states = states
        self._train(states)
        self._train_avg()

        if states.done.any():
            self._br_modes = self._sample_episode_policy(states.done)

        return self._actions

    def _train_avg(self):
        if self._should_train_avg():
            transitions = self._reservoir_buffer.sample(self._batch_size)
            states, actions, rewards, next_states, weights = transitions

            # RLcard repo just one-hots the actions. Likely to get around the
            #   Q/policy origin issue (no action probs for best_action or exploration)
            one_hot = F.one_hot(actions, num_classes=self.n_actions)
            action_probs = self._avg_policy(states)
            # cross entropy loss and do optimizer step
            ce_loss = - (one_hot * action_probs).sum(dim=-1).mean()
            self._avg_policy.reinforce(ce_loss)

            # debugging
            self.writer.add_loss("ce_loss", ce_loss.detach().item())

            return ce_loss

    def _should_train_avg(self):
        return (self._frames_seen > self.replay_start_size
                and self._frames_seen % self.update_frequency == 0
                and len(self._reservoir_buffer) >= self._batch_size)


    def _average_action(self, state):
        # logits = self._avg_policy(state).sum(dim=-1) # Batch x Actions
        logits = self._avg_policy(state) # batch x actions
        probs = F.softmax(logits, dim=-1)
        actions = probs.multinomial(1).squeeze()

        return actions


class PPONFSPPreset(PPOAtariPreset):

    def __init__(self, env, name, device, **hyperparameters):
        hparams = {**default_hyperparameters, **hyperparameters}
        super().__init__(env, name, device, **hparams)
        self.n_actions = env.action_space.n

        self.avg_model = RLNetwork(nn.Sequential(
            nature_features(frames=16),
            NoisyFactorizedLinear(512, self.n_actions),
        )).to(device)

    def agent(self, writer=DummyWriter(), train_steps=float('inf')):
        n_updates = train_steps * self.hyperparameters['epochs'] * self.hyperparameters['minibatches'] / (self.hyperparameters['n_steps'] * self.hyperparameters['n_envs'])

        feature_optimizer = Adam(self.feature_model.parameters(), lr=self.hyperparameters["lr"], eps=self.hyperparameters["eps"])
        value_optimizer = Adam(self.value_model.parameters(), lr=self.hyperparameters["lr"], eps=self.hyperparameters["eps"])
        policy_optimizer = Adam(self.policy_model.parameters(), lr=self.hyperparameters["lr"], eps=self.hyperparameters["eps"])
        sl_optimizer = Adam(self.avg_model.parameters(), lr=self.hyperparameters['lr'], eps=self.hyperparameters['eps'])

        features = FeatureNetwork(
            self.feature_model,
            feature_optimizer,
            scheduler=CosineAnnealingLR(feature_optimizer, n_updates),
            clip_grad=self.hyperparameters["clip_grad"],
            writer=writer
        )

        v = VNetwork(
            self.value_model,
            value_optimizer,
            scheduler=CosineAnnealingLR(value_optimizer, n_updates),
            loss_scaling=self.hyperparameters["value_loss_scaling"],
            clip_grad=self.hyperparameters["clip_grad"],
            writer=writer
        )

        policy = SoftmaxPolicy(
            self.policy_model,
            policy_optimizer,
            scheduler=CosineAnnealingLR(policy_optimizer, n_updates),
            clip_grad=self.hyperparameters["clip_grad"],
            writer=writer
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
            self.hyperparameters['replay_buffer_size'],
            device=self.device,
            store_device="cpu",
        )

        return DeepmindAtariBody(
            PPONFSPAgent(
                features,
                v,
                policy,
                avg_policy,
                reservoir_buffer,
                anticipatory=self.hyperparameters["anticipatory"],
                epsilon=LinearScheduler(
                    self.hyperparameters["clip_initial"],
                    self.hyperparameters["clip_final"],
                    0,
                    n_updates,
                    name='clip',
                    writer=writer
                ),
                epochs=self.hyperparameters["epochs"],
                minibatches=self.hyperparameters["minibatches"],
                n_envs=self.hyperparameters["n_envs"],
                n_steps=self.hyperparameters["n_steps"],
                discount_factor=self.hyperparameters["discount_factor"],
                lam=self.hyperparameters["lam"],
                entropy_loss_scaling=self.hyperparameters["entropy_loss_scaling"],
                writer=writer,
                n_actions=self.n_actions
            )
        )

ppo_nfsp = ParallelPresetBuilder('ppo_nfsp', default_hyperparameters, PPONFSPPreset)

def nat_features():
    return nature_features(16)

def make_ppo_nfsp(env_name, device, _, **kwargs):
    venv = make_vec_env(env_name, device=device, vs_builtin=False)
    test_venv = make_vec_env(env_name, device=device, vs_builtin=True)

    hparams = kwargs.get('hparams', {})
    preset = ppo_nfsp.env(venv).device(device).hyperparameters(
        n_envs=venv.num_envs,
        feature_model_constructor=nat_features,
        **hparams
    ).build()

    experiment = ParallelEnvExperiment(preset, venv, test_env=test_venv)
    return experiment, preset, venv



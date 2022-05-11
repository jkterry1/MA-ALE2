from all.environments import GymVectorEnvironment
from all.experiments import ParallelEnvExperiment
from all.presets import atari
from models import impala_features, impala_value_head, impala_policy_head, nature_features
from env_utils import InvertColorAgentIndicator, make_vec_env
from all.bodies import DeepmindAtariBody
from models import ImpalaCNNLarge
from all import nn
import copy
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.agents import PPO, PPOTestAgent
from all.approximation import VNetwork, FeatureNetwork
from all.logging import DummyWriter
from all.optim import LinearScheduler
from all.policies import SoftmaxPolicy
from all.presets.builder import ParallelPresetBuilder

def nat_features(channels=10):
    return nature_features(channels)


class PPOAgent(PPO):

    def get_buffers(self) -> tuple:
        return ()

    def load_buffers(self, buffers: tuple):
        return


class PPOPreset(atari.PPOAtariPreset):

    def agent(self, writer=DummyWriter(), train_steps=float('inf')):
        n_updates = train_steps * self.hyperparameters['epochs'] * self.hyperparameters['minibatches'] / (
                    self.hyperparameters['n_steps'] * self.hyperparameters['n_envs'])

        feature_optimizer = Adam(self.feature_model.parameters(), lr=self.hyperparameters["lr"],
                                 eps=self.hyperparameters["eps"])
        value_optimizer = Adam(self.value_model.parameters(), lr=self.hyperparameters["lr"],
                               eps=self.hyperparameters["eps"])
        policy_optimizer = Adam(self.policy_model.parameters(), lr=self.hyperparameters["lr"],
                                eps=self.hyperparameters["eps"])

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

        return DeepmindAtariBody(
            PPOAgent(
                features,
                v,
                policy,
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
            ),
            frame_stack=0
        )

    def test_agent(self):
        features = FeatureNetwork(copy.deepcopy(self.feature_model))
        policy = SoftmaxPolicy(copy.deepcopy(self.policy_model))
        return DeepmindAtariBody(PPOTestAgent(features, policy), frame_stack=0)

ppo_preset = ParallelPresetBuilder('ppo_preset', atari.ppo.default_hyperparameters, PPOPreset)


def make_ppo_vec(env_name, device, _, **kwargs):
    n_envs = 12
    venv = make_vec_env(env_name, device=device, vs_builtin=False, num_envs=n_envs)
    test_venv = make_vec_env(env_name, device=device, vs_builtin=True, num_envs=n_envs)

    quiet = kwargs.get('quiet', False)
    hparams = kwargs.get('hparams', {})
    preset = ppo_preset.env(venv).device(device).hyperparameters(
        n_envs=venv.num_envs,
        feature_model_constructor=nat_features,
        **hparams
    ).build()

    experiment = ParallelEnvExperiment(preset, venv, test_env=test_venv, quiet=quiet)
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

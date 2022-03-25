from typing import Dict, Any
import dill
import json
import argparse
import os
import torch
from algorithms.nfsp import save_name
import numpy as np
import random
from experiment_train import trainer_types
import optuna
from optuna.trial import TrialState
import time
import ray
from all.experiments import MultiagentEnvExperiment


parser = argparse.ArgumentParser(description="Run an multiagent Atari benchmark.")

parser.add_argument("--num-gpus", type=int, default=1,
                    help="number of GPUs for this search process")
parser.add_argument("--frames", type=int, default=50e6, help="The number of training frames.")
parser.add_argument("--frames-per-save", type=int, default=None)
parser.add_argument("--trainer-type", type=str, default="nfsp_rainbow")
parser.add_argument("--num-eval-episodes", type=int, default=20,
                    help="how many evaluation episodes to run per training epoch")
parser.add_argument("--local", action='store_true', default=False,
                    help="create study locally (no SQL database)")
parser.add_argument("--envs", type=str, required=True,
                    help="must be comma-separated list of envs with no spaces!")
parser.add_argument("--study-name", type=str, default=None,
                    help="name of shared Optuna study for distributed training")
parser.add_argument("--db-name", type=str, default="maale",
                    help="name of SQL table name. Uses old name as default for testing purposes.")
parser.add_argument("--db-password", type=str)
parser.add_argument("--n-trials", type=int, default=100,
                    help="number of trials for EACH environment, or how many times hparams are sampled.")
parser.add_argument("--from-ckpt", action="store_true",
                    help="learning start from previous trained checkpoints")
args = parser.parse_args()
args.device = 'cuda' if args.num_gpus > 0 else 'cpu'



SQL_ADDRESS = f"mysql://database:{args.db_password}@35.194.57.226/{args.db_name}"

env_list = args.envs.split(',')



with open("plot_data/builtin_env_rewards.json", "r") as fd:
    builtin_rewards = json.load(fd)
with open("plot_data/rand_rewards.json", "r") as fd:
    rand_rewards = json.load(fd)


def sample_common_params(trial: optuna.Trial) -> Dict:
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    adam_eps = trial.suggest_loguniform("eps", 1e-6, 1e-4)

    return {
        "discount_factor": gamma,
        "lr": learning_rate,
        "eps": adam_eps,
    }


def sample_rainbow_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for Rainbow hyperparameters"""
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256, 512])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(5e4), int(1e5), int(1e6)])
    exploration_final_eps = trial.suggest_uniform("exploration_final_eps", 0, 0.2)
    exploration_fraction = trial.suggest_uniform("exploration_fraction", 0, 0.5)
    target_update_interval = trial.suggest_categorical("target_update_interval", [1, 1000, 5000, 10000, 15000, 20000])
    learning_starts = trial.suggest_categorical("learning_starts", [1000, 5000, 10000, 20000])
    noisy_linear_sigma = trial.suggest_uniform("noisy_linear_sigma", 0.1, 0.9)
    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 128, 256, 1000])
    n_quantiles = trial.suggest_int("n_quantiles", 5, 200)

    hyperparams = sample_common_params(trial)
    hyperparams.update({
        "minibatch_size": batch_size,
        "replay_buffer_size": buffer_size,
        "update_frequency": train_freq,
        "initial_exploration": exploration_fraction,
        "final_exploration": exploration_final_eps,
        "target_update_frequency": target_update_interval,
        "replay_start_size": learning_starts,
        "atoms": n_quantiles,
        "sigma": noisy_linear_sigma,
    })
    return hyperparams

def sample_nfsp_rainbow_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for NFSP Rainbow hyperparameters"""
    anticipatory = trial.suggest_loguniform("anticipatory", 0.01, 0.5)

    hyperparams = sample_rainbow_params(trial)
    hyperparams.update({
        "anticipatory": anticipatory,
    })
    return hyperparams


def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for PPO hyperparams."""
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    value_loss_scaling = trial.suggest_uniform("value_loss_scaling", 0, 1)

    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])

    # TODO: account when using multiple envs
    if batch_size > n_steps:
        batch_size = n_steps

    hyperparams = sample_common_params(trial)
    hyperparams.update({
        "n_steps": n_steps,
        "minibatch_size": batch_size,
        "entropy_loss_scaling": ent_coef,
        "clip_range": clip_range,
        "epochs": n_epochs,
        "lam": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "value_loss_scaling": value_loss_scaling,
    })
    return hyperparams


def normalize_score(score: np.ndarray, env_id: str) -> np.ndarray:
    """
    Normalize score to be in [0, 1] where 1 is maximal performance.
    :param score: unnormalized score
    :param env_id: environment id
    :return: normalized score
    """
    assert env_id in rand_rewards
    assert env_id in builtin_rewards

    rand_score = max(rand_rewards[env_id]['mean_rewards'].values())
    builtin_score = builtin_rewards[env_id]['mean_rewards']['first']
    return (score - builtin_score) / (rand_score - builtin_score)


@ray.remote(num_gpus=args.num_gpus)
def train(hparams, seed, trial, env_id):
    # set all hparams sampled from the trial
    buffer_size = hparams.get('replay_buffer_size', 0)
    experiment, preset, env = trainer_types[args.trainer_type](
        env_id, args.device, buffer_size,
        seed=seed,
        num_frames=args.frames,
        hparams=hparams,
        quiet=False,
    )

    is_ma_experiment = isinstance(experiment, MultiagentEnvExperiment)
    if is_ma_experiment: # TODO: not supported by ParallelEnvExperiment yet
        experiment.seed_env(seed)

    save_folder = "checkpoint/" + save_name(args.trainer_type, env_id, buffer_size, args.frames, seed)
    norm_eval_returns = []
    norm_return, avg_norm_return = None, None

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    num_frames_train = int(args.frames)
    frames_per_save = args.frames_per_save or max(num_frames_train // 100, 1)

    # Start from the last preset
    frame_start = 0
    if args.from_ckpt:
        if len(os.listdir(save_folder)) != 0:
            frame_start = sorted([int(ckpt.strip('.pt')) for ckpt in os.listdir(save_folder)])[-1]
            preset = torch.load(f"{save_folder}/{frame_start:09d}.pt")


    for frame in range(frame_start, num_frames_train, frames_per_save):
        experiment.train(frames=frame)
        torch.save(preset, f"{save_folder}/{frame + frames_per_save:09d}.pt")

        eval_returns = experiment.test(episodes=args.num_eval_episodes)
        if is_ma_experiment: # MultiAgentEnvExperiment returns dict, but evals are always one key
            assert len(eval_returns) == 1
            eval_returns = list(eval_returns.values())[0]
            experiment._save_model()  # not implemented in Parallel yet
        # for aid, returns in eval_returns.items():
        mean_return = np.mean(eval_returns)
        norm_return = normalize_score(mean_return, env_id=env_id)
        norm_eval_returns.append(norm_return)
        avg_norm_return = np.mean(norm_eval_returns)

        # Handle pruning based on the intermediate value.
        trial.report(value=avg_norm_return, step=frame + frames_per_save)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # if ckpt is already fully existed
    if frame_start >= num_frames_train:
        eval_returns = experiment.test(episodes=args.num_eval_episodes)
        if is_ma_experiment: # MultiAgentEnvExperiment returns dict, but evals are always one key
            assert len(eval_returns) == 1
            eval_returns = list(eval_returns.values())[0]
            experiment._save_model()  # not implemented in Parallel yet
        # for aid, returns in eval_returns.items():
        mean_return = np.mean(eval_returns)
        norm_return = normalize_score(mean_return, env_id=env_id)
        norm_eval_returns.append(norm_return)
        avg_norm_return = np.mean(norm_eval_returns)

        # Handle pruning based on the intermediate value.
        trial.report(value=avg_norm_return, step=frame + frames_per_save)

    return avg_norm_return

if args.trainer_type == "shared_rainbow":
    sampler_fn = sample_rainbow_params
elif args.trainer_type == "nfsp_rainbow":
    sampler_fn = sample_nfsp_rainbow_params
elif args.trainer_type == "shared_ppo":
    sampler_fn = sample_ppo_params
else:
    raise ValueError

def objective_all(trial):
    """Get hyperparams for trial"""
    hparams = sampler_fn(trial)

    seed = trial.number
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    futures = [train.remote(hparams, seed, trial, env_id) for env_id in env_list]
    norm_returns = ray.get(futures)

    print(hparams)
    print(norm_returns)

    return np.mean(norm_returns)


if __name__ == "__main__":
    if args.local:
        ray.init(num_gpus=args.num_gpus, local_mode=True)
        time.sleep(10)
        study = optuna.create_study(direction="maximize",
                                    study_name=args.study_name,
                                    load_if_exists=True)
    else:
        import pathlib
        temp_dir = pathlib.Path(__file__).parent.resolve().joinpath("raytmp")
        ray.init(num_gpus=args.num_gpus, local_mode=False, _temp_dir=str(temp_dir))
        time.sleep(10)
        study = optuna.create_study(direction="maximize",
                                    storage=SQL_ADDRESS,
                                    study_name=args.study_name,
                                    load_if_exists=True)

    study.optimize(objective_all, n_trials=args.n_trials, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    with open(f"best_params_{args.env}.pkl", 'wb') as fd:
        dill.dump(trial.params, fd)
from typing import Dict, Any
import dill
import json
import argparse
import os
import torch
from algorithms.nfsp import save_name
import numpy as np
from argparse import Namespace
import random
from experiment_train import trainer_types
import optuna
from optuna.trial import TrialState
from multiprocessing import Process


def validate_args(args: Namespace):
    env_list = args.envs.split(',')
    if len(env_list) > 1 and args.num_concurrent > 1:
        raise ValueError("This hasn't been tested!")


parser = argparse.ArgumentParser(description="Run an multiagent Atari benchmark.")

parser.add_argument("--device", default="cuda",
                    help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0).")
parser.add_argument("--replay_buffer_size", default=1000000, type=int,
                    help="The size of the replay buffer, if applicable")
parser.add_argument("--frames", type=int, default=50e6, help="The number of training frames.")
parser.add_argument("--trainer-type", type=str, default="nfsp_rainbow")
parser.add_argument("--num-eval-episodes", type=int, default=20,
                    help="how many evaluation episodes to run per training epoch")
parser.add_argument("--local", action='store_true', default=False,
                    help="create study locally (no SQL database)")
parser.add_argument("--envs", type=str, required=True,
                    help="must be comma-separated list of envs with no spaces!")
parser.add_argument("--num-concurrent", type=int, default=1,
                    help="how many trials to run concurrently")
parser.add_argument("--study-name", type=str, default=None,
                    help="name of shared Optuna study for distributed training")
parser.add_argument("--db-password", type=str)
args = parser.parse_args()
args = validate_args(args)



SQL_ADDRESS = f"mysql://database:{args.db_password}@35.194.57.226/maale"

env_list = args.envs.split(',')



with open("plot_data/builtin_env_rewards.json", "r") as fd:
    builtin_rewards = json.load(fd)
with open("plot_data/rand_rewards.json", "r") as fd:
    rand_rewards = json.load(fd)



def sample_dqn_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for DQN hyperparams.
    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256, 512])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(5e4), int(1e5), int(1e6)])
    exploration_final_eps = trial.suggest_uniform("exploration_final_eps", 0, 0.2)
    exploration_fraction = trial.suggest_uniform("exploration_fraction", 0, 0.5)
    target_update_interval = trial.suggest_categorical("target_update_interval", [1, 1000, 5000, 10000, 15000, 20000])
    learning_starts = trial.suggest_categorical("learning_starts", [0, 1000, 5000, 10000, 20000])

    anticipatory = trial.suggest_loguniform("anticipatory", 0.01, 0.5)
    noisy_linear_sigma = trial.suggest_uniform("noisy_linear_sigma", 0.1, 0.9)

    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 128, 256, 1000])
    subsample_steps = trial.suggest_categorical("subsample_steps", [1, 2, 4, 8])
    gradient_steps = max(train_freq // subsample_steps, 1)
    n_quantiles = trial.suggest_int("n_quantiles", 5, 200)

    hyperparams = {
        "discount_factor": gamma,
        "lr": learning_rate,
        "minibatch_size": batch_size,
        "replay_buffer_size": buffer_size,
        "update_frequency": train_freq,
        "gradient_steps": gradient_steps,
        "initial_exploration": exploration_fraction,
        "final_exploration": exploration_final_eps,
        "target_update_frequency": target_update_interval,
        "replay_start_size": learning_starts,
        "atoms": n_quantiles,
        "sigma": noisy_linear_sigma,
        "anticipatory": anticipatory,
    }
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


def objective(trial, env_id: str, parallel_num: int, hparams: dict):
    """Get hyperparams for trial"""

    seed = trial.number
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # set all hparams sampled from the trial
    experiment, preset, env = trainer_types[args.trainer_type](
        env_id, args.device, args.replay_buffer_size,
        seed=seed,
        num_frames=args.frames,
        hparams=hparams
    )
    experiment.seed_env(seed)
    save_folder = "checkpoint/" \
                  + save_name(args.trainer_type, env_id, args.replay_buffer_size, args.frames, seed) \
                  + f"/{parallel_num}"
    all_eval_returns = []
    norm_eval_returns = []
    norm_return = None

    os.makedirs(save_folder)
    num_frames_train = int(args.frames)
    frames_per_save = num_frames_train // 100
    for frame in range(0, num_frames_train, frames_per_save):
        experiment.train(frames=frame)
        torch.save(preset, f"{save_folder}/{frame + frames_per_save:09d}.pt")

        eval_returns = experiment.test(episodes=args.num_eval_episodes)
        for aid, returns in eval_returns.items():
            mean_return = np.mean(returns)
            norm_return = normalize_score(mean_return, env_id=env_id)
            all_eval_returns.append(mean_return)
            norm_eval_returns.append(norm_return)
        experiment._save_model()

        # Handle pruning based on the intermediate value.
        trial.report(value=norm_return, step=frame + frames_per_save)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return norm_return



if __name__ == "__main__":
    if args.local:
        study = optuna.create_study(direction="maximize",
                                    study_name=args.study_name,
                                    load_if_exists=True)
    else:
        study = optuna.create_study(direction="maximize",
                                    storage=SQL_ADDRESS,
                                    study_name=args.study_name,
                                    load_if_exists=True)

    procs = []
    for env_id in env_list:
        for i in range(args.num_concurrent):
            objective_fn = lambda trial: objective(trial, env_id=env_id, parallel_num=i, hparams=sample_dqn_params(trial))
            proc_target = lambda: study.optimize(objective_fn, n_trials=100//args.num_concurrent, timeout=600)
            p = Process(target=proc_target)
            p.start()
            procs.append(p)
    for p in procs:
        p.join()

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

import optuna
from typing import Dict, Any



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
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(5e4), int(1e5)])
    exploration_final_eps = trial.suggest_uniform("exploration_final_eps", 0, 0.2)
    exploration_fraction = trial.suggest_uniform("exploration_fraction", 0, 0.5)
    target_update_interval = trial.suggest_categorical("target_update_interval", [1, 1000, 5000, 10000, 15000, 20000])
    learning_starts = trial.suggest_categorical("learning_starts", [1000, 5000, 10000, 20000])
    noisy_linear_sigma = trial.suggest_uniform("noisy_linear_sigma", 0.1, 0.9)
    train_freq = trial.suggest_categorical("train_freq", [32, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])
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

def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for PPO hyperparams."""
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512])
    ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
    minibatches = trial.suggest_categorical("minibatches", [1, 4, 8, 16, 32])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    value_loss_scaling = trial.suggest_uniform("value_loss_scaling", 0, 1)

    clip_final = trial.suggest_loguniform("clip_final", 0.0001, 0.1)
    clip_initial = trial.suggest_loguniform("clip_initial", 0.01, 0.5)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.1, 5)

    hyperparams = sample_common_params(trial)
    hyperparams.update({
        "n_steps": n_steps,
        "minibatches": minibatches,
        "entropy_loss_scaling": ent_coef,
        "clip_initial": clip_initial,
        "clip_final": clip_final,
        "epochs": n_epochs,
        "lam": gae_lambda,
        "clip_grad": max_grad_norm,
        "value_loss_scaling": value_loss_scaling,
    })
    return hyperparams

def sample_nfsp_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for extra NFSP hyperparameters"""
    anticipatory = trial.suggest_loguniform("anticipatory", 0.01, 0.5)
    reservoir_buffer_size = trial.suggest_categorical("reservoir_buffer_size", [int(1e5), int(5e5), int(1e6)])
    hyperparams = {
        "anticipatory": anticipatory,
        "reservoir_buffer_size": reservoir_buffer_size,
    }
    return hyperparams

def sample_nfsp_rainbow_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for NFSP Rainbow hyperparameters"""
    hyperparams = sample_rainbow_params(trial)
    hyperparams.update(sample_nfsp_params(trial))
    return hyperparams

def sample_nfsp_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for NFSP Rainbow hyperparameters"""
    hyperparams = sample_ppo_params(trial)
    hyperparams.update(sample_nfsp_params(trial))
    return hyperparams
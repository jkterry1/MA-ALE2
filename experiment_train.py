import argparse
import os
import torch
from algorithms.shared_rainbow import make_rainbow_preset
from algorithms.independent_rainbow import make_indepedent_rainbow
from algorithms.shared_ppo import make_ppo_vec, make_ppo_vec_largenet
from algorithms.rainbow_nfsp import make_nfsp_rainbow
from algorithms.parallel_rainbow import make_parallel_rainbow
from algorithms.parallel_rainbow_nfsp import make_parallel_rainbow_nfsp
from algorithms.ppo_nfsp import make_ppo_nfsp
from shared_utils import save_name
import numpy as np
import time
import random
from glob import glob

import datetime
def datetime_str():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H:%M:%S")

trainer_types = {
    "shared_rainbow": make_rainbow_preset,
    "independent_rainbow": make_indepedent_rainbow,
    "shared_ppo": make_ppo_vec,
    # "shared_ppo_ram": make_ppo_ram_vec,
    "shared_ppo_largenet": make_ppo_vec_largenet,
    "nfsp_rainbow": make_nfsp_rainbow,
    "parallel_rainbow": make_parallel_rainbow,
    "parallel_rainbow_nfsp": make_parallel_rainbow_nfsp,
    "nfsp_ppo": make_ppo_nfsp,
}



def main(return_eval=False):
    parser = argparse.ArgumentParser(description="Run an multiagent Atari benchmark.")
    parser.add_argument("env", help="Name of the Atari game (e.g. Pong).")
    parser.add_argument("trainer_type", help="Name of the type of training method.")
    parser.add_argument(
        "--device",
        default="cuda",
        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--replay_buffer_size",
        default=1000000,
        type=int,
        help="The size of the replay buffer, if applicable",
    )
    parser.add_argument(
        "--frames", type=int, default=50e6, help="The number of training frames."
    )
    parser.add_argument(
        "--experiment-seed", type=int, default=int(time.time()),
        help="The unique id of the experiment run (for running multiple experiments)."
    )
    parser.add_argument("--frames-per-save", type=int, default=None)
    parser.add_argument("--train-against-builtin", action='store_true')
    args = parser.parse_args()

    np.random.seed(args.experiment_seed)
    random.seed(args.experiment_seed)
    torch.manual_seed(args.experiment_seed)

    experiment, preset, env = trainer_types[args.trainer_type](
        args.env, args.device, args.replay_buffer_size,
        seed=args.experiment_seed,
        num_frames=args.frames,
        train_against_builtin=args.train_against_builtin,
    )
    env.seed(args.experiment_seed)
    train_builtin_str = "_builtin" if args.train_against_builtin else ""
    save_folder = "checkpoint/" + save_name(args.trainer_type + train_builtin_str, args.env,
                                            args.replay_buffer_size, args.frames, args.experiment_seed)
    os.makedirs(save_folder)
    num_frames_train = int(args.frames)
    frames_per_save = args.frames_per_save or min(500000, max(num_frames_train // 100, 1))
    for frame in range(0,num_frames_train,frames_per_save):
        experiment.train(frames=frame)
        torch.save(preset, f"{save_folder}/{frame+frames_per_save:09d}.pt")
        checkpoint_files = sorted(glob(f"{save_folder}/*.pt"))
        for ckpt_file in checkpoint_files[:-1]:
            os.remove(ckpt_file)

    if return_eval:
        returns = experiment.test(episodes=1)
        experiment._save_model()
        return returns



if __name__ == "__main__":
    main()

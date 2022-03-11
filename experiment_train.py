import argparse
import os
import torch
from algorithms.shared_rainbow import make_rainbow_preset
from algorithms.independent_rainbow import make_indepedent_rainbow
from algorithms.shared_ppo import make_ppo_vec, make_ppo_vec_largenet
from algorithms.nfsp import make_nfsp_rainbow, save_name
import numpy as np
import time
import random

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

    args = parser.parse_args()

    np.random.seed(args.experiment_seed)
    random.seed(args.experiment_seed)
    torch.manual_seed(args.experiment_seed)

    experiment, preset, env = trainer_types[args.trainer_type](args.env, args.device, args.replay_buffer_size,
                                                               seed=args.experiment_seed,
                                                               num_frames=args.frames)
    env.seed(args.experiment_seed)
    # run_experiment()
    save_folder = "checkpoint/" + save_name(args.trainer_type, args.env, args.replay_buffer_size,
                                            args.frames, args.experiment_seed)
    os.makedirs(save_folder)
    num_frames_train = int(args.frames)
    frames_per_save = num_frames_train//100
    for frame in range(0,num_frames_train,frames_per_save):
        experiment.train(frames=frame)
        torch.save(preset, f"{save_folder}/{frame+frames_per_save:09d}.pt")

    if return_eval:
        returns = experiment.test(episodes=1)
        experiment._save_model()
        return returns



if __name__ == "__main__":
    main()

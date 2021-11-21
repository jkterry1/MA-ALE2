from all_envs import all_environments, builtin_envs
import glob
import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num-parallel", default=1, type=int)
parser.add_argument("--env", default="all", type=str,
                    help="which env to eval - defaults to all built-ins")
args = parser.parse_args()

four_p_envs = {
"warlords_v2",
"quadrapong_v3",
"volleyball_pong_v",
"foozpong_v2",
}

experiment_configs = [
    # ("shared_rainbow", 1000000),
    # ("shared_ppo", 1),
    ('nfsp_rainbow', 1000000),
]

envs = [env for env in builtin_envs if 'tennis' not in env]
if args.env != "all":
    envs = [args.env]  # only run the provided env
num_frames_train = 50_000_000
frames_per_save = num_frames_train//100

eval_frames = 125000
base_folder = "checkpoint"

num_experiments = 3 #5

vs_builtin = True

device = "--device=cuda"

if vs_builtin:
    all_environments = {name:env for name, env in all_environments.items() if name in builtin_envs}

def make_name_pre(trainer, env, buf_size, experiment):
    # return f"{trainer}_{env}_RB{buf_size}_F{num_frames_train}_S{experiment}"
    return f"{trainer}/{env}/RB{buf_size}_F{num_frames_train}_S{experiment}"

from nfsp import save_name

agent_2p_list = ["first_0", "second_0"]
agent_4p_list = agent_2p_list + ["third_0", "fourth_0"]
run_strs = []
for env in all_environments:
    for experiment in range(num_experiments):
        for trainer, buf_size in experiment_configs:
            # handle datetime for multiples; grab most recent
            checkpoint_folder_pre = f"{base_folder}/{make_name_pre(trainer, env, buf_size, experiment)}"
            matches = glob.glob(checkpoint_folder_pre + "*")
            if not matches:
                print(f"Skipping directory {matches}, no logged runs")
                continue  # no logged runs
            checkpoint_folder = matches[-1]

            agent_list = agent_4p_list if env in four_p_envs else agent_2p_list
            for checkpoint in range(frames_per_save, num_frames_train, frames_per_save):
                vs_random_list = ["", "--vs-random"] if not vs_builtin else [""]
                matches = glob.glob(checkpoint_folder + "/*")
                seed_folder = matches[-1]
                if not os.path.isfile(seed_folder + f"/{checkpoint:09d}.pt"):
                    continue  # no checkpoints; compatibility for incomplete runs
                for vs_random in vs_random_list:
                    # if vs_random:
                    #     for agent in agent_list:
                    #         frames = eval_frames
                    #         print(f"workon main_env && python experiment_eval.py {env} {checkpoint:09d} {checkpoint_folder} --frames={frames} --agent={agent} {vs_random}")
                    # else:
                    agent = "first_0"
                    vs_builtin_str = "--vs-builtin" if vs_builtin else ''
                    frames = eval_frames

                    run_strs.append(f"python experiment_eval.py {env} {checkpoint:09d} {seed_folder} --frames={frames} --agent={agent} {vs_random} {vs_builtin_str} {device}")
        # frames = eval_frames*4

num_parallel = args.num_parallel
for run_i in range(len(run_strs) // num_parallel):
    offset = num_parallel * run_i
    runs = run_strs[offset:offset + num_parallel]
    procs = []
    for run in runs:
        procs.append(subprocess.Popen(run.split()))
    for proc in procs:
        proc.wait()
remaining = len(run_strs) % num_parallel
runs = run_strs[-remaining:]
for run in runs:
    subprocess.Popen(run.split())

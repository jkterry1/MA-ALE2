from all_envs import all_environments, builtin_envs
import subprocess

experiment_configs = [
    # ("shared_ppo", 1),
    # ("shared_ppo_largenet", 1),
    # ("shared_rainbow", 1000000),
    # ("independent_rainbow", 1000000),
    # ("independent_rainbow", 100000),
    ('nfsp_rainbow', 1000000),
]

envs = [env for env in builtin_envs if 'tennis' not in env]
num_frames = 50000000
num_experiments = 2
num_parallel = 2
# for env_name in sorted(all_environments):
run_strs = []
for env_name in sorted(envs):
    for exp_num in range(num_experiments):
        for algo_name, replay_size in experiment_configs:
            run_strs.append(f"python experiment_train.py {env_name} {algo_name} --replay_buffer_size={replay_size} --frames={num_frames} --experiment-seed={exp_num}")
for run_i in range(len(run_strs) // num_parallel):
    offset = num_parallel * run_i
    runs = run_strs[offset:offset+num_parallel]
    procs = []
    for run in runs:
        procs.append(subprocess.Popen(run.split()))
    for proc in procs:
        proc.wait()
remaining = len(run_strs) % num_parallel
runs = run_strs[-remaining:]
for run in runs:
    subprocess.Popen(run.split())
if __name__ == '__main__':
    pass
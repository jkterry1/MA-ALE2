import argparse
import pathlib


parser = argparse.ArgumentParser()
parser.add_argument("--study-name", type=str, required=True)
parser.add_argument("--db-password", type=str, required=True)
parser.add_argument("--db-name", type=str, required=True)
parser.add_argument("--n-trials", type=int, default=100)
parser.add_argument("--num-jobs", type=int, default=1,
                    help="how many python processes to split n-trials across")
parser.add_argument("--gpus-per-job", type=int, required=True,
                    help="how many GPUs to allocate per python process (training 6 envs)")
parser.add_argument("--trainer-type", type=str, default="nfsp_rainbow",
                    choices=["nfsp_rainbow","shared_rainbow","shared_ppo"])
# parser.add_argument("--redis-address", type=str, default=None,   # TODO
#                     help="redis address to connect to Head node and share resources")
args = parser.parse_args()


eval_envs = [
    "boxing_v1",
    "double_dunk_v2",
    "ice_hockey_v1",
    "pong_v2",
    "surround_v1",
    "tennis_v2",
]
envs_str = ','.join(eval_envs)

lines = []
for job_i in range(args.num_jobs):
    lines.append(f"python -O hparam_search.py --trainer-type {args.trainer_type} --envs {envs_str} "
                 f"--study-name {args.study_name} "
                 f"--db-password {args.db_password} --db-name {args.db_name} --n-trials {args.n_trials//args.num_jobs} "
                 f"--num-gpus {args.gpus_per_job} \n")


# Remove command file if already exists
cmd_file = pathlib.Path("hparam_search_cmds.txt")
cmd_file.unlink(missing_ok=True)

with open("hparam_search_cmds.txt", "w") as fd:
    fd.writelines(lines)

import argparse
import pathlib


parser = argparse.ArgumentParser()
parser.add_argument("--study-name", type=str, required=True)
parser.add_argument("--study-create", default=False, action="store_true",
                    help="will create study if does not already exist")
parser.add_argument("--db-password", type=str, required=True)
parser.add_argument("--db-name", type=str, required=True)
parser.add_argument("--db-user", type=str, default="database")
parser.add_argument("--max-trials", type=int, default=150)
parser.add_argument("--num-jobs", type=int, default=1,
                    help="how many python processes to split n-trials across")
parser.add_argument("--gpus-per-job", type=int, required=True,
                    help="how many GPUs to allocate per python process (training 6 envs)")
parser.add_argument("--trainer-type", type=str, default="nfsp_rainbow",
                    choices=["nfsp_rainbow","shared_rainbow","shared_ppo","nfsp_ppo","parallel_rainbow","parallel_rainbow_nfsp"])
parser.add_argument("--filename", type=str, default="hparam_search_cmds.txt")
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
                 f"--study-name {args.study_name} {'--study-create' if args.study_create else ''} "
                 f"--db-password {args.db_password} --db-name {args.db_name} --db-user {args.db_user} "
                 f"--max-trials {args.max_trials} --num-gpus {args.gpus_per_job} \n")


# Remove command file if already exists
cmd_file = pathlib.Path("hparam_search_cmds.txt")
cmd_file.unlink(missing_ok=True)

with open(args.filename, "w") as fd:
    fd.writelines(lines)

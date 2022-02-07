import argparse
import pathlib


parser = argparse.ArgumentParser()
parser.add_argument("--study-name", type=str, required=True)
parser.add_argument("--db-password", type=str, required=True)
args = parser.parse_args()


eval_envs = [
    "boxing_v1",
    "double_dunk_v2",
    "ice_hockey_v1",
    "pong_v2",
    "surround_v1",
    "tennis_v2",
]

lines = []
for env in eval_envs:
    lines.append(f"python -O hparam_search.py --env {env} --study-name {args.study_name} --db-password {args.db_password}")


# Remove command file if already exists
cmd_file = pathlib.Path("hparam_search_cmds.txt")
cmd_file.unlink(missing_ok=True)

with open("hparam_search_cmds.txt", "w") as fd:
    fd.writelines(lines)

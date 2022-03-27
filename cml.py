#!/usr/bin/env python3

r"""Launch file to dispatch jobs onto the CML SLURM computation cluster.

Messing up and launching a large number of jobs that fail immediately can slow down the cluster!
Launch debug options and small tests until everything runs as it is supposed to be.

-----------------------------------------------------------------------------------------------------------------------
USAGE: python cml.py file_list --arguments

- Required: file_list : A file containing python calls with a single call per line.
The file may contain spaces, empty lines and comments (#). These will be scrubbed automatically.
Example file_list.sh:

'''
python something.py --someargument # this is the first job

# this is the second job:
python something.py --someotherargument
'''

- Necessary Options:
--conda                   - Name of the chosen anaconda environment.
- Optional but important:
--email                   - Set a custom email, if you dont check your @umiacs.umd.edu account
--name                    - Set a custom name for the job  that will display in squeue.
--min_preemptions         - Use this option to start high-priority jobs only on nodes where no scavengers are running.

- Quality of service options:
--qos, --gpus, --mem,     - These options work exactly as in the usual srun commands. NOTE THAT ALL OPTIONS ARE PER JOB.
--timelimit                 Use "show_qos" and "show_assoc" on the login shell to see your options.
                            4 CPUs are automatically allocated per GPU.

--throttling              - Use this option to reduce the number of jobs that will be launched simultaneously.
--exclude                 - Use this option to exclude some nodes by their node name, e.g. 'cmlgrad05'
                            Separate multiple node names by commas


If you do not want to use anaconda, hard-code or modify the lines

source {"/".join(path.split("/")[:-2])}/etc/profile.d/conda.sh
conda activate {path}

in SBATCH_PROTOTYPE, replacing them with your personal choice of environment.
"""

import argparse
import os
import subprocess
import time
import warnings
import getpass
import random

parser = argparse.ArgumentParser(description="Dispatch a list of python jobs from a given file to the CML cluster")

# Central:
parser.add_argument("file", type=argparse.FileType())

parser.add_argument("--conda", default="dl", type=str, help="Name of the targeted anaconda environment")
parser.add_argument("--email", default=None, type=str, help="Your email if not @umiacs.umd.edu")
parser.add_argument("--min_preemptions", action="store_true", help="Launch on nodes where this user has no scav jobs.")

parser.add_argument("--qos", default="scav", type=str, help="QOS, choose default, medium, high, scav")
parser.add_argument("--name", default=None, type=str, help="Name that will be displayed in squeue. Default: file name")
parser.add_argument("--gpus", default="1", type=int, help="Requested GPUs PER job")
parser.add_argument("--mem", default="32", type=int, help="Requested memory PER job")
parser.add_argument("--timelimit", default=72, type=int, help="Requested hour limit PER job")
parser.add_argument("--throttling", default=None, type=int, help="Launch only this many jobs concurrently")
parser.add_argument("--exclude", default=None, type=str, help="Exclude malfunctioning nodes. Should be a node name.")
parser.add_argument("--nodelist", default=None, type=str, help="Specify nodes to run on.")


args = parser.parse_args()


# Parse and validate input:
if args.name is None:
    dispatch_name = args.file.name
else:
    dispatch_name = args.name

args.conda = args.conda.rstrip("/")
# Parse conda and find environment locations:
raw_output = subprocess.run(["conda", "info", "--envs"], capture_output=True)
environments = [s.split() for s in str(raw_output.stdout).split("\\n")]
path = None
for line in environments:
    if args.conda in line[0]:
        path = line[-1].replace(" ", "")
        break
if path is None:
    raise ValueError(f"Could not find anaconda environment {args.conda}.")
else:
    print(f"Found anaconda environment at path {path}.")

# Usage warnings:
if args.mem > 385:
    raise ValueError("Maximal node memory exceeded.")
if args.gpus > 8:
    raise ValueError("Maximal node GPU number exceeded.")
if args.qos == "high" and args.gpus > 4:
    warnings.warn("QOS only allows for 4 GPUs, GPU request has been reduced to 4.")
    args.gpus = 4
if args.qos == "medium" and args.gpus > 2:
    warnings.warn("QOS only allows for 2 GPUs, GPU request has been reduced to 2.")
    args.gpus = 2
if args.qos == "default" and args.gpus > 1:
    warnings.warn("QOS only allows for 1 GPU, GPU request has been reduced to 1.")
    args.gpus = 1
if args.mem / args.gpus > 48:
    warnings.warn(
        "You are oversubscribing to memory. " "This might leave some GPUs idle as total node memory is consumed."
    )
if args.qos == "high" and args.timelimit > 36:
    warnings.warn("QOS only allows for 36 hours. Timelimit request has been reduced to 48.")
    args.timelimit = 36

# 1) Strip the file_list of comments and blank lines
content = args.file.readlines()
jobs = [c.strip().split("#", 1)[0] for c in content if "python" in c and c[0] != "#"]

print(f"Detected {len(jobs)} jobs.")
if len(jobs) < 1:
    raise ValueError("Detected no valid jobs in given file!")

# Write the clean file list
authkey = random.randint(10 ** 5, 10 ** 6 - 1)
with open(f".cml_job_list_{authkey}.temp.sh", "w") as file:
    file.writelines(chr(10).join(job for job in jobs))
    file.write("\n")


# 2) Decide which nodes not to use in scavenger:
username = getpass.getuser()

all_nodes = set(f"cml{i:02}" for i in range(17)) | set(f"cmlgrad0{i}" for i in range(8))
banned_nodes = set()

safeguard = [username]
if args.min_preemptions:
    try:
        raw_status = subprocess.run("squeue", capture_output=True)
        cluster_status = [s.split() for s in str(raw_status.stdout).split("\\n")]
        for sjob in cluster_status[1:-1]:
            if sjob[1] == "scavenger" and sjob[3] in safeguard and "cml" in sjob[-1]:
                banned_nodes.add(sjob[-1])
    except FileNotFoundError:
        print("Node exclusion only works when called on cml nodes.")
if args.exclude is not None:
    for node_name in args.exclude.replace(" ", "").split(","):
        banned_nodes.add(node_name)
node_list = sorted(all_nodes - banned_nodes)
banned_nodes = sorted(banned_nodes)

# 3) Prepare environment
if not os.path.exists("log"):
    os.makedirs("log")

# 4) Construct the sbatch launch file
if args.qos == "scav":
    cml_account = "scavenger"
elif args.qos in ["high", "very_high", "high_long"]:
    cml_account = "tomg"
else:
    cml_account = "cml"

if args.qos == "cpu":
    args.gpus = 0
    partition = "cpu"
    cpus = 22
elif args.qos == "scav":
    partition = "scavenger"
    cpus = min(args.gpus * 4, 32)
else:
    partition = "dpart"
    cpus = min(args.gpus * 4, 32)

SBATCH_PROTOTYPE = f"""#!/bin/bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name={''.join(e for e in dispatch_name if e.isalnum())}
#SBATCH --array={f"1-{len(jobs)}" if args.throttling is None else f"1-{len(jobs)}%{args.throttling}"}
#SBATCH --output=log/%x_%A_%a.log
#SBATCH --error=log/%x_%A_%a.log
#SBATCH --time={args.timelimit}:00:00
#SBATCH --account={cml_account}
#SBATCH --qos={args.qos if args.qos != "scav" else "scavenger"}
#SBATCH --gres=gpu:{args.gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --partition={partition}
{f"#SBATCH --exclude={','.join(str(node) for node in banned_nodes)}" if banned_nodes else ''}
#SBATCH --mem={args.mem}gb
#SBATCH --mail-user={args.email if args.email is not None else username + "@umiacs.umd.edu"}
#SBATCH --mail-type=FAIL,ARRAY_TASKS
{f"#SBATCH --nodelist={args.nodelist}" if args.nodelist else ''}

source {"/".join(path.split("/")[:-2])}/etc/profile.d/conda.sh
conda activate {path}

export MASTER_PORT=$(shuf -i 2000-65000 -n 1) # Remember that these are fixed across the entire array
export MASTER_ADDR=`/bin/hostname -s`

srun $(head -n $((${{SLURM_ARRAY_TASK_ID}})) .cml_job_list_{authkey}.temp.sh | tail -n 1)

"""


# cleanup:
# rm .cml_job_list_{authkey}.temp.sh
# rm .cml_launch_{authkey}.temp.sh
# trap "rm -f $.cml_launch_{authkey}" EXIT
# this breaks SLURM in some way?

# Write launch commands to file
with open(f".cml_launch_{authkey}.temp.sh", "w") as file:
    file.write(SBATCH_PROTOTYPE)


# 5) Print launch information
print("Launch prototype is ...")
print("---------------")
print(SBATCH_PROTOTYPE)
print("---------------")
print(chr(10).join("srun " + job for job in jobs))

node_string = "all nodes" if len(banned_nodes) == 0 else f'nodes {",".join(str(node) for node in node_list)}'
print(f"Preparing {len(jobs)} jobs as user {username}" f" for launch on {node_string} in 10 seconds...")
print("Terminate if necessary ...")
for _ in range(10):
    time.sleep(1)

# 6) Launch

# Execute file with sbatch
subprocess.run(["/usr/bin/sbatch", f".cml_launch_{authkey}.temp.sh"])
print("Subprocess launched ...")
time.sleep(3)
subprocess.run("squeue")

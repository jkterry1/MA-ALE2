## Install

```
pip install -r requirements.txt 
```

## Run

```
python3 plot_all_one.py plot_data/builtin_results.csv --vs-builtin
```

Generates the plots vs the builtin agent

```
python3 plot_all_one.py plot_data/all_out.txt  
```

Generates the plots vs the random agent

Plots can be found near the data file, i.e. `plot_data/all_out.txt.png`


```
python3 experiment_train.py boxing_v1 nfsp_rainbow  
```

### Basic local hyperparameter search 
```
python3 hparam_search.py --env boxing_v1 --local 
```

### Evaluate with fixed checkpoint 
```
python3 experiment_eval.py boxing_v1 nfsp_rainbow [checkpoint_num] [path/to/checkpoint/dir] 
```
For example:  
`python3 experiment_eval.py pong_v2 000500000 checkpoint/shared_rainbow/pong_v2/RB1000000_F50000000.0_S1636751414`

### Run hyperparameter search on Slurm HPC 

- Generate run command file:  
```
python gen_hparam_search_cmds.py --study-name [name] \ 
    --db-password [password] --db-name [name] --num-concurrent [number]
```
- Start Slurm job (ensure enough resources are given!):
```
python cml.py hparam_search_cmds.txt --conda ma-ale --min_preemptions \
    --gpus [num] --mem [48*num_gpus] > optuna.log 2>&1 
```



## Files Overview

* Environment code
    * all_envs.py  
        * contains list of pettingzoo environments that should be trained
    * env_utils.py
        * contains environment preprocessing code 
    * my_env.py
        * MultiagentPettingZooEnv wrapper needed for NFSP  
* Policy code (each one of these has a function that makes a trainable ALL agent).
    * nfsp.py
    * shared_ppo.py
    * shared_rainbow.py
    * shared_utils.py
    * shared_vqn.py (not used/working!)
    * independent_rainbow.py
    * independent_ppo.py
    * model.py
        * some experimental models to use for policies
* Training code
    * experiment_train.py
        * trains agent returned by policy code
    * gen_train_runs.py
        * Generates many command line calls to experiment_train.py so that many experiments can be run with Slurm, Kabuki, or another job execution service.
* Evaluation code
    * experiment_eval.py
        * evaluates agent returned by checkpoint
        * can evaluate vs random agent, vs builtin agent (on specific environments), or vs trained opponent.
    * ale_rand_test.py
        * evaluates random agent vs random agent on all the environments, reports the results in a json file
    * ale_rand_test_builtin.py
        * evaluates random agent vs builtin agent on all the environments, reports the results in a json file
    * generate_evals.py
        * generates many calls to experiment_eval.py so that the evaluation jobs can be run with Slurm, Kabuki, or another job execution service
* Plotting code
    * plot_all_one.py
        * Looks at input csv file, specific random data file inside plot_data folder, and generates a plots with the results
* Hyperparameter search code 
    * gen_hparam_search_cmds.py 
        * Writes a file with many calls to "hparam_search.py", currently one per environment
        * Set up like this to use the "cml.py" SLURM tool to easily run batch array jobs on CML
    * hparam_search.py
        * Uses optuna with provided study name (allowing distributed) to optimize over normalized env scores
        * This is doing asynchronous optimization between envs, syncing to shared SQL result database
* 

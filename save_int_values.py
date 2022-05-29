"""
Save "int_values.pkl" of optuna intermediate values plot. 
Args: study-name 
"""

import optuna 
import dill 
import argparse 

parser = argparse.ArgumentParser() 
parser.add_argument("--study-name", type=str, required=True)
args = parser.parse_args() 

SQL_ADDRESS = f"mysql://milatest2:passwordyadayada@35.194.57.226/maale"
study = optuna.load_study(storage=SQL_ADDRESS, 
                          study_name=args.study_name)

with open('int_values.pkl','wb') as fd: 
    dill.dump(optuna.visualization.plot_intermediate_values(study), fd)

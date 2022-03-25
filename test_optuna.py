import optuna
from multiprocessing import Process


def objective(trial):
    x = trial.suggest_float("x", -1, 1)
    return x ** 2


study = optuna.create_study()
optim = lambda: study.optimize(objective, n_trials=3)

p = Process(target=optim)
p.start()
p.join()

trials = study.trials
print(len(study.trials))
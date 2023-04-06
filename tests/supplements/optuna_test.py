import pickle
import optuna
import random
import numpy as np
import sys
import os


def func(x1, x2):
    y = (x1**2) - (4.0 * x1) + (x2**2) - x2 - (x1 * x2)
    return y


def test_optuna():
    print(sys.float_info)
    randseed = 42
    random.seed(randseed)
    np.random.seed(seed=randseed)
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=randseed),
        # sampler=optuna.samplers.RandomSampler(seed=randseed),
        # sampler=optuna.samplers.CmaEsSampler(seed=randseed),
        study_name="cmaes",
        direction="minimize"
    )
    distributions = {}
    distributions["x1"] = optuna.distributions.FloatDistribution(0, 5)
    distributions["x2"] = optuna.distributions.FloatDistribution(0, 5)

    # distributions["x1"] = optuna.distributions.UniformDistribution(0, 5)
    # distributions["x2"] = optuna.distributions.UniformDistribution(0, 5)

    script_path = os.path.dirname(os.path.abspath(__file__))
    with open(f"{script_path}/study.pkl", "rb") as f:
        local_study = pickle.load(f)

    for i in range(20):
        print(i)
        t = study.ask(distributions)
        assert t.params["x1"] == local_study.trials[i].params["x1"]
        assert t.params["x2"] == local_study.trials[i].params["x2"]
        objective = func(t.params["x1"], t.params["x2"])
        study.tell(t, objective)
        # print(t.params["x1"], t.params["x2"], objective)
        assert objective == local_study.trials[i].values[0]

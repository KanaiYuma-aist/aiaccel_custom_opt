from tests.base_test import BaseTest

import optuna
import random
import numpy as np


def f(x1, x2):
    y = (x1**2) - (4.0 * x1) + (x2**2) - x2 - (x1 * x2)
    return y


def test_optuna():
    local_point = [
        [1.8727005942368125, 4.75357153204958],
        [3.6599697090570253, 2.993292420985183],
        [0.7800932022121826, 0.7799726016810132],
        [0.2904180608409973, 4.330880728874676],
        [3.005575058716044, 3.540362888980227],
        [0.10292247147901223, 4.8495492608099715],
        [4.162213204002109, 1.0616955533913808],
        [0.9091248360355031, 0.9170225492671691],
        [1.5212112147976886, 2.6237821581611893],
        [2.1597250932105787, 1.4561457009902097],
        [4.725682721151934, 1.8023969329576304],
        [2.9713510998773174, 0.08621811288760961],
        [3.575848060983975, 2.5950671281936977],
        [2.2961993850681535, 1.9622644605131616],
        [2.7377912053520186, 2.1096206119349574],
        [3.0387608534941326, 2.2559943632610056],
        [2.65887938614647, 2.171781358782056],
        [3.1973855499010355, 3.097669715667996],
        [2.5579890515518953, 2.150422807037473],
        [1.725889760436546, 1.469927064314376],

    ]
    local_result = [4.9570596841538315,
                    - 6.233353059626719,
                    - 3.2918940715021465,
                    12.090551582548596,
                    - 4.635838702522881,
                    17.76835432595009,
                    - 3.678335416695216,
                    - 3.7197715447915174,
                    - 3.5016374880215793,
                    - 6.455147702876077,
                    - 3.641971975371461,
                    - 3.39144587522278,
                    - 6.656962375529945,
                    - 6.529799019376814,
                    - 6.890486380924219,
                    - 6.942887042224681,
                    - 6.795529729022854,
                    - 5.972824280432446,
                    - 6.7155107729681145,
                    - 5.771037135365695
                    ]

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

    for i in range(20):
        t = study.ask(distributions)
        assert t.params["x1"] == local_point[i][0]
        assert t.params["x2"] == local_point[i][1]
        objective = f(float(t.params["x1"]), float(t.params["x2"]))
        study.tell(t, objective)
        # print(t.params["x1"], t.params["x2"], objective)
        assert objective == local_result[i]

# Test grid search when given an irregular grid.

# import subprocess
# from pathlib import Path

# import aiaccel
# from aiaccel.config import Config
# from aiaccel.storage.storage import Storage

from tests.base_test import BaseTest

import optuna
import random
import numpy as np


# class OptunaTest(BaseTest):
#     search_algorithm = None

#     def test_run(self, work_dir, create_tmp_config):
#         local_result = [4.9570596841538315,
#                         - 6.233353059626719,
#                         - 3.2918940715021465,
#                         12.090551582548596,
#                         - 4.635838702522881,
#                         17.76835432595009,
#                         - 3.678335416695216,
#                         - 3.7197715447915174,
#                         - 3.5016374880215793,
#                         - 6.455147702876077,
#                         - 3.641971975371461,
#                         - 3.39144587522278,
#                         - 6.656962375529945,
#                         - 6.529799019376814,
#                         - 6.890486380924219,
#                         - 6.942887042224681,
#                         - 6.795529729022854,
#                         - 5.972824280432446,
#                         - 6.7155107729681145,
#                         - 5.771037135365695
#                         ]

#         randseed = 42
#         random.seed(randseed)
#         np.random.seed(seed=randseed)
#         study = optuna.create_study(
#             sampler=optuna.samplers.TPESampler(seed=randseed),
#             # sampler=optuna.samplers.RandomSampler(seed=randseed),
#             # sampler=optuna.samplers.CmaEsSampler(seed=randseed),
#             study_name="cmaes",
#             direction="minimize"
#         )
#         distributions = {}
#         distributions["x1"] = optuna.distributions.FloatDistribution(0, 5)
#         distributions["x2"] = optuna.distributions.FloatDistribution(0, 5)

#         # distributions["x1"] = optuna.distributions.UniformDistribution(0, 5)
#         # distributions["x2"] = optuna.distributions.UniformDistribution(0, 5)

#         for i in range(20):
#             t = study.ask(distributions)
#             objective = f(t.params["x1"], t.params["x2"])
#             study.tell(t, objective)
#             # print(t.params["x1"], t.params["x2"], objective)
#             assert objective == local_result[i]
#     #     test_data_dir = Path(__file__).resolve().parent.joinpath('additional_grid_test', 'test_data')
#     #     config_file = test_data_dir.joinpath('config_{}.yaml'.format(self.search_algorithm))
#     #     config_file = create_tmp_config(config_file)
#     #     self.config = Config(config_file)
#     #     python_file = test_data_dir.joinpath('user.py')

#     #     with self.create_main(python_file):
#     #         storage = Storage(ws=Path(self.config.workspace.get()))
#     #         subprocess.Popen(['aiaccel-start', '--config', str(config_file), '--clean']
#     #                          ).wait(timeout=self.config.batch_job_timeout.get())
#     #     self.evaluate(work_dir, storage)

#     # def evaluate(self, work_dir, storage):
#     #     running = storage.get_num_running()
#     #     ready = storage.get_num_ready()
#     #     finished = storage.get_num_finished()
#     #     assert finished == self.config.trial_number.get()
#     #     assert ready == 0
#     #     assert running == 0
#     #     final_result = work_dir.joinpath(aiaccel.dict_result, aiaccel.file_final_result)
#     #     assert final_result.exists()


# class TestOptunaTPE(OptunaTest):
#     pass


def f(x1, x2):
    y = (x1**2) - (4.0 * x1) + (x2**2) - x2 - (x1 * x2)
    return y


def test_optuna():
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
        objective = f(t.params["x1"], t.params["x2"])
        study.tell(t, objective)
        # print(t.params["x1"], t.params["x2"], objective)
        assert objective == local_result[i]

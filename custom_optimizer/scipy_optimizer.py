from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer

from scipy.optimize import minimize
import sys
from multiprocessing import Process


class ScipyOptimizer(AbstractOptimizer):
    def __init__(self, options: dict) -> None:
        """Initial method of TpeOptimizer.

        Args:
            options (dict): A file name of a configuration.
        """
        super().__init__(options)
        self.study_name = "scipy_neldermead"
        self.resume_objectives = None

        # TODO: get initial
        self.create_numpy_random_generator()
        initial_params = [param['value'] for param in self.params.sample(initial=True, rng=self._rng)]

        self.scipy_process = Process(
            target=minimize,
            args=(self.objective_function, initial_params,),
            kwargs={
                # "method": "Nelder-Mead",  # 1.6900694629777868
                "method": "CG",  # 0.021162316783674685
                # "method": "BFGS", # 0.021162316778463867
                # "method": "COBYLA", # 0.031714451215396365
                # "method": "trust-constr", # 0.021162316778464106
                # "method": "L-BFGS-B", # 0.021162316778463867
                # "method": "TNC", # 0.08367154471697916
                # "method": "SLSQP", # 0.021162316778463867

                # "method": "dogleg", #
                # "method": "trust-ncg", #
                # "method": "trust-exact", #
                # "method": "trust-krylov", #
                "tol": 0.0,
                "options": {"maxiter": self.config.trial_number.get()}
            }
        )

        # warning num_node > 1

    def objective_function(self, X):
        trial_id = self.trial_id.get()

        # finish process
        if trial_id >= self.config.trial_number.get():
            sys.exit()

        # for resume
        if len(self.resume_objectives) > 0:
            return self.resume_objectives.pop(0)

        new_params = []

        for i, param in enumerate(self.params.get_parameter_list()):
            new_param = {
                'parameter_name': param.name,
                'type': param.type,
                'value': X[i]
            }
            new_params.append(new_param)
        self.register_new_parameters(new_params)

        self.trial_id.increment()
        self._serialize(self.trial_id.integer)

        objective = self.storage.result.get_any_trial_objective(trial_id)
        # wait
        while objective is None:
            objective = self.storage.result.get_any_trial_objective(trial_id)

        return objective

    def pre_process(self) -> None:
        """Pre-Procedure before executing optimize processes.
        """
        self.resume()

        self.parameter_list = self.params.get_parameter_list()
        self.resume_objectives = self.storage.result.get_objectives()
        self.scipy_process.start()

    def post_process(self) -> None:
        """Post-procedure after executed processes.
        """
        super().post_process()
        self.scipy_process.kill()

    def inner_loop_main_process(self) -> bool:
        if self.check_finished():
            return False

        if not self.scipy_process.is_alive():
            return False

        return True

    def __getstate__(self):
        obj = super().__getstate__()
        del obj['scipy_process']
        return obj

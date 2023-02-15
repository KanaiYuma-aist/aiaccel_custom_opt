from __future__ import annotations
from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer
from typing import Optional, Union

from scipy.optimize import minimize
import threading
import queue


class ScipyOptimizer(AbstractOptimizer):
    def __init__(self, options: dict) -> None:
        """Initial method of TpeOptimizer.

        Args:
            options (dict): A file name of a configuration.
        """
        super().__init__(options)
        self.study_name = "scipy_neldermead"
        self.resume_objectives = None
        self.parameter_list = self.params.get_parameter_list()

        self.trial_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.running_trial_id = None

        # optimize method in scipy.minimize
        self.method = "Nelder-Mead"
        # self.method = "CG"
        # self.method = "BFGS"
        # self.method = "COBYLA"
        # self.method = "trust-constr"
        # self.method = "L-BFGS-B"
        # self.method = "TNC"
        # self.method = "SLSQP"

        # need additional option
        # self.method = "dogleg"
        # self.method = "trust-ncg"
        # self.method = "trust-exact"
        # self.method = "trust-krylov"

        self.tol = 0.0

        # TODO: warning num_node > 1

    # for scipy.optimize.minimize
    def transfer_function(self, X):
        # for resume
        if len(self.resume_objectives) > 0:
            pop_result = self.resume_objectives.pop(0)
            return pop_result

        self.trial_queue.put(X, block=True, timeout=None)

        # wait
        objective = self.result_queue.get(block=True, timeout=None)

        return objective

    def create_initial_params(self):
        resume_initial_params = self.storage.hp.get_any_trial_params(0)
        if resume_initial_params is None:
            # no resume
            return [param['value'] for param in
                    self.params.sample(initial=True, rng=self._rng)]
        else:
            # resume
            return [param.param_value for param in resume_initial_params]

    def start_scipy_optimize_thread(self):
        bounds = ([(param.lower, param.upper) for param in self.parameter_list])
        # thread of scipy.optimize
        self.scipy_thread = threading.Thread(
            target=minimize,
            args=(self.transfer_function, self.create_initial_params(),),
            kwargs={
                "method": self.method,
                "bounds": bounds,
                "tol": self.tol,
                "options": {"maxiter": self.config.trial_number.get()}
            }
        )
        self.scipy_thread.daemon = True
        self.scipy_thread.start()

    def pre_process(self) -> None:
        """Pre-Procedure before executing optimize processes.
        """
        super().pre_process()

        self.resume_objectives = self.storage.result.get_objectives()
        self.start_scipy_optimize_thread()

    def post_process(self) -> None:
        """Post-procedure after executed processes.
        """
        super().post_process()
        # self.scipy_thread.join(timeout=0)  # finish thread

    def check_result(self) -> None:
        objective = self.storage.result.get_any_trial_objective(self.running_trial_id)
        if objective is not None:
            self.running_trial_id = None
            self.result_queue.put(objective, block=True, timeout=None)

    def generate_initial_parameter(
        self
    ) -> list[dict[str, Union[float, int, str]]]:
        """Generate a list of initial parameters.

        Returns:
            list[dict[str, Union[float, int, str]]]: A created list of initial
            parameters.
        """
        return self.generate_parameter()

    def generate_parameter(
        self, number: Optional[int] = 1
    ) -> Optional[list[dict[str, Union[float, int, str]]]]:
        """Generate parameters.

        Args:
            number (Optional[int]): A number of generating parameters. Defaults
                to 1.

        Returns:
            Optional[list[dict[str, Union[float, int, str]]]]: A list of
            created parameters.
        """
        self.check_result()

        if self.trial_queue.empty():
            return None
        trial = self.trial_queue.get(block=True, timeout=None)

        self.running_trial_id = self.trial_id.get()
        new_params = []

        for i, param in enumerate(self.parameter_list):
            new_param = {
                'parameter_name': param.name,
                'type': param.type,
                'value': trial[i]
            }
            new_params.append(new_param)

        self.logger.info(f'running trial id: {self.running_trial_id}')

        return new_params

    def __getstate__(self):
        obj = super().__getstate__()
        del obj['scipy_thread']
        del obj['trial_queue']
        del obj['result_queue']
        del obj['running_trial_id']
        return obj

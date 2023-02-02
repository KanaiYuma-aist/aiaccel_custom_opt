from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer

from scipy.optimize import minimize
import sys
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

        # warning num_node > 1

    def objective_function(self, X):
        trial_id = self.trial_id.get()

        # finish process
        if trial_id >= self.config.trial_number.get():
            sys.exit()

        # for resume
        if len(self.resume_objectives) > 0:
            pop_result = self.resume_objectives.pop(0)
            return pop_result

        self.running_trial_id = trial_id
        new_params = []

        for i, param in enumerate(self.parameter_list):
            new_param = {
                'parameter_name': param.name,
                'type': param.type,
                'value': X[i]
            }
            new_params.append(new_param)
        self.register_new_parameters(new_params)

        self.trial_id.increment()
        self._serialize(self.trial_id.integer)

        # wait
        objective = self.result_queue.get(block=True, timeout=None)

        return objective

    def pre_process(self) -> None:
        """Pre-Procedure before executing optimize processes.
        """
        super().pre_process()

        self.parameter_list = self.params.get_parameter_list()
        self.resume_objectives = self.storage.result.get_objectives()
        bounds = ([(param.lower, param.upper) for param in self.parameter_list])

        resume_initial_params = self.storage.hp.get_any_trial_params(0)
        if resume_initial_params is None:
            # no resume
            initial_params = [param['value'] for param in
                              self.params.sample(initial=True, rng=self._rng)]
        else:
            # resume
            initial_params = [param.param_value for param in resume_initial_params]

        # thread of scipy.optimize
        self.scipy_thread = threading.Thread(
            target=minimize,
            args=(self.objective_function, initial_params,),
            kwargs={
                "method": self.method,
                "bounds": bounds,
                "tol": 0.0,
                "options": {"maxiter": self.config.trial_number.get()}
            }
        )
        self.scipy_thread.daemon = True

        self.scipy_thread.start()

    def post_process(self) -> None:
        """Post-procedure after executed processes.
        """
        super().post_process()
        # finish thread
        self.scipy_thread.join(timeout=0)

    def inner_loop_main_process(self) -> bool:
        # check result
        objective = self.storage.result.get_any_trial_objective(self.running_trial_id)
        if objective is not None:
            self.running_trial_id = None
            self.result_queue.put(objective, block=True, timeout=None)

        if self.check_finished():
            return False

        if not self.scipy_thread.is_alive():
            return False

        return True

    def __getstate__(self):
        obj = super().__getstate__()
        del obj['scipy_thread']
        del obj['result_queue']
        del obj['running_trial_id']
        return obj

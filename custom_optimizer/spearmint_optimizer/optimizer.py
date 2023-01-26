
import aiaccel.parameter
from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer
from typing import Optional
import numpy as np

# spearmint
import sys
import os

try:
    import simplejson as json
except ImportError:
    import json
from collections import OrderedDict
from spearmint.utils.database.mongodb import MongoDB
from spearmint.utils.parsing import parse_db_address
from spearmint.main import get_suggestion
from spearmint.main import save_job
from spearmint.main import save_hypers
from spearmint.main import load_jobs
from spearmint.main import load_hypers

from spearmint.choosers.default_chooser import init as chooser_init
from spearmint.schedulers.local import init as scheduler_init
from spearmint.resources.resource import Resource

# SpearmintOptimizer を動作させるには、下記セットアップが必要です。

# Download and install MongoDB: https://www.mongodb.org/
# Install the spearmint package using pip: pip install -e aiaccel/original_optimizer/spearmint_optimizer/Spearmint
# export PYTHONPATH="aiaccel/original_optimizer/spearmint_optimizer/Spearmint:aiaccel/original_optimizer:"
# Install pymongo -> version 3.13.0
# Start up a MongoDB daemon instance: mongod --fork --logpath <path/to/logfile\> --dbpath <path/to/dbfolder\>

# 参考: https://github.com/redst4r/Spearmint/tree/python3


class SpearmintOptimizer(AbstractOptimizer):
    def __init__(self, options: dict) -> None:
        """Initial method of TpeOptimizer.

        Args:
            options (dict): A file name of a configuration.
        """
        super().__init__(options)
        self.parameter_pool = {}
        self.parameter_list = []
        self.study_name = "Spearmint"
        self.distributions = None
        self.job_pool = {}
        self.randseed = self.config.randseed.get()
        self.initial_count = 0

        # spearmint setup
        self.spearmint_options, self.expt_dir = self.get_options(self.params)

        self.resource_name = 'Main'
        resources = {self.resource_name: self.resource_factory(self.resource_name, ['main'], self.spearmint_options)}

        self.resource = resources[self.resource_name]
        self.experiment_name = self.spearmint_options.get("experiment-name", 'unnamed-experiment')

        # Connect to the database
        db_address = self.spearmint_options['database']['address']
        self.logger.info('Using database at %s.\n' % db_address)
        self.mongo_db = MongoDB(database_address=db_address)

        # reset database
        self.mongo_db.remove(self.experiment_name, 'jobs')
        self.mongo_db.remove(self.experiment_name, 'hypers')

    def get_options(self, aiaccel_parameters: aiaccel.parameter.HyperParameterConfiguration):

        config_dir = os.path.dirname(__file__) + "/Spearmint/examples/simple/"
        config_file = "config.json"

        # Read in the config file
        expt_dir = os.path.realpath(os.path.expanduser(config_dir))
        if not os.path.isdir(expt_dir):
            raise Exception("Cannot find directory %s" % expt_dir)
        expt_file = os.path.join(expt_dir, config_file)

        try:
            with open(expt_file, 'r') as f:
                options = json.load(f, object_pairs_hook=OrderedDict)
        except Exception:
            raise Exception(
                "config.json did not load properly. Perhaps a spurious comma?")

        variables = OrderedDict()
        for p in aiaccel_parameters.get_parameter_list():
            variables[p.name] = OrderedDict([('type', p.type), ('size', 1), ('min', p.lower), ('max', p.upper)])
        options['variables'] = variables

        # Set sensible defaults for options
        options['chooser'] = options.get('chooser', 'default_chooser')
        if 'tasks' not in options:
            options['tasks'] = {'main': {'type': 'OBJECTIVE',
                                         'likelihood': options.get('likelihood', 'GAUSSIAN')}}

        # Set DB address
        db_address = parse_db_address(options)
        if 'database' not in options:
            options['database'] = {'name': 'spearmint', 'address': db_address}
        else:
            options['database']['address'] = db_address

        if not os.path.exists(expt_dir):
            self.logger.warning("Cannot find experiment directory '%s'. "
                                "Aborting.\n" % (expt_dir))
            sys.exit(-1)

        return options, expt_dir

    def resource_factory(self, resource_name, task_names, config):
        """return a resource object constructed from the resource name, task names, and config dict"""
        scheduler_class = "local"
        scheduler_object = scheduler_init(config)

        max_concurrent = config.get('max-concurrent', 1)
        max_finished_jobs = config.get('max-finished-jobs', np.inf)

        return Resource(resource_name, task_names, scheduler_object,
                        scheduler_class, max_concurrent, max_finished_jobs)

    def pre_process(self) -> None:
        """Pre-Procedure before executing optimize processes.
        """
        super().pre_process()

        np.random.seed(self.seed)
        self.parameter_list = self.params.get_parameter_list()

    def post_process(self) -> None:
        """Post-procedure after executed processes.
        """
        self.check_result()
        super().post_process()

    def my_round(self, val, digit=0):
        p = 10 ** digit
        return (val * p * 2 + 1) // 2 / p

    def check_result(self) -> None:
        """Check the result files and add it to sampler object.

        Returns:
            None
        """

        del_keys = []
        for trial_id, param in self.parameter_pool.items():
            objective = self.storage.result.get_any_trial_objective(trial_id)
            if objective is not None:
                job = self.job_pool[trial_id]
                job['values'] = {'main': objective}
                # job['values'] = {'main': self.my_round(objective, 6)}
                job['status'] = 'complete'
                save_job(job, self.mongo_db, self.experiment_name)
                del_keys.append(trial_id)

        for key in del_keys:
            self.parameter_pool.pop(key)
            self.logger.info(f'trial_id {key} is deleted from parameter_pool')

        self.logger.debug(f'current pool {[k for k, v in self.parameter_pool.items()]}')

    def generate_parameter(self, number: Optional[int] = 1) -> None:
        """Generate parameters.

        Args:
            number (Optional[int]): A number of generating parameters.
        """
        self.check_result()
        self.logger.debug(f'number: {number}, pool: {len(self.parameter_pool)} losses')

        if (
            (len(self.parameter_pool) >= 1)
        ):
            return None

        if len(self.parameter_pool) >= self.config.num_node.get():
            return None

        new_params = []
        suggested_job = get_suggestion(
            chooser_init(self.spearmint_options), self.resource.tasks, self.mongo_db, self.expt_dir,
            self.spearmint_options, self.resource_name
        )

        for param in self.params.get_parameter_list():
            new_param = {
                'parameter_name': param.name,
                'type': param.type,
                'value': suggested_job['params'][param.name]['values'][0]
            }
            new_params.append(new_param)

        trial_id = self.trial_id.get()
        self.parameter_pool[trial_id] = new_params
        self.job_pool[trial_id] = suggested_job
        self.logger.info(f'newly added name: {trial_id} to parameter_pool')

        # for serialize mongo DB
        self.hyper = load_hypers(self.mongo_db, self.experiment_name)
        self.jobs = load_jobs(self.mongo_db, self.experiment_name)

        return new_params

    def generate_initial_parameter(self):
        initial_parameter = super().generate_initial_parameter()

        enqueue_trial = {}
        for hp in self.params.hps.values():
            if hp.initial is not None:
                enqueue_trial[hp.name] = hp.initial

        # all hp.initial is None
        if len(enqueue_trial) == 0:
            return self.generate_parameter()

        spearmint_params = {}
        for param in initial_parameter:
            spearmint_params[param['parameter_name']] = {
                'type': param['type'].lower(),
                'values': [param['value']]
            }

        initial_job = {
            'id': 1,
            'params': spearmint_params
        }

        trial_id = self.trial_id.get()
        self.parameter_pool[trial_id] = initial_parameter
        self.job_pool[trial_id] = initial_job
        self.logger.info(f'newly added name: {trial_id} to parameter_pool')

        # for serialize mongo DB
        self.hyper = load_hypers(self.mongo_db, self.experiment_name)
        self.jobs = load_jobs(self.mongo_db, self.experiment_name)

        return initial_parameter

    def _deserialize(self, trial_id: int) -> None:
        super()._deserialize(trial_id)

        # deserialize mongo DB
        save_hypers(self.hyper, self.mongo_db, self.experiment_name)
        for job in self.jobs:
            save_job(job, self.mongo_db, self.experiment_name)

    def __getstate__(self):
        obj = super().__getstate__()
        del obj['mongo_db']
        return obj

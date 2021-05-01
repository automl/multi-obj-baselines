from copy import deepcopy

from ax import Data, GeneratorRun, Arm
import pandas as pd

from hpbandster.core.worker import Worker
import ConfigSpace as CS
import numpy as np



class MOBOHBWorker(Worker):
    def __init__(self, experiment, search_space, eval_function, seed=42, **kwargs):
        super().__init__(**kwargs)

        self.experiment = experiment
        self.eval_function = eval_function
        self.search_space = search_space
        self.seed = seed

    def tchebycheff_norm(self, cost, rho=0.05):
        w = np.random.random_sample(2)
        w /= np.sum(w)

        w_f = w * cost
        max_k = np.max(w_f)
        rho_sum_wf = rho * np.sum(w_f)
        return max_k + rho_sum_wf

    def compute(self, config_id:int, config: CS.Configuration, budget:float, working_directory:str, *args, **kwargs) -> dict:


        params = deepcopy(config)
        params['budget'] = int(budget)

        params['n_conv_0'] = params['n_conv_0'] if 'n_conv_0' in params else 16
        params['n_conv_1'] = params['n_conv_1'] if 'n_conv_1' in params else 16
        params['n_conv_2'] = params['n_conv_2'] if 'n_conv_2' in params else 16

        params['n_fc_0'] = params['n_fc_0'] if 'n_fc_0' in params else 16
        params['n_fc_1'] = params['n_fc_1'] if 'n_fc_1' in params else 16
        params['n_fc_2'] = params['n_fc_2'] if 'n_fc_2' in params else 16

        params['kernel_size'] = [3, 5, 7][params['kernel_size']]
        params['batch_norm'] = bool(params['batch_norm'])
        params['global_avg_pooling'] = bool(params['global_avg_pooling'])
        params['id'] = str(config_id)

        trial = self.experiment.new_trial(GeneratorRun([Arm(params, name=str(config_id))]))
        data = self.experiment.eval_trial(trial)

        acc = float(data.df[data.df['metric_name'] == 'val_acc_1']['mean'])
        len = float(data.df[data.df['metric_name'] == 'num_params']['mean'])

        return {'loss': (acc, len)}

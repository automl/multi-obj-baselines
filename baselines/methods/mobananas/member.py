import enum
import uuid
from copy import deepcopy
import numpy as np
import pandas as pd
from typing import Dict, Optional
from ax import Experiment, Data, GeneratorRun, Arm
from scipy.stats import truncnorm


class Mutation(enum.IntEnum):
    NONE = -1  # Can be used when only recombination is required
    UNIFORM = 0  # Uniform mutation
    GAUSSIAN = 1  # Gaussian mutation


class Member:
    """
    Class to handle member.
    """


    def __init__(self, search_space,
                 mutation: Mutation,
                 budget = 25,
                 experiment: Experiment = None,
                 x_coordinate: Optional[Dict] = None
                 ) -> None:
        """
        Init
        :param search_space: search_space of the given problem
        :param x_coordinate: Initial coordinate of the member
        :param target_function: The target function that determines the fitness value
        :param mutation: hyperparameter that determines which mutation type use
        :budget number of epochs
        :param experiment: axi experiment to run
        """
        self._space = search_space
        self._budget = budget
        self._id = uuid.uuid4()
        self._x = search_space.sample_configuration().get_dictionary() if not x_coordinate else x_coordinate
        self._age = 0
        self._mutation = mutation
        self._x_changed = True
        self._fit = None
        self._experiment = experiment
        self._num_evals = 0


    @property 
    def fitness(self):
        if self._x_changed:  # Only if budget or architecture has changed we need to evaluate the fitness.
            self._x_changed = False


            params = deepcopy(self._x)
            params['budget'] = int(self._budget)

            params['n_conv_0'] = params['n_conv_0'] if 'n_conv_0' in params else 16
            params['n_conv_1'] = params['n_conv_1'] if 'n_conv_1' in params else 16
            params['n_conv_2'] = params['n_conv_2'] if 'n_conv_2' in params else 16

            params['n_fc_0'] = params['n_fc_0'] if 'n_fc_0' in params else 16
            params['n_fc_1'] = params['n_fc_1'] if 'n_fc_1' in params else 16
            params['n_fc_2'] = params['n_fc_2'] if 'n_fc_2' in params else 16

            params['batch_norm'] = bool(params['batch_norm'])
            params['global_avg_pooling'] = bool(params['global_avg_pooling'])

            trial_name = '{}-{}'.format(self._id, self._num_evals)
            params['id'] = trial_name


            trial = self._experiment.new_trial(GeneratorRun([Arm(params, name=trial_name)]))
            data = self._experiment.eval_trial(trial)
            self._num_evals += 1

            acc = float(data.df[data.df['metric_name'] == 'val_acc_1']['mean'])
            len = float(data.df[data.df['metric_name'] == 'num_params']['mean'])

            self._fit =[acc, len]

        return self._fit  # evaluate or return save variable

    @property
    def x_coordinate(self):
        return self._x

    @x_coordinate.setter
    def x_coordinate(self, value):
        self._x_changed = True
        self._x = value

    @property
    def budget(self):
        return self._budget

    @budget.setter
    def budget(self, value):
        self._x_changed = True
        self._budget = value

    @property
    def id(self):
        return self._id


    def get_truncated_normal(self,mean=0, sd=1, low=0, upp=10):
        return truncnorm(
            (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    def return_train_data(self):

        params = deepcopy(self.x_coordinate)
        hyperparameter_dict = self._space.get_hyperparameters_dict()

        params['n_conv_0'] = params['n_conv_0'] if 'n_conv_0' in params else 0
        params['n_conv_1'] = params['n_conv_1'] if 'n_conv_1' in params else 0
        params['n_conv_2'] = params['n_conv_2'] if 'n_conv_2' in params else 0

        params['n_fc_0'] = params['n_fc_0'] if 'n_fc_0' in params else 0
        params['n_fc_1'] = params['n_fc_1'] if 'n_fc_1' in params else 0
        params['n_fc_2'] = params['n_fc_2'] if 'n_fc_2' in params else 0

        train_data = []
        for key in params.keys():

            if params[key] == True:
                param = 1
            elif params[key] == False:
                param = 0
            else:

                try:
                    param = params[key] / hyperparameter_dict[key].upper
                except:
                    param = params[key]/ (np.sort(hyperparameter_dict[key].choices)[-1])

            train_data.append(param)

        return train_data

    def mutate(self):
        """
        Mutation to create a new offspring
        :return: new member who is based on this member
        """
        new_x = self.x_coordinate.copy()
        hyperparameter_dict = self._space.get_hyperparameters_dict()

        if self._mutation == Mutation.GAUSSIAN:
            keys = np.random.choice(list(new_x.keys()), 3, replace=False)
            for k in keys:

                if self._space.is_mutable_hyperparameter(str(k)):

                    try:

                        mean = new_x[k]
                        upper = hyperparameter_dict[k].upper
                        lower = hyperparameter_dict[k].lower
                        sd = (upper - lower) / 3
                        X = self.get_truncated_normal(mean=mean, sd=sd, low=lower, upp=upper)


                        if str(k) == "lr_init":
                            new_x[k] = X.rvs()
                        else:
                            new_x[k] = int(X.rvs())

                    except:

                        new_x[k] = self._space.sample_hyperparameter(str(k))


        elif self._mutation != Mutation.NONE:
            # We won't consider any other mutation types
            raise NotImplementedError

        child = Member(self._space, self._mutation,self.budget,
                       self._experiment, new_x)

        self._age += 1
        return child



import os
import numpy as np
import torch
import sys
from .member import Member
from .neural_predictor import Neural_Predictor
import numpy as np
from .member import Member
from .member import Mutation
from baselines import nDS_index, crowdingDist
import math


class BANANAS:
    """
    Class to group ensamble of NN
    """

    def __init__(self,neural_predictor, experiment, search_space,
                 initial_samples, num_arch, max_budget, min_budget, eta, select_models, function_evaluations, mutation_type = Mutation.GAUSSIAN):

        self.num_arch = num_arch
        self.num_function_evaluations = function_evaluations
        self.search_space = search_space
        self.experiment = experiment
        self.initial_samples = initial_samples
        self.neural_predictor = neural_predictor
        self.max_budget = max_budget
        self.min_budget = min_budget
        np.random.seed(0)
        self.select = select_models
        self.eta = eta

        self.architecture_list = [Member(self.search_space, mutation_type, self.max_budget,
                                  experiment=self.experiment) for _ in range(self.initial_samples)]

        [Member.fitness for Member in self.architecture_list]


        self.iterations = (self.num_function_evaluations - self.initial_samples)// (self.select//self.eta)

    def steps(self):

        it = 0

        while it < self.iterations:

            it = it + 1
            train_data = [member.return_train_data() for member in self.architecture_list]
            y_train_data = [member.fitness for member in self.architecture_list]
            train_data = [[train_data[i], [-y_train_data[i][0]/10, y_train_data[i][1]]] for i in range(len(train_data))]
            self.neural_predictor.train_models(train_data)

            # choose best configs
            best_configs = self._select_best_architectures_mo(self.num_arch)
            mutated_configs = [member.mutate() for member in best_configs]
            test_data = [member.return_train_data() for member in mutated_configs]
            chosen_models = self.neural_predictor.choose_models(mutated_configs,test_data, self.select)
            chosen_models = self.successive_halving(chosen_models, self.min_budget, self.max_budget, eta=3)

            self.architecture_list.extend(chosen_models)

        return


    def get_budgets(self, min_budget, max_budget,eta = 3):

        budgets = []
        b = max_budget
        while b > min_budget:
            budgets.append(b)
            b = math.ceil(b / eta)

        return budgets


    def successive_halving(self, members, min_budget, max_budget, eta = 3):

        budgets = self.get_budgets(min_budget, max_budget, eta)
        print(budgets)
        print(members)
        for b in budgets[::-1]:
            members[0].budget = b
            for member in members:
                member.budget = b

            fit = [member.fitness for member in members]
            members = self.sort_architectures(members)
            members = members[0:len(members)//eta]

        return members

    def sort_architectures(self, architectures):

        index_list = np.array(list(range(len(architectures))))
        fitness = [member.fitness for member in architectures]
        a, index_return_list = nDS_index(np.array(fitness), index_list)
        b, sort_index = crowdingDist(a, index_return_list)

        sorted = []

        for x in sort_index:
            sorted.extend(x)

        architectures = self.sort_pop(architectures, sorted)

        return architectures


    def sort_pop(self,list1, list2):

        z = [list1[int(m)] for m in list2]

        return z

    def _select_best_architectures_mo(self, num_arch):

            index_list = np.array(list(range(len(self.architecture_list))))
            fitness = [ member.fitness for member in self.architecture_list]
            a, index_return_list = nDS_index(np.array(fitness), index_list)
            b, sort_index = crowdingDist(a, index_return_list)

            sorted = []

            for x in sort_index:
                sorted.extend(x)

            self.architecture_list = self.sort_pop(self.architecture_list, sorted)


            return self.architecture_list[0:num_arch]




def get_MOSHBANANAS(experiment, search_space,
                    initial_samples=20, select_models = 10, num_arch = 20,
                    min_budget = 5, max_budget = 25, function_evaluations = 100,eta = 3):

    # save models and dict so it can be picked up later on

    neural_predictor = Neural_Predictor(num_epochs = 80, num_ensamble_nets = 5)
    banana = BANANAS(neural_predictor, experiment, search_space, initial_samples, num_arch, max_budget,min_budget, eta,  select_models, function_evaluations)
    banana.steps()

    return



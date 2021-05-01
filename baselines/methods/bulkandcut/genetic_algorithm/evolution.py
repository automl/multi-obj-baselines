
import os
import csv
import math
from datetime import datetime

import torch
import numpy as np
from ax import Experiment, GeneratorRun, Arm, Data

from baselines.methods.bulkandcut.model.BNCmodel import BNCmodel
from baselines.methods.bulkandcut.model.blind_model import BlindModel
from baselines.methods.bulkandcut.genetic_algorithm.individual import Individual
from baselines.methods.bulkandcut.bayesian_optimization.optimizer_one_two import OptimizerOneTwo
from baselines.methods.bulkandcut.bayesian_optimization.optimizer_three import OptimizerThree
from baselines.methods.bulkandcut.plot.learning_curve import plot_learning_curves
from baselines.methods.bulkandcut import rng, device


class Evolution:
    """Evolutinary algorithm to jointly optimize validation accuracy and number
    of trainable parameters of a convolutional neural network classifier.

    Parameters
    ----------
    input_shape : tuple
        Image shape in the format (n_channels, height, width)
    n_classes : int
        Number of classes
    work_directory : str
        Path where information about the tested models should be stored
    train_dataset : torch.utils.data.Dataset
        Training dataset
    valid_dataset : torch.utils.data.Dataset
        Validation dataset
    debugging : bool, optional
        If True, the model will be validated after each epoch and learning curves
        will be plotted. If False, models are validaded just after they have been fully
        trainned. The time needed to validate models and plot curves will be deducte
        from the time_budget. By default False.
    """
    def __init__(self,
                 experiment: Experiment,
                 input_shape: tuple,
                 n_classes: int,
                 work_directory: str,
                 train_dataset: "torch.utils.data.Dataset",
                 valid_dataset: "torch.utils.data.Dataset",
                 test_dataset: "torch.utils.data.Dataset",
                 debugging: bool = False
                 ):
        self.experiment = experiment
        # Just variable initializations
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.work_directory = work_directory
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.debugging = debugging

        self.population = []
        self.max_num_epochs = 25  # Project constraint
        self.slimmdown_epochs = int(round(self.max_num_epochs / 3.))

        self.optm_onetwo = OptimizerOneTwo(log_dir=work_directory)
        self.optm_three = OptimizerThree(log_dir=work_directory)

    def run(self, time_budget: float, budget_split: tuple = (.35, .40, .25)):
        """Run evolutionary algorithm for a given time (budget).

        Parameters
        ----------
        time_budget : float
            Total running time in seconds. If in debug modus, time need to validate models and plot
            curves will consume part of the time_budget.
        budget_split : tuple, optional
            Specifies how the total budge will be distributed among the three evolutionary
             phases (initialization, bulk-up and slim-down), by default [.40, .35, .25]

        Raises
        ------
        ValueError
            If len(budget_split) != 3 or the sum of its values is different of 1.
        """

        if len(budget_split) != 3 or np.sum(budget_split) != 1.:
            raise ValueError("Bad budget split")

        self._create_work_directory()

        # Phase 1: Initiated population
        print("Starting phase 1: Initiate population")
        init_pop_budget = budget_split[0] * time_budget
        init_pop_begin = datetime.now()
        # self._train_blind_individual(super_stupid=True)
        # self._train_blind_individual(super_stupid=False)
        while True:
            remaining = init_pop_budget - (datetime.now() - init_pop_begin).seconds
            if remaining < 0:
                break
            print(f"Still {remaining / 60.:.1f} minutes left for the initial phase")
            self._train_naive_individual()


        # Phase 2: Bulk-up
        print("Starting phase 2: Bulk-up")
        bulkup_budget = budget_split[1] * time_budget
        bulkup_begin = datetime.now()
        while True:
            remaining = bulkup_budget - (datetime.now() - bulkup_begin).seconds
            if remaining < 0:
                break
            print(f"Still {remaining / 60.:.1f} minutes left for the bulk-up phase")
            to_bulk = self._select_individual_to_reproduce(transformation="bulk-up")
            self._train_offspring(parent_id=to_bulk, transformation="bulk-up")

        # Phase 3: Slim-down
        print("Starting phase 3: Slim-down")
        slimdown_budget = budget_split[2] * time_budget
        slimdown_begin = datetime.now()
        while True:
            remaining = slimdown_budget - (datetime.now() - slimdown_begin).seconds
            if remaining < 0:
                break
            print(f"Still {remaining / 60.:.1f} minutes left for the slim-down phase")
            to_cut = self._select_individual_to_reproduce(transformation="slim-down")
            self._train_offspring(parent_id=to_cut, transformation="slim-down")

    @property
    def pop_size(self):
        return len(self.population)

    def save_csv(self):
        file_path = os.path.join(self.work_directory, "population_summary.csv")
        with open(file_path, 'w', newline='') as csvfile:
            fieldnames = self.population[0].to_dict().keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for indv in self.population:
                writer.writerow(indv.to_dict())

    def _create_work_directory(self):
        if os.path.exists(self.work_directory):
            raise Exception(f"Directory exists: {self.work_directory}")
        os.makedirs(self.work_directory)
        os.mkdir(os.path.join(self.work_directory, "models"))

    def _get_model_path(self, indv_id: int):
        return os.path.join(
            self.work_directory,
            "models",
            str(indv_id).rjust(4, "0") + ".pt",
        )

    def _train_blind_individual(self, super_stupid: bool):
        indv_id = self.pop_size
        new_model = BlindModel(n_classes=self.n_classes, super_stupid=super_stupid).to(device)
        path_to_model = self._get_model_path(indv_id=indv_id)
        print("Training model", indv_id, "(blind model)")
        learning_curves = new_model.start_training(
            train_dataset=self.train_dataset,
            valid_dataset=self.valid_dataset,
            )
        new_individual = Individual(
            indv_id=indv_id,
            path_to_model=path_to_model,
            summary=new_model.summary,
            depth=1,
            birth_time=new_model.creation_time,
            parent_id=-1,  # No parent
            bulk_counter=0,
            cut_counter=0,
            bulk_offsprings=0,
            cut_offsprings=0,
            optimizer_config={},
            learning_curves=learning_curves,
            n_parameters=new_model.n_parameters,
            )
        new_model.save(file_path=path_to_model)
        new_individual.save_info()
        self.population.append(new_individual)
        self.save_csv()

    def _train_naive_individual(self):
        indv_id = self.pop_size
        new_model = BNCmodel.NEW(self.experiment.search_space, input_shape=self.input_shape, n_classes=self.n_classes)
        dicd_pars = {"depth": new_model.depth, "log_npars": math.log10(new_model.n_parameters)}
        optim_config = self.optm_onetwo.next_pars(dictated_pars=dicd_pars)
        new_model.setup_optimizer(optim_config=optim_config)
        path_to_model = self._get_model_path(indv_id=indv_id)

        # Register in experiment
        trial = self.experiment.new_trial(GeneratorRun([Arm(new_model._parameters_dict)]))
        trial.mark_running()


        print("Training model", indv_id)
        learning_curves = new_model.start_training(
            n_epochs=self.max_num_epochs,
            train_dataset=self.train_dataset,
            valid_dataset=self.valid_dataset,
            test_dataset=self.test_dataset,
            return_all_learning_curvers=self.debugging,
            )
        if self.debugging:
            plot_learning_curves(
                ind_id=indv_id,
                n_pars=new_model.n_parameters,
                curves=learning_curves,
                model_path=path_to_model,
                )
        self.optm_onetwo.register_target(
            config=optim_config,
            learning_curves=learning_curves,
        )
        new_individual = Individual(
            indv_id=indv_id,
            path_to_model=path_to_model,
            summary=new_model.summary,
            depth=new_model.depth,
            birth_time=new_model.creation_time,
            parent_id=-1,  # No parent
            bulk_counter=0,
            cut_counter=0,
            bulk_offsprings=0,
            cut_offsprings=0,
            optimizer_config=optim_config,
            learning_curves=learning_curves,
            n_parameters=new_model.n_parameters,
            parameters=new_model._parameters_dict
            )
        new_model.save(file_path=path_to_model)
        new_individual.save_info()
        self.population.append(new_individual)
        self.save_csv()

        # Save results in Ax experiment
        trial.mark_completed()
        data = Data.from_evaluations({
                trial.arm.name: {
                    'num_params': (np.log10(new_model.n_parameters), 0.0),
                    'val_acc_1': (-1.0 * learning_curves["validation_accuracy"][-1], 0.0),
                    'val_acc_3': (-1.0 * learning_curves["validation_accuracy_3"][-1], 0.0),
                    'tst_acc_1': (-1.0 * learning_curves["test_accuracy"][-1], 0.0),
                    'tst_acc_3': (-1.0 * learning_curves["test_accuracy_3"][-1], 0.0),
                }
            }, trial.index)
        self.experiment.attach_data(data)

    def _train_offspring(self, parent_id: int, transformation: str):
        if transformation not in ["bulk-up", "slim-down"]:
            raise Exception("Unknown transformation")
        bulking = transformation == "bulk-up"

        parent_indv = self.population[parent_id]
        parent_indv.bulk_offsprings += (1 if bulking else 0)
        parent_indv.cut_offsprings += (0 if bulking else 1)
        parent_model = BNCmodel.LOAD(parent_indv.path_to_model)

        child_model = parent_model.bulkup() if bulking else parent_model.slimdown()
        dicd_pars = {"depth": child_model.depth, "log_npars": math.log10(child_model.n_parameters)}
        optimizer = self.optm_onetwo if bulking else self.optm_three
        optim_config = optimizer.next_pars(dictated_pars=dicd_pars)
        child_model.setup_optimizer(optim_config=optim_config)
        child_id = self.pop_size
        path_to_child_model = self._get_model_path(indv_id=child_id)

        # Register in experiment
        trial = self.experiment.new_trial(GeneratorRun([Arm(child_model._parameters_dict)]))
        trial.mark_running()

        print("Training model", child_id)
        learning_curves = child_model.start_training(
            n_epochs=self.max_num_epochs if bulking else self.slimmdown_epochs,
            teacher_model=None if bulking else parent_model,
            train_dataset=self.train_dataset,
            valid_dataset=self.valid_dataset,
            test_dataset=self.test_dataset,
            return_all_learning_curvers=self.debugging,
            )
        if self.debugging:
            plot_learning_curves(
                ind_id=child_id,
                n_pars=child_model.n_parameters,
                curves=learning_curves,
                model_path=path_to_child_model,
                parent_loss=parent_indv.post_training_loss,
                parent_accuracy=parent_indv.post_training_accuracy,
                )
        optimizer.register_target(config=optim_config, learning_curves=learning_curves)
        new_individual = Individual(
            indv_id=child_id,
            path_to_model=path_to_child_model,
            summary=child_model.summary,
            depth=child_model.depth,
            birth_time=child_model.creation_time,
            parent_id=parent_id,
            bulk_counter=parent_indv.bulk_counter + (1 if bulking else 0),
            cut_counter=parent_indv.cut_counter + (0 if bulking else 1),
            bulk_offsprings=0,
            cut_offsprings=0,
            optimizer_config=optim_config,
            learning_curves=learning_curves,
            n_parameters=child_model.n_parameters,
            parameters=child_model._parameters_dict
        )
        self.population.append(new_individual)
        child_model.save(file_path=path_to_child_model)
        new_individual.save_info()
        self.save_csv()

        # Save results in Ax experiment
        trial.mark_completed()
        data = Data.from_evaluations({
                trial.arm.name: {
                    'num_params': (np.log10(child_model.n_parameters), 0.0),
                    'val_acc_1': (-1.0 * learning_curves["validation_accuracy"][-1], 0.0),
                    'val_acc_3': (-1.0 * learning_curves["validation_accuracy_3"][-1], 0.0),
                    'tst_acc_1': (-1.0 * learning_curves["test_accuracy"][-1], 0.0),
                    'tst_acc_3': (-1.0 * learning_curves["test_accuracy_3"][-1], 0.0),
                }
            }, trial.index)
        self.experiment.attach_data(data)

    def _select_individual_to_reproduce(self, transformation: str):
        if transformation not in ["bulk-up", "slim-down"]:
            raise Exception("Unknown transformation")

        # Selection using the "Paretslon-greedy" method, a combination of epslon-greedy
        # and non-dominated sorting. With a probability epslon, it selects a random
        # individual from the Pareto front. with probability (1 - epslon) it selects a
        # random individual from the 2nd Pareto front, as determined by the non-dominated
        # sorting method.
        pareto_fronts = self._non_dominated_sorting(n_fronts=2)
        front_number = 0 if rng.random() < .85 or len(pareto_fronts[1]) == 0 else 1
        candidates = set(pareto_fronts[front_number])

        # Deal with some exclusions:
        # First, blind models are sterile. :-)
        # candidates -= set([0, 1])
        # Then exclude others depending on the transformation
        if transformation == "bulk-up":
            # Exclude individuals that are already too big:
            candidates -= set([i.indv_id for i in self.population if i.n_parameters > int(1E8)])

            # Sergio: Remove candidates that if expanded cross the search space
            candidates -= set([i.indv_id for i in self.population if i._parameters_dict['n_conv_l'] == 3 and i._parameters_dict['n_fc_l'] == 3])

            # Lets give more probability of selection to models with high accuracy
            candidates = list(candidates)  # back to an ordered data structure
            accuracies = [self.population[ind_id].post_training_accuracy for ind_id in candidates]
            random_chance = 100. / self.n_classes
            accuracies = np.square(np.array(accuracies) - random_chance)
            accuracies = accuracies / np.sum(accuracies)  # make it sum to 1
            chosen = rng.choice(candidates, p=accuracies)
        else:
            # Exclude individuals that are already too small:
            candidates -= set([i.indv_id for i in self.population if i.n_parameters < int(1E2)])
            # If possible, exclude individuals that have already been slimed-down:
            already_cut = set([i.indv_id for i in self.population if i.cut_offsprings > 0])
            if len(candidates - already_cut) > 0:
                candidates -= already_cut
            chosen = rng.choice(list(candidates))

        return chosen

    def _get_pareto_front(self, exclude_list=[]):
        # TODO: This function is not perfect: In the rare case of where two identical
        # solutions occur and they are not dominated, none of them will be put in the front.
        # Fix this.
        indv_id, num_of_pars, neg_accuracy,  = [], [], []
        for indv in self.population:
            if indv.indv_id not in exclude_list:
                num_of_pars.append(indv.n_parameters)
                neg_accuracy.append(-indv.post_training_accuracy)
                indv_id.append(indv.indv_id)

        n_indiv = len(indv_id)
        if (n_indiv) == 0:
            return []
        num_of_pars = np.array(num_of_pars)[:, np.newaxis]
        neg_accuracy = np.array(neg_accuracy)[:, np.newaxis]
        not_eye = np.logical_not(np.eye(n_indiv))  # False in the main diag, True elsew.
        indv_id = np.array(indv_id)

        worst_at_num_pars = np.less_equal(num_of_pars, num_of_pars.T)
        worst_at_accuracy = np.less_equal(neg_accuracy, neg_accuracy.T)
        worst_at_both = np.logical_and(worst_at_num_pars, worst_at_accuracy)
        worst_at_both = np.logical_and(worst_at_both, not_eye)  # excludes self-comparisons
        domination = np.any(worst_at_both, axis=0)

        pareto_front = indv_id[np.logical_not(domination)]
        return list(pareto_front)

    def _non_dominated_sorting(self, n_fronts: int):
        pareto_fronts = []  # This will become a list of lists
        exclude_list = []
        for _ in range(n_fronts):
            front = self._get_pareto_front(exclude_list=exclude_list)
            pareto_fronts.append(front)
            exclude_list.extend(front)
        return pareto_fronts

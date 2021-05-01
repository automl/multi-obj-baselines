import math
import numpy as np

from ax import Experiment

from baselines import nDS, computeHV2D
from .member import Member
from .member import Mutation
from .member import Recombination
from .member import ParentSelection


class SHEMOA:
    """
    Succesive Halving Evolutionary MultiObjective Algorithm
    :param population_size: int
    :param mutation_type: hyperparameter to set mutation strategy
    :param recombination_type: hyperparameter to set recombination strategy
    :param sigma: conditional hyperparameter dependent on mutation_type GAUSSIAN
    :param recom_proba: conditional hyperparameter dependent on recombination_type UNIFORM
    :param selection_type: hyperparameter to set selection strategy
    :param total_number_of_function_evaluations: maximum allowed function evaluations
    :param children_per_step: how many children to produce per step
    :param fraction_mutation: balance between sexual and asexual reproduction
    """
    def __init__(
        self,
        search_space,
        experiment: Experiment,
        population_size: int = 10,
        budget_min: int = 5,
        budget_max: int = 50,
        eta: int = 2,
        init_time: float = 0.0,
        mutation_type: Mutation = Mutation.UNIFORM,
        recombination_type:
        Recombination = Recombination.UNIFORM,
        sigma: float = 1.,
        recom_proba: float = 0.5,
        selection_type: ParentSelection = ParentSelection.TOURNAMENT,
        total_number_of_function_evaluations: int = 200,
        children_per_step: int = 1,
        fraction_mutation: float = .5
    ):
        assert 0 <= fraction_mutation <= 1
        assert 0 < children_per_step
        assert 0 < total_number_of_function_evaluations
        assert 0 < sigma
        assert 0 < population_size

        self.current_budget = 0 # first budget
        self.init_time = init_time
        self.experiment = experiment

        # Compute succesive halving values
        budgets, evals = get_budgets(
            budget_min,
            budget_max,
            eta,
            total_number_of_function_evaluations,
            population_size
        )

        # Initialize population
        self.population = [
            Member(
                search_space,
                budgets[0],
                mutation_type,
                recombination_type,
                'dummy.txt',
                sigma,
                recom_proba,
                experiment=self.experiment
            ) for _ in range(population_size)
        ]

        # NOW WE CANNOT SORT WITH DOUBLE FITNESS, SORT ONLY BY ACCURACY
        self.population.sort(key=lambda x: (15 - x.fitness[0]) * x.fitness[1])

        self.pop_size = population_size
        self.selection = selection_type
        self.max_func_evals = total_number_of_function_evaluations
        self._func_evals = population_size
        self.num_children = children_per_step
        self.frac_mutants = fraction_mutation
        # will store the optimization trajectory and lets you easily observe how often
        # a new best member was generated
        self.trajectory = [self.population[0]]

        # list of different budgets
        self.budgets = budgets
        # list of how many function evaluations to do per budget
        self.evals = evals

    def get_average_fitness(self) -> float:
        """Helper to quickly access average population fitness"""
        # TODO: COMPUTE approx of HYPERVOLUME CONTRIBUTION OF EACH ONE
        return np.mean(list(map(lambda x: (15 - x.fitness[0]) * x.fitness[1], self.population)))

    def select_parents(self):
        """
        Method that implements all selection mechanism.
        For ease of computation we assume that the population members are sorted according to their fitness
        :return: list of ids of selected parents.
        """
        parent_ids = []
        if self.selection == ParentSelection.NEUTRAL:
            parent_ids = np.random.choice(self.pop_size, self.num_children)
        elif self.selection == ParentSelection.FITNESS:
            p = np.array([x.fitness for x in self.population])
            p = (p.max() - p) + 0.0001
            p = p / p.sum()
            parent_ids = np.random.choice(self.pop_size, self.num_children, p=p)

        elif self.selection == ParentSelection.TOURNAMENT:
            k = 3
            parent_ids = [np.random.choice(self.pop_size, min(k, self.pop_size), replace=False).min()
                          for i in range(self.num_children)]
        else:
            raise NotImplementedError
        return parent_ids

    def remove_member(self, fitness):
        # BE CAREFUL: fitness must be a list
        for m in self.population:
            if list(m.fitness) == fitness:
                self.population.remove(m)
                break
        else:
            raise Warning("remove_member did not found the member to remove")
        return m.id

    def step(self) -> float:
        """
        Performs one step of parent selection -> offspring creation -> survival selection
        :return: average population fitness
        """
        # Step 2: Parent selection
        parent_ids = self.select_parents()
        children = []
        for pid in parent_ids:
            if np.random.uniform() < self.frac_mutants:
                children.append(self.population[pid].mutate())
            else:
                children.append(self.population[pid].recombine(np.random.choice(self.population)))
            self._func_evals += 1

        # Step 4: Survival selection
        # (\mu + \lambda)-selection i.e. combine offspring and parents in one sorted list, keep the #pop_size best
        self.population.extend(children)
        costs = np.array([[x.fitness[0], x.fitness[1]] for x in self.population])

        fronts = nDS(costs)

        if len(fronts[-1]) == 1:
            r_member = fronts[-1][0].tolist()
            r_id = self.remove_member(r_member)
            # with open(self.name_file, "a") as myfile:
            #     myfile.write('REMOVED: '+ str(r_id)+'\n')
        else:
            # compute the contribution to the hypervolume
            reference_point = [np.log10(10**8), 0]
            # sort the front for hv calculation
            sfront_indexes = np.argsort(fronts[-1][:, 1])
            sfront = fronts[-1][sfront_indexes, :]

            hv = computeHV2D(sfront, reference_point)
            min_hvc = hv
            min_point = sfront[0]
            sfront = sfront.tolist()
            for point in sfront:
                front_wout = sfront.copy()
                front_wout.remove(point)
                hvc = hv - computeHV2D(front_wout, reference_point)
                if hvc < min_hvc:
                    min_hvc = hvc
                    min_point = point

            r_id = self.remove_member(min_point)

            # with open(self.name_file, "a") as myfile:
            #     myfile.write('REMOVED: '+ str(r_id)+'\n')

        self.population.sort(key=lambda x: (15 - x.fitness[0]) * x.fitness[1])
        self.trajectory.append(self.population[0])
        return self.get_average_fitness()

    def optimize(self):
        """
        Simple optimization loop that stops after a predetermined number of function evaluations
        :return:
        """
        step = 1
        for b in range(len(self.budgets)):
            if b > 0:
                self.current_budget = b
                # Change budget for all members in population (train them for bigger budget)
                for m in self.population:
                    m.budget = self.budgets[b]
                # Re-order the population given the new fitness with bigger budget
                self.population.sort(key=lambda x: (15 - x.fitness[0]) * x.fitness[1])
            for e in range(self.evals[b]):
                avg_fitness = self.step()
                print(step)
                step += 1

        # Calculate pareto front of population
        costs = np.array([x.fitness for x in self.population])
        fronts = nDS(costs)

        pareto = []
        pareto_front = fronts[0].tolist()
        for m in self.population:
            if list(m.fitness) in pareto_front:
                pareto.append(m)
        return pareto


def get_budgets(bmin, bmax, eta, max_evals, pop_size):
    # Size of all budgets
    budgets = []
    b = bmax
    while b > bmin:
        budgets.append(b)
        b = math.ceil(b / eta)

    # Number of function evaluations to do per budget
    evals = []
    min_evals = math.ceil((max_evals-pop_size) / sum([eta**i for i in range(len(budgets))]))
    for _ in range(len(budgets)):
        evals.append(min_evals)
        min_evals = eta * min_evals

    return np.flip(np.array(budgets)), np.flip(np.array(evals))
from baselines.problems import get_flowers
from baselines.problems.flowers import FlowersSearchSpace
from baselines.problems import get_fashion
from baselines.problems.fashion import FashionSearchSpace
from baselines import save_experiment
from baselines.methods.shemoa import SHEMOA
from baselines.methods.shemoa import Mutation, Recombination, ParentSelection


if __name__ == '__main__':

    # Parameters Flowers
    N_init = 100
    min_budget = 5
    max_budget = 25
    max_function_evals = 15000
    mutation_type = Mutation.UNIFORM
    recombination_type = Recombination.UNIFORM
    selection_type = ParentSelection.TOURNAMENT
    search_space = FlowersSearchSpace()
    experiment = get_flowers('SHEMOA')

    # Parameters Fashion
    # N_init = 10
    # min_budget = 5
    # max_budget = 25
    # max_function_evals = 150
    # mutation_type = Mutation.UNIFORM
    # recombination_type = Recombination.UNIFORM
    # selection_type = ParentSelection.TOURNAMENT
    # search_space = FashionSearchSpace()
    # experiment = get_fashion('SHEMOA')

    #################
    #### SH-EMOA ####
    #################
    ea = SHEMOA(
        search_space,
        experiment,
        N_init, min_budget, max_budget,
        mutation_type=mutation_type,
        recombination_type=recombination_type,
        selection_type=selection_type,
        total_number_of_function_evaluations=max_function_evals
    )
    ea.optimize()
    save_experiment(experiment, f'{experiment.name}.pickle')

from baselines.methods.mobohb.run_mobohb import get_MOBOHB
from baselines.problems import get_flowers
from baselines.problems.flowers import FlowersSearchSpace
from baselines.problems import get_fashion
from baselines.problems.fashion import FashionSearchSpace
from baselines import save_experiment


if __name__ == '__main__':

    # Parameters Flowers
    N_init = 50
    num_candidates = 24
    gamma = 0.10
    min_budget = 5
    max_budget = 25
    max_function_evals = 2000
    search_space = FlowersSearchSpace()
    experiment = get_flowers('MOBOHB')

    # Parameters Fashion
    # N_init = 10
    # num_candidates = 24
    # gamma = 0.10
    # min_budget = 5
    # max_budget = 25
    # max_function_evals = 150
    # search_space = FlowersSearchSpace()
    # experiment = get_flowers('MOBOHB')


    ################
    #### MOBOHB ####
    ################
    get_MOBOHB(
        experiment,
        search_space,
        num_initial_samples=N_init,
        num_candidates=num_candidates,
        gamma=gamma,
        num_iterations=max_function_evals,
        min_budget=min_budget,
        max_budget=max_budget,
    )
    save_experiment(experiment, f'{experiment.name}.pickle')

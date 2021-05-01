from baselines.problems import get_flowers
from baselines.problems.flowers import FlowersSearchSpace
from baselines.problems import get_fashion
from baselines.problems.fashion import FashionSearchSpace
from baselines import save_experiment
from baselines.methods.mobananas import get_MOSHBANANAS


if __name__ == '__main__':

    # Parameters Flowers
    N_init = 10
    min_budget = 5
    max_budget = 25
    max_function_evals = 10000
    num_arch=20
    select_models=10
    eta=3
    search_space = FlowersSearchSpace()
    experiment = get_flowers('MOSHBANANAS')

    # Parameters Fashion
    # N_init = 10
    # min_budget = 5
    # max_budget = 25
    # max_function_evals = 400
    # num_arch=20
    # select_models=10
    # eta=3
    # search_space = FashionSearchSpace()
    # experiment = get_fashion('MOSHBANANAS')


    #####################
    #### MOSHBANANAS ####
    #####################
    get_MOSHBANANAS(
        experiment,
        search_space,
        initial_samples=N_init,
        select_models=select_models,
        num_arch=num_arch,
        min_budget=min_budget,
        max_budget=max_budget,
        function_evaluations=max_function_evals,
        eta=eta
    )
    save_experiment(experiment, f'{experiment.name}.pickle')

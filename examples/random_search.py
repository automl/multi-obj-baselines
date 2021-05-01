from ax import Models

from baselines.problems import get_flowers
from baselines.problems import get_branin_currin
from baselines.problems import get_fashion
from baselines import save_experiment

if __name__ == '__main__':

    # Parameters Flowers
    N = 20000   # Number of samples (it is not important)
    experiment = get_flowers('RandomSearch')  # Function to get the problem

    # Parameters Fashion
    # N = 20000   # Number of samples (it is not important)
    # experiment = get_fashion('RandomSearch')  # Function to get the problem

    #######################
    #### Random Search ####
    #######################
    for _ in range(N):
        experiment.new_trial(Models.SOBOL(experiment.search_space).gen(1))
        experiment.fetch_data()

    print(experiment.fetch_data().df)
    save_experiment(experiment, f'{experiment.name}.pickle')

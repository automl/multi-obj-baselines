from ax import Metric
from ax import MultiObjective
from ax import ObjectiveThreshold
from ax import MultiObjectiveOptimizationConfig

from baselines import MultiObjectiveSimpleExperiment
from .fashionnet import evaluate_network
from .search_space import CustomSearchSpace

def get_fashion(name=None):

    val_acc_1 = Metric('val_acc_1', True)
    val_acc_3 = Metric('val_acc_3', True)
    tst_acc_1 = Metric('tst_acc_1', True)
    tst_acc_3 = Metric('tst_acc_3', True)
    num_params = Metric('num_params', True)

    objective = MultiObjective([val_acc_1, num_params])
    thresholds = [
        ObjectiveThreshold(val_acc_1, 0.0),
        ObjectiveThreshold(num_params, 8.0)
    ]
    optimization_config = MultiObjectiveOptimizationConfig(
        objective=objective,
        objective_thresholds=thresholds
    )

    return MultiObjectiveSimpleExperiment(
        name=name,
        search_space=CustomSearchSpace().as_ax_space(),
        evaluation_function=evaluate_network,
        optimization_config=optimization_config,
        extra_metrics=[val_acc_3, tst_acc_1, tst_acc_3]
    )

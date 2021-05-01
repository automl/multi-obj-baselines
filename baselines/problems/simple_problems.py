import torch
import numpy as np

from botorch.test_functions.multi_objective import BraninCurrin

from ax import Metric
from ax.core.search_space import SearchSpace
from ax.core.objective import MultiObjective
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.outcome_constraint import ObjectiveThreshold
from ax.core.optimization_config import MultiObjectiveOptimizationConfig

from baselines import MultiObjectiveSimpleExperiment

def get_branin_currin(name=None):

    metric_a = Metric('a', False)
    metric_b = Metric('b', False)

    objective = MultiObjective([metric_a, metric_b])
    thresholds = [
        ObjectiveThreshold(metric_a, 0.0),
        ObjectiveThreshold(metric_b, 8.0)
    ]
    optimization_config = MultiObjectiveOptimizationConfig(
        objective=objective,
        objective_thresholds=thresholds
    )

    x1 = RangeParameter(
        name="x1", lower=0, upper=1, parameter_type=ParameterType.FLOAT
    )
    x2 = RangeParameter(
        name="x2", lower=0, upper=1, parameter_type=ParameterType.FLOAT
    )

    search_space = SearchSpace(
        parameters=[x1, x2],
    )

    branin_currin = BraninCurrinEvalFunction()

    return MultiObjectiveSimpleExperiment(
        name=name,
        search_space=search_space,
        eval_function=branin_currin,
        optimization_config=optimization_config,
    )

class BraninCurrinEvalFunction:
    def __init__(self):
        self.branin_currin = BraninCurrin(negate=True).to(
        dtype=torch.double, 
        device= torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    def __call__(self, x):
        x = torch.tensor([x['x1'], x['x2']])
        return {
            'a': (float(self.branin_currin(x)[0]), 0.0),
            'b': (float(self.branin_currin(x)[1]), 0.0),
        }

    def discrete_call(self, x):
        return self(x)['a'][0]

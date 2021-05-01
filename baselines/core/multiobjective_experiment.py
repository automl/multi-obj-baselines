from typing import Optional, Dict, Any, List
import pickle

from ax import Arm
from ax import Metric
from ax import Experiment
from ax import SearchSpace
from ax import SimpleExperiment
from ax import OptimizationConfig

from ax.core.simple_experiment import TEvaluationFunction
from ax.core.simple_experiment import unimplemented_evaluation_function

class MultiObjectiveSimpleExperiment(SimpleExperiment):

    def __init__(
        self,
        search_space: SearchSpace,
        optimization_config: OptimizationConfig,
        name: Optional[str] = None,
        eval_function: TEvaluationFunction = unimplemented_evaluation_function,
        status_quo: Optional[Arm] = None,
        properties: Optional[Dict[str, Any]] = None,
        extra_metrics: Optional[List[Metric]] = None,
    ):
        super(MultiObjectiveSimpleExperiment, self).__init__(
            search_space=search_space,
            name=name,
            evaluation_function=eval_function,
            status_quo=status_quo,
            properties=properties
        )

        self.optimization_config = optimization_config

        if extra_metrics is not None:
            for metric in extra_metrics:
                Experiment.add_tracking_metric(self, metric)


def save_experiment(experiment: Experiment, filename: str):
    with open(filename, 'wb') as file:
        pickle.dump(experiment, file)


def load_experiment(filename: str):
    with open(filename, 'rb') as file:
        return pickle.load(file)

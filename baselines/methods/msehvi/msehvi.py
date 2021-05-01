from ax.core.generator_run import GeneratorRun
from baselines.core.multiobjective_experiment import MultiObjectiveSimpleExperiment
import torch
import numpy as np
from copy import deepcopy
from typing import Optional, Callable

from ax import Models, Experiment, Data, MultiObjective
from ax.core import ObservationFeatures
from ax.core.observation import observations_from_data
from ax.modelbridge import MultiObjectiveTorchModelBridge
from ax.modelbridge.factory import DEFAULT_EHVI_BATCH_LIMIT
from ax.modelbridge.registry import MODEL_KEY_TO_MODEL_SETUP
from ax.models.torch.botorch_defaults import get_and_fit_model
from ax.utils.common.typeutils import checked_cast
from botorch.models import GenericDeterministicModel
from botorch.models.model import Model
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultitaskMultivariateNormal

class MSEHVI:

    def __init__(
        self,
        experiment: Experiment,
        discrete_metric: str,
        discrete_function: Callable,
    ):

        self.experiment = experiment
        self.discrete_m = discrete_metric
        self.discrete_f = discrete_function

        self.experiment_simple = MultiObjectiveSimpleExperiment(
            name=self.experiment.name,
            search_space=self.experiment.search_space,
            eval_function=self.experiment.evaluation_function,
            optimization_config=self.experiment.optimization_config,
        )

        self.metrics = [
            m for m in self.experiment.optimization_config.objective.metrics
        ]

        # Initialize simple experiment with current observed data
        d = self.experiment.fetch_data().df
        for i in sorted(experiment.trials.keys()):
            trial_simple = self.experiment_simple.new_trial(
                GeneratorRun([experiment.trials[i].arm])
            )
            trial_simple._time_created = experiment.trials[i]._time_created
            trial_simple._time_completed = experiment.trials[i]._time_completed

            index = d['trial_index'] == i
            m_1 = (index) & (d['metric_name'] == self.metrics[0].name)
            m_2 = (index) & (d['metric_name'] == self.metrics[1].name)
            data = Data.from_evaluations({
                trial_simple.arm.name: {
                    self.metrics[0].name: (d[m_1]['mean'].values[0], 0.0),
                    self.metrics[1].name: (d[m_2]['mean'].values[0], 0.0),
                }
            }, trial_simple.index)

            self.experiment_simple.attach_data(data)


    def step(self):

        trial_simple = self.experiment_simple.new_trial(
            get_MOO_MSEHVI(
                self.discrete_m,
                self.discrete_f,
                self.experiment_simple,
                self.experiment_simple.fetch_data()
            ).gen(1)
        )
        trial = self.experiment.new_trial(GeneratorRun([trial_simple.arm]))
        
        trial_simple.mark_running()
        d = self.experiment.fetch_data().df
        trial_simple.mark_completed()

        index = d['trial_index'] == trial.index
        m_1 = (index) & (d['metric_name'] == self.metrics[0].name)
        m_2 = (index) & (d['metric_name'] == self.metrics[1].name)
        data = Data.from_evaluations({
            trial_simple.arm.name: {
                self.metrics[0].name: (d[m_1]['mean'].values[0], 0.0),
                self.metrics[1].name: (d[m_2]['mean'].values[0], 0.0),
            }
        }, trial_simple.index)
        self.experiment_simple.attach_data(data)



def get_MOO_MSEHVI(
    discrete_metric: str,
    discrete_function: Callable,
    experiment: Experiment,
    data: Data,
    dtype: torch.dtype = torch.double,
    device: Optional[torch.device] = None,
) -> MultiObjectiveTorchModelBridge:
    """Instantiates a multi-objective model that used Mixed Surrogate EHVI.

    Requires `objective_thresholds`,
    a list of `ax.core.ObjectiveThresholds`, for every objective being optimized.
    An arm only improves hypervolume if it is strictly better than all
    objective thresholds.

    `objective_thresholds` can be passed in the optimization_config or
    passed directly here.
    """
    device = device or (
        torch.device("cuda") if torch.cuda.is_available() 
        else torch.device("cpu")
    )
    if not isinstance(experiment.optimization_config.objective, MultiObjective):
        raise ValueError(
            "Multi-objective optimization requires multiple objectives."
        )
    if data.df.empty:
        raise ValueError("MultiObjectiveOptimization requires non-empty data.")

    return checked_cast(
        MultiObjectiveTorchModelBridge,
        Models.MOO(
            experiment=experiment,
            data=data,
            search_space=experiment.search_space,
            torch_dtype=dtype,
            torch_device=device,
            model_constructor=MixedDeterministicModelConstructor(
                discrete_metric, discrete_function, experiment, data
            ),
            default_model_gen_options={
                "acquisition_function_kwargs": {"sequential": True},
                "optimizer_kwargs": {
                    # having a batch limit is very important for avoiding
                    # memory issues in the initialization
                    "batch_limit": DEFAULT_EHVI_BATCH_LIMIT
                },
            },
        ),
    )


class GPyTorchWithDeterministicCost(Model):
    num_outputs = 2  # to inform GPyTorchModel API

    def __init__(self, g1, g2, d, i):
        super().__init__()
        self.g1 = g1
        self.g2 = g2
        self.d = d
        self.i = i

    def posterior(self, X, **kwargs):
        d_posterior = self.d.posterior(X, )

        g1_posterior = self.g1.posterior(X, )
        g2_posterior = self.g2.posterior(X, )

        if self.i == 0:
            g1_posterior.mvn *= 10e-10
            g1_posterior.mean[..., 0] = d_posterior.mean[..., 0]
        else:
            g2_posterior.mvn *= 10e-10
            g2_posterior.mean[..., 0] = d_posterior.mean[..., 0]

        mvns = [g1_posterior.mvn, g2_posterior.mvn]
        return GPyTorchPosterior(
            mvn=MultitaskMultivariateNormal.from_independent_mvns(mvns=mvns)
        )


class MixedDeterministicModelConstructor:

    def __init__(
            self,
            d_metric: str,
            d_function: Callable,
            experiment: Experiment,
            data: Data
    ):
        self.d_metric = d_metric
        self.d_function = d_function
        self.experiment = experiment

        observations = observations_from_data(experiment, data)
        t_obs_feat, t_obs_data = list(
            zip(*[(obs.features, obs.data) for obs in observations])
        )
        t_search_space = deepcopy(experiment.search_space)

        self.transforms = {}
        for t in MODEL_KEY_TO_MODEL_SETUP['MOO'].transforms:
            t_instance = t(t_search_space, t_obs_feat, t_obs_data)
            t_search_space = t_instance.transform_search_space(t_search_space)
            t_obs_feat = t_instance.transform_observation_features(t_obs_feat)
            t_obs_data = t_instance.transform_observation_data(
                t_obs_data, t_obs_feat
            )
            self.transforms[t.__name__] = t_instance

        self.search_space = t_search_space

    def __call__(
            self,
            Xs,
            Ys,
            Yvars,
            task_features,
            fidelity_features,
            metric_names,
            state_dict=None,
            refit_model=True,
            **kwargs,
    ):

        def f(x):
            """Run self.d_function on the untransformed values of x

            """
            x = x.detach().cpu().numpy()
            res_total = np.zeros((*x.shape[:-1], 1))

            for i in range(len(x)):
                parameters = {
                    p: float(x[i, ..., j])
                    for j, p in enumerate(self.search_space.parameters)
                }
                observation_features = [
                    ObservationFeatures(parameters=parameters)
                ]

                for t in reversed(list(self.transforms.values())):
                    observation_features = t.untransform_observation_features(
                        observation_features
                    )
                params = observation_features[0].parameters

                mean = list(self.transforms.values())[-1].Ymean
                std = list(self.transforms.values())[-1].Ystd
                res = (self.d_function(params) - mean[self.d_metric])
                res = res / std[self.d_metric]
                res_total[i, ..., 0] = res

            return torch.as_tensor(res_total)

        gp1 = get_and_fit_model(
            [Xs[0]],
            [Ys[0]],
            [Yvars[0]],
            task_features,
            fidelity_features,
            metric_names[0]
        )
        gp2 = get_and_fit_model(
            [Xs[1]],
            [Ys[1]],
            [Yvars[1]],
            task_features,
            fidelity_features,
            metric_names[1]
        )
        dt = GenericDeterministicModel(f, num_outputs=1)

        return GPyTorchWithDeterministicCost(
            gp1, gp2, dt,
            metric_names.index(self.d_metric)
        )

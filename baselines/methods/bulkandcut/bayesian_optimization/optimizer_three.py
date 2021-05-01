import os

from baselines.methods.bulkandcut.bayesian_optimization.constrained_bayesian_optimizer \
    import ConstrainedBayesianOptimizer


class OptimizerThree(ConstrainedBayesianOptimizer):
    """
    Optimizer used on phase 3 (slim-down)
    """

    def __init__(self, log_dir: str):
        self.log_path = os.path.join(log_dir, "BO_Three.csv")
        parameter_bounds = {
            "lr_exp": (-5., -2.),
            # "w_decay_exp": (-4., -1.),  # weight_decay = 10^w_decay_exp
            # The parameters bellow are observed but not controlled by the optimizer:
            "depth": (1., 15.),  # Depth of the network
            "log_npars": (0., 8.),  # log10 of the number of parameters of the network
        }
        super().__init__(par_bounds=parameter_bounds)

    def register_target(self, config, learning_curves):
        valid_loss = learning_curves["train_loss"][-1]
        super().register_target(
            par_values=config,
            target=valid_loss
        )
        self.save_csv(self.log_path)

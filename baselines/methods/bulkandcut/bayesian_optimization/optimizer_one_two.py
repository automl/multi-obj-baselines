import os

from baselines.methods.bulkandcut.bayesian_optimization.constrained_bayesian_optimizer \
    import ConstrainedBayesianOptimizer


class OptimizerOneTwo(ConstrainedBayesianOptimizer):
    """
    Optimizer used on phases 1 and 2 (initialization and bulk-up)
    """

    def __init__(self, log_dir: str):
        self.log_path = os.path.join(log_dir, "BO_OneTwo.csv")
        parameter_bounds = {
            "lr_exp": (-5., -0.),  # LR = 10^lr_exp
            #"w_decay_exp": (-4., -1.),  # weight_decay = 10^w_decay_exp
            #"lr_sched_gamma": (1., 1.),  # 1. is equivalent to no schedule
            #"lr_sched_step_size": (2., 50.),
            # The parameters bellow are observed but not controlled by the optimizer.
            "depth": (1., 15.),  # Depth of the network
            "log_npars": (0., 8.),  # log10 of the number of parameters of the network
        }
        # The baseline (default configuration) is included in the search space.
        # default conf = {
        #     "lr_exp" : math.log10(2.244958736283895e-05),
        #     "w_decay_exp" : -2,
        #     "lr_sched_gamma" : 1.,  # No schedule
        #     "lr_sched_step_size" : 25.,  # This is irrelevant, because lr_sched_gamma=1.
        # }
        super().__init__(par_bounds=parameter_bounds)

    def register_target(self, config, learning_curves):
        valid_loss = learning_curves["validation_loss"][-1]
        super().register_target(
            par_values=config,
            target=valid_loss
        )
        self.save_csv(self.log_path)

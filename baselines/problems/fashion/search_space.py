"""A common search space for all the experiments
"""

import ConfigSpace as CS
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from ConfigSpace.hyperparameters import CategoricalHyperparameter


class CustomSearchSpace(CS.ConfigurationSpace):

    def __init__(self):
        super(CustomSearchSpace, self).__init__()

        # Convolution
        n_conv_l = UniformIntegerHyperparameter("n_conv_l", 1, 3, default_value=3)
        n_conv_0 = UniformIntegerHyperparameter("n_conv_0", 16, 1024, default_value=128, log=True)
        n_conv_1 = UniformIntegerHyperparameter("n_conv_1", 16, 1024, default_value=128, log=True)
        n_conv_2 = UniformIntegerHyperparameter("n_conv_2", 16, 1024, default_value=128, log=True)

        # Dense
        n_fc_l = UniformIntegerHyperparameter("n_fc_l", 1, 3, default_value=3)
        n_fc_0 = UniformIntegerHyperparameter("n_fc_0", 2, 512, default_value=32, log=True)
        n_fc_1 = UniformIntegerHyperparameter("n_fc_1", 2, 512, default_value=32, log=True)
        n_fc_2 = UniformIntegerHyperparameter("n_fc_2", 2, 512, default_value=32, log=True)

        # Kernel Size
        ks = CategoricalHyperparameter("kernel_size", choices=[7, 5, 3], default_value=5)

        # Learning Rate
        lr = UniformFloatHyperparameter('lr_init', 0.00001, 1.0, default_value=0.001, log=True)

        # Use Batch Normalization
        bn = CategoricalHyperparameter("batch_norm", choices=[False, True], default_value=False)

        # Batch size
        bs = UniformIntegerHyperparameter('batch_size', 1, 512, default_value=128, log=True)

        # Global Avg Pooling
        ga = CategoricalHyperparameter("global_avg_pooling", choices=[False, True], default_value=True)

        # Conditions

        cond1 = CS.conditions.InCondition(n_conv_2, n_conv_l, [3])
        cond2 = CS.conditions.InCondition(n_conv_1, n_conv_l, [2, 3])

        cond3 = CS.conditions.InCondition(n_fc_2, n_fc_l, [3])
        cond4 = CS.conditions.InCondition(n_fc_1, n_fc_l, [2, 3])
        cond5 = CS.conditions.InCondition(n_fc_0, n_fc_l, [1, 2, 3])

        self.not_mutables = ['n_conv_l', 'n_fc_l']

        self.add_hyperparameters([n_conv_l, n_conv_0, n_conv_1, n_conv_2])
        self.add_hyperparameters([n_fc_l, n_fc_0, n_fc_1, n_fc_2])
        self.add_hyperparameters([ks, lr, bn, bs, ga])
        self.add_conditions([cond1, cond2, cond3, cond4, cond5])

    def as_uniform_space(self):

        # Convolution
        n_conv_l = self.get_hyperparameter('n_conv_l')
        n_conv_0 = self.get_hyperparameter('n_conv_0')
        n_conv_1 = self.get_hyperparameter('n_conv_1')
        n_conv_2 = self.get_hyperparameter('n_conv_2')

        # Dense
        n_fc_l = self.get_hyperparameter('n_fc_l')
        n_fc_0 = self.get_hyperparameter('n_fc_0')
        n_fc_1 = self.get_hyperparameter('n_fc_1')
        n_fc_2 = self.get_hyperparameter('n_fc_2')

        # Kernel Size
        ks = UniformIntegerHyperparameter('kernel_size', 0, 2, default_value=1)

        # Learning Rate
        lr = self.get_hyperparameter('lr_init')

        # Use Batch Normalization
        bn = UniformIntegerHyperparameter("batch_norm", 0, 1, default_value=1)

        # Batch size
        bs = self.get_hyperparameter('batch_size')

        # Global Avg Pooling
        ga = UniformIntegerHyperparameter('global_avg_pooling', 0, 1, default_value=1)

        # Conditions
        cond1 = CS.conditions.InCondition(n_conv_2, n_conv_l, [3])
        cond2 = CS.conditions.InCondition(n_conv_1, n_conv_l, [2, 3])

        cond3 = CS.conditions.InCondition(n_fc_2, n_fc_l, [3])
        cond4 = CS.conditions.InCondition(n_fc_1, n_fc_l, [2, 3])
        cond5 = CS.conditions.InCondition(n_fc_0, n_fc_l, [1, 2, 3])

        cs = CS.ConfigurationSpace()

        cs.add_hyperparameters([n_conv_l, n_conv_0, n_conv_1, n_conv_2])
        cs.add_hyperparameters([n_fc_l, n_fc_0, n_fc_1, n_fc_2])
        cs.add_hyperparameters([ks, lr, bn, bs, ga])
        cs.add_conditions([cond1, cond2, cond3, cond4, cond5])
        return cs

    def as_ax_space(self):
        from ax import ParameterType, RangeParameter, FixedParameter, ChoiceParameter, SearchSpace

        # Convolution
        n_conv_l = RangeParameter('n_conv_l', ParameterType.INT, 1, 3)
        n_conv_0 = RangeParameter('n_conv_0', ParameterType.INT, 16, 1024, True)
        n_conv_1 = RangeParameter('n_conv_1', ParameterType.INT, 16, 1024, True)
        n_conv_2 = RangeParameter('n_conv_2', ParameterType.INT, 16, 1024, True)

        # Dense
        n_fc_l = RangeParameter('n_fc_l', ParameterType.INT, 1, 3)
        n_fc_0 = RangeParameter('n_fc_0', ParameterType.INT, 2, 512, True)
        n_fc_1 = RangeParameter('n_fc_1', ParameterType.INT, 2, 512, True)
        n_fc_2 = RangeParameter('n_fc_2', ParameterType.INT, 2, 512, True)

        # Kernel Size
        ks = ChoiceParameter('kernel_size', ParameterType.INT, values=[3, 5, 7])

        # Learning Rate
        lr =  RangeParameter('lr_init', ParameterType.FLOAT, 0.00001, 1.0, True)

        # Use Batch Normalization
        bn = ChoiceParameter('batch_norm', ParameterType.BOOL, values=[True, False])

        # Batch size
        bs = RangeParameter('batch_size', ParameterType.INT, 1, 512, True)

        # Global Avg Pooling
        ga = ChoiceParameter('global_avg_pooling', ParameterType.BOOL, values=[True, False])

        b = FixedParameter('budget', ParameterType.INT, 25)

        i = FixedParameter('id', ParameterType.STRING, 'dummy')

        return SearchSpace(
            parameters=[n_conv_l, n_conv_0, n_conv_1, n_conv_2, n_fc_l, n_fc_0, n_fc_1, n_fc_2, ks, lr, bn, bs, ga, b, i],
        )


    def sample_hyperparameter(self, hp):
        if not self.is_mutable_hyperparameter(hp):
            raise Exception("Hyperparameter {} is not mutable and must be fixed".format(hp))
        return self.get_hyperparameter(hp).sample(self.random)

    def is_mutable_hyperparameter(self, hp):
        return hp not in self.not_mutables
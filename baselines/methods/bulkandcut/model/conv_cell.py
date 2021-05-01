from copy import deepcopy

import torch


class ConvCell(torch.nn.Module):

    @classmethod
    def NEW(cls, index, parameters, in_elements: int):
        # sample
        out_elements = parameters[f'n_conv_{index}']
        kernel_size = parameters['kernel_size']
        conv = torch.nn.Conv2d(
            in_channels=in_elements,
            out_channels=out_elements,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            )
        bnorm = torch.nn.BatchNorm2d(num_features=out_elements) if parameters['batch_norm'] else torch.nn.Identity()

        return cls(conv_layer=conv, batch_norm=bnorm)

    def __init__(self, conv_layer, batch_norm, dropout_p: float = .5, is_first_cell: bool = False):
        super(ConvCell, self).__init__()
        self.conv = conv_layer
        self.act = torch.nn.ReLU()
        self.bnorm = batch_norm
        self.is_first_cell = is_first_cell  # This changes how the cell is pruned

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.bnorm(x)
        return x

    def downstream_morphism(self):
        identity_layer = torch.nn.Conv2d(
            in_channels=self.out_elements,
            out_channels=self.out_elements,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )

        with torch.no_grad():
            # Initiate weights and biases with the identity function
            torch.nn.init.dirac_(identity_layer.weight)
            torch.nn.init.zeros_(identity_layer.bias)

            # And add some noise to break the symmetry
            identity_layer.weight += torch.rand_like(identity_layer.weight) * 1E-5
            identity_layer.bias += torch.rand_like(identity_layer.bias) * 1E-5

        # Batch-norm morphism (is this the best way?):
        if isinstance(self.bnorm, torch.nn.BatchNorm2d):
            bnorm = torch.nn.BatchNorm2d(num_features=self.out_elements)
            bnorm.weight = torch.nn.Parameter(deepcopy(self.bnorm.weight))
            bnorm.running_var = torch.square(deepcopy(self.bnorm.weight).detach()) - self.bnorm.eps
            bnorm.bias = torch.nn.Parameter(deepcopy(self.bnorm.bias))
            bnorm.running_mean = deepcopy(self.bnorm.bias).detach()
        else:
            bnorm = torch.nn.Identity()

        return ConvCell(conv_layer=identity_layer, batch_norm=bnorm)

    @torch.no_grad()
    def prune(self, out_selected, amount: float = .1):

        num_out_elements = len(out_selected)
        conv_weight = self.conv.weight[out_selected]
        conv_bias = self.conv.bias[out_selected]

        if self.is_first_cell:
            num_in_elements = self.in_elements
            in_selected = None  # should be ignored by the calling code
        else:
            # Upstream filters with the lowest L1 norms will be pruned
            elements_to_prune = int(amount * self.in_elements)  # implicit floor
            num_in_elements = self.in_elements - elements_to_prune
            w_l1norm = torch.sum(
                input=torch.abs(self.conv.weight),
                dim=[0, 2, 3],
            )
            candidates = torch.argsort(w_l1norm)[:2 * elements_to_prune]
            idx_to_prune = torch.randperm(candidates.size(0))[:elements_to_prune]
            in_selected = torch.arange(self.in_elements)
            for kill in idx_to_prune:
                in_selected = torch.cat((in_selected[:kill], in_selected[kill + 1:]))
            conv_weight = conv_weight[:, in_selected]


        # Pruning the convolution:
        pruned_conv = torch.nn.Conv2d(
            in_channels=num_in_elements,
            out_channels=num_out_elements,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        pruned_conv.weight = torch.nn.Parameter(deepcopy(conv_weight))
        pruned_conv.bias = torch.nn.Parameter(deepcopy(conv_bias))

        if isinstance(self.bnorm, torch.nn.BatchNorm2d):
            # Pruning the batch norm:
            bnorm_weight = self.bnorm.weight[out_selected]
            bnorm_bias = self.bnorm.bias[out_selected]
            bnorm_running_var = self.bnorm.running_var[out_selected]
            bnorm_running_mean = self.bnorm.running_mean[out_selected]
            pruned_bnorm = torch.nn.BatchNorm2d(num_features=num_out_elements)
            pruned_bnorm.weight = torch.nn.Parameter(deepcopy(bnorm_weight))
            pruned_bnorm.bias = torch.nn.Parameter(deepcopy(bnorm_bias))
            pruned_bnorm.running_var = deepcopy(bnorm_running_var)
            pruned_bnorm.bnorm_running_mean = deepcopy(bnorm_running_mean)
        else:
            pruned_bnorm = torch.nn.Identity()

        # "Pruning" dropout:
        # drop_p = self.drop.p * (1. - amount)

        # Wrapping it all up:
        pruned_cell = ConvCell(
            conv_layer=pruned_conv,
            batch_norm=pruned_bnorm,
            # dropout_p=drop_p,
            is_first_cell=self.is_first_cell,
            )
        return pruned_cell, in_selected

    @property
    def in_elements(self):
        return self.conv.in_channels

    @property
    def out_elements(self):
        return self.conv.out_channels

    @property
    def kernel_size(self):
        return self.conv.kernel_size

    @property
    def padding(self):
        return self.conv.padding

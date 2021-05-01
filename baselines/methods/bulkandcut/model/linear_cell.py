from copy import deepcopy

import torch
from ax import SearchSpace


class LinearCell(torch.nn.Module):

    @classmethod
    def NEW(cls, index, parameters, in_elements: int):


        out_elements = parameters[f'n_fc_{index}']
        ll = torch.nn.Linear(in_features=in_elements, out_features=out_elements)
        return cls(linear_layer=ll)

    def __init__(self, linear_layer, dropout_p=.5):
        super(LinearCell, self).__init__()
        self.linear = linear_layer
        self.act = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        return x

    def downstream_morphism(self):
        identity_layer = torch.nn.Linear(
            in_features=self.out_elements,
            out_features=self.out_elements,
        )

        with torch.no_grad():
            # Initiate weights and biases with the identity function
            torch.nn.init.eye_(identity_layer.weight)
            torch.nn.init.zeros_(identity_layer.bias)

            # And add some noise to break the symmetry
            identity_layer.weight += torch.rand_like(identity_layer.weight) * 1E-5
            identity_layer.bias += torch.rand_like(identity_layer.bias) * 1E-5

        return LinearCell(linear_layer=identity_layer)

    @torch.no_grad()
    def prune(self, out_selected, amount: float):
        # TODO: improve commentary

        elements_to_prune = int(amount * self.in_elements)  # implicit floor
        num_in_elements = self.in_elements - elements_to_prune
        num_out_elements = self.out_elements if out_selected is None else len(out_selected)

        # Upstream units with the lowest L1 norms will be pruned
        w_l1norm = torch.sum(
            input=torch.abs(self.linear.weight),
            dim=0,
        )
        candidates = torch.argsort(w_l1norm)[:2 * elements_to_prune]
        idx_to_prune = torch.randperm(candidates.size(0))[:elements_to_prune]
        in_selected = torch.arange(self.in_elements)
        for kill in idx_to_prune:
            in_selected = torch.cat((in_selected[:kill], in_selected[kill + 1:]))

        pruned_layer = torch.nn.Linear(
            in_features=num_in_elements,
            out_features=num_out_elements,
            )

        weight = self.linear.weight[:, in_selected]
        bias = self.linear.bias
        if out_selected is not None:
            weight = weight[out_selected]
            bias = bias[out_selected]
        pruned_layer.weight = torch.nn.Parameter(deepcopy(weight))
        pruned_layer.bias = torch.nn.Parameter(deepcopy(bias))


        # Wrapping it up:
        pruned_cell = LinearCell(
            linear_layer=pruned_layer,
            )

        return pruned_cell, in_selected

    @property
    def in_elements(self):
        return self.linear.in_features

    @property
    def out_elements(self):
        return self.linear.out_features

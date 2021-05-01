from copy import deepcopy

import torch


class ModelHead(torch.nn.Module):

    @classmethod
    def NEW(cls, in_elements, out_elements):
        linear_layer = torch.nn.Linear(
            in_features=in_elements,
            out_features=out_elements,
        )
        return ModelHead(linear_layer=linear_layer)

    def __init__(self, linear_layer):
        super(ModelHead, self).__init__()
        self.layer = linear_layer

    @property
    def in_elements(self):
        return self.layer.in_features

    @property
    def out_elements(self):
        return self.layer.out_features

    def forward(self, x):
        return self.layer(x)

    def bulkup(self):
        return deepcopy(self)

    @torch.no_grad()
    def slimdown(self, amount: float):

        elements_to_prune = int(amount * self.in_elements) 
        num_in_elements = self.in_elements - elements_to_prune
        new_layer = torch.nn.Linear(
            in_features=num_in_elements,
            out_features=self.out_elements,
        )

        # Upstream units with the lowest L1 norms will be pruned
        w_l1norm = torch.sum(
            input=torch.abs(self.layer.weight),
            dim=0,
        )
        candidates = torch.argsort(w_l1norm)[:2 * elements_to_prune]
        idx_to_prune = torch.randperm(candidates.size(0))[:elements_to_prune]
        in_selected = torch.arange(self.in_elements)
        for kill in idx_to_prune:
            in_selected = torch.cat((in_selected[:kill], in_selected[kill + 1:]))

        weight = deepcopy(self.layer.weight.data[:, in_selected])
        bias = deepcopy(self.layer.bias)
        new_layer.weight = torch.nn.Parameter(weight)
        new_layer.bias = torch.nn.Parameter(bias)

        narrower_head = ModelHead(linear_layer=new_layer)

        return narrower_head, in_selected

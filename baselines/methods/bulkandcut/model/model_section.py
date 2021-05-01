from copy import deepcopy

import torch
from ax import SearchSpace

from baselines.methods.bulkandcut.model.linear_cell import LinearCell
from baselines.methods.bulkandcut.model.conv_cell import ConvCell
from baselines.methods.bulkandcut.model.skip_connection import SkipConnection
from baselines.methods.bulkandcut import rng, device


class ModelSection(torch.nn.Module):

    @classmethod
    def NEW(cls, index: int, parameters, in_elements: int, section_type: str):
        if section_type not in ["linear", "conv"]:
            raise Exception(f"Unknown section type: {section_type}")

        if section_type == "linear":
            first_cell = LinearCell.NEW(index, parameters, in_elements=in_elements)
            last_op = torch.nn.Identity()
        else:
            first_cell = ConvCell.NEW(index, parameters, in_elements=in_elements)
            last_op = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        cells = torch.nn.ModuleList([first_cell])
        return ModelSection(cells=cells, last_op=last_op)

    def __init__(self,
                 cells: "torch.nn.ModuleList",
                 last_op: "torch.nn.Module",
                 skip_cnns: "torch.nn.ModuleList" = None
                 ):
        super(ModelSection, self).__init__()
        self.cells = cells
        self.last_op = last_op
        self.skip_cnns = skip_cnns if skip_cnns is not None else torch.nn.ModuleList()

    def __len__(self):
        return len(self.cells)

    def __iter__(self):
        return self.cells.__iter__()

    @property
    def in_elements(self):
        return self.cells[0].in_elements

    @property
    def out_elements(self):
        return self.cells[-1].out_elements

    @property
    def skip_connections_summary(self):
        if len(self.skip_cnns) == 0:
            return "\t None\n"
        summary = ""
        for sc in self.skip_cnns:
            summary += "\t" + str(sc) + "\n"
        return summary

    def mark_as_first_section(self):
        self.cells[0].is_first_cell = True

    def forward(self, x):
        n_cells = len(self.cells)
        x = self.cells[0](x)
        x_buffer = self._build_forward_buffer(buffer_shape=x.shape)

        for i in range(1, len(self.cells)):
            if i in x_buffer:
                x += x_buffer[i]
            for sk in self.skip_cnns:
                if sk.source == i:
                    x_buffer[sk.destiny] += sk(x)
            x = self.cells[i](x)
        if n_cells + 1 in x_buffer:
            x += x_buffer[n_cells + 1]
        x = self.last_op(x)
        return x

    def _build_forward_buffer(self, buffer_shape):
        addresses = {skcnn.destiny for skcnn in self.skip_cnns}  # a set
        buffer = {addr: torch.zeros(size=buffer_shape).to(device) for addr in addresses}  # a dict
        return buffer

    def bulkup(self):
        # Adds a new cell
        sel_cell = rng.integers(low=0, high=len(self.cells))
        identity_cell = self.cells[sel_cell].downstream_morphism()
        new_cell_set = deepcopy(self.cells)
        new_cell_set.insert(index=sel_cell + 1, module=identity_cell)

        # Adjust skip connection addressing
        new_skip_cnns = deepcopy(self.skip_cnns)
        for skcnn in new_skip_cnns:
            skcnn.adjust_addressing(inserted_cell=sel_cell + 1)

        # Stochastically add a skip connection
        if rng.random() < .7:
            candidates = self._skip_connection_candidates()
            if len(candidates) > 0:
                chosen = rng.choice(candidates)
                new_skip_connection = SkipConnection(source=chosen[0], destiny=chosen[1])
                new_skip_cnns.append(new_skip_connection)

        deeper_section = ModelSection(
            cells=new_cell_set,
            skip_cnns=new_skip_cnns,
            last_op=deepcopy(self.last_op),
            )
        return deeper_section

    def _skip_connection_candidates(self):
        n_cells = len(self.cells)
        if (n_cells) < 3:
            return []

        already_connected = [(sk.source, sk.destiny) for sk in self.skip_cnns]
        candidates = []
        for source in range(1, n_cells - 1):
            for destiny in range(source + 2, n_cells + 1):
                if (source, destiny) not in already_connected:
                    candidates.append((source, destiny))

        return candidates

    def slimdown(self, out_selected, amount: float):
        narrower_cells = torch.nn.ModuleList()
        for cell in self.cells[::-1]:
            pruned_cell, out_selected = cell.prune(
                out_selected=out_selected,
                amount=amount,
                )
            narrower_cells.append(pruned_cell)

        narrower_section = ModelSection(
            cells=narrower_cells[::-1],
            skip_cnns=deepcopy(self.skip_cnns),
            last_op=deepcopy(self.last_op),
            )
        return narrower_section, out_selected

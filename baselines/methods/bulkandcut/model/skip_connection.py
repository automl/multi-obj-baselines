import torch


class SkipConnection(torch.nn.Module):

    def __init__(self, source: int, destiny: int):
        super(SkipConnection, self).__init__()
        initial_gain = torch.rand(1) * 1E-6
        self.weight = torch.nn.Parameter(data=initial_gain, requires_grad=True)

        self.source = source
        self.destiny = destiny

    def __str__(self):
        summary_str = f"from {self.source} to {self.destiny}"
        summary_str += f" with weight {self.weight.item():.4e}"
        return summary_str

    def forward(self, x):
        x = self.weight * x
        return x

    def adjust_addressing(self, inserted_cell: int):
        if self.source > inserted_cell:
            self.source += 1
        if self.destiny > inserted_cell:
            self.destiny += 1

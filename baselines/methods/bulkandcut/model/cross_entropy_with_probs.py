import torch
import torch.nn.functional as F


class CrossEntropyWithProbs(torch.nn.Module):
    """Cross entropy for soft labels. PyTorch, unlike TensorFlow or Keras, requires this
    workaround because CrossEntropyLoss demands that labels are given in a LongTensor.

    This code was shamelessly copied from Snorkel:
    https://github.com/snorkel-team/snorkel/blob/master/snorkel/classification/loss.py
    """

    def __init__(self, weight: "torch.Tensor" = None, reduction: str = "mean"):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        num_points, num_classes = input.shape

        cum_losses = input.new_zeros(num_points)
        for y in range(num_classes):
            target_temp = input.new_full((num_points,), y, dtype=torch.long)
            y_loss = F.cross_entropy(input, target_temp, reduction="none")
            if self.weight is not None:
                y_loss = y_loss * self.weight[y]
            cum_losses += target[:, y].float() * y_loss

        if self.reduction == "none":
            return cum_losses
        elif self.reduction == "mean":
            return cum_losses.mean()
        elif self.reduction == "sum":
            return cum_losses.sum()
        else:
            raise ValueError("Keyword 'reduction' must be one of ['none', 'mean', 'sum']")

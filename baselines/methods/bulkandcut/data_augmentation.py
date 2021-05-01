import torch
from torchvision.transforms.functional import hflip

from baselines.methods.bulkandcut import rng


class DataAugmentation():
    def __init__(self, n_classes: int, mixup_alpha: float = .25):
        self.n_classes = n_classes
        self.alpha = mixup_alpha

    def __call__(self, data, targets):
        data = self._hflip(data)
        targets = self._onehot(targets)
        data, targets = self._mixup(data, targets)
        return data, targets

    def _hflip(self, data):
        batch_size = data.size(0)
        mask = torch.rand(size=(batch_size,)) > .5
        data[mask] = hflip(data[mask])
        return data

    def _mixup(self, data, targets):
        """
        This function was adapted from:
            https://github.com/hysts/pytorch_mixup/blob/master/utils.py.
        To the author my gratitude. :-)
        """
        batch_size = data.size(0)
        indices = torch.randperm(n=batch_size)
        data2 = data[indices]
        targets2 = targets[indices]

        # Original code:
        # lambda_ = torch.FloatTensor([rng.beta(a=alpha, b=alpha)])
        # data = data * lambda_ + data2 * (1 - lambda_)
        # targets = targets * lambda_ + targets2 * (1 - lambda_)

        # My modification:
        lambda_ = torch.FloatTensor(rng.beta(a=self.alpha, b=self.alpha, size=batch_size))
        lamb_data = lambda_.reshape((-1, 1, 1, 1))
        lamb_targ = lambda_.reshape((-1, 1))
        data = data * lamb_data + data2 * (1 - lamb_data)
        targets = targets * lamb_targ + targets2 * (1 - lamb_targ)

        return data, targets

    def _onehot(self, label):
        template = torch.zeros(label.size(0), self.n_classes)
        ohe = template.scatter_(1, label.view(-1, 1), 1)
        return ohe

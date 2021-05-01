import torch

class Accuracy:

    def __init__(self):
        self.reset()

        self.sum = 0
        self.cnt = 0

    def reset(self):
        self.sum = 0
        self.cnt = 0

    def __call__(self, y_true, y_pred):
        self.sum += torch.sum(y_true == y_pred).to('cpu').numpy()
        self.cnt += y_true.size(0)

        return self.sum / self.cnt

class AccuracyTop1:

    def __init__(self):
        self.reset()

        self.sum = 0
        self.cnt = 0

    def reset(self):
        self.sum = 0
        self.cnt = 0

    def __call__(self, y_true, y_pred):

        self.sum += y_pred.topk(1)[1].eq(y_true.argmax(-1).reshape(-1, 1).expand(-1, 1)).float().sum().to('cpu').numpy()
        self.cnt += y_pred.size(0)

        return self.sum / self.cnt

class AccuracyTop3:

    def __init__(self):
        self.reset()

        self.sum = 0
        self.cnt = 0

    def reset(self):
        self.sum = 0
        self.cnt = 0

    def __call__(self, y_true, y_pred):

        self.sum += y_pred.topk(3)[1].eq(y_true.argmax(-1).reshape(-1, 1).expand(-1, 3)).float().sum().to('cpu').numpy()
        self.cnt += y_pred.size(0)

        return self.sum / self.cnt

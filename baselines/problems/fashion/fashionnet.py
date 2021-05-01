"""A common simple Neural Network for the experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import tqdm
import numpy as np

from .utils import Accuracy, AccuracyTop1, AccuracyTop3
import pathlib




class FashionNet(nn.Module):
    """
    The model to optimize
    """
    def __init__(self, config, input_shape=(3, 28, 28), num_classes=10):
        super(FashionNet, self).__init__()

        inp_ch = input_shape[0]

        layers = []
        for i in range(config['n_conv_l']):

            out_ch = config['n_conv_{}'.format(i)]

            ks = config['kernel_size'] if config['kernel_size'] > 2 else [3, 5, 7][config['kernel_size']]
            layers.append(nn.Conv2d(inp_ch, out_ch, kernel_size=ks, padding=(ks - 1) // 2))
            layers.append(nn.ReLU(inplace=False))

            if config['batch_norm']:
                layers.append(nn.BatchNorm2d(out_ch))

            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            inp_ch = out_ch
        
        self.conv_layers = nn.Sequential(*layers)
        self.pooling = nn.AdaptiveAvgPool2d(1) if config['global_avg_pooling'] else nn.Identity()
        self.output_size = num_classes

        self.fc_layers = nn.ModuleList()

        inp_n = self._get_conv_output(input_shape)
        
        layers = [nn.Flatten()]
        for i in range(config['n_fc_l']):
            out_n = config['n_fc_{}'.format(i)]

            layers.append(nn.Linear(inp_n, out_n))
            layers.append(nn.ReLU(inplace=False))

            inp_n = out_n
        
        layers.append(nn.Linear(inp_n, num_classes))
        self.fc_layers = nn.Sequential(*layers)

        self.time_train = 0

    # generate input sample and forward to get shape
    def _get_conv_output(self, shape):
        bs = 1
        input = torch.autograd.Variable(torch.rand(bs, *shape))
        output_feat = self.conv_layers(input)
        output_feat = self.pooling(output_feat)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pooling(x)
        x = self.fc_layers(x)
        return x

    def train_fn(self, optimizer, criterion, loader, device):
        """
        Training method
        :param optimizer: optimization algorithm
        :param criterion: loss function
        :param loader: data loader for either training or testing set
        :param device: torch device
        :param train: boolean to indicate if training or test set is used
        :return: (accuracy, loss) on the data
        """
        accuracy = AccuracyTop1()
        self.train()

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # Step
            optimizer.zero_grad()
            logits = self(images)

            loss = criterion(logits, labels.argmax(-1))
            loss.backward()
            optimizer.step()

            acc = accuracy(labels, logits)

        return acc

    def eval_fn(self, loader, device):
        """
        Evaluation method
        :param loader: data loader for either training or testing set
        :param device: torch device
        :param train: boolean to indicate if training or test set is used
        :return: accuracy on the data
        """
        accuracy1 = AccuracyTop1()
        accuracy3 = AccuracyTop3()
        self.eval()

        with torch.no_grad():  # no gradient needed
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = self(images)
                acc1 = accuracy1(labels, outputs)
                acc3 = accuracy3(labels, outputs)

        return acc1, acc3


def evaluate_network(config, budget=None):

    budget = budget if budget else config['budget']

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    path = lambda x: str(pathlib.Path(__file__).parent.absolute().joinpath('data').joinpath(x))

    # Read train datasets
    x_train = torch.tensor(np.load(path('x_train.npy'))).float()
    x_train = x_train.permute(0, 3, 1, 2)

    y_train = torch.tensor(np.load(path('y_train.npy'))).long()

    ds_train = torch.utils.data.TensorDataset(x_train, y_train)
    ds_train = torch.utils.data.DataLoader(ds_train, batch_size=config['batch_size'], shuffle=True)

    # Read val datasets
    x_val = torch.tensor(np.load(path('x_val.npy'))).float()
    x_val = x_val.permute(0, 3, 1, 2)

    y_val = torch.tensor(np.load(path('y_val.npy'))).long()

    # Read Test datasets
    x_test = torch.tensor(np.load(path('x_test.npy'))).float()
    x_test = x_test.permute(0, 3, 1, 2)
    
    y_test = torch.tensor(np.load(path('y_test.npy'))).long()


    ds_val = torch.utils.data.TensorDataset(x_val, y_val)
    ds_val = torch.utils.data.DataLoader(ds_val, batch_size=config['batch_size'], shuffle=True)

    ds_test = torch.utils.data.TensorDataset(x_test, y_test)
    ds_test = torch.utils.data.DataLoader(ds_test, batch_size=config['batch_size'], shuffle=True)


    # Create model
    net = FashionNet(config, (1, 28, 28), 10).to(device)

    # Train
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr_init'])
    criterion = torch.nn.CrossEntropyLoss()

    t = tqdm.tqdm(total=budget)
    for epoch in range(budget):
        acc = net.train_fn(optimizer, criterion, ds_train, device)
        t.set_postfix(train_acc=acc)
        t.update()

    num_params = np.sum(p.numel() for p in net.parameters())
    
    val_acc1, val_acc3 = net.eval_fn(ds_val, device)
    tst_acc1, tst_acc3 = net.eval_fn(ds_test, device)

    t.set_postfix(
        train_acc=acc,
        val_acc=val_acc1,
        tst_acc=tst_acc1,
        len=np.log10(num_params))
    t.close()

    return {
        'val_acc_1': (-100.0 * val_acc1, 0.0),
        'val_acc_3': (-100.0 * val_acc3, 0.0),
        'tst_acc_1': (-100.0 * tst_acc1, 0.0),
        'tst_acc_3': (-100.0 * tst_acc3, 0.0),
        'num_params': (np.log10(num_params), 0.0),        
    }


def extract_num_parameters(config):
    total = 0

    s = (config['kernel_size'] * config['kernel_size'] * 3 + 1) * config['n_conv_0'] + config['batch_norm'] * config['n_conv_0'] * 2
    total += s
    s = ((config['kernel_size'] * config['kernel_size'] * config['n_conv_0'] + 1) * config[
        'n_conv_1']) + config['batch_norm'] * config['n_conv_1'] * 2 if config['n_conv_l'] > 1 else 0
    # print(s)
    total += s
    s = ((config['kernel_size'] * config['kernel_size'] * config['n_conv_1'] + 1) * config[
        'n_conv_2']) + config['batch_norm'] * config['n_conv_2'] * 2 if config['n_conv_l'] > 2 else 0
    total += s
    after_conv = {1: 64, 2: 16, 3: 4}[config['n_conv_l']] if not config['global_avg_pooling'] else 1
    after_conv *= config['n_conv_' + str(config['n_conv_l'] - 1)]

    s = config['n_fc_0'] * after_conv + config['n_fc_0']
    total += s
    s = (config['n_fc_1'] * config['n_fc_0'] + config['n_fc_1']) if config['n_fc_l'] > 1 else 0
    total += s
    s = (config['n_fc_2'] * config['n_fc_1'] + config['n_fc_2']) if config['n_fc_l'] > 2 else 0
    total += s

    s = 10 * config['n_fc_' + str(config['n_fc_l'] - 1)] + 10

    # print(s)
    total += s

    return np.log10(total)

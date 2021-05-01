import os
import uuid
import pathlib
import numpy as np
import torch

import baselines.methods.bulkandcut as bnc
from baselines.problems import get_flowers
from baselines.problems import flowers
from baselines.problems import get_fashion
from baselines.problems import fashion
from baselines import save_experiment


def get_datasets(path):
    x_train = torch.tensor(np.load(path('x_train.npy'))).float()
    x_train = x_train.permute(0, 3, 1, 2)
    y_train = torch.tensor(np.load(path('y_train.npy'))).long()

    ds_train = torch.utils.data.TensorDataset(x_train, y_train)

    x_val = torch.tensor(np.load(path('x_val.npy'))).float()
    x_val = x_val.permute(0, 3, 1, 2)
    y_val = torch.tensor(np.load(path('y_val.npy'))).long()

    ds_val = torch.utils.data.TensorDataset(x_val, y_val)


    x_test = torch.tensor(np.load(path('x_test.npy'))).float()
    x_test = x_test.permute(0, 3, 1, 2)
    y_test = torch.tensor(np.load(path('y_test.npy'))).long()

    ds_test = torch.utils.data.TensorDataset(x_test, y_test)

    return ds_train, ds_val, ds_test


if __name__ == '__main__':

    # Parameters Flowers
    input_shape = (3, 16, 16)
    num_classes = 17
    budget = 24 * 3600
    path = lambda x: str(
        pathlib.Path(flowers.__file__).
        parent.absolute().joinpath('data').joinpath(x)
    )
    experiment = get_flowers('BNC')

    # Parameters Fashion
    # input_shape = (1, 28, 28)
    # num_classes = 10
    # budget = 24 * 3600
    # path = lambda x: str(
    #     pathlib.Path(fashion.__file__).
    #     parent.absolute().joinpath('data').joinpath(x)
    # )
    # experiment = get_fashion('BNC')


    ################
    #### MOBOHB ####
    ################
    # Run a full optimization:
    ds_train, ds_val, ds_test = get_datasets(path)
    work_dir = os.path.join('bulkandcutoutput', f"{str(uuid.uuid4())}")
    evolution = bnc.Evolution(
        experiment,
        input_shape=input_shape,
        n_classes=num_classes,
        work_directory=work_dir,
        train_dataset=ds_train,
        valid_dataset=ds_val,
        test_dataset=ds_test,
        debugging=False,
        )
    evolution.run(time_budget=budget)

    save_experiment(experiment, f'{experiment.name}.pickle')
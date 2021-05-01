from datetime import datetime
from copy import deepcopy
from collections import defaultdict
from math import ceil
from typing import Dict

import numpy as np
import torch
import torchsummary
import tqdm
from ax import SearchSpace
from ax.modelbridge import get_sobol

from baselines.methods.bulkandcut.model.model_section import ModelSection
from baselines.methods.bulkandcut.model.model_head import ModelHead
from baselines.methods.bulkandcut.data_augmentation import DataAugmentation
from baselines.methods.bulkandcut.model.average_meter import AverageMeter
from baselines.methods.bulkandcut.model.cross_entropy_with_probs import CrossEntropyWithProbs
from baselines.methods.bulkandcut import rng, device


class BNCmodel(torch.nn.Module):

    @classmethod
    def LOAD(cls, file_path: str) -> "BNCmodel":
        return torch.load(f=file_path).to(device)

    @classmethod
    def NEW(cls, search_space: SearchSpace, input_shape, n_classes: int) -> "BNCmodel":
        # Sample

        parameters = get_sobol(search_space).gen(1).arms[0].parameters

        n_conv_sections = parameters['n_conv_l']
        n_linear_sections = parameters['n_fc_l']

        # Convolutional layers
        conv_sections = torch.nn.ModuleList()
        in_elements = input_shape[0]
        for i in range(n_conv_sections):
            conv_section = ModelSection.NEW(i, parameters, in_elements=in_elements, section_type="conv")
            in_elements = conv_section.out_elements
            conv_sections.append(conv_section)
        conv_sections[0].mark_as_first_section()

        #x = torch.autograd.Variable(torch.rand(1, *input_shape))
        #for conv in conv_sections:
        #    x = conv(x)
        #n_outputs = x.data.view(1, -1).size(1)


        # Fully connected (i.e. linear) layers
        linear_sections = torch.nn.ModuleList()
        for i in range(n_linear_sections):
            linear_section = ModelSection.NEW(i, parameters, in_elements=in_elements, section_type="linear")
            in_elements = linear_section.out_elements
            linear_sections.append(linear_section)

        head = ModelHead.NEW(
            in_elements=in_elements,
            out_elements=n_classes,
        )

        return BNCmodel(
            conv_sections=conv_sections,
            linear_sections=linear_sections,
            head=head,
            input_shape=input_shape,
            parameters=parameters).to(device)

    def __init__(self,
                 conv_sections: "torch.nn.ModuleList[ModelSection]",
                 linear_sections: "torch.nn.ModuleList[ModelSection]",
                 head: "ModelHead",
                 input_shape: tuple,
                 parameters: Dict):
        super(BNCmodel, self).__init__()
        self.conv_sections = conv_sections
        self.glob_av_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.linear_sections = linear_sections
        self.head = head
        self.input_shape = input_shape
        self.n_classes = head.out_elements
        self._parameters_dict = parameters

        self.data_augment = DataAugmentation(n_classes=self.n_classes)
        self.loss_func_CE_soft = CrossEntropyWithProbs().to(device)
        self.loss_func_CE_hard = torch.nn.CrossEntropyLoss().to(device)
        self.loss_func_MSE = torch.nn.MSELoss().to(device)
        self.creation_time = datetime.now()

    @property
    def n_parameters(self):
        return np.sum(par.numel() for par in self.parameters())

    @property
    def depth(self):
        n_cells = sum([len(lin_sec) for lin_sec in self.linear_sections])
        n_cells += sum([len(conv_sec) for conv_sec in self.conv_sections])
        return n_cells

    @property
    def summary(self):
        # Pytorch summary:
        model_summary = torchsummary.summary(
            model=self,
            input_data=self.input_shape,
            device=device,
            verbose=0,
            depth=5,
            )
        summary_str = str(model_summary) + "\n\n"
        # Skip connection info:
        summary_str += "Skip connections\n" + "-" * 30 + "\n"
        for cs, conv_sec in enumerate(self.conv_sections):
            summary_str += f"Convolutional section #{cs + 1}:\n"
            summary_str += conv_sec.skip_connections_summary
        for ls, lin_sec in enumerate(self.linear_sections):
            summary_str += f"Linear section #{ls + 1}:\n"
            summary_str += lin_sec.skip_connections_summary
        return summary_str

    def setup_optimizer(self, optim_config: dict):
        # self.optimizer = torch.optim.AdamW(
        #     params=self.parameters(),
        #     lr=10 ** optim_config["lr_exp"],
        #     weight_decay=10. ** optim_config["w_decay_exp"],
        #     )

        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=10 ** optim_config["lr_exp"])
        # st_size = optim_config["lr_sched_step_size"] if "lr_sched_step_size" in optim_config else 1
        # gamma = optim_config["lr_sched_gamma"] if "lr_sched_gamma" in optim_config else 1.
        # self.LR_schedule = torch.optim.lr_scheduler.StepLR(
        #    optimizer=self.optimizer,
        #    step_size=int(st_size),
        #    gamma=gamma,
        #    )

    def save(self, file_path):
        torch.save(obj=self, f=file_path)

    def forward(self, x):
        # convolutional cells
        for conv_sec in self.conv_sections:
            x = conv_sec(x)
        x = self.glob_av_pool(x)
        # flattening
        x = x.view(x.size(0), -1)
        # linear cells
        for lin_sec in self.linear_sections:
            x = lin_sec(x)
        x = self.head(x)
        return x

    def bulkup(self) -> "BNCmodel":
        new_conv_sections = deepcopy(self.conv_sections)
        new_linear_sections = deepcopy(self.linear_sections)

        new_params = deepcopy(self._parameters_dict)

        # There is a p chance of adding a convolutional cell
        if new_params['n_conv_l'] < 3 and (rng.uniform() < .7 or new_params['n_fc_l'] == 3):
            sel_section = rng.integers(low=0, high=len(self.conv_sections))
            new_conv_sections[sel_section] = self.conv_sections[sel_section].bulkup()
            new_params['n_conv_l'] += 1

            i = 0
            for sect in new_conv_sections:
                for cell in sect.cells:
                    new_params[f'n_conv_{i}'] = cell.out_elements
                    i += 1

        # And a (1-p) chance of adding a linear cell
        else:
            sel_section = rng.integers(low=0, high=len(self.linear_sections))
            new_linear_sections[sel_section] = self.linear_sections[sel_section].bulkup()

            new_params['n_fc_l'] += 1
            i = 0
            for sect in new_linear_sections:
                for cell in sect.cells:
                    new_params[f'n_fc_{i}'] = cell.out_elements
                    i += 1

        new_head = self.head.bulkup()  # just returns a copy

        return BNCmodel(
            conv_sections=new_conv_sections,
            linear_sections=new_linear_sections,
            head=new_head,
            input_shape=self.input_shape,
            parameters=new_params,
            ).to(device)

    def slimdown(self) -> "BNCmodel":
        # Prune head
        new_head, out_selected = self.head.slimdown(
            amount=rng.triangular(left=.04, right=.06, mode=.05),
            )
        # Prune linear sections
        new_linear_sections = torch.nn.ModuleList()
        for lin_sec in self.linear_sections[::-1]:
            new_linear_section, out_selected = lin_sec.slimdown(
                out_selected=out_selected,
                amount=rng.triangular(left=.065, right=.085, mode=.075),
                )
            new_linear_sections.append(new_linear_section)


        # Prune convolutional sections
        new_conv_sections = torch.nn.ModuleList()
        for conv_sec in self.conv_sections[::-1]:
            new_conv_section, out_selected = conv_sec.slimdown(
                out_selected=out_selected,
                amount=rng.triangular(left=.09, right=.11, mode=.10),
                )
            new_conv_sections.append(new_conv_section)

        new_params = deepcopy(self._parameters_dict)
        i = 0
        for sect in new_conv_sections:
            for cell in sect.cells:
                new_params[f'n_conv_{i}'] = cell.out_elements
                i += 1
        i = 0
        for sect in new_linear_sections:
            for cell in sect.cells:
                new_params[f'n_fc_{i}'] = cell.out_elements
                i += 1


        return BNCmodel(
            conv_sections=new_conv_sections[::-1],
            linear_sections=new_linear_sections[::-1],
            head=new_head,
            input_shape=self.input_shape,
            parameters=new_params,
            ).to(device)

    def start_training(self,
                       n_epochs: int,
                       train_dataset: "torch.utils.data.Dataset",
                       valid_dataset: "torch.utils.data.Dataset",
                       test_dataset: "torch.utils.data.Dataset",
                       teacher_model: "BNCmodel" = None,
                       return_all_learning_curvers: bool = False,
                       ):
        # Batch size is automatically tuned according to the available hardware.
        # The goal is to have the batch size as big as possible. First we try to fit the whole
        # dataset inside a two batches. If we run out of memory, we successively halve
        # the batch size until it fits the memory.
        batch_size = len(train_dataset) / 8.

        while True:
            # Create Dataloaders:
            train_data_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=int(ceil(batch_size)),
                shuffle=True,
                )
            valid_data_loader = torch.utils.data.DataLoader(
                dataset=valid_dataset,
                batch_size=int(ceil(batch_size)),
                shuffle=False,
                )
            test_data_loader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=int(ceil(batch_size)),
                shuffle=False,
            )

            try:
                learning_curves = self._train_n_epochs(
                    n_epochs=n_epochs,
                    train_data_loader=train_data_loader,
                    valid_data_loader=valid_data_loader,
                    test_data_loader=test_data_loader,
                    teacher_model=teacher_model,
                    return_all_learning_curvers=return_all_learning_curvers,
                )
                return learning_curves
            # Exception handling adapted from FairSeq (https://github.com/pytorch/fairseq/)
            except RuntimeError as exc:
                batch_size /= 2.
                if "out of memory" in str(exc) and batch_size >= 1:
                    print("WARNING: ran out of memory, trying again with smaller batch size:",
                          int(batch_size),
                          )
                    for p in self.parameters():
                        if p.grad is not None:
                            del p.grad
                    torch.cuda.empty_cache()
                else:
                    raise exc

    def _train_n_epochs(self,
                        n_epochs: int,
                        train_data_loader: "torch.utils.data.DataLoader",
                        valid_data_loader: "torch.utils.data.DataLoader",
                        test_data_loader: "torch.utils.data.DataLoader",
                        teacher_model: "BNCmodel",
                        return_all_learning_curvers: bool,
                        ):
        learning_curves = defaultdict(list)

        # If a parent model was provided, its logits will be used as targets (knowledge
        # distilation). In this case we are going to use a simple MSE as loss function.
        loss_func = self.loss_func_CE_soft if teacher_model is None else self.loss_func_MSE

        # Pre-training validation loss:
        print("Pre-training evaluation:")
        initial_loss, _, _ = self.evaluate(
            data_loader=valid_data_loader,
            split_name="validation",
            )
        learning_curves["validation_loss"].append(initial_loss)
        print("\n")

        for epoch in range(1, n_epochs + 1):
            train_loss = self._train_one_epoch(
                train_data_loader=train_data_loader,
                teacher_model=teacher_model,
                loss_function=loss_func,
                )

            # Register perfomance of the current epoch:
            learning_curves["train_loss"].append(train_loss)
            status_str = f"Epoch {epoch} results -- "
            # status_str += f"learning rate: {self.LR_schedule.get_last_lr()[0]:.3e}, "
            status_str += f"training loss: {learning_curves['train_loss'][-1]:.3f}, "
            if return_all_learning_curvers or epoch == n_epochs:
                # If required, I'm going to monitor all sorts of learning curves,
                # otherwise I'll measure performance just once after the last epoch.
                train_loss_at_eval, train_accuracy, _ = self.evaluate(
                    data_loader=train_data_loader,
                    split_name="training",
                    )
                valid_loss, valid_accuracy_1, valid_accuracy_3 = self.evaluate(
                    data_loader=valid_data_loader,
                    split_name="validation",
                    )
                test_loss, test_accuracy_1, test_accuracy_3 = self.evaluate(
                    data_loader=test_data_loader,
                    split_name="test",
                )
                learning_curves["train_loss_at_eval"].append(train_loss_at_eval)
                learning_curves["train_accuracy"].append(train_accuracy)
                learning_curves["validation_loss"].append(valid_loss)
                learning_curves["validation_accuracy"].append(valid_accuracy_1)
                learning_curves["validation_accuracy_3"].append(valid_accuracy_3)
                learning_curves["test_loss"].append(test_loss)
                learning_curves["test_accuracy"].append(test_accuracy_1)
                learning_curves["test_accuracy_3"].append(test_accuracy_3)

                status_str += f"validation loss: {valid_loss:.3f}, "
                status_str += f"validation accuracy: {valid_accuracy_1:.3f}"
            print(status_str + "\n")

        return learning_curves

    def _train_one_epoch(self, train_data_loader, teacher_model, loss_function):
        self.train()
        if teacher_model is not None:
            teacher_model.eval()

        batch_losses = AverageMeter()
        tqdm_ = tqdm.tqdm(train_data_loader, disable=False)
        for images, labels in tqdm_:
            batch_size = images.size(0)
            labels = labels.argmax(-1)
            # Apply data augmentation
            images, labels = self.data_augment(data=images, targets=labels)
            images = images.to(device)

            # If a teacher model was given, we use its predictions as targets,
            # otherwise we stick to the image labels.
            if teacher_model is not None:
                with torch.no_grad():
                    targets = teacher_model(images)
                targets = targets.to(device)
            else:
                targets = labels.to(device)

            # Forward- and backprop:
            self.optimizer.zero_grad()
            logits = self(images)
            loss = loss_function(input=logits, target=targets)
            loss.backward()
            self.optimizer.step()

            # Register training loss of the current batch:
            loss_value = loss.item()
            batch_losses.update(val=loss_value, n=batch_size)
            tqdm_.set_description(desc=f"Training loss: {loss_value:.3f}")

        # self.LR_schedule.step()
        return batch_losses()

    @torch.no_grad()
    def evaluate(self, data_loader, split_name):
        self.eval()

        average_loss = AverageMeter()
        average_accuracy_1 = AverageMeter()
        average_accuracy_3 = AverageMeter()
        tqdm_ = tqdm.tqdm(data_loader, disable=False)
        for images, labels in tqdm_:
            batch_size = images.size(0)

            # No data augmentation here!
            images = images.to(device)
            labels = labels.to(device)

            # Loss:
            logits = self(images)

            loss_value = self.loss_func_CE_hard(input=logits, target=labels.argmax(-1))
            average_loss.update(val=loss_value.item(), n=batch_size)

            # Top-3 accuracy:
            top3_accuracy = self._accuracy(outputs=logits, targets=labels.argmax(-1), topk=(3,))
            average_accuracy_3.update(val=top3_accuracy[0], n=batch_size)

            # Top-1 accuracy:
            top1_accuracy = self._accuracy(outputs=logits, targets=labels.argmax(-1), topk=(1,))
            average_accuracy_1.update(val=top1_accuracy[0], n=batch_size)

            tqdm_.set_description(f"Evaluating on the {split_name} split:")

        return average_loss(), average_accuracy_1(), average_accuracy_3()

    @torch.no_grad()
    def _accuracy(self, outputs, targets, topk=(1,)):
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.T
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        accuracies = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            accuracies.append(correct_k.mul_(100.0/batch_size).item())
        return accuracies

import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold   # We use 3-fold stratified cross-validation
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from baselines import nDS_index, crowdingDist
import torch.nn.functional as F


def sort_array(fit):
    index_list = np.array(list(range(len(fit))))

    a, index_return_list = nDS_index(np.array(fit), index_list)
    b, sort_index = crowdingDist(a, index_return_list)

    sorted_ = []
    for i, x in enumerate(sort_index):
        sorted_.extend(x)

    sorted_ = [sorted_.index(i) for i in range((len(fit)))]

    return sorted_



class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.fc2 = torch.nn.Linear(13, 10)
        torch.nn.init.normal_(self.fc2.weight)
        torch.nn.init.normal_(self.fc2.bias)

        self.fc3 = torch.nn.Linear(10, 2)
        torch.nn.init.normal_(self.fc3.weight)
        torch.nn.init.normal_(self.fc3.bias)

    def forward(self, x):

        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x



    def train_fn(self, train_data, num_epochs):
        """
        Training method
        :param optimizer: optimization algorithm
        """
        self.train()
        batch_size = 32

        train_loader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True)

        loss_criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)

        for epoch in range(num_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            i = 0
            for data in train_loader:
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

                inputs = torch.stack(data[0]).to(device)
                inputs = torch.transpose(inputs, 0, 1)
                inputs.type(torch.FloatTensor)

                # get the inputs; data is a list of [inputs, labels]
                y_value = torch.stack(data[1])
                y_value = torch.transpose(y_value, 0, 1)
                y_value.type(torch.FloatTensor)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                self.double()
                outputs = self(inputs)
                loss = loss_criterion(outputs, y_value)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

                if (epoch + 1) % 20 == 0:
                    print('[%d] loss: %.2f' %
                              (epoch + 1, running_loss))
                running_loss = 0.0

        return



    def predict(self, x):

        self.eval()
        self.double()

        x = [float(m) for m in x]

        train_loader = DataLoader(dataset=[x],
                                  shuffle=True)

        for d in train_loader:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            inputs = torch.stack(d).to(device)
            inputs = torch.transpose(inputs, 0, 1)

            output = self(inputs)


        return output



class Neural_Predictor:
    """
    Class to group ensamble of NN
    """

    def __init__(self, num_epochs, num_ensamble_nets):

        self.num_epochs = num_epochs
        self.num_ensamble_nets = num_ensamble_nets
        self.networks = [Net() for i in range(self.num_ensamble_nets)]
        self.all_architecture = []

    def train_models(self, x):


        for model in self.networks:
            model.train_fn(x, self.num_epochs)



    def ensamble_predict(self, x):

        predictions = [model.predict(x).tolist()[0] for model in self.networks]
        predictions = [ [- pred[0]*10, pred[1]] for pred in predictions]


        mean1 = np.mean([pred[0] for pred in predictions])
        mean2 = np.mean([pred[1] for pred in predictions])
        return [mean1, mean2], predictions



    def independent_thompson_sampling_for_mo(self, x, arches_in, num_models):

        arches = arches_in.copy()
        mean_list = []
        prediction_list = [[] for _ in range(num_models)]

        for arch in range(len(arches)):

            mean, predictions = self.ensamble_predict(x[arch])
            mean_list.append(mean)

            for i in range(num_models):
                prediction_list[i].extend([predictions[i]])

        sorted_mean = sort_array(mean_list)

        fit = []
        for i in range(num_models):
            fit.append(sort_array(prediction_list[i]))


        prob_ = []
        for i in range(len(arches_in)):
            prob1 = self.independent_thompson_sampling(sorted_mean[i], [f[i] for f in fit])
            prob_.append(prob1)

        return prob_

    def sort_pop(self,list1, list2):

        z = []
        for m in list2:
            z.append(list1[int(m)])

        return z

    def independent_thompson_sampling(self, mean, predictions_fixed):

        M = self.num_ensamble_nets
        squared_differences = np.sum([np.square(np.abs(predictions_fixed[i]) - mean) for i in range(len(predictions_fixed))])
        var = np.sqrt( (squared_differences) / (M - 1))
        prob = np.random.normal(mean, var)

        return prob

    def choose_models(self, architectures, test_data, select_models):

        architectures = architectures.copy()

        arch_lists = []
        probs = self.independent_thompson_sampling_for_mo(test_data, architectures, self.num_ensamble_nets)


        for _ in range(select_models):
            max_index = probs.index(min(probs))
            arch_lists.append(architectures[max_index])
            probs.pop(max_index)
            architectures.pop(max_index)

        return arch_lists

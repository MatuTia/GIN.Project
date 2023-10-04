from copy import deepcopy

import numpy as np
import torch.nn
import torch_geometric.nn.pool as pooling
from torch import Tensor, concat, FloatTensor
from torch.nn import ModuleList, Sigmoid, ReLU
from torch.nn.functional import binary_cross_entropy
from torch.optim import Adam
from torch_geometric.data import Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataLoader

from model import GINlayer, MLPlayer
from utils import accuracy


class NeuralNetwork(torch.nn.Module):

    def __init__(self, layer_list: list[torch.nn.Module], classifier: torch.nn.Module):
        super().__init__()
        self.layer_list = ModuleList(modules=layer_list)
        self.classifier = classifier

    def forward(self, data: BaseData) -> Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        y_list = list()

        for layer in self.layer_list:
            x = layer(x=x, edge_index=edge_index)

            # readout
            y_list.append(pooling.global_add_pool(x=x, batch=batch))

        # concat
        y = concat(tensors=y_list, dim=1)

        y = self.classifier(y)
        return Sigmoid()(y).reshape([-1])


def model_definition(train_eps: bool, hidden_size: int) -> NeuralNetwork:
    gin_list = [GINlayer(MLPlayer([37, hidden_size, 37], ReLU()), train_eps) for _ in range(5)]
    classifier = MLPlayer([37 * 5, 37, 1], ReLU())
    return NeuralNetwork(gin_list, classifier)


def model_train(model: NeuralNetwork, train_set: Dataset, val_set: Dataset, lr: float, patience: int,
                batch_size: int, device: torch.device) -> tuple[float, dict, np.ndarray]:
    max_accuracy = 0
    stop_counter = 0
    parameters = None
    accuracy_progress = list()

    val_len = len(val_set)
    val_batch = DataLoader(val_set, batch_size=val_len).__iter__().__next__().to(device)
    val_output = val_batch.y.to(device)

    train_len = len(train_set)
    train_batch = DataLoader(train_set, batch_size=train_len).__iter__().__next__().to(device)
    train_output = train_batch.y.to(device)

    optimizer = Adam(model.parameters(), lr)

    for epoch in range(1000):
        if stop_counter > patience:
            return max_accuracy, parameters, np.array(accuracy_progress)

        model.train()
        for batch in DataLoader(train_set, batch_size, shuffle=True).__iter__():
            optimizer.zero_grad()
            prediction = model(batch.to(device))
            binary_cross_entropy(prediction, batch.y.type(FloatTensor).to(device)).backward()
            optimizer.step()

        model.eval()
        train_accuracy = accuracy(model(train_batch), train_output, train_len)
        val_accuracy = accuracy(model(val_batch), val_output, val_len)
        accuracy_progress.append([train_accuracy, val_accuracy])

        if val_accuracy > max_accuracy:
            max_accuracy = val_accuracy
            parameters = deepcopy(model.state_dict())
            stop_counter = 0
        else:
            stop_counter += 1

    return max_accuracy, parameters, np.array(accuracy_progress)


def model_eval(model: torch.nn.Module, dataset: Dataset, device: torch.device) -> float:
    model.eval()
    data = DataLoader(dataset, batch_size=len(dataset)).__iter__().__next__().to(device)
    return accuracy(model(data), data.y.type(FloatTensor).to(device), len(dataset))

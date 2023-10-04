from math import sqrt

import torch.nn
import torch_geometric.nn
from torch import Tensor, empty, matmul
from torch.nn import BatchNorm1d, Parameter, ModuleList
from torch.nn.init import uniform_


class Linear(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.W = Parameter(empty((in_features, out_features)))
        self.b = Parameter(empty(out_features))

        uniform_(self.W, -1/sqrt(in_features), 1/sqrt(in_features))
        uniform_(self.b, -1/sqrt(out_features), 1/sqrt(out_features))

    def forward(self, x: Tensor) -> Tensor:
        return self.b + matmul(x, self.W)


class MLPlayer(torch.nn.Module):

    def __init__(self, channels: list[int], activation_function: "None | torch.nn.Module" = None):
        super().__init__()
        self.channels = channels
        self.activation_function = activation_function

        # neuroni interni
        self.neurons = ModuleList()
        self.batch_norms = ModuleList()

        for i in range(len(channels) - 2):
            self.neurons.append(Linear(channels[i], channels[i + 1]))
            self.batch_norms.append(BatchNorm1d(channels[i + 1]))

        # neurone output
        self.output_neuron = Linear(channels[-2], channels[-1])

    def forward(self, x: Tensor) -> Tensor:
        for neuron, norm in zip(self.neurons, self.batch_norms):
            x = neuron(x)
            x = norm(x)
            x = self.activation_function(x)

        return self.output_neuron(x)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.channels})'


class GINlayer(torch_geometric.nn.MessagePassing):

    def __init__(self, MLP: torch.nn.Module, eps: float = 0, train_eps: bool = False):
        super().__init__(aggr='sum')
        self.MLP = MLP
        self.eps = Parameter(Tensor([eps]), requires_grad=train_eps)

    def forward(self, edge_index: Tensor, x: Tensor) -> Tensor:
        y = self.propagate(edge_index, x=x)
        y += (1 + self.eps) * x
        return self.MLP(y)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.MLP})'

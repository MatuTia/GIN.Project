from itertools import product

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch import Tensor, where, as_tensor
from torch_geometric.data import Dataset


def dataset_split(dataset: Dataset, train_size: float, val_size: float) -> tuple[Dataset, Dataset, Dataset]:
    dataset_len = len(dataset)
    f_split = int(train_size * dataset_len)
    s_split = int((train_size + val_size) * dataset_len)

    train_set = dataset[:f_split]
    val_set = dataset[f_split:s_split]
    test_set = dataset[s_split:]

    return train_set, val_set, test_set


def cross_validation_index(dataset: Dataset, num_folders: int) -> list[tuple[int, int]]:
    folder_size = len(dataset) // num_folders
    return [(folder_size*folder, folder_size*(folder+1)) for folder in range(num_folders)]


def hyper_parameters_combination(*hyper_parameters: list) -> list[tuple]:
    return [element for element in product(*hyper_parameters)]


def accuracy(prediction: Tensor, correct: Tensor, tensor_len: int) -> int:
    prediction = where(as_tensor(prediction > 0.5), 1, 0)
    return where(as_tensor(prediction == correct), 1, 0).sum().item() / tensor_len


def plot_accuracy_progress(accuracy_progress: np.ndarray, test: float):
    sns.set_style(style="whitegrid")
    train_accuracy = accuracy_progress[:, 0]
    val_accuracy = accuracy_progress[:, 1]

    x = range(len(train_accuracy))
    plt.plot(x, train_accuracy, label="Training")
    plt.plot(x, val_accuracy, label="Validation")
    plt.hlines(test, xmin=0, xmax=len(x), colors="purple", linestyles="--", label="Test")
    plt.ylim(-0.1, 1.1)

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc="lower right")
    sns.despine(left=True, bottom=True)
    plt.savefig(f'../AccuracyProgress.svg', format='svg', transparent=True)

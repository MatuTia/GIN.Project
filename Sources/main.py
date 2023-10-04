import os
import time
import logging

import torch
from torch import manual_seed, cuda, use_deterministic_algorithms
from torch_geometric.datasets.tu_dataset import TUDataset

from model_validation import model_selection, model_assessment
from utils import hyper_parameters_combination, dataset_split, cross_validation_index, plot_accuracy_progress

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def main(require_deterministic: bool):

    start = time.time()
    device = torch.device('cuda' if cuda.is_available() else 'cpu')
    logging.basicConfig(filename=f'../log.txt', level=logging.INFO, format="%(asctime)s %(message)s")
    logging.info(f'Run on {device}')

    if require_deterministic:
        manual_seed(42)
        use_deterministic_algorithms(True)

    dataset = TUDataset(root='/tmp/NCI1', name='NCI1').shuffle()

    list_patience = [10, 20, 40]
    list_hidden_size = [16, 37]
    list_train_eps = [False, True]
    list_lr = [0.01, 0.005]
    combination = hyper_parameters_combination(list_patience, list_hidden_size, list_train_eps, list_lr)
    train_set, val_set, test_set = dataset_split(dataset, 0.66, 0.17)
    cross_validation = cross_validation_index(train_set, 6)

    patience, hidden_size, train_eps, lr = model_selection(train_set, cross_validation, combination, device)

    logging.info(f'Best configuration of hyper-parameter has patience: {patience}, hidden_size: {hidden_size},'
                 f'train_eps: {train_eps}, lr: {lr}')

    test_accuracy, accuracy_progress = model_assessment(train_set, val_set, test_set, patience,
                                                        hidden_size, train_eps, lr, device)

    plot_accuracy_progress(accuracy_progress, test_accuracy)

    end = time.time()
    logging.info(f'Execution time: {end - start}')

    logging.shutdown()


if __name__ == '__main__':
    main(True)

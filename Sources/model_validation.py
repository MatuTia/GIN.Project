import logging

import numpy as np
import torch
from torch_geometric.data import Dataset

from nn import model_definition, model_train, model_eval


def model_selection(train_set: Dataset, cross_validation: list[tuple[int, int]], combination: list[tuple],
                    device: torch.device) -> tuple:

    max_accuracy = 0
    best_combination = None

    for patience, hidden_size, train_eps, lr in combination:
        logging.info(f'Configuration: patience {int(patience)}, hidden_size {hidden_size}, '
                     f'train_eps {bool(train_eps)}, lr {lr:.4f}')

        accuracy = 0

        for left, right in cross_validation:
            val_folder = train_set[left:right]
            train_folder = train_set[:left] + train_set[right:]

            logging.info("New folder")

            model = model_definition(train_eps, hidden_size).to(device)
            folder_accuracy = model_train(model, train_folder, val_folder, lr, patience, 113, device)[0]
            logging.info(f'Accuracy on folder {folder_accuracy}')
            accuracy += folder_accuracy

        logging.info(f'Configuration Accuracy: {accuracy / len(cross_validation):.4f}')

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_combination = (patience, hidden_size, train_eps, lr)

    logging.info(f'Best configuration Accuracy: {max_accuracy / len(cross_validation):.4f}')
    return best_combination


def model_assessment(train_set: Dataset, val_set: Dataset, test_set: Dataset, patience: int, hidden_size: int,
                     train_eps: bool, lr: float, device: torch.device) -> tuple[float, np.ndarray]:

    model = model_definition(train_eps, hidden_size).to(device)
    val_accuracy, parameters, accuracy_progress = model_train(model, train_set, val_set, lr, patience, 113, device)

    model.load_state_dict(parameters)
    test_accuracy = model_eval(model, test_set, device)
    logging.info(f'Final network validation_accuracy {val_accuracy:.4f}, test_accuracy {test_accuracy:.4f}')
    return test_accuracy, accuracy_progress

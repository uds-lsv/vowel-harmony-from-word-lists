from cltoolkit.models import Form
from copy import deepcopy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Tuple

from eff.util import constants


def generate_batch(batch: List[Tuple[Tensor, Tensor, Tensor]]) \
    -> Tuple[List[Tensor], Tensor, Tensor]:
    """ Pads sequences to the same length and moves them to GPU/CPU. 
    Called by the PyTorch DataLoader.

    Returns
    -------
    Tuple[List[Tensor], Tensor, Tensor]
        The padded batch
    """
    target_indices = [item[0].to(constants.device) for item in batch]
    x = [item[1] for item in batch]
    y = [item[2] for item in batch]
    x_padded = pad_sequence(x)
    y_padded = pad_sequence(y)
    # for some reason the padded tensors are of the shape seq_len*batch_size,
    # therefore transpose is applied.
    return target_indices, x_padded.T.to(constants.device), \
         y_padded.T.to(constants.device)


# def generate_batch_cpu(batch: List[Tuple[Tensor, Tensor, Tensor]]) \
#     -> Tuple[List[Tensor], Tensor, Tensor]:
#     target_indices = [item[0].to(constants.device) for item in batch]
#     x = [item[1] for item in batch]
#     y = [item[2] for item in batch]
#     x_padded = pad_sequence(x)
#     y_padded = pad_sequence(y)
#     # for some reason the padded tensors are of the shape seq_len*batch_size,
#     # therefore transpose is applied.
#     return target_indices, x_padded.T, \
#          y_padded.T


def get_train_test_split(words: List[str], test_size=0.3) -> Tuple[List[str], List[str]]:
    """ Wrapper of the scipy function. Also sets a random seed.

    Parameters
    ----------
    words : List[str]
        List of sequences to be split
    test_size : float, optional
        Size of the test set, by default 0.3

    Returns
    -------
    Tuple[List[str], List[str]]
        The train and test set
    """
    words_copy = deepcopy(words)
    np.random.RandomState(constants.random_seed).shuffle(words_copy)
    return train_test_split(words_copy, test_size=test_size, shuffle=False)


def get_train_test_valid_split(words: List[str], test_size=0.3, valid_size=0.1) -> \
    Tuple[List[str], List[str], List[str]]:
    """ Same as get_train_test_split, but now the train set is split in a train set proper
        and a validation set comprising 10% of the train data.

    Parameters
    ----------
    words : List[str]
        List of sequences to be split
    test_size : float, optional
        Size of the test set, by default 0.3
    valid_size : float, optional
        Size of the validtion set as a fraction of the train set, by default 0.1

    Returns
    -------
    Tuple[List[str], List[str], List[str]]
        
    """
    words_copy = deepcopy(words)
    np.random.RandomState(constants.random_seed).shuffle(words_copy)
    train_valid, test = train_test_split(words_copy, test_size=test_size, shuffle=False)
    train, valid = train_test_split(train_valid, test_size=valid_size, shuffle=False)
    return train, valid, test


def get_class_weights(class_weight: Dict[int, int], ignore_classes: List[int], 
    classes: List[int], y: List[int]) -> Tensor:
    """ Computes class weights for loss function with sklearn, for the case that a phoneme is not present
        in the labels (CrossEntropyLoss requires that all labels in the test set are also present in the
        train set).

    Parameters
    ----------
    class_weight : Dict[int, int]
        Label counts
    ignore_classes : List[int]
        Classes not in the output distribution
    classes : List[int]
        List of all classes
    y : List[int]
        List of classes that actually are in the labels

    Returns
    -------
    Tensor
        Final class weights
    """
    weights = compute_class_weight(class_weight=class_weight, classes=np.array(classes), \
        y=np.array(y))
    weights = [w if idx not in ignore_classes else 0. for idx, w in enumerate(weights)]
    return Tensor(weights).to(constants.device)


def get_class_weights_balanced(ignore_classes: List[int], classes: List[int], y: List[int]) \
    -> Tensor:
    """ Same as above, but with balanced class weights (see docuementation of the sklearn Function). """
    weights = compute_class_weight(class_weight='balanced', classes=np.array(classes), \
        y=np.array(y))
    weights = [w if idx not in ignore_classes else 0. for idx, w in enumerate(weights)]
    return Tensor(weights).to(constants.device)


def get_class_weights_uniform(ignore_classes: List[int], classes: List[int], y: List[int]) \
    -> Tensor:
    """ Same as above, but with uniform class weights. """
    weights = compute_class_weight(class_weight=None, classes=np.array(classes), y=np.array(y))
    weights = [w if idx not in ignore_classes else 0. for idx, w in enumerate(weights)]
    return Tensor(weights).to(constants.device)
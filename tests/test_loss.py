import pytest

from collections import Counter
import torch
from torch.nn import CrossEntropyLoss
import numpy as np

from eff.train import get_class_weights, get_class_weights_balanced, get_class_weights_uniform
from eff.util import constants


torch.manual_seed(0)


n_classes = 20
n_samples = 100
classes = range(n_classes)
pad_idx = classes[0]
eos_idx = classes[1]
mask_idx = classes[2]


@pytest.fixture
def dummy_targets():
    targets = np.random.randint(0, n_classes, size=10)
    for cla in classes:
        if cla not in targets:
            targets = np.append(targets, cla)
    print(targets)
    return targets


@pytest.fixture
def class_weights_balanced_ignore(dummy_targets):
    print(dummy_targets)
    return get_class_weights_balanced(ignore_classes=[pad_idx, mask_idx], classes=classes, \
         y=dummy_targets)


@pytest.fixture
def class_weights_balanced_all(dummy_targets): 
    return get_class_weights_balanced(ignore_classes=[], classes=classes, y=dummy_targets)


@pytest.fixture
def class_weights_uniform_ignore(dummy_targets):
    return get_class_weights_uniform(ignore_classes=[pad_idx, mask_idx], classes=classes, \
        y=dummy_targets)


@pytest.fixture
def class_weights_uniform_all(dummy_targets): 
    return get_class_weights_uniform(ignore_classes=[], classes=classes, y=dummy_targets)


@pytest.fixture
def class_weights_ignore(dummy_targets):
    class_weight = Counter(dummy_targets)
    return get_class_weights(class_weight=class_weight, ignore_classes=[pad_idx, mask_idx], \
          classes=classes, y=dummy_targets)


@pytest.fixture
def class_weights_all(dummy_targets):
    class_weight = Counter(dummy_targets)
    return get_class_weights(class_weight=class_weight, ignore_classes=[], classes=classes, \
         y=dummy_targets)


@pytest.fixture
def weighted_ce_loss_balanced_ignore(class_weights_balanced_ignore):
    # return WeightedCELoss(weight=class_weights_ignore, eos_idx=eos_idx)
    return CrossEntropyLoss(weight=class_weights_balanced_ignore)


@pytest.fixture
def weighted_ce_loss_balanced_all(class_weights_balanced_all):
    # return WeightedCELoss(weight=class_weights_all, eos_idx=eos_idx)
    return CrossEntropyLoss(weight=class_weights_balanced_all)


def test_class_weights_balanced_ignore(class_weights_balanced_ignore):
    weights = class_weights_balanced_ignore
    assert weights[pad_idx] == 0.
    assert weights[mask_idx] == 0.


def test_class_weights_balanced_all(class_weights_balanced_all):
    weights = class_weights_balanced_all
    assert weights[pad_idx] != 0.
    assert weights[mask_idx] != 0.


def test_class_weights_uniform_ignore(class_weights_uniform_ignore):
    weights = class_weights_uniform_ignore
    assert weights[pad_idx] == 0.
    assert weights[mask_idx] == 0.


def test_class_weights_uniform_all(class_weights_uniform_all):
    weights = class_weights_uniform_all
    assert weights[pad_idx] == 1.
    assert weights[mask_idx] == 1.


def test_class_weights_ignore(class_weights_ignore):
    weights = class_weights_ignore
    assert weights[pad_idx] == 0.
    assert weights[mask_idx] == 0.


def test_class_weights_all(class_weights_all):
    weights = class_weights_all
    assert weights[pad_idx] != 0.
    assert weights[mask_idx] != 0.


def test_ce_loss(weighted_ce_loss_balanced_ignore, weighted_ce_loss_balanced_all):
    target = torch.randint(0,n_classes,(8,10)).to(constants.device) # 8 items of length 10
    logits = torch.randn(8,10,n_classes).to(constants.device) # 8 items of length 10 for 20 classes
    print(target.shape)
    print(logits.shape)
    target_flat = target.view(-1) # CELoss needs a 1D target tensor
    logits_flat = logits.flatten(end_dim=1) # cat sequences to a single sequence
    print(target_flat.shape)
    print(logits_flat.shape)
    loss1 = weighted_ce_loss_balanced_ignore(logits_flat, target_flat).item()
    loss2 = weighted_ce_loss_balanced_all(logits_flat, target_flat).item()
    assert loss1 != loss2
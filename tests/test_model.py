import pytest

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from eff.model.lstm import LstmLM
from eff.train import get_class_weights_balanced
from eff.util import constants


torch.manual_seed(0)


n_classes = 20
n_samples = 100
classes = list(range(n_classes))
pad_idx = classes[0]
eos_idx = classes[1]
mask_idx = classes[2]


def test_nlm():
    input = torch.randint(0,n_classes,(8,10)).to(constants.device)
    target = torch.randint(0,n_classes,(8,10)).to(constants.device) # 8 items of length 10
    target_flat = target.flatten(end_dim=1)
    train_labels = list(set(target_flat.cpu().tolist()))  
    missing_labels = list(set(range(n_classes)).difference(set(train_labels)))
    train_labels = train_labels + missing_labels
    class_weight = get_class_weights_balanced(ignore_classes=[pad_idx, eos_idx], classes=classes, \
         y=train_labels)
    criterion = CrossEntropyLoss(weight=class_weight)
    model = LstmLM(
        input_dim=n_classes,
        output_dim=n_classes,
        embedding_dim=64,
        hidden_dim=256,
        dropout=0.33,
        n_layers=2,
        loss_fn=criterion
    )
    model.to(constants.device)
    optimizer = Adam(model.parameters())
    output = model(input)
    output_flat = output.flatten(end_dim=1)
    loss = criterion(output_flat, target_flat)
    loss.backward()
    optimizer.step()
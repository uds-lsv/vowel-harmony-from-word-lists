from copy import deepcopy
import numpy as np
import torch
from torch import Tensor
from typing import List, Tuple

from eff.analysis import log_normalized_logits

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


def log_epoch(epoch, epoch_loss, perplexity):
    print("{}\t{}\t{}\t".format(
        epoch, round(epoch_loss, 4), round(perplexity, 2)
))


def check_converged(curr_loss, prev_loss, delta_loss=0.001):
    diff = prev_loss-curr_loss
    # print("{} > {}".format(diff, delta_loss))
    return np.abs(diff) <= delta_loss, diff < 0


def valid_batch(x: Tensor, 
                y: Tensor, 
                model: torch.nn.Module, 
                criterion: torch.nn.Module) -> Tuple[Tensor, Tensor, float]:
    model.eval()
    logits = model(x)
    logits_flat = logits.flatten(end_dim=1)
    y_flat = y.flatten()
    loss = criterion(logits_flat, y_flat)
    # return logits_flat, y_flat, loss.item()

    return logits, y, loss.item()


def train_batch(x: Tensor, 
                y: Tensor, 
                model: torch.nn.Module, 
                optimizer: torch.optim.Optimizer, 
                criterion: torch.nn.Module) -> None:
    model.train()
    optimizer.zero_grad()
    logits = model(x)
    logits_flat = logits.flatten(end_dim=1)
    y_flat = y.flatten()
    loss = criterion(logits_flat, y_flat)
    loss.backward()
    optimizer.step()
    # return logits_flat, y_flat, loss.item()


def validate(model: torch.nn.Module, 
             valid_loader: torch.utils.data.DataLoader, 
             criterion: torch.nn.Module) -> float:
    model.eval()
    with torch.no_grad():
        running_loss = 0
        # perplexity = 0
        nitems = 0
        for _, x, y in valid_loader:
            bsize = x.shape[0]
            nitems +=bsize
            logits_flat, y_flat, loss = valid_batch(x, y, model, criterion)            
            running_loss += loss # since reduction='mean'
            # perplexity += batch_perplexity
    valid_loss = running_loss/nitems
    # valid_perplexity = perplexity/nitems
    return valid_loss


def test(model: torch.nn.Module, 
         test_loader: torch.utils.data.DataLoader,
         criterion: torch.nn.Module) -> Tuple[List[List[float]], List[int]]:
    with torch.no_grad():
        running_loss = 0
        # perplexity = 0
        nitems = 0
        logprobs = []
        targets = []
        target_indices = []
        for indices, x, y in test_loader:
            bsize = x.shape[0]
            nitems += bsize
            logits_flat, y_flat, loss = valid_batch(x, y, model, criterion)
            # logprobs += batch_logprobs
            targets += y_flat.cpu().tolist()
            logprobs += torch.abs(log_normalized_logits(logits_flat)).cpu().tolist()
            running_loss += loss
            # perplexity += batch_perplexity
            # print(indices)
            target_indices += [idx.cpu().tolist() for idx in indices]
    test_loss = running_loss/nitems
    # test_perplexity = perplexity/nitems
    print("Test loss: {}\nTest perplexity: {}".format(test_loss, \
        0))
    return logprobs, target_indices, targets


def train(model: torch.nn.Module, 
          train_loader: torch.utils.data.DataLoader, 
          valid_loader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module, 
          max_epochs=100,
          patience=5):

    converged = False
    patience_lost = 0
    n_epochs = 0
    best_epoch = 0
    prev_valid_loss = np.inf
    best_valid_loss = np.inf
    best_model_state = deepcopy(model.state_dict())

    print("Epoch\tLoss\tPerplexity")
    
    while not converged and n_epochs < max_epochs:
        for _, x, y in train_loader:
            train_batch(x, y, model, optimizer, criterion)
            
        valid_loss = validate(model, valid_loader, criterion)  

        # save model if best valid loss achieved
        if valid_loss < best_valid_loss:
            best_model_state = deepcopy(model.state_dict())
            best_epoch = n_epochs + 1
            best_valid_loss = valid_loss
        elif valid_loss > best_valid_loss:
            model.load_state_dict(best_model_state)

        # check if converged or max epochs reached
        loss_below_delta, loss_increased = check_converged(valid_loss, prev_valid_loss)

        if loss_below_delta and not loss_below_delta:
            patience_lost += 1
            print("\tPatience lost, remaining patience: {}".format(patience-patience_lost))
        converged = (patience_lost - patience == 0) or loss_increased

        # converged = check_converged(valid_perplexity, prev_valid_perplexity)
        n_epochs += 1
        # save average epoch loss for next iteration
        # prev_valid_perplexity = valid_perplexity
        prev_valid_loss = valid_loss
        log_epoch(n_epochs, valid_loss, 0)
    print("Best epoch: {}, best valid loss: {}".format(best_epoch, round(best_valid_loss, 2)))
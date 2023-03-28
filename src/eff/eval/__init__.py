import numpy as np
import torch
import torch.nn.functional as F
from typing import List


# def check_normalized(probs: List[float], epsilon=1e-05) -> None:
#     S = sum(probs)
#     delta = np.abs(1-S)
#     if delta > epsilon:
#         raise Exception("Probabilities not normalized, delta={}".format(delta))


# def normalize_logits(logits):
#     normalized = F.softmax(logits[0], dim=-1) # remove batch dimension
#     for probs in normalized:
#         check_normalized(probs.cpu())
#     return normalized


# def get_perplexity(indices, logits):
#     probs = F.softmax(logits, dim=-1)
#     N = len(indices)
#     eos_idx = ((indices == 2).nonzero())
#     eos_idx = eos_idx.item()
#     for d in probs:
#         check_normalized(d.cpu().tolist())
#     indices = indices[None, :]
#     true_probs = torch.gather(probs, -1, indices)
#     true_probs = true_probs[0][:eos_idx]
#     P_true = np.prod(true_probs.cpu().tolist())
#     pp = 2**((-1/N)*np.log2(P_true))
#     return pp
    # return 2**(-1*sum([1/len(probs)*np.log2(prob) for prob in probs]))

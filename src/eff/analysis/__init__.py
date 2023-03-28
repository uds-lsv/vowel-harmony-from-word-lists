import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon
import torch
import torch.nn.functional as F
from typing import List, Tuple


def normalize_logits(logits: torch.Tensor) -> torch.Tensor:
    """ Normalizes logits to a probability distribution over vowels + mask - stop symbol along the last dimension.

    Parameters
    ----------
    logits : torch.Tensor
        Logits saved in the test step
    Returns
    -------
    torch.Tensor
        Normalized logits
    """
    normalized = F.softmax(logits, dim=2)
    return normalized


def log_normalized_logits(logits: torch.Tensor) -> torch.Tensor:
    """ Applies logarithm with basis two to the normalized logits, i. e. turning
        turning them into surprisal values.

    Parameters
    ----------
    logits : torch.Tensor
        Normalized logits

    Returns
    -------
    torch.Tensor
        Surprisal values
    """
    normalized = normalize_logits(logits)
    log_normalized = torch.log2(normalized)
    return log_normalized


def surprisal(logprobs: List[List[float]], 
              targets: List[List[int]], 
              target_indices: List[List[int]],
              ignore_vowel_index: int) -> List[float]:
    """ Find logprobs of target vowel indices,

    Parameters
    ----------
    logprobs : List[List[float]]
        Negative log-likelihoods of all indices in the corpus.
    targets : List[List[int]]
        Target (char) indices, used to index the second level of logprobs.
    target_indices : List[List[int]]
        Indices of the vowels. Used to index the first level of logprobs
    ignore_vowel_index : int
        Ignore this vowel index (e. g. the first one since it doesn't follow vowel harmony).

    Returns
    -------
    List[float]
        List of surprisal values
    """

    surprisals = []
    for ids, trg, lpr in zip(target_indices, targets, logprobs):
        # print(ids, trg)
        for pos, idx in enumerate(ids):
            if pos != ignore_vowel_index:
                # print("here", pos, idx)
                char_idx = trg[idx]
                surprisals.append(lpr[idx][char_idx])
    return surprisals



# def word_surprisal(logprobs: List[List[float]], 
#                    targets: List[List[int]], 
#                    target_indices: List[List[int]],
#                    ignore_vowel_index: int) -> List[float]:
#     surprisals = []
#     for ids, trg, lpr in zip(target_indices, targets, logprobs):
#         surprisals.append([])
#         for pos, idx in enumerate(ids):
#             if pos != ignore_vowel_index:
#                 char_idx = trg[idx]
#                 surprisals[-1].append(lpr[idx][char_idx])
#     return surprisals


# def positional_surprisal(logprobs: List[List[float]], 
#                          targets: List[List[int]],
#                          position: List[int]) -> List[float]:
#     """ Calculates surprisal on the specified positions. 
#         Used for replacment experiments & Feature surprisal.

#     Parameters
#     ----------
#     logprobs : List[List[float]]
#         Negative log probabilities by word, position
#     targets : List[List[int]]
#         Target char indices by word, position
#     position : List[int]
#         Positions at which surprisal is calculated

#     Returns
#     -------
#     List[float]
#         [description]
#     """
#     surprisals = []
#     for pos, trg, lpr in zip(position, targets, logprobs):
#         char_idx = trg[pos]
#         spr = lpr[pos][char_idx]
#         surprisals.append(spr)
#     return surprisals


def mann_whitney_test(x: List[float], y:List[float]) -> Tuple[float, float, float]:
    """ Performs a Mann-Whitney U-test on the input distributions. 
        Also calculates rank-biserial coefficient as f - (1 - f), f = U1/(n1*n2).

    Parameters
    ----------
    x : List[float]
        The first distribution.
    y : List[float]
        The second distribution.

    Returns
    -------
    Tuple[float, float, float]
        test statistic, p-value, effect size
    """
    n1 = len(x)
    n2 = len(y)
    U1, p_value = mannwhitneyu(x, y)
    # f is the common language effect size
    f = U1/(n1*n2) 
    # u = 1 - f, unexplained difference in ranks
    r = f - (1-f)
    return U1, p_value, r


def wilcoxon_signed_rank_test(x: List[float], y: List[float]) -> Tuple[float, float, float]:
    """ Performs a Wilcoxon signed-rank test on the input distributions.
        Should also calculate effect size as T/S, S being the sum of ranks.

    Parameters
    ----------
    x : List[float]
        The first distribution.
    y : List[float]
        The second distribution.

    Returns
    -------
    Tuple[float, float, float]
        test statistic, p-value, effect size
    """
    assert len(x) == len(y), "Expected y to be of length {}, is {}!".format(len(x), len(y))
    T, p_value = wilcoxon(x, y, )
    r = T/sum(range(1, len(x)+1))
    return T, p_value, r


def get_significance_level(p_value: float) -> str:
    """ Symbolizes significance level as 
            "ns" -> not significant,
            "*"  -> p < 0.05,
            "**" -> p < 0.01 

    Parameters
    ----------
    p_value : float

    Returns
    -------
    str
        the significance level
    """

    significane_level = None
    if p_value < 0.01:
        significane_level = r"$\bf{**}$"
    elif p_value < 0.05:
        significane_level = r"$\bf{*}$"
    else:
        significane_level = r"$\bf{ns}$"
    return significane_level


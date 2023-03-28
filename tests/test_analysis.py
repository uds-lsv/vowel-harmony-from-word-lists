import pytest

import numpy as np
import torch

from eff.analysis import normalize_logits, log_normalized_logits
from eff.data import get_word_list
from eff.data.dataset import CLTSDataset
from eff.train.dataset import ConsonantMaskingTestSet

batch_size = 8
seq_length = 10
n_classes = 20

word_list = get_word_list('northeuralex')


@pytest.fixture
def lang_fin():
    return word_list.languages[0]


@pytest.fixture
def clts_dataset(lang_fin):
    return CLTSDataset(lang_fin.forms)


@pytest.fixture
def consonant_masking_test_dataset(clts_dataset):
    return ConsonantMaskingTestSet(
        words=clts_dataset.words,
        input_alphabet=clts_dataset.input_alphabet,
        output_alphabet=clts_dataset.output_alphabet,
        bipa=clts_dataset.bipa, 
    )


@pytest.fixture
def logits_flat():
    return torch.randn(batch_size,seq_length,n_classes)


def test_normalize_logits(logits_flat):
    """ Test if logits are normalized properly. """
    normalized = normalize_logits(logits_flat)
    assert torch.sum(torch.sum(normalized, dim=2)) == batch_size * seq_length


def test_log_normalized_logits(logits_flat):
    """ Ensures that logits are normalized properly. """
    log_normalized_test = log_normalized_logits(logits_flat)
    for seq_logporbs in log_normalized_test:
        for phoneme_logprobs in seq_logporbs:
            assert round(sum([2**logprob for logprob in phoneme_logprobs]).item(),4) == 1.


# @pytest.mark.parametrize("targets,len_expected", 
#     [
#         ([1, 13, 1, 11, 1, 9, 2, 0, 0, 0], 2)
#     ]
# )
# def test_surprisal(clts_dataset, targets, len_expected):
#     np.random.seed(0)
#     alphabet = clts_dataset.output_alphabet
#     bipa = clts_dataset.bipa
#     logprobs = np.abs(np.random.randn(seq_length, len(alphabet))).tolist()
#     # print([alphabet.idx2char(idx) for idx in targets])
#     # print(targets)
#     spr = surprisal(logprobs, targets, alphabet, bipa)
#     assert len(targets) == len(logprobs)
#     assert len(spr) == len_expected


# def test_test_statistics():
#     np.random.seed(0)
#     data = dict(
#         V=np.random.normal(2, 1.5, 100),
#         C=np.random.normal(1.5, 1, 100),
#         E=np.random.normal(1.7, 2, 100)
#     )
#     res = test_statistics(data)
#     print(res)
#     assert False
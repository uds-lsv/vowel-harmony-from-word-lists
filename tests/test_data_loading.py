import pytest

from torch.utils.data import DataLoader

from eff.data import get_word_list
from eff.data.dataset import CLTSDataset
from eff.train import generate_batch, get_train_test_split, get_train_test_valid_split
from eff.train.dataset import TrainDataset


word_list = get_word_list('northeuralex')


@pytest.fixture
def lang_fin():
    return word_list.languages[0]


@pytest.fixture
def clts_dataset(lang_fin):
    return CLTSDataset(lang_fin.forms)


@pytest.fixture
def train_dataset(clts_dataset):
    return TrainDataset(
        words=clts_dataset.words,
        input_alphabet=clts_dataset.input_alphabet,
        output_alphabet=clts_dataset.output_alphabet,
        bipa=clts_dataset.bipa, 
        masking=0.25
        )


def test_train_test_split(clts_dataset):
    """ Tests train/test splitting a DictTuple of Form ojects directly. It might be better to
        preprocess the forms first in a separate class since we have to cast the DictTuple
        to a list, which may have side effects.
    """
    train1, test1 = get_train_test_split(clts_dataset.words)
    train2, test2 = get_train_test_split(clts_dataset.words)
    assert train1 == train2
    assert test1 == test2


def test_train_test_valid_split(clts_dataset):
    train1, valid1, test1 = get_train_test_valid_split(clts_dataset.words, \
        test_size=0.3, valid_size=0.1)
    train2, valid2, test2 = get_train_test_valid_split(clts_dataset.words, \
        test_size=0.3, valid_size=0.1)
    assert train1 == train2
    assert valid1 == valid2
    assert test1 == test2


def test_generate_batch(train_dataset):
    batch_size = 10
    batch = []
    max_len = 0
    for idx in range(batch_size):
        idx, x, y = train_dataset[idx]
        batch.append(tuple((idx, x ,y)))
        if len(x) > max_len:
            max_len = len(x)
    batch = generate_batch(batch)
    assert batch[1].shape[0] == batch_size
    assert batch[1].shape[1] == max_len


def test_generate_batch_dataloader(train_dataset):
    train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=generate_batch)
    for _ in train_loader:
        continue

from collections import defaultdict
from pathlib import Path
import pytest
from torch.nn.modules.loss import CrossEntropyLoss

from eff.data import get_word_list
from eff.data.dataset import CLTSDataset
from eff.model.lstm import LstmLM
from eff.util.util import save_results, load_results


word_list = get_word_list('northeuralex')

base_path = Path("./out/test")


@pytest.fixture
def lang_fin():
    return word_list.languages[0]

@pytest.fixture
def clts_dataset(lang_fin):
    return CLTSDataset(lang_fin.forms)


def test_save_load_results(clts_dataset):
    clts_ds = clts_dataset
    result = defaultdict(lambda: {})
    for k1 in ['key1', 'key2', 'key3']:
        for k2 in ['key4', 'key5', 'key6']:
            result[k1][k2] = 0
    criterion = CrossEntropyLoss()
    model = LstmLM(
        input_dim=10,
        output_dim=10,
        embedding_dim=64,
        hidden_dim=256,
        dropout=0.33,
        n_layers=2,
        loss_fn=criterion
    )
    save_results(base_path, "test", clts_ds, result, criterion, model)
    clts_ds_loaded, result_loaded, _, _ = load_results(base_path, "test")
    assert result == result_loaded
    assert clts_ds == clts_ds_loaded


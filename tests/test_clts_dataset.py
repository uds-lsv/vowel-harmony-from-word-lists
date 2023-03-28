import pytest
from pathlib import Path

from eff.data import get_word_list
from eff.data.dataset import CLTSDataset


word_list = get_word_list('northeuralex')


@pytest.fixture
def lang_fin():
    return word_list.languages[0]

@pytest.fixture
def clts_dataset(lang_fin):
    return CLTSDataset(lang_fin.forms)


@pytest.mark.parametrize("form_id,expected", 
    [
        (94, "sɑtɛŋkɑri"), # sɑtɛːŋkɑːri
        (373, "lɑtikɔ"), # lɑːtikːɔ 
        (390, "kɔwkʊ"), # kɔwkːʊ
        (307, "ɛtæisys"), # ɛtæi̯syːs
        (890, "ɔlɑkirɛisæn"), # ɔlːɑ_kiːrɛi̯sːæːn, _ is word boundary.
    ]
)
def test_preprocessing(clts_dataset, lang_fin, form_id, expected):
    """ Test if all long and half-long segments are converted to their short version. """
    form = lang_fin.forms[form_id]
    preprocessed = clts_dataset._preprocess_word(form)
    assert ''.join(s for s in preprocessed) == expected
    

@pytest.mark.parametrize("sequences,unique",
    [
     ([list("silmæ"), list("silmæ"), list("siltæ"), list("silmæsæ")], [list("silmæsæ"), list("siltæ")]),
     ([['i', 'p', 'sʰ', 'u', 'l'], ['sʰ', 'a', 'l', 'g', 'a', 'c', 'c'], ['sʰ', 'o', 'n', 'dʰ', 'o', 'p̚']],
       [['i', 'p', 'sʰ', 'u', 'l'], ['sʰ', 'a', 'l', 'g', 'a', 'c', 'c'], ['sʰ', 'o', 'n', 'dʰ', 'o', 'p̚']])
    ]
)
def test_unify_sequences(clts_dataset, sequences, unique):
    unified = clts_dataset._unify_sequences(sequences)
    print(unified)
    assert unique == unified
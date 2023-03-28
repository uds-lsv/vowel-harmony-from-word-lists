import pytest

from eff.data import get_word_list
from eff.data.dataset import CLTSDataset
from eff.train.dataset import ConsonantMaskingTestSet, TrainDataset, UnmaskedTestSet, \
    VowelMaskingTestSet
from eff.util import constants

test_forms = [94, 373, 390, 307, 890]

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
        masking=0.25,
        )


@pytest.fixture
def unmasked_test_dataset(clts_dataset):
    return UnmaskedTestSet(
        words=clts_dataset.words,
        input_alphabet=clts_dataset.input_alphabet,
        output_alphabet=clts_dataset.output_alphabet,
        bipa=clts_dataset.bipa, 
    )


@pytest.fixture
def consonant_masking_test_dataset(clts_dataset):
    return ConsonantMaskingTestSet(
        words=clts_dataset.words,
        input_alphabet=clts_dataset.input_alphabet,
        output_alphabet=clts_dataset.output_alphabet,
        bipa=clts_dataset.bipa, 
    )


@pytest.fixture
def vowel_masking_test_dataset(clts_dataset):
    return VowelMaskingTestSet(
        words=clts_dataset.words,
        input_alphabet=clts_dataset.input_alphabet,
        output_alphabet=clts_dataset.output_alphabet,
        bipa=clts_dataset.bipa, 
    )


@pytest.mark.parametrize("form_id", test_forms)
def test_train_input_output_size(train_dataset, clts_dataset, lang_fin, form_id):
    """ Test if input and output tensors are of equal size. """
    form = lang_fin.forms[form_id]
    word = clts_dataset._preprocess_word(form)
    _, x, y = train_dataset._get_word_tensor(word)
    assert x.shape == y.shape


@pytest.mark.parametrize("form_id", test_forms)
def test_train_random(train_dataset, clts_dataset, lang_fin, form_id):
    """ Test if random masking during training is not random. """
    form = lang_fin.forms[form_id]
    word = clts_dataset._preprocess_word(form)
    w1_idx, w1_x, w1_y = train_dataset._get_word_tensor(word)
    w2_idx, w2_x, w2_y = train_dataset._get_word_tensor(word)
    assert w1_x.tolist() == w2_x.tolist()
    assert w1_y.tolist() == w2_y.tolist()
    assert w1_idx.tolist() == w2_idx.tolist()


@pytest.mark.parametrize("form_id", test_forms)
def test_test_unmasked_input_output_size(unmasked_test_dataset, clts_dataset, lang_fin, form_id):
    form = lang_fin.forms[form_id]
    word = clts_dataset._preprocess_word(form)
    _, x, y = unmasked_test_dataset._get_word_tensor(word)
    assert x.shape == y.shape


@pytest.mark.parametrize("form_id", test_forms)
def test_test_unmasked_random(unmasked_test_dataset, clts_dataset, lang_fin, form_id):
    """ Test if random masking during training is not random. """
    form = lang_fin.forms[form_id]
    word = clts_dataset._preprocess_word(form)
    w1_idx, w1_x, w1_y = unmasked_test_dataset._get_word_tensor(word)
    w2_idx, w2_x, w2_y = unmasked_test_dataset._get_word_tensor(word)
    assert w1_x.tolist() == w2_x.tolist()
    assert w1_y.tolist() == w2_y.tolist()
    assert w1_idx.tolist() == w2_idx.tolist()


@pytest.mark.parametrize("form_id", test_forms)
def test_test_consonant_masking_input_output_size(consonant_masking_test_dataset, clts_dataset, \
    lang_fin, form_id):
    form = lang_fin.forms[form_id]
    word = clts_dataset._preprocess_word(form)
    _, x, y = consonant_masking_test_dataset._get_word_tensor(word)
    assert x.shape == y.shape


@pytest.mark.parametrize("form_id", test_forms)
def test_test_consonant_masking_random(consonant_masking_test_dataset, clts_dataset, \
    lang_fin, form_id):
    """ Test if random masking during training is not random. """
    form = lang_fin.forms[form_id]
    word = clts_dataset._preprocess_word(form)
    w1_idx, w1_x, w1_y = consonant_masking_test_dataset._get_word_tensor(word)
    w2_idx, w2_x, w2_y = consonant_masking_test_dataset._get_word_tensor(word)
    assert w1_x.tolist() == w2_x.tolist()
    assert w1_y.tolist() == w2_y.tolist()
    assert w1_idx.tolist() == w2_idx.tolist()


@pytest.mark.parametrize("form_id", test_forms)
def test_test_vowel_masking_input_output_size(vowel_masking_test_dataset, clts_dataset, \
    lang_fin, form_id):
    form = lang_fin.forms[form_id]
    word = clts_dataset._preprocess_word(form)
    print(word)
    _, X, Y = vowel_masking_test_dataset._get_word_tensor(word)
    for x, y in zip(X, Y):
        assert x.shape == y.shape


@pytest.mark.parametrize("form_id", test_forms)
def test_test_vowel_masking_random(vowel_masking_test_dataset, clts_dataset, lang_fin, form_id):
    """ Test if random masking during training is not random. """
    form = lang_fin.forms[form_id]
    word = clts_dataset._preprocess_word(form)
    w1_idx, w1_x, w1_y = vowel_masking_test_dataset._get_word_tensor(word)
    w2_idx, w2_x, w2_y = vowel_masking_test_dataset._get_word_tensor(word)
    for x1, x2 in zip(w1_x, w2_x): 
        assert x1.tolist() == x2.tolist()
    for y1, y2 in zip(w1_y, w2_y):
        assert y1.tolist() == y2.tolist()
    for i1, i2 in zip(w1_idx, w2_idx):
        assert i1.tolist() == i2.tolist()

@pytest.mark.parametrize("form_id,expected", [
        # (94, [ # sɑtɛːŋkɑːri
        #         ["s", constants.mask, "t", "ɛ", "ŋ", "k", "ɑ", "r", "i"], 
        #         ["s", constants.mask, "t", constants.mask, "ŋ", "k", "ɑ", "r", "i"], 
        #         ["s", constants.mask, "t", constants.mask, "ŋ", "k", constants.mask, "r", "i"], 
        # ]),
        # (373,[ # lɑːtikːɔ
        #         ["l", constants.mask, "t", "i", "k", "ɔ"],
        #         ["l", constants.mask, "t", constants.mask, "k", "ɔ"],
                
        # ])
        (94, [ # sɑtɛːŋkɑːri
                # (1, ["s", "ɑ", "t", constants.mask, "ŋ", "k", constants.mask, "r", constants.mask]),
                (3, ["s", constants.mask, "t", "ɛ", "ŋ", "k", constants.mask, "r", constants.mask]), 
                (6, ["s", constants.mask, "t", constants.mask, "ŋ", "k", "ɑ", "r", constants.mask]), 
                (8, ["s", constants.mask, "t", constants.mask, "ŋ", "k", constants.mask, "r", "i"])
        ]),
        (373,[ # lɑːtikːɔ
                # (1, ["l", "ɑ", "t", constants.mask, "k", constants.mask]),
                (3, ["l", constants.mask, "t", "i", "k", constants.mask]),
                (5, ["l", constants.mask, "t", constants.mask, "k", "ɔ"])
                
        ])
        # more?
])
def test_vowel_masking_get_items(vowel_masking_test_dataset, clts_dataset, lang_fin, \
    form_id, expected):
    """ Test if vowel masking happens in the right way, i. e. at least one vowel is masked and
        at least one vowel is unmasked, with masking proceeding from left to right.
    """
    form = lang_fin.forms[form_id]
    word = clts_dataset._preprocess_word(form)
    indices, items = vowel_masking_test_dataset._compile_items(word)
    for idx_item, item, exp in zip(indices, items, expected):
        idx_exp, item_exp = exp[0], exp[1]
        assert idx_exp == idx_item
        assert item_exp == item

@pytest.mark.parametrize("form_id,expected", [
        # sɑtɛːŋkɑːri
        (94, 
                ([1, 3, 6, 8], [constants.mask, "ɑ", constants.mask, "ɛ", constants.mask, constants.mask, "ɑ", constants.mask, "i"]), 
        ),
        # lɑːtikːɔ
        (373, 
                ([1, 3, 5], [constants.mask, "ɑ", constants.mask, "i", constants.mask, "ɔ"]),
                
        )
        # more?
])
def test_consonant_masking_get_items(consonant_masking_test_dataset, clts_dataset, lang_fin, \
    form_id, expected):
    """ Test if consonant masking happens in the right way, i. e. all consonants are masked. """
    print(expected)
    form = lang_fin.forms[form_id]
    word = clts_dataset._preprocess_word(form)
    target_indices, x, y = consonant_masking_test_dataset._get_word_tensor(word)
    assert target_indices.cpu().tolist() == expected[0]
    assert len(x) == len(y)


def test_test_sets_of_equal_size(vowel_masking_test_dataset, consonant_masking_test_dataset, \
    unmasked_test_dataset):
    """ Tests sets should be of equal size to avoid bias for one of the conditions. """
    assert len(unmasked_test_dataset) == len(consonant_masking_test_dataset)
    assert len(vowel_masking_test_dataset) != len(consonant_masking_test_dataset)

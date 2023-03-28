import pytest
from pyclts import CLTS

from eff.data import get_word_list
from eff.data.alphabet import InputAlphabet, OutputAlphabet
from eff.data.dataset import CLTSDataset
from eff.util import constants


word_list = get_word_list('northeuralex')

bipa = CLTS().bipa


@pytest.fixture
def lang_fin():
    return word_list.languages[0]


@pytest.fixture
def clts_dataset(lang_fin):
    return CLTSDataset(lang_fin.forms)


def test_input_alphabet(clts_dataset):
    input_alphabet = InputAlphabet()
    for word in clts_dataset.words:
        input_alphabet.add_word(word)
        
    print(input_alphabet.symbols)
    assert len(input_alphabet) == 40
    assert constants.sos in input_alphabet
    assert constants.mask in input_alphabet
    assert constants.pad in input_alphabet
    assert constants.eos not in input_alphabet


def test_output_alphabet(clts_dataset):
    output_alphabet = OutputAlphabet()
    for word in clts_dataset.words:
        for char in word:
            if bipa[char].type != 'consonant':
                output_alphabet.add_char(char)

    print(output_alphabet.symbols)
    assert len(output_alphabet) == 23
    assert constants.eos in output_alphabet
    assert constants.mask in output_alphabet
    assert constants.sos not in output_alphabet
    assert constants.pad in output_alphabet
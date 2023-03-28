from cltoolkit.wordlist import Wordlist
import json
from pathlib import Path
from pycldf import Dataset
from typing import List


def get_word_list(dataset_name: str) -> Wordlist:
    """ Load language data from pycldf as a WordList object (contains multiple languages).

    Parameters
    ----------
    dataset_name : str
        String identifying the dataset. Either 'wold' or 'northeuralex'

    Returns
    -------
    Wordlist
        The word list(s)
    """
    if dataset_name == 'wold':
        from lexibank_wold import Dataset as WOLD
        word_list = Wordlist([Dataset.from_metadata(WOLD().cldf_dir.joinpath('cldf-metadata.json'))])
    elif dataset_name == 'northeuralex':
        from lexibank_northeuralex import Dataset as NEL
        word_list = Wordlist([Dataset.from_metadata(NEL().cldf_dir.joinpath('cldf-metadata.json'))])
    return word_list


def load_ipa_transcriptions(json_path: Path) -> List[List[str]]:
    # source: https://github.com/tatuylonen/wiktextract
    """ Load IPA transcriptions from wiktionary data.

    Parameters
    ----------
    json_path : Path
        Path to the wiktionary dump

    Returns
    -------
    List[List[str]]
        The IPA transcriptions
    """
    entries = []
    with open(json_path.absolute(), 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            entries.append(data)
    ipa_transcriptions = []
    for entry in entries:
        if 'sounds' in entry:
            sounds = entry['sounds']
            if len(sounds) > 0 and 'ipa' in sounds[0]:
                ipa_phonemic = sounds[0]['ipa']
                ipa_transcriptions.append(ipa_phonemic)
    return ipa_transcriptions

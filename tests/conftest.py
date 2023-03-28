import pathlib

import pytest

import itertools

from pycldf import Dataset
from pyclts import CLTS


@pytest.fixture
def tests_dir():
    return pathlib.Path(__file__).parent


@pytest.fixture
def repos(tests_dir):
    return tests_dir / 'repos'


@pytest.fixture
def clts(repos):
    return CLTS(repos / 'clts')


@pytest.fixture
def ds_carvalhopurus(repos):
    return Dataset.from_metadata(repos / "carvalhopurus" / "cldf" / "cldf-metadata.json")


@pytest.fixture
def ds_wold(repos):
    return Dataset.from_metadata(repos / "wold" / "cldf" / "cldf-metadata.json")


@pytest.fixture
def sequences(ds_wold):
    return [
        f.cldf.segments for f in 
        ds_wold.objects("LanguageTable")["English"].forms if f.cldf.segments]

@pytest.fixture
def graphemes(sequences):
    return list(itertools.chain(*sequences))


@pytest.fixture
def bipa(clts):
    return clts.bipa


@pytest.fixture
def sca(clts):
    return clts.soundclasses_dict["sca"]


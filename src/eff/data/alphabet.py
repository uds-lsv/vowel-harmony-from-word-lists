# based on https://github.com/tpimentelms/frontload-disambiguation/blob/main/src/h01_data/alphabet.py
from abc import ABC

from eff.util import constants


class Alphabet(ABC):
    PAD_IDX = 0
    MASK_IDX = 1

    def __init__(self) -> None:
        self._chars2idx = {
            constants.pad: self.PAD_IDX,
            constants.mask: self.MASK_IDX
        }
        self._chars_count = {
            constants.pad : 0,
            constants.mask : 0
        }
        self._idx2chars = {idx: char for char, idx in self._chars2idx.items()}
        self._updated = True

    def add_char(self, char):
        if char not in self._chars2idx:
            new_idx = len(self._chars2idx)
            self._chars2idx[char] = new_idx
            self._idx2chars[new_idx] = char
            self._chars_count[char] = 1
            self._updated = False
        else:
            self._chars_count[char] += 1

    def add_word(self, word):
        for char in word:
            self.add_char(char)

    def word2idx(self, word):
        return [self._chars2idx[char] for char in word]

    def char2idx(self, char):
        return self._chars2idx[char]

    def idx2word(self, idx_word):
        if not self._updated:
            self._idx2chars = {idx: char for char, idx in self._chars2idx.items()}
            self._updated = True
        return [self._idx2chars[idx] for idx in idx_word]

    def idx2char(self, idx_char):
        if not self._updated:
            self._idx2chars = {idx: char for char, idx in self._chars2idx.items()}
            self._updated = True
        return self._idx2chars[idx_char]

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            if self._chars2idx == other._chars2idx:
                return True
        return False

    def __len__(self):
        return len(self._chars2idx)

    def __contains__(self, char):
        return char in self._chars2idx

    def __iter__(self):
        for char in self._chars2idx:
            yield char 

    @property
    def symbols(self):
        return list(self._chars2idx.keys())

    @property
    def indices(self):
        return list(self._idx2chars.keys())


class InputAlphabet(Alphabet):

    SOS_IDX = 2

    def __init__(self) -> None:
        super(InputAlphabet, self).__init__()
        self._idx2chars[self.SOS_IDX] = constants.sos
        self._chars2idx[constants.sos] = self.SOS_IDX

class OutputAlphabet(Alphabet):

    EOS_IDX = 2

    def __init__(self) -> None:
        super(OutputAlphabet, self).__init__()
        self._idx2chars[self.EOS_IDX] = constants.eos
        self._chars2idx[constants.eos] = self.EOS_IDX

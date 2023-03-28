from abc import ABC, abstractmethod
import numpy as np
from pycldf import Dataset
from pyclts import TranscriptionSystem
import random
from torch import LongTensor
from torch.utils.data import Dataset
from typing import List, Tuple

from eff.data.alphabet import InputAlphabet, OutputAlphabet
from eff.util import constants


class BaseDataset(Dataset, ABC):
    """ Base dataset class """
    def __init__(self, 
        words: List[str], 
        input_alphabet: InputAlphabet, 
        output_alphabet: OutputAlphabet, 
        bipa: TranscriptionSystem
        ) -> None:
        self._words = words
        self._input_alphabet = input_alphabet
        self._output_alphabet = output_alphabet
        self._bipa = bipa
        self._target_indices, self._X, self._Y = self._get_data_tensors(self._words)
    
    @abstractmethod
    def _get_data_tensors(self, words: List[List[str]]) \
        -> Tuple[List[LongTensor], List[LongTensor], List[LongTensor]]:
        """ Outer loop, populates input, target and indexing variabels. Indices point to unmasked
            vowel poisitions and are later used to retrieve the surprisal values. """
        raise NotImplementedError("Please implement me!")

    @abstractmethod
    def _get_word_tensor(self, word: List[str]) -> Tuple[LongTensor, LongTensor, LongTensor]:
        """ Inner loop, derives input, target and index tensors from a word form. """
        raise NotImplementedError("Please implement me!")

    def __getitem__(self, index):
        return self._target_indices[index], self._X[index], self._Y[index]

    def __len__(self):
        return len(self._X)


class TrainDataset(BaseDataset):
    """ Train set with random input masking """

    def __init__(self,
        words: List[str], 
        input_alphabet: InputAlphabet, 
        output_alphabet: OutputAlphabet, 
        bipa: TranscriptionSystem,
        masking: float
        ) -> None:
        """ 
        Parameters
        ----------
        words : List[str]
            List of words from a CLTSDataset object
        input_alphabet : InputAlphabet
            The input alphabet
        output_alphabet : OutputAlphabet
            The output alphabet
        bipa : TranscriptionSystem
            Transcription needed to access phonological information
        masking : float
            Fraction of input items to be masked
        """
        self.masking = masking
        super(TrainDataset, self).__init__(words, input_alphabet, output_alphabet, bipa)

    def _get_data_tensors(self, words: List[List[str]]) \
        -> Tuple[List[LongTensor], List[LongTensor], List[LongTensor]]:
        target_indices, X, Y = [], [], []
        for word in words:
            target_indices_word, x_word_idx, y_word_idx = self._get_word_tensor(word)
            X.append(x_word_idx)
            Y.append(y_word_idx)
            target_indices.append(target_indices_word)
        return target_indices, X, Y

    def _get_word_tensor(self, word: str) -> Tuple[LongTensor, LongTensor, LongTensor]:
        """ Get word tensor with random input masking

        Parameters
        ----------
        word : str
            The word form

        Returns
        -------
        Tuple[LongTensor, LongTensor, LongTensor]
            Masked input tensor, output tensor, index tensor
        """
        masked_indices = list(np.random.RandomState(constants.random_seed).choice(range(len(word)), \
            round(self.masking*len(word)))) 
        
        x_word_idx = [self._input_alphabet.char2idx(constants.sos)] + \
                [self._input_alphabet.char2idx(char) 
                    if idx not in masked_indices
                    else self._input_alphabet.char2idx(constants.mask) for idx, char in enumerate(word)] 
        y_word_idx = \
        [
            self._output_alphabet.char2idx(char) 
            if self._bipa[char].type in ['vowel', 'diphthong']
            else self._output_alphabet.char2idx(constants.mask) 
            for char in word 
        ] + \
        [self._output_alphabet.char2idx(constants.eos)]    
        target_indices_word = [
            idx for idx, char in enumerate(word) 
            if self._bipa[char].type in ['vowel', 'diphthong']
        ]
        return LongTensor(target_indices_word), LongTensor(x_word_idx), LongTensor(y_word_idx)


class ConsonantMaskingTestSet(BaseDataset):
    """ Test set with masked consonantal positions. """
    def __init__(self,
        words: List[str], 
        input_alphabet: InputAlphabet, 
        output_alphabet: OutputAlphabet, 
        bipa: TranscriptionSystem,
        ) -> None:
        """ 
        Parameters
        ----------
        words : List[str]
            List of words from a CLTSDataset object
        input_alphabet : InputAlphabet
            The input alphabet
        output_alphabet : OutputAlphabet
            The output alphabet
        bipa : TranscriptionSystem
            Transcription needed to access phonological information
        """
        super(ConsonantMaskingTestSet, self).__init__(words, input_alphabet, output_alphabet, bipa)

    def _get_data_tensors(self, words: List[List[str]]) \
        -> Tuple[List[LongTensor], List[LongTensor], List[LongTensor]]:
        target_indices, X, Y = [], [], []
        for word in words:
            target_indices_word, x_word_idx, y_word_idx = self._get_word_tensor(word)
            X.append(x_word_idx)
            Y.append(y_word_idx)
            target_indices.append(target_indices_word)
        return target_indices, X, Y

    def _get_word_tensor(self, word: str) -> Tuple[LongTensor, LongTensor, LongTensor]:
        """ All consonant symbols/indices are replaced by the mask symbol. """
        x_word_idx = [self._input_alphabet.char2idx(constants.sos)] + \
                    [self._input_alphabet.char2idx(char) 
                        if self._bipa[char].type != 'consonant' 
                        else self._input_alphabet.char2idx(constants.mask) for char in word]
        y_word_idx = \
        [
            self._output_alphabet.char2idx(char) 
            if self._bipa[char].type in ['vowel', 'diphthong']
            else self._output_alphabet.char2idx(constants.mask) 
            for char in word 
        ] + \
        [self._output_alphabet.char2idx(constants.eos)] 

        target_indices_word = [
            idx for idx, char in enumerate(word) 
            if self._bipa[char].type in ['vowel', 'diphthong']
        ]
        return LongTensor(target_indices_word), LongTensor(x_word_idx), LongTensor(y_word_idx)


class UnmaskedTestSet(BaseDataset):
    """ Test set with no replacement, used as a control condition and for later VH system comparisions. """
    def __init__(self,
        words: List[str], 
        input_alphabet: InputAlphabet, 
        output_alphabet: OutputAlphabet, 
        bipa: TranscriptionSystem,
        ) -> None:
        super(UnmaskedTestSet, self).__init__(words, input_alphabet, output_alphabet, bipa)

    def _get_data_tensors(self, words: List[List[str]]) \
        -> Tuple[List[LongTensor], List[LongTensor], List[LongTensor]]:
        target_indices, X, Y = [], [], []
        for word in words:
            target_indices_word, x_word_idx, y_word_idx = self._get_word_tensor(word)
            X.append(x_word_idx)
            Y.append(y_word_idx)
            target_indices.append(target_indices_word)
        return target_indices, X, Y

    def _get_word_tensor(self, word: str) -> Tuple[LongTensor, LongTensor]:
        x_word_idx = [self._input_alphabet.char2idx(constants.sos)] + \
                [self._input_alphabet.char2idx(char) for char in word] 
        y_word_idx = \
        [
            self._output_alphabet.char2idx(char) 
            if self._bipa[char].type in ['vowel', 'diphthong']
            else self._output_alphabet.char2idx(constants.mask) 
            for char in word 
        ] + \
        [self._output_alphabet.char2idx(constants.eos)]    

        target_indices_word = [
            idx for idx, char in enumerate(word) 
            if self._bipa[char].type in ['vowel', 'diphthong']
        ]
        return LongTensor(target_indices_word), LongTensor(x_word_idx), LongTensor(y_word_idx)


class VowelMaskingTestSet(BaseDataset):
    """ Vowel masking test set. In a word form, all vowel positions except for the last position 
        are masked once. Surprisal will be reported for the first
        unmasked vowel position. Thus, a (made-up) item [matilda] will be split into 2 items [m,<m>,t,i,l,d,a]
        and [m,<m>,t,<m>,l,d,a], with <m> representing the mask symbol. Thus, 
    """
    
    def __init__(self,
        words: List[str], 
        input_alphabet: InputAlphabet, 
        output_alphabet: OutputAlphabet, 
        bipa: TranscriptionSystem,
        ) -> None:
        super(VowelMaskingTestSet, self).__init__(words, input_alphabet, output_alphabet, bipa)

    def _get_data_tensors(self, words: List[List[str]]) \
        -> Tuple[List[LongTensor], List[LongTensor], List[LongTensor]]:
        target_indices, X, Y = [], [], []
        for word in words:
            target_indices_word, X_word, Y_word = self._get_word_tensor(word)
            X.extend(X_word)
            Y.extend(Y_word)
            target_indices.extend(target_indices_word)
        return target_indices, X, Y

    def _get_word_tensor(self, word: str) -> Tuple[LongTensor, LongTensor, LongTensor]:
        """ Adapted for split items (inner loop over split forms). """
        target_indices_word, X_word, Y_word = [], [], []
        target_indices_items, items = self._compile_items(word)
        for item, target_index in zip(items, target_indices_items):
            x_word_idx = [self._input_alphabet.char2idx(constants.sos)] + \
                [self._input_alphabet.char2idx(char) for char in item]
            y_word_idx = \
            [
                self._output_alphabet.char2idx(char) 
                if self._bipa[char].type in ['vowel', 'diphthong']
                else self._output_alphabet.char2idx(constants.mask) 
                for char in item
            ] + \
            [self._output_alphabet.char2idx(constants.eos)]    

            X_word.append(LongTensor(x_word_idx))
            Y_word.append(LongTensor(y_word_idx))
            target_indices_word.append(LongTensor([target_index]))
        return target_indices_word, X_word, Y_word

    def _compile_items(self, word: List[str]) -> Tuple[List[int], List[str]]:
        """ Method to split a single word form into multiple items as described above. """
        indices = []
        items = []
        for i in range(len(word)):
            char_i = word[i]
            if self._bipa[char_i].type in ['vowel', 'diphthong']:
                item = [
                    word[j] if (i == j or self._bipa[word[j]].type == 'consonant')
                    else constants.mask
                    for j in range(len(word))
                ]
                items.append(item)
                indices.append(i)
        # First vowel position is left out since it is not conditioned on another vowel.
        items = items[1:] if len(items) > 1 else items
        indices = indices[1:] if len(indices) > 1 else indices
        return indices, items

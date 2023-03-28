from anytree import Node
from cltoolkit.models import Form
from lingpy.sequence.sound_classes import syllabify
from pyclts import CLTS
from typing import List, Union

from eff.data.alphabet import InputAlphabet, OutputAlphabet
from eff.util import constants


class CLTSDataset():
    """ Class containing all data used for training & testing a language model. """

    def __init__(self, forms: Union[List[Form], List[str]], min_len=2, \
        unique_sequences=False) -> None:
        """ 
        Parameters
        ----------
        forms : Union[List[Form], List[str]]
            List of Form objects derived from a pycldf Language object.
        min_len : int, optional
            minimum length of a word to be included.
        """
        self._input_alphabet = InputAlphabet()
        self._output_alphabet = OutputAlphabet()
        self._bipa = CLTS().bipa
        self.min_len = min_len
        self.unique_sequences = unique_sequences
        if isinstance(forms[0], Form):
            self._words = self._preprocess_data(forms)
        if isinstance(forms[0], str):
            self._words = self._preprocess_data_list(forms)
        if self.unique_sequences:
            self._words = self._unify_sequences(self.words)
    
    def _preprocess_data(self, forms: List[Form]) -> List[List[str]]:
        """ Outer loop for 

        Parameters
        ----------
        forms : List[Form]
            List of forms

        Returns
        -------
        List[List[str]]
            List of purged strings
        """
        words = []
        for form in forms:
            word = self._preprocess_word(form)
            # at least 2 vowel phonemes
            n_vowels = len([s for s in word if self.bipa[s].type == 'vowel'])
            if n_vowels >= self.min_len:
                words.append(word)
                self._input_alphabet.add_word(word)
                for char in word:
                    if self.bipa[char].type in ['vowel', 'diphthong']:
                        self._output_alphabet.add_char(char)
        return words

    def _preprocess_data_list(self, forms: List[str]) -> List[List[str]]:
        words = []
        for form in forms:
            word = self._preprocess_word_list(form)
            n_vowels = len([s for s in word if self.bipa[s].type == 'vowel'])
            if n_vowels >= self.min_len:
                words.append(word)
                self._input_alphabet.add_word(word)
                for char in word:
                    if self.bipa[char].type in ['vowel', 'diphthong']:
                        self._output_alphabet.add_char(char)
        return words

    def _preprocess_word(self, form: Form) -> List[str]:
        """ Removes symbols not needed for the analysis

        Parameters
        ----------
        form : Form

        Returns
        -------
        List[str]
            The purged string
        """
        segments = form.data['Segments']
        # treat composita like all other words
        segments = [s for s in segments if s != constants.marker]
        segemnts = [s for s in segments if s != constants.word_boundary]
        # ignore tones
        segments = [s for s in segments if s not in constants.tones_mandarin]
        # map long and half-long sounds to their short counterpart
        segments = [s.replace(constants.mid_long_segment_marker, '') for s in segments]
        segments = [s.replace(constants.long_segment_marker, '') for s in segments]
        return segments

    def _preprocess_word_list(self, form: str) -> List[str]:
        segments = list(form)
        # treat composita like all other words
        # remove ipa slashes
        # segments = segments[1:-1]
        segments = [s for s in segments if s != constants.marker]
        segemnts = [s for s in segments if s != constants.word_boundary]
        # ignore tones
        segments = [s for s in segments if s not in constants.tones_mandarin]
        # map long and half-long sounds to their short counterpart
        segments = [s.replace(constants.mid_long_segment_marker, '') for s in segments]
        segments = [s.replace(constants.long_segment_marker, '') for s in segments]
        # remove special wiktionary characters
        segments = [s for s in segments if s not in constants.wikt_chars]
        segments = [s for s in segments if s != '']
        segments = [s for s in segments if s != ' ']
        return segments

    def _unify_sequences(self, sequences: List[str]) -> List[str]:
        """ Ensures that only unique sequences (forms) are presented to the model by first building a 
            phoneme tree, and then dropping repeating sentences.

        Parameters
        ----------
        sequences : List[str]
            List of sentences to be unified

        Returns
        -------
        List[str]
            List of unified sequences
        """
       
        sequence_tree = Node("root", parent=None)
        pre = sequence_tree
        unique_sequences = []

        for seq in sequences:
            syllables = syllabify(seq, output="nested")
            syllables = [tuple(syl) for syl in syllables]
            # syllables = [''.join([s for s in syl]) for syl in syllables]
            for nex in syllables:
                node_exists = False
                for child in pre.children:
                    if child.name == nex:
                        pre = child
                        node_exists = True
                        break
                if not node_exists:
                    node = Node(nex, parent=pre)
                    pre = node
            pre = sequence_tree

        for leaf in sequence_tree.leaves:
            name = []
            for anc in leaf.ancestors:
                if not anc.is_root:
                    name.extend(list(anc.name))
            name.extend(leaf.name)
            unique_sequences.append(name)
       
        return unique_sequences

    def __len__(self):
        return len(self.words)

    def __eq__(self, other) -> bool:
        if (isinstance(other, self.__class__)):
            if self.words == other.words and self.input_alphabet == other.input_alphabet \
                and self.output_alphabet == other.output_alphabet: \
            return True
        return False

    @property
    def words(self):
        return self._words

    @property
    def bipa(self):
        return self._bipa

    @property
    def pad_idx(self):
        return self._output_alphabet.PAD_IDX

    @property
    def mask_idx(self):
        return self._output_alphabet.MASK_IDX

    @property
    def input_alphabet(self):
        return self._input_alphabet

    @property
    def output_alphabet(self):
        return self._output_alphabet

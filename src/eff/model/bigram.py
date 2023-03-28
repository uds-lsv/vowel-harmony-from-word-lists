import nltk
import numpy as np


class LidstoneBigramLM:

    # epsilon same as in Goldsmith & Riggle 2012
    def __init__(self, succession_counts, grapheme_counts, V, epsilon=0.5):
        self.succession_counts = succession_counts
        self.grapheme_counts = grapheme_counts
        self.epsilon = epsilon
        self.V = V
        self.N = sum(grapheme_counts.values())

    def _get_succession_prob(self, succession):
        g1 = succession[0]
        g_count = self.grapheme_counts[g1]
        s_count = self.succession_counts[succession]
        return (s_count + self.epsilon) / (g_count + len(self.V)*self.epsilon) 

    def _get_grapheme_prob(self, grapheme):
        g_count = self.grapheme_counts[grapheme]
        return (g_count + self.epsilon) /(self.N + len(self.V)*self.epsilon)

    def perplexity(self, word):
        successions = list(nltk.bigrams(word))
        cond_entr = -sum([np.log2(self._get_succession_prob(succession)) for succession in successions])
        return 2**(1/len(successions))*cond_entr
    
    def pmi(self, succession):
        g1 = succession[0]
        g2 = succession[1]
        p_s = self._get_succession_prob(succession)
        p_g1 = self._get_grapheme_prob(g1)
        p_g2 = self._get_grapheme_prob(g2)
        return np.log2(p_s/(p_g2))

    def count(self, succession):
        return self.succession_counts[succession]
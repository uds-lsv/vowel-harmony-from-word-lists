import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(0)

# class OCPLoss(nn.Module):
#     loss_cls = nn.CrossEntropyLoss
#     def __init__(self, alphabet: Alphabet, beta=0.25):
#         super().__init__()
#         self.alphabet = alphabet
#         self.beta = beta
#         self.criterion = self.loss_cls(ignore_index=alphabet.PAD_IDX, reduction='none')

#     def __call__(self, logits, y):
#         losses = self.criterion(logits, y)
#         predictions = torch.argmax(logits, 1)
#         if self.beta > 0:
#             sequences = self.get_sequences(y)
#             for i in range(1, len(y)):
#                 if sequences[i] == sequences[i-1]:
#                     normalization_term = self.get_normalization_term(predictions[i-1], predictions[i], y[i], losses[i])
#                     losses[i] = losses[i] + normalization_term
#                 # loss = losss + normalization_term
#         return losses.mean()

#     def get_sequences(self, y):
#         start_index = 0
#         seq_boundaries = list(np.where(np.array(y.cpu()) == self.alphabet.EOS_IDX)[0])
#         seq_boundaries = [start_index] + seq_boundaries + [len(y)]
#         sequences = {}
#         for w, b in enumerate(seq_boundaries):
#             for i in range(start_index, b+1):
#                 sequences[i] = w
#             start_index = b+1
#         return sequences

#     def get_centrality(self, sound: Union[Vowel, Diphthong]):
#         return sound.from_sound.centrality if sound.type == 'diphthong' else sound.centrality

#     def get_normalization_term(self, prev_pred, pred, y, loss):
#         # y = y.item()
#         pred = pred.item()
#         prev_pred = prev_pred.item()

#         char_prev, char_pred = self.alphabet.idx2char(prev_pred), self.alphabet.idx2char(pred)
#         sound_prev, sound_pred = bipa[char_prev], bipa[char_pred]
#         # This is specific to Finnish
#         if sound_prev.type in ['vowel', 'diphthong'] and sound_pred.type in ['vowel', 'diphthong']:
#             prev_centr = self.get_centrality(sound_prev) 
#             pred_centr = self.get_centrality(sound_pred)
#             if ((prev_centr in ['front', 'near-front'] and pred_centr in ['back', 'near-back']) \
#                 or (prev_centr in ['back', 'near-back'] and pred_centr in ['front', 'near-front'])) \
#             and not sound_pred in [bipa['i'], bipa['ɛ']] \
#             and not sound_prev in [bipa['i'], bipa['ɛ']]:
#                 # print(sound_prev.grapheme, sound_pred.grapheme)
#                 return self.beta*loss
#         return 0


class WeightedCELoss:
    def __init__(self, weight, eos_idx):
        self.eos_idx = eos_idx
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, reduction='mean')


    # def _get_perplexity(self, seq, trg_seq):
    #     slen = seq.shape[0]
    #     eos = ((trg_seq == 2).nonzero()).item()
    #     logprobs = F.log_softmax(seq, dim=1)
    #     logprobs = logprobs[range(slen), trg_seq]
    #     logprobs = logprobs[:eos+1]
    #     pp = 2**(-1/len(logprobs)*sum(logprobs.cpu().tolist()))
    #     return pp

    def _get_logprobs(self, seq, trg_seq):
        slen = seq.shape[0]
        eos = ((trg_seq == 2).nonzero()).item()
        logprobs = F.log_softmax(seq, dim=1)
        logprobs = logprobs[range(slen), trg_seq]
        logprobs = logprobs[:eos] # ignore eos
        # print(trg_seq, logprobs)
        return logprobs.cpu().tolist()
    
    def __call__(self, logits, target):
        bloss = 0
        # bpp = 0
        blogprobs = []
        bsize = logits.shape[0]
        for i in range(bsize):
            logits_seq = logits[i]
            trg_seq = target[i]
            # bpp += self._get_perplexity(logits_seq, trg_seq)
            # blogprobs += [-1*sum(self._get_logprobs(seq, trg_seq))]
            blogprobs += [self._get_logprobs(logits_seq, trg_seq)]
            bloss += self.loss_fn(logits_seq, trg_seq)
        # return bloss, bpp, blogprobs
        return bloss, blogprobs

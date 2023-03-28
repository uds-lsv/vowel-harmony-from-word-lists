# based on https://github.com/tpimentelms/frontload-disambiguation/blob/main/src/h02_learn/model/lstm.py
import numpy as np
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

# from eff.model.base_pytorch import BasePytorchLM
from eff.util import constants


torch.manual_seed(0)


class LstmLM(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim, hidden_dim, n_layers, loss_fn,
            dropout=0.3):
        super(LstmLM, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(self.input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                                  dropout=(dropout if n_layers > 1 else 0),
                                  batch_first=True)
        self.dropout = nn.Dropout(dropout)
        # self.linear = nn.Linear(hidden_dim, embedding_dim)
        # self.out = nn.Linear(hidden_dim, self.alphabet_size)
        self.out = torch.nn.Linear(hidden_dim, self.output_dim)
        self.loss_fn = loss_fn

        # self.out.weight = self.embedding.weight

    # def init_hidden(self, batch_size):
    #      # Initialize hidden state with zeros
    #     h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
    #     # Initialize cell state
    #     c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
    #     return h0, c0


    def forward(self, inputs):
        batch_size = inputs.shape[0]
        # embedded = self.embedding(inputs)
        # embedded = self.dropout(embedded)
        # embedded = embedded[:, None, :] # add missing dimension for lstm input
        # out, lstm_hidden = self.lstm(embedded, lstm_hidden)
        # out = self.linear(out)
        # out = self.out(out)

        x = self.embedding(inputs)
        x = self.dropout(x)
        # h0, c0 = self.init_hidden(batch_size)        
        # c_t, _ = self.lstm(x, (h0.detach(), c0.detach()))
        c_t, _ = self.lstm(x)
        # c_t, _ = self.lstm(x)
        # x = self.linear(c_t)
        out = self.out(c_t)

        return out


    def get_loss(self, logits, y):
        loss, pp, logprobs = self.loss_fn(logits, y)
        return loss, pp, logprobs


# class LstmLM(BasePytorchLM):
#     # pylint: disable=arguments-differ,too-many-instance-attributes
#     name = 'lstm-lm'

#     def __init__(self, alphabet, embedding_size, hidden_size,
#                  nlayers, dropout, masked):
#         super().__init__(alphabet, embedding_size, hidden_size,
#                          nlayers, dropout)
#         self.masked = masked
#         self.embedding = nn.Embedding(self.alphabet_size, embedding_size)

#         self.lstm = nn.LSTM(embedding_size, hidden_size, nlayers,
#                             dropout=(dropout if nlayers > 1 else 0),
#                             batch_first=True)

#         self.dropout = nn.Dropout(dropout)
#         self.linear = nn.Linear(hidden_size, embedding_size)
#         self.out = nn.Linear(embedding_size, self.alphabet_size)

#         # Tie weights
#         if not self.masked:
#             self.out.weight = self.embedding.weight

#     def forward(self, x):
#         x_emb = self.get_embeddings(x)

#         c_t, _ = self.lstm(x_emb)
#         c_t = self.dropout(c_t).contiguous()

#         hidden = F.relu(self.linear(c_t))
#         logits = self.out(hidden)
#         return logits

#     def get_embeddings(self, instance):
#         emb = self.dropout(self.embedding(instance))

#         return emb

#     def get_args(self):
#         return {
#             'nlayers': self.nlayers,
#             'hidden_size': self.hidden_size,
#             'embedding_size': self.embedding_size,
#             'dropout': self.dropout_p,
#             'alphabet': self.alphabet,
#             'masked': self.masked
#         }
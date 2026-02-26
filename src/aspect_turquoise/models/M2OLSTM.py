import torch
from torch import nn
import torch.nn.functional as F


class M2OLSTM(nn.Module):

    def __init__(
            self,
            vocab_size, embed_dim=200,
            num_layers=2, hidden_size=256, bidirectional=False,
    ):
        super(M2OLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0,
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.linear = nn.Linear(
            in_features=hidden_size * (2 if bidirectional else 1),
            out_features=vocab_size,
        )

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, _ = self.lstm(embedded)
        last_hidden = outputs[:, -1, :]
        logits = self.linear(last_hidden)
        return logits

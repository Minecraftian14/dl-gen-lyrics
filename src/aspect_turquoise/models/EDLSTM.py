import torch
from torch import nn
import torch.nn.functional as F


class _Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_size=256, hidden_size=256, num_layers=2, dropout=0.1):
        super(_Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_size, hidden_size,
            num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.linear_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.linear_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        """
        :param x: [=] (batch_size, seq_len) and is integral
        :return: hidden [=] (batch_size, hidden_size), cell [=] (batch_size, hidden_size)
        """

        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs [=] (batch_size, seq_len, 2 * hidden_size)  # 2 for BiDirec
        # hidden  [=] (2 * n_layers, batch_size, hidden_size)
        # cell    [=] (2 * n_layers, batch_size, hidden_size)

        # Concatenate forward and backward hidden states
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        cell_cat = torch.cat((cell[-2], cell[-1]), dim=1)

        # Project to decoder size
        hidden = torch.tanh(self.linear_hidden(hidden_cat)).unsqueeze(0)
        cell = torch.tanh(self.linear_cell(cell_cat)).unsqueeze(0)

        return hidden, cell

class _Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_size=256, hidden_size=256, num_layers=1, dropout=0.1):
        super(_Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.linear_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        """
        :param x: [=] (batch_size, 1) and is integral
        :return: predictions [=] (batch_size, vocab_size), hidden [=] (batch_size, hidden_size), cell [=] (batch_size, hidden_size)
        """

        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # outputs [=] (batch_size, seq_len, hidden_size)
        # hidden  [=] (n_layers, batch_size, hidden_size)
        # cell    [=] (n_layers, batch_size, hidden_size)

        predictions = self.linear_out(outputs)
        # predictions [=] (batch_size, 1, vocab_size)

        return predictions[:, 0, :], hidden, cell


class EDLSTM(nn.Module):
    def __init__(self, start_token_index, encoder: _Encoder, decoder: _Decoder, device):
        super(EDLSTM, self).__init__()
        self.start_token_index = start_token_index
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, x: 'Sample', y=None, max_evals=10):
        """
        :param x: [=] (batch_size, seq_len)
        :return: predictions [=] (batch_size, vocab_size),
                 hidden [=] (batch_size, hidden_size),
                 cell [=] (batch_size, hidden_size)
        """
        batch_size = x.shape[0]
        vocab_size = self.decoder.linear_out.out_features

        hidden, cell = self.encoder(x)


        if y is None:
            y = torch.ones(batch_size, 1, dtype=torch.long).to(self.device) * self.start_token_index
            p = torch.zeros(batch_size, max_evals, vocab_size).to(self.device)

            for i in range(max_evals):
                predictions, hidden, cell = self.decoder(y, hidden, cell)
                p[:, i] = predictions

                best_guess = predictions.argmax(1)
                y = (y * 0) + best_guess

        else:
            p = torch.zeros(batch_size, y.shape[1], vocab_size).to(self.device)

            for i in range(y.shape[1]):
                predictions, hidden, cell = self.decoder(y[:, i:i + 1], hidden, cell)
                p[:, i] = predictions

        return p

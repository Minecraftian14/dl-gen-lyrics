from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class M2OLSTM(nn.Module):

    def __init__(
            self,
            vocab_size, embed_dim=200,
            num_layers=2, hidden_size=256, bidirectional=True,
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

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, _ = self.lstm(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)

        idx = (lengths - 1).view(-1, 1).expand(len(lengths), outputs.size(2)).unsqueeze(1)
        last_hidden = outputs.gather(1, idx).squeeze(1)

        logits = self.linear(last_hidden)
        return logits

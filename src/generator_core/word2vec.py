import random
from collections import Counter

import torch
import torch.nn as nn


class Vocabulary:
    def __init__(self, token_stream, min_freq=5):
        counter = Counter()
        for tokens in token_stream:
            counter.update(tokens)

        self.itos = ['<PAD>', '<UNK>']
        self.itos += [w for w, c in counter.items() if c >= min_freq]

        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, token):
        return self.stoi.get(token, self.stoi['<UNK>'])

    def __len__(self):
        return len(self.itos)


class SkipGramDataset(torch.utils.data.Dataset):
    def __init__(self, token_stream, vocab, window_size=5):
        self.token_stream = token_stream
        self.vocab = vocab
        self.window_size = window_size

        self.stream = token_stream()
        self.window = []
        self.window_index = 0
        for i in range(window_size):
            self.window.append(self.vocab.encode(next(self.stream)))

    def __iter__(self):
        return self

    def __next__(self):
        if not self.stream:
            self.stream = self.token_stream()
            self.window = []
            self.window_index = 0

        try:
            center = self.window[self.window_index]
            self.window_index += 1


        except StopIteration as e:
            raise e

        if self.current < self.limit:
            self.current += 1
            return self.current
        else:
            # Raises StopIteration when the sequence is exhausted
            raise StopIteration

    def __getitem__(self, idx):
        center = self.data[idx]

        window = random.randint(1, self.window_size)
        start = max(0, idx - window)
        end = min(len(self.data), idx + window + 1)

        context = []
        for i in range(start, end):
            if i != idx:
                context.append(self.data[i])

        if not context:
            context = [center]

        context_word = random.choice(context)

        return center, context_word


class SkipGramWord2Vec(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.input_emb = nn.Embedding(vocab_size, emb_dim)
        self.output_emb = nn.Embedding(vocab_size, emb_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.input_emb.weight, -0.5, 0.5)
        nn.init.zeros_(self.output_emb.weight)

    def forward(self, words):
        center_words, pos_words, neg_words = words

        center_emb = self.input_emb(center_words)
        pos_emb = self.output_emb(pos_words)
        neg_emb = self.output_emb(neg_words)

        # Positive score
        pos_score = torch.sum(center_emb * pos_emb, dim=1)
        pos_loss = torch.log(torch.sigmoid(pos_score) + 1e-10)

        # Negative score
        neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze()
        neg_loss = torch.sum(torch.log(torch.sigmoid(-neg_score) + 1e-10), dim=1)

        loss = -(pos_loss + neg_loss)
        return loss.mean()

import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence


class TransformerDataset(data.Dataset):
    def __init__(self, teal):
        self.teal = teal

    def __len__(self):
        return len(self.teal.ds_data)

    def __getitem__(self, index):
        teal = self.teal
        sample = teal.ds_data.iloc[index]

        lyrics = sample['lyrics']
        tag = sample['tag']

        # tokenize lyrics
        tokens = teal.tokenize_text(lyrics)

        # get context words
        context_words = teal.get_context_words(lyrics)

        # genre token
        genre_token = f"genre {tag}"

        # build conditioning text
        cond_text = genre_token + " " + " ".join(context_words)

        # tokenize conditioning
        cond_tokens = teal.tokenize_text(cond_text)

        return tokens, cond_tokens

    def collate_fn(self, batch):
        tokens, cond_tokens = zip(*batch)

        input_ids = []
        target_ids = []

        for t, c in zip(tokens, cond_tokens):
            full = c + t

            input_ids.append(torch.tensor(full[:-1], dtype=torch.long))
            target_ids.append(torch.tensor(full[1:], dtype=torch.long))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        target_ids = pad_sequence(target_ids, batch_first=True, padding_value=0)

        return (input_ids,), target_ids
        
        
class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=3000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0)/d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]    

class GQAAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_groups):
        super().__init__()

        assert d_model % n_heads == 0
        assert n_heads % n_groups == 0

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(3000, 3000), diagonal=1).bool()
        )

        self.n_heads = n_heads
        self.n_groups = n_groups
        self.head_dim = d_model // n_heads
        self.heads_per_group = n_heads // n_groups

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, n_groups * self.head_dim)
        self.v_proj = nn.Linear(d_model, n_groups * self.head_dim)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, key_padding_mask=None):
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_groups, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_groups, self.head_dim).transpose(1, 2)

        k = k.repeat_interleave(self.heads_per_group, dim=1)
        v = v.repeat_interleave(self.heads_per_group, dim=1)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # causal mask
        mask = self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0)
        attn = attn.masked_fill(mask, float('-inf'))

        # padding mask
        if key_padding_mask is not None:
            pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool)
            attn = attn.masked_fill(pad_mask, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=0.1, training=self.training)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps

        # learnable parameters
        self.gamma = nn.Parameter(torch.ones(d_model))  # scale
        self.beta = nn.Parameter(torch.zeros(d_model))  # shift

    def forward(self, x):
        # x shape: (B, T, D)

        mean = x.mean(dim=-1, keepdim=True)           # (B, T, 1)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)

        x_norm = (x - mean) / torch.sqrt(variance + self.eps)

        return self.gamma * x_norm + self.beta

class GELUFFN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d*4),
            nn.GELU(),
            nn.Linear(d*4, d)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        d_model = config["d_model"]
        self.dropout = nn.Dropout(0.1)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.attn = GQAAttention(
            d_model,
            config["n_heads"],
            config["n_groups"]
        )

        self.ffn = GELUFFN(d_model)

    def forward(self, x, key_padding_mask=None):
        attn_out = self.attn(self.norm1(x), key_padding_mask)
        x = x + self.dropout(attn_out)
        
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, config, embedding_weights=None):
        super().__init__()

        d_model = config["d_model"]

        if embedding_weights is None:
            self.embed = nn.Embedding(vocab_size, d_model)
        else:
            self.embed = nn.Embedding.from_pretrained(
                embedding_weights,
                freeze=False,
                padding_idx=0
            )

        self.pe = SinusoidalPE(d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config["n_layers"])
        ])

        self.norm = LayerNorm(d_model)

        self.head = nn.Linear(d_model, vocab_size)
        self.head.weight = self.embed.weight
        for name, p in self.named_parameters():
            if p.dim() > 1 and "embed" not in name:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        key_padding_mask = (x == 0)

        x = self.embed(x)
        x = self.pe(x)

        for block in self.blocks:
            x = block(x, key_padding_mask)

        x = self.norm(x)
        return self.head(x)
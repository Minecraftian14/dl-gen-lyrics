import numpy as np


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

        # MAX_LEN = 128

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
    

class RoPE(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        B, T, D = x.shape
        half = D // 2

        freqs = torch.arange(half, device=x.device).float()
        freqs = 1.0 / (10000 ** (freqs / half))

        pos = torch.arange(T, device=x.device).float()
        angles = pos[:, None] * freqs[None, :]

        sin = angles.sin()[None, :, :]
        cos = angles.cos()[None, :, :]

        x1 = x[..., :half]
        x2 = x[..., half:half*2]

        x_rope_1 = x1 * cos - x2 * sin
        x_rope_2 = x1 * sin + x2 * cos

        return torch.cat([x_rope_1, x_rope_2, x[..., half*2:]], dim=-1)
    

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        return self.weight * x / torch.sqrt(norm + self.eps)

def get_norm(norm_type, d):
    return RMSNorm(d) if norm_type=="rmsnorm" else nn.LayerNorm(d)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, x):
        B, T, _ = x.shape
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        out, _ = self.attn(x, x, x, attn_mask=mask)
        return out

class SimpleGQA(nn.Module):
    def __init__(self, d_model, n_heads, n_groups):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_groups == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_groups = n_groups
        self.head_dim = d_model // n_heads
        self.heads_per_group = n_heads // n_groups

        # Projections
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, n_groups * self.head_dim)
        self.v = nn.Linear(d_model, n_groups * self.head_dim)
        self.o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, _ = x.shape

        # Project
        q = self.q(x)  # (B, T, d_model)
        k = self.k(x)  # (B, T, n_groups * head_dim)
        v = self.v(x)

        # Reshape
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  
        # (B, n_heads, T, head_dim)

        k = k.view(B, T, self.n_groups, self.head_dim).transpose(1, 2)  
        # (B, n_groups, T, head_dim)

        v = v.view(B, T, self.n_groups, self.head_dim).transpose(1, 2)

        # Expand k, v to match heads
        k = k.repeat_interleave(self.heads_per_group, dim=1)  
        v = v.repeat_interleave(self.heads_per_group, dim=1)
        # Now: (B, n_heads, T, head_dim)

        # Attention
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        
        # Softmax
        attn = torch.softmax(attn, dim=-1)

        out = attn @ v  # (B, n_heads, T, head_dim)

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        return self.o(out)

def get_attention(attn_type, d_model, n_heads, n_groups):
    return MultiHeadAttention(d_model, n_heads) if attn_type=="mha" else SimpleGQA(d_model, n_heads, n_groups)


class SwiGLU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w1 = nn.Linear(d, d*2)
        self.w2 = nn.Linear(d, d)

    def forward(self, x):
        x1, x2 = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(x1) * x2)

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

def get_ffn(ffn_type, d):
    return SwiGLU(d) if ffn_type=="swiglu" else GELUFFN(d)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, config):
        super().__init__()
        self.norm1 = get_norm(config["norm"], d_model)
        self.norm2 = get_norm(config["norm"], d_model)

        self.attn = get_attention(config["attn"], d_model, n_heads, config['n_groups'])
        self.ffn = get_ffn(config["ffn"], d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, config, embedding_weights=None):
        super().__init__()

        if embedding_weights is None:
            self.embed = nn.Embedding(vocab_size, config["d_model"])
        else:
            self.embed = nn.Embedding.from_pretrained(embedding_weights, freeze=False, padding_idx=0)

        self.pe_type = config["pe"]

        if self.pe_type == "sinusoidal":
            self.pe = SinusoidalPE(config["d_model"])
        elif self.pe_type == "rope":
            self.pe = RoPE(config["d_model"])

        self.blocks = nn.ModuleList([
            TransformerBlock(config["d_model"], config["n_heads"], config)
            for _ in range(config["n_layers"])
        ])

        self.norm = get_norm(config["norm"], config["d_model"])
        self.head = nn.Linear(config["d_model"], vocab_size)

    def forward(self, x):
        x = self.embed(x)

        if self.pe_type in ["sinusoidal", "rope"]:
            x = self.pe(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.head(x)











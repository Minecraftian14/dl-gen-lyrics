"""
src/model/bigru.py

Bidirectional GRU language model for conditional lyrics generation.

Architecture
------------
  Embedding (vocab_size × embed_dim)
      │
  Dropout
      │
  BiGRU stack (num_layers layers, hidden_dim per direction)
      │  The forward & backward hidden states are concatenated,
      │  giving an effective hidden size of 2 × hidden_dim.
      │
  Layer Normalization
      │
  Projection Linear (2×hidden_dim → hidden_dim)
      │
  ReLU
      │
  Dropout
      │
  Output Linear (hidden_dim → vocab_size)
      │
  LogSoftmax (used during inference; training uses CrossEntropyLoss directly)

Conditional Generation
----------------------
Genre and theme tokens are part of the vocabulary (special tokens). They
are prepended to each sequence during preprocessing, so the model learns
to attend to them through its recurrent memory — no separate conditioning
mechanism is needed.

Hidden-state bridging
---------------------
For generation, the last hidden state of one chunk is passed as the
initial hidden state of the next, enabling long-range coherence across
stanzas.

Note on Bidirectionality at Inference Time
------------------------------------------
True bidirectional processing requires the full future context, which
isn't available during left-to-right generation. We follow the common
practice of using the *forward* hidden states only at inference time,
while the full BiGRU (both directions) is used during training for
richer gradient signal and better embeddings.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from ..GRU import config


class BiGRULyricsModel(nn.Module):
    """
    Bidirectional GRU language model.

    Args:
        vocab_size  : Size of the SentencePiece vocabulary.
        embed_dim   : Token embedding dimension (256).
        hidden_dim  : GRU hidden size per direction (512 → 1024 total).
        num_layers  : Number of stacked GRU layers.
        dropout     : Dropout probability (applied between layers & on embed).
        pad_id      : Padding token ID (for embedding padding_idx).
    """

    def __init__(
        self,
        vocab_size:  int   = config.SP_VOCAB_SIZE,
        embed_dim:   int   = config.EMBEDDING_DIM,
        hidden_dim:  int   = config.HIDDEN_DIM,
        num_layers:  int   = config.NUM_LAYERS,
        dropout:     float = config.DROPOUT,
        pad_id:      int   = 0,
        word2vec_weights=None,
        word2vec_frozen=True,
    ) -> None:
        super().__init__()

        self.vocab_size  = vocab_size
        self.embed_dim   = embed_dim
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers
        self.dropout_p   = dropout
        self.pad_id      = pad_id
        self.bidirectional = True
        self.num_directions = 2  # BiGRU

        # ── Embedding ──────────────────────────────────────────────────
        if word2vec_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(
                word2vec_weights, freeze=word2vec_frozen, padding_idx=pad_id
            )
        else:
            self.embedding = nn.Embedding(
                vocab_size, embed_dim, padding_idx=pad_id
            )
            nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.embed_drop = nn.Dropout(dropout)

        # ── BiGRU Stack ────────────────────────────────────────────────
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        # ── Post-GRU projection ────────────────────────────────────────
        # Concatenated forward + backward → hidden_dim
        self.layer_norm  = nn.LayerNorm(hidden_dim * self.num_directions)
        self.proj        = nn.Linear(hidden_dim * self.num_directions, hidden_dim)
        self.proj_drop   = nn.Dropout(dropout)

        # ── Output head ───────────────────────────────────────────────
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        # Weight tying: share embedding weights with output projection
        # (maps back from hidden → vocab space using the same basis vectors)
        if embed_dim == hidden_dim:
            self.output_proj.weight = self.embedding.weight  # type: ignore

        # Initialise linear layers
        self._init_weights()

    # ──────────────────────────────────────────────────────────────────
    def _init_weights(self) -> None:
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.zero_()

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    # ──────────────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids: torch.Tensor,             # (B, T)
        hidden:    Optional[torch.Tensor] = None,  # (num_layers*2, B, H)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids : (batch, seq_len) token IDs.
            hidden    : Initial hidden state for the GRU, or None.

        Returns:
            logits : (batch, seq_len, vocab_size)  — raw (un-normalised) scores.
            hidden : (num_layers * num_directions, batch, hidden_dim)
        """
        # Embedding
        x = self.embedding(input_ids)        # (B, T, E)
        x = self.embed_drop(x)

        # BiGRU
        gru_out, hidden = self.gru(x, hidden)   # (B, T, 2H), (2*L, B, H)

        # LayerNorm + projection
        out = self.layer_norm(gru_out)           # (B, T, 2H)
        out = F.relu(self.proj(out))             # (B, T, H)
        out = self.proj_drop(out)

        # Logits
        logits = self.output_proj(out)           # (B, T, V)
        return logits, hidden

    # ──────────────────────────────────────────────────────────────────
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return zero initial hidden state."""
        return torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_dim,
            device=device,
        )

    # ──────────────────────────────────────────────────────────────────
    def forward_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Extract only the *forward* direction hidden states.
        Used during auto-regressive inference to seed the next step.

        BiGRU hidden layout (num_layers * 2, batch, H):
          Layer 0 fwd → index 0
          Layer 0 bwd → index 1
          Layer 1 fwd → index 2
          Layer 1 bwd → index 3
          …
        """
        fwd_indices = list(range(0, self.num_layers * 2, 2))
        return hidden[fwd_indices]  # (num_layers, B, H)

    # ──────────────────────────────────────────────────────────────────
    def forward_only_gru(self) -> nn.GRU:
        """
        Return a *unidirectional* GRU constructed from the forward weights
        of the trained BiGRU. Used internally for inference.
        """
        fwd_gru = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        with torch.no_grad():
            for layer in range(self.num_layers):
                # Copy forward weights from the BiGRU
                fwd_gru.weight_ih_l0 if layer == 0 else None
                src_ih  = getattr(self.gru, f"weight_ih_l{layer}")
                src_hh  = getattr(self.gru, f"weight_hh_l{layer}")
                src_bih = getattr(self.gru, f"bias_ih_l{layer}")
                src_bhh = getattr(self.gru, f"bias_hh_l{layer}")

                dst_ih  = getattr(fwd_gru, f"weight_ih_l{layer}")
                dst_hh  = getattr(fwd_gru, f"weight_hh_l{layer}")
                dst_bih = getattr(fwd_gru, f"bias_ih_l{layer}")
                dst_bhh = getattr(fwd_gru, f"bias_hh_l{layer}")

                dst_ih.copy_(src_ih)
                dst_hh.copy_(src_hh)
                dst_bih.copy_(src_bih)
                dst_bhh.copy_(src_bhh)
        return fwd_gru

    # ──────────────────────────────────────────────────────────────────
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:  # type: ignore
        return (
            f"BiGRULyricsModel("
            f"vocab={self.vocab_size}, embed={self.embed_dim}, "
            f"hidden={self.hidden_dim}, layers={self.num_layers}, "
            f"params={self.count_parameters():,})"
        )

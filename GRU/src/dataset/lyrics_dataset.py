"""
src/dataset/lyrics_dataset.py

PyTorch Dataset that:
  - Takes pre-tokenized (integer-encoded) lyric sequences.
  - Slides a fixed-length window over each sequence.
  - Returns (input_ids, target_ids) pairs for next-token prediction.
  - Provides a collate function for dynamic padding.

Also contains build_dataloaders() — the top-level factory function
called by train.py.
"""

from __future__ import annotations

import os
import pickle
import logging
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class LyricsDataset(Dataset):
    """
    Sliding-window next-token-prediction dataset.

    Each item:
        input_ids  : int tensor of shape (seq_len,)
        target_ids : int tensor of shape (seq_len,)  — input shifted by 1

    Args:
        token_sequences : List of integer ID lists (one per song).
        seq_len         : Window length (default: config.SEQ_LEN).
        pad_id          : Padding token ID.
        stride          : Sliding-window stride (default = seq_len // 2).
    """

    def __init__(
        self,
        token_sequences: List[List[int]],
        seq_len: int = config.SEQ_LEN,
        pad_id: int = 0,
        stride: Optional[int] = None,
    ) -> None:
        self.seq_len = seq_len
        self.pad_id  = pad_id
        self.stride  = stride or (seq_len // 2)

        # Build flat list of (sequence, start_idx) windows
        self._windows: List[Tuple[List[int], int]] = []
        for seq in token_sequences:
            # Need at least seq_len + 1 tokens to form one window
            if len(seq) < 2:
                continue
            # Pad short sequences so every song gets ≥1 window
            if len(seq) < seq_len + 1:
                padded = seq + [pad_id] * (seq_len + 1 - len(seq))
                self._windows.append((padded, 0))
                continue
            # Sliding window
            for start in range(0, len(seq) - seq_len, self.stride):
                self._windows.append((seq, start))

        logger.info(
            "LyricsDataset: %d songs → %d sliding windows (seq_len=%d, stride=%d)",
            len(token_sequences), len(self._windows), seq_len, self.stride,
        )

    # ──────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        seq, start = self._windows[idx]
        chunk = seq[start : start + self.seq_len + 1]

        # Pad if needed (last window may be short)
        if len(chunk) < self.seq_len + 1:
            chunk = chunk + [self.pad_id] * (self.seq_len + 1 - len(chunk))

        input_ids  = torch.tensor(chunk[:-1], dtype=torch.long)
        target_ids = torch.tensor(chunk[1:],  dtype=torch.long)
        return input_ids, target_ids


# ─────────────────────────────────────────────
# Collate function
# ─────────────────────────────────────────────

def collate_fn(batch: List[Tuple[Tensor, Tensor]], pad_id: int = 0):
    """
    Collate a batch of (input_ids, target_ids) pairs.
    All sequences from LyricsDataset are the same length (seq_len),
    so no padding is needed here — this is a safety wrapper.
    """
    inputs, targets = zip(*batch)
    return torch.stack(inputs), torch.stack(targets)


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def build_dataloaders(
    token_sequences: List[List[int]],
    pad_id: int = 0,
    seq_len: int = config.SEQ_LEN,
    batch_size: int = config.BATCH_SIZE,
    val_split: float = config.VALIDATION_SPLIT,
    num_workers: int = config.NUM_WORKERS,
    seed: int = config.RANDOM_SEED,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from a list of token sequences.

    Returns:
        train_loader, val_loader
    """
    dataset = LyricsDataset(token_sequences, seq_len=seq_len, pad_id=pad_id)

    n_val   = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)

    _collate = lambda b: collate_fn(b, pad_id=pad_id)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
        drop_last=False,
    )

    logger.info(
        "DataLoaders ready — train: %d batches | val: %d batches",
        len(train_loader), len(val_loader),
    )
    return train_loader, val_loader


# ─────────────────────────────────────────────
# Serialisation helpers
# ─────────────────────────────────────────────

def save_token_sequences(sequences: List[List[int]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(sequences, f)
    logger.info("Saved %d token sequences → %s", len(sequences), path)


def load_token_sequences(path: str) -> List[List[int]]:
    with open(path, "rb") as f:
        seqs = pickle.load(f)
    logger.info("Loaded %d token sequences from %s", len(seqs), path)
    return seqs

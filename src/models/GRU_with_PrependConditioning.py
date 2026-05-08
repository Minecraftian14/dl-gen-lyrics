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

import logging
import os
import pickle
import sys
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import optim
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, random_split

from dl_trainer import Trainer
from generator_core import *
from models import Midnight

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


class config:
    # ─────────────────────────────────────────────
    # Paths
    # ─────────────────────────────────────────────
    DATA_PATH = "data/song_lyrics.csv"
    PROCESSED_DIR = "data/processed"
    SP_MODEL_PREFIX = "data/processed/spm_lyrics"  # SentencePiece model prefix
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    EVAL_OUTPUT_DIR = "eval_outputs"

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

    # ─────────────────────────────────────────────
    # Dataset Filtering
    # ─────────────────────────────────────────────
    LANGUAGE_FILTER = "en"  # keep only English lyrics
    MIN_LYRIC_TOKENS = 30  # drop very short songs
    MAX_LYRIC_TOKENS = 1024  # truncate very long songs
    MAX_ROWS = 200_000  # cap dataset size for feasibility

    # ─────────────────────────────────────────────
    # Genre Configuration
    # ─────────────────────────────────────────────
    SUPPORTED_GENRES = [
        "pop", "rock", "hip-hop", "country", "r&b",
        "metal", "jazz", "indie", "electronic", "folk",
        "punk", "soul", "blues", "reggae", "classical",
        "other"
    ]

    # ─────────────────────────────────────────────
    # Preprocessing
    # ─────────────────────────────────────────────
    TFIDF_TOP_K = 5  # top-K theme keywords per song
    TFIDF_THEME_VOCAB = 200  # max distinct theme tokens added to SPM vocab
    TFIDF_MAX_FEATURES = 10_000  # TF-IDF vocabulary size

    # ─────────────────────────────────────────────
    # Special Tokens
    # ─────────────────────────────────────────────
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"  # beginning of sequence
    EOS_TOKEN = "<eos>"  # end of sequence
    NEWLINE_TOKEN = "<newline>"  # preserves line breaks in lyrics
    GENRE_TOKEN_FMT = "<genre_{genre}>"  # e.g. <genre_rock>
    THEME_TOKEN_FMT = "<theme_{word}>"  # e.g. <theme_love>

    # ─────────────────────────────────────────────
    # SentencePiece Tokenizer
    # ─────────────────────────────────────────────
    SP_VOCAB_SIZE = 16_000
    SP_MODEL_TYPE = "bpe"  # byte-pair encoding
    SP_CHARACTER_COVERAGE = 0.9995

    # ─────────────────────────────────────────────
    # Model Architecture
    # ─────────────────────────────────────────────
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512  # per direction; total = 1024
    NUM_LAYERS = 2
    DROPOUT = 0.3
    BIDIRECTIONAL = True

    # ─────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────
    BATCH_SIZE = 64
    SEQ_LEN = 128  # tokens per training window
    EPOCHS = 30
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP = 1.0
    TEACHER_FORCING_RATIO = 1.0  # 1.0 = always use ground truth
    VALIDATION_SPLIT = 0.05
    RANDOM_SEED = 42
    NUM_WORKERS = 2

    # ─────────────────────────────────────────────
    # Inference / Generation
    # ─────────────────────────────────────────────
    TEMPERATURE = 0.85  # sampling temperature
    TOP_K = 50  # top-k filtering
    TOP_P = 0.92  # nucleus sampling
    MAX_GEN_TOKENS = 300  # max tokens per stanza
    LINES_PER_STANZA = 4

    # ─────────────────────────────────────────────
    # Post-Training Evaluation
    # ─────────────────────────────────────────────
    EVAL_NUM_STANZAS = 3
    EVAL_START_PHRASES = [
        "i walk alone through",
        "she said goodbye to",
        "we dance under the",
        "lost in the sound of",
    ]
    BERTSCORE_MODEL = "distilbert-base-uncased"
    SELF_BLEU_N = 4  # BLEU-n for self-BLEU


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
            vocab_size: int = config.SP_VOCAB_SIZE,
            embed_dim: int = config.EMBEDDING_DIM,
            hidden_dim: int = config.HIDDEN_DIM,
            num_layers: int = config.NUM_LAYERS,
            dropout: float = config.DROPOUT,
            pad_id: int = 0,
            word2vec_weights=None,
            word2vec_frozen=True,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.pad_id = pad_id
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
        self.layer_norm = nn.LayerNorm(hidden_dim * self.num_directions)
        self.proj = nn.Linear(hidden_dim * self.num_directions, hidden_dim)
        self.proj_drop = nn.Dropout(dropout)

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
            input_ids: torch.Tensor,  # (B, T)
            hidden: Optional[torch.Tensor] = None,  # (num_layers*2, B, H)
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
        x = self.embedding(input_ids)  # (B, T, E)
        x = self.embed_drop(x)

        # BiGRU
        gru_out, hidden = self.gru(x, hidden)  # (B, T, 2H), (2*L, B, H)

        # LayerNorm + projection
        out = self.layer_norm(gru_out)  # (B, T, 2H)
        out = F.relu(self.proj(out))  # (B, T, H)
        out = self.proj_drop(out)

        # Logits
        logits = self.output_proj(out)  # (B, T, V)
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
                src_ih = getattr(self.gru, f"weight_ih_l{layer}")
                src_hh = getattr(self.gru, f"weight_hh_l{layer}")
                src_bih = getattr(self.gru, f"bias_ih_l{layer}")
                src_bhh = getattr(self.gru, f"bias_hh_l{layer}")

                dst_ih = getattr(fwd_gru, f"weight_ih_l{layer}")
                dst_hh = getattr(fwd_gru, f"weight_hh_l{layer}")
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
        self.pad_id = pad_id
        self.stride = stride or (seq_len // 2)

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
        chunk = seq[start: start + self.seq_len + 1]

        # Pad if needed (last window may be short)
        if len(chunk) < self.seq_len + 1:
            chunk = chunk + [self.pad_id] * (self.seq_len + 1 - len(chunk))

        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        target_ids = torch.tensor(chunk[1:], dtype=torch.long)
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

    n_val = max(1, int(len(dataset) * val_split))
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


sure_fire_keywords = [
    "chorus", "verse", "stanza", "interlude",
]
two_fire_keywords = [
    "intro", "outro",
    "bridge",
    "hook",
]
other_words = [
    "music",
    "pre-", "post-",
]
structure_tokens = [
    "[", "]", "(", ")", "<", ">", "{", "}",
    ":",
]


class Cobalt(Solution):

    def __init__(self, ds_data: pd.DataFrame):
        """
        Initialize all mid-state helpers
        """

        # For accessing shared tokenizer and embeddings
        self.midnight = Midnight(ds_data, skip_model_loading=True)

        self.ds_data: pd.DataFrame = self.midnight.ds_data
        self.custom_tokens = self.midnight.custom_tokens
        self.genre_to_id = self.midnight.genre_to_id
        self.id_to_genre = self.midnight.id_to_genre
        self.tfidf = self.midnight.tfidf
        self.feature_names = self.midnight.feature_names
        self.vocabulary = self.midnight.vocabulary
        self.embedder = self.midnight.embedder
        self.training_data = self._prepare_training_data()
        self.language_model = self._prepare_language_model()

    @cached()
    def _prepare_training_data(self, ds_data=None):
        if ds_data is None: ds_data = self.ds_data
        lyrics = ds_data['lyrics']
        genre = ds_data['tag'].map(lambda g: f"<genre_{g}>")
        context_words = ds_data['lyrics'].map(lambda l: " ".join([f"<theme_{t}>" for t in self.get_context_words(l)]))
        return genre.str.cat([context_words, lyrics], sep=" ").map(self.tokenize_text)

    def _optimizer(self, parameters):
        return optim.AdamW(parameters, lr=3e-4, weight_decay=1e-5)

    def _model_train_step(self, model, data): return model(data)

    def _model_criteria_step(self, criterion, preds, truth):
        preds = preds[0].permute(0, 2, 1)
        return criterion(preds, truth)

    @cached()
    def _prepare_language_model(self):
        model = BiGRULyricsModel(
            vocab_size=self.vocabulary.vocab_size(),
            embed_dim=512,
            hidden_dim=512,
            num_layers=2,
            dropout=0.3,
            pad_id=0,
            word2vec_weights=self.embedder.embeddings.weight,
            word2vec_frozen=True,
        )
        dataloader = data.DataLoader(
            LyricsDataset(self.training_data),
            batch_size=64,
            shuffle=True,
            collate_fn=collate_fn,
        )
        model.trainer = Trainer(
            model=model,
            train_dataloader=dataloader,
            criterion=nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1),
            optimizer=self._optimizer,
            epochs=1,
            device='cpu',
            record_per_batch_training_loss=True,
            model_train_step=self._model_train_step,
            model_criteria_step=self._model_criteria_step,
        )
        return model

    def clean_text(self, text: str) -> str:
        return self.midnight.clean_text(text)

    def pollute_text(self, text: str) -> str:
        return self.midnight.pollute_text(text)

    def get_context_words(self, text, k=5):
        return self.midnight.get_context_words(text, k=k)

    def annotate_text(self, id: int, k=5) -> Annotation:
        return self.midnight.annotate_text(id, k=k)

    def tokenize_text(self, data: str) -> list[int]:
        return self.midnight.tokenize_text(data)

    def detokenize_ids(self, data: list[int]) -> list[str]:
        return self.midnight.detokenize_ids(data)

    @torch.no_grad()
    def embed_tokens(self, data):
        return self.midnight.embed_tokens(data)

    @torch.no_grad()
    def inference(self,
                  genre: str, context_words: list[str],
                  starting_words="",
                  starting_token="<SONG_START>", end_token="<SONG_END>",
                  max_len=40, temperature=1.0, top_k=50,
                  ):
        def ends_with(sequence, pattern):
            seq_len, pat_len = sequence.size(1), pattern.size(0)
            if seq_len < pat_len: return False
            return torch.equal(sequence[0, seq_len - pat_len:], pattern)

        def sample_top_k(logits, k=50):
            k = min(k, logits.size(-1))
            top_k_logits, top_k_indices = torch.topk(logits, k)
            probs = F.softmax(top_k_logits, dim=-1)
            sampled_idx = torch.multinomial(probs, num_samples=1)
            return top_k_indices[sampled_idx]

        starting_ids = self.tokenize_text(starting_token + starting_words)
        ending_ids = self.tokenize_text(end_token)
        genre_id = self.tokenize_text(f"<genre_{genre}>")
        context_ids = self.tokenize_text(" ".join([f"<theme_{t}>" for t in context_words]))
        annotation_ids = genre_id + context_ids

        starting_ids = annotation_ids + starting_ids

        device = next(self.language_model.parameters()).device
        input_ids = torch.tensor(starting_ids, device=device).unsqueeze(0)
        ending_ids = torch.tensor(ending_ids, device=device)

        self.language_model.eval()
        for _ in range(max_len):
            preds, _ = self.language_model(input_ids)
            preds = preds[0, -1]
            preds = preds / temperature
            next_token = sample_top_k(preds, k=top_k)

            input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)
            if ends_with(input_ids, ending_ids): break

        input_ids = input_ids.squeeze(0).tolist()
        return self.detokenize_ids(input_ids)

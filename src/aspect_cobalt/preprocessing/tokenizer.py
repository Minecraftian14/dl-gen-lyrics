"""
src/preprocessing/tokenizer.py

SentencePiece BPE tokenizer:
  - Trains on the annotated lyrics corpus (including all special tokens).
  - Provides encode / decode helpers used by the dataset and inference.
"""

from __future__ import annotations

import os
import logging
import tempfile
from typing import List, Optional

import sentencepiece as spm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from ..GRU import config

logger = logging.getLogger(__name__)


class LyricsTokenizer:
    """
    Thin wrapper around a SentencePiece BPE model.

    Training
    --------
    tokenizer = LyricsTokenizer()
    tokenizer.train(list_of_annotated_lyric_strings, extra_special_tokens)
    tokenizer.save("data/processed/spm_lyrics")

    Inference
    ---------
    tokenizer = LyricsTokenizer.load("data/processed/spm_lyrics")
    ids  = tokenizer.encode("hello world")
    text = tokenizer.decode([123, 456, 789])
    """

    def __init__(self) -> None:
        self._model: Optional[spm.SentencePieceProcessor] = None
        self.model_path: Optional[str] = None

    # ──────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────
    @property
    def vocab_size(self) -> int:
        self._check_loaded()
        return self._model.get_piece_size()

    @property
    def pad_id(self) -> int:
        return self._model.piece_to_id(config.PAD_TOKEN)

    @property
    def unk_id(self) -> int:
        return self._model.unk_id()

    @property
    def bos_id(self) -> int:
        return self._model.piece_to_id(config.BOS_TOKEN)

    @property
    def eos_id(self) -> int:
        return self._model.piece_to_id(config.EOS_TOKEN)

    @property
    def newline_id(self) -> int:
        return self._model.piece_to_id(config.NEWLINE_TOKEN)

    # ──────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────
    def train(
        self,
        corpus: List[str],
        extra_special_tokens: List[str],
        model_prefix: str = config.SP_MODEL_PREFIX,
    ) -> "LyricsTokenizer":
        """
        Train a SentencePiece BPE model on the given corpus.

        Args:
            corpus:               List of annotated lyric strings.
            extra_special_tokens: All genre/theme tokens + newline token.
            model_prefix:         Output file prefix (writes .model and .vocab).
        """
        logger.info(
            "Training SentencePiece (vocab=%d, type=%s) on %d documents …",
            config.SP_VOCAB_SIZE, config.SP_MODEL_TYPE, len(corpus),
        )

        # Write corpus to a temp file (spm.train requires a file path)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as fh:
            for line in corpus:
                # Each sentence on its own line; <newline> tokens already
                # embedded as text, so we just write the string.
                fh.write(line.strip() + "\n")
            corpus_path = fh.name

        # <pad>/<unk>/<bos>/<eos> are reserved via pad_id/unk_id/bos_id/eos_id;
        # including them in user_defined_symbols too raises a RuntimeError.
        spm_reserved = {
            config.PAD_TOKEN,
            config.UNK_TOKEN,
            config.BOS_TOKEN,
            config.EOS_TOKEN,
        }
        user_defined = [config.NEWLINE_TOKEN]  # always keep newline token
        seen = {config.NEWLINE_TOKEN}
        for t in extra_special_tokens:
            if t not in spm_reserved and t not in seen:
                user_defined.append(t)
                seen.add(t)
        user_symbols = ",".join(user_defined)

        os.makedirs(os.path.dirname(model_prefix), exist_ok=True)

        # Safety check: meta pieces (user_defined) + 4 reserved must fit in vocab
        n_meta = len(user_defined) + 4  # +4 for pad/unk/bos/eos
        if n_meta >= config.SP_VOCAB_SIZE:
            raise ValueError(
                f"Too many special tokens ({n_meta}) for vocab_size={config.SP_VOCAB_SIZE}. "
                "Reduce TFIDF_THEME_VOCAB in config.py."
            )
        logger.info(
            "SPM budget: %d reserved + %d user_defined = %d meta | %d for BPE subwords",
            4, len(user_defined), n_meta, config.SP_VOCAB_SIZE - n_meta,
        )

        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=model_prefix,
            vocab_size=config.SP_VOCAB_SIZE,
            model_type=config.SP_MODEL_TYPE,
            character_coverage=config.SP_CHARACTER_COVERAGE,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece=config.PAD_TOKEN,
            unk_piece=config.UNK_TOKEN,
            bos_piece=config.BOS_TOKEN,
            eos_piece=config.EOS_TOKEN,
            user_defined_symbols=user_symbols,
            # Treat whitespace before piece as part of the piece (byte-level)
            byte_fallback=False,  # disabled: byte pieces eat ~256 vocab slots
            allow_whitespace_only_pieces=True,
            # Do NOT split on special symbols
            split_by_unicode_script=False,
            split_by_number=False,
            split_digits=True,
        )

        os.remove(corpus_path)
        # load() is a classmethod that returns a new object — call the
        # underlying loader directly on self so self._model is populated.
        model_path = model_prefix + ".model"
        self._model = spm.SentencePieceProcessor()
        self._model.load(model_path)
        self.model_path = model_path
        logger.info("SentencePiece trained → %s.model  (vocab=%d)", model_prefix, self.vocab_size)
        return self

    # ──────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────
    def save(self, model_prefix: str = config.SP_MODEL_PREFIX) -> None:
        """No-op: SentencePiece already wrote files during train()."""
        logger.info("SentencePiece model already saved at %s.model", model_prefix)

    @classmethod
    def load(cls, model_prefix: str = config.SP_MODEL_PREFIX) -> "LyricsTokenizer":
        """Load a previously trained SentencePiece model."""
        model_path = model_prefix + ".model"
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"SentencePiece model not found at {model_path}. "
                "Run train.py first."
            )
        obj = cls()
        obj._model = spm.SentencePieceProcessor()
        obj._model.load(model_path)
        obj.model_path = model_path
        logger.info("Loaded SentencePiece model from %s (vocab=%d)", model_path, obj.vocab_size)
        return obj

    # ──────────────────────────────────────────
    # Encoding / Decoding
    # ──────────────────────────────────────────
    def _check_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError("Tokenizer not loaded. Call .train() or .load() first.")

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """Encode a string to a list of token IDs."""
        self._check_loaded()
        ids = self._model.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode a list of token IDs to a string."""
        self._check_loaded()
        if skip_special:
            special = {self.pad_id, self.bos_id, self.eos_id}
            ids = [i for i in ids if i not in special]
        text = self._model.decode(ids)
        # Restore newlines
        text = text.replace(config.NEWLINE_TOKEN, "\n")
        # Clean residual genre/theme tokens from output
        text = _strip_annotation_tokens(text)
        return text.strip()

    def id_to_piece(self, idx: int) -> str:
        self._check_loaded()
        return self._model.id_to_piece(idx)

    def piece_to_id(self, piece: str) -> int:
        self._check_loaded()
        return self._model.piece_to_id(piece)


# ──────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────

import re as _re

_ANNOTATION_RE = _re.compile(r"<(genre|theme)_[^>]+>")


def _strip_annotation_tokens(text: str) -> str:
    """Remove genre/theme special tokens from decoded output."""
    return _ANNOTATION_RE.sub("", text).strip()

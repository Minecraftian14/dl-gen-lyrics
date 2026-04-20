"""
src/inference/generator.py

Lyrics generator — takes a start phrase + genre and generates N stanzas.

Generation strategy
-------------------
* Top-k + nucleus (top-p) sampling with temperature scaling.
* Repetition penalty (CTRL-style) — previously generated tokens have their
  logits divided by `repetition_penalty`, breaking "all my all my" loops.
* N-gram blocking — any n-gram already generated is hard-blocked.
* Synthetic newline fallback — if the model never emits <newline> (common
  after just 1 epoch), a newline is inserted every `tokens_per_line` tokens
  so stanza truncation still works correctly.
* Stanza chaining — hidden state is passed across stanzas; full prompt is
  re-encoded each time (single last-token seeding gave no context).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from ..GRU import config
from ..model.bigru import BiGRULyricsModel
from ..preprocessing.tokenizer import LyricsTokenizer
from ..preprocessing.annotator import (
    genre_to_token,
    normalize_genre,
    ThemeExtractor,
    build_annotation_prefix,
)

logger = logging.getLogger(__name__)

_SYNTHETIC_NL = -2   # sentinel value used in token_ids list for fallback newlines


# ─────────────────────────────────────────────
# Sampling utilities
# ─────────────────────────────────────────────

def _apply_repetition_penalty(
    logits:  torch.Tensor,
    gen_ids: List[int],
    penalty: float,
) -> torch.Tensor:
    """
    CTRL repetition penalty: divide positive logits / multiply negative logits
    for tokens already generated. penalty > 1 makes re-use less likely.
    """
    if penalty == 1.0 or not gen_ids:
        return logits
    for token_id in set(gen_ids):
        if 0 <= token_id < logits.size(0):
            if logits[token_id] > 0:
                logits[token_id] /= penalty
            else:
                logits[token_id] *= penalty
    return logits


def _apply_ngram_block(
    logits:  torch.Tensor,
    gen_ids: List[int],
    n:       int,
) -> torch.Tensor:
    """
    Hard-block tokens that would complete an already-seen n-gram.
    Builds a map of (n-1)-gram -> [next tokens] from gen_ids,
    then blocks those next tokens if the current tail matches.
    """
    if n < 2 or len(gen_ids) < n - 1:
        return logits
    ngram_map: Dict[Tuple, List[int]] = defaultdict(list)
    for i in range(len(gen_ids) - n + 1):
        prefix = tuple(gen_ids[i : i + n - 1])
        ngram_map[prefix].append(gen_ids[i + n - 1])
    current_prefix = tuple(gen_ids[-(n - 1):])
    for token_id in ngram_map.get(current_prefix, []):
        if 0 <= token_id < logits.size(0):
            logits[token_id] = float("-inf")
    return logits


def _top_k_top_p_filter(
    logits: torch.Tensor,
    top_k:  int,
    top_p:  float,
) -> torch.Tensor:
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_val = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth_val, float("-inf"))
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = (cum_probs - F.softmax(sorted_logits, dim=-1)) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)
    return logits


def _sample_token(
    logits:             torch.Tensor,
    temperature:        float,
    top_k:              int,
    top_p:              float,
    repetition_penalty: float,
    ngram_block:        int,
    gen_ids:            List[int],
) -> int:
    logits = logits.clone().float()
    if gen_ids:
        logits = _apply_repetition_penalty(logits, gen_ids, repetition_penalty)
        if ngram_block > 0:
            logits = _apply_ngram_block(logits, gen_ids, ngram_block)
    logits = logits / max(temperature, 1e-8)
    logits = _top_k_top_p_filter(logits, top_k, top_p)
    # Safety: if all -inf (over-blocked), fall back to unrestricted top-10
    if not torch.isfinite(logits).any():
        logits = _top_k_top_p_filter(logits.clone().fill_(0.0), top_k=10, top_p=1.0)
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


# ─────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────

class LyricsGenerator:
    """
    End-to-end lyrics generator with repetition penalty and n-gram blocking.

    Args:
        model              : Trained BiGRULyricsModel.
        tokenizer          : Fitted LyricsTokenizer.
        extractor          : Fitted ThemeExtractor (optional).
        device             : torch.device.
        repetition_penalty : CTRL penalty > 1.0 (1.0 = disabled).
        ngram_block        : Block n-grams of this size (0 = disabled).
        tokens_per_line    : Synthetic newline fallback cadence in tokens.
    """

    def __init__(
        self,
        model:              BiGRULyricsModel,
        tokenizer:          LyricsTokenizer,
        extractor:          Optional[ThemeExtractor] = None,
        device:             Optional[torch.device]   = None,
        repetition_penalty: float = 1.3,
        ngram_block:        int   = 3,
        tokens_per_line:    int   = 10,
    ) -> None:
        self.model              = model.eval()
        self.tokenizer          = tokenizer
        self.extractor          = extractor
        self.repetition_penalty = repetition_penalty
        self.ngram_block        = ngram_block
        self.tokens_per_line    = tokens_per_line
        self.device             = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self._eos_id     = tokenizer.eos_id
        self._pad_id     = tokenizer.pad_id
        self._newline_id = tokenizer.newline_id

        # If <newline> landed on a reserved id (0-3) it wasn't truly registered
        if self._newline_id <= 3:
            logger.warning(
                "<newline> token id=%d (≤3 means it's a reserved slot, not truly "
                "registered). Using synthetic line-break every %d tokens.",
                self._newline_id, tokens_per_line,
            )
            self._use_synthetic_nl = True
        else:
            self._use_synthetic_nl = False

        logger.info(
            "Generator | newline_id=%d synthetic_nl=%s "
            "rep_penalty=%.2f ngram_block=%d tokens_per_line=%d",
            self._newline_id, self._use_synthetic_nl,
            repetition_penalty, ngram_block, tokens_per_line,
        )

    # ──────────────────────────────────────────
    def _build_prompt_ids(self, start_phrase: str, genre: str) -> List[int]:
        genre_norm   = normalize_genre(genre)
        theme_tokens = []
        if self.extractor is not None:
            theme_tokens = self.extractor.get_theme_tokens(start_phrase)
        prefix      = build_annotation_prefix(genre_norm, theme_tokens)
        full_prompt = f"{prefix} {start_phrase.lower().strip()}"
        ids         = self.tokenizer.encode(full_prompt, add_bos=False, add_eos=False)
        logger.debug("Prompt (%d tokens): %s …", len(ids), ids[:15])
        return ids

    # ──────────────────────────────────────────
    @torch.no_grad()
    def _generate_tokens(
        self,
        prompt_ids:     List[int],
        max_tokens:     int,
        temperature:    float,
        top_k:          int,
        top_p:          float,
        initial_hidden: Optional[torch.Tensor],
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Auto-regressive generation from prompt_ids.

        Token list uses _SYNTHETIC_NL (-2) as a newline sentinel when
        the model is in synthetic-newline mode.
        """
        prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        _, hidden = self.model(prompt_t, initial_hidden)

        generated:           List[int] = []
        current_id:          int       = prompt_ids[-1]
        tokens_since_nl:     int       = 0

        for _ in range(max_tokens):
            x = torch.tensor([[current_id]], dtype=torch.long, device=self.device)
            logits, hidden = self.model(x, hidden)
            logits_vec = logits[0, 0]

            next_id = _sample_token(
                logits_vec,
                temperature        = temperature,
                top_k              = top_k,
                top_p              = top_p,
                repetition_penalty = self.repetition_penalty,
                ngram_block        = self.ngram_block,
                gen_ids            = generated[-64:],
            )

            if next_id == self._eos_id:
                break

            # Synthetic newline injection
            if self._use_synthetic_nl:
                tokens_since_nl += 1
                if tokens_since_nl >= self.tokens_per_line:
                    generated.append(_SYNTHETIC_NL)
                    tokens_since_nl = 0
            else:
                if next_id == self._newline_id:
                    tokens_since_nl = 0
                else:
                    tokens_since_nl += 1

            generated.append(next_id)
            current_id = next_id

        return generated, hidden

    # ──────────────────────────────────────────
    def generate_stanzas(
        self,
        start_phrase:          str,
        genre:                 str,
        num_stanzas:           int   = config.EVAL_NUM_STANZAS,
        lines_per_stanza:      int   = config.LINES_PER_STANZA,
        max_tokens_per_stanza: int   = config.MAX_GEN_TOKENS,
        temperature:           float = config.TEMPERATURE,
        top_k:                 int   = config.TOP_K,
        top_p:                 float = config.TOP_P,
    ) -> str:
        logger.info(
            "Generating %d stanza(s) | genre='%s' | start='%s'",
            num_stanzas, genre, start_phrase,
        )
        nl_sentinel = _SYNTHETIC_NL if self._use_synthetic_nl else self._newline_id
        hidden      = None
        stanzas:    List[str] = []

        for _ in range(num_stanzas):
            prompt_ids = self._build_prompt_ids(start_phrase, genre)

            token_ids, hidden = self._generate_tokens(
                prompt_ids     = prompt_ids,
                max_tokens     = max_tokens_per_stanza,
                temperature    = temperature,
                top_k          = top_k,
                top_p          = top_p,
                initial_hidden = hidden,
            )

            token_ids = _truncate_to_n_newlines(token_ids, nl_sentinel, lines_per_stanza)
            text      = _decode_tokens(token_ids, self.tokenizer, nl_sentinel)
            text      = _post_process(text)
            stanzas.append(text)

        header = f"[Genre: {normalize_genre(genre).title()}]\n\n"
        return header + "\n\n".join(stanzas)

    # ──────────────────────────────────────────
    def generate_raw_tokens(
        self,
        start_phrase: str,
        genre:        str,
        max_tokens:   int   = config.MAX_GEN_TOKENS,
        temperature:  float = config.TEMPERATURE,
        top_k:        int   = config.TOP_K,
        top_p:        float = config.TOP_P,
    ) -> List[int]:
        prompt_ids   = self._build_prompt_ids(start_phrase, genre)
        token_ids, _ = self._generate_tokens(
            prompt_ids, max_tokens, temperature, top_k, top_p, None
        )
        return [t for t in token_ids if t >= 0]


# ─────────────────────────────────────────────
# Decoding & post-processing
# ─────────────────────────────────────────────

def _truncate_to_n_newlines(
    token_ids:  List[int],
    nl_id:      int,
    n:          int,
) -> List[int]:
    count = 0
    for i, tid in enumerate(token_ids):
        if tid == nl_id:
            count += 1
            if count >= n:
                return token_ids[: i + 1]
    return token_ids


def _decode_tokens(
    token_ids: List[int],
    tokenizer: LyricsTokenizer,
    nl_id:     int,
) -> str:
    """
    Decode token_ids to a string.
    Synthetic newline sentinels (_SYNTHETIC_NL = -2) become real "\n".
    Real <newline> tokens are handled by tokenizer.decode().
    """
    if nl_id == _SYNTHETIC_NL:
        # Walk token list, split on synthetic sentinels
        parts:   List[str] = []
        segment: List[int] = []
        for tid in token_ids:
            if tid == _SYNTHETIC_NL:
                if segment:
                    parts.append(tokenizer.decode(segment, skip_special=True))
                    segment = []
                parts.append("\n")
            elif tid >= 0:
                segment.append(tid)
        if segment:
            parts.append(tokenizer.decode(segment, skip_special=True))
        return "".join(parts)
    else:
        real = [t for t in token_ids if t >= 0]
        return tokenizer.decode(real, skip_special=True)


def _post_process(text: str) -> str:
    import re
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [ln.strip().capitalize() if ln.strip() else "" for ln in text.split("\n")]
    return "\n".join(lines).strip()

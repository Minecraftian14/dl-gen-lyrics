"""
generate.py — Inference entry point for the BiGRU Lyrics Generation System.

Given a trained checkpoint, a start phrase, and a genre, generates lyrics
and prints them to stdout (and optionally saves to a file).

Usage
-----
  python generate.py \
      --checkpoint checkpoints/best_model.pt \
      --start_phrase "i walk alone in" \
      --genre "rock" \
      --num_stanzas 3

  python generate.py \
      --checkpoint checkpoints/best_model.pt \
      --start_phrase "she said goodbye to" \
      --genre "pop" \
      --num_stanzas 4 \
      --temperature 0.9 \
      --top_k 60 \
      --top_p 0.95 \
      --output generated_lyrics.txt
"""

from __future__ import annotations

import os
import sys
import pickle
import logging
import argparse

import torch

import config
from ..preprocessing.tokenizer  import LyricsTokenizer
from ..training.trainer         import Trainer
from ..inference.generator      import LyricsGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate lyrics with BiGRU")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a saved checkpoint (.pt file).",
    )
    parser.add_argument(
        "--start_phrase",
        type=str,
        required=True,
        help="Starting phrase for lyrics generation (3-4 words recommended).",
    )
    parser.add_argument(
        "--genre",
        type=str,
        required=True,
        choices=config.SUPPORTED_GENRES,
        help=f"Music genre. Choices: {config.SUPPORTED_GENRES}",
    )
    parser.add_argument(
        "--num_stanzas",
        type=int,
        default=config.EVAL_NUM_STANZAS,
        help="Number of stanzas to generate.",
    )
    parser.add_argument(
        "--lines_per_stanza",
        type=int,
        default=config.LINES_PER_STANZA,
        help="Lines per stanza.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=config.TEMPERATURE,
        help="Sampling temperature (higher = more random).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=config.TOP_K,
        help="Top-K sampling cutoff.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.3,
        help="Repetition penalty > 1 suppresses token re-use (1.0 = disabled).",
    )
    parser.add_argument(
        "--ngram_block",
        type=int,
        default=3,
        help="Block repeated n-grams of this size during generation (0 = disabled).",
    )
    parser.add_argument(
        "--tokens_per_line",
        type=int,
        default=10,
        help="Fallback: insert newline every N tokens if model never emits <newline>.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=config.TOP_P,
        help="Nucleus sampling probability threshold.",
    )
    parser.add_argument(
        "--sp_model",
        type=str,
        default=config.SP_MODEL_PREFIX,
        help="SentencePiece model prefix (without .model extension).",
    )
    parser.add_argument(
        "--extractor",
        type=str,
        default=os.path.join(config.PROCESSED_DIR, "theme_extractor.pkl"),
        help="Path to saved ThemeExtractor pickle.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save generated lyrics as a .txt file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override: 'cpu' or 'cuda'. Auto-detects if not set.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Device: %s", device)

    # ── Load tokenizer ────────────────────────────────────────────────
    tokenizer = LyricsTokenizer.load(args.sp_model)

    # ── Load ThemeExtractor (optional; improves conditioning quality) ──
    extractor = None
    if os.path.exists(args.extractor):
        with open(args.extractor, "rb") as f:
            extractor = pickle.load(f)
        logger.info("Loaded ThemeExtractor from %s", args.extractor)
    else:
        logger.warning(
            "ThemeExtractor not found at %s — generating without theme tokens.",
            args.extractor,
        )

    # ── Load model ───────────────────────────────────────────────────
    model, ckpt_meta = Trainer.load_checkpoint(args.checkpoint, device=device)
    logger.info(
        "Loaded model: epoch=%d, val_loss=%.4f",
        ckpt_meta["epoch"], ckpt_meta["val_loss"],
    )

    # ── Build generator ───────────────────────────────────────────────
    generator = LyricsGenerator(
        model=model,
        tokenizer=tokenizer,
        extractor=extractor,
        device=device,
        repetition_penalty=args.repetition_penalty,
        ngram_block=args.ngram_block,
        tokens_per_line=args.tokens_per_line,
    )

    # ── Generate ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Generating lyrics for genre='{args.genre}'")
    print(f"Start phrase : '{args.start_phrase}'")
    print(f"Stanzas      : {args.num_stanzas}")
    print(f"Temperature  : {args.temperature}  Top-k: {args.top_k}  Top-p: {args.top_p}")
    print("=" * 60 + "\n")

    lyrics = generator.generate_stanzas(
        start_phrase=args.start_phrase,
        genre=args.genre,
        num_stanzas=args.num_stanzas,
        lines_per_stanza=args.lines_per_stanza,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    print(lyrics)
    print("\n" + "=" * 60)

    # ── Save ─────────────────────────────────────────────────────────
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(f"Genre: {args.genre}\n")
            f.write(f"Start phrase: {args.start_phrase}\n\n")
            f.write(lyrics)
        logger.info("Lyrics saved → %s", args.output)


if __name__ == "__main__":
    main()

"""
preprocess.py — Run ONCE to prepare all data for training.

Saves to data/processed/:
  token_sequences.pkl   — list of integer-encoded lyric sequences
  spm_lyrics.model      — trained SentencePiece BPE model
  spm_lyrics.vocab      — vocabulary file
  theme_extractor.pkl   — fitted TF-IDF ThemeExtractor

Usage
-----
  python preprocess.py --data_path data/song_lyrics.csv

Then train any number of times without re-preprocessing:
  python train.py --output_dir checkpoints/
"""

from __future__ import annotations

import os
import sys
import pickle
import logging
import argparse
import random

import numpy as np
import pandas as pd

import config
from ..preprocessing.cleaner   import clean_lyrics, clean_genre
from ..preprocessing.annotator import annotate_dataframe, ThemeExtractor
from ..preprocessing.tokenizer import LyricsTokenizer
from ..dataset.lyrics_dataset   import save_token_sequences


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(config.LOG_DIR, "preprocess.log")),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Paths for cached artefacts
# ─────────────────────────────────────────────

SEQ_CACHE       = os.path.join(config.PROCESSED_DIR, "token_sequences.pkl")
EXTRACTOR_CACHE = os.path.join(config.PROCESSED_DIR, "theme_extractor.pkl")
SP_MODEL        = config.SP_MODEL_PREFIX + ".model"


def _all_cached() -> bool:
    return (
        os.path.exists(SEQ_CACHE)
        and os.path.exists(EXTRACTOR_CACHE)
        and os.path.exists(SP_MODEL)
    )


# ─────────────────────────────────────────────
# Step 1 & 2: Load and clean
# ─────────────────────────────────────────────

def load_and_clean(data_path: str) -> pd.DataFrame:
    logger.info("Loading dataset from %s …", data_path)
    df = pd.read_csv(
        data_path,
        usecols=["title", "tag", "artist", "lyrics", "language"],
    )
    logger.info("Raw rows: %d", len(df))

    df = df[df["language"] == config.LANGUAGE_FILTER].copy()
    logger.info("After language filter (%s): %d rows", config.LANGUAGE_FILTER, len(df))

    df = df.dropna(subset=["lyrics", "tag"])

    if len(df) > config.MAX_ROWS:
        df = df.sample(n=config.MAX_ROWS, random_state=config.RANDOM_SEED)
        logger.info("Sampled %d rows.", config.MAX_ROWS)

    logger.info("Cleaning lyrics …")
    df["lyrics"] = df["lyrics"].apply(clean_lyrics)
    df["tag"]    = df["tag"].apply(clean_genre)

    df = df[df["lyrics"].str.len() > 10]
    df["approx_tokens"] = df["lyrics"].str.split().apply(len)
    df = df[
        (df["approx_tokens"] >= config.MIN_LYRIC_TOKENS) &
        (df["approx_tokens"] <= config.MAX_LYRIC_TOKENS)
    ]
    logger.info("After cleaning + length filter: %d rows", len(df))
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# Step 3: Annotate
# ─────────────────────────────────────────────

def annotate(df: pd.DataFrame):
    logger.info("Annotating (genre tokens + TF-IDF theme extraction) …")
    df, extractor = annotate_dataframe(df, extractor=None)

    with open(EXTRACTOR_CACHE, "wb") as f:
        pickle.dump(extractor, f)
    logger.info("ThemeExtractor saved → %s", EXTRACTOR_CACHE)
    return df, extractor


# ─────────────────────────────────────────────
# Step 4 & 5: Train tokenizer + encode
# ─────────────────────────────────────────────

def tokenise(df: pd.DataFrame, extractor: ThemeExtractor):
    tokenizer = LyricsTokenizer()

    extra_specials = extractor.get_all_genre_tokens() + extractor.get_all_theme_tokens()
    seen, dedup = set(), []
    for t in extra_specials:
        if t not in seen:
            dedup.append(t)
            seen.add(t)

    tokenizer.train(
        corpus=df["annotated"].tolist(),
        extra_special_tokens=dedup,
    )

    logger.info("Encoding all songs …")
    sequences = [
        tokenizer.encode(text, add_bos=False, add_eos=True)
        for text in df["annotated"]
    ]
    save_token_sequences(sequences, SEQ_CACHE)
    logger.info("Token sequences saved → %s", SEQ_CACHE)
    return tokenizer


# ─────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess the Genius lyrics dataset (run once before training)."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=config.DATA_PATH,
        help="Path to song_lyrics.csv",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run preprocessing even if cached files already exist.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if _all_cached() and not args.force:
        logger.info(
            "All preprocessed files already exist in %s.\n"
            "  %s\n  %s\n  %s\n"
            "Skipping. Use --force to rerun.",
            config.PROCESSED_DIR, SEQ_CACHE, EXTRACTOR_CACHE, SP_MODEL,
        )
        return

    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    df              = load_and_clean(args.data_path)
    df, extractor   = annotate(df)
    tokenizer       = tokenise(df, extractor)

    logger.info("=" * 50)
    logger.info("Preprocessing complete.")
    logger.info("  Vocab size    : %d", tokenizer.vocab_size)
    logger.info("  SPM model     : %s", SP_MODEL)
    logger.info("  Sequences     : %s", SEQ_CACHE)
    logger.info("  Extractor     : %s", EXTRACTOR_CACHE)
    logger.info("You can now run: python train.py")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()

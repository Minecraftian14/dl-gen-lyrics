"""
train.py — Training entry point for the BiGRU Lyrics Generation System.

Assumes preprocessing has already been run via:
  python preprocess.py --data_path data/song_lyrics.csv

Loads cached artefacts from data/processed/ and goes straight to training.

Usage
-----
  python train.py
  python train.py --output_dir checkpoints/ --epochs 30
  python train.py --resume checkpoints/epoch_005.pt   # resume from checkpoint
  python train.py --no_eval                            # skip post-training eval
"""

from __future__ import annotations

import os
import sys
import json
import logging
import argparse
import pickle
import random

import numpy as np
import torch

import config
from src.preprocessing.tokenizer import LyricsTokenizer
from src.dataset.lyrics_dataset  import build_dataloaders, load_token_sequences
from src.model.bigru             import BiGRULyricsModel
from src.training.trainer        import Trainer
from src.inference.generator     import LyricsGenerator
from src.evaluation.evaluator    import Evaluator


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(config.LOG_DIR, "train.log")),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

SEQ_CACHE       = os.path.join(config.PROCESSED_DIR, "token_sequences.pkl")
EXTRACTOR_CACHE = os.path.join(config.PROCESSED_DIR, "theme_extractor.pkl")
SP_MODEL_PREFIX = config.SP_MODEL_PREFIX


def _check_preprocessed() -> None:
    """Abort early with a clear message if preprocessing hasn't been run."""
    missing = []
    if not os.path.exists(SEQ_CACHE):
        missing.append(SEQ_CACHE)
    if not os.path.exists(EXTRACTOR_CACHE):
        missing.append(EXTRACTOR_CACHE)
    if not os.path.exists(SP_MODEL_PREFIX + ".model"):
        missing.append(SP_MODEL_PREFIX + ".model")
    if missing:
        logger.error(
            "Preprocessed files not found:\n  %s\n\n"
            "Run preprocessing first:\n"
            "  python preprocess.py --data_path data/song_lyrics.csv",
            "\n  ".join(missing),
        )
        sys.exit(1)


# ─────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the BiGRU Lyrics Model (preprocessing must be done first)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=config.CHECKPOINT_DIR,
        help="Directory to save checkpoints (default: checkpoints/).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.BATCH_SIZE,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config.LEARNING_RATE,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="Path to a checkpoint .pt file to resume training from.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early-stopping patience (epochs without val-loss improvement).",
    )
    parser.add_argument(
        "--no_eval",
        action="store_true",
        help="Skip post-training evaluation.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    set_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Guard: fail fast if preprocessing wasn't done ─────────────────
    _check_preprocessed()

    # ── Load cached artefacts ─────────────────────────────────────────
    logger.info("Loading preprocessed artefacts from %s …", config.PROCESSED_DIR)
    tokenizer = LyricsTokenizer.load(SP_MODEL_PREFIX)
    sequences = load_token_sequences(SEQ_CACHE)

    with open(EXTRACTOR_CACHE, "rb") as f:
        extractor = pickle.load(f)
    logger.info(
        "Loaded: vocab=%d | sequences=%d | extractor fitted=%s",
        tokenizer.vocab_size, len(sequences), extractor._vectorizer is not None,
    )

    # ── DataLoaders ───────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        token_sequences=sequences,
        pad_id=tokenizer.pad_id,
        batch_size=args.batch_size,
    )

    # ── Model ─────────────────────────────────────────────────────────
    start_epoch = 1
    if args.resume:
        logger.info("Resuming from checkpoint: %s", args.resume)
        model, ckpt = Trainer.load_checkpoint(args.resume, device=device)
        start_epoch = ckpt["epoch"] + 1
        logger.info("Resuming from epoch %d", start_epoch)
    else:
        model = BiGRULyricsModel(
            vocab_size=tokenizer.vocab_size,
            embed_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            pad_id=tokenizer.pad_id,
        )

    logger.info("Model: %s", model)
    logger.info("Trainable parameters: %s", f"{model.count_parameters():,}")

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=args.output_dir,
    )

    if args.resume and args.resume:
        # Restore optimizer + scheduler state for true resume
        ckpt = torch.load(args.resume, map_location=device)
        if "optimizer_state_dict" in ckpt:
            trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            trainer.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        trainer.best_val_loss = ckpt.get("val_loss", float("inf"))
        trainer.best_epoch    = ckpt.get("epoch", 0)

    if args.lr != config.LEARNING_RATE:
        for pg in trainer.optimizer.param_groups:
            pg["lr"] = args.lr

    # ── Train ─────────────────────────────────────────────────────────
    trainer.train(epochs=args.epochs, patience=args.patience)

    # ── Post-training evaluation ───────────────────────────────────────
    if not args.no_eval:
        logger.info("Running post-training evaluation …")
        generator = LyricsGenerator(
            model=model,
            tokenizer=tokenizer,
            extractor=extractor,
            device=device,
        )
        evaluator = Evaluator(
            model=model,
            tokenizer=tokenizer,
            generator=generator,
            device=device,
        )
        for genre in ["pop", "rock", "hip-hop"]:
            results = evaluator.run_full_evaluation(
                start_phrases=config.EVAL_START_PHRASES,
                genre=genre,
                references=None,
            )
            evaluator.print_report(results)
            out_path = os.path.join(config.EVAL_OUTPUT_DIR, f"eval_{genre}.json")
            with open(out_path, "w") as f:
                json.dump(
                    {k: v for k, v in results.items() if k != "samples"},
                    f, indent=2,
                )
            logger.info("Saved eval results → %s", out_path)

    logger.info("All done.")


if __name__ == "__main__":
    main()

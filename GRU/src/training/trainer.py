"""
src/training/trainer.py

Training loop for the BiGRU lyrics model.

Key design decisions
--------------------
* Teacher Forcing ratio = 1.0 — the ground-truth previous token is always
  fed as input (rather than the model's own prediction). This stabilises
  learning and prevents error accumulation across timesteps.
* Loss: Categorical Cross-Entropy (nn.CrossEntropyLoss, ignores pad tokens).
* Per-epoch training BLEU: computed over a small fixed reference subset to
  track linguistic quality without full generation rollout.
* Checkpoint strategy: save best model by validation loss.
* TensorBoard logging for loss and BLEU curves.
"""

from __future__ import annotations

import os
import time
import math
import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import config
from src.model.bigru import BiGRULyricsModel

logger = logging.getLogger(__name__)

smoother = SmoothingFunction().method4


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _compute_training_bleu(
    model: BiGRULyricsModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
    n: int = 4,
) -> float:
    """
    Compute corpus BLEU on a subset of the validation loader using
    the model's greedy predictions (argmax, NOT generation rollout).
    This is the per-epoch evaluation metric as defined in the spec:
    the model predicts the next token given the ground-truth prefix.
    """
    model.eval()
    references_all: List[List[List[str]]] = []
    hypotheses_all: List[List[str]]       = []

    with torch.no_grad():
        for i, (input_ids, target_ids) in enumerate(loader):
            if i >= max_batches:
                break
            input_ids  = input_ids.to(device)   # (B, T)
            target_ids = target_ids.to(device)  # (B, T)

            logits, _ = model(input_ids)          # (B, T, V)
            preds = logits.argmax(dim=-1)          # (B, T)

            # Convert IDs → string tokens for BLEU
            for b in range(preds.size(0)):
                hyp = [str(t.item()) for t in preds[b]]
                ref = [str(t.item()) for t in target_ids[b]]
                hypotheses_all.append(hyp)
                references_all.append([ref])

    if not hypotheses_all:
        return 0.0

    bleu = corpus_bleu(
        references_all,
        hypotheses_all,
        weights=tuple([1.0 / n] * n),
        smoothing_function=smoother,
    )
    return float(bleu)


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────

class Trainer:
    """
    Manages the full training and validation loop.

    Args:
        model       : BiGRULyricsModel instance.
        train_loader: Training DataLoader.
        val_loader  : Validation DataLoader.
        device      : torch.device.
        output_dir  : Directory to save checkpoints.
    """

    def __init__(
        self,
        model:        BiGRULyricsModel,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        device:       torch.device,
        output_dir:   str = config.CHECKPOINT_DIR,
    ) -> None:
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.output_dir   = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Loss — ignore pad tokens
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=model.pad_id,
            label_smoothing=0.1,   # mild smoothing to reduce overconfidence
        )

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=(0.9, 0.999),
        )

        # LR scheduler: cosine annealing over all epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.EPOCHS,
            eta_min=config.LEARNING_RATE * 0.05,
        )

        # TensorBoard
        self.writer = SummaryWriter(log_dir=config.LOG_DIR)

        # State
        self.best_val_loss  = float("inf")
        self.best_epoch     = 0
        self.train_losses:  List[float] = []
        self.val_losses:    List[float] = []
        self.train_bleus:   List[float] = []
        self.val_bleus:     List[float] = []

    # ──────────────────────────────────────────
    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch. Returns average cross-entropy loss."""
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        t0 = time.time()

        for step, (input_ids, target_ids) in enumerate(self.train_loader):
            input_ids  = input_ids.to(self.device)   # (B, T)
            target_ids = target_ids.to(self.device)  # (B, T)

            # ── Teacher Forcing (ratio = 1.0) ─────────────────────────
            # The model always receives ground-truth tokens as input;
            # it predicts the shifted target. No scheduled sampling needed.
            logits, _ = self.model(input_ids)         # (B, T, V)

            # Reshape for CrossEntropyLoss: (B*T, V) vs (B*T,)
            B, T, V = logits.shape
            loss = self.criterion(
                logits.reshape(B * T, V),
                target_ids.reshape(B * T),
            )

            # Gradient update
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP)
            self.optimizer.step()

            # Track non-pad tokens for accurate per-token loss
            non_pad = (target_ids != self.model.pad_id).sum().item()
            total_loss   += loss.item() * non_pad
            total_tokens += non_pad

            if step % 200 == 0:
                elapsed = time.time() - t0
                logger.info(
                    "  Epoch %d | step %d/%d | loss=%.4f | %.1fs",
                    epoch, step, len(self.train_loader), loss.item(), elapsed,
                )

        avg_loss = total_loss / max(total_tokens, 1)
        return avg_loss

    # ──────────────────────────────────────────
    def _validate_epoch(self) -> Tuple[float, float]:
        """Compute validation loss and BLEU. Returns (loss, bleu)."""
        self.model.eval()
        total_loss   = 0.0
        total_tokens = 0

        with torch.no_grad():
            for input_ids, target_ids in self.val_loader:
                input_ids  = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)

                logits, _ = self.model(input_ids)
                B, T, V   = logits.shape
                loss = self.criterion(
                    logits.reshape(B * T, V),
                    target_ids.reshape(B * T),
                )
                non_pad       = (target_ids != self.model.pad_id).sum().item()
                total_loss   += loss.item() * non_pad
                total_tokens += non_pad

        avg_loss = total_loss / max(total_tokens, 1)
        bleu     = _compute_training_bleu(self.model, self.val_loader, self.device)
        return avg_loss, bleu

    # ──────────────────────────────────────────
    def _save_checkpoint(self, epoch: int, val_loss: float, tag: str = "best") -> str:
        path = os.path.join(self.output_dir, f"{tag}_model.pt")
        torch.save(
            {
                "epoch":      epoch,
                "val_loss":   val_loss,
                "model_state_dict":     self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                # Hyper-params embedded for reproducibility
                "hparams": {
                    "vocab_size":  self.model.vocab_size,
                    "embed_dim":   self.model.embed_dim,
                    "hidden_dim":  self.model.hidden_dim,
                    "num_layers":  self.model.num_layers,
                    "dropout":     self.model.dropout_p,
                    "pad_id":      self.model.pad_id,
                },
            },
            path,
        )
        logger.info("Checkpoint saved → %s  (epoch=%d, val_loss=%.4f)", path, epoch, val_loss)
        return path

    # ──────────────────────────────────────────
    def train(
        self,
        epochs:           int  = config.EPOCHS,
        patience:         int  = 5,
        save_every:       int  = 5,
    ) -> BiGRULyricsModel:
        """
        Full training loop.

        Args:
            epochs:     Number of training epochs.
            patience:   Early stopping patience (epochs without val-loss improvement).
            save_every: Also save a checkpoint every N epochs regardless of performance.

        Returns:
            Trained model (loaded from best checkpoint).
        """
        logger.info("Starting training: %d epochs, device=%s", epochs, self.device)
        logger.info("Model: %s", self.model)

        no_improve = 0

        for epoch in range(1, epochs + 1):
            t_epoch = time.time()

            # ── Training ──────────────────────────────────────────────
            train_loss = self._train_epoch(epoch)

            # ── Validation ────────────────────────────────────────────
            val_loss, val_bleu = self._validate_epoch()

            # ── Training BLEU (greedy next-token accuracy proxy) ──────
            train_bleu = _compute_training_bleu(
                self.model, self.train_loader, self.device, max_batches=10
            )

            self.scheduler.step()

            # ── Logging ───────────────────────────────────────────────
            elapsed = time.time() - t_epoch
            ppl = math.exp(min(val_loss, 20))  # cap for display
            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | ppl=%.1f | "
                "train_bleu=%.4f | val_bleu=%.4f | lr=%.2e | %.1fs",
                epoch, epochs,
                train_loss, val_loss, ppl,
                train_bleu, val_bleu,
                self.optimizer.param_groups[0]["lr"],
                elapsed,
            )

            # TensorBoard
            self.writer.add_scalar("Loss/train",      train_loss, epoch)
            self.writer.add_scalar("Loss/val",        val_loss,   epoch)
            self.writer.add_scalar("BLEU/train",      train_bleu, epoch)
            self.writer.add_scalar("BLEU/val",        val_bleu,   epoch)
            self.writer.add_scalar("Perplexity/val",  ppl,        epoch)
            self.writer.add_scalar(
                "LR", self.optimizer.param_groups[0]["lr"], epoch
            )

            # History
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_bleus.append(train_bleu)
            self.val_bleus.append(val_bleu)

            # ── Checkpointing ─────────────────────────────────────────
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch    = epoch
                no_improve         = 0
                self._save_checkpoint(epoch, val_loss, tag="best")
            else:
                no_improve += 1

            if epoch % save_every == 0:
                self._save_checkpoint(epoch, val_loss, tag=f"epoch_{epoch:03d}")

            # ── Early Stopping ────────────────────────────────────────
            if no_improve >= patience:
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    patience,
                )
                break

        logger.info(
            "Training complete. Best epoch: %d, best val_loss: %.4f",
            self.best_epoch, self.best_val_loss,
        )
        self.writer.close()
        return self.model

    # ──────────────────────────────────────────
    @staticmethod
    def load_checkpoint(
        path: str,
        device: Optional[torch.device] = None,
    ) -> Tuple[BiGRULyricsModel, dict]:
        """
        Load model + metadata from a checkpoint file.

        Returns:
            (model, checkpoint_dict)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(path, map_location=device)
        hp   = ckpt["hparams"]

        model = BiGRULyricsModel(
            vocab_size=hp["vocab_size"],
            embed_dim=hp["embed_dim"],
            hidden_dim=hp["hidden_dim"],
            num_layers=hp["num_layers"],
            dropout=hp["dropout"],
            pad_id=hp["pad_id"],
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        logger.info(
            "Loaded checkpoint from %s (epoch=%d, val_loss=%.4f)",
            path, ckpt["epoch"], ckpt["val_loss"],
        )
        return model, ckpt

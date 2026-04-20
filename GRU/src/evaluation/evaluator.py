"""
src/evaluation/evaluator.py

Post-Training Evaluation
------------------------
Computes four metrics on model-generated lyrics:

Reference-based (require human references):
  1. BERTScore — contextual embedding similarity (P, R, F1).
  2. MAUVE     — distributional divergence between human & generated text.

Reference-free (no references needed):
  3. Perplexity   — exponentiated cross-entropy under the model itself.
  4. Self-BLEU    — measures diversity; lower = more diverse outputs.

Usage
-----
from src.evaluation.evaluator import Evaluator

evaluator = Evaluator(model, tokenizer, generator, device)
results = evaluator.run_full_evaluation(
    start_phrases=["i walk alone through", "she said goodbye to"],
    genre="rock",
    references=["actual lyric 1", "actual lyric 2"],  # optional
)
print(results)
"""

from __future__ import annotations

import math
import logging
import random
from itertools import combinations
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from nltk.translate.bleu_score import (
    sentence_bleu,
    corpus_bleu,
    SmoothingFunction,
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import config
from src.model.bigru import BiGRULyricsModel
from src.preprocessing.tokenizer import LyricsTokenizer
from src.inference.generator import LyricsGenerator

logger = logging.getLogger(__name__)
smoother = SmoothingFunction().method4


# ─────────────────────────────────────────────
# Metric implementations
# ─────────────────────────────────────────────

# ── 1. BERTScore ──────────────────────────────

def compute_bertscore(
    hypotheses: List[str],
    references: List[str],
    model_type: str = config.BERTSCORE_MODEL,
    device:     str = "cpu",
) -> Dict[str, float]:
    """
    Compute BERTScore F1 between generated lyrics and human references.

    Args:
        hypotheses : List of generated lyric strings.
        references : List of corresponding reference lyric strings.
        model_type : Underlying BERT model for scoring.
        device     : 'cpu' or 'cuda'.

    Returns:
        Dict with 'precision', 'recall', 'f1' (averaged across pairs).
    """
    try:
        from bert_score import score as bert_score_fn
        P, R, F = bert_score_fn(
            cands=hypotheses,
            refs=references,
            model_type=model_type,
            device=device,
            verbose=False,
        )
        result = {
            "precision": float(P.mean()),
            "recall":    float(R.mean()),
            "f1":        float(F.mean()),
        }
        logger.info("BERTScore: P=%.4f R=%.4f F1=%.4f", result["precision"],
                    result["recall"], result["f1"])
        return result
    except ImportError:
        logger.warning("bert_score not installed. Skipping BERTScore.")
        return {"precision": None, "recall": None, "f1": None}
    except Exception as e:
        logger.error("BERTScore failed: %s", e)
        return {"precision": None, "recall": None, "f1": None}


# ── 2. MAUVE ──────────────────────────────────

def compute_mauve(
    generated_texts: List[str],
    reference_texts: List[str],
    device_id:       int = -1,          # -1 = CPU
    max_text_length: int = 256,
) -> Optional[float]:
    """
    Compute MAUVE score (Pillutla et al., 2021) between generated and
    reference text distributions.

    A score close to 1 means the generated distribution matches human text.

    Args:
        generated_texts : Model outputs.
        reference_texts : Human-written references.
        device_id       : GPU device id (-1 for CPU).
        max_text_length : Truncation length for MAUVE featuriser.

    Returns:
        MAUVE score (float), or None if library is unavailable.
    """
    try:
        import mauve
        out = mauve.compute_mauve(
            p_text=generated_texts,
            q_text=reference_texts,
            device_id=device_id,
            max_text_length=max_text_length,
            verbose=False,
        )
        score = float(out.mauve)
        logger.info("MAUVE score: %.4f", score)
        return score
    except ImportError:
        logger.warning("mauve-text not installed. Skipping MAUVE.")
        return None
    except Exception as e:
        logger.error("MAUVE failed: %s", e)
        return None


# ── 3. Perplexity ─────────────────────────────

def compute_perplexity(
    model:     BiGRULyricsModel,
    tokenizer: LyricsTokenizer,
    texts:     List[str],
    device:    torch.device,
    max_len:   int = config.SEQ_LEN,
) -> float:
    """
    Compute model perplexity on a list of text strings.

    PPL = exp( -1/N * sum_i log P(w_i | w_{<i}) )

    Lower perplexity = better language model fit.

    Args:
        model     : Trained BiGRULyricsModel.
        tokenizer : Fitted LyricsTokenizer.
        texts     : Generated or reference text strings to evaluate.
        device    : Computation device.
        max_len   : Max sequence length for perplexity computation.

    Returns:
        Average perplexity across all texts.
    """
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_id, reduction="sum"
    )
    model.eval()
    total_loss   = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            ids = tokenizer.encode(text, add_bos=True, add_eos=True, max_length=max_len + 1)
            if len(ids) < 2:
                continue
            ids_t = torch.tensor([ids], dtype=torch.long, device=device)
            inp = ids_t[:, :-1]   # (1, T)
            tgt = ids_t[:, 1:]    # (1, T)

            logits, _ = model(inp)          # (1, T, V)
            B, T, V   = logits.shape
            loss = criterion(
                logits.reshape(B * T, V),
                tgt.reshape(B * T),
            )
            non_pad      = (tgt != tokenizer.pad_id).sum().item()
            total_loss   += loss.item()
            total_tokens += non_pad

    if total_tokens == 0:
        return float("inf")

    avg_nll = total_loss / total_tokens
    ppl     = math.exp(min(avg_nll, 100))
    logger.info("Perplexity: %.2f  (avg NLL=%.4f)", ppl, avg_nll)
    return ppl


# ── 4. Self-BLEU ──────────────────────────────

def compute_self_bleu(
    generated_texts: List[str],
    n:               int = config.SELF_BLEU_N,
    max_pairs:       int = 500,
) -> float:
    """
    Compute Self-BLEU — measures how similar generated texts are to each
    other (diversity metric). Lower Self-BLEU = more diverse outputs.

    For each generated text, every OTHER generated text is treated as a
    "reference", and BLEU-n is computed. The average is Self-BLEU.

    Args:
        generated_texts : Multiple generated lyric strings.
        n               : BLEU-n order.
        max_pairs       : Cap on number of pairs to keep runtime feasible.

    Returns:
        Mean Self-BLEU score (float).
    """
    if len(generated_texts) < 2:
        logger.warning("Self-BLEU requires at least 2 texts. Returning 0.0.")
        return 0.0

    # Tokenise at word level (for Self-BLEU, whitespace tokenisation suffices)
    tokenised = [t.lower().split() for t in generated_texts]

    scores: List[float] = []
    pairs  = list(range(len(tokenised)))

    # Randomly sample pairs to avoid O(n^2) cost on large sets
    indices = list(range(len(tokenised)))
    random.shuffle(indices)
    sampled = indices[:max_pairs] if len(indices) > max_pairs else indices

    weights = tuple([1.0 / n] * n)

    for i in sampled:
        refs = [tokenised[j] for j in range(len(tokenised)) if j != i]
        if not refs:
            continue
        hyp = tokenised[i]
        s   = sentence_bleu(
            refs, hyp,
            weights=weights,
            smoothing_function=smoother,
        )
        scores.append(s)

    self_bleu = float(sum(scores) / len(scores)) if scores else 0.0
    logger.info("Self-BLEU-%d: %.4f (diversity: %.4f)", n, self_bleu, 1 - self_bleu)
    return self_bleu


# ─────────────────────────────────────────────
# Evaluator class — orchestrates all metrics
# ─────────────────────────────────────────────

class Evaluator:
    """
    Orchestrates post-training evaluation.

    Args:
        model     : Trained BiGRULyricsModel.
        tokenizer : Fitted LyricsTokenizer.
        generator : LyricsGenerator instance.
        device    : torch.device.
    """

    def __init__(
        self,
        model:     BiGRULyricsModel,
        tokenizer: LyricsTokenizer,
        generator: LyricsGenerator,
        device:    torch.device,
    ) -> None:
        self.model     = model
        self.tokenizer = tokenizer
        self.generator = generator
        self.device    = device

    # ──────────────────────────────────────────
    def generate_samples(
        self,
        start_phrases: List[str],
        genre:         str,
        num_stanzas:   int = config.EVAL_NUM_STANZAS,
    ) -> List[str]:
        """Generate one full lyric per start phrase."""
        samples = []
        for phrase in start_phrases:
            text = self.generator.generate_stanzas(
                start_phrase=phrase,
                genre=genre,
                num_stanzas=num_stanzas,
            )
            samples.append(text)
            logger.info("Generated sample for '%s':\n%s\n", phrase, text[:200])
        return samples

    # ──────────────────────────────────────────
    def run_full_evaluation(
        self,
        start_phrases: List[str],
        genre:         str,
        references:    Optional[List[str]] = None,
        num_stanzas:   int = config.EVAL_NUM_STANZAS,
    ) -> Dict[str, object]:
        """
        Run the full post-training evaluation suite.

        Args:
            start_phrases : List of 3-4 word seed phrases.
            genre         : Target genre for generation.
            references    : Optional human-written reference lyrics (for
                            BERTScore and MAUVE). Must be same length as
                            start_phrases.
            num_stanzas   : Stanzas to generate per sample.

        Returns:
            Dictionary of metric names → values.
        """
        logger.info("=== Post-Training Evaluation ===")
        logger.info("Genre: %s | Phrases: %d | Stanzas/sample: %d",
                    genre, len(start_phrases), num_stanzas)

        # 1. Generate samples
        generated = self.generate_samples(start_phrases, genre, num_stanzas)

        results: Dict[str, object] = {
            "genre":     genre,
            "n_samples": len(generated),
            "samples":   generated,
        }

        # ── Reference-free metrics ─────────────────────────────────────
        logger.info("Computing reference-free metrics …")

        ppl = compute_perplexity(
            self.model, self.tokenizer, generated, self.device
        )
        results["perplexity"] = ppl

        self_bleu = compute_self_bleu(generated, n=config.SELF_BLEU_N)
        results["self_bleu"]  = self_bleu
        results["diversity"]  = round(1.0 - self_bleu, 4)

        # ── Reference-based metrics ────────────────────────────────────
        if references is not None:
            if len(references) != len(generated):
                logger.warning(
                    "references length (%d) != generated length (%d). "
                    "Truncating to shorter.", len(references), len(generated)
                )
                n = min(len(references), len(generated))
                references = references[:n]
                generated_for_ref = generated[:n]
            else:
                generated_for_ref = generated

            logger.info("Computing BERTScore …")
            bert = compute_bertscore(
                generated_for_ref,
                references,
                device=str(self.device),
            )
            results["bertscore"] = bert

            logger.info("Computing MAUVE …")
            mauve_score = compute_mauve(
                generated_for_ref,
                references,
                device_id=0 if self.device.type == "cuda" else -1,
            )
            results["mauve"] = mauve_score
        else:
            results["bertscore"] = "No references provided"
            results["mauve"]     = "No references provided"

        # ── Summary ────────────────────────────────────────────────────
        logger.info("=== Evaluation Results ===")
        for k, v in results.items():
            if k != "samples":
                logger.info("  %-20s %s", k, v)

        return results

    # ──────────────────────────────────────────
    def print_report(self, results: Dict[str, object]) -> None:
        """Pretty-print evaluation results."""
        print("\n" + "=" * 60)
        print("POST-TRAINING EVALUATION REPORT")
        print("=" * 60)
        print(f"Genre          : {results.get('genre')}")
        print(f"Samples        : {results.get('n_samples')}")
        print()
        print("── Reference-Free Metrics ──")
        print(f"  Perplexity   : {results.get('perplexity', 'N/A'):.2f}")
        print(f"  Self-BLEU    : {results.get('self_bleu', 'N/A'):.4f}")
        print(f"  Diversity    : {results.get('diversity', 'N/A'):.4f}")
        print()
        bert = results.get("bertscore")
        if isinstance(bert, dict):
            print("── Reference-Based Metrics ──")
            print(f"  BERTScore P  : {bert.get('precision', 'N/A'):.4f}")
            print(f"  BERTScore R  : {bert.get('recall', 'N/A'):.4f}")
            print(f"  BERTScore F1 : {bert.get('f1', 'N/A'):.4f}")
            print(f"  MAUVE        : {results.get('mauve', 'N/A')}")
        else:
            print(f"  BERTScore    : {bert}")
            print(f"  MAUVE        : {results.get('mauve')}")
        print()
        print("── Generated Samples ──")
        for i, s in enumerate(results.get("samples", []), 1):
            print(f"\n[Sample {i}]\n{s[:500]}{'…' if len(s) > 500 else ''}")
        print("=" * 60)

# BiGRU Lyrics Generation — Architecture & Design Document

## 1. End-to-End Pipeline

```
Genius CSV
    │
    ▼
┌────────────────────────────────────────────────────────┐
│  PREPROCESSING                                          │
│                                                        │
│  cleaner.py        — HTML removal, URL stripping,      │
│                      ASCII artefacts, lowercasing,     │
│                      contraction expansion,            │
│                      punctuation standardisation,      │
│                      <newline> token injection         │
│                                                        │
│  annotator.py      — Genre → <genre_X> token           │
│                      TF-IDF → top-K <theme_X> tokens  │
│                      Builds: "<genre_rock> <theme_love>│
│                               <bos> lyrics… <eos>"    │
│                                                        │
│  tokenizer.py      — SentencePiece BPE (16K vocab)     │
│                      All special tokens in vocab       │
└────────────────────────────────────────────────────────┘
    │
    ▼  token_sequences.pkl
┌────────────────────────────────────────────────────────┐
│  DATASET (lyrics_dataset.py)                           │
│                                                        │
│  Sliding window over token sequences                   │
│  (input_ids[t], target_ids[t] = input_ids[t+1])        │
│  → Train / Val DataLoaders                             │
└────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────┐
│  MODEL (bigru.py)                                      │
│                                                        │
│  Embedding(16K, 256)                                   │
│      │ Dropout(0.3)                                    │
│  BiGRU(256→512×2, 2 layers)   ← teacher forcing       │
│      │ LayerNorm(1024)                                 │
│  Linear(1024→512)                                      │
│      │ ReLU + Dropout(0.3)                             │
│  Linear(512→16K)     ← output logits                  │
└────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────┐
│  TRAINING (trainer.py)                                 │
│                                                        │
│  Loss       : CrossEntropyLoss (pad-masked)            │
│  Optimizer  : AdamW + CosineAnnealingLR                │
│  Teacher Forcing ratio = 1.0 (always ground truth)     │
│  Per-epoch BLEU on val set (greedy next-token pred)    │
│  Checkpoint: best val-loss + every N epochs            │
│  TensorBoard: loss, BLEU, perplexity, LR curves        │
└────────────────────────────────────────────────────────┘
    │
    ▼  best_model.pt
┌────────────────────────────────────────────────────────┐
│  INFERENCE (generator.py)                              │
│                                                        │
│  Input:  start_phrase + genre                          │
│  Prompt: <genre_X> <theme_X>... <bos> <start tokens>  │
│  Sampling: temperature + top-k + nucleus (top-p)       │
│  Stanza: count <newline> tokens until 4 lines          │
│  Chained stanzas via last hidden state                 │
└────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────┐
│  POST-TRAINING EVALUATION (evaluator.py)               │
│                                                        │
│  Reference-free:                                       │
│    Perplexity  — NLL under trained model               │
│    Self-BLEU   — diversity of generated outputs        │
│                                                        │
│  Reference-based (if refs provided):                   │
│    BERTScore   — contextual embedding similarity       │
│    MAUVE       — distributional divergence             │
└────────────────────────────────────────────────────────┘
```

---

## 2. Model Architecture Detail

### Bidirectional GRU

```
Token IDs  →  Embedding(256)  →  BiGRU Layer 1
                                    ├── Forward GRU  (256 → 512)
                                    └── Backward GRU (256 → 512)
                                    concat → 1024
                              →  BiGRU Layer 2
                                    ├── Forward GRU  (1024 → 512)
                                    └── Backward GRU (1024 → 512)
                                    concat → 1024
                              →  LayerNorm(1024)
                              →  Linear(1024 → 512) + ReLU
                              →  Dropout(0.3)
                              →  Linear(512 → vocab_size)
                              →  Logits (16K)
```

**Parameter count** (default config):
- Embedding: 16,000 × 256 ≈ 4.1M
- BiGRU L1:  3 × (256×512 + 512×512) × 2 ≈ 3.1M
- BiGRU L2:  3 × (1024×512 + 512×512) × 2 ≈ 6.3M
- Projection: 1024×512 ≈ 0.5M
- Output: 512×16,000 ≈ 8.2M
- **Total: ~22M parameters**

### Weight Tying
When `embed_dim == hidden_dim`, the output projection shares weights with
the embedding matrix, reducing parameter count and regularising the model.

---

## 3. Preprocessing Design Decisions

### Contraction Expansion
Two-pass approach:
1. `contractions` library — handles standard English contractions (don't, won't)
2. Custom slang map — handles lyric-specific slang (gonna, wanna, gotta, etc.)

### Newline Preservation
Lyric line structure is semantically meaningful (rhyme schemes, rhythm).
Rather than discarding newlines as whitespace, they are converted to
`<newline>` special tokens, allowing the model to learn verse/chorus structure.

### Section Header Removal
`[Chorus]`, `[Verse 1]`, `[Bridge]` labels are removed. The model learns
structural transitions implicitly from the `<newline>` pattern.

---

## 4. Annotation Strategy

### Genre Tokens
```
clean_genre("Hip Hop") → "hip-hop" → "<genre_hip-hop>"
```
16 supported genres; anything unrecognised maps to `<genre_other>`.

### Theme Tokens (TF-IDF)
- Corpus-level TF-IDF with sublinear term frequency (log scaling)
- Extended English + lyrics stop-words removed
- Min 3-character alpha tokens only
- Top-5 keywords per song → `<theme_love>`, `<theme_dark>`, etc.

The annotation prefix acts as soft conditioning:
```
<genre_rock> <theme_fire> <theme_storm> <theme_rage> <bos> i walk alone...
```

---

## 5. Training Details

### Teacher Forcing (ratio = 1.0)
At every timestep, the model receives the ground-truth previous token,
not its own prediction. This:
- Prevents error accumulation during training
- Gives stable, fast convergence
- Is standard practice for RNN language models

### Loss
`CrossEntropyLoss` with:
- `ignore_index=pad_id` — padding doesn't contribute to loss
- `label_smoothing=0.1` — mild regularisation, prevents overconfidence

### Optimiser
`AdamW` with cosine LR annealing:
- Initial LR: 3e-4, final LR: ~1.5e-5
- Weight decay: 1e-5
- Gradient clipping: 1.0

### Checkpoint Strategy
- Save `best_model.pt` whenever validation loss improves
- Save `epoch_NNN.pt` every 5 epochs (recovery points)
- Early stopping with patience=5

---

## 6. Inference (Generation)

### Full Prompt Construction
```
prompt = "<genre_rock> <theme_fire> <theme_night> <bos> i walk alone in"
```

### Sampling Strategy
1. **Temperature scaling**: `logits /= T` (T=0.85 by default)
2. **Top-k filtering**: keep only top-50 logits, mask rest to -inf
3. **Nucleus (top-p)**: keep minimum set summing to ≥ 0.92 cumulative prob
4. **Multinomial sampling**: draw from the filtered distribution

### Stanza Chaining
Hidden state is passed across stanzas, giving the model memory of what
it already "sang". The last generated token seeds the next stanza.

---

## 7. Evaluation Metrics

| Metric      | Type       | What it measures                          | Better |
|-------------|------------|-------------------------------------------|--------|
| BLEU        | Training   | n-gram overlap (next-token accuracy proxy)| Higher |
| Perplexity  | Post-train | How well model predicts its own outputs   | Lower  |
| Self-BLEU   | Post-train | Diversity of multiple generated texts     | Lower  |
| BERTScore   | Post-train | Semantic similarity to human lyrics       | Higher |
| MAUVE       | Post-train | Distributional closeness to human text    | Higher |

---

## 8. File Index

| File                              | Purpose                              |
|-----------------------------------|--------------------------------------|
| `config.py`                       | All hyperparameters and paths        |
| `train.py`                        | Training entry point                 |
| `generate.py`                     | Inference entry point                |
| `src/preprocessing/cleaner.py`    | Text cleaning & normalisation        |
| `src/preprocessing/annotator.py`  | Genre + TF-IDF annotation            |
| `src/preprocessing/tokenizer.py`  | SentencePiece BPE wrapper            |
| `src/dataset/lyrics_dataset.py`   | PyTorch Dataset + DataLoaders        |
| `src/model/bigru.py`              | BiGRU architecture                   |
| `src/training/trainer.py`         | Training loop, checkpointing         |
| `src/inference/generator.py`      | Lyrics generation with sampling      |
| `src/evaluation/evaluator.py`     | Post-training evaluation suite       |

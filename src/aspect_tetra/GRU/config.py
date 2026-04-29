"""
config.py — Central configuration for the BiGRU Lyrics Generation System.
All hyperparameters, paths, and special tokens are defined here.
"""

import os

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
DATA_PATH          = "data/song_lyrics.csv"
PROCESSED_DIR      = "data/processed"
SP_MODEL_PREFIX    = "data/processed/spm_lyrics"   # SentencePiece model prefix
CHECKPOINT_DIR     = "checkpoints"
LOG_DIR            = "logs"
EVAL_OUTPUT_DIR    = "eval_outputs"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Dataset Filtering
# ─────────────────────────────────────────────
LANGUAGE_FILTER    = "en"          # keep only English lyrics
MIN_LYRIC_TOKENS   = 30            # drop very short songs
MAX_LYRIC_TOKENS   = 1024          # truncate very long songs
MAX_ROWS           = 200_000       # cap dataset size for feasibility

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
TFIDF_TOP_K        = 5             # top-K theme keywords per song
TFIDF_THEME_VOCAB  = 200           # max distinct theme tokens added to SPM vocab
TFIDF_MAX_FEATURES = 10_000        # TF-IDF vocabulary size

# ─────────────────────────────────────────────
# Special Tokens
# ─────────────────────────────────────────────
PAD_TOKEN          = "<pad>"
UNK_TOKEN          = "<unk>"
BOS_TOKEN          = "<bos>"       # beginning of sequence
EOS_TOKEN          = "<eos>"       # end of sequence
NEWLINE_TOKEN      = "<newline>"   # preserves line breaks in lyrics
GENRE_TOKEN_FMT    = "<genre_{genre}>"   # e.g. <genre_rock>
THEME_TOKEN_FMT    = "<theme_{word}>"    # e.g. <theme_love>

# ─────────────────────────────────────────────
# SentencePiece Tokenizer
# ─────────────────────────────────────────────
SP_VOCAB_SIZE      = 16_000
SP_MODEL_TYPE      = "bpe"         # byte-pair encoding
SP_CHARACTER_COVERAGE = 0.9995

# ─────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────
EMBEDDING_DIM      = 256
HIDDEN_DIM         = 512           # per direction; total = 1024
NUM_LAYERS         = 2
DROPOUT            = 0.3
BIDIRECTIONAL      = True

# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
BATCH_SIZE         = 20
SEQ_LEN            = 1024           # tokens per training window
EPOCHS             = 30
LEARNING_RATE      = 3e-4
WEIGHT_DECAY       = 1e-5
GRAD_CLIP          = 1.0
TEACHER_FORCING_RATIO = 1.0        # 1.0 = always use ground truth
VALIDATION_SPLIT   = 0.05
RANDOM_SEED        = 42
NUM_WORKERS        = 2

# ─────────────────────────────────────────────
# Inference / Generation
# ─────────────────────────────────────────────
TEMPERATURE        = 0.85          # sampling temperature
TOP_K              = 50            # top-k filtering
TOP_P              = 0.92          # nucleus sampling
MAX_GEN_TOKENS     = 300           # max tokens per stanza
LINES_PER_STANZA   = 4

# ─────────────────────────────────────────────
# Post-Training Evaluation
# ─────────────────────────────────────────────
EVAL_NUM_STANZAS   = 3
EVAL_START_PHRASES = [
    "i walk alone through",
    "she said goodbye to",
    "we dance under the",
    "lost in the sound of",
]
BERTSCORE_MODEL    = "distilbert-base-uncased"
SELF_BLEU_N        = 4             # BLEU-n for self-BLEU

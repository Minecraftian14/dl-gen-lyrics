"""
src/preprocessing/annotator.py

Annotation & extraction stage:
  1. Genre annotation  — maps each song to a <genre_X> special token.
  2. Theme annotation  — uses TF-IDF over the corpus to extract top-K
                         thematic keyword tokens per song (<theme_X>).

The annotated tokens are prepended to each lyric sequence so the model
learns to condition on both genre and thematic context.
"""

from __future__ import annotations

import re
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from ..GRU import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Genre normalisation helpers
# ─────────────────────────────────────────────

_GENRE_ALIAS: dict[str, str] = {
    "rap":          "hip-hop",
    "hip hop":      "hip-hop",
    "hiphop":       "hip-hop",
    "r&b / soul":   "r&b",
    "rhythm and blues": "r&b",
    "rnb":          "r&b",
    "heavy metal":  "metal",
    "hard rock":    "rock",
    "alternative":  "indie",
    "alt":          "indie",
    "electronic dance music": "electronic",
    "edm":          "electronic",
    "dance":        "electronic",
    "classical music": "classical",
    "gospel":       "soul",
    "latin":        "other",
    "world":        "other",
}

_STOPWORDS_EXTRA = {
    # generic filler words common in lyrics but not thematic
    "oh", "ah", "na", "la", "yeah", "hey", "ooh", "uh",
    "gonna", "wanna", "gotta", "like", "just", "know",
    "got", "get", "let", "say", "said", "come", "go",
    "make", "take", "give", "think", "feel", "tell",
    "see", "look", "need", "way", "time", "day", "night",
    "yes", "no", "so", "do", "be", "is", "am", "are",
    "newline",  # our placeholder token
}


def normalize_genre(raw_genre: str) -> str:
    """Map raw genre string to one of the SUPPORTED_GENRES."""
    g = raw_genre.lower().strip()
    # try alias first
    g = _GENRE_ALIAS.get(g, g)
    if g in config.SUPPORTED_GENRES:
        return g
    # partial match
    for supported in config.SUPPORTED_GENRES:
        if supported in g or g in supported:
            return supported
    return "other"


def genre_to_token(genre: str) -> str:
    """Convert genre string → special genre token."""
    g = normalize_genre(genre)
    return config.GENRE_TOKEN_FMT.format(genre=g.replace(" ", "_").replace("&", "and"))


def theme_word_to_token(word: str) -> str:
    """Convert theme keyword → special theme token."""
    safe = re.sub(r"[^a-z0-9_]", "_", word.lower())
    return config.THEME_TOKEN_FMT.format(word=safe)


# ─────────────────────────────────────────────
# TF-IDF Keyword Extractor
# ─────────────────────────────────────────────

class ThemeExtractor:
    """
    Fits a TF-IDF model over the entire lyrics corpus, then extracts
    the top-K thematic keywords per document.

    Usage
    -----
    extractor = ThemeExtractor()
    extractor.fit(list_of_cleaned_lyrics)
    theme_tokens = extractor.get_theme_tokens(lyric_text, top_k=5)
    """

    def __init__(
        self,
        max_features: int = config.TFIDF_MAX_FEATURES,
        top_k: int = config.TFIDF_TOP_K,
    ):
        self.max_features = max_features
        self.top_k = top_k
        self._vectorizer: TfidfVectorizer | None = None
        self._feature_names: np.ndarray | None = None

    # ------------------------------------------------------------------
    def fit(self, corpus: List[str]) -> "ThemeExtractor":
        """Fit TF-IDF on the full corpus (list of cleaned lyric strings)."""
        logger.info("Fitting TF-IDF on %d documents …", len(corpus))

        # Build a combined stop-word list
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        stop = set(ENGLISH_STOP_WORDS) | _STOPWORDS_EXTRA

        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words=list(stop),
            token_pattern=r"(?u)\b[a-z][a-z]{2,}\b",  # min 3-char alpha tokens
            ngram_range=(1, 1),
            sublinear_tf=True,
        )
        self._vectorizer.fit(corpus)
        self._feature_names = np.array(self._vectorizer.get_feature_names_out())
        logger.info("TF-IDF vocabulary size: %d", len(self._feature_names))
        return self

    # ------------------------------------------------------------------
    def get_top_keywords(self, text: str, top_k: int | None = None) -> List[str]:
        """Return top-K thematic keywords for a single document."""
        if self._vectorizer is None:
            raise RuntimeError("Call .fit() before .get_top_keywords()")
        top_k = top_k or self.top_k
        vec = self._vectorizer.transform([text])
        scores = vec.toarray()[0]
        indices = np.argsort(scores)[::-1][:top_k]
        return [self._feature_names[i] for i in indices if scores[i] > 0]

    # ------------------------------------------------------------------
    def get_theme_tokens(self, text: str, top_k: int | None = None) -> List[str]:
        """Return top-K theme tokens (<theme_word>) for a single document."""
        keywords = self.get_top_keywords(text, top_k)
        return [theme_word_to_token(kw) for kw in keywords]

    # ------------------------------------------------------------------
    def get_all_theme_tokens(self, top_n: int | None = None) -> List[str]:
        """
        Return the top-N most globally salient <theme_X> tokens to add
        to the SentencePiece vocabulary.

        We rank by corpus-wide IDF (inverse document frequency): words
        that appear across many songs are the most genre-defining themes.
        Capping at config.TFIDF_THEME_VOCAB (default 200) keeps the
        number of meta_pieces well below the SPM vocab_size limit.
        """
        if self._feature_names is None or self._vectorizer is None:
            return []
        top_n = top_n or getattr(config, 'TFIDF_THEME_VOCAB', 200)
        # IDF scores: higher = more discriminative across documents
        idf_scores = self._vectorizer.idf_
        # Pick indices of the top-N highest-IDF words
        top_indices = idf_scores.argsort()[::-1][:top_n]
        return [theme_word_to_token(self._feature_names[i]) for i in top_indices]

    # ------------------------------------------------------------------
    def get_all_genre_tokens(self) -> List[str]:
        return [genre_to_token(g) for g in config.SUPPORTED_GENRES]


# ─────────────────────────────────────────────
# Annotation pipeline (corpus-level)
# ─────────────────────────────────────────────

def build_annotation_prefix(
    genre: str,
    theme_tokens: List[str],
) -> str:
    """
    Constructs the annotation prefix prepended to each lyric sequence.

    Format:  <genre_rock> <theme_love> <theme_night> <bos>
    """
    g_token = genre_to_token(genre)
    parts = [g_token] + theme_tokens + [config.BOS_TOKEN]
    return " ".join(parts)


def annotate_dataframe(
    df: pd.DataFrame,
    text_col: str = "lyrics",
    genre_col: str = "tag",
    extractor: ThemeExtractor | None = None,
) -> Tuple[pd.DataFrame, ThemeExtractor]:
    """
    Annotate a lyrics dataframe in-place:
      - df['genre_token']   : genre special token string
      - df['theme_tokens']  : space-separated theme token string
      - df['annotated']     : full annotation prefix + cleaned lyrics

    Returns the modified dataframe and the fitted ThemeExtractor.
    """
    logger.info("Building genre tokens …")
    df["genre_token"] = df[genre_col].apply(genre_to_token)

    # Fit or reuse ThemeExtractor
    # Guard: also fit if the extractor exists but has never been fitted yet
    if extractor is None:
        extractor = ThemeExtractor()
    if extractor._vectorizer is None:
        extractor.fit(df[text_col].tolist())

    logger.info("Extracting theme keywords per song (TF-IDF) …")
    # Vectorised: transform the whole corpus at once, then pick top-K per row
    tfidf_matrix = extractor._vectorizer.transform(df[text_col].tolist())
    feature_names = extractor._feature_names

    theme_lists = []
    top_k = extractor.top_k
    for i in range(tfidf_matrix.shape[0]):
        row_scores = tfidf_matrix[i].toarray()[0]
        top_indices = row_scores.argsort()[::-1][:top_k]
        keywords = [feature_names[j] for j in top_indices if row_scores[j] > 0]
        theme_lists.append([theme_word_to_token(kw) for kw in keywords])
    df["theme_tokens"] = [" ".join(t) for t in theme_lists]

    logger.info("Building annotated sequences …")
    theme_token_col = df["theme_tokens"].str.split()
    annotated = [
        f"{build_annotation_prefix(genre, themes)} {lyrics} {config.EOS_TOKEN}"
        for genre, themes, lyrics in zip(
            df[genre_col], theme_token_col, df[text_col]
        )
    ]
    df["annotated"] = annotated

    return df, extractor

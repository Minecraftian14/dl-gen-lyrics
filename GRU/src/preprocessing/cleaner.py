"""
src/preprocessing/cleaner.py

Handles all text cleaning and normalization steps:
  - HTML tag removal
  - Contributor / timestamp noise removal
  - URL stripping
  - ASCII artefact removal
  - Uniform lowercasing
  - Contraction expansion
  - Punctuation standardization
  - Newline preservation (converted to <newline> token placeholder)
"""

import re
import html
import unicodedata
import contractions
from bs4 import BeautifulSoup


# ─────────────────────────────────────────────
# Contraction & Slang Expansion Map
# ─────────────────────────────────────────────
SLANG_MAP = {
    r"\bgonna\b":   "going to",
    r"\bwanna\b":   "want to",
    r"\bgotta\b":   "got to",
    r"\bkinda\b":   "kind of",
    r"\bsorta\b":   "sort of",
    r"\boutta\b":   "out of",
    r"\blotta\b":   "lot of",
    r"\blemme\b":   "let me",
    r"\bgimme\b":   "give me",
    r"\btellin\b":  "telling",
    r"\bcomin\b":   "coming",
    r"\bfeelin\b":  "feeling",
    r"\bwakin\b":   "waking",
    r"\btakin\b":   "taking",
    r"\bmakin\b":   "making",
    r"\bbreaking\b":"breaking",
    r"\bdunno\b":   "do not know",
    r"\byeah\b":    "yes",
    r"\bya\b":      "you",
    r"\bem\b":      "them",
    r"\bcause\b":   "because",
    r"\b'cause\b":  "because",
    r"\bcuz\b":     "because",
}

# Contributor noise patterns commonly found in Genius scrapes
_CONTRIBUTOR_RE = re.compile(
    r"(\d+\s+contributors?).*?lyrics",
    re.IGNORECASE | re.DOTALL
)
_TIMESTAMP_RE = re.compile(
    r"\b(\d{1,2}[:/]\d{2}(\s*[ap]m)?)\b",
    re.IGNORECASE
)
_URL_RE = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE
)
# Repeated punctuation (e.g. "!!!!", "......")
_REPEAT_PUNCT_RE = re.compile(r"([!?.,;:]){2,}")
# Characters that are not printable ASCII or common unicode letters/punctuation
_NON_ASCII_RE = re.compile(r"[^\x20-\x7E\n]")
# Multiple blank lines → single blank line
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
# Section headers like [Chorus], [Verse 1], etc.
_SECTION_HDR_RE = re.compile(r"\[.*?\]")


def _remove_html(text: str) -> str:
    """Strip HTML tags and unescape HTML entities."""
    text = html.unescape(text)
    text = BeautifulSoup(text, "html.parser").get_text(separator="\n")
    return text


def _remove_contributor_noise(text: str) -> str:
    """Remove 'N Contributors Lyrics' headers injected by Genius."""
    text = _CONTRIBUTOR_RE.sub("", text)
    return text


def _remove_urls(text: str) -> str:
    return _URL_RE.sub("", text)


def _remove_timestamps(text: str) -> str:
    return _TIMESTAMP_RE.sub("", text)


def _remove_section_headers(text: str) -> str:
    """Remove bracketed section markers like [Chorus], [Bridge]."""
    return _SECTION_HDR_RE.sub("", text)


def _normalize_unicode(text: str) -> str:
    """Normalize unicode to NFC and replace lookalike characters."""
    text = unicodedata.normalize("NFC", text)
    # Fancy quotes → straight quotes
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # Em/en dashes → hyphen
    text = text.replace("\u2014", " - ").replace("\u2013", " - ")
    # Ellipsis character → three dots
    text = text.replace("\u2026", "...")
    return text


def _remove_non_ascii_artefacts(text: str) -> str:
    """Remove erratic ASCII / non-printable characters."""
    return _NON_ASCII_RE.sub("", text)


def _expand_contractions(text: str) -> str:
    """
    Two-pass expansion:
      1. Library-based (handles can't, won't, I'm, etc.)
      2. Slang map (gonna, wanna, etc.)
    """
    try:
        text = contractions.fix(text)
    except Exception:
        pass  # contractions library may fail on edge cases
    for pattern, replacement in SLANG_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _standardize_punctuation(text: str) -> str:
    """
    - Collapse repeated punctuation: '!!!' → '!'
    - Ensure space after punctuation
    - Remove stray hyphens/underscores used as decorators
    """
    # Collapse repeated punctuation
    text = _REPEAT_PUNCT_RE.sub(r"\1", text)
    # Remove decorator lines like '------' or '______'
    text = re.sub(r"[-_]{3,}", "", text)
    # Ensure space after , . ! ? ; :
    text = re.sub(r"([,.!?;:])(?=[^\s])", r"\1 ", text)
    # Collapse multiple spaces (but NOT newlines)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def _preserve_newlines(text: str) -> str:
    """
    Replace newlines with <newline> placeholder so the tokenizer
    treats them as vocabulary items rather than whitespace.
    Also collapse 3+ consecutive blank lines to 2.
    """
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    text = text.replace("\n", " <newline> ")
    return text


def clean_lyrics(text: str, preserve_newlines: bool = True) -> str:
    """
    Full cleaning pipeline applied to a single lyric string.

    Args:
        text:              Raw lyric string from the dataset.
        preserve_newlines: If True, newlines are converted to <newline> tokens.

    Returns:
        Cleaned, normalized lyric string ready for tokenization.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = _remove_html(text)
    text = _remove_contributor_noise(text)
    text = _remove_urls(text)
    text = _remove_timestamps(text)
    text = _remove_section_headers(text)
    text = _normalize_unicode(text)
    text = _remove_non_ascii_artefacts(text)
    text = text.lower()
    text = _expand_contractions(text)
    text = _standardize_punctuation(text)

    if preserve_newlines:
        text = _preserve_newlines(text)

    # Final strip
    text = text.strip()
    return text


def clean_genre(genre: str) -> str:
    """Normalize a genre string to lowercase, strip whitespace."""
    if not isinstance(genre, str):
        return "other"
    g = genre.lower().strip()
    return g if g else "other"

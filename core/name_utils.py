"""
Shared name normalization utilities used across MLB and NBA prediction models.

The various sportsbooks and stats providers spell player names inconsistently
(accents, hyphens, suffixes). These helpers provide a single source of truth.
"""

import unicodedata
import pandas as pd


def normalize_name(name):
    """
    Normalize a player name for matching.

    - Lowercases
    - Strips accents (e.g. "Pérez" -> "perez", "Jokić" -> "jokic")
    - Removes leading/trailing whitespace

    Returns "" for null/NaN inputs so it's safe to apply to DataFrame columns.
    """
    # pd.isna catches None, float NaN, NaT, and pd.NA in one shot.
    try:
        if pd.isna(name):
            return ""
    except (TypeError, ValueError):
        # pd.isna raises on some custom types; fall through to str() below.
        pass
    text = str(name)
    # NFKD splits accented characters into base + combining mark
    decomposed = unicodedata.normalize('NFKD', text)
    stripped = ''.join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.lower().strip()


def names_match(query_name, candidate_name):
    """
    Returns True if every whitespace-separated token of `query_name`
    appears in the normalized `candidate_name`.

    This avoids the "Cameron Johnson matches Jalen Johnson" problem caused
    by last-name-only matching.
    """
    query_norm = normalize_name(query_name)
    candidate_norm = normalize_name(candidate_name)
    if not query_norm or not candidate_norm:
        return False
    parts = query_norm.split()
    return all(part in candidate_norm for part in parts)


def filter_by_name(df, name_col, query_name):
    """
    Filter `df` to rows whose `name_col` matches `query_name` using full-name
    token matching. Returns a sliced DataFrame (potentially empty).
    """
    if df is None or df.empty:
        return df
    return df[df[name_col].apply(lambda x: names_match(query_name, x))]

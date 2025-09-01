"""
Common tag set and canonicalization utilities for LaMP-2 classification.
"""
from __future__ import annotations

from typing import Optional, Dict, List

# Default closed set (15 labels) aligned with LaMP-2 attributes
DEFAULT_ALLOWED_TAGS: List[str] = [
    "action",
    "based on a book",
    "classic",
    "comedy",
    "dark comedy",
    "dystopia",
    "fantasy",
    "psychology",
    "romance",
    "sci-fi",
    "social commentary",
    "thought-provoking",
    "true story",
    "twist ending",
    "violence",
]

def build_id_maps(labels):
    id2tag = {i+1: t for i, t in enumerate(labels)}
    tag2id = {t: i for i, t in id2tag.items()}
    return id2tag, tag2id

# Backward compatibility
ID2TAG: Dict[int, str] = {i + 1: t for i, t in enumerate(DEFAULT_ALLOWED_TAGS)}
TAG2ID: Dict[str, int] = {t: i for i, t in ID2TAG.items()}

_CANON = {
    "sci fi": "sci-fi",
    "scifi": "sci-fi",
    "science fiction": "sci-fi",
    "thought provoking": "thought-provoking",
    "thoughtprovoking": "thought-provoking",
    "dark-comedy": "dark comedy",
    "darkcomedy": "dark comedy",
    "based on book": "based on a book",
    "based-on-a-book": "based on a book",
    "based-on book": "based on a book",
}


def canonicalize(s: str) -> str:
    s0 = (s or "").strip().lower()
    s0 = s0.replace("_", " ").replace("—", "-").replace("–", "-")
    # Collapse spaces
    s0 = " ".join(s0.split())
    s0 = _CANON.get(s0, s0)
    return s0


def match_allowed(s: str, allowed=None) -> Optional[str]:
    """Return canonical allowed tag if string matches approximately; else None."""
    import re

    allowed = DEFAULT_ALLOWED_TAGS if allowed is None else allowed
    c = canonicalize(s)
    if c in allowed:
        return c
    # Try punctuation-stripped exact and space-insensitive match
    c2 = re.sub(r"[^a-z0-9\- ]+", "", c)
    for t in allowed:
        if c2 == t:
            return t
        if c2.replace(" ", "") == t.replace(" ", ""):
            return t
    return None


__all__ = [
    "DEFAULT_ALLOWED_TAGS",
    "build_id_maps",
    "ID2TAG",
    "TAG2ID",
    "canonicalize",
    "match_allowed",
]


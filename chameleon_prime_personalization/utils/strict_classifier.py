"""
ID-mode prompt builder and output parser for closed-set classification.
"""
from __future__ import annotations

from typing import Optional, Tuple

from .tags import ID2TAG

PROMPT_ID_MODE = (
    "You are a tag classifier.\n"
    "Choose exactly ONE tag ID from the list and output the ID ONLY (integer, no text).\n"
    "Tags:\n{lines}\n\n"
    "Input:\n{content}\n\n"
    "Answer with the ID only on a single line."
)


def build_id_prompt(content: str, *, id2tag: dict) -> str:
    lines = "\n".join([f"{i}: {t}" for i, t in id2tag.items()])
    return PROMPT_ID_MODE.format(lines=lines, content=content or "(no content)")


def parse_id_output(text: str) -> Tuple[Optional[int], Optional[str]]:
    import re
    m = re.search(r"\b(\d{1,2})\b", (text or ""))
    if not m:
        return None, None
    i = int(m.group(1))
    return i, ID2TAG.get(i)


__all__ = ["build_id_prompt", "parse_id_output"]


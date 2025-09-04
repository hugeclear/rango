import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any


# Paths to the expanded dev set
DATA_Q = Path("data/lamp_expanded/dev_questions_expanded.json")
DATA_Y = Path("data/lamp_expanded/dev_outputs_expanded.json")


def _canon(x: str) -> str:
    """Normalize labels to reduce trivial mismatches (e.g., sci fi -> sci-fi)."""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("science fiction", "sci-fi").replace("sci fi", "sci-fi")
    return s


def load_dataset(limit: int | None = None) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Load questions and gold dict (id -> gold).

    The expanded files are expected to be aligned and include ids present in both.
    Returns:
      - questions: list of question dicts
      - gold: mapping from str(id) -> raw gold string
    """
    if not DATA_Q.exists() or not DATA_Y.exists():
        raise FileNotFoundError(
            "Expanded dataset not found. Ensure expand_evaluation_robust.py has been run."
        )
    qs = json.load(open(DATA_Q, "r"))
    ys = json.load(open(DATA_Y, "r"))

    if limit is not None:
        qs = qs[:limit]
        ys = ys[:limit]

    gold: Dict[str, str] = {}
    for y in ys:
        if isinstance(y, dict):
            _id = str(y.get("id"))
            _out = y.get("output") or y.get("answer") or y.get("label") or ""
        else:
            # If a plain string was stored, rely on aligned index (rare)
            # Here we skip as id is not available.
            continue
        gold[_id] = str(_out)

    # Ensure all questions have golds
    questions = [q for q in qs if str(q.get("id")) in gold]
    return questions, gold


def mcnemar_pvalue(b_flags: List[bool], c_flags: List[bool]) -> Tuple[float, Tuple[int, int]]:
    """Compute McNemar two-sided p-value.

    - Use exact binomial test when b+c < 25
    - Otherwise use continuity-corrected chi-square approximation
    Returns (p_value, (b, c)) where
      b = count(baseline correct, chameleon incorrect)
      c = count(baseline incorrect, chameleon correct)
    """
    if len(b_flags) != len(c_flags):
        raise ValueError("Flag lists must be of equal length")

    b = 0
    c = 0
    for b_ok, c_ok in zip(b_flags, c_flags):
        if b_ok and not c_ok:
            b += 1
        elif (not b_ok) and c_ok:
            c += 1

    n = b + c
    if n == 0:
        return 1.0, (b, c)

    # Exact binomial test for small samples
    if n < 25:
        from math import comb

        x = min(b, c)
        # Two-sided p-value
        tail = 0.0
        for k in range(0, x + 1):
            tail += comb(n, k) * (0.5 ** n)
        p = min(1.0, 2.0 * tail)
        return float(p), (b, c)

    # Continuity-corrected chi-square for larger samples
    from math import fabs
    stat = (fabs(b - c) - 1.0) ** 2 / float(n)
    # Survival function of chi-square with df=1: sf(x) ≈ exp(-x/2)
    from math import exp
    p = exp(-stat / 2.0)
    return float(p), (b, c)


def user_histogram(questions: List[Dict[str, Any]]) -> Dict[str, int]:
    """Simple histogram of user buckets derived from id[:-1]."""
    from collections import Counter

    def _uid(qid: Any) -> str:
        s = str(qid) if qid is not None else ""
        return s[:-1]

    return dict(Counter(_uid(q.get("id")) for q in questions))


#!/usr/bin/env python3
"""
LaMP-2 Dataset Preflight Validation

Validates that all samples meet strict requirements for zero-fallback experiments:
- user_id exists (or can be deterministically generated)
- profile is non-empty and parseable to [{tag, description}] format
- question (movie description) is non-empty
- gold label is in allowed tag set
- No unknown tags, missing fields, or broken JSON

Usage:
    python tools/validate_lamp2.py --dataset data/evaluation/lamp2_expanded_eval.jsonl --report results/diagnostics/lamp2_preflight.md

Exit codes:
    0: PASS - all samples valid, ready for strict mode
    2: File not found or invalid
    3: FAIL - validation errors found
"""

import argparse
import json
import re
import sys
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

ALLOWED = [
    "action", "based on a book", "classic", "comedy", "dark comedy", "dystopia",
    "fantasy", "psychology", "romance", "sci-fi", "social commentary",
    "thought-provoking", "true story", "twist ending", "violence"
]

def canon_tag(s: str) -> str | None:
    """Canonicalize tag string to match allowed tags exactly"""
    x = (s or "").strip().lower()
    x = x.replace("_", " ").replace("-", " ").replace("  ", " ")
    x = re.sub(r"\s+", " ", x)
    
    # Handle common variants
    variants = {
        "sci fi": "sci-fi", 
        "science fiction": "sci-fi",
        "thought provoking": "thought-provoking",
        "based on a book": "based on a book",
        "dark comedy": "dark comedy",
    }
    x = variants.get(x, x)
    
    # Match against allowed tags
    for allowed in ALLOWED:
        if x == allowed or x == allowed.replace("-", " "):
            return allowed
    if x == "sci fi": 
        return "sci-fi"
    
    return None

def parse_profile_any(sample):
    """Parse profile from any format to standardized [{tag, description}] format"""
    # Already structured
    if isinstance(sample.get("profile"), list):
        ok = []
        for p in sample["profile"]:
            t = canon_tag(p.get("tag", ""))
            d = (p.get("description") or "").strip()
            if t and d: 
                ok.append({"tag": t, "description": d})
        if ok: 
            return ok
    
    # Text-based (user_profile / preferences etc.)
    raw = sample.get("user_profile") or sample.get("preferences") or ""
    if isinstance(raw, str) and raw.strip():
        items = []
        for line in raw.splitlines():
            # Match pattern: - tag: description
            m = re.match(r"^\s*-\s*([A-Za-z0-9 \-]+)\s*:\s*(.+)$", line)
            if m:
                t = canon_tag(m.group(1))
                d = m.group(2).strip()
                if t and d: 
                    items.append({"tag": t, "description": d})
        if items: 
            return items
    
    raise ValueError("profile_missing_or_unparsable")

def stable_user_id(profile_list):
    """Generate deterministic user_id from profile content"""
    s = json.dumps(profile_list, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def main():
    parser = argparse.ArgumentParser(description="LaMP-2 preflight validation")
    parser.add_argument("--dataset", required=True, help="JSONL dataset file")
    parser.add_argument("--report", default="", help="Output validation report (markdown)")
    
    args = parser.parse_args()
    
    path = Path(args.dataset)
    if not path.exists():
        print(f"[ERR] dataset not found: {path}", file=sys.stderr)
        sys.exit(2)
    
    n = 0
    bad_json = 0
    missing = defaultdict(int)
    unknown_tags = Counter()
    id_dupe = Counter()
    user_id_set = set()
    derived_user_ids = 0
    label_dist = Counter()
    
    rows = []
    with path.open() as f:
        for ln, line in enumerate(f, start=1):
            if not line.strip(): 
                continue
            try:
                obj = json.loads(line)
            except Exception:
                bad_json += 1
                continue
            
            n += 1
            sid = obj.get("id")
            id_dupe[sid] += 1
            
            # Profile validation
            try:
                prof = parse_profile_any(obj)
            except Exception:
                missing["profile"] += 1
                prof = []
            
            # User ID validation
            uid = (obj.get("user_id") or "").strip() if isinstance(obj.get("user_id"), str) else obj.get("user_id")
            uid = str(uid) if uid not in (None, "") else ""
            if not uid and prof:
                uid = stable_user_id(prof)
                derived_user_ids += 1
            if not uid:
                missing["user_id"] += 1
            user_id_set.add(uid)
            
            # Question validation
            q = obj.get("question") or obj.get("movie") or ""
            if not isinstance(q, str) or not q.strip():
                missing["question"] += 1
            
            # Gold label validation
            graw = obj.get("gold") or obj.get("reference") or ""
            g = canon_tag(graw) if isinstance(graw, str) else None
            if not g:
                missing["gold"] += 1
            else:
                label_dist[g] += 1
            
            # Track unknown tags in profile (for reference)
            for p in prof:
                if p["tag"] not in ALLOWED:
                    unknown_tags[p["tag"]] += 1
            
            rows.append({
                "id": sid,
                "user_id": uid,
                "gold": g or graw,
                "has_profile": bool(prof)
            })
    
    # Check for duplicate IDs
    dup_ids = [k for k, c in id_dupe.items() if c > 1]
    
    # Overall validation result
    ok = (bad_json == 0 and 
          all(v == 0 for v in missing.values()) and 
          len(dup_ids) == 0)
    
    # Generate report
    md = []
    md += ["# LaMP-2 Preflight Report", ""]
    md += [f"- Dataset: `{path}`", f"- Total lines: {n}"]
    md += [f"- JSON errors: **{bad_json}**", ""]
    md += ["## Missing Fields"]
    for k in ("user_id", "profile", "question", "gold"):
        md += [f"- {k}: **{missing[k]}**"]
    md += [""]
    md += ["## Duplicated `id`", f"- count: **{len(dup_ids)}**"]
    if dup_ids[:10]:
        md += [f"- first 10: {dup_ids[:10]}"]
    md += ["", "## Label Distribution (gold)"]
    for tag in ALLOWED:
        md += [f"- {tag}: {label_dist[tag]}"]
    md += ["", f"## Derived user_id: **{derived_user_ids}** (deterministic generation)"]
    if unknown_tags:
        md += ["", "## Unknown tags found in profile (reference)", str(dict(unknown_tags))]
    md += ["", f"## Result: {'✅ PASS' if ok else '❌ FAIL'}"]
    
    report_content = "\n".join(md)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(report_content)
    
    print(report_content)
    sys.exit(0 if ok else 3)

if __name__ == "__main__":
    main()
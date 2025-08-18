#!/usr/bin/env bash
set -Eeuo pipefail
shopt -s nullglob

# ===== 設定（必要に応じて export で上書き可）=====
: "${COMP_THRESH:=0.95}"
: "${QUAL_THRESH:=0.60}"
: "${SRC:=/home/nakata/master_thesis/rango/data/evaluation/lamp2_backup_eval.jsonl}"
: "${CANARY_N:=50}"  # 小さめで安定実行（必要なら 200 等に）
: "${PATTERN:=regex:^Answer:\\s*([A-Za-z0-9_\\- ]+)\\s*$}"

: "${RUN_DIR:=runs/canary10_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/out.log"

# すべての出力をログに（切断しても継続）
exec > >(stdbuf -oL -eL tee -a "$LOG") 2>&1

echo "[start] RUN_DIR=$RUN_DIR"
echo "[env] COMP_THRESH=$COMP_THRESH QUAL_THRESH=$QUAL_THRESH CANARY_N=$CANARY_N"

# ===== 1) CANARY 作成 =====
CANARY_FILE="$RUN_DIR/lamp2_canary_${CANARY_N}.jsonl"
python3 - "$SRC" "$CANARY_FILE" "$CANARY_N" <<'PY'
import os, sys, random
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
random.seed(42)
with open(src,encoding='utf-8',errors='ignore') as f:
    lines=[l for l in f if l.strip()]
sel = lines[:n] if len(lines)>=n else lines
os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(dst,'w') as f: f.writelines(sel)
print(f"[canary] created: {dst} lines={len(sel)}")
PY

# ===== 2) ablation（quality 用）=====
echo "[ablation] start"
conda run -n faiss310 python scripts/verification/ablation_study.py \
  --data "$CANARY_FILE" \
  --runs-dir "$RUN_DIR" \
  --treatments gate_curriculum \
  --seed 42 \
  --strict-output "$PATTERN" \
  --reask-on-format-fail --reask-max-retries 2 \
  --reask-temperature 0.0 \
  --selector "cos+tags+ppr" \
  --selector-weights "alpha=1.0,beta=0.4,gamma=0.6,lambda=0.3" \
  --mmr-lambda 0.3 \
  --adaptive-k "min=1,max=5,tau=0.05" \
  --neg-curriculum "easy:1,medium:0,hard:0" \
  --anti-hub on --ppr-restart 0.15 --hub-degree-cap 200 \
  --generate-report
echo "[ablation] done"

# ===== 3) strict（compliance 用）=====
echo "[strict] start"
conda run -n faiss310 python run_strict_compliance_test.py \
  --data "$CANARY_FILE" \
  --system-prompt prompts/lamp2_system_strict.txt \
  --user-template prompts/lamp2_user_template_strict.txt \
  --output "$RUN_DIR/strict_report.json" \
  --target-compliance "$COMP_THRESH"
echo "[strict] done"

# ===== 4) ゲート判定（strict compliance + ablation quality）=====
echo "[gate] start"
python3 - "$RUN_DIR" "$COMP_THRESH" "$QUAL_THRESH" <<'PY'
import json, os, re, sys
r, C, Q = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
ab_jsons=[os.path.join(r,'ablation_report.json'), os.path.join(r,'ablation_study_report.json')]
ab_json=next((p for p in ab_jsons if os.path.exists(p)), '')
log_ab=os.path.join(r,'out.log')
strict_json=os.path.join(r,'strict_report.json')

def num(x):
    if isinstance(x,str):
        m=re.search(r'([0-9.]+)',x); return float(m.group(1)) if m else None
    return float(x) if x is not None else None

def parse_strict(p):
    if not os.path.exists(p): return None
    d=json.load(open(p))
    comp=(d.get('format_compliance_rate') or d.get('compliance_rate')
          or d.get('compliance') or d.get('format_compliance'))
    comp=num(comp)
    if comp and comp>1.0: comp/=100.0
    return comp

def extract_quality_any(d):
    pools=(d.get('conditions') or d.get('results') or [])
    best=None; v2=None
    if isinstance(pools,list):
        for c in pools:
            if not isinstance(c,dict): continue
            name=(c.get('name') or c.get('condition') or '').lower()
            q=c.get('quality') if c.get('quality') is not None else c.get('qual')
            if q is None: continue
            q=float(q)
            if 'v2' in name and 'complete' in name: v2=q
            best=q if best is None else max(best,q)
    if v2 is not None: return v2
    if best is not None: return best
    for k,v in d.items():
        if isinstance(v,dict):
            q=v.get('quality') if v.get('quality') is not None else v.get('qual')
            if q is not None: return float(q)
    q=d.get('quality') if d.get('quality') is not None else d.get('qual')
    return float(q) if q is not None else None

def parse_ablation(p, log):
    fmt=qual=None
    if p:
        try:
            d=json.load(open(p))
            fmt=(d.get('format_compliance_rate') or d.get('format_compliance')
                 or d.get('format_compliance_ratio') or d.get('compliance'))
            fmt=num(fmt)
            if fmt and fmt>1.0: fmt/=100.0
            qual=extract_quality_any(d)
        except Exception:
            pass
    if (fmt is None or qual is None) and os.path.exists(log):
        txt=open(log,errors='ignore').read()
        if fmt is None:
            m=re.search(r'(?:Format\\s+)?Compliance(?:\\s+Rate)?:\\s*([0-9.]+)',txt,re.I)
            fmt=float(m.group(1)) if m else None
            if fmt and fmt>1.0: fmt/=100.0
        if qual is None:
            m=re.search(r'V2-Complete.*?\\bqual[:=]\\s*([0-9.]+)',txt,re.I|re.S) or re.search(r'\\bQuality\\s+([0-9.]+)',txt,re.I)
            qual=float(m.group(1)) if m else None
    return fmt,qual

comp_strict=parse_strict(strict_json)
fmt_ab,qual=parse_ablation(ab_json,log_ab)
comp=comp_strict if comp_strict is not None else fmt_ab

ok=(comp is not None and comp>=C) and (qual is not None and qual>=Q)
fmt = lambda x: 'N/A' if x is None else f'{x:.3f}'
line = f"CANARY(10) GATE | compliance={fmt(comp)} (>= {C}) | quality={fmt(qual)} (>= {Q}) | {'PASS' if ok else 'FAIL'}"
print(line)
open(os.path.join(r,'gate_result.txt'),'w').write(line+"\\n")
PY
echo "[gate] done"
echo "[done] artifacts in $RUN_DIR"

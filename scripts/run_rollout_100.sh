set -euo pipefail

# ===== 0) しきい値 & 入力 =====
export COMP_THRESH="${COMP_THRESH:-0.95}"
export QUAL_THRESH="${QUAL_THRESH:-0.60}"
export SRC="${SRC:-/home/nakata/master_thesis/rango/data/evaluation/lamp2_backup_eval.jsonl}"
export PATTERN='regex:^Answer:\s*([A-Za-z0-9_\- ]+)\s*$'

# ===== 1) 入れ物（全量評価） =====
export RUN_DIR="runs/rollout100_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
export FULL_FILE="$RUN_DIR/lamp2_full.jsonl"
# 全量コピー（必要なら事前にフィルタしたSRCを渡す）
python3 - <<'PY'
import os
src=os.environ["SRC"]; dst=os.environ["FULL_FILE"]
with open(src) as f, open(dst,"w") as g:
    for line in f:
        if line.strip(): g.write(line)
print(dst)
PY

# ===== 2) ablation（quality算出） =====
conda run -n faiss310 python scripts/verification/ablation_study.py \
  --data "$FULL_FILE" \
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
  --generate-report | tee "$RUN_DIR/out.log"

# ===== 3) strict（compliance算出；同 RUN_DIR） =====
conda run -n faiss310 python run_strict_compliance_test.py \
  --data "$FULL_FILE" \
  --system-prompt prompts/lamp2_system_strict.txt \
  --user-template prompts/lamp2_user_template_strict.txt \
  --output "$RUN_DIR/strict_report.json" \
  --target-compliance "${COMP_THRESH}" \
| tee -a "$RUN_DIR/out.log"

# ===== 4) ゲート判定（strictのcompliance + ablationのV2-Complete品質） =====
RUN_DIR="$RUN_DIR" COMP_THRESH="$COMP_THRESH" QUAL_THRESH="$QUAL_THRESH" python3 - <<'PY' | tee "$RUN_DIR/gate_result.txt"
import json, os, re, sys
r=os.environ['RUN_DIR']
ab_jsons=[os.path.join(r,'ablation_report.json'), os.path.join(r,'ablation_study_report.json')]
ab_json=next((p for p in ab_jsons if os.path.exists(p)),'')
log_ab=os.path.join(r,'out.log')
strict_json=os.path.join(r,'strict_report.json')

def fnum(x):
    if isinstance(x,str):
        m=re.search(r'([0-9.]+)',x); return float(m.group(1)) if m else None
    return float(x) if x is not None else None

def parse_strict(p):
    if not os.path.exists(p): return None
    d=json.load(open(p))
    v=(d.get('format_compliance_rate') or d.get('compliance_rate')
       or d.get('compliance') or d.get('format_compliance'))
    v=fnum(v); 
    if v and v>1: v/=100.0
    return v

def extract_v2_quality(d):
    pools=d.get('conditions') or d.get('results') or []
    for c in pools if isinstance(pools,list) else []:
        if not isinstance(c,dict): continue
        name=(c.get('name') or c.get('condition') or '').lower()
        q=c.get('quality') if c.get('quality') is not None else c.get('qual')
        if q is not None and 'v2' in name and 'complete' in name:
            return float(q)
    for k,v in d.items():
        if isinstance(v,dict) and 'v2' in str(k).lower() and 'complete' in str(k).lower():
            q=v.get('quality') if v.get('quality') is not None else v.get('qual')
            if q is not None: return float(q)
    return None

def extract_best_quality(d):
    pools=d.get('conditions') or d.get('results') or []
    best=None
    for c in pools if isinstance(pools,list) else []:
        if not isinstance(c,dict): continue
        q=c.get('quality') if c.get('quality') is not None else c.get('qual')
        if q is not None:
            q=float(q); best = q if best is None else max(best,q)
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
            fmt=fnum(fmt); 
            if fmt and fmt>1: fmt/=100.0
            qual=extract_v2_quality(d) or extract_best_quality(d)
        except Exception:
            pass
    if (fmt is None or qual is None) and os.path.exists(log):
        txt=open(log,errors='ignore').read()
        if fmt is None:
            m=re.search(r'(?:Format\s+)?Compliance(?:\s+Rate)?:\s*([0-9.]+)',txt,re.I)
            if m: 
                fmt=float(m.group(1)); 
                if fmt>1: fmt/=100.0
        if qual is None:
            m=re.search(r'V2-Complete.*?\bqual[:=]\s*([0-9.]+)',txt,re.I|re.S) or re.search(r'\bQuality\s+([0-9.]+)',txt,re.I)
            if m: qual=float(m.group(1))
    return fmt,qual

comp = parse_strict(strict_json)
fmt_ab, qual = parse_ablation(ab_json, log_ab)
if comp is None: comp = fmt_ab

C=float(os.environ.get('COMP_THRESH','0.95'))
Q=float(os.environ.get('QUAL_THRESH','0.60'))
ok=(comp is not None and comp>=C) and (qual is not None and qual>=Q)
fmt=lambda x:'N/A' if x is None else f'{x:.3f}'
print(f"ROLLOUT(100) GATE | compliance={fmt(comp)} (>= {C}) | quality={fmt(qual)} (>= {Q}) | {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 2)
PY

echo "[done] artifacts in $RUN_DIR"
ls -lh "$RUN_DIR"/{strict_report.json,ablation_*report.json,out.log,gate_result.txt} 2>/dev/null || true

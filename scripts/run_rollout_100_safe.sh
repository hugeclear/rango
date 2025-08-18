#!/usr/bin/env bash
set -Eeuo pipefail

# ===== logging: すべて時刻付きで out.log に残す =====
RUN_DIR="runs/rollout100_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
exec > >(awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0 }' | tee -a "$RUN_DIR/out.log") 2>&1
echo "[start] pid=$$ run_dir=$RUN_DIR"

trap 'echo "[ERROR] line=$LINENO status=$?"' ERR
trap 'echo "[exit] status=$?"' EXIT

# ===== thresholds & inputs =====
COMP_THRESH="${COMP_THRESH:-0.95}"
QUAL_THRESH="${QUAL_THRESH:-0.60}"
SRC="${SRC:-/home/nakata/master_thesis/rango/data/evaluation/lamp2_backup_eval.jsonl}"
PATTERN='regex:^Answer:\s*([A-Za-z0-9_\- ]+)\s*$'

# ===== conda の絶対パス指定（必要なら修正）=====
CONDA_BIN="${CONDA_BIN:-/home/nakata/anaconda3/bin/conda}"
if ! "$CONDA_BIN" --version >/dev/null 2>&1; then
  echo "[fatal] conda not found at $CONDA_BIN"
  exit 127
fi
echo "[info] conda=$("$CONDA_BIN" --version) bin=$CONDA_BIN"

# ===== dataset（全量そのままコピー）=====
FULL_FILE="$RUN_DIR/lamp2_full.jsonl"
python3 - <<PY
src="$SRC"; dst="$FULL_FILE"
import sys
with open(src) as f, open(dst,"w") as g:
    for L in f:
        if L.strip(): g.write(L)
print(dst)
PY
echo "[ok] dataset -> $FULL_FILE (size=$(wc -l < "$FULL_FILE") lines)"

# ===== ablation =====
echo "[ablation] start"
"$CONDA_BIN" run -n faiss310 --no-capture-output \
  python scripts/verification/ablation_study.py \
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
    --generate-report | tee -a "$RUN_DIR/out.log"
echo "[ablation] done"

# ===== strict =====
echo "[strict] start"
"$CONDA_BIN" run -n faiss310 --no-capture-output \
  python run_strict_compliance_test.py \
    --data "$FULL_FILE" \
    --system-prompt prompts/lamp2_system_strict.txt \
    --user-template prompts/lamp2_user_template_strict.txt \
    --output "$RUN_DIR/strict_report.json" \
    --target-compliance "$COMP_THRESH" | tee -a "$RUN_DIR/out.log"
echo "[strict] done"

# ===== gate (strict compliance + ablation quality) =====
echo "[gate] start"
COMP_THRESH="$COMP_THRESH" QUAL_THRESH="$QUAL_THRESH" RUN_DIR="$RUN_DIR" \
python3 - <<'PY' | tee "$RUN_DIR/gate_result.txt"
import json, os, re
r=os.environ['RUN_DIR']
ab_json=next((p for p in [os.path.join(r,'ablation_report.json'),
                          os.path.join(r,'ablation_study_report.json')]
              if os.path.exists(p)), '')
log_ab=os.path.join(r,'out.log')
strict_json=os.path.join(r,'strict_report.json')

def fnum(x):
    if isinstance(x,str):
        m=re.search(r'([0-9.]+)',x); return float(m.group(1)) if m else None
    return float(x) if x is not None else None

def parse_strict(p):
    if not os.path.exists(p): return None
    d=json.load(open(p))
    v=(d.get('format_compliance_rate') or d.get('compliance_rate') or
       d.get('compliance') or d.get('format_compliance'))
    v=fnum(v); 
    if v and v>1: v/=100.0
    return v

def pick_quality(d):
    pools=d.get('conditions') or d.get('results') or []
    # V2-Complete 優先
    for c in pools if isinstance(pools,list) else []:
        if isinstance(c,dict):
            nm=(c.get('name') or c.get('condition') or '').lower()
            q=c.get('quality') if c.get('quality') is not None else c.get('qual')
            if q is not None and 'v2' in nm and 'complete' in nm:
                return float(q)
    # fallback: 最大値
    best=None
    for c in pools if isinstance(pools,list) else []:
        if isinstance(c,dict):
            q=c.get('quality') if c.get('quality') is not None else c.get('qual')
            if q is not None:
                q=float(q); best = q if best is None else max(best,q)
    if best is not None: return best
    # さらに fallback
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
            fmt=(d.get('format_compliance_rate') or d.get('format_compliance') or
                 d.get('format_compliance_ratio') or d.get('compliance'))
            fmt=fnum(fmt); 
            if fmt and fmt>1: fmt/=100.0
            qual=pick_quality(d)
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
            m=re.search(r'V2-Complete.*?\bqual[:=]\s*([0-9.]+)',txt,re.I|re.S) \
               or re.search(r'\bQuality\s+([0-9.]+)',txt,re.I)
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
PY
echo "[gate] done"

echo "[done] artifacts in $RUN_DIR"
ls -lh "$RUN_DIR"/{strict_report.json,ablation_*report.json,out.log,gate_result.txt} 2>/dev/null || true

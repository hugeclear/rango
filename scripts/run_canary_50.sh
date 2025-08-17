#!/usr/bin/env bash
set -Eeuo pipefail

# ==== 設定（必要なら上書き可）====
export COMP_THRESH="${COMP_THRESH:-0.95}"   # compliance 閾値
export QUAL_THRESH="${QUAL_THRESH:-0.60}"   # quality   閾値

# conda 初期化（失敗しても続行）
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/anaconda3/etc/profile.d/conda.sh" || true
fi
conda activate faiss310 2>/dev/null || true

# CANARY ファイルの自動検出（指定がなければ runs/ から最新を探す）
if [[ -z "${CANARY_FILE:-}" || ! -f "$CANARY_FILE" ]]; then
  CANARY_FILE="$(find runs -type f -name 'lamp2_canary_50.jsonl' -print0 2>/dev/null | xargs -0 ls -t 2>/dev/null | head -1 || true)"
fi
if [[ -z "${CANARY_FILE:-}" || ! -f "$CANARY_FILE" ]]; then
  echo "[ERROR] 50件の canary データ(lamp2_canary_50.jsonl)が見つかりません。CANARY_FILE を指定してください。"
  exit 3
fi

# RUN_DIR をデータに合わせて決定
export RUN_DIR="${RUN_DIR:-"$(dirname "$CANARY_FILE")"}"
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR"
echo "CANARY_FILE=$CANARY_FILE"

# ---- 1) strict（形式準拠だけ確認; 外部データを必ず使うため --samples 0）----
echo "== STRICT compliance test =="
conda run -n faiss310 python run_strict_compliance_test.py \
  --data "$CANARY_FILE" \
  --samples 0 \
  --system-prompt prompts/lamp2_system_strict.txt \
  --user-template prompts/lamp2_user_template_strict.txt \
  --output "$RUN_DIR/strict_report.json" \
  --target-compliance "${COMP_THRESH}" \
| tee -a "$RUN_DIR/out.log"

# ---- 2) ablation（同じ 50件で quality を出す）----
echo "== Ablation study (quality) =="
PATTERN='regex:^Answer:\s*([A-Za-z0-9_\- ]+)\s*$'
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
  --generate-report | tee -a "$RUN_DIR/out.log"

# ---- 3) ゲート判定（strictの compliance と ablation の quality を合算判定）----
python3 - <<'PY' | tee "$RUN_DIR/gate_result.txt"
import json, os, re, sys
r=os.environ['RUN_DIR']
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
  comp=d.get('format_compliance_rate') or d.get('compliance_rate') or d.get('compliance') or d.get('format_compliance')
  comp=num(comp); 
  if comp and comp>1.0: comp/=100.0
  return comp

def extract_quality(d):
  pools=d.get('conditions') or d.get('results') or []
  best=v2=None
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
  # 辞書/トップレベルも一応見る
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
      fmt=d.get('format_compliance_rate') or d.get('format_compliance') or d.get('format_compliance_ratio') or d.get('compliance')
      fmt=num(fmt)
      if fmt and fmt>1.0: fmt/=100.0
      qual=extract_quality(d)
    except: 
      pass
  if (fmt is None or qual is None) and os.path.exists(log):
    txt=open(log,errors='ignore').read()
    if fmt is None:
      m=re.search(r'(?:Format\s+)?Compliance(?:\s+Rate)?:\s*([0-9.]+)',txt,re.I)
      fmt=float(m.group(1)) if m else None
      if fmt and fmt>1.0: fmt/=100.0
    if qual is None:
      m=re.search(r'V2-Complete.*?\bqual[:=]\s*([0-9.]+)',txt,re.I|re.S) or re.search(r'\bQuality\s+([0-9.]+)',txt,re.I)
      qual=float(m.group(1)) if m else None
  return fmt,qual

comp_strict=parse_strict(strict_json)
fmt_ab,qual=parse_ablation(ab_json,log_ab)
comp=comp_strict if comp_strict is not None else fmt_ab

C=float(os.environ.get('COMP_THRESH','0.95'))
Q=float(os.environ.get('QUAL_THRESH','0.60'))
ok=(comp is not None and comp>=C) and (qual is not None and qual>=Q)
fmt=lambda x:'N/A' if x is None else f'{x:.3f}'
print(f"CANARY(50) GATE | compliance={fmt(comp)} (>= {C}) | quality={fmt(qual)} (>= {Q}) | {'PASS' if ok else 'FAIL'}")
print(f"- strict   : {strict_json if os.path.exists(strict_json) else '(not found)'}")
print(f"- ablation : {ab_json if ab_json else '(not found)'}")
print(f"- log      : {log_ab if os.path.exists(log_ab) else '(not found)'}")
sys.exit(0 if ok else 2)
PY

echo "Done. See $RUN_DIR/{out.log,strict_report.json,ablation_*report.json,gate_result.txt}"

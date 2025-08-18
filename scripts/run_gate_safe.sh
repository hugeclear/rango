#!/usr/bin/env bash
# robust gate pipeline: ablation → strict → gate
set -Eeuo pipefail

# ---- config ----
CONDA_ENV="${CONDA_ENV:-faiss310}"
COMP_THRESH="${COMP_THRESH:-0.95}"
QUAL_THRESH="${QUAL_THRESH:-0.60}"
PATTERN='regex:^Answer:\s*([A-Za-z0-9_\- ]+)\s*$'
SRC="${SRC:-/home/nakata/master_thesis/rango/data/evaluation/lamp2_backup_eval.jsonl}"
CANARY_N="${CANARY_N:-50}"

ts() { date "+%Y%m%d_%H%M%S"; }
RUN_DIR="runs/rollout_$(ts)"
OUT="${OUT:-$RUN_DIR/out.log}"     # __OUT_DEFAULT_GUARD__
mkdir -p "$RUN_DIR"

echo "=== run_gate_safe ===" | tee -a "$OUT"
echo "RUN_DIR=$RUN_DIR" | tee -a "$OUT"
echo "CONDA_ENV=$CONDA_ENV  COMP_THRESH=$COMP_THRESH  QUAL_THRESH=$QUAL_THRESH" | tee -a "$OUT"

# ---- conda detection ----
CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
if [[ -z "${CONDA_BIN}" ]]; then
  for c in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" "/opt/conda/bin/conda"; do
    [[ -x "$c" ]] && CONDA_BIN="$c" && break
  done
fi
if [[ -z "${CONDA_BIN}" || ! -x "${CONDA_BIN}" ]]; then
  echo "ERROR: conda not found" | tee -a "$OUT"; exit 2
fi
echo "CONDA_BIN=$CONDA_BIN" | tee -a "$OUT"

# env exists?
if ! "$CONDA_BIN" info --envs | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  echo "ERROR: conda env '$CONDA_ENV' not found" | tee -a "$OUT"; exit 2
fi

# required scripts
REQUIRED=(
  "scripts/verification/ablation_study.py"
  "run_strict_compliance_test.py"
)
for s in "${REQUIRED[@]}"; do
  if [[ ! -f "$s" ]]; then
    echo "ERROR: required script missing: $s" | tee -a "$OUT"; exit 2
  fi
done

# data check
if [[ ! -r "$SRC" ]]; then
  echo "ERROR: SRC not readable: $SRC" | tee -a "$OUT"; exit 2
fi

# ---- 1) canary 作成 ----
CANARY_FILE="$RUN_DIR/lamp2_canary_${CANARY_N}.jsonl"
SRC="$SRC" CANARY_FILE="$CANARY_FILE" CANARY_N="$CANARY_N" python3 - <<'PY' | tee -a "$OUT"
import os, sys
src=os.environ["SRC"]; dst=os.environ["CANARY_FILE"]; n=int(os.environ["CANARY_N"])
with open(src) as f:
    lines=[l for l in f if l.strip()]
with open(dst,"w") as g:
    g.writelines(lines[:n] if len(lines)>=n else lines)
print(f"{dst}")
PY
echo "[canary] -> $CANARY_FILE  lines=$(wc -l < "$CANARY_FILE")" | tee -a "$OUT"

# ---- 2) ablation（quality算出；V2-Completeがログに出る）----
"$CONDA_BIN" run -n "$CONDA_ENV" --no-capture-output \
  python scripts/verification/ablation_study.py \
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
    --generate-report | tee -a "$OUT"

# ---- 3) strict（format compliance；非0でも先へ）----
set +e
"$CONDA_BIN" run -n "$CONDA_ENV" --no-capture-output \
  python run_strict_compliance_test.py \
    --data "$CANARY_FILE" \
    --system-prompt prompts/lamp2_system_strict.txt \
    --user-template prompts/lamp2_user_template_strict.txt \
    --output "$RUN_DIR/strict_report.json" \
    --target-compliance "$COMP_THRESH" \
    --samples "$CANARY_N" | tee -a "$OUT"
STRICT_STATUS=${PIPESTATUS[0]}
set -e
echo "[strict] exit=$STRICT_STATUS" | tee -a "$OUT"
if [ "${STRICT_STATUS:-1}" -ne 0 ]; then  # __STRICT_SOFT_RETRY__
  echo "[strict] soft-retry with stronger constraint" | tee -a "$OUT"
  STRONG_PROMPT="$RUN_DIR/system_strict_strong.txt"
  cp prompts/lamp2_system_strict.txt "$STRONG_PROMPT"
  cat >> "$STRONG_PROMPT" <<'EOP'

# === DO-NOT-BREAK RULES ===
- Output EXACTLY one line in the form: 'Answer: <tag>'
- Do NOT add any other words (no 'because', no explanation).
- If you add anything extra, the run FAILS.
EOP

  "$CONDA_BIN" run -n "$CONDA_ENV" --no-capture-output \
    python run_strict_compliance_test.py \
      --data "$CANARY_FILE" \
      --system-prompt "$STRONG_PROMPT" \
      --user-template prompts/lamp2_user_template_strict.txt \
      --output "$RUN_DIR/strict_report.json" \
      --target-compliance "$COMP_THRESH" \
      --samples 50 | tee -a "$OUT"
  STRICT_STATUS=$?
  echo "[strict] retry exit=$STRICT_STATUS" | tee -a "$OUT"
fi  # __STRICT_SOFT_RETRY__

# ---- 4) gate 判定（JSON→なければログの V2-Complete をフォールバック）----
python3 scripts/gate_eval.py \
  --run-dir "$RUN_DIR" \
  --comp-thresh "$COMP_THRESH" \
  --qual-thresh "$QUAL_THRESH" \
  | tee "$RUN_DIR/gate_result.txt"

echo "[done] artifacts in $RUN_DIR" | tee -a "$OUT"

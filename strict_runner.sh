#!/usr/bin/env bash
# fault-tolerant STRICT runner (keeps going; always returns 0)

# ← ここは Bash 必須。zsh から呼ばれたら Bash で再実行
if [ -z "${BASH_VERSION-}" ]; then exec bash "$0" "$@"; fi

set -u   # -e/-o pipefail は付けない（途中で落とさない）

# 0) 出力先（Python からも読めるよう export）
RUN="results/bench/strict_run_$(date +%Y%m%d_%H%M%S)"
export RUN
mkdir -p "$RUN"
echo "[OUTDIR] $RUN"

# 1) 事前チェック（落とさない）
for f in data/evaluation/lamp2_expanded_eval.jsonl data/user_priors.jsonl; do
  if [ ! -s "$f" ]; then
    echo "[WARN] missing: $f" | tee -a "$RUN/benchmark.log"
  else
    echo "[OK] found: $f" | tee -a "$RUN/benchmark.log"
  fi
done

# 2) ベンチ本体（exit codeを吸収）
export PYTHONUNBUFFERED=1
LAYERS=(-4 -3 -2 -1)
python3 tools/run_benchmark_lamp2.py \
  --data_path data --split test --limit 500 --seed 42 \
  --alpha_personal 6.0 --alpha_general -1.0 \
  --norm_scale 0.9 --edit_gate_threshold 0.0 \
  --target_layers "${LAYERS[@]}" \
  --mode id --calibrate \
  --strict --prior_mode user \
  --user_prior_path data/user_priors.jsonl \
  --out_dir "$RUN" \
  >"$RUN/benchmark.log" 2>&1
echo "[RC] run_benchmark=$?" | tee -a "$RUN/status.txt"

# 3) STRICT監査
if [ -s "$RUN/predictions.jsonl" ]; then
  python3 tools/validate_strict_results.py "$RUN/predictions.jsonl" \
    >"$RUN/strict_audit.log" 2>&1
  echo "[RC] strict_audit=$?" | tee -a "$RUN/status.txt"

  if command -v jq >/dev/null 2>&1; then
    jq -r '.prior.source' "$RUN/predictions.jsonl" | sort | uniq -c > "$RUN/prior_sources.txt"
  else
    echo "[INFO] jqなし。prior sources 集計スキップ" | tee -a "$RUN/benchmark.log"
  fi
else
  echo "[WARN] predictions.jsonl 不在のため監査スキップ" | tee -a "$RUN/benchmark.log"
fi

# 4) 効果検出＆ゲート診断
if [ -s "$RUN/predictions.jsonl" ]; then
  python3 tools/detect_editing_effects.py "$RUN/predictions.jsonl" --verbose \
    >"$RUN/effects.log" 2>&1
  echo "[RC] effects_verbose=$?" | tee -a "$RUN/status.txt"

  python3 tools/detect_editing_effects.py "$RUN/predictions.jsonl" \
    >"$RUN/effects.json" 2>>"$RUN/effects.log"
  echo "[RC] effects_json=$?" | tee -a "$RUN/status.txt"

  python3 tools/diagnose_gate_health.py "$RUN/predictions.jsonl" \
    >"$RUN/gate_diag.json" 2>>"$RUN/diagnose.log"
  echo "[RC] gate_diag=$?" | tee -a "$RUN/status.txt"
fi

# 5) 簡易MDレポート（失敗しても続行）
python3 - <<'PY' || true
import json, os, pathlib
run=os.environ.get("RUN")
p=pathlib.Path(run); report=p/"report.md"
try:
  E=json.load(open(p/"effects.json"))
  G=json.load(open(p/"gate_diag.json")) if (p/"gate_diag.json").exists() else {}
  n=E.get("n"); b=E.get("b"); c=E.get("c")
  delta=E.get("delta_acc"); pval=E.get("p_value")
  gate=G.get("gate_rate","-"); cos=G.get("mean_cos_theta","-"); L2=G.get("mean_l2","-")
  bc=(b or 0)+(c or 0)
  verdict=("GO" if (bc>=30 and (c or 0)>(b or 0) and (delta or 0)>0) else "NO-GO")
  md=f"""# STRICT FINAL REPORT

- n={n}  b={b}  c={c}  b+c={bc}
- Δacc={delta}  p={pval}
- gate_rate={gate}  mean_cosθ={cos}  mean_L2={L2}

**VERDICT: {verdict}**
"""
  report.write_text(md)
  (p/"status.json").write_text(json.dumps({"ok":True},ensure_ascii=False,indent=2))
except Exception as e:
  report.write_text("# STRICT FINAL REPORT\n\n(レポート生成に失敗。ログを参照)\n")
  (p/"status.json").write_text(json.dumps({"ok":False,"reason":str(e)},ensure_ascii=False,indent=2))
print(f"[OK] wrote {report}")
PY

# 6) 成果物一覧（最後まで必ず到達）
echo "=== ARTIFACTS ($RUN) ==="; ls -1 "$RUN" || true
echo "=== STATUS ==="; cat "$RUN/status.txt" 2>/dev/null || echo "(no status)"
echo "=== REPORT HEAD ==="; head -n 40 "$RUN/report.md" 2>/dev/null || echo "(no report)"
exit 0

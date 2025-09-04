#!/usr/bin/env bash
set -u  # 故障しても最後までログを残したいので -e は付けない

# 推奨: 高速ローカルキャッシュ
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HOME/.cache/huggingface}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

BASE="results/bench"
STAMP=$(date +%Y%m%d_%H%M%S)
ROOT="${BASE}/queue_${STAMP}"
mkdir -p "$ROOT"

echo "[OUTROOT] $ROOT"

common=(
  --data_path data --split test --limit 100 --seed 42
  --alpha_personal 6.0 --alpha_general -1.0
  --norm_scale 0.9 --edit_gate_threshold 0.0
  --target_layers -4 -3 -2 -1
  --mode id --calibrate
  --strict --prior_mode user
  --user_prior_path data/user_priors.jsonl
)

run_job () {
  name="$1"; shift
  out="${ROOT}/${name}"
  mkdir -p "$out"
  if [ -s "${out}/predictions.jsonl" ]; then
    echo "[SKIP] ${name} (predictions.jsonl あり)"
  else
    echo "[RUN ] ${name}"
    PYTHONUNBUFFERED=1 python tools/run_benchmark_lamp2.py "${common[@]}" "$@" --out_dir "$out" \
      >"${out}/benchmark.log" 2>&1
    echo "[RC] $name: $?" | tee -a "${ROOT}/status.txt"
  fi
  # 効果検出（失敗しても続行）
  if [ -s "${out}/predictions.jsonl" ]; then
    python tools/detect_editing_effects.py "${out}/predictions.jsonl" > "${out}/effects.json" 2>> "${out}/effects.log" || true
  fi
}

# ---- キュー（直列）----
run_job pmi_off                     # 既定（PMIなし）
run_job pmi_on      --use_pmi       # PMI補正あり
run_job calib_entropy --target_entropy 2.2 --max_top1_share 0.40

# 集約レポートを作成（Pythonワンライナー）
python - <<'PY' "$ROOT"
import json, sys, pathlib, collections
root=pathlib.Path(sys.argv[1])
rows=[]
for name in ["pmi_off","pmi_on","calib_entropy"]:
    d=root/name
    if not (d/"effects.json").exists(): continue
    E=json.load(open(d/"effects.json"))
    # 分布（jq不使用）
    import json as js
    bcnt=collections.Counter(); ccnt=collections.Counter()
    for line in open(d/"predictions.jsonl"):
        o=js.loads(line)
        bcnt[o["baseline"]]+=1; ccnt[o["chameleon"]]+=1
    rows.append({
        "name": name,
        "n": E.get("n"), "b": E.get("b"), "c": E.get("c"),
        "bc": (E.get("b",0)+E.get("c",0)),
        "delta_acc": E.get("delta_acc"), "p": E.get("p_value"),
        "baseline_top": bcnt.most_common(1)[0] if bcnt else ("-",0),
        "cham_top": ccnt.most_common(1)[0] if ccnt else ("-",0),
    })
md=["# BREAKTHROUGH QUEUE REPORT","","root: "+str(root),"","| run | n | b | c | b+c | Δacc | p | baseline_top | chameleon_top |","|---|---:|--:|--:|--:|--:|--:|---|---|"]
for r in rows:
    md.append(f"| {r['name']} | {r['n']} | {r['b']} | {r['c']} | {r['bc']} | {r['delta_acc']} | {r['p']} | {r['baseline_top']} | {r['cham_top']} |")
pathlib.Path(root/"report.md").write_text("\n".join(md))
print(f"[OK] wrote {root}/report.md")
PY

echo "[DONE] See: ${ROOT}/report.md"
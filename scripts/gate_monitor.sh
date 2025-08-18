#!/usr/bin/env bash
set -Eeuo pipefail
RUN_DIR="${1:-$(python3 scripts/latest_run_dir.py || true)}"
[[ -n "${RUN_DIR:-}" && -d "$RUN_DIR" ]] || { echo "RUN_DIR invalid"; exit 1; }
LOG="$RUN_DIR/out.log"
echo "RUN_DIR=$RUN_DIR"
echo "--- tail(out.log) ---"
[[ -f "$LOG" ]] && tail -n 120 "$LOG" || echo "no out.log yet"

# quick stats
if [[ -f "$LOG" ]]; then
  LOG="$LOG" python3 - <<'PY'
import os,re
p=os.environ.get("LOG","")
txt=open(p,errors="ignore").read() if p and os.path.exists(p) else ""
vals=[float(x) for x in re.findall(r'Compliance:\s*([0-9.]+)', txt)]
print("strict_compliance_last=", f"{vals[-1]:.3f}" if vals else "N/A", f"(n={len(vals)})")
m=re.search(r'Format Compliance Rate:\s*([0-9.]+)', txt)
print("ablation_format_compliance=", m.group(1) if m else "N/A")
v2=re.search(r'V2-Complete.*?qual[:=]\s*([0-9.]+)', txt, re.I|re.S)
print("v2_qual_from_log=", v2.group(1) if v2 else "N/A")
PY
fi

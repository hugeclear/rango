#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
from pathlib import Path
import time
import itertools as it
import pandas as pd

# ── Path layout (script is inside rango/)
RANGO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RANGO_DIR.parent                     # /home/nakata/master_thesis
EVALUATOR = RANGO_DIR / "chameleon_evaluator.py"    # /home/nakata/.../rango/chameleon_evaluator.py
RESULTS_DIR = PROJECT_ROOT / "results"              # /home/nakata/master_thesis/results
RUNLOG_DIR = RESULTS_DIR / "run_logs"
RUNLOG_DIR.mkdir(parents=True, exist_ok=True)

def run_evaluation(
    mode: str,
    gen: str,
    layers: str,
    alpha: float,
    beta: float,
    data_path: str,
    target_edit_ratio: float = 0.02,
    adaptive_alpha: bool = False,
    last_k_tokens: int = 0,
    timeout_sec: int = 600,
    quiet: bool = False,
    run_idx: int = 0,
    total: int = 0,
) -> bool:
    cmd = [
        sys.executable, str(EVALUATOR),            # ← 絶対パスで evaluator を叩く
        "--mode", mode,
        "--gen", gen,
        "--layers", layers,
        "--alpha", str(alpha),
        "--beta", str(beta),
        "--data_path", data_path,
        "--target_edit_ratio", str(target_edit_ratio),
        "--last_k_tokens", str(last_k_tokens),
    ]
    if adaptive_alpha:
        cmd.append("--adaptive_alpha")

    env = os.environ.copy()
    env.update({
        "TRANSFORMERS_VERBOSITY": "error",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONWARNINGS": "ignore",
        "LOGLEVEL": "WARNING",
    })

    if not quiet:
        print(f"[{run_idx}/{total}] cwd={PROJECT_ROOT} eval={EVALUATOR}")
        print(f"[{run_idx}/{total}] {' '.join(cmd)}")

    try:
        if quiet:
            stamp = int(time.time())
            base = f"run_{stamp}_{run_idx:04d}"
            out_path = RUNLOG_DIR / f"{base}.out"
            err_path = RUNLOG_DIR / f"{base}.err"
            with open(out_path, "w") as out, open(err_path, "w") as err:
                res = subprocess.run(
                    cmd, env=env, cwd=str(PROJECT_ROOT),
                    stdout=out, stderr=err, timeout=timeout_sec, check=False
                )
        else:
            res = subprocess.run(
                cmd, env=env, cwd=str(PROJECT_ROOT),
                timeout=timeout_sec, check=False, capture_output=False
            )
        if res.returncode != 0:
            if not quiet:
                print(f"  ❌ FAILED (rc={res.returncode})")
            return False
        return True

    except subprocess.TimeoutExpired:
        if not quiet:
            print(f"  ⏱️  TIMEOUT after {timeout_sec}s")
        return False
    except Exception as e:
        if not quiet:
            print(f"  ❌ ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Chameleon Hyperparameter Sweep")
    parser.add_argument("--mode", choices=["demo", "full"], default="demo")
    parser.add_argument("--gen", choices=["greedy", "sample"], default="sample")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--target_edit_ratio", type=float, default=0.02)
    parser.add_argument("--adaptive_alpha", action="store_true")
    parser.add_argument("--last_k_tokens", type=int, nargs='+', default=[0])
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Root directory that contains LaMP-2 data (expects raw/LaMP-2/merged.json or answers.json)",
    )
    args = parser.parse_args()

    # 探索空間
    layers_options = [
        "model.layers.27",
        "model.layers.20",
        "model.layers.20,model.layers.27",
    ]
    alpha_options = [0.2, 0.3, 0.4, 0.5]
    beta_options  = [-0.03, -0.05, -0.07, -0.1]
    target_options = [args.target_edit_ratio]
    lastk_options  = args.last_k_tokens
    adaptive_opts  = [True] if args.adaptive_alpha else [False, True]

    combos = list(it.product(layers_options, alpha_options, beta_options,
                             target_options, lastk_options, adaptive_opts))
    total_all = len(combos)

    if args.offset > 0 or args.limit is not None:
        end = (args.offset + args.limit) if args.limit is not None else None
        combos = combos[args.offset:end]

    total = len(combos)

    print("🔬 Starting Chameleon Hyperparameter Sweep")
    print(f"   Mode: {args.mode} | Generation: {args.gen}")
    print(f"   Layers: {len(layers_options)} | Alpha: {len(alpha_options)} | Beta: {len(beta_options)}")
    print(f"   Target edit ratios: {len(target_options)} | Last k tokens: {len(lastk_options)} | Adaptive: {len(adaptive_opts)}")
    print(f"   Total (all): {total_all} | Scheduled (this run): {total} (offset={args.offset}, limit={args.limit})")
    print("=" * 60)

    successful_runs = 0
    failed_runs = 0
    for idx, (layers, alpha, beta, target_ratio, last_k, adaptive) in enumerate(combos, 1):
        if not args.quiet:
            print(f"\n[{idx}/{total}] Testing:")
            print(f"  Layers: {layers}")
            print(f"  Alpha: {alpha}, Beta: {beta}")
            print(f"  Target ratio: {target_ratio}, Last-k: {last_k}, Adaptive: {adaptive}")

        ok = run_evaluation(
            mode=args.mode, gen=args.gen, layers=layers,
            alpha=alpha, beta=beta, data_path=args.data_path,
            target_edit_ratio=target_ratio,
            adaptive_alpha=adaptive, last_k_tokens=last_k,
            timeout_sec=args.timeout, quiet=args.quiet,
            run_idx=idx, total=total,
        )
        if ok:
            successful_runs += 1
            if not args.quiet:
                print("  ✅ SUCCESS")
        else:
            failed_runs += 1
            if not args.quiet:
                print("  ❌ FAILED")
        time.sleep(0.5)

    print("\n" + "=" * 60)
    print("🎯 Sweep Complete!")
    print(f"   Successful runs: {successful_runs}/{total}")
    print(f"   Failed runs:     {failed_runs}/{total}")

    csv_file = RESULTS_DIR / "experiment_results.csv"
    if not csv_file.exists():
        print(f"❌ Results CSV not found: {csv_file}")
        print("   (Check individual logs under results/run_logs/)")
        return

    try:
        df = pd.read_csv(csv_file)
        if len(df) == 0:
            print("⚠️  CSV is empty."); return

        tail_n = max(1, successful_runs) if successful_runs > 0 else min(10, len(df))
        df_recent = df.tail(tail_n)
        df_sorted = df_recent.sort_values(['cham_acc', 'impr_abs_acc'], ascending=[False, False])

        print("\n🏆 TOP 5 CONFIGURATIONS (from recent runs):")
        print("=" * 100)
        top_5 = df_sorted.head(5)
        for rank, (_, row) in enumerate(top_5.iterrows(), 1):
            print(f"\nRank {rank}:")
            print(f"  Layers: {row['layers']}")
            print(f"  Alpha: {row['alpha']:.2f}, Beta: {row['beta']:.3f}")
            print(f"  Acc: {row['baseline_acc']:.4f} → {row['cham_acc']:.4f} ({row['impr_rel_acc_pct']:+.1f}%)")
            print(f"  Diag: edit_ratio={row.get('avg_edit_ratio', 0.0):.4e}, hook_calls={row.get('hook_calls_mean', 0.0):.1f}")
            try:
                if row['cham_acc'] == row['baseline_acc'] and row.get('avg_edit_ratio', 0.0) >= 0.02 and row.get('hook_calls_mean', 0.0) >= 1:
                    print("  ⚠️  [ALERT] Outputs look insensitive to current edits")
                if row.get('hook_calls_mean', 0.0) < len(str(row['layers']).split(',')):
                    print("  🐛 [BUG] Hooks not firing for some layers")
            except Exception:
                pass

        best_impr = df_recent['impr_abs_acc'].max()
        mean_edit = df_recent.get('avg_edit_ratio', pd.Series([0.0]*len(df_recent))).mean()
        mean_hooks = df_recent.get('hook_calls_mean', pd.Series([0.0]*len(df_recent))).mean()

        print(f"\n📈 OVERALL (recent {len(df_recent)}):")
        print(f"  Best improvement: {best_impr:+.4f}")
        print(f"  Avg edit ratio:   {mean_edit:.4e}")
        print(f"  Avg hook calls:   {mean_hooks:.1f}")

        if best_impr >= 0.02:
            print("  ✅ SUCCESS: Found ≥ +0.02 absolute accuracy improvement")
        else:
            print("  ⚠️  No ≥ +0.02 improvement in this slice")

    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        print("   (CSV schema mismatch or partial runs?)")

    print("\n" + "=" * 60)
    print(f"📝 Full details: {csv_file}")
    print(f"🗂️  Per-run logs: {RESULTS_DIR/'run_logs'}/*.out|*.err (when --quiet)")

if __name__ == "__main__":
    main()
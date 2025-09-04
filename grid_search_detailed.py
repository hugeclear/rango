import os
import json
import itertools
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from utils_eval_detailed import load_dataset, mcnemar_pvalue, _canon


OUTDIR = Path("results/phase3c_grid_detailed")
OUTDIR.mkdir(parents=True, exist_ok=True)


def _compute_flags(preds: List[Any], gts: List[Any]) -> Tuple[List[bool], float]:
    to_norm = lambda s: _canon(s)
    flags = [to_norm(p) == to_norm(g) for p, g in zip(preds, gts)]
    acc = (sum(flags) / len(flags)) if flags else 0.0
    return flags, acc


def _stage1_grid() -> List[Tuple[float, float, float, float]]:
    # Coarse exploration
    grid = {
        "alpha_personal": [1.25, 1.75, 2.25],
        "alpha_general": [-1.0, -0.5, 0.0, 0.5],
        "norm_scale": [0.9, 1.0, 1.2],
        "edit_gate_threshold": [0.022, 0.026, 0.030],
    }
    return list(
        itertools.product(
            grid["alpha_personal"],
            grid["alpha_general"],
            grid["norm_scale"],
            grid["edit_gate_threshold"],
        )
    )


def _neighborhood(val: float, pool: List[float], width: int = 1) -> List[float]:
    uniq = sorted(set(pool))
    if val in uniq:
        idx = uniq.index(val)
    else:
        idx = min(range(len(uniq)), key=lambda i: abs(uniq[i] - val))
    left = max(0, idx - width)
    right = min(len(uniq), idx + width + 1)
    return uniq[left:right]


def stage2_grid(best_row: Dict[str, float]) -> List[Tuple[float, float, float, float]]:
    ap_pool = [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75]
    ag_pool = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25]
    ns_pool = [0.9, 1.0, 1.1, 1.2, 1.3]
    gt_pool = [0.020, 0.022, 0.024, 0.026, 0.028, 0.030]

    aps = _neighborhood(float(best_row["alpha_personal"]), ap_pool, width=2)
    ags = _neighborhood(float(best_row["alpha_general"]), ag_pool, width=2)
    nss = _neighborhood(float(best_row["norm_scale"]), ns_pool, width=1)
    gts = _neighborhood(float(best_row["edit_gate_threshold"]), gt_pool, width=2)
    return list(itertools.product(aps, ags, nss, gts))


def run_all() -> Dict[str, Any]:
    from chameleon_evaluator import ChameleonEvaluator

    # Prefer GPU explicitly when available by writing a small override config
    cfg_path = Path("config.yaml")
    cfg_cuda_path = OUTDIR / "config_cuda.yaml"
    use_cuda = False
    try:
        import torch
        use_cuda = torch.cuda.is_available()
    except Exception:
        use_cuda = False

    passed_config_path = str(cfg_path)
    if use_cuda:
        # Prepare a minimal config that enforces CUDA (merged with defaults inside evaluator)
        cuda_cfg = {
            "model": {
                "name": "./chameleon_prime_personalization/models/base_model",
                "device": "cuda",
                "torch_dtype": "float16",
            }
        }
        try:
            with open(cfg_cuda_path, "w", encoding="utf-8") as f:
                import yaml
                yaml.safe_dump(cuda_cfg, f)
            passed_config_path = str(cfg_cuda_path)
            print(f"[GPU] CUDA available. Forcing device=cuda via {passed_config_path}")
        except Exception as e:
            print(f"[GPU] Failed to write CUDA override config: {e}. Falling back to default config.yaml")

    # Load data (use full expanded set)
    questions, gold = load_dataset(limit=None)
    N = len(questions)
    if N < 50:
        raise SystemExit(f"Expanded dev set too small for grid search: N={N}")

    # Create single evaluator and preload theta
    evaluator = ChameleonEvaluator(config_path=passed_config_path, data_path="./")
    evaluator.test_samples_cache = questions
    # Load preferred theta vectors
    if hasattr(evaluator, "chameleon_editor"):
        evaluator.chameleon_editor.load_theta_vectors(
            "results/phase3c_theta_corrected/theta_p_lamp2_corrected.json",
            "results/phase3c_theta_corrected/theta_n_lamp2_corrected.json",
        )
        try:
            P0 = evaluator.chameleon_editor.personal_direction.clone()
            N0 = evaluator.chameleon_editor.neutral_direction.clone()
        except Exception:
            P0 = N0 = None
    else:
        P0 = N0 = None

    # Baseline once (fixed across all combos)
    base_res = evaluator.evaluation_engine.evaluate_baseline(questions, gold)
    baseline_flags, baseline_acc = _compute_flags(base_res.predictions, base_res.ground_truths)

    # Stage 1
    rows: List[Dict[str, Any]] = []
    combos = _stage1_grid()
    print(f"[Stage1] combos={len(combos)}, N={N}")
    for i, (ap, ag, ns, egt) in enumerate(combos, 1):
        print(f"[{i}/{len(combos)}] ap={ap}, ag={ag}, ns={ns}, gate={egt}")
        # Rescale theta directions if available
        if hasattr(evaluator, "chameleon_editor") and (P0 is not None):
            try:
                ns_val = float(ns)
                evaluator.chameleon_editor.personal_direction = P0 * ns_val
                evaluator.chameleon_editor.neutral_direction = N0 * ns_val
            except Exception:
                pass

        ch_res = evaluator.evaluation_engine.evaluate_chameleon(
            questions,
            gold,
            alpha_personal=float(ap),
            alpha_neutral=float(ag),
            target_layers=evaluator.config['chameleon'].get('target_layers'),
            target_edit_ratio=float(egt),
        )
        c_flags, c_acc = _compute_flags(ch_res.predictions, ch_res.ground_truths)
        pval, (b, c) = mcnemar_pvalue(baseline_flags, c_flags)
        rows.append(
            {
                "alpha_personal": ap,
                "alpha_general": ag,
                "norm_scale": ns,
                "edit_gate_threshold": egt,
                "baseline_acc": baseline_acc,
                "chameleon_acc": c_acc,
                "improvement": c_acc - baseline_acc,
                "p_value": pval,
                "discordant_b": b,
                "discordant_c": c,
            }
        )
        if (c_acc - baseline_acc) > 0.05 and pval < 0.10:
            print("  hint: promising region")

    df1 = pd.DataFrame(rows).sort_values("improvement", ascending=False)
    df1.to_csv(OUTDIR / "results_stage1.csv", index=False)
    best1 = df1.iloc[0].to_dict()
    json.dump({"best": best1}, open(OUTDIR / "best_stage1.json", "w"), indent=2)
    print("[Stage1] best:", best1)

    # Stage 2 around best of stage 1
    rows2: List[Dict[str, Any]] = []
    combos2 = stage2_grid(best1)
    print(f"[Stage2] combos={len(combos2)} around Stage1 best")
    for i, (ap, ag, ns, egt) in enumerate(combos2, 1):
        print(f"[{i}/{len(combos2)}] ap={ap}, ag={ag}, ns={ns}, gate={egt}")
        if hasattr(evaluator, "chameleon_editor") and (P0 is not None):
            try:
                ns_val = float(ns)
                evaluator.chameleon_editor.personal_direction = P0 * ns_val
                evaluator.chameleon_editor.neutral_direction = N0 * ns_val
            except Exception:
                pass
        ch_res = evaluator.evaluation_engine.evaluate_chameleon(
            questions,
            gold,
            alpha_personal=float(ap),
            alpha_neutral=float(ag),
            target_layers=evaluator.config['chameleon'].get('target_layers'),
            target_edit_ratio=float(egt),
        )
        c_flags, c_acc = _compute_flags(ch_res.predictions, ch_res.ground_truths)
        pval, (b, c) = mcnemar_pvalue(baseline_flags, c_flags)
        rows2.append(
            {
                "alpha_personal": ap,
                "alpha_general": ag,
                "norm_scale": ns,
                "edit_gate_threshold": egt,
                "baseline_acc": baseline_acc,
                "chameleon_acc": c_acc,
                "improvement": c_acc - baseline_acc,
                "p_value": pval,
                "discordant_b": b,
                "discordant_c": c,
            }
        )

    df2 = pd.DataFrame(rows2).sort_values("improvement", ascending=False)
    df2.to_csv(OUTDIR / "results_stage2.csv", index=False)
    best2 = df2.iloc[0].to_dict()
    json.dump({"best": best2}, open(OUTDIR / "best_stage2.json", "w"), indent=2)
    print("[Stage2] best:", best2)

    # Consolidate and write final best
    df_all = pd.concat([df1, df2], ignore_index=True).sort_values("improvement", ascending=False)
    df_all.to_csv(OUTDIR / "results_all.csv", index=False)
    best = df_all.iloc[0].to_dict()
    payload = {
        "config": {
            "alpha_personal": float(best["alpha_personal"]),
            "alpha_general": float(best["alpha_general"]),
            "norm_scale": float(best["norm_scale"]),
            "edit_gate_threshold": float(best["edit_gate_threshold"]),
        },
        "improvement": float(best["improvement"]),
        "p_value": float(best["p_value"]),
        "N": int(N),
    }
    json.dump(payload, open(OUTDIR / "best_config.json", "w"), indent=2)
    print("[ALL] best:", best)

    return {
        "N": N,
        "stage1_top": best1,
        "stage2_top": best2,
        "final_best": payload,
    }


if __name__ == "__main__":
    run_all()

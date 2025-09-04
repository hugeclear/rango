import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from utils_eval_detailed import load_dataset, mcnemar_pvalue, _canon
from grid_search_detailed import stage2_grid, OUTDIR


def _compute_flags(preds: List[str], gts: List[str]):
    to_norm = lambda s: _canon(s)
    flags = [to_norm(p) == to_norm(g) for p, g in zip(preds, gts)]
    acc = (sum(flags) / len(flags)) if flags else 0.0
    return flags, acc


def main():
    from chameleon_evaluator import ChameleonEvaluator

    # Load anchor (best of Stage 1 or explicit config)
    best_anchor_path = OUTDIR / "best_stage1.json"
    if not best_anchor_path.exists():
        raise SystemExit("best_stage1.json not found; run grid_search_detailed.py first")

    best_anchor = json.load(open(best_anchor_path))
    best = best_anchor.get("best", best_anchor)

    # Load (possibly expanded) dataset
    questions, gold = load_dataset(limit=None)
    N = len(questions)

    evaluator = ChameleonEvaluator(config_path="config.yaml", data_path="./")
    evaluator.test_samples_cache = questions
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

    # Baseline once
    base_res = evaluator.evaluation_engine.evaluate_baseline(questions, gold)
    baseline_flags, baseline_acc = _compute_flags(base_res.predictions, base_res.ground_truths)

    rows: List[Dict[str, Any]] = []
    combos2 = stage2_grid(best)
    print(f"[Stage2/RERUN] combos={len(combos2)}, N={N}")
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

    df = pd.DataFrame(rows).sort_values("improvement", ascending=False)
    df.to_csv(OUTDIR / "results_stage2_rerun.csv", index=False)
    best2 = df.iloc[0].to_dict()
    json.dump({"best": best2}, open(OUTDIR / "best_stage2_rerun.json", "w"), indent=2)

    # Merge into results_all and update best_config.json
    base_all_path = OUTDIR / "results_all.csv"
    if base_all_path.exists():
        base_all = pd.read_csv(base_all_path)
        all_new = pd.concat([base_all, df], ignore_index=True).sort_values("improvement", ascending=False)
    else:
        all_new = df
    all_new.to_csv(base_all_path, index=False)

    payload = {
        "config": {
            "alpha_personal": float(best2["alpha_personal"]),
            "alpha_general": float(best2["alpha_general"]),
            "norm_scale": float(best2["norm_scale"]),
            "edit_gate_threshold": float(best2["edit_gate_threshold"]),
        },
        "improvement": float(best2["improvement"]),
        "p_value": float(best2["p_value"]),
        "N": int(N),
    }
    json.dump(payload, open(OUTDIR / "best_config.json", "w"), indent=2)
    print("[Stage2/RERUN] best:", best2)


if __name__ == "__main__":
    main()


import json
import random
import itertools
from pathlib import Path
from typing import List, Dict, Tuple
import os

import numpy as np
import pandas as pd
from scipy.stats import chi2

# Reproducibility
random.seed(42)
np.random.seed(42)
try:
    import torch
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
except Exception:
    pass

DATA_Q = Path("data/lamp_expanded/dev_questions_expanded.json")
DATA_Y = Path("data/lamp_expanded/dev_outputs_expanded.json")
THETA_P = Path("results/phase3c_theta_corrected/theta_p_lamp2_corrected.json")
THETA_N = Path("results/phase3c_theta_corrected/theta_n_lamp2_corrected.json")


def load_dataset() -> Tuple[List[Dict], Dict[str, str]]:
    if not DATA_Q.exists() or not DATA_Y.exists():
        raise FileNotFoundError("Expanded dataset not found. Run expand_evaluation_robust.py first.")
    qs = json.load(open(DATA_Q, "r"))
    ys = json.load(open(DATA_Y, "r"))
    # Build gold dict mapping id -> output
    gold = {}
    for y in ys:
        # Accept flexible keys
        _id = str(y.get("id"))
        _out = y.get("output") or y.get("answer") or y.get("label") or y
        gold[_id] = str(_out)
    # Ensure all questions have golds
    qs = [q for q in qs if str(q.get("id")) in gold]
    return qs, gold


def mcnemar_from_flags(b_flags: List[bool], c_flags: List[bool]) -> Tuple[float, Tuple[int, int]]:
    b = c = 0
    for b_ok, c_ok in zip(b_flags, c_flags):
        if b_ok and not c_ok:
            b += 1
        elif (not b_ok) and c_ok:
            c += 1
    if (b + c) == 0:
        return 1.0, (b, c)
    # Continuity-corrected McNemar chi-squared
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = float(chi2.sf(stat, 1))
    return p, (b, c)


def evaluate_with_evaluator(alpha_p: float, alpha_g: float, norm_scale: float, gate: float):
    """Run baseline and chameleon on the expanded dataset via EvaluationEngine.

    Returns: (baseline_flags, chameleon_flags)
    """
    from chameleon_evaluator import ChameleonEvaluator

    # Initialize evaluator (data_path default is not used since we pass samples directly)
    evaluator = ChameleonEvaluator(config_path="config.yaml", data_path="./")

    # Override theta vectors
    if hasattr(evaluator, "chameleon_editor"):
        evaluator.chameleon_editor.load_theta_vectors(str(THETA_P), str(THETA_N))
        if norm_scale is not None and abs(norm_scale - 1.0) > 1e-9:
            try:
                evaluator.chameleon_editor.personal_direction *= float(norm_scale)
                evaluator.chameleon_editor.neutral_direction *= float(norm_scale)
            except Exception:
                pass

    # Load data
    questions, gold = load_dataset()

    # Run baseline & chameleon using the evaluation engine directly
    baseline = evaluator.evaluation_engine.evaluate_baseline(questions, gold)
    chameleon = evaluator.evaluation_engine.evaluate_chameleon(
        questions,
        gold,
        alpha_personal=float(alpha_p),
        alpha_neutral=float(alpha_g),
        target_layers=evaluator.config['chameleon'].get('target_layers'),
        target_edit_ratio=float(gate),  # map gate to target_edit_ratio
    )

    # Build flags based on returned aligned predictions/ground_truths
    def to_norm(s):
        return str(s).strip().lower()

    b_flags = [to_norm(p) == to_norm(g) for p, g in zip(baseline.predictions, baseline.ground_truths)]
    c_flags = [to_norm(p) == to_norm(g) for p, g in zip(chameleon.predictions, chameleon.ground_truths)]
    return (b_flags, baseline), (c_flags, chameleon)


def run_grid():
    results_dir = Path("results/phase3c_grid_final")
    results_dir.mkdir(parents=True, exist_ok=True)

    grid = {
        "alpha_personal": [1.5, 2.0, 2.5],
        "alpha_general": [-0.5, 0.0],
        "norm_scale": [1.0, 1.2],
        "edit_gate_threshold": [0.02, 0.025, 0.03],
    }

    combos = list(itertools.product(
        grid["alpha_personal"],
        grid["alpha_general"],
        grid["norm_scale"],
        grid["edit_gate_threshold"],
    ))

    # Preload data and run baseline once using a single evaluator instance
    from chameleon_evaluator import ChameleonEvaluator
    qs, gold = load_dataset()
    # Optional: limit number of samples via env var
    n_limit = int(os.getenv("GRID_N", "0") or 0)
    if n_limit > 0:
        qs = qs[:n_limit]
    evalr = ChameleonEvaluator(config_path="config.yaml", data_path="./")
    # Ensure finalize uses our provided samples
    evalr.test_samples_cache = qs
    # Load theta once
    if hasattr(evalr, "chameleon_editor"):
        evalr.chameleon_editor.load_theta_vectors(str(THETA_P), str(THETA_N))
        # Keep originals to rescale cleanly per combo
        try:
            P0 = evalr.chameleon_editor.personal_direction.clone()
            N0 = evalr.chameleon_editor.neutral_direction.clone()
        except Exception:
            P0 = N0 = None
    base_baseline = evalr.evaluation_engine.evaluate_baseline(qs, gold)
    def _flags(preds, gts):
        tn = lambda s: str(s).strip().lower()
        return [tn(p) == tn(g) for p, g in zip(preds, gts)]
    baseline_flags = _flags(base_baseline.predictions, base_baseline.ground_truths)

    # Optional: limit number of combinations for smoke runs
    max_combos = int(os.getenv("GRID_MAX_COMBOS", "0") or 0)
    if max_combos > 0:
        combos = combos[:max_combos]

    rows = []
    for i, (ap, ag, ns, egt) in enumerate(combos, 1):
        print(f"[{i}/{len(combos)}] α_p={ap}, α_g={ag}, norm={ns}, gate={egt}")
        # Rescale theta directions per combo without reloading the model
        if hasattr(evalr, "chameleon_editor") and (P0 is not None):
            try:
                ns_val = float(ns)
                evalr.chameleon_editor.personal_direction = P0 * ns_val
                evalr.chameleon_editor.neutral_direction = N0 * ns_val
            except Exception:
                pass

        ch_res = evalr.evaluation_engine.evaluate_chameleon(
            qs, gold,
            alpha_personal=float(ap),
            alpha_neutral=float(ag),
            target_layers=evalr.config['chameleon'].get('target_layers'),
            target_edit_ratio=float(egt),
        )
        c_flags = _flags(ch_res.predictions, ch_res.ground_truths)
        b_flags = baseline_flags
        b_acc = sum(b_flags) / len(b_flags)
        c_acc = sum(c_flags) / len(c_flags)
        p, (b, c) = mcnemar_from_flags(b_flags, c_flags)
        rows.append(
            {
                "alpha_personal": ap,
                "alpha_general": ag,
                "norm_scale": ns,
                "edit_gate_threshold": egt,
                "baseline_acc": b_acc,
                "chameleon_acc": c_acc,
                "improvement": c_acc - b_acc,
                "p_value": p,
                "discordant_b": b,
                "discordant_c": c,
            }
        )
        if (c_acc - b_acc) > 0.05 and p < 0.05:
            print(f"  ✓ 有意な改善: +{(c_acc - b_acc) * 100:.1f} pt, p={p:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "results.csv", index=False)

    best = df.loc[df["improvement"].idxmax()]
    best_payload = {
        "config": {
            "alpha_personal": float(best["alpha_personal"]),
            "alpha_general": float(best["alpha_general"]),
            "norm_scale": float(best["norm_scale"]),
            "edit_gate_threshold": float(best["edit_gate_threshold"]),
        },
        "improvement": float(best["improvement"]),
        "p_value": float(best["p_value"]),
    }
    json.dump(best_payload, open(results_dir / "best_config.json", "w"), indent=2)
    print("\n最適設定:", best_payload)


if __name__ == "__main__":
    run_grid()

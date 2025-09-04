import json
from pathlib import Path

import yaml


def main():
    best_path = Path("results/phase3c_grid_final/best_config.json")
    if not best_path.exists():
        raise FileNotFoundError("best_config.json not found. Run grid_search_robust.py first.")

    best = json.load(open(best_path, "r"))
    cfg = {
        "chameleon": {
            "alpha_personal": best["config"]["alpha_personal"],
            "alpha_general": best["config"]["alpha_general"],
            "norm_scale": best["config"]["norm_scale"],
            "edit_gate_threshold": best["config"]["edit_gate_threshold"],
            "theta_vectors": {
                "personal": "results/phase3c_theta_corrected/theta_p_lamp2_corrected.json",
                "neutral": "results/phase3c_theta_corrected/theta_n_lamp2_corrected.json",
            },
        },
        "generation": {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
        },
        "evaluation": {"min_samples": 100, "use_mcnemar": True},
    }

    with open("config_optimized.yaml", "w") as f:
        yaml.dump(cfg, f)
    print("\u2713 config_optimized.yaml を作成")


if __name__ == "__main__":
    main()


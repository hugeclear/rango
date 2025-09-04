import json
import yaml
from pathlib import Path


BEST_PATH = Path("results/phase3c_grid_detailed/best_config.json")
OUT_PATH = Path("config_optimized.yaml")


def main():
    if not BEST_PATH.exists():
        raise SystemExit(f"Best config not found: {BEST_PATH}")
    best = json.load(open(BEST_PATH))
    conf = best["config"]
    cfg = {
        "chameleon": {
            "alpha_personal": float(conf["alpha_personal"]),
            "alpha_general": float(conf["alpha_general"]),
            "norm_scale": float(conf["norm_scale"]),
            "edit_gate_threshold": float(conf["edit_gate_threshold"]),
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
        "evaluation": {
            "min_samples": max(100, int(best.get("N", 100))),
            "use_mcnemar": True,
        },
    }
    yaml.safe_dump(cfg, open(OUT_PATH, "w"))
    print(f"✓ wrote {OUT_PATH}")


if __name__ == "__main__":
    main()


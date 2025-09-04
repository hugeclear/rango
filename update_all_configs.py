import json
import glob
from pathlib import Path

configs = glob.glob("**/generation_config.json", recursive=True)
for config_path in configs:
    if Path(config_path).exists():
        with open(config_path) as f:
            cfg = json.load(f)
        cfg.update({
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        })
        Path(config_path).rename(f"{config_path}.bak")
        with open(config_path, 'w') as f:
            json.dump(cfg, f, indent=2)
        print(f"✓ {config_path} を更新")

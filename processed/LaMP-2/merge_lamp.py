import json
from pathlib import Path

q_path = Path("processed/LaMP-2/queries.json")
p_path = Path("processed/LaMP-2/profiles.json")
out_path = Path("processed/LaMP-2/merged.json")

# id → 映画説明 の辞書を作成
queries = { r["id"]: r["input"] for r in json.loads(q_path.read_text(encoding="utf-8")) }
# id → タグ履歴リスト の辞書を作成
profiles = { r["id"]: r["profile"] for r in json.loads(p_path.read_text(encoding="utf-8")) }

merged = []
for uid, inp in queries.items():
    prof = profiles.get(uid, [])
    merged.append({
        "id": uid,
        "input": inp,
        "profile": prof
    })

out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"✅ Written {len(merged)} records to {out_path}")
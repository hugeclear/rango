import json, os

def load_labels_from_jsonl(path):
    labs = []
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            g = (o.get("gold") or o.get("label") or "").strip().lower()
            if g: labs.append(g)
    labs = sorted(set(labs))
    return labs

def resolve_label_set(data_dir, split, fallback_labels):
    # 予測データがない初回は gold を含む元データから抽出する実装に合わせて変更してください。
    # 最低限は fallback を返す。
    pred_path = os.path.join(data_dir, f"lamp2_{split}.gold.jsonl")
    if os.path.exists(pred_path):
        labs = load_labels_from_jsonl(pred_path)
        if labs: return labs
    return fallback_labels
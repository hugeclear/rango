#!/usr/bin/env python3
import os, json, argparse
import numpy as np
import pandas as pd
from collections import Counter
from sentence_transformers import SentenceTransformer

DEF_ID_CANDS = ["user_id","uid","user","author_id","profile_id","id"]
DEF_TEXT_CANDS = ["question","input","profile","history","preferences","tags","text","content","prompt"]

def autodetect_cols(df, id_key=None, text_keys=None):
    if id_key is None:
        for k in DEF_ID_CANDS:
            if k in df.columns:
                id_key = k
                break
    if id_key is None:
        raise ValueError(f"ID列が見つかりません。候補: {DEF_ID_CANDS}")

    if not text_keys:
        text_keys = [k for k in DEF_TEXT_CANDS if k in df.columns]
    if not text_keys:
        # 文字列比率が高い列を自動選定
        text_keys = [c for c in df.columns if df[c].map(lambda x:isinstance(x,str)).mean()>0.7]
    if not text_keys:
        raise ValueError("テキスト列が見つかりません。--text_keys を指定してください。")
    return id_key, text_keys

def make_joiner(text_keys):
    def join_row(r):
        vals=[]
        for k in text_keys:
            v = r.get(k, "")
            if isinstance(v, (list, tuple)):
                buf=[]
                for item in v:
                    if isinstance(item, dict):
                        s = " ".join(str(item.get(tk,"")) for tk in ("text","title","category"))
                        buf.append(s.strip())
                    else:
                        buf.append(str(item))
                v = " ".join([x for x in buf if x])
            elif isinstance(v, dict):
                v = " ".join(str(v.get(tk,"")) for tk in ("text","title","category"))
            elif not isinstance(v, str):
                v = str(v)
            if v:
                vals.append(v)
        return " [SEP] ".join(vals)
    return join_row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_path", required=True)
    ap.add_argument("--out_dir", default="./embeddings")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--id_key", default=None)
    ap.add_argument("--text_keys", nargs="*", default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    id_key, text_keys = autodetect_cols(df, args.id_key, args.text_keys)
    print(f"[INFO] id_key={id_key}, text_keys={text_keys}")

    join_row = make_joiner(text_keys)
    df["_text"] = df.apply(join_row, axis=1)
    g = df.groupby(id_key)["_text"].apply(lambda s: " [DOC] ".join(s.tolist())).reset_index()

    # デバイス自動
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    print(f"[INFO] device={device}, model={args.model}")
    model = SentenceTransformer(args.model, device=device)

    texts = g["_text"].tolist()
    embs = model.encode(
        texts, batch_size=args.batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=False
    ).astype(np.float32)

    out_npy = os.path.join(args.out_dir, "lamp2_user_embeddings.npy")
    out_idx = os.path.join(args.out_dir, "lamp2_user_embeddings.index.json")
    np.save(out_npy, embs)
    # インデックスは {user_id, n_docs}
    cnt = Counter(g[id_key])
    with open(out_idx, "w", encoding="utf-8") as f:
        json.dump([{ "user_id": str(uid), "n_docs": int(n)} for uid, n in cnt.items()],
                  f, ensure_ascii=False, indent=2)

    print(f"[OK] saved: {out_npy} shape={embs.shape}")
    print(f"[OK] saved: {out_idx} users={len(g)}")

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

class CollaborativeUserReps:
    """
    - 入力:
        pool: graphrag_cfs_weights/cfs_pool.parquet
              (cols: user_id, neighbor_id, weight, ppr_score, cos_sim)
        emb:  embeddings/lamp2_user_embeddings.npy (shape: [N, D], L2正規化済み)
        id_map: ppr_results/id_map.json  ※内部idx->外部user_id の辞書
                (なければ embeddings の index.json から {i: rec["user_id"]} を組んでもOK)

    - 出力:
        .fused_vector(user_id, lam) -> 協調埋め込み (D,)
        .export_all(out_npy, lam)   -> 全ユーザの協調埋め込み (N, D)
    """
    def __init__(self, pool: str, emb: str, id_map: str):
        self.pool_path = Path(pool)
        self.emb_path  = Path(emb)
        self.idmap_path= Path(id_map)

        # Embeddings
        self.E = np.load(self.emb_path)            # (N, D)
        self.N, self.D = self.E.shape

        # ID map: internal_idx -> user_id (str)
        with open(self.idmap_path, "r", encoding="utf-8") as f:
            id_map_dict: Dict[str, str] = json.load(f)
        # keys 文字列の可能性があるので int 化
        self.idx2uid: Dict[int, str] = {int(k): str(v) for k, v in id_map_dict.items()}
        self.uid2idx: Dict[str, int] = {v: k for k, v in self.idx2uid.items()}

        # Pool
        df = pd.read_parquet(self.pool_path)
        # 型をそろえる
        df["user_id"] = df["user_id"].astype(str)
        df["neighbor_id"] = df["neighbor_id"].astype(str)

        # self-loop は二重カウントを避けるため neighbor 側から除外（ユーザ本体ベースベクトルで担保）
        self.pool = df[df["user_id"] != df["neighbor_id"]].copy()

        # 事前アグリゲート: user_id -> (neighbor_idx array, weight array)
        groups = self.pool.groupby("user_id")
        self.user_to_neighbors: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for uid, g in groups:
            # ID→内部idxへ（存在しないIDは落とす）
            nids = []
            wts  = []
            for nid, w in zip(g["neighbor_id"].values, g["weight"].values):
                idx = self.uid2idx.get(str(nid))
                if idx is not None:
                    nids.append(idx); wts.append(float(w))
            if len(nids) > 0:
                self.user_to_neighbors[uid] = (np.asarray(nids, dtype=np.int64),
                                               np.asarray(wts, dtype=np.float32))

    def fused_vector(self, user_id: str, lam: float = 0.5, renorm: bool = True) -> Optional[np.ndarray]:
        """v' = norm( (1-λ) v_u + λ * Σ_j (ŵ_j v_j) ),  ŵ はweightの和で正規化"""
        uid = str(user_id)
        uidx = self.uid2idx.get(uid)
        if uidx is None:
            return None

        v_u = self.E[uidx]  # (D,)
        pair = self.user_to_neighbors.get(uid)
        if pair is None:
            # 近傍なしなら素のベクトルを返す
            return v_u.copy()

        nidx, w = pair
        if nidx.size == 0:
            return v_u.copy()

        # 重み正規化（総和>0を保証）
        sw = float(w.sum())
        if sw <= 0:
            v_nei = v_u
        else:
            v_nei = (self.E[nidx] * (w[:, None] / sw)).sum(axis=0)  # (D,)

        v = (1.0 - lam) * v_u + lam * v_nei
        if renorm:
            nrm = np.linalg.norm(v)
            if nrm > 0:
                v = v / nrm
        return v.astype(np.float32)

    def export_all(self, out_npy: str, lam: float = 0.5, renorm: bool = True) -> np.ndarray:
        """id_mapの順（idx昇順）で (N, D) を書き出す"""
        V = np.zeros((self.N, self.D), dtype=np.float32)
        miss = 0
        for idx in range(self.N):
            uid = self.idx2uid[idx]
            v = self.fused_vector(uid, lam=lam, renorm=renorm)
            if v is None:
                V[idx] = self.E[idx]
                miss += 1
            else:
                V[idx] = v
        Path(out_npy).parent.mkdir(parents=True, exist_ok=True)
        np.save(out_npy, V)
        print(f"✅ exported: {out_npy}  shape={V.shape}, fallback={miss}")
        return V

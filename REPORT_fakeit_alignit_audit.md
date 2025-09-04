# Fake it → Align it 実装監査レポート

**監査日**: 2025-08-24  
**対象リポジトリ**: /home/nakata/master_thesis/rango  
**監査者**: Claude Code (リファクタリングエンジニア)  

---

## Executive Summary

**論文準拠度**: ❌ **部分不適合** - Fake it実装が不完全  
**CFS-GraphRAG統合度**: ✅ **適合** - プロンプト強化のみに限定  
**Hook実装**: ✅ **適合** - 射影式・弱編集停止・二重登録防止すべて実装済み  
**緊急度**: **HIGH** - Fake itパイプライン未実装により論文再現性に影響

---

## 1. Fake it（合成データ生成）監査結果

### 1.1 実装状況

| コンポーネント | 期待実装 | 現状 | 適合度 |
|--------------|---------|------|--------|
| **PCA選択** | 履歴からtop-k選択 | ❌ 未実装 | 不適合 |
| **LLM自己生成インサイト** | personal/neutral洞察生成 | ❌ 未実装 | 不適合 |
| **合成データ作成** | インサイトベース質問応答ペア | ❌ 未実装 | 不適合 |
| **SVD θ_P推定** | 個人化応答埋め込み→第1主成分 | ❌ 未実装 | 不適合 |
| **CCS θ_N推定** | personal vs neutral分離超平面 | ❌ 未実装 | 不適合 |
| **永続化** | `.npy/.jsonl`形式保存 | ❌ 未実装 | 不適合 |

### 1.2 発見された代替実装

**現行 θ ベクトル生成**:
- ファイル: `enhance_theta_vectors.py`
- 手法: 手動で固定値生成 (非論文準拠)
```python
# 現行実装: 固定値による生成
theta_p = [0.1] * 3072  # 固定値
theta_n = [-0.05] * 3072  # 固定値
```

**論文要求実装**:
```python
# 期待実装: SVD/CCSベースの動的推定
theta_p = SVD(personal_embeddings)[0]  # 第1主成分
theta_n = CCS(personal_vs_neutral_embeddings)  # 分離超平面
```

### 1.3 影響度評価

- **論文再現性**: CRITICAL - 主要アルゴリズムが未実装
- **パフォーマンス**: MODERATE - 固定θでも基本動作可能
- **スケーラビリティ**: HIGH - ユーザー個別θ生成不可

---

## 2. Align it（推論時編集）監査結果

### 2.1 射影式編集実装

**現行実装**: ✅ **論文準拠**
```python
# chameleon_evaluator.py:504-516
def _project(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """論文準拠: 投影成分計算 ⟨x,v⟩/||v||² * v"""
    v_norm = v.norm() + 1e-8
    v_normalized = v / v_norm
    dot_product = torch.sum(x * v_normalized, dim=-1, keepdim=True)
    projection = dot_product * v_normalized
    return projection

# 投影ベース編集式: h' = h + α_p*proj_p - |α_n|*proj_n
base_edit = float(alpha_personal) * proj_p - float(abs(alpha_neutral)) * proj_n
```

**数学的妥当性**: ✅ 完全準拠
- 内積計算: `⟨x,v⟩`
- 正規化: `v/||v||`
- 投影: `(⟨x,v⟩/||v||²) * v`
- 符号制御: `α>0, β<0`

### 2.2 last_k_tokens=16実装

**現行実装**: ✅ **要件準拠**
```python
# chameleon_evaluator.py:452-453
if last_k_tokens == 0:
    last_k_tokens = 16  # Default last_k enforcement

# chameleon_evaluator.py:537-541
if len(shape) == 3 and last_k_tokens > 0:
    k = min(last_k_tokens, t)
    mask = torch.zeros_like(edit)
    mask[:, -k:, :] = 1
    edit = edit * mask
```

### 2.3 弱編集早期停止

**現行実装**: ✅ **要件準拠**
```python
# chameleon_evaluator.py:561-574
weak_threshold = 0.005  # 1e-3より緩和済み
if ratio < weak_threshold:
    self._weak_streak += 1
    # Early-stop after 2 consecutive weak edits
    if self._weak_streak >= 2:
        self._editing_disabled = True
else:
    self._weak_streak = 0  # Reset streak on strong edit
```

### 2.4 二重登録防止

**現行実装**: ✅ **要件準拠**
```python
# chameleon_evaluator.py:221-222, 437-438
self._registered_layers = set()
self._hook_handles = []

# 登録時チェック
new_layers = [l for l in target_layers if l not in self._registered_layers]

# 削除時完全クリーンアップ
def remove_editing_hooks(self):
    for h in self._hook_handles:
        h.remove()
    self._hook_handles = []
    self._registered_layers.clear()
```

---

## 3. GraphRAG統合監査結果

### 3.1 分離原則の遵守

**監査結果**: ✅ **完全分離**

GraphRAG使用箇所:
1. **プロンプト強化のみ**: `scripts/run_graph_chameleon_eval.py:307`
```python
enhanced_prompt = self._build_graph_enhanced_prompt(
    prompt, user_profile, collab_users, user_id
)
```

2. **θ推定への非使用**: 確認済み - GraphRAGはθ生成に関与しない

### 3.2 協調フィルタリング実装

**現行実装**: ✅ **適切分離**
```python
# PPRベース協調ユーザー取得
def get_collaborative_context(self, user_id: int) -> List[int]:
    user_ppr = self.ppr_scores[user_id]
    top_indices = np.argsort(user_ppr)[-self.top_k-1:-1][::-1]
    return [int(idx) for idx in top_indices if user_ppr[idx] >= min_score]

# プロンプト強化（θ推定と分離）
def _build_graph_enhanced_prompt(self, prompt, user_profile, collab_users, user_id):
    collab_text = f"Similar users: {collab_users[:3]}"
    return f"{collab_text}\nUser preferences: {...}\nQuestion: {prompt}"
```

---

## 4. 成果物永続化監査結果

### 4.1 現行永続化

**既存実装**:
```
chameleon_prime_personalization/processed/LaMP-2/
├── theta_p.json  # 手動生成固定値
├── theta_n.json  # 手動生成固定値
├── theta_p.npy   # numpy配列版
└── theta_n.npy   # numpy配列版
```

**期待永続化構造**:
```
runs/personalization/
├── theta_cache/{uid}_theta_p.npy     # ユーザー個別SVD推定
├── theta_cache/{uid}_theta_n.npy     # ユーザー個別CCS推定
├── synthetic/{uid}_pairs.jsonl       # 合成Q&Aペア
└── insights/{uid}_insights.json      # LLM生成インサイト
```

### 4.2 DIAGログ現状

**現行DIAGログ**:
```
[DIAG] Generation complete: hook_calls=8, avg_edit_ratio=3.50e-04, suggested_alpha=0.200
[DIAG] tok_cache_hit_rate=0.000 (0/1)
```

**期待DIAGログ拡張**:
```
[DIAG] FAKEIT user=U42 pairs=24 p_ins_len=56 n_ins_len=43 svd_ok=True ccs_ok=True
```

---

## 5. 修復優先度と推奨アクション

### 5.1 CRITICAL（即時修復要）

1. **新規Fake itパイプライン実装**
   - ファイル: `scripts/pipeline_fakeit_build_directions.py`
   - 機能: PCA選択→LLM自己生成→合成データ→SVD/CCS推定

2. **オーケストレータ統合**
   - ファイル: `scripts/run_graph_chameleon_all.py`
   - 機能: θキャッシュ存在チェック→未存在時Fake it実行

### 5.2 HIGH（早期修復推奨）

1. **DIAGログ拡張**
   - Fake it実行時の1行要約ログ

2. **ユニットテスト追加**
   - ファイル: `tests/test_fakeit_alignit.py`
   - テスト: θ正規化、編集比率、弱編集停止

### 5.3 MEDIUM（継続改善）

1. **永続化構造統一**
   - 現行`.json`から期待`.npy`形式への移行

---

## 6. 受け入れ基準チェック

| 基準 | 現状 | 必要アクション |
|------|------|---------------|
| ✅ Align it実装 | 射影編集完全実装済み | 維持 |
| ❌ Fake it実装 | 未実装 | **新規作成必須** |
| ✅ 分離原則 | GraphRAG=プロンプトのみ | 維持 |
| ❌ 再現性 | 固定θのみ | **動的θ生成実装** |
| ✅ 観測性 | per-sample DIAG完備 | 拡張 |
| ✅ 回帰なし | 既存CLI/grid search維持 | 維持 |

---

## 7. 実装ロードマップ

### Phase 1: Fake it パイプライン（必須）
- [ ] `pipeline_fakeit_build_directions.py`新規作成
- [ ] SVD/CCS推定アルゴリズム実装
- [ ] 永続化機能実装

### Phase 2: オーケストレータ統合（必須）
- [ ] `run_graph_chameleon_all.py`へのθキャッシュ統合
- [ ] CLI フラグ追加（`--build-theta-if-missing`）

### Phase 3: 観測性向上（推奨）
- [ ] DIAG ログ拡張
- [ ] ユニットテスト追加

**予想実装期間**: 1-2日（Phase 1-2）+ 0.5日（Phase 3）

---

## 8. リスク評価

**技術リスク**: 
- SVD/CCS実装複雑性 → numpy/sklearn活用で軽減
- LLM自己生成品質 → few-shot prompt engineering適用

**スケジュールリスク**:
- 論文準拠要求 → Fake it実装が最優先

**品質リスク**:
- 既存機能への影響 → 段階的統合とテスト実施

---

**監査完了**: 2025-08-24  
**次回アクション**: Fake it パイプライン実装開始
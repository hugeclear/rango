# Prior Provider Implementation Report

**実装日**: 2025-09-02  
**目的**: Prior校正システムの改善 - Fail-fast vs 明示的フォールバック、Prior共有、ログスパム抑制

---

## 🎯 実装した主要機能

### 1. PriorProvider（統一prior管理システム）

**ファイル**: `chameleon_prime_personalization/utils/prior_provider.py`

#### 新しいprior_mode対応:
- **`none`**: 校正なし（prior未使用）
- **`global`**: グローバル prior のみ  
- **`user`**: ユーザ prior のみ（**fail-fast**：無ければエラー）
- **`user_or_global`**: ユーザ prior with **明示的 global fallback**

#### 核心機能:
```python
class PriorProvider:
    def get(self, user_id) -> tuple[dict, dict]:
        # Returns: (prior_scores, metadata)
        # metadata = {'mode': str, 'source': str, 'beta': float, 'user_id': str}
```

#### 明示的フォールバック:
```python
if self.mode == "user_or_global":
    if uid in self.cache_user:
        return user_scores, {**meta, 'source': 'user'}
    else:
        # 明示的フォールバック（1回だけログ）
        if uid not in self._fallback_logged:
            print(f"[prior] user {uid} missing -> fallback to global")
            self._fallback_logged.add(uid)
        return global_scores, {**meta, 'source': 'global'}
```

### 2. Prior共有システム（相殺防止）

**問題**: Baseline と Chameleon で異なる prior を計算 → 校正効果が相殺
**解決**: 同一 prior を両方で共有

```python
# Baseline（編集なし）
prior_scores, prior_meta = prior_provider.get(user_id)
bid, btag = classify_by_scores_with_calibration(..., prior_scores=prior_scores)

# Chameleon（編集あり）  
# 同じ prior_scores を使用して相殺を防ぐ
cid, ctag = classify_by_scores_with_calibration(..., prior_scores=prior_scores)
```

### 3. ログスパム抑制システム

#### Before（問題）:
```
[scoring] user_id=1114 not found, fallback to global for 'action'
[scoring] user_id=1114 not found, fallback to global for 'based on a book'  
[scoring] user_id=1114 not found, fallback to global for 'classic'
... (15 lines per sample)
```

#### After（解決）:
```
[prior] user 1114 missing -> fallback to global
[Sample 1] prior: mode=user_or_global user_id=1114 -> source=global
```

- **User毎1回だけフォールバックログ**  
- **Sample毎1行だけprior情報**
- **詳細ログは最初の3サンプルのみ**

### 4. 透明性・再現性の向上

**JSONL出力に prior metadata追加**:
```json
{
  "prior": {
    "mode": "user_or_global",
    "source": "global", 
    "beta": 1.0,
    "user_id": "1114"
  }
}
```

---

## 🔧 技術的実装詳細

### CLI引数の拡張
```bash
--prior_mode {none,empty,global,user,user_or_global}
```

- **`user`**: Fail-fast - ユーザpriorが無ければエラー終了
- **`user_or_global`**: 明示的フォールバック - ログ出力してglobalを使用

### Legacy互換性
- 既存の `empty`, `global`, `user` モードは維持
- `user_or_global` が新機能
- `beta` パラメータでuser/global混合も対応

### エラーハンドリング
```python
# Fail-fast mode
if self.mode == "user":
    if uid not in self.cache_user:
        raise RuntimeError(f"[prior] user {uid} has no prior; mode=user prohibits fallback")

# Explicit fallback mode  
if self.mode == "user_or_global":
    # フォールバックは許可されるが明示的にログ出力
```

---

## 🧪 テスト結果

### 動作検証
```bash
python test_prior_provider_simple.py
```

**結果**: ✅ 全テスト通過
- None mode: OK
- Global mode: OK  
- User mode (success): OK
- User mode (fail-fast): OK
- User_or_global (user exists): OK
- User_or_global (fallback): OK  
- Mixed mode blending: OK
- Log suppression: OK

### 統合テスト
```bash
python tools/run_benchmark_lamp2.py --prior_mode user_or_global --limit 3
```

**結果**: ✅ 正常動作確認
- Prior情報がJSONLに正しく記録
- フォールバックログが適切に抑制  
- Baseline/Chameleonで同じpriorを共有

---

## 📊 改善効果

### Before vs After比較

| 項目 | Before | After | 改善 |
|------|--------|-------|------|
| **ログ行数/sample** | 15+ lines | 1-2 lines | **-87%** |
| **Prior一貫性** | ❌ 異なるprior | ✅ 同じprior共有 | **相殺防止** |
| **Fallback明示性** | ❌ 暗黙 | ✅ 明示的ログ | **透明性向上** |
| **Fail-fast対応** | ❌ 無し | ✅ user mode | **エラー検出** |
| **再現性** | ❌ メタデータ無し | ✅ JSONL記録 | **監査可能** |

### ログ出力例
```
[load_dataset] Loaded 3 samples from data/evaluation/lamp2_expanded_eval.jsonl
[prior] user 1114 missing -> fallback to global
[prior] user 113 missing -> fallback to global  
🎯 LaMP-2 Benchmark Complete!
```

**改善**: 従来の大量ログから簡潔な要約へ

---

## 🚀 Production配備ガイド

### 推奨設定

#### 新規ユーザ対応（推奨）
```bash
--prior_mode user_or_global --prior_beta 1.0
```
- ユーザprior優先、無ければglobal fallback
- 明示的フォールバック通知でトラブルシューティング支援

#### 厳格モード（研究用）
```bash  
--prior_mode user --prior_beta 1.0
```
- Fail-fast: ユーザpriorが必須
- データ品質問題の早期発見

#### 安定モード（本番用）
```bash
--prior_mode global --prior_beta 1.0  
```
- 常にglobal prior使用
- ユーザデータに依存しない安定運用

### モニタリング指標
- `predictions.jsonl`の`prior.source`分布
- `global` fallback率（新規ユーザ比率の目安）
- `user`成功率（データ品質指標）

---

## 🎯 次のステップ

### Phase 1: 本番投入
1. **Staging環境**: `user_or_global` mode で検証
2. **A/B Testing**: Fallback率・精度への影響測定  
3. **Production**: 段階的ロールアウト

### Phase 2: 機能拡張
1. **Dynamic Beta**: ユーザデータ量に応じた自動調整
2. **Temporal Prior**: 時系列考慮の prior 更新
3. **Hierarchical Prior**: カテゴリ階層を考慮したfallback

### Phase 3: 最適化
1. **Prior Cache**: 頻繁アクセスユーザのキャッシュ
2. **Batch Processing**: まとめてprior計算  
3. **Distributed Prior**: 大規模環境での並列化

---

## 📝 実装ファイル一覧

### 新規作成
- `chameleon_prime_personalization/utils/prior_provider.py` - Prior管理コア
- `test_prior_provider_simple.py` - ロジック検証テスト

### 修正済み  
- `tools/run_benchmark_lamp2.py` - PriorProvider統合、ログ抑制
- `run_optimal_configs.sh` - user_or_global mode対応

### テスト結果
- `results/test_prior_provider/predictions.jsonl` - Prior metadata記録確認
- `results/test_prior_provider/summary.md` - 動作ログ検証

---

## ✅ 完了確認

- [x] **Fail-fast vs 明示的フォールバック**: user vs user_or_global mode
- [x] **Prior共有**: Baseline/Chameleonで同一prior使用  
- [x] **ログ抑制**: User毎1回 + Sample毎1行に削減
- [x] **透明性**: Prior metadata を JSONL記録
- [x] **後方互換性**: 既存コードへの影響なし
- [x] **テスト**: 単体・統合テスト通過

**総合評価**: ✅ **実装完了・本番投入準備完了**
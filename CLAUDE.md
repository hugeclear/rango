# CLAUDE_CODE_TASK.md

## プロジェクト概要

**研究課題**: LLMパーソナライゼーション（Chameleon + PriME手法）  
**現在フェーズ**: 第1フェーズ - Chameleon実装と評価  
**目標**: LaMPベンチマークでChameleonの有効性を定量的に実証  
**研究室**: Paikラボ  

---

## タスク: LaMPベンチマーク自動評価システム構築

### 背景
- Chameleonは埋め込み空間でユーザーの「らしさ」を編集するパーソナライゼーション手法
- 自己生成データ + SVD + 推論時リアルタイム編集の組み合わせ
- LaMPデータセットで定量評価が必要（第1フェーズ完了条件）

### 求める成果物
**完全自動化された評価システム** - 研究室メンバーがワンコマンドで実行可能

---

## STEP 1: リポジトリ分析

### 確認事項
1. **データファイル**
   - `merged.json` の存在と構造
   - LaMP-2データセットの状態
   - サンプル数、ユーザー数、データ形式

2. **既存実装**
   - Chameleon関連コード
   - 評価スクリプト
   - 前処理スクリプト

3. **環境設定**
   - requirements.txt
   - 設定ファイル
   - ディレクトリ構造

**出力**: リポジトリ現状分析レポート

---

## STEP 2: 評価フレームワーク実装

### 必須実装ファイル

#### `chameleon_evaluator.py`
```python
class LaMPDataLoader:
    """LaMP-2データの読み込みとユーザー分割"""
    
class SelfDataGenerator:
    """自己生成データ作成 (personal vs neutral)"""
    
class ChameleonEditor:
    """
    核心実装:
    1. 埋め込み抽出 (PyTorchフック使用)
    2. SVD方向学習 
    3. 推論時リアルタイム編集
    """
    
class EvaluationEngine:
    """ベースライン vs Chameleon比較評価"""
```

#### `run_evaluation.py`
```python
"""
実行エントリーポイント:
- 環境チェック
- パラメータ設定  
- 評価実行
- 結果保存・レポート生成
"""
```

#### `config.yaml`
```yaml
model:
  name: "meta-llama/Llama-3.2-3B-Instruct"
  max_length: 512
  batch_size: 4

chameleon:
  num_self_generated: 10
  target_layers: ["model.layers.16", "model.layers.20"] 
  alpha_personal: 1.5
  alpha_general: -0.8

evaluation:
  max_users: 10
  metrics: ["exact_match", "bleu_score"]
```

---

## STEP 3: Chameleon技術実装詳細

### 埋め込み抽出 (重要)
```python
def extract_embeddings(self, texts, target_layers):
    """
    PyTorchフックでTransformer中間層から抽出
    
    Flow:
    1. テキスト → tokenizer → model
    2. register_forward_hook(target_layer)
    3. フォワードパス中にhookが作動
    4. shape: (batch, seq_len, hidden_dim) → (hidden_dim,)
    """
```

### 方向学習 (SVD)
```python
def learn_directions(self, personal_embeddings, neutral_embeddings):
    """
    Flow:
    1. diff = personal_embeddings - neutral_embeddings
    2. U, S, Vt = torch.svd(diff)
    3. personal_direction = Vt[:, 0]  # 第1主成分
    4. general_direction = Vt[:, 1]   # 第2主成分
    """
```

### 推論時編集
```python
def edit_during_generation(self, input_text):
    """
    Flow:
    1. editing_hookを指定層に登録
    2. hook内: output += α_p * personal_dir + α_g * general_dir  
    3. model.generate()でパーソナライズ生成
    """
```

---

## STEP 4: 評価・実行システム

### 実行コマンド設計
```bash
# デモ実行 (3ユーザー、5分)
python run_evaluation.py --mode demo

# 本格評価 (10ユーザー、30-60分)
python run_evaluation.py --mode full

# 結果確認
python run_evaluation.py --mode results
```

### 評価メトリクス
```python
metrics = {
    "exact_match": 完全一致率,
    "bleu_score": n-gram重複度,  
    "improvement_rate": (chameleon - baseline) / baseline,
    "significance": t検定のp値
}
```

### 出力形式
```
results/
├── evaluation_YYYYMMDD_HHMMSS/
│   ├── results.csv           # ユーザー別スコア
│   ├── summary.json          # 全体統計
│   ├── visualization.png     # 改善率分布図
│   └── report.md            # 実験レポート
```

---

## STEP 5: 期待する最終成果
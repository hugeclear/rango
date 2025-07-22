# 🦎 Chameleon LaMP自動評価システム

**LLMパーソナライゼーション（Chameleon手法）のLaMPベンチマーク自動評価システム**

Paikラボ研究プロジェクト - 第1フェーズ完了目標

---

## 🎯 プロジェクト概要

**目的**: LaMP-2映画タグ付けタスクでChameleonパーソナライゼーション手法の定量的評価を自動実行

**Chameleon手法**:
- 自己生成データによるユーザー「らしさ」学習
- SVDによる方向ベクトル抽出 
- 推論時リアルタイム埋め込み編集

**評価内容**:
- ベースライン vs Chameleon性能比較
- 統計的有意性検定
- 完全自動化された実行環境

---

## 🚀 クイックスタート

### 1. 環境セットアップ
```bash
# 依存関係インストール
pip install -r requirements.txt

# 必要なNLTKデータダウンロード
python -c "import nltk; nltk.download('punkt')"
```

### 2. デモ実行（5分）
```bash
# 3ユーザー、軽量テスト
python run_evaluation.py --mode demo
```

### 3. 本格評価（30-60分）
```bash
# 10ユーザー、完全評価
python run_evaluation.py --mode full
```

### 4. 結果確認
```bash
# 最新の評価結果表示
python run_evaluation.py --mode results
```

---

## 📊 期待する出力例

```
=== Chameleon LaMP-2 Evaluation Results ===
Total Users Evaluated: 10
Baseline Performance: 0.627 ± 0.124
Chameleon Performance: 0.741 ± 0.098
Average Improvement: +18.2%
Users with Improvement: 8/10 (80%)
Statistical Significance: p < 0.05

✅ Statistically significant improvement!
📁 Results saved to: results/evaluation_20250722_143052/
```

---

## 🏗️ システム構成

### 核心ファイル

| ファイル | 機能 | 説明 |
|---------|------|------|
| `chameleon_evaluator.py` | メイン評価フレームワーク | Chameleon実装＋評価エンジン |
| `run_evaluation.py` | 自動実行スクリプト | ワンコマンド実行インターフェース |
| `config.yaml` | 設定ファイル | モデル・パラメータ設定 |
| `requirements.txt` | 依存関係 | 必要ライブラリ一覧 |

### データファイル

| パス | 内容 | 必須 |
|------|------|------|
| `chameleon_prime_personalization/data/raw/LaMP-2/merged.json` | LaMP-2データセット | ✅ |
| `processed/LaMP-2/theta_p.json` | パーソナル方向ベクトル | ✅ |
| `processed/LaMP-2/theta_n.json` | ニュートラル方向ベクトル | ✅ |
| `processed/LaMP-2/personal_insights.json` | 自己生成データ | 参考 |

---

## 🧠 Chameleon技術詳細

### 1. 埋め込み抽出
```python
# PyTorchフックでTransformer中間層から抽出
def extract_embeddings(texts, target_layers):
    """
    Flow:
    1. テキスト → tokenizer → model入力
    2. register_forward_hook で指定層の出力捕獲
    3. shape: (batch, seq_len, hidden_dim) → (hidden_dim,)
    """
```

### 2. 方向学習 (SVD)
```python
def learn_directions(personal_embeddings, neutral_embeddings):
    """
    Flow:
    1. diff = personal_embeddings - neutral_embeddings
    2. U, S, Vt = torch.svd(diff)
    3. personal_direction = Vt[:, 0]  # 第1主成分
    4. neutral_direction = Vt[:, 1]   # 第2主成分
    """
```

### 3. 推論時編集
```python
def edit_during_generation(input_text):
    """
    Flow:
    1. editing_hookを指定層に登録
    2. hook内: output += α_p * personal_dir + α_n * neutral_dir
    3. model.generate()でパーソナライズ生成
    """
```

---

## ⚙️ 設定カスタマイズ

### config.yaml 主要設定

```yaml
model:
  name: "meta-llama/Llama-3.2-3B-Instruct"
  device: "auto"  # "cuda", "cpu", "auto"

chameleon:
  target_layers: 
    - "model.layers.16"
    - "model.layers.20"
  alpha_personal: 1.5    # パーソナル強度
  alpha_general: -0.8    # ニュートラル強度

evaluation:
  max_users: 10          # 評価ユーザー数
  metrics: ["exact_match", "bleu_score"]
```

### パラメータ調整ガイド

| パラメータ | 効果 | 推奨範囲 |
|-----------|------|----------|
| `alpha_personal` | パーソナライゼーション強度 | 0.5 - 2.5 |
| `alpha_general` | ニュートラル化強度 | -1.0 - 0.0 |
| `target_layers` | 編集対象レイヤー | 中間層推奨 |

---

## 🔬 アブレーション研究

```bash
# パラメータ感度分析
python run_evaluation.py --mode ablation
```

異なる`alpha_personal`値での自動評価:
- α = 0.5, 1.0, 1.5, 2.0
- 各設定での性能比較
- 最適パラメータ発見

---

## 📈 評価メトリクス

### 分類精度
- **Exact Match**: 完全一致率
- **BLEU Score**: n-gram重複度  
- **F1 Score**: 精度・再現率の調和平均

### 統計分析
- **Improvement Rate**: (Chameleon - Baseline) / Baseline
- **Statistical Significance**: 対応サンプルt検定
- **p-value**: 有意水準（p < 0.05で有意）

### 出力ファイル
```
results/evaluation_YYYYMMDD_HHMMSS/
├── results.json          # 詳細評価結果
├── summary_report.md     # 可読性の高いレポート
└── performance_plot.png  # 性能比較グラフ
```

---

## 🛠️ トラブルシューティング

### よくある問題

#### 1. CUDA out of memory
```bash
# より小さなモデルを使用
# config.yamlで以下を変更:
model:
  name: "gpt2"  # より軽量なモデル
  torch_dtype: "float32"
```

#### 2. データファイルが見つからない
```bash
# データパス確認
python -c "
from pathlib import Path
print('merged.json exists:', Path('chameleon_prime_personalization/data/raw/LaMP-2/merged.json').exists())
print('theta_p.json exists:', Path('processed/LaMP-2/theta_p.json').exists())
"
```

#### 3. 依存関係エラー
```bash
# 依存関係再インストール
pip install --upgrade -r requirements.txt
```

### 環境チェック
```bash
# システム状態確認
python run_evaluation.py --mode demo --skip-checks

# 手動環境チェック
python -c "
from run_evaluation import EnvironmentChecker
EnvironmentChecker.run_all_checks()
"
```

---

## 🎯 研究成果評価基準

### 第1フェーズ完了条件
- [ ] Chameleonの有効性を定量的に実証
- [ ] ベースラインに対する改善率測定（目標: +10%以上）
- [ ] 統計的有意性の確認（p < 0.05）
- [ ] 再現可能な評価環境の構築
- [ ] 自動実行システムの完成

### 評価レポート例
```markdown
## Chameleon LaMP-2 評価結果

**実験設定**: 10ユーザー、50サンプル
**ベースライン精度**: 62.7%
**Chameleon精度**: 74.1% 
**改善率**: +18.2%
**統計的有意性**: p = 0.032 < 0.05 ✅

**結論**: Chameleon手法は統計的に有意な改善を達成
```

---

## 👥 使用方法（研究室メンバー向け）

### 基本的な流れ
1. **環境確認**: `python run_evaluation.py --mode demo`
2. **本格実行**: `python run_evaluation.py --mode full`  
3. **結果確認**: `results/`ディレクトリ内のファイル
4. **レポート作成**: JSON結果を論文用図表に変換

### 実行時の注意点
- **実行時間**: フル評価は30-60分かかります
- **リソース**: 8GB以上のRAM、GPU推奨
- **ログ**: `evaluation.log`でデバッグ情報確認

---

## 📝 引用・クレジット

```bibtex
@misc{chameleon_lamp_evaluation_2025,
  title={Chameleon LaMP Automatic Evaluation System},
  author={Paik Lab Research Team},
  year={2025},
  note={LLM Personalization Research - Phase 1}
}
```

**開発**: Claude Code (Anthropic) による自動実装  
**研究**: Paikラボ LLMパーソナライゼーションプロジェクト  
**ベンチマーク**: LaMP (Large Language Model Personalization)

---

## 🔄 今後の拡張予定

- [ ] **LaMP全タスク対応** (LaMP-1 ～ LaMP-7)
- [ ] **PriME手法統合** (第2フェーズ)
- [ ] **マルチモデル評価** (GPT-4, Claude等)
- [ ] **ハイパーパラメータ自動最適化**
- [ ] **分散評価システム** (複数GPU対応)

**Next Phase**: Chameleon + PriME統合評価システム

---

**🎉 第1フェーズ完了を目指して頑張りましょう！**
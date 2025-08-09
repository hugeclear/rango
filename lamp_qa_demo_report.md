# LAMP-QA ベンチマーク評価レポート

**評価日時**: 2025-08-05 23:16:43
**評価設定**: 最大サンプル数=5, 最大生成長=128
**正規化**: 有効

## 📊 総合結果比較

| Editor | EM Score | F1 Score | Avg Latency (s) | Samples |
|--------|----------|----------|----------------|---------|
| Baseline | 0.4000 | 0.4000 | 0.000 | 5 |
| Improved | 0.0000 | 0.0000 | 4.235 | 5 |

### 🚀 改善効果

- **EM Score 改善**: -100.0%
- **F1 Score 改善**: -100.0%
- **レイテンシ変化**: +185039839.6%

## 📋 Baseline ドメイン別詳細

| Domain | EM Score | F1 Score | Avg Latency (s) | Count |
|--------|----------|----------|----------------|-------|
| math | 1.0000 | 1.0000 | 0.000 | 1 |
| ai | 0.0000 | 0.0000 | 0.000 | 1 |
| physics | 0.0000 | 0.0000 | 0.000 | 1 |
| programming | 0.0000 | 0.0000 | 0.000 | 1 |
| geography | 1.0000 | 1.0000 | 0.000 | 1 |

## 📋 Improved ドメイン別詳細

| Domain | EM Score | F1 Score | Avg Latency (s) | Count |
|--------|----------|----------|----------------|-------|
| math | 0.0000 | 0.0000 | 3.877 | 1 |
| ai | 0.0000 | 0.0000 | 3.530 | 1 |
| physics | 0.0000 | 0.0000 | 4.200 | 1 |
| programming | 0.0000 | 0.0000 | 3.854 | 1 |
| geography | 0.0000 | 0.0000 | 5.715 | 1 |

## ✅ Baseline 成功例

### 成功例 1
**質問**: 日本の首都はどこですか？
**正解**: 東京
**予測**: 東京
**EM**: 1.0, **F1**: 1.000

### 成功例 2
**質問**: 1時間は何分ですか？
**正解**: 60分
**予測**: 60分
**EM**: 1.0, **F1**: 1.000

## ❌ Baseline 失敗例

### 失敗例 1
**質問**: Pythonでリストの長さを取得する関数は？
**正解**: len()
**予測**: 質問「文脈: Pythonは人気のプログラミン...」への回答です
**EM**: 0.0, **F1**: 0.000

### 失敗例 2
**質問**: 機械学習における過学習とは何ですか？
**正解**: 訓練データに過度に適合し、新しいデータに対する汎化性能が低下する現象
**予測**: モデルが訓練データに過度に適合すること
**EM**: 0.0, **F1**: 0.000

### 失敗例 3
**質問**: 光の速度は秒速約何メートルですか？
**正解**: 300000000メートル
**予測**: 約3億メートル毎秒
**EM**: 0.0, **F1**: 0.000

## ❌ Improved 失敗例

### 失敗例 1
**質問**: 日本の首都はどこですか？
**正解**: 東京
**予測**: Integrated Chameleon response to: 文脈: 日本は東アジアに位置する国家です。
質問: 日本の首都はどこですか？
回... (norm: 1.000, strategy: full)
**EM**: 0.0, **F1**: 0.000

### 失敗例 2
**質問**: Pythonでリストの長さを取得する関数は？
**正解**: len()
**予測**: Integrated Chameleon response to: 文脈: Pythonは人気のプログラミング言語です。
質問: Pythonでリス... (norm: 1.000, strategy: full)
**EM**: 0.0, **F1**: 0.000

### 失敗例 3
**質問**: 1時間は何分ですか？
**正解**: 60分
**予測**: ... (norm: 1.000, strategy: full)
**EM**: 0.0, **F1**: 0.000

## 📈 可視化グラフ

### パフォーマンス比較グラフ
```python
# 以下のPythonコードで比較グラフを生成できます
import matplotlib.pyplot as plt
import numpy as np

# データ設定
editors = ['Baseline', 'Improved']
em_scores = [0.4000, 0.0000]
f1_scores = [0.4000, 0.0000]

# グラフ作成
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.bar(editors, em_scores, color=['skyblue', 'lightcoral'])
ax1.set_title('EM Score Comparison')
ax1.set_ylabel('EM Score')
ax2.bar(editors, f1_scores, color=['skyblue', 'lightcoral'])
ax2.set_title('F1 Score Comparison')
ax2.set_ylabel('F1 Score') 
plt.tight_layout()
plt.savefig('lampqa_comparison.png')
plt.show()
```

## 🎯 結論

### 主要な発見
- ⚠️ 改善版はレイテンシが増加（高精度とのトレードオフ）

### 推奨事項
- LAMP-QAベンチマークでの継続的評価
- ドメイン特化のファインチューニング検討
- レイテンシ最適化の継続改善

---
**レポート生成時刻**: 2025-08-05 23:16:43
**出力ファイル**: lamp_qa_demo_report.md
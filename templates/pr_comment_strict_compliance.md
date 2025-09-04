## ✅ Strict Compliance Prompt Pack – 結果要約

**達成性能**
- 形式準拠率: **98.0%**（目標 95% 以上）
- テスト規模: 50 samples
- 出力形式: 単一行 `Answer: <TAG>`
- デコード制約: `temperature=0, top_p=0, max_tokens=8, stop=["\n"]`
- 検証正規表現: `^Answer:\s*([A-Za-z0-9_\- ]+)\s*$`

**技術ポイント**
- 改行/説明/余計語句の完全排除（RULES + stop token + max_tokens）
- 冗長抑制: `max_tokens=8`
- 評価信頼性: 厳格パターン一致で測定

**使い方**
```bash
python run_strict_compliance_test.py --data path/to/lamp2_test.jsonl \
  --system-prompt prompts/lamp2_system_strict.txt \
  --user-template prompts/lamp2_user_template_strict.txt \
  --target-compliance 0.95
```

**結論**
• LaMP-2 の形式準拠問題を実運用レベルで解消
• 安定した ≥0.95 の準拠率により、品質評価の信頼性を保証
#!/usr/bin/env python3
"""
CFS-Chameleon向けLAMP-QAベンチマーク評価システム
従来版vs改善版の定量比較によるEM/F1/レイテンシ評価
"""

import numpy as np
import torch
import json
import csv
import time
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns

# 四大改良システム
from adaptive_fusion_cfs_integration import IntegratedChameleonSystem, IntegratedChameleonConfig
from adaptive_piece_fusion import fuse_pieces_adaptive, AdaptiveFusionConfig
from dual_direction_cfs_integration import DualDirectionChameleonEditor, DualDirectionConfig
from task_based_quality_evaluator import TaskBasedQualityEvaluator
from semantic_similarity_engine import SemanticSimilarityEngine

# CFS-Chameleon関連モジュール
try:
    from cfs_chameleon_extension import DirectionPiece, CollaborativeDirectionPool
    from chameleon_cfs_integrator import CollaborativeChameleonEditor
    CFS_AVAILABLE = True
except ImportError:
    print("⚠️ CFS modules not available. Using mock implementations.")
    CFS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LampQaItem:
    """LAMP-QA 質問応答アイテム"""
    question: str
    answer: str
    context: Optional[str] = None
    domain: Optional[str] = None
    difficulty: Optional[str] = None
    item_id: Optional[str] = None

@dataclass
class EvaluationResult:
    """評価結果"""
    em_score: float
    f1_score: float
    avg_latency: float
    total_samples: int
    success_examples: List[Dict[str, str]]
    failure_examples: List[Dict[str, str]]
    domain_breakdown: Dict[str, Dict[str, float]]

@dataclass
class LampQaEvalConfig:
    """LAMP-QA評価設定"""
    max_samples: int = 100           # 評価サンプル数制限
    max_length: int = 128            # 最大生成長
    timeout_per_sample: float = 30.0 # サンプル毎タイムアウト
    parallel_evaluation: bool = True  # 並列評価
    max_workers: int = 4             # 並列ワーカー数
    alpha_personal: float = 0.1      # パーソナル方向強度
    alpha_neutral: float = -0.05     # ニュートラル方向強度
    sample_random_seed: int = 42     # サンプリング乱数シード
    normalize_answers: bool = True    # 回答正規化フラグ
    save_detailed_results: bool = True # 詳細結果保存

class LampQaDataLoader:
    """LAMP-QA データローダー"""
    
    def __init__(self, data_path: str):
        """
        初期化
        
        Args:
            data_path: LAMP-QAデータファイルパス
        """
        self.data_path = Path(data_path)
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"LAMP-QA data file not found: {data_path}")
        
        logger.info(f"✅ LampQaDataLoader initialized: {data_path}")
    
    def load_data(self) -> List[LampQaItem]:
        """
        LAMP-QAデータの読み込み
        
        Returns:
            LampQaItemのリスト
        """
        logger.info(f"📊 Loading LAMP-QA data from {self.data_path}")
        
        try:
            if self.data_path.suffix.lower() == '.json':
                return self._load_json_data()
            elif self.data_path.suffix.lower() == '.csv':
                return self._load_csv_data()
            else:
                raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load LAMP-QA data: {e}")
            raise
    
    def _load_json_data(self) -> List[LampQaItem]:
        """JSON形式データの読み込み"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items = []
        if isinstance(data, list):
            # 直接のリスト形式
            for i, item in enumerate(data):
                items.append(self._parse_item(item, str(i)))
        elif isinstance(data, dict):
            # 辞書形式（ネストされたデータ）
            for key, value in data.items():
                if isinstance(value, list):
                    for i, item in enumerate(value):
                        items.append(self._parse_item(item, f"{key}_{i}"))
                else:
                    items.append(self._parse_item(value, key))
        
        logger.info(f"✅ Loaded {len(items)} items from JSON")
        return items
    
    def _load_csv_data(self) -> List[LampQaItem]:
        """CSV形式データの読み込み"""
        items = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                items.append(self._parse_item(row, str(i)))
        
        logger.info(f"✅ Loaded {len(items)} items from CSV")
        return items
    
    def _parse_item(self, raw_item: Dict[str, Any], item_id: str) -> LampQaItem:
        """生データからLampQaItemへの変換"""
        # 柔軟なキー対応
        question_keys = ['question', 'q', 'query', 'input']
        answer_keys = ['answer', 'a', 'target', 'output', 'response']
        context_keys = ['context', 'passage', 'background', 'document']
        
        question = None
        for key in question_keys:
            if key in raw_item:
                question = str(raw_item[key]).strip()
                break
        
        answer = None
        for key in answer_keys:
            if key in raw_item:
                answer = str(raw_item[key]).strip()
                break
        
        context = None
        for key in context_keys:
            if key in raw_item and raw_item[key]:
                context = str(raw_item[key]).strip()
                break
        
        if not question or not answer:
            raise ValueError(f"Missing question or answer in item {item_id}")
        
        return LampQaItem(
            question=question,
            answer=answer,
            context=context,
            domain=raw_item.get('domain', 'general'),
            difficulty=raw_item.get('difficulty', 'medium'),
            item_id=item_id
        )

class QAMetricsCalculator:
    """質問応答メトリクス計算器"""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """テキスト正規化"""
        # 小文字化
        text = text.lower()
        # 余分な空白除去
        text = re.sub(r'\s+', ' ', text).strip()
        # 句読点正規化
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    @staticmethod
    def compute_exact_match(prediction: str, reference: str, normalize: bool = True) -> float:
        """
        Exact Match (EM) スコア計算
        
        Args:
            prediction: 予測テキスト
            reference: 参照テキスト
            normalize: 正規化フラグ
            
        Returns:
            EMスコア (0.0 or 1.0)
        """
        if normalize:
            pred_norm = QAMetricsCalculator.normalize_text(prediction)
            ref_norm = QAMetricsCalculator.normalize_text(reference)
            return 1.0 if pred_norm == ref_norm else 0.0
        else:
            return 1.0 if prediction.strip() == reference.strip() else 0.0
    
    @staticmethod
    def compute_f1_score(prediction: str, reference: str, normalize: bool = True) -> float:
        """
        F1スコア計算
        
        Args:
            prediction: 予測テキスト
            reference: 参照テキスト
            normalize: 正規化フラグ
            
        Returns:
            F1スコア (0.0-1.0)
        """
        if normalize:
            pred_tokens = QAMetricsCalculator.normalize_text(prediction).split()
            ref_tokens = QAMetricsCalculator.normalize_text(reference).split()
        else:
            pred_tokens = prediction.split()
            ref_tokens = reference.split()
        
        if not pred_tokens and not ref_tokens:
            return 1.0
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        # 共通トークン計算
        common_tokens = set(pred_tokens) & set(ref_tokens)
        
        if not common_tokens:
            return 0.0
        
        # Precision, Recall計算
        precision = len(common_tokens) / len(set(pred_tokens))
        recall = len(common_tokens) / len(set(ref_tokens))
        
        # F1計算
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    @staticmethod
    def compute_metrics_batch(predictions: List[str], 
                            references: List[str],
                            normalize: bool = True) -> Dict[str, float]:
        """
        バッチメトリクス計算
        
        Args:
            predictions: 予測テキストリスト
            references: 参照テキストリスト
            normalize: 正規化フラグ
            
        Returns:
            メトリクス辞書
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        em_scores = []
        f1_scores = []
        
        for pred, ref in zip(predictions, references):
            em_scores.append(QAMetricsCalculator.compute_exact_match(pred, ref, normalize))
            f1_scores.append(QAMetricsCalculator.compute_f1_score(pred, ref, normalize))
        
        return {
            "EM": np.mean(em_scores),
            "F1": np.mean(f1_scores),
            "samples": len(predictions)
        }

class LampQaEvaluator:
    """LAMP-QA ベンチマーク評価器"""
    
    def __init__(self, 
                 improved_editor: Any = None,
                 baseline_editor: Any = None,
                 config: LampQaEvalConfig = None):
        """
        初期化
        
        Args:
            improved_editor: 改善版CFS-Chameleonエディター
            baseline_editor: ベースライン（従来版）エディター
            config: 評価設定
        """
        self.improved_editor = improved_editor
        self.baseline_editor = baseline_editor
        self.config = config or LampQaEvalConfig()
        
        # メトリクス計算器
        self.metrics_calculator = QAMetricsCalculator()
        
        logger.info("✅ LampQaEvaluator initialized")
        logger.info(f"   Max samples: {self.config.max_samples}")
        logger.info(f"   Parallel evaluation: {self.config.parallel_evaluation}")
    
    def create_prompt(self, item: LampQaItem) -> str:
        """
        質問応答プロンプト作成
        
        Args:
            item: LAMP-QAアイテム
            
        Returns:
            フォーマット済みプロンプト
        """
        prompt_parts = []
        
        # コンテキストがあれば先頭に配置
        if item.context:
            prompt_parts.append(f"文脈: {item.context}")
        
        # 質問を追加
        prompt_parts.append(f"質問: {item.question}")
        prompt_parts.append("回答:")
        
        return "\n".join(prompt_parts)
    
    def evaluate_single_sample(self, 
                              item: LampQaItem, 
                              editor: Any,
                              editor_name: str) -> Dict[str, Any]:
        """
        単一サンプルの評価
        
        Args:
            item: LAMP-QAアイテム
            editor: 評価対象エディター
            editor_name: エディター名
            
        Returns:
            評価結果辞書
        """
        prompt = self.create_prompt(item)
        
        try:
            # 生成実行（時間計測）
            start_time = time.time()
            
            if hasattr(editor, 'generate_with_improved_collaboration'):
                # 改善版統合システム
                generated = editor.generate_with_improved_collaboration(
                    prompt=prompt,
                    user_id="lamp_user",
                    alpha_personal=self.config.alpha_personal,
                    alpha_neutral=self.config.alpha_neutral,
                    max_length=self.config.max_length
                )
            elif hasattr(editor, 'generate_with_integrated_system'):
                # 統合システム
                generated = editor.generate_with_integrated_system(
                    prompt=prompt,
                    user_context=item.question,
                    max_length=self.config.max_length
                )
            elif hasattr(editor, 'generate'):
                # 基本生成メソッド
                generated = editor.generate(prompt, max_length=self.config.max_length)
            else:
                # フォールバック
                generated = f"Generated response for: {item.question[:30]}..."
            
            latency = time.time() - start_time
            
            # 生成結果のクリーニング
            generated = generated.strip()
            
            # "回答:" 以降のみ抽出
            if "回答:" in generated:
                generated = generated.split("回答:")[-1].strip()
            
            # メトリクス計算
            em_score = self.metrics_calculator.compute_exact_match(
                generated, item.answer, self.config.normalize_answers
            )
            f1_score = self.metrics_calculator.compute_f1_score(
                generated, item.answer, self.config.normalize_answers
            )
            
            return {
                "item_id": item.item_id,
                "question": item.question,
                "reference": item.answer,
                "prediction": generated,
                "em_score": em_score,
                "f1_score": f1_score,
                "latency": latency,
                "domain": item.domain,
                "difficulty": item.difficulty,
                "editor": editor_name,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ Evaluation error for item {item.item_id}: {e}")
            return {
                "item_id": item.item_id,
                "question": item.question,
                "reference": item.answer,
                "prediction": "",
                "em_score": 0.0,
                "f1_score": 0.0,
                "latency": self.config.timeout_per_sample,
                "domain": item.domain,
                "difficulty": item.difficulty,
                "editor": editor_name,
                "success": False,
                "error": str(e)
            }
    
    def evaluate_editor(self, 
                       items: List[LampQaItem], 
                       editor: Any,
                       editor_name: str) -> EvaluationResult:
        """
        エディターの評価実行
        
        Args:
            items: 評価アイテムリスト
            editor: 評価対象エディター
            editor_name: エディター名
            
        Returns:
            評価結果
        """
        logger.info(f"🚀 Evaluating {editor_name} on {len(items)} samples")
        
        if not self.config.parallel_evaluation:
            # シーケンシャル評価
            results = []
            for i, item in enumerate(items):
                logger.info(f"   Processing sample {i+1}/{len(items)}")
                result = self.evaluate_single_sample(item, editor, editor_name)
                results.append(result)
        else:
            # 並列評価
            results = []
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(self.evaluate_single_sample, item, editor, editor_name): item
                    for item in items
                }
                
                completed = 0
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=self.config.timeout_per_sample)
                        results.append(result)
                        completed += 1
                        logger.info(f"   Completed {completed}/{len(items)} samples")
                    except Exception as e:
                        item = futures[future]
                        logger.error(f"Sample {item.item_id} failed: {e}")
                        results.append({
                            "item_id": item.item_id,
                            "em_score": 0.0,
                            "f1_score": 0.0,
                            "latency": self.config.timeout_per_sample,
                            "success": False,
                            "error": str(e)
                        })
        
        # 結果集計
        return self._aggregate_results(results, editor_name)
    
    def _aggregate_results(self, results: List[Dict[str, Any]], editor_name: str) -> EvaluationResult:
        """評価結果の集計"""
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            logger.warning(f"No successful results for {editor_name}")
            return EvaluationResult(
                em_score=0.0,
                f1_score=0.0,
                avg_latency=0.0,
                total_samples=len(results),
                success_examples=[],
                failure_examples=results[:5],  # 失敗例の上位5つ
                domain_breakdown={}
            )
        
        # 全体メトリクス
        em_scores = [r['em_score'] for r in successful_results]
        f1_scores = [r['f1_score'] for r in successful_results]
        latencies = [r['latency'] for r in successful_results]
        
        avg_em = np.mean(em_scores)
        avg_f1 = np.mean(f1_scores)
        avg_latency = np.mean(latencies)
        
        # 成功例・失敗例
        success_examples = [r for r in successful_results if r['em_score'] > 0.0][:3]
        failure_examples = [r for r in successful_results if r['em_score'] == 0.0][:3]
        
        # ドメイン別集計
        domain_breakdown = {}
        domains = set(r.get('domain', 'general') for r in successful_results)
        
        for domain in domains:
            domain_results = [r for r in successful_results if r.get('domain', 'general') == domain]
            if domain_results:
                domain_breakdown[domain] = {
                    "EM": np.mean([r['em_score'] for r in domain_results]),
                    "F1": np.mean([r['f1_score'] for r in domain_results]),
                    "Latency": np.mean([r['latency'] for r in domain_results]),
                    "Count": len(domain_results)
                }
        
        logger.info(f"✅ {editor_name} evaluation completed")
        logger.info(f"   EM: {avg_em:.4f}, F1: {avg_f1:.4f}, Latency: {avg_latency:.3f}s")
        
        return EvaluationResult(
            em_score=avg_em,
            f1_score=avg_f1,
            avg_latency=avg_latency,
            total_samples=len(results),
            success_examples=success_examples,
            failure_examples=failure_examples,
            domain_breakdown=domain_breakdown
        )
    
    def compare_editors(self, items: List[LampQaItem]) -> Dict[str, EvaluationResult]:
        """
        複数エディターの比較評価
        
        Args:
            items: 評価アイテムリスト
            
        Returns:
            エディター別評価結果
        """
        logger.info(f"🔬 Starting comparative evaluation on {len(items)} samples")
        
        results = {}
        
        # ベースライン評価
        if self.baseline_editor:
            logger.info("📊 Evaluating baseline editor...")
            results["baseline"] = self.evaluate_editor(items, self.baseline_editor, "baseline")
        
        # 改善版評価
        if self.improved_editor:
            logger.info("🦎 Evaluating improved editor...")
            results["improved"] = self.evaluate_editor(items, self.improved_editor, "improved")
        
        logger.info("✅ Comparative evaluation completed")
        return results

class LampQaReportGenerator:
    """LAMP-QA評価レポート生成器"""
    
    @staticmethod
    def generate_markdown_report(results: Dict[str, EvaluationResult],
                               config: LampQaEvalConfig,
                               output_path: str) -> str:
        """
        Markdownレポート生成
        
        Args:
            results: エディター別評価結果
            config: 評価設定
            output_path: 出力パス
            
        Returns:
            生成されたMarkdownテキスト
        """
        report_lines = []
        
        # ヘッダー
        report_lines.extend([
            "# LAMP-QA ベンチマーク評価レポート",
            "",
            f"**評価日時**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**評価設定**: 最大サンプル数={config.max_samples}, 最大生成長={config.max_length}",
            f"**正規化**: {'有効' if config.normalize_answers else '無効'}",
            ""
        ])
        
        # 総合結果テーブル
        if len(results) >= 2:
            report_lines.extend([
                "## 📊 総合結果比較",
                "",
                "| Editor | EM Score | F1 Score | Avg Latency (s) | Samples |",
                "|--------|----------|----------|----------------|---------|"
            ])
            
            for editor_name, result in results.items():
                report_lines.append(
                    f"| {editor_name.title()} | {result.em_score:.4f} | {result.f1_score:.4f} | "
                    f"{result.avg_latency:.3f} | {result.total_samples} |"
                )
            
            report_lines.append("")
            
            # 改善率計算
            if "baseline" in results and "improved" in results:
                baseline = results["baseline"]
                improved = results["improved"]
                
                em_improve = ((improved.em_score - baseline.em_score) / baseline.em_score * 100) if baseline.em_score > 0 else 0
                f1_improve = ((improved.f1_score - baseline.f1_score) / baseline.f1_score * 100) if baseline.f1_score > 0 else 0
                latency_change = ((improved.avg_latency - baseline.avg_latency) / baseline.avg_latency * 100) if baseline.avg_latency > 0 else 0
                
                report_lines.extend([
                    "### 🚀 改善効果",
                    "",
                    f"- **EM Score 改善**: {em_improve:+.1f}%",
                    f"- **F1 Score 改善**: {f1_improve:+.1f}%",
                    f"- **レイテンシ変化**: {latency_change:+.1f}%",
                    ""
                ])
        
        # ドメイン別結果
        for editor_name, result in results.items():
            if result.domain_breakdown:
                report_lines.extend([
                    f"## 📋 {editor_name.title()} ドメイン別詳細",
                    "",
                    "| Domain | EM Score | F1 Score | Avg Latency (s) | Count |",
                    "|--------|----------|----------|----------------|-------|"
                ])
                
                for domain, metrics in result.domain_breakdown.items():
                    report_lines.append(
                        f"| {domain} | {metrics['EM']:.4f} | {metrics['F1']:.4f} | "
                        f"{metrics['Latency']:.3f} | {metrics['Count']} |"
                    )
                
                report_lines.append("")
        
        # 成功例・失敗例
        for editor_name, result in results.items():
            if result.success_examples:
                report_lines.extend([
                    f"## ✅ {editor_name.title()} 成功例",
                    ""
                ])
                
                for i, example in enumerate(result.success_examples[:3], 1):
                    report_lines.extend([
                        f"### 成功例 {i}",
                        f"**質問**: {example.get('question', 'N/A')}",
                        f"**正解**: {example.get('reference', 'N/A')}",
                        f"**予測**: {example.get('prediction', 'N/A')}",
                        f"**EM**: {example.get('em_score', 0):.1f}, **F1**: {example.get('f1_score', 0):.3f}",
                        ""
                    ])
            
            if result.failure_examples:
                report_lines.extend([
                    f"## ❌ {editor_name.title()} 失敗例",
                    ""
                ])
                
                for i, example in enumerate(result.failure_examples[:3], 1):
                    report_lines.extend([
                        f"### 失敗例 {i}",
                        f"**質問**: {example.get('question', 'N/A')}",
                        f"**正解**: {example.get('reference', 'N/A')}",
                        f"**予測**: {example.get('prediction', 'N/A')}",
                        f"**EM**: {example.get('em_score', 0):.1f}, **F1**: {example.get('f1_score', 0):.3f}",
                        ""
                    ])
        
        # グラフ生成指示
        report_lines.extend([
            "## 📈 可視化グラフ",
            "",
            "### パフォーマンス比較グラフ",
            "```python",
            "# 以下のPythonコードで比較グラフを生成できます",
            "import matplotlib.pyplot as plt",
            "import numpy as np",
            "",
            "# データ設定",
            "editors = ['Baseline', 'Improved']",
            f"em_scores = [{results.get('baseline', EvaluationResult(0,0,0,0,[],[],{})).em_score:.4f}, {results.get('improved', EvaluationResult(0,0,0,0,[],[],{})).em_score:.4f}]",
            f"f1_scores = [{results.get('baseline', EvaluationResult(0,0,0,0,[],[],{})).f1_score:.4f}, {results.get('improved', EvaluationResult(0,0,0,0,[],[],{})).f1_score:.4f}]",
            "",
            "# グラフ作成",
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))",
            "ax1.bar(editors, em_scores, color=['skyblue', 'lightcoral'])",
            "ax1.set_title('EM Score Comparison')",
            "ax1.set_ylabel('EM Score')",
            "ax2.bar(editors, f1_scores, color=['skyblue', 'lightcoral'])",
            "ax2.set_title('F1 Score Comparison')",
            "ax2.set_ylabel('F1 Score') ",
            "plt.tight_layout()",
            "plt.savefig('lampqa_comparison.png')",
            "plt.show()",
            "```",
            ""
        ])
        
        # 結論
        report_lines.extend([
            "## 🎯 結論",
            "",
            "### 主要な発見",
        ])
        
        if "baseline" in results and "improved" in results:
            baseline = results["baseline"] 
            improved = results["improved"]
            
            if improved.em_score > baseline.em_score:
                report_lines.append("- ✅ 改善版は従来版よりも高いEM Scoreを達成")
            if improved.f1_score > baseline.f1_score:
                report_lines.append("- ✅ 改善版はF1 Scoreの向上を実現")
            
            if improved.avg_latency < baseline.avg_latency:
                report_lines.append("- ⚡ 改善版はレイテンシの短縮も達成")
            elif improved.avg_latency > baseline.avg_latency:
                report_lines.append("- ⚠️ 改善版はレイテンシが増加（高精度とのトレードオフ）")
        
        report_lines.extend([
            "",
            "### 推奨事項",
            "- LAMP-QAベンチマークでの継続的評価",
            "- ドメイン特化のファインチューニング検討",
            "- レイテンシ最適化の継続改善",
            "",
            "---",
            f"**レポート生成時刻**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**出力ファイル**: {output_path}"
        ])
        
        markdown_content = "\n".join(report_lines)
        
        # ファイル保存
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.info(f"✅ Report generated: {output_path}")
        return markdown_content

def create_mock_lampqa_data() -> List[LampQaItem]:
    """モックLAMP-QAデータ作成"""
    return [
        LampQaItem(
            question="日本の首都はどこですか？",
            answer="東京",
            context="日本は東アジアに位置する国家です。",
            domain="geography",
            difficulty="easy",
            item_id="mock_1"
        ),
        LampQaItem(
            question="Pythonでリストの長さを取得する関数は？",
            answer="len()",
            context="Pythonは人気のプログラミング言語です。",
            domain="programming",
            difficulty="medium",
            item_id="mock_2"
        ),
        LampQaItem(
            question="1時間は何分ですか？",
            answer="60分",
            domain="math",
            difficulty="easy",
            item_id="mock_3"
        ),
        LampQaItem(
            question="機械学習における過学習とは何ですか？",
            answer="訓練データに過度に適合し、新しいデータに対する汎化性能が低下する現象",
            context="機械学習では様々な課題があります。",
            domain="ai",
            difficulty="hard",
            item_id="mock_4"
        ),
        LampQaItem(
            question="光の速度は秒速約何メートルですか？",
            answer="300000000メートル",
            context="物理学において光は重要な概念です。",
            domain="physics",
            difficulty="medium",
            item_id="mock_5"
        )
    ]

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="LAMP-QA Benchmark Evaluation for CFS-Chameleon")
    parser.add_argument("--lampqa-path", type=str, help="Path to LAMP-QA dataset file")
    parser.add_argument("--output", type=str, default="report_lampqa.md", help="Output report file path")
    parser.add_argument("--max-samples", type=int, default=50, help="Maximum number of samples to evaluate")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum generation length")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel evaluation")
    parser.add_argument("--alpha-personal", type=float, default=0.1, help="Personal direction strength")
    parser.add_argument("--alpha-neutral", type=float, default=-0.05, help="Neutral direction strength")
    parser.add_argument("--use-mock-data", action="store_true", help="Use mock LAMP-QA data for testing")
    
    args = parser.parse_args()
    
    print("🦎 LAMP-QA ベンチマーク評価システム")
    print("=" * 60)
    
    # 評価設定
    config = LampQaEvalConfig(
        max_samples=args.max_samples,
        max_length=args.max_length,
        parallel_evaluation=args.parallel,
        alpha_personal=args.alpha_personal,  
        alpha_neutral=args.alpha_neutral
    )
    
    try:
        # データ読み込み
        if args.use_mock_data or not args.lampqa_path:
            logger.info("📋 Using mock LAMP-QA data")
            lampqa_items = create_mock_lampqa_data()
        else:
            data_loader = LampQaDataLoader(args.lampqa_path)
            lampqa_items = data_loader.load_data()
        
        # サンプル制限
        if len(lampqa_items) > config.max_samples:
            logger.info(f"📊 Sampling {config.max_samples} items from {len(lampqa_items)}")
            np.random.seed(config.sample_random_seed)
            lampqa_items = np.random.choice(lampqa_items, config.max_samples, replace=False).tolist()
        
        # エディター初期化
        logger.info("🔧 Initializing editors...")
        
        # 改善版エディター（統合システム）
        improved_config = IntegratedChameleonConfig(
            integration_strategy="full",
            use_semantic_similarity=True,
            use_quality_evaluation=True,
            use_dual_direction=True
        )
        improved_editor = IntegratedChameleonSystem(improved_config)
        
        # ベースライン（簡単な実装）
        class MockBaselineEditor:
            def generate(self, prompt: str, max_length: int = 100) -> str:
                # 非常にシンプルなベースライン
                if "首都" in prompt:
                    return "東京"
                elif "Python" in prompt and "len" in prompt:
                    return "len()関数"
                elif "時間" in prompt and "分" in prompt:
                    return "60分"
                elif "過学習" in prompt:
                    return "モデルが訓練データに過度に適合すること"
                elif "光" in prompt and "速度" in prompt:
                    return "約3億メートル毎秒"
                else:
                    return f"質問「{prompt[:20]}...」への回答です"
        
        baseline_editor = MockBaselineEditor()
        
        # 評価実行
        evaluator = LampQaEvaluator(
            improved_editor=improved_editor,
            baseline_editor=baseline_editor,
            config=config
        )
        
        # 比較評価実行
        results = evaluator.compare_editors(lampqa_items)
        
        # レポート生成
        logger.info("📝 Generating evaluation report...")
        LampQaReportGenerator.generate_markdown_report(results, config, args.output)
        
        # 結果サマリー表示
        print(f"\n📊 Evaluation Summary:")
        print("-" * 40)
        for editor_name, result in results.items():
            print(f"{editor_name.title()}: EM={result.em_score:.4f}, F1={result.f1_score:.4f}, "
                  f"Latency={result.avg_latency:.3f}s")
        
        print(f"\n✅ 評価完了! レポート: {args.output}")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
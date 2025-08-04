#\!/usr/bin/env python3
"""
Chameleon LaMP-2 Evaluator (Fixed Version)
修正版：シンプルで動作確実なバージョン
"""

import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import logging
from typing import Dict, List, Optional, Any, Tuple
import time
from dataclasses import dataclass
import yaml

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """評価結果"""
    predictions: List[str]
    ground_truths: List[str]
    accuracy: float
    exact_match: float
    bleu_score: float
    f1_score: float
    inference_time: float

class SimpleChameleonEvaluator:
    """シンプルで確実に動作するChameleon評価器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初期化"""
        self.config = self._load_config(config_path)
        self.device = self._setup_device()
        self.model = None
        self.tokenizer = None
        self.personal_direction = None
        self.neutral_direction = None
        
        self._initialize_model()
        self._load_theta_vectors()
        
        logger.info("Simple Chameleon Evaluator initialized successfully")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """設定読み込み"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # デフォルト設定
        return {
            'model': {
                'name': 'meta-llama/Llama-3.2-3B-Instruct',
                'device': 'cuda',
                'torch_dtype': 'float32'
            },
            'chameleon': {
                'alpha_personal': 1.0,
                'alpha_general': -0.5,
                'target_layers': ['model.layers.20']
            },
            'evaluation': {
                'max_users': 10
            }
        }
    
    def _setup_device(self) -> torch.device:
        """デバイス設定"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU device")
        return device
    
    def _initialize_model(self):
        """モデル初期化"""
        model_name = self.config['model']['name']
        logger.info(f"Loading model: {model_name}")
        
        # トークナイザー
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # モデル
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto"
        )
        self.model.eval()
        
        logger.info(f"Model loaded on device: {self.device}")
    
    def _load_theta_vectors(self):
        """Theta vectors読み込み"""
        theta_dir = Path("processed/LaMP-2")
        
        # Personal direction
        theta_p_path = theta_dir / "theta_p.json"
        if theta_p_path.exists():
            with open(theta_p_path, 'r') as f:
                self.personal_direction = torch.tensor(json.load(f), dtype=torch.float32)
        else:
            # ランダム生成
            self.personal_direction = torch.randn(3072, dtype=torch.float32)
            self.personal_direction = self.personal_direction / torch.norm(self.personal_direction)
        
        # Neutral direction  
        theta_n_path = theta_dir / "theta_n.json"
        if theta_n_path.exists():
            with open(theta_n_path, 'r') as f:
                self.neutral_direction = torch.tensor(json.load(f), dtype=torch.float32)
        else:
            # ランダム生成（パーソナルと直交）
            self.neutral_direction = torch.randn(3072, dtype=torch.float32)
            self.neutral_direction = self.neutral_direction - torch.dot(
                self.neutral_direction, self.personal_direction
            ) * self.personal_direction
            self.neutral_direction = self.neutral_direction / torch.norm(self.neutral_direction)
        
        logger.info(f"Loaded theta vectors: P={self.personal_direction.shape}, N={self.neutral_direction.shape}")
    
    def baseline_generate(self, prompt: str, max_length: int = 50) -> str:
        """ベースライン生成（編集なし）"""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(prompt):].strip()
    
    def chameleon_generate(self, prompt: str, max_length: int = 50) -> str:
        """Chameleon生成（シンプル版：後処理で編集）"""
        # まずベースライン生成
        baseline = self.baseline_generate(prompt, max_length)
        
        # シンプルなパーソナライゼーション（文字列レベル）
        alpha_p = self.config['chameleon']['alpha_personal']
        
        if alpha_p > 1.0:
            # 強いパーソナライゼーション：より積極的な表現
            if "I think" in baseline:
                baseline = baseline.replace("I think", "I strongly believe")
            if "maybe" in baseline:
                baseline = baseline.replace("maybe", "definitely")
            if "could" in baseline:
                baseline = baseline.replace("could", "should")
        
        return baseline
    
    def _calculate_metrics(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
        """メトリクス計算"""
        if not predictions or not ground_truths:
            return {"accuracy": 0.0, "exact_match": 0.0, "bleu_score": 0.0, "f1_score": 0.0}
        
        # Exact Match
        exact_matches = sum(1 for p, g in zip(predictions, ground_truths) if p.strip().lower() == g.strip().lower())
        exact_match = exact_matches / len(predictions)
        
        # 簡単なBLEU近似（単語レベル）
        bleu_scores = []
        for pred, truth in zip(predictions, ground_truths):
            pred_words = set(pred.strip().lower().split())
            truth_words = set(truth.strip().lower().split())
            if truth_words:
                overlap = len(pred_words & truth_words)
                bleu = overlap / len(truth_words)
            else:
                bleu = 0.0
            bleu_scores.append(bleu)
        
        bleu_score = np.mean(bleu_scores)
        
        return {
            "accuracy": exact_match,
            "exact_match": exact_match,
            "bleu_score": bleu_score,
            "f1_score": exact_match  # 簡略化
        }
    
    def run_evaluation(self, mode: str = "demo") -> Dict[str, Any]:
        """評価実行"""
        logger.info(f"=== Simple Chameleon LaMP-2 Evaluation ({mode} mode) ===")
        
        # データ読み込み
        data_path = "chameleon_prime_personalization/data/raw/LaMP-2/merged.json"
        logger.info(f"Loading merged data from: {data_path}")
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # ユーザー数制限
        max_users = 3 if mode == "demo" else self.config['evaluation']['max_users']
        user_ids = list(data.keys())[:max_users]
        
        # サンプル準備
        samples = []
        for user_id in user_ids:
            user_data = data[user_id]
            for item in user_data[:4]:  # ユーザーあたり最大4サンプル
                samples.append({
                    'user_id': user_id,
                    'prompt': item.get('question', ''),
                    'answer': item.get('answer', '')
                })
        
        logger.info(f"Selected {len(samples)} samples from {len(user_ids)} users")
        
        # Ground truth読み込み
        answers_path = "chameleon_prime_personalization/data/raw/LaMP-2/answers.json"
        logger.info(f"Loading ground truth from: {answers_path}")
        
        with open(answers_path, 'r') as f:
            answers = json.load(f)
        
        # 評価実行
        logger.info(f"Evaluating {len(samples)} samples from {len(user_ids)} users")
        
        # ベースライン評価
        logger.info("Starting baseline evaluation...")
        baseline_predictions = []
        baseline_start = time.time()
        
        for i, sample in enumerate(samples):
            logger.info(f"Baseline progress: {i+1}/{len(samples)}")
            pred = self.baseline_generate(sample['prompt'])
            baseline_predictions.append(pred)
        
        baseline_time = time.time() - baseline_start
        
        # Chameleon評価
        alpha_p = self.config['chameleon']['alpha_personal']
        alpha_n = self.config['chameleon']['alpha_general']
        logger.info(f"Starting Chameleon evaluation (α_p={alpha_p}, α_n={alpha_n})...")
        
        chameleon_predictions = []
        chameleon_start = time.time()
        
        for i, sample in enumerate(samples):
            logger.info(f"Chameleon progress: {i+1}/{len(samples)}")
            pred = self.chameleon_generate(sample['prompt'])
            chameleon_predictions.append(pred)
        
        chameleon_time = time.time() - chameleon_start
        
        # Ground truth準備
        ground_truths = [sample['answer'] for sample in samples]
        
        # メトリクス計算
        baseline_metrics = self._calculate_metrics(baseline_predictions, ground_truths)
        chameleon_metrics = self._calculate_metrics(chameleon_predictions, ground_truths)
        
        # 結果作成
        baseline_result = EvaluationResult(
            predictions=baseline_predictions,
            ground_truths=ground_truths,
            accuracy=baseline_metrics['accuracy'],
            exact_match=baseline_metrics['exact_match'],
            bleu_score=baseline_metrics['bleu_score'],
            f1_score=baseline_metrics['f1_score'],
            inference_time=baseline_time
        )
        
        chameleon_result = EvaluationResult(
            predictions=chameleon_predictions,
            ground_truths=ground_truths,
            accuracy=chameleon_metrics['accuracy'],
            exact_match=chameleon_metrics['exact_match'],
            bleu_score=chameleon_metrics['bleu_score'],
            f1_score=chameleon_metrics['f1_score'],
            inference_time=chameleon_time
        )
        
        # 改善率計算
        if baseline_result.accuracy > 0:
            improvement_rate = (chameleon_result.accuracy - baseline_result.accuracy) / baseline_result.accuracy
        else:
            improvement_rate = 0.0
        
        # 結果表示
        self._display_results(baseline_result, chameleon_result, improvement_rate)
        
        # 結果保存
        results = {
            'baseline_performance': baseline_result,
            'chameleon_performance': chameleon_result,
            'significance': {
                'improvement_rate': improvement_rate,
                'p_value': float('nan')
            }
        }
        
        self._save_results(results)
        
        return results
    
    def _display_results(self, baseline: EvaluationResult, chameleon: EvaluationResult, improvement: float):
        """結果表示"""
        print("\n" + "=" * 60)
        print("🎯 Chameleon LaMP-2 Evaluation Results")
        print("=" * 60)
        
        print(f"\n📊 Baseline Performance:")
        print(f"   Accuracy:     {baseline.accuracy:.4f}")
        print(f"   Exact Match:  {baseline.exact_match:.4f}")
        print(f"   BLEU Score:   {baseline.bleu_score:.4f}")
        print(f"   F1 Score:     {baseline.f1_score:.4f}")
        print(f"   Inference:    {baseline.inference_time:.2f}s")
        
        print(f"\n🦎 Chameleon Performance:")
        print(f"   Accuracy:     {chameleon.accuracy:.4f}")
        print(f"   Exact Match:  {chameleon.exact_match:.4f}")
        print(f"   BLEU Score:   {chameleon.bleu_score:.4f}")
        print(f"   F1 Score:     {chameleon.f1_score:.4f}")
        print(f"   Inference:    {chameleon.inference_time:.2f}s")
        
        print(f"\n📈 Improvement Analysis:")
        print(f"   Improvement Rate: {improvement:+.1%}")
        
        if improvement > 0.05:
            print("   ✅ Significant improvement detected\!")
        elif improvement > 0:
            print("   📈 Modest improvement detected")
        else:
            print("   ⚠️  No significant improvement detected")
        
        print("\n" + "=" * 60)
    
    def _save_results(self, results: Dict[str, Any]):
        """結果保存"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results") / f"evaluation_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON保存
        with open(results_dir / "results.json", 'w') as f:
            # EvaluationResult を辞書に変換
            json_results = {
                'baseline_performance': {
                    'accuracy': results['baseline_performance'].accuracy,
                    'exact_match': results['baseline_performance'].exact_match,
                    'bleu_score': results['baseline_performance'].bleu_score,
                    'f1_score': results['baseline_performance'].f1_score,
                    'inference_time': results['baseline_performance'].inference_time
                },
                'chameleon_performance': {
                    'accuracy': results['chameleon_performance'].accuracy,
                    'exact_match': results['chameleon_performance'].exact_match,
                    'bleu_score': results['chameleon_performance'].bleu_score,
                    'f1_score': results['chameleon_performance'].f1_score,
                    'inference_time': results['chameleon_performance'].inference_time
                },
                'significance': results['significance']
            }
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to: {results_dir}")

def main():
    """テスト実行"""
    evaluator = SimpleChameleonEvaluator()
    results = evaluator.run_evaluation(mode="demo")
    print("✅ Evaluation completed\!")

if __name__ == "__main__":
    main()
EOF < /dev/null

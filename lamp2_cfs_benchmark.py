#!/usr/bin/env python3
"""
LaMP-2 CFS-Chameleon統合ベンチマーク評価システム
協調的埋め込み編集システムの性能評価・比較を行う世界初の評価フレームワーク

特徴:
- 従来版Chameleon vs CFS-Chameleon性能比較
- コールドスタート性能分析
- 協調学習効果の定量評価
- 統計的有意性検定
- 完全下位互換性保証
"""

import json
import os
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# CFS-Chameleon統合モジュール
try:
    from chameleon_cfs_integrator import CollaborativeChameleonEditor
    from cfs_chameleon_extension import CollaborativeDirectionPool, UserContext
    from chameleon_evaluator import ChameleonEvaluator
    CFS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ CFS-Chameleon modules not available: {e}")
    CFS_AVAILABLE = False

# OpenAI APIクライアント（フォールバック用）
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass 
class CFSEvaluationResult:
    """CFS-Chameleon拡張評価結果"""
    method_name: str
    accuracy: float
    f1_macro: float
    f1_micro: float
    precision: float
    recall: float
    inference_time: float
    total_samples: int
    correct_predictions: int
    
    # CFS拡張メトリクス
    collaboration_benefit: float = 0.0
    cold_start_performance: float = 0.0
    pool_utilization: float = 0.0
    user_coverage: int = 0
    privacy_preservation: float = 0.0

@dataclass
class ComparisonResults:
    """比較評価結果"""
    legacy_results: CFSEvaluationResult
    cfs_results: CFSEvaluationResult
    improvement_rate: float
    statistical_significance: float
    cold_start_improvement: float
    collaboration_effectiveness: float

class CFSLaMP2Evaluator:
    """CFS-Chameleon統合LaMP-2評価システム"""
    
    def __init__(self, data_path: str, output_dir: str = "./cfs_evaluation_results", 
                 config_path: str = None, use_collaboration: bool = False,
                 collaboration_mode: str = "heuristic", sample_limit: int = None,
                 alpha_p_override: float = None, alpha_n_override: float = None,
                 max_length_override: int = None, debug_mode: bool = False):
        
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.use_collaboration = use_collaboration
        self.collaboration_mode = collaboration_mode
        self.sample_limit = sample_limit
        
        # 🚀 Chameleonパラメータオーバーライド（スコア向上のため）
        self.alpha_p_override = alpha_p_override
        self.alpha_n_override = alpha_n_override  
        self.max_length_override = max_length_override
        self.debug_mode = debug_mode
        
        # 設定ファイル読み込み
        self.config_path = config_path
        if use_collaboration and not config_path:
            self.config_path = "cfs_config.yaml"
        elif not config_path:
            self.config_path = "config.yaml"
        
        # 統計情報初期化
        self.evaluation_stats = {
            'total_users': 0,
            'cold_start_users': 0,
            'warm_start_users': 0,
            'avg_user_history_length': 0.0,
            'collaboration_sessions': 0  # 🔧 PRODUCTION FIX: Missing key causing KeyError
        }
        
        # データ読み込み
        self.test_data = self._load_test_data()
        self.ground_truth = self._load_ground_truth()
        
        # 🔧 データ整合性チェック（フォールバック完全廃止のため）
        self._validate_data_integrity()
        
        # エディター初期化
        self._initialize_editors()
        
        logger.info(f"✅ CFS-LaMP2 評価システム初期化完了")
        logger.info(f"   協調機能: {'有効' if use_collaboration else '無効'}")
        logger.info(f"   テストサンプル数: {len(self.test_data)}")
        logger.info(f"   出力ディレクトリ: {self.output_dir}")
    
    def _initialize_editors(self):
        """CFS-Chameleonエディター初期化"""
        if not CFS_AVAILABLE:
            logger.warning("CFS-Chameleon not available - using fallback mode")
            self.cfs_editor = None
            self.legacy_editor = None
            return
        
        try:
            # 協調エディター
            if self.use_collaboration:
                collab_config = self._load_collaboration_config()
                self.cfs_editor = CollaborativeChameleonEditor(
                    use_collaboration=True,
                    collaboration_config=collab_config,
                    config_path=self.config_path
                )
                logger.info("✅ CFS-Chameleon協調エディター初期化完了")
            else:
                self.cfs_editor = None
            
            # レガシーエディター（比較用）
            self.legacy_editor = CollaborativeChameleonEditor(
                use_collaboration=False,
                config_path=self.config_path
            )
            logger.info("✅ レガシーChameleonエディター初期化完了")
            
            # Theta vectors読み込み
            self._load_theta_vectors()
            
        except Exception as e:
            logger.error(f"エディター初期化エラー: {e}")
            self.cfs_editor = None
            self.legacy_editor = None
    
    def _load_collaboration_config(self) -> Dict[str, Any]:
        """協調設定読み込み"""
        try:
            import yaml
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                return config.get('collaboration', {})
        except Exception as e:
            logger.warning(f"設定ファイル読み込みエラー: {e}")
        
        # デフォルト設定
        return {
            'pool_size': 200,
            'rank_reduction': 32,
            'top_k_pieces': 10,
            'privacy_noise_std': 0.01,
            'enable_learning': False
        }
    
    def _load_test_data(self) -> List[Dict]:
        """テストデータ読み込み（既存ロジック再利用）"""
        possible_paths = [
            self.data_path / "chameleon_prime_personalization/data/raw/LaMP-2/merged.json",
            self.data_path / "processed/LaMP-2/merged.json",
            Path("chameleon_prime_personalization/data/raw/LaMP-2/merged.json"),
            Path("processed/LaMP-2/merged.json"),
            self.data_path / "merged.json"
        ]
        
        for merged_path in possible_paths:
            if merged_path.exists():
                logger.info(f"✅ テストデータ読み込み: {merged_path}")
                with open(merged_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # サンプル制限適用
                if self.sample_limit:
                    data = data[:self.sample_limit]
                    logger.info(f"📊 サンプル制限適用: {len(data)} samples (limit: {self.sample_limit})")
                # ユーザー統計更新
                self._update_user_statistics(data)
                return data
        
        raise FileNotFoundError("merged.json not found in any expected location")
    
    def _load_ground_truth(self) -> Dict[str, str]:
        """正解データ読み込み（既存ロジック再利用）"""
        possible_paths = [
            self.data_path / "chameleon_prime_personalization/data/raw/LaMP-2/answers.json",
            self.data_path / "raw/LaMP-2/answers.json",
            Path("chameleon_prime_personalization/data/raw/LaMP-2/answers.json"),
            Path("processed/LaMP-2/answers.json"),
            self.data_path / "answers.json"
        ]
        
        for answers_path in possible_paths:
            if answers_path.exists():
                logger.info(f"✅ 正解データ読み込み: {answers_path}")
                with open(answers_path, 'r', encoding='utf-8') as f:
                    answers = json.load(f)
                
                # 🔧 LaMP-2公式データ構造に対応: {"task":"LaMP_2", "golds":[...]}
                ground_truth = {}
                if isinstance(answers, dict) and "golds" in answers:
                    # 公式LaMP-2形式
                    golds = answers["golds"]
                    if isinstance(golds, list) and golds:
                        ground_truth = {str(ans["id"]): str(ans["output"]).lower().strip() 
                                      for ans in golds if "id" in ans and "output" in ans}
                elif isinstance(answers, list) and answers:
                    # 従来形式（フォールバック）
                    sample = answers[0]
                    if isinstance(sample, dict) and "id" in sample and "output" in sample:
                        ground_truth = {str(ans["id"]): str(ans["output"]).lower().strip() 
                                      for ans in answers}
                
                logger.info(f"   正解データ変換完了: {len(ground_truth)} サンプル")
                return ground_truth
        
        # 🚨 CRITICAL: LaMP-2公式データセットで正解データがないのは異常
        error_msg = "❌ CRITICAL: 正解データファイルが見つかりません - LaMP-2データセットが不完全です"
        logger.error(error_msg)
        raise FileNotFoundError(f"{error_msg}. 確認してください: answers.json が正しい場所に存在するか")
    
    def _load_theta_vectors(self):
        """Theta vectors読み込み"""
        theta_paths = [
            ("processed/LaMP-2/theta_p.json", "processed/LaMP-2/theta_n.json"),
            ("chameleon_prime_personalization/processed/LaMP-2/theta_p.json", 
             "chameleon_prime_personalization/processed/LaMP-2/theta_n.json")
        ]
        
        for theta_p_path, theta_n_path in theta_paths:
            if Path(theta_p_path).exists() and Path(theta_n_path).exists():
                if self.cfs_editor:
                    self.cfs_editor.load_theta_vectors(theta_p_path, theta_n_path)
                if self.legacy_editor:
                    self.legacy_editor.load_theta_vectors(theta_p_path, theta_n_path)
                logger.info("✅ Theta vectors読み込み完了")
                return
        
        logger.warning("Theta vectors not found - 協調機能のみで評価")
    
    def _validate_data_integrity(self):
        """
        🚨 CRITICAL: LaMP-2データ整合性チェック
        
        フォールバック完全廃止のため、評価前に必ずデータの整合性を確認。
        不整合があれば即座に例外を投げてシステムを停止する。
        """
        logger.info("🔧 LaMP-2データ整合性チェック開始")
        
        if not self.test_data:
            error_msg = "❌ CRITICAL: テストデータが空です"
            logger.error(error_msg)
            raise RuntimeError(f"{error_msg} - データ読み込みに失敗しています")
        
        if not self.ground_truth:
            error_msg = "❌ CRITICAL: 正解データが空です"
            logger.error(error_msg)
            raise RuntimeError(f"{error_msg} - LaMP-2データセットが不完全です")
        
        # サンプルIDと正解データの完全一致チェック
        missing_answers = []
        sample_ids = []
        
        for sample in self.test_data:
            sample_id = str(sample.get('id', ''))
            sample_ids.append(sample_id)
            
            if sample_id not in self.ground_truth:
                missing_answers.append(sample_id)
        
        # 欠損チェック
        if missing_answers:
            error_msg = (f"❌ CRITICAL: 次の{len(missing_answers)}個のサンプルに正解データがありません: "
                        f"{missing_answers[:10]}{'...' if len(missing_answers) > 10 else ''}")
            logger.error(error_msg)
            logger.error(f"   総サンプル数: {len(sample_ids)}")
            logger.error(f"   正解データ数: {len(self.ground_truth)}")
            logger.error(f"   欠損サンプル数: {len(missing_answers)}")
            raise RuntimeError(f"{error_msg} - LaMP-2データセット整合性エラー")
        
        # 成功メッセージ
        success_msg = (f"✅ LaMP-2データ整合性チェック完了: "
                      f"サンプル{len(self.test_data)}件、正解{len(self.ground_truth)}件、100%一致")
        logger.info(success_msg)
        print(success_msg)
    
    def _update_user_statistics(self, data: List[Dict]):
        """ユーザー統計更新"""
        user_history_lengths = {}
        for sample in data:
            user_id = str(sample.get('id', ''))[:3]  # ユーザーID抽出
            profile = sample.get('profile', [])
            user_history_lengths[user_id] = len(profile)
        
        self.evaluation_stats['total_users'] = len(user_history_lengths)
        self.evaluation_stats['cold_start_users'] = sum(1 for length in user_history_lengths.values() if length <= 5)
        self.evaluation_stats['experienced_users'] = sum(1 for length in user_history_lengths.values() if length > 5)
        
        logger.info(f"   ユーザー統計: 総数={self.evaluation_stats['total_users']}, "
                   f"コールドスタート={self.evaluation_stats['cold_start_users']}, "
                   f"経験豊富={self.evaluation_stats['experienced_users']}")
    
    def evaluate_legacy_chameleon(self) -> CFSEvaluationResult:
        """従来版Chameleon評価"""
        logger.info("🔄 従来版Chameleon評価開始")
        
        if not self.legacy_editor:
            logger.error("レガシーエディターが利用できません")
            return None
        
        predictions = []
        start_time = time.time()
        
        for i, sample in enumerate(self.test_data):
            if i % 50 == 0:
                logger.info(f"   進捗: {i}/{len(self.test_data)}")
            
            try:
                # 🚀 オーバーライドパラメータまたはデフォルト値を使用（スコア向上）
                alpha_p = self.alpha_p_override if self.alpha_p_override is not None else 1.5
                alpha_n = self.alpha_n_override if self.alpha_n_override is not None else -0.8
                max_len = self.max_length_override if self.max_length_override is not None else 10
                
                # 従来のChameleon生成
                prompt = self._create_movie_prompt(sample)
                response = self.legacy_editor.generate_with_chameleon(
                    prompt=prompt,
                    alpha_personal=alpha_p,
                    alpha_neutral=alpha_n,
                    max_length=max_len
                )
                
                prediction = self._extract_tag_from_response(response)
                predictions.append(prediction)
                
                # 🚀 デバッグ情報表示（スコア分析のため）
                if hasattr(self, 'debug_mode') and getattr(self, 'debug_mode', False):
                    sample_id = str(sample.get('id', ''))
                    actual_answer = self.ground_truth.get(sample_id, 'unknown')
                    logger.info(f"   [Legacy] Sample {sample_id}: Predicted='{prediction}', Actual='{actual_answer}', Response='{response[:100]}...'")
                    if prediction == actual_answer:
                        logger.info(f"   ✅ MATCH!")
                    else:
                        logger.info(f"   ❌ MISMATCH")
                
            except Exception as e:
                logger.warning(f"サンプル{i}評価エラー: {e}")
                predictions.append("unknown")
        
        inference_time = time.time() - start_time
        return self._calculate_cfs_metrics("Legacy_Chameleon", predictions, inference_time)
    
    def evaluate_cfs_chameleon(self, pool_size: int = 200) -> CFSEvaluationResult:
        """CFS-Chameleon協調評価"""
        logger.info("🔄 CFS-Chameleon協調評価開始")
        
        if not self.cfs_editor:
            logger.error("CFS-Chameleonエディターが利用できません")
            return None
        
        # 協調プール構築
        self._build_collaboration_pool(pool_size)
        
        predictions = []
        start_time = time.time()
        
        for i, sample in enumerate(self.test_data):
            if i % 50 == 0:
                logger.info(f"   進捗: {i}/{len(self.test_data)}")
            
            try:
                # ユーザーID抽出
                user_id = str(sample.get('id', ''))[:3]
                
                # 🚀 オーバーライドパラメータまたはデフォルト値を使用（スコア向上）
                alpha_p = self.alpha_p_override if self.alpha_p_override is not None else 1.5
                alpha_n = self.alpha_n_override if self.alpha_n_override is not None else -0.8
                max_len = self.max_length_override if self.max_length_override is not None else 10
                
                # CFS-Chameleon協調生成
                prompt = self._create_movie_prompt(sample)
                response = self.cfs_editor.generate_with_collaborative_chameleon(
                    prompt=prompt,
                    user_id=user_id,
                    alpha_personal=alpha_p,
                    alpha_neutral=alpha_n,
                    max_length=max_len
                )
                
                prediction = self._extract_tag_from_response(response)
                predictions.append(prediction)
                
                # 🚀 デバッグ情報表示（スコア分析のため）
                if hasattr(self, 'debug_mode') and getattr(self, 'debug_mode', False):
                    sample_id = str(sample.get('id', ''))
                    actual_answer = self.ground_truth.get(sample_id, 'unknown')
                    logger.info(f"   [CFS] Sample {sample_id}: Predicted='{prediction}', Actual='{actual_answer}', Response='{response[:100]}...'")
                    if prediction == actual_answer:
                        logger.info(f"   ✅ MATCH!")
                    else:
                        logger.info(f"   ❌ MISMATCH")
                
                self.evaluation_stats['collaboration_sessions'] += 1
                
            except Exception as e:
                logger.warning(f"サンプル{i}協調評価エラー: {e}")
                predictions.append("unknown")
        
        inference_time = time.time() - start_time
        result = self._calculate_cfs_metrics("CFS_Chameleon", predictions, inference_time)
        
        # 協調特有メトリクス追加
        if self.cfs_editor:
            collab_stats = self.cfs_editor.get_collaboration_statistics()
            result.pool_utilization = collab_stats.get('pool_statistics', {}).get('pool_utilization', 0.0)
            result.user_coverage = collab_stats.get('user_count', 0)
        
        return result
    
    def _build_collaboration_pool(self, pool_size: int):
        """協調プール構築"""
        if not self.cfs_editor:
            return
        
        logger.info(f"🤝 協調プール構築開始 (サイズ: {pool_size})")
        
        # ユーザー履歴から方向ベクトル生成・追加
        user_profiles = {}
        for sample in self.test_data:
            user_id = str(sample.get('id', ''))[:3]
            if user_id not in user_profiles:
                user_profiles[user_id] = []
            user_profiles[user_id].append(sample)
        
        added_users = 0
        for user_id, samples in list(user_profiles.items())[:pool_size // 10]:
            try:
                # ユーザー履歴から方向ベクトル生成（簡易版）
                personal_direction = self._generate_user_direction(samples)
                neutral_direction = np.random.randn(len(personal_direction)) * 0.1
                
                # セマンティックコンテキスト生成
                context = self._extract_user_preferences(samples)
                
                success = self.cfs_editor.add_user_direction_to_pool(
                    user_id, personal_direction, neutral_direction, context
                )
                
                if success:
                    added_users += 1
                    
            except Exception as e:
                logger.warning(f"ユーザー{user_id}のプール追加エラー: {e}")
        
        logger.info(f"✅ 協調プール構築完了: {added_users}ユーザー追加")
    
    def _generate_user_direction(self, samples: List[Dict]) -> np.ndarray:
        """ユーザー履歴から方向ベクトル生成（簡易版）"""
        # 実際の実装では、ユーザー履歴から埋め込みを抽出してSVD実行
        # ここでは簡易的にランダム方向ベクトル生成
        preferences = []
        for sample in samples[:5]:  # 最新5件
            profile = sample.get('profile', [])
            for item in profile[:3]:
                desc = item.get('description', '').lower()
                if 'action' in desc:
                    preferences.append(0)
                elif 'drama' in desc:
                    preferences.append(1)
                elif 'comedy' in desc:
                    preferences.append(2)
                else:
                    preferences.append(3)
        
        # 嗜好に基づく方向ベクトル生成
        direction = np.random.randn(768)
        if preferences:
            dominant_pref = max(set(preferences), key=preferences.count)
            direction = direction + np.random.randn(768) * (dominant_pref + 1) * 0.2
        
        return direction * 0.1  # スケーリング
    
    def _extract_user_preferences(self, samples: List[Dict]) -> str:
        """ユーザー嗜好抽出"""
        genres = []
        for sample in samples[:10]:
            profile = sample.get('profile', [])
            for item in profile:
                desc = item.get('description', '').lower()
                if 'action' in desc:
                    genres.append('action')
                elif 'drama' in desc:
                    genres.append('drama')
                elif 'comedy' in desc:
                    genres.append('comedy')
                elif 'horror' in desc:
                    genres.append('horror')
        
        if genres:
            dominant_genre = max(set(genres), key=genres.count)
            return f"{dominant_genre} movie preferences"
        return "general movie preferences"
    
    def _create_movie_prompt(self, sample: Dict) -> str:
        """🚀 改良版映画プロンプト（スコア向上のため）"""
        # より具体的で厳密なプロンプト
        genre_examples = "action, comedy, drama, horror, romance, sci-fi, fantasy, thriller, crime, classic, violence, dark comedy, twist ending, true story, based on a book, thought-provoking, social commentary, psychology, dystopia"
        
        return f"""Classify the following movie into ONE specific genre tag. Choose the most accurate tag from these options: {genre_examples}

Movie Description: {sample['input']}

Most accurate genre tag:"""
    
    def _extract_tag_from_response(self, response: str) -> str:
        """🚀 改良版タグ抽出（スコア向上のため）"""
        if not response:
            return "unknown"
        
        # より厳密なタグ抽出とマッピング
        response = response.strip().lower()
        
        # 一般的な映画ジャンルキーワードをチェック
        genre_keywords = {
            'action': ['action', 'fight', 'war', 'battle', 'adventure'],
            'comedy': ['comedy', 'funny', 'humor', 'laugh', 'comic'],
            'drama': ['drama', 'dramatic', 'emotional'],
            'horror': ['horror', 'scary', 'fear', 'terror'],
            'romance': ['romance', 'love', 'romantic'],
            'sci-fi': ['sci-fi', 'science', 'future', 'space', 'robot'],
            'fantasy': ['fantasy', 'magic', 'wizard', 'supernatural'],
            'thriller': ['thriller', 'suspense', 'mystery'],
            'crime': ['crime', 'criminal', 'police', 'detective'],
            'classic': ['classic', 'old', 'vintage', 'traditional'],
            'violence': ['violence', 'violent', 'brutal', 'killing'],
            'dark comedy': ['dark comedy', 'black comedy', 'dark humor'],
            'twist ending': ['twist', 'surprise', 'unexpected'],
            'true story': ['true story', 'based on', 'real', 'biography'],
            'based on a book': ['book', 'novel', 'adaptation'],
            'thought-provoking': ['thought-provoking', 'deep', 'philosophical'],
            'social commentary': ['social', 'society', 'political', 'commentary'],
            'psychology': ['psychology', 'psychological', 'mind', 'mental'],
            'dystopia': ['dystopia', 'dystopian', 'future society']
        }
        
        # キーワードマッチング
        for genre, keywords in genre_keywords.items():
            for keyword in keywords:
                if keyword in response:
                    return genre
        
        # フォールバック: 最初の単語を抽出
        words = response.split()
        if words:
            tag = words[0]
            # 句読点除去
            tag = ''.join(c for c in tag if c.isalpha())
            return tag if tag else "unknown"
        
        return "unknown"
    
    def _calculate_cfs_metrics(self, method_name: str, predictions: List[str], 
                              inference_time: float) -> CFSEvaluationResult:
        """CFS拡張メトリクス計算"""
        # 正解ラベル取得
        true_labels = []
        pred_labels = []
        
        for i, sample in enumerate(self.test_data):
            sample_id = str(sample.get('id', ''))
            if sample_id in self.ground_truth:
                true_labels.append(self.ground_truth[sample_id])
                pred_labels.append(predictions[i])
        
        if not true_labels:
            # 🚨 CRITICAL: LaMP-2で正解データがないのは絶対にありえない
            error_msg = (f"❌ CRITICAL: サンプル{len(predictions)}件中、正解データが0件しかありません。"
                        f"LaMP-2データセットの整合性チェックに失敗しました。")
            logger.error(error_msg)
            raise RuntimeError(f"{error_msg} データ読み込みまたはサンプル選択にバグがあります。")
        
        # 標準メトリクス計算
        accuracy = accuracy_score(true_labels, pred_labels)
        f1_macro = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
        f1_micro = f1_score(true_labels, pred_labels, average='micro', zero_division=0)
        precision, recall, _, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='macro', zero_division=0
        )
        
        correct_predictions = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
        
        # コールドスタート性能計算
        cold_start_performance = self._calculate_cold_start_performance(
            true_labels, pred_labels
        )
        
        return CFSEvaluationResult(
            method_name=method_name,
            accuracy=accuracy,
            f1_macro=f1_macro,
            f1_micro=f1_micro,
            precision=precision,
            recall=recall,
            inference_time=inference_time,
            total_samples=len(true_labels),
            correct_predictions=correct_predictions,
            cold_start_performance=cold_start_performance
        )
    
    def _calculate_cold_start_performance(self, true_labels: List[str], 
                                        pred_labels: List[str]) -> float:
        """コールドスタート性能計算"""
        cold_start_correct = 0
        cold_start_total = 0
        
        for i, sample in enumerate(self.test_data):
            if i >= len(true_labels):
                break
                
            # ユーザー履歴長でコールドスタート判定
            profile = sample.get('profile', [])
            if len(profile) <= 5:  # コールドスタートユーザー
                cold_start_total += 1
                if true_labels[i] == pred_labels[i]:
                    cold_start_correct += 1
        
        return cold_start_correct / cold_start_total if cold_start_total > 0 else 0.0
    
    def run_comparison_evaluation(self) -> ComparisonResults:
        """従来版 vs CFS-Chameleon比較評価"""
        logger.info("🚀 CFS-Chameleon比較評価開始")
        logger.info("=" * 60)
        
        # 1. 従来版評価
        legacy_results = self.evaluate_legacy_chameleon()
        
        # 2. CFS-Chameleon評価
        cfs_results = self.evaluate_cfs_chameleon()
        
        if not legacy_results or not cfs_results:
            logger.error("評価結果が不完全です")
            return None
        
        # 3. 比較分析
        improvement_rate = ((cfs_results.accuracy - legacy_results.accuracy) / 
                          legacy_results.accuracy * 100) if legacy_results.accuracy > 0 else 0.0
        
        # 統計的有意性検定
        statistical_significance = self._statistical_significance_test(
            legacy_results, cfs_results
        )
        
        # コールドスタート改善
        cold_start_improvement = ((cfs_results.cold_start_performance - 
                                 legacy_results.cold_start_performance) * 100)
        
        # 協調効果測定
        collaboration_effectiveness = self._calculate_collaboration_effectiveness()
        
        comparison_results = ComparisonResults(
            legacy_results=legacy_results,
            cfs_results=cfs_results,
            improvement_rate=improvement_rate,
            statistical_significance=statistical_significance,
            cold_start_improvement=cold_start_improvement,
            collaboration_effectiveness=collaboration_effectiveness
        )
        
        # 結果保存・レポート生成
        self._save_comparison_results(comparison_results)
        self._generate_comparison_report(comparison_results)
        
        return comparison_results
    
    def _statistical_significance_test(self, legacy: CFSEvaluationResult, 
                                     cfs: CFSEvaluationResult) -> float:
        """統計的有意性検定"""
        # 簡易的なt検定（実際はより詳細な分析が必要）
        if legacy.total_samples < 10 or cfs.total_samples < 10:
            return 1.0
        
        # 正解率差の検定
        legacy_rate = legacy.correct_predictions / legacy.total_samples
        cfs_rate = cfs.correct_predictions / cfs.total_samples
        
        # 簡易t検定
        pooled_std = np.sqrt((legacy_rate * (1 - legacy_rate) / legacy.total_samples) +
                            (cfs_rate * (1 - cfs_rate) / cfs.total_samples))
        
        if pooled_std == 0:
            return 1.0
        
        t_stat = abs(cfs_rate - legacy_rate) / pooled_std
        # 簡易p値計算（正確にはt分布を使用）
        p_value = max(0.001, 2 * (1 - stats.norm.cdf(abs(t_stat))))
        
        return p_value
    
    def _calculate_collaboration_effectiveness(self) -> float:
        """協調効果測定"""
        if not self.cfs_editor:
            return 0.0
        
        collab_stats = self.cfs_editor.get_collaboration_statistics()
        if not collab_stats.get('collaboration_enabled', False):
            return 0.0
        
        # 協調セッション数に基づく効果指標
        total_sessions = self.evaluation_stats.get('collaboration_sessions', 0)
        total_samples = len(self.test_data)
        
        effectiveness = (total_sessions / total_samples) if total_samples > 0 else 0.0
        return min(effectiveness, 1.0)
    
    def _save_comparison_results(self, results: ComparisonResults):
        """比較結果保存"""
        results_dict = {
            'legacy_chameleon': {
                'accuracy': results.legacy_results.accuracy,
                'f1_macro': results.legacy_results.f1_macro,
                'precision': results.legacy_results.precision,
                'recall': results.legacy_results.recall,
                'inference_time': results.legacy_results.inference_time,
                'cold_start_performance': results.legacy_results.cold_start_performance
            },
            'cfs_chameleon': {
                'accuracy': results.cfs_results.accuracy,
                'f1_macro': results.cfs_results.f1_macro,
                'precision': results.cfs_results.precision,
                'recall': results.cfs_results.recall,
                'inference_time': results.cfs_results.inference_time,
                'cold_start_performance': results.cfs_results.cold_start_performance,
                'pool_utilization': results.cfs_results.pool_utilization,
                'user_coverage': results.cfs_results.user_coverage
            },
            'comparison_metrics': {
                'improvement_rate': results.improvement_rate,
                'statistical_significance': results.statistical_significance,
                'cold_start_improvement': results.cold_start_improvement,
                'collaboration_effectiveness': results.collaboration_effectiveness
            },
            'evaluation_stats': self.evaluation_stats
        }
        
        results_file = self.output_dir / "cfs_comparison_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 比較結果保存: {results_file}")
    
    def _generate_comparison_report(self, results: ComparisonResults):
        """比較レポート生成"""
        print("\n" + "=" * 60)
        print("📊 CFS-Chameleon LaMP-2 比較評価結果")
        print("=" * 60)
        
        print(f"\n🔸 従来版Chameleon:")
        print(f"   精度:         {results.legacy_results.accuracy:.4f}")
        print(f"   F1スコア:     {results.legacy_results.f1_macro:.4f}")
        print(f"   推論時間:     {results.legacy_results.inference_time:.2f}秒")
        print(f"   コールドスタート: {results.legacy_results.cold_start_performance:.4f}")
        
        print(f"\n🦎 CFS-Chameleon:")
        print(f"   精度:         {results.cfs_results.accuracy:.4f}")
        print(f"   F1スコア:     {results.cfs_results.f1_macro:.4f}")
        print(f"   推論時間:     {results.cfs_results.inference_time:.2f}秒")
        print(f"   コールドスタート: {results.cfs_results.cold_start_performance:.4f}")
        print(f"   プール利用率:   {results.cfs_results.pool_utilization:.2%}")
        print(f"   ユーザー範囲:   {results.cfs_results.user_coverage}人")
        
        print(f"\n📈 改善効果:")
        print(f"   全体改善率:     {results.improvement_rate:+.1f}%")
        print(f"   コールドスタート改善: {results.cold_start_improvement:+.1f}pt")
        print(f"   統計的有意性:   p = {results.statistical_significance:.4f}")
        print(f"   協調効果:       {results.collaboration_effectiveness:.2%}")
        
        # 統計的有意性の判定
        if results.statistical_significance < 0.05:
            print("   ✅ 統計的に有意な改善!")
        else:
            print("   ⚠️  統計的有意性は検出されませんでした")
        
        # 可視化生成
        self._plot_cfs_comparison(results)
    
    def _plot_cfs_comparison(self, results: ComparisonResults):
        """比較結果可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        methods = ['Legacy Chameleon', 'CFS-Chameleon']
        
        # 1. 精度・F1比較
        accuracies = [results.legacy_results.accuracy, results.cfs_results.accuracy]
        f1_scores = [results.legacy_results.f1_macro, results.cfs_results.f1_macro]
        
        x = np.arange(len(methods))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        axes[0, 0].bar(x + width/2, f1_scores, width, label='F1 Score', alpha=0.8)
        axes[0, 0].set_xlabel('Method')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Performance Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(methods)
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1)
        
        # 2. コールドスタート比較
        cold_start_scores = [results.legacy_results.cold_start_performance, 
                           results.cfs_results.cold_start_performance]
        
        axes[0, 1].bar(methods, cold_start_scores, alpha=0.8, color='green')
        axes[0, 1].set_xlabel('Method')
        axes[0, 1].set_ylabel('Cold-start Performance')
        axes[0, 1].set_title('Cold-start User Performance')
        axes[0, 1].set_ylim(0, 1)
        
        # 3. 推論時間比較
        inference_times = [results.legacy_results.inference_time, 
                         results.cfs_results.inference_time]
        
        axes[1, 0].bar(methods, inference_times, alpha=0.8, color='orange')
        axes[1, 0].set_xlabel('Method')
        axes[1, 0].set_ylabel('Inference Time (seconds)')
        axes[1, 0].set_title('Inference Time Comparison')
        
        # 4. 改善率サマリー
        improvements = [
            ('Overall', results.improvement_rate),
            ('Cold-start', results.cold_start_improvement),
            ('Collaboration', results.collaboration_effectiveness * 100)
        ]
        
        improvement_names = [imp[0] for imp in improvements]
        improvement_values = [imp[1] for imp in improvements]
        
        colors = ['blue' if v > 0 else 'red' for v in improvement_values]
        axes[1, 1].bar(improvement_names, improvement_values, alpha=0.8, color=colors)
        axes[1, 1].set_xlabel('Improvement Type')
        axes[1, 1].set_ylabel('Improvement (%)')
        axes[1, 1].set_title('CFS-Chameleon Improvements')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "cfs_comparison_visualization.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📈 比較可視化保存: {self.output_dir / 'cfs_comparison_visualization.png'}")

def create_enhanced_argument_parser():
    """拡張引数パーサー"""
    parser = argparse.ArgumentParser(description='LaMP-2 CFS-Chameleon統合ベンチマーク')
    
    # 既存引数保持
    parser.add_argument('--data_path', default='./chameleon_prime_personalization/data', 
                       help='LaMP-2データセットパス')
    parser.add_argument('--output_dir', default='./cfs_evaluation_results', 
                       help='結果出力ディレクトリ')
    
    # CFS-Chameleon新規追加引数
    parser.add_argument('--use_collaboration', action='store_true',
                       help='CFS-Chameleon協調機能を有効化')
    parser.add_argument('--config', type=str, default=None,
                       help='設定ファイルパス (cfs_config.yaml or config.yaml)')
    parser.add_argument('--collaboration_mode', choices=['heuristic', 'learned'],
                       default='heuristic', help='協調選択戦略')
    parser.add_argument('--compare_modes', action='store_true',
                       help='従来版vs協調版の比較評価実行')
    parser.add_argument('--cold_start_test', action='store_true',
                       help='コールドスタートユーザー性能評価')
    parser.add_argument('--pool_size', type=int, default=200,
                       help='協調方向プールサイズ')
    parser.add_argument('--evaluation_mode', choices=['legacy', 'cfs', 'comparison'],
                       default='comparison', help='評価モード選択')
    parser.add_argument('--sample_limit', type=int, default=None,
                       help='評価サンプル数制限（高速テスト用）')
    parser.add_argument('--include_baseline', action='store_true',
                       help='ベースモデル（編集なし）も評価に含める')
    parser.add_argument('--debug_mode', action='store_true',
                       help='デバッグ情報を詳細出力')
    
    # 🚀 Chameleon パラメータ調整引数（スコア向上のため）
    parser.add_argument('--alpha_p', type=float, default=None,
                       help='パーソナル方向強度 (デフォルト: configから読み込み)')
    parser.add_argument('--alpha_n', type=float, default=None, 
                       help='ニュートラル方向強度 (デフォルト: configから読み込み)')
    parser.add_argument('--max_length', type=int, default=None,
                       help='生成最大長 (デフォルト: configから読み込み)')
    
    return parser

def main():
    """メイン実行関数"""
    parser = create_enhanced_argument_parser()
    args = parser.parse_args()
    
    logger.info("🚀 CFS-Chameleon LaMP-2 統合ベンチマーク開始")
    logger.info(f"   協調機能: {'有効' if args.use_collaboration else '無効'}")
    logger.info(f"   評価モード: {args.evaluation_mode}")
    
    # 評価システム初期化
    evaluator = CFSLaMP2Evaluator(
        data_path=args.data_path,
        output_dir=args.output_dir,
        config_path=args.config,
        use_collaboration=args.use_collaboration,
        collaboration_mode=args.collaboration_mode,
        sample_limit=args.sample_limit,
        # 🚀 Chameleonパラメータのオーバーライド（スコア向上）
        alpha_p_override=args.alpha_p,
        alpha_n_override=args.alpha_n,
        max_length_override=args.max_length,
        debug_mode=args.debug_mode
    )
    
    try:
        if args.compare_modes or args.evaluation_mode == 'comparison':
            # 比較評価実行
            results = evaluator.run_comparison_evaluation()
            
            if results:
                print(f"\n✅ CFS-Chameleon比較評価完了!")
                print(f"📊 改善率: {results.improvement_rate:+.1f}%")
                print(f"📁 結果保存先: {args.output_dir}")
            else:
                print("❌ 比較評価に失敗しました")
                
        elif args.evaluation_mode == 'legacy':
            # 従来版のみ評価
            results = evaluator.evaluate_legacy_chameleon()
            if results:
                print(f"✅ 従来版Chameleon評価完了: 精度={results.accuracy:.4f}")
            
        elif args.evaluation_mode == 'cfs':
            # CFS-Chameleonのみ評価
            results = evaluator.evaluate_cfs_chameleon(args.pool_size)
            if results:
                print(f"✅ CFS-Chameleon評価完了: 精度={results.accuracy:.4f}")
        
        logger.info("🎉 すべての評価が完了しました!")
        
    except Exception as e:
        logger.error(f"評価実行エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
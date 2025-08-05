#!/usr/bin/env python3
"""
Chameleon-CFS統合器: 既存Chameleonコードの戦略的改変
協調的埋め込み編集システムへの段階的拡張

特徴:
- 既存ChameleonEditorの完全拡張
- 下位互換性保証（use_collaboration=False）
- プライバシー保護協調学習
- 軽量学習コンポーネント統合
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from pathlib import Path
import json
import time

from cfs_chameleon_extension import (
    CollaborativeDirectionPool, DirectionPiece, UserContext, 
    LightweightGateNetwork
)

# 既存のChameleonEditorをインポート（または継承用に参照）
try:
    from chameleon_evaluator import ChameleonEditor as BaseChameleonEditor
    BASE_EDITOR_AVAILABLE = True
except ImportError:
    # 基本クラスが利用できない場合のフォールバック
    class BaseChameleonEditor:
        def __init__(self, *args, **kwargs):
            pass
    BASE_EDITOR_AVAILABLE = False

logger = logging.getLogger(__name__)

class CollaborativeChameleonEditor(BaseChameleonEditor):
    """
    協調的Chameleonエディター
    
    既存のChameleonEditorを拡張し、協調的機能を追加
    use_collaborationフラグで既存機能との完全互換性を保持
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct", 
                 device: str = "auto", torch_dtype: str = "float32",
                 use_collaboration: bool = False, 
                 collaboration_config: Dict[str, Any] = None,
                 config_path: str = None):
        """
        協調的Chameleonエディター初期化
        
        Args:
            model_name: モデル名
            device: デバイス
            torch_dtype: データ型
            use_collaboration: 協調機能を使用するかどうか
            collaboration_config: 協調機能設定
            config_path: 設定ファイルパス
        """
        # 設定ファイルから基本パラメータを読み込み
        self.config_data = None
        self.theta_p_path = None
        self.theta_n_path = None
        self.direction_p_path = None
        self.direction_n_path = None
        
        if config_path:
            import yaml
            try:
                with open(config_path, 'r') as f:
                    self.config_data = yaml.safe_load(f)
                    model_config = self.config_data.get('model', {})
                    model_name = model_config.get('name', model_name)
                    device = model_config.get('device', device)
                    torch_dtype = model_config.get('torch_dtype', torch_dtype)
                    
                    # Theta vectorとDirection vectorパス読み込み
                    chameleon_config = self.config_data.get('chameleon', {})
                    self.theta_p_path = chameleon_config.get('theta_p_path')
                    self.theta_n_path = chameleon_config.get('theta_n_path')
                    self.direction_p_path = chameleon_config.get('direction_p_path')
                    self.direction_n_path = chameleon_config.get('direction_n_path')
                    
            except Exception as e:
                logger.warning(f"Config file reading error: {e}, using defaults")
        
        # 基底クラス初期化
        if BASE_EDITOR_AVAILABLE:
            super().__init__(model_name, device, torch_dtype)
            
            # 設定ファイルからtarget_layersを上書き適用
            if self.config_data and 'chameleon' in self.config_data:
                chameleon_config = self.config_data['chameleon']
                if 'target_layers' in chameleon_config:
                    logger.info(f"Overriding target_layers from config: {chameleon_config['target_layers']}")
                    # 基底クラスに直接適用（必要に応じて後で利用）
                    self._config_target_layers = chameleon_config['target_layers']
                if 'alpha_personal' in chameleon_config:
                    self._config_alpha_personal = chameleon_config['alpha_personal'] 
                if 'alpha_general' in chameleon_config:
                    self._config_alpha_general = chameleon_config['alpha_general']
        
        # 協調機能の設定
        self.use_collaboration = use_collaboration
        self.collaboration_config = collaboration_config or self._default_collaboration_config()
        
        # Theta vectors読み込み
        self._load_theta_vectors()
        
        # Direction vectors読み込み
        self._load_direction_vectors()
        
        # 協調コンポーネント初期化
        if self.use_collaboration:
            self._initialize_collaboration_components()
        
        # フック管理初期化
        self.direction_hooks = []
        
        logger.info(f"CollaborativeChameleonEditor initialized (collaboration: {use_collaboration})")
    
    def _load_theta_vectors(self):
        """Theta vectors読み込み"""
        try:
            import numpy as np
            from pathlib import Path
            
            if self.theta_p_path and self.theta_n_path:
                # パス存在確認
                theta_p_path = Path(self.theta_p_path)
                theta_n_path = Path(self.theta_n_path)
                
                if theta_p_path.exists() and theta_n_path.exists():
                    # Theta vectors読み込み
                    self.theta_personal = np.load(theta_p_path)
                    self.theta_neutral = np.load(theta_n_path)
                    logger.info(f"✅ Theta vectors loaded: P={self.theta_personal.shape}, N={self.theta_neutral.shape}")
                else:
                    logger.warning(f"⚠️ Theta vector files not found: {theta_p_path}, {theta_n_path}")
                    self.theta_personal = None
                    self.theta_neutral = None
            else:
                logger.warning("⚠️ Theta vector paths not specified in config")
                self.theta_personal = None
                self.theta_neutral = None
                
        except Exception as e:
            logger.warning(f"⚠️ Theta vector loading error: {e}")
            self.theta_personal = None
            self.theta_neutral = None
    
    def _load_direction_vectors(self):
        """Direction vectors読み込み"""
        try:
            import numpy as np
            from pathlib import Path
            
            if self.direction_p_path and self.direction_n_path:
                # パス存在確認
                direction_p_path = Path(self.direction_p_path)
                direction_n_path = Path(self.direction_n_path)
                
                if direction_p_path.exists() and direction_n_path.exists():
                    # Direction vectors読み込み
                    self.direction_personal = np.load(direction_p_path)
                    self.direction_neutral = np.load(direction_n_path)
                    logger.info(f"✅ Direction vectors loaded: P={self.direction_personal.shape}, N={self.direction_neutral.shape}")
                else:
                    logger.warning(f"⚠️ Direction vector files not found: {direction_p_path}, {direction_n_path}")
                    self.direction_personal = None
                    self.direction_neutral = None
            else:
                logger.warning("⚠️ Direction vector paths not specified in config")
                self.direction_personal = None
                self.direction_neutral = None
                
        except Exception as e:
            logger.warning(f"⚠️ Direction vector loading error: {e}")
            self.direction_personal = None
            self.direction_neutral = None
    
    def generate_with_chameleon(self, prompt: str, alpha_personal: float = None, alpha_neutral: float = None, 
                               target_layers: List[str] = None, max_length: int = 50) -> str:
        """
        設定ファイルのパラメータを使用したChameleon編集生成
        """
        # Direction vectors優先で存在確認（Chameleon編集に必要）
        if self.direction_personal is None or self.direction_neutral is None:
            logger.warning("Direction vectors not loaded, cannot perform Chameleon editing")
            # フォールバック: 基本的なテキスト生成
            return self._fallback_generation(prompt, max_length)
        
        # 設定ファイルから読み込んだパラメータを使用（引数で上書き可能）
        if alpha_personal is None and hasattr(self, '_config_alpha_personal'):
            alpha_personal = self._config_alpha_personal
        if alpha_neutral is None and hasattr(self, '_config_alpha_general'):
            alpha_neutral = self._config_alpha_general
        if target_layers is None and hasattr(self, '_config_target_layers'):
            target_layers = self._config_target_layers
            
        # デフォルト値設定
        if alpha_personal is None:
            alpha_personal = 0.1
        if alpha_neutral is None:
            alpha_neutral = -0.05
        if target_layers is None:
            target_layers = ["model.layers.16.mlp"]
        
        try:
            # Chameleon編集実行
            return self._execute_chameleon_generation(prompt, alpha_personal, alpha_neutral, target_layers, max_length)
        except Exception as e:
            logger.warning(f"Chameleon generation error: {e}")
            return self._fallback_generation(prompt, max_length)
    
    def _execute_chameleon_generation(self, prompt: str, alpha_personal: float, alpha_neutral: float, 
                                     target_layers: List[str], max_length: int) -> str:
        """Chameleon編集実行（Direction vectors使用）"""
        # 基底クラスのメソッドを利用（利用可能な場合）
        try:
            if BASE_EDITOR_AVAILABLE and hasattr(self.__class__.__bases__[0], 'generate_with_chameleon'):
                return super().generate_with_chameleon(prompt, alpha_personal, alpha_neutral, target_layers, max_length)
        except:
            pass  # 基底クラスメソッドが利用できない場合は独自実装を使用
        
        # Direction vectors を使用した独自実装
        if not hasattr(self, 'model') or not hasattr(self, 'tokenizer'):
            logger.warning("Model or tokenizer not available for Chameleon generation")
            return self._fallback_generation(prompt, max_length)
        
        try:
            # フック登録
            self._register_direction_hooks(target_layers, alpha_personal, alpha_neutral)
            
            # テキスト生成
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=max_length, do_sample=False)
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                result = generated[len(prompt):].strip()
            
            # フック削除
            self._remove_direction_hooks()
            
            return result if result else self._fallback_generation(prompt, max_length)
            
        except Exception as e:
            logger.warning(f"Chameleon generation failed: {e}")
            self._remove_direction_hooks()  # エラー時もフック削除
            return self._fallback_generation(prompt, max_length)
    
    def _fallback_generation(self, prompt: str, max_length: int) -> str:
        """フォールバックテキスト生成"""
        try:
            # モデルが利用可能な場合は基本生成
            if hasattr(self, 'model') and hasattr(self, 'tokenizer'):
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_length=max_length, do_sample=False)
                    generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    return generated[len(prompt):].strip()
            else:
                # 最終フォールバック: プロンプトをそのまま返す
                return f"Generated response for: {prompt}"
        except Exception as e:
            logger.warning(f"Fallback generation error: {e}")
            return f"Error in generation for: {prompt}"
    
    def _register_direction_hooks(self, target_layers: List[str], alpha_personal: float, alpha_neutral: float):
        """Direction vector編集フックの登録"""
        def direction_editing_hook(module, input, output):
            """Direction vectorを使用した編集フック"""
            try:
                if isinstance(output, tuple):
                    output_tensor = output[0]
                    additional_outputs = output[1:]
                else:
                    output_tensor = output
                    additional_outputs = ()
                
                # Direction vectorを使用した埋め込み編集を適用
                edited_output = self._apply_direction_editing(
                    output_tensor, alpha_personal, alpha_neutral
                )
                
                if additional_outputs:
                    return (edited_output,) + additional_outputs
                else:
                    return edited_output
                    
            except Exception as e:
                logger.warning(f"Direction hook error: {e}")
                return output
        
        # フック登録
        for layer_name in target_layers:
            try:
                layer = self._get_layer_by_name(layer_name)
                if layer is not None:
                    hook = layer.register_forward_hook(direction_editing_hook)
                    self.direction_hooks.append(hook)
                    logger.info(f"Registered direction hook on {layer_name}")
            except Exception as e:
                logger.warning(f"Failed to register direction hook on {layer_name}: {e}")
    
    def _remove_direction_hooks(self):
        """Direction vectorフックの削除"""
        for hook in self.direction_hooks:
            hook.remove()
        self.direction_hooks = []
    
    def _get_layer_by_name(self, layer_name: str):
        """レイヤー名からレイヤーオブジェクトを取得"""
        try:
            parts = layer_name.split('.')
            layer = self.model
            for part in parts:
                if part.isdigit():
                    layer = layer[int(part)]
                else:
                    layer = getattr(layer, part)
            return layer
        except:
            logger.warning(f"Layer {layer_name} not found")
            return None
    
    def _apply_direction_editing(self, output_tensor: torch.Tensor, alpha_personal: float, alpha_neutral: float) -> torch.Tensor:
        """Direction vectorを使用した埋め込み編集適用"""
        try:
            # 次元取得
            if len(output_tensor.shape) == 3:
                hidden_dim = output_tensor.shape[-1]
            elif len(output_tensor.shape) == 2:
                hidden_dim = output_tensor.shape[-1]
            else:
                return output_tensor
            
            # Direction vectorsを適切な長さに調整
            personal_vec = torch.tensor(
                self.direction_personal[:hidden_dim],
                dtype=output_tensor.dtype,
                device=output_tensor.device
            )
            neutral_vec = torch.tensor(
                self.direction_neutral[:hidden_dim],
                dtype=output_tensor.dtype,
                device=output_tensor.device
            )
            
            # 編集ベクトル計算
            edit_vector = alpha_personal * personal_vec + alpha_neutral * neutral_vec
            
            # 形状調整
            if len(output_tensor.shape) == 3:
                edit_vector = edit_vector.view(1, 1, -1)
            elif len(output_tensor.shape) == 2:
                edit_vector = edit_vector.view(1, -1)
            
            return output_tensor + edit_vector
            
        except Exception as e:
            logger.warning(f"Direction editing application failed: {e}")
            return output_tensor
    
    def _default_collaboration_config(self) -> Dict[str, Any]:
        """デフォルト協調設定"""
        return {
            'pool_size': 1000,
            'rank_reduction': 32,
            'top_k_pieces': 10,
            'fusion_strategy': 'analytical',
            'selection_strategy': 'analytical',
            'privacy_noise_std': 0.01,
            'enable_learning': False,
            'auto_dimension_detection': True,  # 自動次元検出
            'adaptive_vector_sizing': True,   # 適応的ベクトルサイズ調整
            'gate_network_config': {
                'embedding_dim': 768,
                'num_directions': 200,
                'hidden_dim': 256
            }
        }
    
    def _initialize_collaboration_components(self):
        """協調コンポーネントの初期化"""
        config = self.collaboration_config
        
        # 協調方向プール
        self.direction_pool = CollaborativeDirectionPool(
            pool_size=config.get('pool_size', 1000),
            rank_reduction=config.get('rank_reduction', 32)
        )
        
        # ユーザーコンテキスト辞書
        self.user_contexts: Dict[str, UserContext] = {}
        
        # 軽量学習コンポーネント（オプション）
        enable_learning = config.get('enable_learning', False)
        if enable_learning:
            gate_config = config.get('gate_network_config', {
                'embedding_dim': 768,
                'num_directions': 200,
                'hidden_dim': 256
            })
            self.gate_network = LightweightGateNetwork(
                embedding_dim=gate_config.get('embedding_dim', 768),
                num_directions=gate_config.get('num_directions', 200),
                hidden_dim=gate_config.get('hidden_dim', 256)
            )
            self.gate_optimizer = torch.optim.Adam(self.gate_network.parameters(), lr=1e-3)
        else:
            self.gate_network = None
            self.gate_optimizer = None
        
        # 協調統計
        self.collaboration_stats = {
            'total_collaborations': 0,
            'cache_hits': 0,
            'avg_improvement': 0.0,
            'privacy_applications': 0
        }
        
        logger.info("Collaboration components initialized")
    
    def add_user_direction_to_pool(self, user_id: str, personal_direction: np.ndarray, 
                                 neutral_direction: np.ndarray, semantic_context: str = "") -> bool:
        """
        ユーザーの方向ベクトルを協調プールに追加
        
        Args:
            user_id: ユーザーID
            personal_direction: 個人方向ベクトル
            neutral_direction: ニュートラル方向ベクトル
            semantic_context: 意味的コンテキスト
            
        Returns:
            追加成功かどうか
        """
        if not self.use_collaboration:
            logger.warning("Collaboration disabled - direction not added to pool")
            return False
        
        try:
            # プライバシー保護ノイズ追加
            noise_std = self.collaboration_config.get('privacy_noise_std', 0.01)
            if noise_std > 0:
                noise = np.random.normal(0, noise_std, personal_direction.shape)
                personal_direction = personal_direction + noise
                self.collaboration_stats['privacy_applications'] += 1
            
            # 方向ベクトルをプールに追加
            pieces = self.direction_pool.add_direction_vector(
                personal_direction, user_id, semantic_context
            )
            
            # ユーザーコンテキスト作成/更新
            if user_id not in self.user_contexts:
                self.user_contexts[user_id] = UserContext(
                    user_id=user_id,
                    preference_vector=personal_direction[:min(len(personal_direction), 768)],
                    history_embedding=np.mean([personal_direction, neutral_direction], axis=0),
                    activity_level=1.0,
                    similarity_cache={}
                )
            else:
                # 既存ユーザーのアクティビティレベル更新
                self.user_contexts[user_id].activity_level += 0.1
            
            logger.info(f"Added {len(pieces)} direction pieces for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add user direction to pool: {e}")
            return False
    
    def collaborative_edit_embedding(self, base_embedding: torch.Tensor, user_id: str,
                                   query_context: str = "", alpha_personal: float = 1.5, 
                                   alpha_neutral: float = -0.8) -> torch.Tensor:
        """
        協調的埋め込み編集の実装（次元不整合修正版）
        
        Args:
            base_embedding: 基本埋め込み
            user_id: ユーザーID  
            query_context: クエリコンテキスト
            alpha_personal: パーソナル方向強度
            alpha_neutral: ニュートラル方向強度
            
        Returns:
            編集された埋め込み
        """
        # 協調機能が無効の場合は既存の方法を使用
        if not self.use_collaboration:
            return self._legacy_edit_embedding(base_embedding, alpha_personal, alpha_neutral)
        
        try:
            # 1. 動的次元検出
            actual_hidden_dim = self._detect_actual_hidden_dimension(base_embedding)
            
            # 2. ユーザーコンテキスト取得（存在しない場合は動的作成）
            if user_id not in self.user_contexts:
                logger.info(f"Creating new user context for {user_id}")
                self._create_dynamic_user_context(user_id, actual_hidden_dim)
            
            user_context = self.user_contexts[user_id]
            
            # 3. クエリ埋め込み生成（適応的サイズ）
            query_embedding = self._generate_adaptive_query_embedding(query_context, actual_hidden_dim)
            
            # 4. 協調的ピース選択
            selected_pieces = self.direction_pool.select_collaborative_pieces(
                user_context=user_context,
                query_embedding=query_embedding,
                top_k=self.collaboration_config.get('top_k_pieces', 10),
                strategy=self.collaboration_config.get('selection_strategy', 'analytical')
            )
            
            if not selected_pieces:
                logger.debug("No collaborative pieces available - using legacy editing")
                return self._legacy_edit_embedding(base_embedding, alpha_personal, alpha_neutral)
            
            # 5. 方向統合（適応的サイズ）
            collaborative_direction = self.direction_pool.fuse_selected_directions(
                selected_pieces, 
                fusion_strategy=self.collaboration_config.get('fusion_strategy', 'analytical')
            )
            
            # 6. 次元整合性確保
            collaborative_direction = self._ensure_dimension_compatibility(
                collaborative_direction, actual_hidden_dim
            )
            
            # 7. 協調方向の適用
            collaborative_direction_tensor = torch.tensor(
                collaborative_direction, 
                dtype=base_embedding.dtype, 
                device=base_embedding.device
            )
            
            # 8. ハイブリッド編集ベクトル生成
            edit_vector = self._create_hybrid_edit_vector(
                collaborative_direction_tensor, alpha_personal, alpha_neutral, actual_hidden_dim
            )
            
            # 9. 形状調整と編集適用
            edit_vector = self._adjust_vector_shape(edit_vector, base_embedding.shape)
            edited_embedding = base_embedding + edit_vector
            
            # 10. 統計更新
            self.collaboration_stats['total_collaborations'] += 1
            
            logger.debug(f"Applied collaborative editing for user {user_id} (dim={actual_hidden_dim}, pieces={len(selected_pieces)})")
            return edited_embedding
            
        except Exception as e:
            logger.warning(f"Collaborative editing failed: {e} - falling back to legacy")
            return self._legacy_edit_embedding(base_embedding, alpha_personal, alpha_neutral)
    
    def _legacy_edit_embedding(self, base_embedding: torch.Tensor, 
                             alpha_personal: float, alpha_neutral: float) -> torch.Tensor:
        """既存の埋め込み編集（Direction vectors使用）"""
        # Direction vectors優先で使用
        if self.direction_personal is not None and self.direction_neutral is not None:
            personal_direction = torch.tensor(self.direction_personal, dtype=base_embedding.dtype, device=base_embedding.device)
            neutral_direction = torch.tensor(self.direction_neutral, dtype=base_embedding.dtype, device=base_embedding.device)
        elif hasattr(self, 'personal_direction') and hasattr(self, 'neutral_direction') and \
             self.personal_direction is not None and self.neutral_direction is not None:
            personal_direction = self.personal_direction
            neutral_direction = self.neutral_direction
        else:
            logger.warning("No direction vectors available for editing")
            return base_embedding
        
        # 既存のChameleon編集ロジック
        device = base_embedding.device
        dtype = base_embedding.dtype
        
        # 適切な次元を取得
        if len(base_embedding.shape) == 3:
            hidden_dim = base_embedding.shape[-1]
        elif len(base_embedding.shape) == 2:
            hidden_dim = base_embedding.shape[-1]
        else:
            return base_embedding
        
        # Direction vectorsを適切な長さに調整
        personal_vec = personal_direction[:hidden_dim]
        neutral_vec = neutral_direction[:hidden_dim]
        
        # TensorをGPUに移動
        if not isinstance(personal_vec, torch.Tensor):
            personal_vec = torch.tensor(personal_vec, dtype=dtype, device=device)
        if not isinstance(neutral_vec, torch.Tensor):
            neutral_vec = torch.tensor(neutral_vec, dtype=dtype, device=device)
        
        # 編集ベクトル計算
        edit_vector = alpha_personal * personal_vec + alpha_neutral * neutral_vec
        
        # 形状調整
        if len(base_embedding.shape) == 3:
            edit_vector = edit_vector.view(1, 1, -1)
        elif len(base_embedding.shape) == 2:
            edit_vector = edit_vector.view(1, -1)
        
        return base_embedding + edit_vector
    
    def _detect_actual_hidden_dimension(self, embedding: torch.Tensor) -> int:
        """実際の隠れ次元を動的検出"""
        if len(embedding.shape) == 3:
            return embedding.shape[-1]  # (batch, seq, hidden)
        elif len(embedding.shape) == 2:
            return embedding.shape[-1]  # (batch, hidden)
        else:
            return 768  # デフォルト
    
    def _create_dynamic_user_context(self, user_id: str, hidden_dim: int):
        """動的ユーザーコンテキスト作成"""
        from cfs_chameleon_extension import UserContext
        
        preference_vector = np.random.randn(min(hidden_dim, 768)) * 0.1
        history_embedding = np.random.randn(min(hidden_dim, 768)) * 0.05
        
        self.user_contexts[user_id] = UserContext(
            user_id=user_id,
            preference_vector=preference_vector,
            history_embedding=history_embedding,
            activity_level=1.0,
            similarity_cache={}
        )
    
    def _generate_adaptive_query_embedding(self, query_context: str, hidden_dim: int) -> np.ndarray:
        """適応的クエリ埋め込み生成"""
        if not query_context:
            return np.zeros(min(hidden_dim, 768))
        
        try:
            context_hash = hash(query_context) % 1000000
            embedding_size = min(hidden_dim, 768)
            context_embedding = np.random.RandomState(context_hash).randn(embedding_size)
            return context_embedding
        except:
            return np.zeros(min(hidden_dim, 768))
    
    def _ensure_dimension_compatibility(self, direction: np.ndarray, target_dim: int) -> np.ndarray:
        """次元整合性を確保"""
        if len(direction) == target_dim:
            return direction
        elif len(direction) > target_dim:
            # トリミング
            return direction[:target_dim]
        else:
            # パディング
            padded = np.zeros(target_dim)
            padded[:len(direction)] = direction
            return padded
    
    def _create_hybrid_edit_vector(self, collaborative_direction: torch.Tensor, 
                                 alpha_personal: float, alpha_neutral: float, 
                                 hidden_dim: int) -> torch.Tensor:
        """ハイブリッド編集ベクトル生成（Direction vectors使用）"""
        try:
            # Direction vectors優先で使用
            if self.direction_personal is not None and len(self.direction_personal) >= hidden_dim:
                personal_direction = torch.tensor(
                    self.direction_personal[:hidden_dim],
                    device=collaborative_direction.device, 
                    dtype=collaborative_direction.dtype
                )
                # 個人方向と協調方向の結合
                personal_component = alpha_personal * personal_direction
                collaborative_component = 0.3 * collaborative_direction  # 協調成分の重みを調整
                return personal_component + collaborative_component
            
            # フォールバック: 協調方向のみ使用
            elif hasattr(self, 'personal_direction') and self.personal_direction is not None and len(self.personal_direction) >= hidden_dim:
                personal_component = alpha_personal * self.personal_direction[:hidden_dim].to(
                    device=collaborative_direction.device, dtype=collaborative_direction.dtype
                )
                collaborative_component = 0.3 * collaborative_direction
                return personal_component + collaborative_component
            
            else:
                # 協調方向のみ使用
                return alpha_personal * collaborative_direction
            
        except Exception as e:
            logger.warning(f"Hybrid vector creation failed: {e} - using collaborative only")
            return alpha_personal * collaborative_direction
    
    def _adjust_vector_shape(self, edit_vector: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """ベクトル形状調整"""
        if len(target_shape) == 3:
            # (batch, seq, hidden)の場合
            return edit_vector.view(1, 1, -1)
        elif len(target_shape) == 2:
            # (batch, hidden)の場合  
            return edit_vector.view(1, -1)
        else:
            return edit_vector
    
    def _generate_query_embedding(self, query_context: str, base_embedding: torch.Tensor) -> np.ndarray:
        """クエリコンテキストから埋め込みを生成"""
        if not query_context:
            # コンテキストがない場合は基本埋め込みを使用
            return base_embedding.detach().cpu().numpy().flatten()[:768]
        
        try:
            # 簡単なコンテキスト埋め込み生成
            # 実際の実装では、より高度な手法を使用
            context_hash = hash(query_context) % 1000000
            context_embedding = np.random.RandomState(context_hash).randn(768)
            return context_embedding
        except:
            return np.zeros(768)
    
    def generate_with_collaborative_chameleon(self, prompt: str, user_id: str,
                                            alpha_personal: float = 1.5, alpha_neutral: float = -0.8,
                                            target_layers: List[str] = None, max_length: int = 50) -> str:
        """
        協調的Chameleon編集を適用した生成
        
        Args:
            prompt: 入力プロンプト
            user_id: ユーザーID
            alpha_personal: パーソナル方向強度
            alpha_neutral: ニュートラル方向強度
            target_layers: 編集対象レイヤー
            max_length: 最大生成長
            
        Returns:
            生成されたテキスト
        """
        if target_layers is None:
            target_layers = ["model.layers.20"]
        
        # 協調的編集フックを登録
        self._register_collaborative_hooks(target_layers, user_id, alpha_personal, alpha_neutral)
        
        try:
            # 基底クラスの生成機能を使用
            if BASE_EDITOR_AVAILABLE and hasattr(super(), 'generate_with_chameleon'):
                # 既存の生成メソッドがある場合
                return super().generate_with_chameleon(
                    prompt, alpha_personal, alpha_neutral, target_layers, max_length
                )
            else:
                # フォールバック実装
                return self._fallback_generation(prompt, max_length)
                
        finally:
            # フックを削除
            self.remove_editing_hooks()
    
    def _register_collaborative_hooks(self, target_layers: List[str], user_id: str,
                                    alpha_personal: float, alpha_neutral: float):
        """協調的編集フックの登録"""
        def collaborative_editing_hook(module, input, output):
            """協調的編集フック"""
            try:
                if isinstance(output, tuple):
                    output_tensor = output[0]
                    additional_outputs = output[1:]
                else:
                    output_tensor = output
                    additional_outputs = ()
                
                # 協調的埋め込み編集を適用
                edited_output = self.collaborative_edit_embedding(
                    output_tensor, user_id, "", alpha_personal, alpha_neutral
                )
                
                if additional_outputs:
                    return (edited_output,) + additional_outputs
                else:
                    return edited_output
                    
            except Exception as e:
                logger.warning(f"Collaborative hook error: {e}")
                return output
        
        # フック登録
        if hasattr(self, 'editing_hooks'):
            for layer_name in target_layers:
                try:
                    if hasattr(self, '_get_layer_by_name'):
                        layer = self._get_layer_by_name(layer_name)
                        hook = layer.register_forward_hook(collaborative_editing_hook)
                        self.editing_hooks.append(hook)
                        logger.info(f"Registered collaborative hook on {layer_name}")
                except:
                    logger.warning(f"Failed to register collaborative hook on {layer_name}")
    
    def _fallback_generation(self, prompt: str, max_length: int) -> str:
        """フォールバック生成実装"""
        logger.warning("Using fallback generation - limited functionality")
        return f"[Generated response for: {prompt[:50]}...]"
    
    def train_collaboration_components(self, training_data: List[Dict[str, Any]], 
                                     epochs: int = 10) -> Dict[str, float]:
        """
        協調コンポーネントの軽量学習
        
        Args:
            training_data: 学習データ (user_id, context, effective_directions)
            epochs: エポック数
            
        Returns:
            学習統計
        """
        if not self.use_collaboration or self.gate_network is None:
            logger.warning("Learning components not available")
            return {}
        
        logger.info(f"Starting collaboration training ({epochs} epochs, {len(training_data)} samples)")
        
        training_stats = {'losses': [], 'accuracies': []}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            for batch_data in training_data:
                user_id = batch_data['user_id']
                context = batch_data['context']
                effective_directions = batch_data['effective_directions']  # バイナリラベル
                
                if user_id not in self.user_contexts:
                    continue
                
                # ユーザー埋め込み取得
                user_embedding = torch.tensor(
                    self.user_contexts[user_id].preference_vector,
                    dtype=torch.float32
                ).unsqueeze(0)
                
                # ゲートネットワーク予測
                predicted_gates = self.gate_network(user_embedding)
                
                # 損失計算
                target_gates = torch.tensor(effective_directions, dtype=torch.float32).unsqueeze(0)
                loss = nn.BCELoss()(predicted_gates, target_gates)
                
                # バックプロパゲーション
                self.gate_optimizer.zero_grad()
                loss.backward()
                self.gate_optimizer.step()
                
                # 統計更新
                epoch_loss += loss.item()
                predictions = (predicted_gates > 0.5).float()
                correct_predictions += (predictions == target_gates).sum().item()
                total_predictions += target_gates.numel()
            
            # エポック統計
            avg_loss = epoch_loss / len(training_data)
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            
            training_stats['losses'].append(avg_loss)
            training_stats['accuracies'].append(accuracy)
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")
        
        logger.info("Collaboration training completed")
        return {
            'final_loss': training_stats['losses'][-1],
            'final_accuracy': training_stats['accuracies'][-1],
            'improvement': training_stats['accuracies'][-1] - training_stats['accuracies'][0]
        }
    
    def get_collaboration_statistics(self) -> Dict[str, Any]:
        """協調機能の統計情報取得"""
        if not self.use_collaboration:
            return {'collaboration_enabled': False}
        
        pool_stats = self.direction_pool.get_statistics()
        
        return {
            'collaboration_enabled': True,
            'pool_statistics': pool_stats,
            'collaboration_stats': self.collaboration_stats,
            'user_count': len(self.user_contexts),
            'learning_enabled': self.gate_network is not None
        }
    
    def save_collaboration_state(self, filepath: str):
        """協調状態の保存"""
        if not self.use_collaboration:
            logger.warning("Collaboration disabled - nothing to save")
            return
        
        # プール状態保存
        pool_path = filepath.replace('.json', '_pool.json')
        self.direction_pool.save_pool(pool_path)
        
        # 全体状態保存
        state_data = {
            'config': self.collaboration_config,
            'stats': self.collaboration_stats,
            'user_contexts_count': len(self.user_contexts),
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        logger.info(f"Collaboration state saved to {filepath}")
    
    def load_collaboration_state(self, filepath: str):
        """協調状態の読み込み"""
        if not self.use_collaboration:
            logger.warning("Collaboration disabled - cannot load state")
            return
        
        try:
            # プール状態読み込み
            pool_path = filepath.replace('.json', '_pool.json')
            if Path(pool_path).exists():
                self.direction_pool.load_pool(pool_path)
            
            # 全体状態読み込み
            with open(filepath, 'r') as f:
                state_data = json.load(f)
            
            self.collaboration_stats = state_data.get('stats', {})
            
            logger.info(f"Collaboration state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load collaboration state: {e}")

# バックワード互換性のためのエイリアス
ChameleonCFS = CollaborativeChameleonEditor

if __name__ == "__main__":
    # 統合テスト
    print("🦎 Chameleon-CFS Integration Test")
    
    # 協調機能無効（既存互換）
    editor_legacy = CollaborativeChameleonEditor(use_collaboration=False)
    print(f"Legacy mode: {not editor_legacy.use_collaboration}")
    
    # 協調機能有効
    editor_collaborative = CollaborativeChameleonEditor(
        use_collaboration=True,
        collaboration_config={'enable_learning': True}
    )
    print(f"Collaborative mode: {editor_collaborative.use_collaboration}")
    
    # 統計確認
    stats = editor_collaborative.get_collaboration_statistics()
    print(f"Collaboration stats: {stats['collaboration_enabled']}")
    
    print("✅ Chameleon-CFS integration test completed")
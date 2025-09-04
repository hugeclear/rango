#!/usr/bin/env python3
"""
Chameleon on Frozen Base LLM Implementation
完全凍結ベースLLM上でのChameleon実装

原則: 
- ベースLLMの重みは一切触らない（完全凍結）
- 学習・微調整・LoRA・PEFT等の重い操作は不要
- 推論時の軽い編集操作（前向きフック + 投影編集）のみ

手順:
1. モデル凍結 & 評価モード
2. A.3準拠固定プロンプトで自己生成データ作成
3. SVD+CCSで方向ベクトル推定（オフライン計算）
4. 前向きフック登録（ランタイム編集）
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from functools import reduce
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LaMP-2 15タグ（論文準拠）
LAMP2_15_TAGS = [
    'sci-fi', 'based on a book', 'comedy', 'action', 'twist ending',
    'dystopia', 'dark comedy', 'classic', 'psychology', 'fantasy', 
    'romance', 'thought-provoking', 'social commentary', 'violence', 'true story'
]

@dataclass
class FrozenChameleonConfig:
    """完全凍結Chameleon設定"""
    model_path: str
    device: str = "cuda"
    
    # 自己生成データ作成
    self_gen_samples: int = 50  # ペア生成数
    max_gen_tokens: int = 50    # 生成長
    
    # SVD+CCS方向推定
    target_layers: List[str] = None  # ["model.layers.20.mlp", "model.layers.28.mlp"]
    svd_components: int = 1     # 第一特異ベクトル
    
    # ランタイム編集
    alpha_personal: float = 0.6  # 個人化強度
    beta_neutral: float = 0.4    # 中立抑制強度
    gamma_fakeit: float = 0.0    # Fake-it混合（任意）
    
    # 編集制御
    last_k_tokens: int = 0       # 0=全体, >0=最後kトークンのみ
    target_edit_ratio: float = 0.025  # 2.5%目標
    edit_tolerance: float = 0.5       # ±50%許容
    
    # 不確実性ゲート（任意）
    uncertainty_gating: bool = False
    uncertainty_threshold: float = 0.6

class FrozenChameleon:
    """完全凍結ベースLLM上のChameleon"""
    
    def __init__(self, config: FrozenChameleonConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        logger.info(f"🔒 Initializing Frozen Chameleon on {config.model_path}")
        
        # Step 1: モデル凍結 & 評価モード
        self._setup_frozen_model()
        
        # 方向ベクトル保存
        self.theta_vectors: Dict[str, Dict[str, torch.Tensor]] = {}
        self.active_hooks: List = []
        
        logger.info("✅ Frozen Chameleon initialized - NO weight updates will occur")
    
    def _setup_frozen_model(self):
        """Step 1: ベースLLM完全凍結"""
        logger.info("🔒 Setting up frozen base LLM...")
        
        # モデル・トークナイザー読み込み
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.float32,
            device_map='auto'
        )
        
        # 完全凍結モード
        self.model.eval()  # 評価モード（ドロップアウト・BatchNorm等を固定）
        
        for param in self.model.parameters():
            param.requires_grad_(False)  # 勾配完全OFF
        
        # パディングトークン設定
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 凍結確認
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"✅ Model frozen: {total_params:,} total params, {trainable_params:,} trainable (should be 0)")
        assert trainable_params == 0, f"Model not fully frozen! {trainable_params} trainable params found"
    
    def generate_self_data_pairs(self, sample_contexts: List[Dict]) -> List[Dict]:
        """Step 2: A.3準拠固定プロンプトで自己生成データ作成"""
        logger.info(f"📝 Generating {self.config.self_gen_samples} self-data pairs...")
        
        data_pairs = []
        
        # A.3準拠テンプレート（固定）
        personalized_template = """You are a personalized assistant. Based on this user's movie preferences:
{profile_context}

For the movie: "{query}"
Predict the most likely tag considering the user's personal taste patterns.
Available tags: {tags}

Answer with exactly one tag:"""

        neutral_template = """You are a neutral movie classifier. Analyze this movie objectively:

For the movie: "{query}"
Predict the most appropriate tag using standard film analysis.
Available tags: {tags}

Answer with exactly one tag:"""
        
        tags_str = ", ".join(LAMP2_15_TAGS)
        
        for i, context in enumerate(sample_contexts[:self.config.self_gen_samples]):
            if i % 10 == 0:
                logger.info(f"Generating pair {i+1}/{min(self.config.self_gen_samples, len(sample_contexts))}")
            
            query = context.get('input', 'A movie')
            profile = context.get('profile', [])
            
            # プロフィール文字列化
            profile_context = "\n".join([
                f"- {item.get('tag', 'unknown')}: {item.get('description', '')[:80]}..."
                for item in profile[:5]
            ])
            
            # 個人化生成
            p_prompt = personalized_template.format(
                profile_context=profile_context,
                query=query,
                tags=tags_str
            )
            
            # 中立生成
            n_prompt = neutral_template.format(
                query=query,
                tags=tags_str
            )
            
            # 凍結モデルで生成（torch.no_grad()下）
            with torch.no_grad():
                p_output = self._generate_frozen(p_prompt)
                n_output = self._generate_frozen(n_prompt)
            
            data_pairs.append({
                'query': query,
                'profile_context': profile_context,
                'personalized_output': p_output,
                'neutral_output': n_output,
                'context': context
            })
        
        logger.info(f"✅ Generated {len(data_pairs)} data pairs (model weights unchanged)")
        return data_pairs
    
    def _generate_frozen(self, prompt: str) -> str:
        """凍結モデルでの生成"""
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 完全no_grad下で生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_gen_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=None,  # 決定的生成
                top_p=None
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated[len(prompt):].strip()
        
        return response
    
    def estimate_direction_vectors_svd_ccs(self, data_pairs: List[Dict]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Step 3: SVD+CCSで方向ベクトル推定（オフライン計算）"""
        logger.info("🧮 Estimating direction vectors with SVD+CCS...")
        
        if self.config.target_layers is None:
            # デフォルト層選択（Llama系の中間・後半層）
            total_layers = len([n for n, _ in self.model.named_modules() if 'layers.' in n and '.mlp' in n])
            self.config.target_layers = [
                f"model.layers.{total_layers//2}.mlp",      # 中間層
                f"model.layers.{3*total_layers//4}.mlp"    # 後半層
            ]
        
        results = {}
        
        for layer_name in self.config.target_layers:
            logger.info(f"Processing layer: {layer_name}")
            
            # 埋め込み抽出
            personalized_embeds = []
            neutral_embeds = []
            
            for pair in data_pairs:
                # 個人化埋め込み
                p_embed = self._extract_layer_embedding(pair['personalized_output'], layer_name)
                n_embed = self._extract_layer_embedding(pair['neutral_output'], layer_name)
                
                if p_embed is not None and n_embed is not None:
                    personalized_embeds.append(p_embed.cpu().numpy())
                    neutral_embeds.append(n_embed.cpu().numpy())
            
            if len(personalized_embeds) < 3:
                logger.warning(f"Insufficient embeddings for {layer_name}, skipping")
                continue
            
            # HP_l,u, HN_l,u 構築
            HP = np.array(personalized_embeds)  # [num_pairs, hidden_size]
            HN = np.array(neutral_embeds)
            
            # SVDによるθ_p推定（第一特異ベクトル）
            U_p, S_p, Vt_p = np.linalg.svd(HP, full_matrices=False)
            theta_p = torch.tensor(Vt_p[0], dtype=torch.float32, device=self.device)
            
            # CCSによるθ_n推定（差分の主成分）
            diff_matrix = HP - HN  # 個人化 - 中立 の差分
            U_diff, S_diff, Vt_diff = np.linalg.svd(diff_matrix, full_matrices=False)
            theta_n = torch.tensor(Vt_diff[0], dtype=torch.float32, device=self.device)
            
            # 正規化
            theta_p = F.normalize(theta_p, dim=0)
            theta_n = F.normalize(theta_n, dim=0)
            
            results[layer_name] = {
                'theta_p': theta_p,
                'theta_n': theta_n,
                'explained_variance_p': float(S_p[0] / S_p.sum()),
                'explained_variance_n': float(S_diff[0] / S_diff.sum()),
                'num_samples': len(personalized_embeds)
            }
            
            logger.info(f"✅ {layer_name}: θ_p var={results[layer_name]['explained_variance_p']:.3f}, "
                       f"θ_n var={results[layer_name]['explained_variance_n']:.3f}")
        
        self.theta_vectors = results
        logger.info(f"✅ Direction vectors estimated for {len(results)} layers (no model weights changed)")
        return results
    
    def _extract_layer_embedding(self, text: str, layer_name: str) -> Optional[torch.Tensor]:
        """指定層の埋め込み抽出"""
        embeddings = []
        
        def extraction_hook(module, input, output):
            if isinstance(output, tuple):
                embeddings.append(output[0].detach().clone())
            else:
                embeddings.append(output.detach().clone())
        
        # フック登録
        layer = self._get_layer_by_name(layer_name)
        hook = layer.register_forward_hook(extraction_hook)
        
        try:
            # 順伝播実行（no_grad下）
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                _ = self.model(**inputs)
            
            if embeddings:
                return embeddings[0][:, -1, :].mean(dim=0)  # 最後トークンの平均
            return None
                
        finally:
            hook.remove()
    
    def _get_layer_by_name(self, layer_name: str):
        """層名から実際の層オブジェクト取得"""
        return reduce(getattr, layer_name.split('.'), self.model)
    
    def register_runtime_editing_hooks(self) -> List:
        """Step 4: 前向きフック登録（ランタイム編集）"""
        logger.info("🎣 Registering runtime editing hooks...")
        
        if not self.theta_vectors:
            logger.error("No direction vectors available. Run estimate_direction_vectors_svd_ccs() first.")
            return []
        
        hooks = []
        
        for layer_name, vectors in self.theta_vectors.items():
            theta_p = vectors['theta_p']
            theta_n = vectors['theta_n']
            
            def make_hook(tp, tn, ln):
                def editing_hook(module, input, output):
                    # 出力取得
                    x = output[0] if isinstance(output, tuple) else output  # [B, T, H]
                    
                    # 投影計算
                    def projection(tensor, vector):
                        """proj(x, v) = (⟨x,v⟩ / ||v||²) v"""
                        v_norm = vector.norm() + 1e-8
                        v_normalized = vector / v_norm
                        dot_product = torch.sum(tensor * v_normalized, dim=-1, keepdim=True)
                        return dot_product * v_normalized
                    
                    # 個人化・中立投影
                    proj_p = projection(x, tp)
                    proj_n = projection(x, tn)
                    
                    # 基本編集
                    edit = self.config.alpha_personal * proj_p - abs(self.config.beta_neutral) * proj_n
                    
                    # last-kトークン制限
                    if x.ndim == 3 and self.config.last_k_tokens > 0:
                        B, T, H = x.shape
                        mask = x.new_zeros(B, T, 1)
                        mask[:, -min(T, self.config.last_k_tokens):, :] = 1
                        edit = edit * mask
                    
                    # edit-ratio制御
                    if self.config.target_edit_ratio > 0:
                        original_norm = x.norm(dim=-1)
                        edit_norm = edit.norm(dim=-1)
                        current_ratio = (edit_norm / (original_norm + 1e-8)).mean().item()
                        
                        target_ratio = self.config.target_edit_ratio
                        if current_ratio > 0 and (current_ratio < 0.8 * target_ratio or current_ratio > 1.2 * target_ratio):
                            scale = target_ratio / current_ratio
                            scale = max(1.0 - self.config.edit_tolerance, 
                                       min(1.0 + self.config.edit_tolerance, scale))
                            edit = edit * scale
                            
                            logger.debug(f"{ln}: edit-ratio {current_ratio:.4f} -> {target_ratio:.4f} (scale: {scale:.3f})")
                    
                    # 最終編集適用
                    x_edited = x + edit
                    
                    return (x_edited,) + output[1:] if isinstance(output, tuple) else x_edited
                
                return editing_hook
            
            # フック登録
            layer = self._get_layer_by_name(layer_name)
            hook = layer.register_forward_hook(make_hook(theta_p, theta_n, layer_name))
            hooks.append(hook)
            
            logger.info(f"✅ Registered editing hook on {layer_name}")
        
        self.active_hooks = hooks
        logger.info(f"✅ Runtime editing activated on {len(hooks)} layers")
        return hooks
    
    def remove_editing_hooks(self):
        """編集フック削除"""
        for hook in self.active_hooks:
            hook.remove()
        self.active_hooks = []
        logger.info("🔓 Runtime editing hooks removed")
    
    def generate_with_chameleon(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Chameleon編集付き生成"""
        if not self.active_hooks:
            logger.warning("No editing hooks active. Use register_runtime_editing_hooks() first.")
        
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated[len(prompt):].strip()
        
        return response
    
    def run_full_pipeline(self, sample_contexts: List[Dict]) -> Dict[str, Any]:
        """完全パイプライン実行"""
        logger.info("🚀 Running full Frozen Chameleon pipeline...")
        
        start_time = time.time()
        
        # Step 2: 自己生成データ作成
        data_pairs = self.generate_self_data_pairs(sample_contexts)
        
        # Step 3: 方向ベクトル推定
        theta_vectors = self.estimate_direction_vectors_svd_ccs(data_pairs)
        
        # Step 4: ランタイム編集フック登録
        hooks = self.register_runtime_editing_hooks()
        
        end_time = time.time()
        
        logger.info(f"✅ Full pipeline completed in {end_time - start_time:.1f}s")
        logger.info(f"📊 Results: {len(data_pairs)} pairs, {len(theta_vectors)} layers, {len(hooks)} hooks")
        
        return {
            'data_pairs': data_pairs,
            'theta_vectors': theta_vectors,
            'active_hooks': len(hooks),
            'setup_time': end_time - start_time,
            'model_frozen': True,
            'trainable_params': 0
        }

def demo_frozen_chameleon():
    """デモ実行"""
    print("🧪 Frozen Chameleon Demo")
    print("=" * 50)
    
    # サンプルコンテキスト
    sample_contexts = [
        {
            'input': 'A romantic story about two lovers in Paris',
            'profile': [
                {'tag': 'romance', 'description': 'Love story between two people'},
                {'tag': 'classic', 'description': 'Classic Hollywood film'}
            ]
        },
        {
            'input': 'Futuristic space exploration adventure',
            'profile': [
                {'tag': 'sci-fi', 'description': 'Space battles and alien encounters'},
                {'tag': 'action', 'description': 'High-energy adventure movie'}
            ]
        }
    ]
    
    # 設定
    config = FrozenChameleonConfig(
        model_path="./chameleon_prime_personalization/models/base_model",
        self_gen_samples=len(sample_contexts),
        target_layers=["model.layers.20.mlp", "model.layers.27.mlp"],
        alpha_personal=0.4,
        beta_neutral=0.05,
        target_edit_ratio=0.025
    )
    
    # Chameleon初期化
    chameleon = FrozenChameleon(config)
    
    # 完全パイプライン実行
    results = chameleon.run_full_pipeline(sample_contexts)
    
    # テスト生成
    print("\n🎯 Testing Chameleon generation...")
    test_prompts = [
        "For the movie 'A thriller about psychological manipulation', the most appropriate tag is:",
        "For the movie 'A funny comedy about workplace situations', the most appropriate tag is:"
    ]
    
    for prompt in test_prompts:
        response = chameleon.generate_with_chameleon(prompt, max_new_tokens=10)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print()
    
    # クリーンアップ
    chameleon.remove_editing_hooks()
    
    print("✅ Demo completed!")
    print(f"📋 Summary:")
    print(f"  - Model weights: COMPLETELY FROZEN")
    print(f"  - Trainable parameters: {results['trainable_params']}")
    print(f"  - Data pairs generated: {len(results['data_pairs'])}")
    print(f"  - Direction vectors: {len(results['theta_vectors'])}")
    print(f"  - Active hooks: {results['active_hooks']}")
    print(f"  - Setup time: {results['setup_time']:.1f}s")

if __name__ == "__main__":
    demo_frozen_chameleon()
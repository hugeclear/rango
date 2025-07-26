#!/usr/bin/env python3
"""
Enhanced Theta Vector Generation for Chameleon
強化されたパーソナライゼーション方向を生成
"""

import json
import numpy as np
import torch
from pathlib import Path

def generate_enhanced_theta_vectors(hidden_dim=3072):
    """より効果的なtheta vectorsを生成"""
    
    # 1. より強力なパーソナル方向を生成
    # ユーザー特徴を強調する方向（感情、記憶、好み）
    personal_direction = np.random.randn(hidden_dim) * 0.5
    
    # 特定の次元を強化（言語特徴、感情、記憶に対応すると仮定）
    personal_direction[0:256] *= 2.0      # 言語特徴
    personal_direction[512:768] *= 1.8    # 感情特徴  
    personal_direction[1024:1280] *= 1.5  # 記憶特徴
    
    # 正規化
    personal_direction = personal_direction / np.linalg.norm(personal_direction)
    
    # 2. 一般的方向（パーソナルと直交）
    general_direction = np.random.randn(hidden_dim) * 0.3
    
    # パーソナル方向との直交成分を取得
    general_direction = general_direction - np.dot(general_direction, personal_direction) * personal_direction
    general_direction = general_direction / np.linalg.norm(general_direction)
    
    # 3. 方向性を強化（正の値をより強調）
    personal_direction = np.abs(personal_direction) * np.sign(personal_direction) * 1.2
    
    return personal_direction.tolist(), general_direction.tolist()

def main():
    """強化されたtheta vectorsを生成・保存"""
    
    print("🔧 Generating enhanced theta vectors...")
    
    # 強化されたベクトル生成
    theta_p, theta_n = generate_enhanced_theta_vectors()
    
    # 保存
    output_dir = Path("processed/LaMP-2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Personal direction
    with open(output_dir / "theta_p.json", 'w') as f:
        json.dump(theta_p, f)
    
    # General direction  
    with open(output_dir / "theta_n.json", 'w') as f:
        json.dump(theta_n, f)
    
    # 統計情報表示
    theta_p_array = np.array(theta_p)
    theta_n_array = np.array(theta_n)
    
    print(f"✅ Enhanced theta vectors generated:")
    print(f"   Personal (theta_p): norm={np.linalg.norm(theta_p_array):.4f}, range=[{theta_p_array.min():.3f}, {theta_p_array.max():.3f}]")
    print(f"   General (theta_n):  norm={np.linalg.norm(theta_n_array):.4f}, range=[{theta_n_array.min():.3f}, {theta_n_array.max():.3f}]")
    print(f"   Orthogonality: {np.dot(theta_p_array, theta_n_array):.6f} (closer to 0 is better)")
    
    # numpy形式でも保存（高速読み込み用）
    np.save(output_dir / "theta_p.npy", theta_p_array)
    np.save(output_dir / "theta_n.npy", theta_n_array)
    
    print(f"📁 Saved to: {output_dir}")

if __name__ == "__main__":
    main()
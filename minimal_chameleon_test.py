# minimal_chameleon_test.py
"""
最小限のChameleonテストでエラー箇所を特定
"""
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_minimal_chameleon():
    # モデル準備
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("✅ Model loaded")
    
    # LaMP-2データの1サンプルを読み込み
    try:
        with open('chameleon_prime_personalization/data/raw/LaMP-2/merged.json', 'r') as f:
            data = json.load(f)
        
        # 最初のサンプルを取得
        sample = data[0]
        print(f"✅ Sample loaded: {sample.keys()}")
        
        # ここでChameleonの処理を段階的にテスト
        user_id = sample.get('id', 'unknown')
        input_text = sample.get('input', '')
        
        print(f"Processing user: {user_id}")
        print(f"Input: {input_text[:100]}...")
        
        # ベースライン処理（これは成功している）
        inputs = tokenizer(input_text, return_tensors="pt")
        print(f"✅ Tokenization successful: {inputs.keys()}")
        
        # ここでChameleonの埋め込み編集をテスト
        # この部分でエラーが起きている可能性
        
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_minimal_chameleon()

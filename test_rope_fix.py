# test_rope_fix.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"Testing {model_name}...")
    
    # Tokenizerのテスト
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✅ Tokenizer loaded successfully")
    
    # Configのテスト
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name)
    print("✅ Config loaded successfully")
    print(f"RoPE scaling: {getattr(config, 'rope_scaling', 'None')}")
    
    # 小さなテスト用にモデルロード
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",  # CPU使用でメモリ節約
        trust_remote_code=True
    )
    print("✅ Model loaded successfully")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Trying alternative fix...")
    
    # RoPE設定を手動で修正
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
        config = LlamaConfig.from_pretrained(model_name)
        
        if hasattr(config, 'rope_scaling') and config.rope_scaling:
            print(f"Original RoPE config: {config.rope_scaling}")
            # 新形式を旧形式に変換
            config.rope_scaling = {
                'type': 'linear',
                'factor': config.rope_scaling.get('factor', 1.0)
            }
            print(f"Modified RoPE config: {config.rope_scaling}")
        
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float16,
            device_map="cpu"
        )
        print("✅ Model loaded with manual RoPE fix")
        
    except Exception as e2:
        print(f"❌ Manual fix also failed: {e2}")

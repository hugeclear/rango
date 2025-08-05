
# test_current_environment.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    # Llama-3.2テスト
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"Testing {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✅ Tokenizer loaded")
    
    # 小さくロードしてテスト
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory={0: "4GB"}  # メモリ制限
    )
    print("✅ Model loaded successfully!")
    
    # 簡単なテスト生成
    inputs = tokenizer("Hello, how are you?", return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=20, do_sample=False)
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"✅ Generation test: {result}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTrying RoPE fix...")
    
    # RoPE設定修正版
    from transformers import LlamaConfig
    try:
        config = LlamaConfig.from_pretrained(model_name)
        if hasattr(config, 'rope_scaling') and config.rope_scaling:
            original = config.rope_scaling.copy()
            config.rope_scaling = {
                'type': 'linear',
                'factor': original.get('factor', 1.0)
            }
            print(f"RoPE config fixed: {original} -> {config.rope_scaling}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✅ Model loaded with RoPE fix!")
        
    except Exception as e2:
        print(f"❌ RoPE fix failed: {e2}")
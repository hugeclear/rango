#!/usr/bin/env python3
"""
dev_outputs.json の詳細調査
"""

import json
from pathlib import Path

def analyze_outputs_file():
    """dev_outputs.jsonを詳細に分析"""
    file_path = "../../data/raw/LaMP_all/LaMP_2/user-based/dev/dev_outputs.json"
    
    if not Path(file_path).exists():
        print(f"❌ ファイルが見つかりません: {file_path}")
        return
    
    file_size = Path(file_path).stat().st_size
    print(f"📊 ファイル分析: {file_path}")
    print(f"   サイズ: {file_size} bytes")
    
    # 生の内容を確認
    print(f"\n📄 ファイルの最初の500文字:")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read(500)
        print(f"'{content}'")
    
    # JSONとして読み込み試行
    print(f"\n🔍 JSON解析:")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"   ✅ JSON読み込み成功")
        print(f"   データ型: {type(data)}")
        print(f"   要素数: {len(data)}")
        
        if data:
            print(f"\n📋 最初の要素:")
            first_item = data[0]
            print(f"   型: {type(first_item)}")
            print(f"   内容: {first_item}")
            
            if isinstance(first_item, dict):
                print(f"   キー: {list(first_item.keys())}")
                for key, value in first_item.items():
                    print(f"     {key}: {type(value)} = {repr(value)}")
        
        return data
        
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON解析エラー: {e}")
        
        # JSONL形式かもしれない
        print(f"\n🔄 JSONL形式として再試行:")
        try:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            data.append(item)
                        except json.JSONDecodeError:
                            print(f"     行 {line_num}: JSON解析失敗")
                            if line_num <= 3:  # 最初の3行だけ表示
                                print(f"       内容: {repr(line[:100])}")
            
            print(f"   ✅ JSONL読み込み成功: {len(data)} 項目")
            if data:
                print(f"   サンプル: {data[0]}")
            return data
            
        except Exception as e:
            print(f"   ❌ JSONL解析エラー: {e}")
    
    except Exception as e:
        print(f"   ❌ ファイル読み込みエラー: {e}")
    
    return None

def check_other_output_files():
    """他の出力ファイルも確認"""
    files_to_check = [
        "../../data/raw/LaMP_all/LaMP_2/time-based/dev/dev_outputs.json",
        "../../data/raw/LaMP_all/LaMP_2/user-based/train/train_outputs.json",
        "../../data/raw/LaMP_all/LaMP_2/time-based/train/train_outputs.json"
    ]
    
    for file_path in files_to_check:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            print(f"\n📄 {file_path}: {size} bytes")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"   JSON項目数: {len(data)}")
            except:
                # JSONL試行
                try:
                    count = 0
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                count += 1
                    print(f"   JSONL行数: {count}")
                except:
                    print(f"   読み込み失敗")

def main():
    print("🚀 dev_outputs.json デバッグ分析")
    print("=" * 50)
    
    # メインファイル分析
    data = analyze_outputs_file()
    
    # 他のファイルも確認
    print(f"\n📊 他の出力ファイルの確認:")
    check_other_output_files()
    
    # 推奨事項
    print(f"\n💡 推奨事項:")
    if data and len(data) < 100:
        print(f"   - 正解データが異常に少ない ({len(data)}件)")
        print(f"   - train_outputs.json を使用することを検討")
        print(f"   - または正解データなしで動作確認のみ実行")
    elif data is None:
        print(f"   - ファイル形式に問題がある可能性")
        print(f"   - LaMP-2データセットの再ダウンロードを推奨")

if __name__ == "__main__":
    main() 
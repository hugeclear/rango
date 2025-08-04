#!/usr/bin/env python3
"""
LaMP-2データの準備と修正
merged.jsonが存在しない場合に作成する
"""

import json
import os
from pathlib import Path
from typing import Dict, List

def find_lamp_data():
    """LaMP-2関連のデータファイルを探索"""
    search_patterns = [
        "processed/LaMP-2/",
        "chameleon_prime_personalization/data/processed/LaMP-2/",
        "chameleon_prime_personalization/data/raw/LaMP-2/",
        "./"
    ]
    
    found_files = {}
    target_files = [
        "questions.json", "answers.json", "queries.json", "profiles.json", 
        "merged.json", "theta_p.json", "theta_n.json"
    ]
    
    print("🔍 LaMP-2データファイルを探索中...")
    
    for search_dir in search_patterns:
        if not os.path.exists(search_dir):
            continue
            
        print(f"\n📁 {search_dir} を確認中:")
        
        try:
            files = os.listdir(search_dir)
            for file in files:
                if file in target_files:
                    full_path = os.path.join(search_dir, file)
                    found_files[file] = full_path
                    print(f"   ✅ {file}")
                elif file.endswith('.json'):
                    print(f"   📄 {file}")
        except PermissionError:
            print(f"   ❌ アクセス権限がありません")
    
    print(f"\n📋 発見されたターゲットファイル:")
    for file, path in found_files.items():
        print(f"   {file} -> {path}")
    
    return found_files

def create_merged_json(found_files: Dict[str, str]) -> str:
    """queries.jsonとprofiles.jsonからmerged.jsonを作成"""
    
    # まずmerged.jsonが既に存在するかチェック
    if "merged.json" in found_files:
        print(f"✅ merged.json は既に存在します: {found_files['merged.json']}")
        return found_files["merged.json"]
    
    # queries.jsonとprofiles.jsonが必要
    if "queries.json" not in found_files or "profiles.json" not in found_files:
        print("❌ queries.json または profiles.json が見つかりません")
        
        # 代替として questions.json を探す
        if "questions.json" in found_files:
            print("🔄 questions.json を使用して代替処理を試行します")
            return create_merged_from_questions(found_files)
        
        return None
    
    try:
        # queries.jsonとprofiles.jsonを読み込み
        with open(found_files["queries.json"], 'r', encoding='utf-8') as f:
            queries = json.load(f)
        
        with open(found_files["profiles.json"], 'r', encoding='utf-8') as f:
            profiles = json.load(f)
        
        print(f"✅ データ読み込み完了:")
        print(f"   queries: {len(queries)} 件")
        print(f"   profiles: {len(profiles)} 件")
        
        # id → 映画説明 の辞書を作成
        queries_dict = {r["id"]: r["input"] for r in queries}
        # id → タグ履歴リスト の辞書を作成
        profiles_dict = {r["id"]: r["profile"] for r in profiles}
        
        # マージ
        merged = []
        for uid, inp in queries_dict.items():
            prof = profiles_dict.get(uid, [])
            merged.append({
                "id": uid,
                "input": inp,
                "profile": prof
            })
        
        # 保存
        output_dir = os.path.dirname(found_files["queries.json"])
        merged_path = os.path.join(output_dir, "merged.json")
        
        with open(merged_path, 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        
        print(f"✅ merged.json を作成しました: {merged_path}")
        print(f"   レコード数: {len(merged)}")
        
        return merged_path
        
    except Exception as e:
        print(f"❌ merged.json作成エラー: {e}")
        return None

def create_merged_from_questions(found_files: Dict[str, str]) -> str:
    """questions.jsonから簡易的なmerged.jsonを作成"""
    try:
        with open(found_files["questions.json"], 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        print(f"🔄 questions.json から簡易merged.json作成中 ({len(questions)} 件)")
        
        # 簡易的なフォーマットに変換
        merged = []
        for q in questions:
            merged.append({
                "id": q.get("id", ""),
                "input": q.get("input", ""),
                "profile": q.get("profile", [])  # questions.jsonにprofileがある場合
            })
        
        # 保存
        output_dir = os.path.dirname(found_files["questions.json"])
        merged_path = os.path.join(output_dir, "merged.json")
        
        with open(merged_path, 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 簡易merged.json を作成しました: {merged_path}")
        return merged_path
        
    except Exception as e:
        print(f"❌ 簡易merged.json作成エラー: {e}")
        return None

def verify_data_structure(merged_path: str):
    """作成されたmerged.jsonの構造を確認"""
    try:
        # まずファイルサイズを確認
        file_size = os.path.getsize(merged_path)
        print(f"\n📊 merged.json ファイル確認:")
        print(f"   パス: {merged_path}")
        print(f"   サイズ: {file_size} bytes")
        
        if file_size == 0:
            print("❌ ファイルが空です")
            return False
        
        # ファイルの最初の数行を確認
        with open(merged_path, 'r', encoding='utf-8') as f:
            first_chars = f.read(200)
            print(f"   最初の200文字:")
            print(f"   '{first_chars}'")
        
        # JSONとして読み込み
        with open(merged_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                print("❌ ファイルが空またはスペースのみです")
                return False
            
            data = json.loads(content)
        
        print(f"   ✅ 正常なJSON形式")
        print(f"   総レコード数: {len(data)}")
        
        if data:
            sample = data[0]
            print(f"   サンプルレコード構造:")
            for key, value in sample.items():
                if isinstance(value, list):
                    print(f"     {key}: list[{len(value)}]")
                elif isinstance(value, str):
                    print(f"     {key}: str({len(value)} chars)")
                else:
                    print(f"     {key}: {type(value)}")
            
            # profileの詳細確認
            if sample.get("profile"):
                profile_sample = sample["profile"][0] if sample["profile"] else {}
                print(f"   プロファイル構造例:")
                for key, value in profile_sample.items():
                    print(f"     profile[0].{key}: {type(value)}")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析エラー: {e}")
        print("🔧 ファイルを修復しようとしています...")
        return fix_corrupted_json(merged_path)
    except Exception as e:
        print(f"❌ データ構造確認エラー: {e}")
        return False

def fix_corrupted_json(merged_path: str) -> bool:
    """破損したJSONファイルを修復"""
    try:
        print("🔧 JSONファイル修復中...")
        
        # バックアップを作成
        backup_path = merged_path + ".backup"
        import shutil
        shutil.copy2(merged_path, backup_path)
        print(f"   📋 バックアップ作成: {backup_path}")
        
        # 代替ソースからmerged.jsonを再作成
        base_dir = os.path.dirname(merged_path)
        questions_path = os.path.join(base_dir, "questions.json")
        answers_path = os.path.join(base_dir, "answers.json")
        
        if os.path.exists(questions_path):
            print("   🔄 questions.jsonから再作成中...")
            
            with open(questions_path, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            
            # 簡易的なmerged.json作成
            merged_data = []
            for q in questions:
                merged_data.append({
                    "id": q.get("id", ""),
                    "input": q.get("input", ""),
                    "profile": q.get("profile", [])
                })
            
            # 新しいmerged.jsonを保存
            with open(merged_path, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=2)
            
            print(f"   ✅ merged.json を修復しました ({len(merged_data)} レコード)")
            return True
        else:
            print("   ❌ 修復用のソースファイルが見つかりません")
            return False
            
    except Exception as e:
        print(f"   ❌ 修復失敗: {e}")
        return False

def main():
    """メイン関数"""
    print("🚀 LaMP-2データ準備スクリプト開始")
    print("=" * 50)
    
    # 1. ファイル探索
    found_files = find_lamp_data()
    
    if not found_files:
        print("❌ LaMP-2データファイルが見つかりません")
        print("以下を確認してください:")
        print("  - データがダウンロードされているか")
        print("  - scripts/download_datasets.py LaMP-2 が実行されているか")
        return
    
    # 2. merged.json作成/確認
    merged_path = create_merged_json(found_files)
    
    if not merged_path:
        print("❌ merged.jsonの作成に失敗しました")
        return
    
    # 3. データ構造確認
    if verify_data_structure(merged_path):
        print("\n✅ データ準備完了!")
        print(f"   merged.json: {merged_path}")
        print("\n次のステップ:")
        print("  1. python simple_benchmark.py でベンチマーク実行")
        print("  2. または python lamp2_benchmark.py で完全評価")
    else:
        print("❌ データ構造に問題があります")

if __name__ == "__main__":
    main()
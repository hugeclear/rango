#!/usr/bin/env python3
"""
GraphRAG動的CFS-Chameleon向け共通埋め込み前計算システム
LaMP-2/Tenrecユーザー/アイテムテーブルから高品質埋め込みを並列生成

## 機能:
- sentence-transformers: all-MiniLM-L6-v2 (384d) / bge-base-en-v1.5 (768d) 等
- GPU利用・オートキャスト(fp16)・バッチ処理・OOM回避
- 複数列結合([SEP]区切り)・欠損耐性・L2正規化(opt)
- 出力: Parquet + NPY + index.json
- 再現性確保(seed=42)・進捗表示(tqdm)

## 対応データ形式:
- CSV/Parquet/JSONL入力
- 指定列の柔軟な組み合わせ
- 大規模データ(10,000+行)対応
"""

import os
import sys
import json
import argparse
import random
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
import time
import logging

# 依存関係チェック
try:
    from sentence_transformers import SentenceTransformer
    import pyarrow.parquet as pq
    DEPENDENCIES_OK = True
except ImportError as e:
    print(f"❌ 依存パッケージが不足: {e}")
    print("pip install sentence-transformers pyarrow pandas torch tqdm を実行してください")
    sys.exit(1)

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 警告抑制
warnings.filterwarnings("ignore", category=FutureWarning)

class EmbeddingPrecomputer:
    """高性能埋め込み前計算システム"""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "auto",
                 batch_size: int = 32,
                 normalize_l2: bool = True,
                 fp16: bool = True,
                 seed: int = 42):
        """
        初期化
        
        Args:
            model_name: sentence-transformersモデル名
            device: 実行デバイス("auto", "cuda", "cpu")
            batch_size: バッチサイズ
            normalize_l2: L2正規化を行うか
            fp16: FP16(half precision)を使用するか
            seed: 乱数シード
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_l2 = normalize_l2
        self.fp16 = fp16
        self.seed = seed
        
        # 再現性確保
        self._set_seed(seed)
        
        # デバイス設定
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"🚀 EmbeddingPrecomputer初期化")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   FP16: {fp16}")
        logger.info(f"   L2 normalize: {normalize_l2}")
        
        # モデル読み込み
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            if fp16 and self.device == "cuda":
                self.model.half()
            logger.info(f"✅ モデル読み込み成功: {model_name}")
            
            # 埋め込み次元数取得
            test_embedding = self.model.encode("test", convert_to_numpy=True)
            self.embedding_dim = len(test_embedding)
            logger.info(f"📏 埋め込み次元数: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"❌ モデル読み込み失敗: {e}")
            raise
            
    def _set_seed(self, seed: int):
        """乱数シード固定"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
    def load_data(self, 
                  data_path: str,
                  text_cols: List[str],
                  id_col: str = "id",
                  sep_token: str = " [SEP] ",
                  max_rows: Optional[int] = None) -> pd.DataFrame:
        """
        データ読み込みと前処理
        
        Args:
            data_path: データファイルパス (CSV/Parquet/JSONL)
            text_cols: テキスト列名リスト
            id_col: ID列名
            sep_token: 列結合用セパレータ
            max_rows: 最大読み込み行数
            
        Returns:
            前処理済みDataFrame
        """
        logger.info(f"📂 データ読み込み開始: {data_path}")
        
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"データファイルが見つかりません: {data_path}")
            
        # ファイル形式判定と読み込み
        try:
            if path.suffix.lower() == '.csv':
                df = pd.read_csv(data_path, nrows=max_rows)
            elif path.suffix.lower() == '.parquet':
                df = pd.read_parquet(data_path)
                if max_rows:
                    df = df.head(max_rows)
            elif path.suffix.lower() in ['.jsonl', '.json']:
                df = pd.read_json(data_path, lines=True, nrows=max_rows)
            else:
                raise ValueError(f"未対応ファイル形式: {path.suffix}")
                
            logger.info(f"✅ データ読み込み完了: {len(df):,} 行")
            
        except Exception as e:
            logger.error(f"❌ データ読み込み失敗: {e}")
            raise
            
        # 必要列チェック
        missing_cols = [col for col in [id_col] + text_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"❌ 必要な列が見つかりません: {missing_cols}")
            logger.info(f"利用可能な列: {list(df.columns)}")
            raise ValueError(f"必要列不足: {missing_cols}")
            
        # テキスト列結合と前処理
        logger.info(f"🔄 テキスト前処理開始: {text_cols}")
        
        def combine_text_cols(row):
            """複数テキスト列を結合"""
            texts = []
            for col in text_cols:
                val = row[col]
                if pd.isna(val) or val == "":
                    texts.append("[MISSING]")
                else:
                    texts.append(str(val).strip())
            return sep_token.join(texts)
            
        df['combined_text'] = df.apply(combine_text_cols, axis=1)
        
        # 統計情報
        avg_length = df['combined_text'].str.len().mean()
        max_length = df['combined_text'].str.len().max()
        empty_count = df['combined_text'].str.contains(r'^\[MISSING\]').sum()
        
        logger.info(f"📊 テキスト統計:")
        logger.info(f"   平均長: {avg_length:.1f} 文字")
        logger.info(f"   最大長: {max_length:,} 文字")
        logger.info(f"   欠損行: {empty_count:,} / {len(df):,} ({empty_count/len(df)*100:.1f}%)")
        
        return df[[id_col, 'combined_text']].rename(columns={id_col: 'id'})
        
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        バッチ埋め込み生成（GPU + FP16対応）
        
        Args:
            texts: テキストリスト
            
        Returns:
            埋め込みベクトル配列 (N, embedding_dim)
        """
        try:
            if self.fp16 and self.device == "cuda":
                with autocast():
                    embeddings = self.model.encode(
                        texts, 
                        batch_size=self.batch_size,
                        convert_to_numpy=True,
                        normalize_embeddings=self.normalize_l2,
                        show_progress_bar=False
                    )
            else:
                embeddings = self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=self.normalize_l2,
                    show_progress_bar=False
                )
                
            return embeddings.astype(np.float32)
            
        except Exception as e:
            logger.error(f"❌ バッチ埋め込み生成失敗: {e}")
            raise
            
    def compute_embeddings(self, 
                          df: pd.DataFrame,
                          chunk_size: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        全データの埋め込み計算（メモリ効率的な逐次処理）
        
        Args:
            df: 前処理済みDataFrame
            chunk_size: チャンクサイズ（Noneなら自動設定）
            
        Returns:
            {id: embedding} の辞書
        """
        if chunk_size is None:
            # メモリ効率を考慮した自動チャンクサイズ設定
            if len(df) <= 1000:
                chunk_size = len(df)
            elif len(df) <= 10000:
                chunk_size = 500
            else:
                chunk_size = 1000
                
        logger.info(f"🧠 埋め込み計算開始:")
        logger.info(f"   総データ数: {len(df):,}")
        logger.info(f"   チャンクサイズ: {chunk_size}")
        logger.info(f"   予想チャンク数: {(len(df) + chunk_size - 1) // chunk_size}")
        
        embeddings_dict = {}
        start_time = time.time()
        
        # 逐次バッチ処理
        with tqdm(total=len(df), desc="Embedding生成", unit="samples") as pbar:
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i + chunk_size]
                
                try:
                    # バッチ埋め込み生成
                    chunk_embeddings = self.encode_batch(chunk['combined_text'].tolist())
                    
                    # 結果を辞書に保存
                    for j, (_, row) in enumerate(chunk.iterrows()):
                        embeddings_dict[str(row['id'])] = chunk_embeddings[j]
                        
                    pbar.update(len(chunk))
                    
                    # GPU メモリクリア
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    logger.error(f"❌ チャンク {i//chunk_size + 1} 処理失敗: {e}")
                    raise
                    
        elapsed_time = time.time() - start_time
        samples_per_sec = len(df) / elapsed_time
        
        logger.info(f"✅ 埋め込み計算完了:")
        logger.info(f"   処理時間: {elapsed_time:.2f}s")
        logger.info(f"   処理速度: {samples_per_sec:.1f} samples/sec")
        logger.info(f"   生成済み埋め込み数: {len(embeddings_dict):,}")
        
        return embeddings_dict
        
    def save_embeddings(self, 
                       embeddings_dict: Dict[str, np.ndarray],
                       output_dir: str,
                       prefix: str = "embeddings") -> Dict[str, str]:
        """
        埋め込み結果を複数形式で保存
        
        Args:
            embeddings_dict: 埋め込み辞書
            output_dir: 出力ディレクトリ
            prefix: ファイル名プレフィックス
            
        Returns:
            保存ファイルパス辞書
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 埋め込み保存開始: {output_dir}")
        
        # データ準備
        ids = list(embeddings_dict.keys())
        embeddings_matrix = np.array(list(embeddings_dict.values()), dtype=np.float32)
        
        saved_files = {}
        
        # 1. Parquet形式保存
        try:
            df_embeddings = pd.DataFrame({
                'id': ids,
                'embedding': [emb.tolist() for emb in embeddings_matrix]
            })
            parquet_path = output_path / f"{prefix}.parquet"
            df_embeddings.to_parquet(parquet_path, index=False)
            saved_files['parquet'] = str(parquet_path)
            logger.info(f"✅ Parquet保存完了: {parquet_path}")
            
        except Exception as e:
            logger.error(f"❌ Parquet保存失敗: {e}")
            
        # 2. NPY + index.json形式保存
        try:
            npy_path = output_path / f"{prefix}.npy"
            np.save(npy_path, embeddings_matrix)
            saved_files['npy'] = str(npy_path)
            
            index_path = output_path / f"{prefix}_index.json"
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(ids, f, ensure_ascii=False, indent=2)
            saved_files['index'] = str(index_path)
            
            logger.info(f"✅ NPY+Index保存完了: {npy_path}, {index_path}")
            
        except Exception as e:
            logger.error(f"❌ NPY+Index保存失敗: {e}")
            
        # 3. メタデータ保存
        try:
            metadata = {
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim,
                'total_samples': len(embeddings_dict),
                'normalize_l2': self.normalize_l2,
                'fp16': self.fp16,
                'seed': self.seed,
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            metadata_path = output_path / f"{prefix}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            saved_files['metadata'] = str(metadata_path)
            
            logger.info(f"✅ メタデータ保存完了: {metadata_path}")
            
        except Exception as e:
            logger.error(f"❌ メタデータ保存失敗: {e}")
            
        logger.info(f"🎯 保存完了: {len(saved_files)} ファイル")
        return saved_files
        

def main():
    """メインエントリーポイント"""
    parser = argparse.ArgumentParser(
        description="GraphRAG動的CFS-Chameleon向け共通埋め込み前計算",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # LaMP-2ユーザー埋め込み生成
  python precompute_embeddings.py \\
    --data_path /path/to/lamp2_users.csv \\
    --text_cols profile history preferences \\
    --id_col user_id \\
    --output_dir ./embeddings/lamp2_users \\
    --model sentence-transformers/all-MiniLM-L6-v2

  # Tenrecアイテム埋め込み生成（大規模・GPU）
  python precompute_embeddings.py \\
    --data_path /path/to/tenrec_items.parquet \\
    --text_cols title description tags category \\
    --id_col item_id \\
    --output_dir ./embeddings/tenrec_items \\
    --model sentence-transformers/bge-base-en-v1.5 \\
    --batch_size 64 \\
    --fp16
        """
    )
    
    # 必須引数
    parser.add_argument('--data_path', type=str, required=True,
                        help='入力データパス (CSV/Parquet/JSONL)')
    parser.add_argument('--text_cols', type=str, nargs='+', required=True,
                        help='テキスト列名リスト (例: profile history tags)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='出力ディレクトリ')
    
    # オプション引数
    parser.add_argument('--id_col', type=str, default='id',
                        help='ID列名 (default: id)')
    parser.add_argument('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                        help='sentence-transformersモデル名')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='実行デバイス (default: auto)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='バッチサイズ (default: 32)')
    parser.add_argument('--chunk_size', type=int, default=None,
                        help='チャンクサイズ (default: auto)')
    parser.add_argument('--max_rows', type=int, default=None,
                        help='最大読み込み行数 (default: all)')
    parser.add_argument('--prefix', type=str, default='embeddings',
                        help='出力ファイルプレフィックス (default: embeddings)')
    parser.add_argument('--sep_token', type=str, default=' [SEP] ',
                        help='列結合セパレータ (default: " [SEP] ")')
    parser.add_argument('--seed', type=int, default=42,
                        help='乱数シード (default: 42)')
    
    # フラグ
    parser.add_argument('--no_normalize', action='store_true',
                        help='L2正規化を無効化')
    parser.add_argument('--no_fp16', action='store_true',
                        help='FP16を無効化')
    parser.add_argument('--verbose', action='store_true',
                        help='詳細ログ表示')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    try:
        # 埋め込み前計算システム初期化
        precomputer = EmbeddingPrecomputer(
            model_name=args.model,
            device=args.device,
            batch_size=args.batch_size,
            normalize_l2=not args.no_normalize,
            fp16=not args.no_fp16,
            seed=args.seed
        )
        
        # データ読み込み・前処理
        df = precomputer.load_data(
            data_path=args.data_path,
            text_cols=args.text_cols,
            id_col=args.id_col,
            sep_token=args.sep_token,
            max_rows=args.max_rows
        )
        
        # 埋め込み計算
        embeddings_dict = precomputer.compute_embeddings(
            df=df,
            chunk_size=args.chunk_size
        )
        
        # 結果保存
        saved_files = precomputer.save_embeddings(
            embeddings_dict=embeddings_dict,
            output_dir=args.output_dir,
            prefix=args.prefix
        )
        
        # 完了レポート
        print(f"\n🎉 埋め込み前計算完了!")
        print(f"📊 処理統計:")
        print(f"   モデル: {args.model}")
        print(f"   処理数: {len(embeddings_dict):,} samples")
        print(f"   次元数: {precomputer.embedding_dim}")
        print(f"   出力先: {args.output_dir}")
        
        print(f"\n📁 保存ファイル:")
        for file_type, file_path in saved_files.items():
            print(f"   {file_type}: {file_path}")
            
    except Exception as e:
        logger.error(f"❌ 実行失敗: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
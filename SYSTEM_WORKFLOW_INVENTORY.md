# System Workflow Inventory

このドキュメントは、リポジトリ内の **GraphRAG / Chameleon / CFS / Evaluator / Pipeline** を自動走査して概観をまとめたものです。

## 1. 主要セクションと該当ファイル
### Chameleon/Evaluator
- `chameleon_evaluator.py`
- `chameleon_evaluator_fixed.py`
- `chameleon_frozen_base.py`
- `chameleon_paper_compliant.py`
- `manifold_chameleon_evaluator.py`

### Personalization/Theta
- `enhance_theta_vectors.py`
- `lamp2_corrected_theta_retrainer.py`
- `neutral_direction_generator.py`
- `results/phase3c_theta_corrected/theta_n_lamp2_corrected.json`
- `results/phase3c_theta_corrected/theta_p_lamp2_corrected.json`

### GraphRAG/Indexing/Retrieval
- `build_faiss_index.py`
- `embeddings`
- `faiss`
- `faiss/lamp2_users`
- `faiss/lamp2_users_hnsw`
- `faiss/lamp2_users_hnsw_clamp`
- `faiss/lamp2_users_ivf_auto`
- `faiss/lamp2_users_ivf_correct`
- `faiss/lamp2_users_ivf_fix`
- `graphrag_cfs_weights`
- `graphrag_cfs_weights_strict`
- `precompute_embeddings.py`
- `rag`
- `rag/__pycache__`

### CFS Integration / Fusion
- `adaptive_fusion_cfs_integration.py`
- `adaptive_piece_fusion.py`
- `cfs_chameleon_demo.py`
- `cfs_chameleon_extension.py`
- `cfs_chameleon_graphrag_implementation_report.md`
- `cfs_chameleon_with_graphrag_part2.md`
- `cfs_comprehensive_evaluation.py`
- `cfs_comprehensive_evaluation_20250805_133209.csv`
- `cfs_comprehensive_evaluation_20250805_133209.json`
- `cfs_comprehensive_results_report.md`
- `cfs_config.yaml`
- `cfs_demo_results`
- `cfs_evaluation_results`
- `cfs_evaluation_utils.py`
- `cfs_improved_integration.py`
- `cfs_quality_integration.py`
- `cfs_quick_evaluation.py`
- `cfs_quick_evaluation_20250805_133938.csv`
- `cfs_semantic_integration.py`
- `collab_fusion_loader.py`
- `dual_direction_cfs_integration.py`

### PPR/Graph Utilities
- `integrate_ppr_cfs.py`
- `ppr_results`
- `ppr_results_user100`
- `run_ppr.py`

### Pipelines/Schedulers
- `production`
- `rango-gate.service`
- `rango-gate.timer`
- `scripts`
- `scripts/__pycache__`
- `scripts/cfs_v2`
- `scripts/cfs_v2/__pycache__`
- `scripts/tools`
- `scripts/verification`
- `scripts/verification/__pycache__`
- `scripts/verification/utils`
- `scripts/verification/utils/__pycache__`
- `scripts/verification/v0`
- `scripts/verification/v0/__pycache__`
- `scripts/verification/v1`
- `scripts/verification/v1/__pycache__`
- `scripts/verification/v2`
- `scripts/verification/v2/__pycache__`
- `sweep_chameleon.py`
- `yaml_batch_runner.py`

### Benchmarks/Evals
- `eval`
- `eval/__pycache__`
- `lamp2_benchmark.py`
- `lamp2_paper_compliant.py`
- `lampqa_cfs_benchmark.py`
- `lampqa_evaluator.py`
- `metrics`
- `run_evaluation.py`
- `run_improved_evaluation.py`
- `test_cfs_chameleon_compatibility.py`
- `test_cfs_chameleon_units.py`
- `test_cfs_simple.py`
- `test_chameleon_full.py`
- `test_current_environement.py`
- `test_current_environemnt.py`
- `test_dependencies.py`
- `test_edit_ratio_control.py`
- `test_fixed_cfs_chameleon.py`
- `test_fixed_performance_differences.py`
- `test_improved_prompt.py`
- `test_label_leakage_prevention.py`
- `test_lamp2_paper_compliant.py`
- `test_projection_editing_mathematics.py`
- `test_prompt_with_model.py`
- `test_rope_fix.py`
- `test_uncertainty_estimation.py`

### Configs/Data
- `alpha_batch_config.yaml`
- `config.yaml`
- `config/lamp2_eval_config.yaml`
- `config/prod_baseline.yaml`
- `config/prod_cfs_v2.yaml`
- `config/w2_evaluation.yaml`
- `config_optimized.yaml`
- `data`
- `data/__pycache__`
- `data/evaluation`
- `data/lamp_expanded`
- `data/raw`
- `data/raw/LaMP`
- `data/raw/LaMP-2`
- `data/raw/LaMP/.git`
- `data/raw/LaMP/.git/branches`
- `data/raw/LaMP/.git/hooks`
- `data/raw/LaMP/.git/info`
- `data/raw/LaMP/.git/logs`
- `data/raw/LaMP/.git/logs/refs`
- `data/raw/LaMP/.git/logs/refs/heads`
- `data/raw/LaMP/.git/logs/refs/remotes`
- `data/raw/LaMP/.git/logs/refs/remotes/origin`
- `data/raw/LaMP/.git/objects`
- `data/raw/LaMP/.git/objects/info`
- `data/raw/LaMP/.git/objects/pack`
- `data/raw/LaMP/.git/refs`
- `data/raw/LaMP/.git/refs/heads`
- `data/raw/LaMP/.git/refs/remotes`
- `data/raw/LaMP/.git/refs/remotes/origin`
- `data/raw/LaMP/.git/refs/tags`
- `data/raw/LaMP/LaMP`
- `data/raw/LaMP/LaMP/data`
- `data/raw/LaMP/LaMP/metrics`
- `data/raw/LaMP/LaMP/prompts`
- `data/raw/LaMP/LaMP/utils`
- `data/raw/LaMP/PEFT`
- `data/raw/LaMP/PEFT/data`
- `data/raw/LaMP/ROPG`
- `data/raw/LaMP/ROPG/data`
- `data/raw/LaMP/ROPG/models`
- `data/raw/LaMP/ROPG/prompts`
- `data/raw/LaMP/ROPG/trainers`
- `data/raw/LaMP/ROPG/utils`
- `data/raw/LaMP/RSPG`
- `data/raw/LaMP/RSPG/data`
- `data/raw/LaMP/RSPG/metrics`
- `data/raw/LaMP/RSPG/modeling`
- `data/raw/LaMP/RSPG/utils`
- `data/raw/LaMP/data`
- `data/raw/LaMP/data/avocado`
- `data/raw/LaMP/eval`
- `data/raw/LaMP_all`
- `data/raw/LaMP_all/LaMP_1`
- `data/raw/LaMP_all/LaMP_1/time-based`
- `data/raw/LaMP_all/LaMP_1/time-based/dev`
- `data/raw/LaMP_all/LaMP_1/time-based/test`
- `data/raw/LaMP_all/LaMP_1/time-based/train`
- `data/raw/LaMP_all/LaMP_1/user-based`
- `data/raw/LaMP_all/LaMP_1/user-based/dev`
- `data/raw/LaMP_all/LaMP_1/user-based/test`
- `data/raw/LaMP_all/LaMP_1/user-based/train`
- `data/raw/LaMP_all/LaMP_2`
- `data/raw/LaMP_all/LaMP_2/time-based`
- `data/raw/LaMP_all/LaMP_2/time-based/dev`
- `data/raw/LaMP_all/LaMP_2/time-based/test`
- `data/raw/LaMP_all/LaMP_2/time-based/train`
- `data/raw/LaMP_all/LaMP_2/user-based`
- `data/raw/LaMP_all/LaMP_2/user-based/dev`
- `data/raw/LaMP_all/LaMP_2/user-based/test`
- `data/raw/LaMP_all/LaMP_2/user-based/train`
- `data/raw/LaMP_all/LaMP_3`
- `data/raw/LaMP_all/LaMP_3/time-based`
- `data/raw/LaMP_all/LaMP_3/time-based/dev`
- `data/raw/LaMP_all/LaMP_3/time-based/test`
- `data/raw/LaMP_all/LaMP_3/time-based/train`
- `data/raw/LaMP_all/LaMP_3/user-based`
- `data/raw/LaMP_all/LaMP_3/user-based/dev`
- `data/raw/LaMP_all/LaMP_3/user-based/test`
- `data/raw/LaMP_all/LaMP_3/user-based/train`
- `data/raw/LaMP_all/LaMP_4`
- `data/raw/LaMP_all/LaMP_4/time-based`
- `data/raw/LaMP_all/LaMP_4/time-based/dev`
- `data/raw/LaMP_all/LaMP_4/time-based/test`
- `data/raw/LaMP_all/LaMP_4/time-based/train`
- `data/raw/LaMP_all/LaMP_4/user-based`
- `data/raw/LaMP_all/LaMP_4/user-based/dev`
- `data/raw/LaMP_all/LaMP_4/user-based/test`
- `data/raw/LaMP_all/LaMP_4/user-based/train`
- `data/raw/LaMP_all/LaMP_5`
- `data/raw/LaMP_all/LaMP_5/time-based`
- `data/raw/LaMP_all/LaMP_5/time-based/dev`
- `data/raw/LaMP_all/LaMP_5/time-based/test`
- `data/raw/LaMP_all/LaMP_5/time-based/train`
- `data/raw/LaMP_all/LaMP_5/user-based`
- `data/raw/LaMP_all/LaMP_5/user-based/dev`
- `data/raw/LaMP_all/LaMP_5/user-based/test`
- `data/raw/LaMP_all/LaMP_5/user-based/train`
- `data/raw/LaMP_all/LaMP_6`
- `data/raw/LaMP_all/LaMP_6/time-based`
- `data/raw/LaMP_all/LaMP_6/time-based/dev`
- `data/raw/LaMP_all/LaMP_6/time-based/test`
- `data/raw/LaMP_all/LaMP_6/time-based/train`
- `data/raw/LaMP_all/LaMP_6/user-based`
- `data/raw/LaMP_all/LaMP_6/user-based/dev`
- `data/raw/LaMP_all/LaMP_6/user-based/test`
- `data/raw/LaMP_all/LaMP_6/user-based/train`
- `data/raw/LaMP_all/LaMP_7`
- `data/raw/LaMP_all/LaMP_7/time-based`
- `data/raw/LaMP_all/LaMP_7/time-based/dev`
- `data/raw/LaMP_all/LaMP_7/time-based/test`
- `data/raw/LaMP_all/LaMP_7/time-based/train`
- `data/raw/LaMP_all/LaMP_7/user-based`
- `data/raw/LaMP_all/LaMP_7/user-based/dev`
- `data/raw/LaMP_all/LaMP_7/user-based/test`
- `data/raw/LaMP_all/LaMP_7/user-based/train`
- `datasets`
- `templates`

## 2. ざっくり依存関係（主要ファイルの import 抽出）
- `build_faiss_index.py` → argparse, faiss, numpy, pandas, pyarrow, scipy, tqdm, warnings
- `cfs_chameleon_demo.py` → cfs_chameleon_extension, chameleon_cfs_integrator, chameleon_evaluator, datetime, numpy, torch
- `cfs_chameleon_extension.py` → dataclasses, hashlib, numpy, torch
- `cfs_chameleon_graphrag_implementation_report.md` → (stdlibのみ)
- `cfs_chameleon_with_graphrag_part2.md` → (stdlibのみ)
- `cfs_comprehensive_evaluation.py` → pandas, subprocess
- `cfs_comprehensive_evaluation_20250805_133209.csv` → (stdlibのみ)
- `cfs_comprehensive_evaluation_20250805_133209.json` → (stdlibのみ)
- `cfs_comprehensive_results_report.md` → (stdlibのみ)
- `cfs_config.yaml` → (stdlibのみ)
- `cfs_evaluation_utils.py` → dataclasses, matplotlib, numpy, pandas, scipy, seaborn, sklearn, warnings
- `cfs_improved_integration.py` → cfs_chameleon_extension, chameleon_cfs_integrator, improved_direction_pieces_generator, numpy, torch
- `cfs_quality_integration.py` → cfs_chameleon_extension, cfs_improved_integration, chameleon_cfs_integrator, numpy, task_based_quality_evaluator, torch
- `cfs_quick_evaluation.py` → pandas, subprocess
- `cfs_quick_evaluation_20250805_133938.csv` → (stdlibのみ)
- `cfs_semantic_integration.py` → cfs_chameleon_extension, chameleon_cfs_integrator, numpy, semantic_similarity_engine, torch
- `chameleon_evaluator.py` → causal_inference, csv, dataclasses, datetime, nltk, numpy, scipy, threading, torch, transformers, yaml
- `lamp2_corrected_theta_retrainer.py` → chameleon_evaluator, numpy, sklearn, torch
- `precompute_embeddings.py` → argparse, numpy, pandas, pyarrow, sentence_transformers, torch, tqdm, warnings
- `rag/cluster.py` → numpy, sklearn
- `rag/diversity.py` → numpy, sklearn
- `rag/retrieval.py` → numpy, pandas

## 3. パイプライン（概念図）
```mermaid
flowchart TD
  A[LaMP-2 Data] -->|expand (N=100)| B[LAMP Expanded Dev]
  B --> C[ChameleonEvaluator (Baseline)]
  B --> D[ChameleonEvaluator (Chameleon)]
  subgraph Personalization
    P1[theta_p.json]:::theta
    P2[theta_n.json]:::theta
  end
  P1 --> D
  P2 --> D
  subgraph GridSearch
    G1[alpha_personal]
    G2[alpha_general]
    G3[norm_scale]
    G4[edit_gate]
  end
  G1 --> D
  G2 --> D
  G3 --> D
  G4 --> D
  subgraph Retrieval
    R1[GraphRAG / FAISS]
    R2[PPR/Graph Utils]
  end
  R1 --> C
  R1 --> D
  R2 --> R1
  D --> E[McNemar Test]
  E --> F[best_config.json]
  F --> Y[config_optimized.yaml]
classDef theta fill:#eef,stroke:#66f,stroke-width:1px
```

## 4. 動作の要点（解釈）
- **ChameleonEvaluator** が Expanded Dev セットに対して Baseline/Chameleon の2系統を評価。
- **Personalization (theta)** は `results/phase3c_theta_corrected/` の JSON を参照。
- **GridSearch** では α_p/α_g/norm_scale/edit_gate を系統探索し、**McNemar** で有意性判定。
- **GraphRAG/FAISS/PPR** は Retrieval レイヤで、必要に応じて Evaluator の入出力時に利用（コード上は `rag/`, `build_faiss_index.py`, `precompute_embeddings.py`, `integrate_ppr_cfs.py` など）。
- **Schedulers/Pipelines** は `yaml_batch_runner.py`, `sweep_chameleon.py`, `rango-gate.*`, `scripts/` 等で実行バッチや定期タスクをサポート。

## 5. 主要エントリポイント（実行例）
- グリッドサーチ: `python grid_search_robust.py`
- ベスト設定出力: `python apply_best_config.py` → `config_optimized.yaml`
- 最終評価: `python run_w2_evaluation.py --config config_optimized.yaml`

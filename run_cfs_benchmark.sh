#!/bin/bash

# 🚀 CFS-Chameleon LaMP-2 ベンチマーク実行スクリプト
# 使用方法: ./run_cfs_benchmark.sh [quick|full|debug|compare]

set -e  # エラーで停止

# 設定
GPU_DEVICE="0"
BASE_CMD="CUDA_VISIBLE_DEVICES=${GPU_DEVICE} python lamp2_cfs_benchmark.py"

# 色付きログ
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING:${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $1"
}

# GPU確認
check_gpu() {
    log "🔍 GPU確認中..."
    if ! nvidia-smi > /dev/null 2>&1; then
        log_error "nvidia-smiが見つかりません。CUDAが正しくインストールされているか確認してください。"
        exit 1
    fi
    
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    log "✅ GPU確認完了"
}

# 依存関係確認
check_dependencies() {
    log "🔍 依存関係確認中..."
    
    # Python環境確認
    if ! python --version > /dev/null 2>&1; then
        log_error "Pythonが見つかりません"
        exit 1
    fi
    
    # 必要なファイル確認
    if [ ! -f "lamp2_cfs_benchmark.py" ]; then
        log_error "lamp2_cfs_benchmark.pyが見つかりません"
        exit 1
    fi
    
    if [ ! -f "cfs_config.yaml" ]; then
        log_error "cfs_config.yamlが見つかりません"
        exit 1
    fi
    
    log "✅ 依存関係確認完了"
}

# クイックテスト（3サンプル、高速）
run_quick_test() {
    log "🚀 クイックテスト実行開始 (3サンプル)"
    log "   パラメータ: α_p=0.3, α_n=-0.1, max_length=30"
    
    $BASE_CMD \
        --compare_modes \
        --use_collaboration \
        --config cfs_config.yaml \
        --sample_limit 3 \
        --alpha_p 0.3 \
        --alpha_n -0.1 \
        --max_length 30
    
    log "✅ クイックテスト完了"
}

# デバッグモードテスト
run_debug_test() {
    log "🐛 デバッグテスト実行開始 (5サンプル)"
    log "   詳細出力でスコア分析を行います"
    
    $BASE_CMD \
        --compare_modes \
        --use_collaboration \
        --config cfs_config.yaml \
        --sample_limit 5 \
        --alpha_p 0.3 \
        --alpha_n -0.1 \
        --max_length 30 \
        --debug_mode
    
    log "✅ デバッグテスト完了"
}

# フル評価（10+サンプル、本格評価）
run_full_evaluation() {
    log "🎯 フル評価実行開始 (20サンプル)"
    log "   本格的な性能評価を行います（時間がかかります）"
    
    $BASE_CMD \
        --compare_modes \
        --use_collaboration \
        --config cfs_config.yaml \
        --sample_limit 20 \
        --alpha_p 0.3 \
        --alpha_n -0.1 \
        --max_length 30
    
    log "✅ フル評価完了"
}

# パラメータ最適化テスト
run_optimization_test() {
    log "⚙️  パラメータ最適化テスト開始"
    
    # 複数のパラメータ組み合わせをテスト
    params=(
        "0.1 -0.02 64"   # 軽微編集
        "0.3 -0.1 30"    # バランス型（推奨）
        "0.5 -0.2 20"    # 中程度編集
    )
    
    for param in "${params[@]}"; do
        read -r alpha_p alpha_n max_len <<< "$param"
        log "🧪 テスト中: α_p=${alpha_p}, α_n=${alpha_n}, max_length=${max_len}"
        
        $BASE_CMD \
            --compare_modes \
            --use_collaboration \
            --config cfs_config.yaml \
            --sample_limit 5 \
            --alpha_p $alpha_p \
            --alpha_n $alpha_n \
            --max_length $max_len
        
        echo "----------------------------------------"
    done
    
    log "✅ パラメータ最適化テスト完了"
}

# CFS単体評価
run_cfs_only() {
    log "🦎 CFS-Chameleon単体評価開始 (10サンプル)"
    
    $BASE_CMD \
        --use_collaboration \
        --config cfs_config.yaml \
        --evaluation_mode cfs \
        --sample_limit 10 \
        --alpha_p 0.3 \
        --alpha_n -0.1 \
        --max_length 30
    
    log "✅ CFS-Chameleon単体評価完了"
}

# レガシー単体評価
run_legacy_only() {
    log "🔸 従来版Chameleon単体評価開始 (10サンプル)"
    
    $BASE_CMD \
        --config cfs_config.yaml \
        --evaluation_mode legacy \
        --sample_limit 10 \
        --alpha_p 0.3 \
        --alpha_n -0.1 \
        --max_length 30
    
    log "✅ 従来版Chameleon単体評価完了"
}

# 結果表示
show_results() {
    log "📊 結果ファイル一覧:"
    
    if [ -d "cfs_evaluation_results" ]; then
        ls -la cfs_evaluation_results/
        
        if [ -f "cfs_evaluation_results/cfs_comparison_results.json" ]; then
            log "📈 最新の比較結果:"
            echo "----------------------------------------"
            cat cfs_evaluation_results/cfs_comparison_results.json | python -m json.tool | head -20
            echo "----------------------------------------"
        fi
    else
        log_warn "結果ディレクトリが見つかりません"
    fi
}

# ヘルプ表示
show_help() {
    echo "🚀 CFS-Chameleon LaMP-2 ベンチマーク実行スクリプト"
    echo ""
    echo "使用方法:"
    echo "  ./run_cfs_benchmark.sh [オプション]"
    echo ""
    echo "オプション:"
    echo "  quick      - クイックテスト (3サンプル、推奨開始点)"
    echo "  debug      - デバッグモード (5サンプル、詳細出力)"
    echo "  full       - フル評価 (20サンプル、本格評価)"
    echo "  optimize   - パラメータ最適化テスト"
    echo "  cfs        - CFS-Chameleon単体評価"
    echo "  legacy     - 従来版Chameleon単体評価"
    echo "  results    - 結果表示のみ"
    echo "  help       - このヘルプを表示"
    echo ""
    echo "実行例:"
    echo "  ./run_cfs_benchmark.sh quick      # クイックテスト"
    echo "  ./run_cfs_benchmark.sh debug      # デバッグモード"
    echo "  ./run_cfs_benchmark.sh full       # フル評価"
    echo ""
    echo "スコア向上のコツ:"
    echo "  • α_p=0.3, α_n=-0.1 が推奨パラメータ"
    echo "  • max_length=30 で適切な長さの出力"
    echo "  • debug モードで予測と正解を詳細確認"
}

# メイン処理
main() {
    echo "🦎 CFS-Chameleon LaMP-2 ベンチマークシステム"
    echo "================================================"
    
    # 基本チェック
    check_gpu
    check_dependencies
    
    # 引数処理
    case "${1:-quick}" in
        "quick"|"q")
            run_quick_test
            ;;
        "debug"|"d")
            run_debug_test
            ;;
        "full"|"f")
            run_full_evaluation
            ;;
        "optimize"|"opt"|"o")
            run_optimization_test
            ;;
        "cfs"|"c")
            run_cfs_only
            ;;
        "legacy"|"l")
            run_legacy_only
            ;;
        "results"|"r")
            show_results
            exit 0
            ;;
        "help"|"h"|"-h"|"--help")
            show_help
            exit 0
            ;;
        *)
            log_error "不明なオプション: $1"
            show_help
            exit 1
            ;;
    esac
    
    # 結果表示
    echo ""
    show_results
    
    echo ""
    log "🎉 実行完了! 結果は cfs_evaluation_results/ に保存されました"
    log "📊 比較結果を確認するには: ./run_cfs_benchmark.sh results"
}

# スクリプト実行
main "$@"
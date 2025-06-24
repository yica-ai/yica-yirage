#!/bin/bash

# YICA运行时优化器启动脚本
# 支持GPU加速的存算一体架构优化器

set -e

# 默认配置
YICA_MODE=${YICA_MODE:-"runtime"}
YICA_CONFIG=${YICA_CONFIG:-"default"}
YICA_GPU_ENABLED=${YICA_GPU_ENABLED:-"true"}
YICA_LOG_LEVEL=${YICA_LOG_LEVEL:-"INFO"}
YICA_MONITORING_ENABLED=${YICA_MONITORING_ENABLED:-"true"}
YICA_ML_OPTIMIZATION_ENABLED=${YICA_ML_OPTIMIZATION_ENABLED:-"true"}

# 目录配置
YICA_ROOT=${YICA_ROOT:-"/workspace/yica-optimizer"}
YICA_RUNTIME_DIR=${YICA_RUNTIME_DIR:-"/workspace/yica-runtime"}
YICA_CONFIG_DIR=${YICA_CONFIG_DIR:-"${YICA_RUNTIME_DIR}/configs"}
YICA_LOG_DIR=${YICA_LOG_DIR:-"${YICA_RUNTIME_DIR}/logs"}
YICA_CHECKPOINT_DIR=${YICA_CHECKPOINT_DIR:-"${YICA_RUNTIME_DIR}/checkpoints"}
YICA_MODEL_DIR=${YICA_MODEL_DIR:-"${YICA_RUNTIME_DIR}/models"}

# 日志函数
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $1"
}

# 检查GPU可用性
check_gpu() {
    if [ "$YICA_GPU_ENABLED" = "true" ]; then
        log_info "检查GPU可用性..."
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
            if [ $? -eq 0 ]; then
                log_info "GPU检查通过"
                export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"all"}
                return 0
            else
                log_error "GPU不可用，但GPU模式已启用"
                return 1
            fi
        else
            log_error "nvidia-smi命令不可用"
            return 1
        fi
    else
        log_info "CPU模式启动"
        export CUDA_VISIBLE_DEVICES=""
        return 0
    fi
}

# 初始化目录结构
init_directories() {
    log_info "初始化目录结构..."
    
    # 创建必要目录
    mkdir -p "$YICA_CONFIG_DIR"
    mkdir -p "$YICA_LOG_DIR"
    mkdir -p "$YICA_CHECKPOINT_DIR"
    mkdir -p "$YICA_MODEL_DIR"
    
    # 设置权限
    chmod 755 "$YICA_RUNTIME_DIR"
    chmod 755 "$YICA_CONFIG_DIR"
    chmod 755 "$YICA_LOG_DIR"
    chmod 755 "$YICA_CHECKPOINT_DIR"
    chmod 755 "$YICA_MODEL_DIR"
    
    log_info "目录初始化完成"
}

# 生成默认配置
generate_default_config() {
    local config_file="$YICA_CONFIG_DIR/runtime_config.json"
    
    if [ ! -f "$config_file" ]; then
        log_info "生成默认运行时配置..."
        
        cat > "$config_file" << EOF
{
    "runtime": {
        "mode": "$YICA_MODE",
        "gpu_enabled": $YICA_GPU_ENABLED,
        "log_level": "$YICA_LOG_LEVEL",
        "monitoring_enabled": $YICA_MONITORING_ENABLED,
        "ml_optimization_enabled": $YICA_ML_OPTIMIZATION_ENABLED
    },
    "performance_monitor": {
        "collection_frequency_hz": 1000,
        "sliding_window_size": 100,
        "anomaly_detection_enabled": true,
        "metrics_export_enabled": true
    },
    "ml_optimizer": {
        "model_type": "lstm",
        "learning_rate": 0.001,
        "batch_size": 32,
        "sequence_length": 50,
        "hidden_size": 128,
        "num_layers": 2,
        "online_learning_enabled": true
    },
    "reinforcement_optimizer": {
        "algorithm": "q_learning",
        "learning_rate": 0.1,
        "discount_factor": 0.95,
        "epsilon": 0.1,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01
    },
    "multi_objective_optimizer": {
        "algorithm": "nsga2",
        "population_size": 50,
        "max_generations": 100,
        "crossover_rate": 0.8,
        "mutation_rate": 0.1
    },
    "hardware": {
        "cim_array_size": 256,
        "spm_size_mb": 32,
        "memory_bandwidth_gbps": 1024,
        "compute_units": 64
    },
    "optimization_objectives": {
        "performance_weight": 0.4,
        "energy_efficiency_weight": 0.3,
        "latency_weight": 0.3,
        "target_performance_improvement": 0.15,
        "target_energy_reduction": 0.20
    }
}
EOF
        log_info "默认配置已生成: $config_file"
    else
        log_info "使用现有配置: $config_file"
    fi
}

# 启动性能监控
start_performance_monitor() {
    if [ "$YICA_MONITORING_ENABLED" = "true" ]; then
        log_info "启动性能监控器..."
        
        # 启动监控后台进程
        nohup "$YICA_ROOT/build/tests/yica/test_runtime_optimizer" \
            --mode=monitor \
            --config="$YICA_CONFIG_DIR/runtime_config.json" \
            --log-dir="$YICA_LOG_DIR" \
            > "$YICA_LOG_DIR/performance_monitor.log" 2>&1 &
        
        echo $! > "$YICA_LOG_DIR/performance_monitor.pid"
        log_info "性能监控器已启动 (PID: $(cat $YICA_LOG_DIR/performance_monitor.pid))"
    fi
}

# 启动ML优化器
start_ml_optimizer() {
    if [ "$YICA_ML_OPTIMIZATION_ENABLED" = "true" ]; then
        log_info "启动ML优化器..."
        
        # 启动ML优化器后台进程
        nohup "$YICA_ROOT/build/tests/yica/test_runtime_optimizer" \
            --mode=ml_optimizer \
            --config="$YICA_CONFIG_DIR/runtime_config.json" \
            --model-dir="$YICA_MODEL_DIR" \
            --checkpoint-dir="$YICA_CHECKPOINT_DIR" \
            --log-dir="$YICA_LOG_DIR" \
            > "$YICA_LOG_DIR/ml_optimizer.log" 2>&1 &
        
        echo $! > "$YICA_LOG_DIR/ml_optimizer.pid"
        log_info "ML优化器已启动 (PID: $(cat $YICA_LOG_DIR/ml_optimizer.pid))"
    fi
}

# 启动主运行时
start_yica_runtime() {
    log_info "启动YICA运行时优化器..."
    
    # 构建启动命令
    local cmd="$YICA_ROOT/build/tests/yica/test_runtime_optimizer"
    local args=(
        "--mode=runtime"
        "--config=$YICA_CONFIG_DIR/runtime_config.json"
        "--log-dir=$YICA_LOG_DIR"
        "--checkpoint-dir=$YICA_CHECKPOINT_DIR"
        "--model-dir=$YICA_MODEL_DIR"
        "--log-level=$YICA_LOG_LEVEL"
    )
    
    if [ "$YICA_GPU_ENABLED" = "true" ]; then
        args+=("--gpu-enabled")
    fi
    
    # 启动主进程
    exec "$cmd" "${args[@]}"
}

# 清理函数
cleanup() {
    log_info "正在清理资源..."
    
    # 停止后台进程
    if [ -f "$YICA_LOG_DIR/performance_monitor.pid" ]; then
        local pid=$(cat "$YICA_LOG_DIR/performance_monitor.pid")
        if kill -0 "$pid" 2>/dev/null; then
            log_info "停止性能监控器 (PID: $pid)"
            kill -TERM "$pid"
            wait "$pid" 2>/dev/null || true
        fi
        rm -f "$YICA_LOG_DIR/performance_monitor.pid"
    fi
    
    if [ -f "$YICA_LOG_DIR/ml_optimizer.pid" ]; then
        local pid=$(cat "$YICA_LOG_DIR/ml_optimizer.pid")
        if kill -0 "$pid" 2>/dev/null; then
            log_info "停止ML优化器 (PID: $pid)"
            kill -TERM "$pid"
            wait "$pid" 2>/dev/null || true
        fi
        rm -f "$YICA_LOG_DIR/ml_optimizer.pid"
    fi
    
    # 清理运行时PID文件
    rm -f "$YICA_LOG_DIR/yica-runtime.pid"
    
    log_info "清理完成"
}

# 信号处理
trap cleanup EXIT INT TERM

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode=*)
            YICA_MODE="${1#*=}"
            shift
            ;;
        --config=*)
            YICA_CONFIG="${1#*=}"
            shift
            ;;
        --gpu-enabled)
            YICA_GPU_ENABLED="true"
            shift
            ;;
        --no-gpu)
            YICA_GPU_ENABLED="false"
            shift
            ;;
        --log-level=*)
            YICA_LOG_LEVEL="${1#*=}"
            shift
            ;;
        --help|-h)
            echo "YICA运行时优化器启动脚本"
            echo ""
            echo "使用方法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --mode=MODE              运行模式 (runtime|monitor|ml_optimizer)"
            echo "  --config=CONFIG          配置名称 (default|custom)"
            echo "  --gpu-enabled            启用GPU加速"
            echo "  --no-gpu                 禁用GPU"
            echo "  --log-level=LEVEL        日志级别 (DEBUG|INFO|WARN|ERROR)"
            echo "  --help, -h               显示此帮助信息"
            echo ""
            echo "环境变量:"
            echo "  YICA_MODE                运行模式"
            echo "  YICA_GPU_ENABLED         GPU启用状态"
            echo "  YICA_LOG_LEVEL           日志级别"
            echo "  YICA_MONITORING_ENABLED  性能监控启用状态"
            echo "  YICA_ML_OPTIMIZATION_ENABLED  ML优化启用状态"
            exit 0
            ;;
        *)
            log_error "未知参数: $1"
            exit 1
            ;;
    esac
done

# 主启动流程
main() {
    log_info "启动YICA运行时优化器..."
    log_info "模式: $YICA_MODE"
    log_info "GPU启用: $YICA_GPU_ENABLED"
    log_info "监控启用: $YICA_MONITORING_ENABLED"
    log_info "ML优化启用: $YICA_ML_OPTIMIZATION_ENABLED"
    
    # 检查GPU
    if ! check_gpu; then
        if [ "$YICA_GPU_ENABLED" = "true" ]; then
            log_error "GPU检查失败，退出"
            exit 1
        fi
    fi
    
    # 初始化环境
    init_directories
    generate_default_config
    
    # 创建运行时PID文件
    echo $$ > "$YICA_LOG_DIR/yica-runtime.pid"
    
    # 启动组件
    case "$YICA_MODE" in
        "runtime")
            start_performance_monitor
            start_ml_optimizer
            start_yica_runtime
            ;;
        "monitor")
            start_performance_monitor
            wait
            ;;
        "ml_optimizer")
            start_ml_optimizer
            wait
            ;;
        *)
            log_error "未知运行模式: $YICA_MODE"
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@" 
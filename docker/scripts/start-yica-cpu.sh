#!/bin/bash

# YICA运行时优化器CPU启动脚本
# 不依赖GPU驱动，支持GPU行为模拟

set -e

# 默认配置
YICA_MODE=${YICA_MODE:-"runtime"}
YICA_CONFIG=${YICA_CONFIG:-"cpu"}
YICA_GPU_SIMULATION=${YICA_GPU_SIMULATION:-"true"}
YICA_CPU_ONLY=${YICA_CPU_ONLY:-"true"}
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
YICA_SIMULATION_DIR=${YICA_SIMULATION_DIR:-"${YICA_RUNTIME_DIR}/simulation"}

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

# 检查CPU环境
check_cpu_environment() {
    log_info "检查CPU计算环境..."
    
    # 检查CPU核心数
    local cpu_cores=$(nproc)
    log_info "可用CPU核心数: $cpu_cores"
    
    # 检查内存
    local memory_gb=$(free -g | awk '/^Mem:/{print $2}')
    log_info "可用内存: ${memory_gb}GB"
    
    # 检查OpenMP支持
    if command -v gcc &> /dev/null; then
        if gcc -fopenmp -dM -E - < /dev/null | grep -q "_OPENMP"; then
            log_info "OpenMP支持: 已启用"
            export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$cpu_cores}
        else
            log_warn "OpenMP支持: 未检测到"
        fi
    fi
    
    # 检查BLAS库
    if ldconfig -p | grep -q "libopenblas\|libmkl\|libblas"; then
        log_info "BLAS库: 已安装"
    else
        log_warn "BLAS库: 未检测到优化的BLAS库"
    fi
    
    log_info "CPU环境检查完成"
}

# 初始化GPU模拟环境
init_gpu_simulation() {
    if [ "$YICA_GPU_SIMULATION" = "true" ]; then
        log_info "初始化GPU模拟环境..."
        
        # 创建模拟GPU设备信息
        mkdir -p "$YICA_SIMULATION_DIR/gpu_devices"
        
        # 模拟GPU 0
        cat > "$YICA_SIMULATION_DIR/gpu_devices/gpu0.json" << EOF
{
    "device_id": 0,
    "name": "Simulated NVIDIA A100",
    "compute_capability": "8.6",
    "memory_total_mb": 16384,
    "memory_free_mb": 15360,
    "utilization_percent": 0,
    "temperature_celsius": 35,
    "power_watts": 50,
    "clock_graphics_mhz": 1410,
    "clock_memory_mhz": 1215
}
EOF

        # 模拟GPU 1
        cat > "$YICA_SIMULATION_DIR/gpu_devices/gpu1.json" << EOF
{
    "device_id": 1,
    "name": "Simulated NVIDIA A100",
    "compute_capability": "8.6",
    "memory_total_mb": 16384,
    "memory_free_mb": 15360,
    "utilization_percent": 0,
    "temperature_celsius": 36,
    "power_watts": 52,
    "clock_graphics_mhz": 1410,
    "clock_memory_mhz": 1215
}
EOF

        # 创建模拟的nvidia-smi脚本
        cat > "$YICA_SIMULATION_DIR/nvidia-smi" << 'EOF'
#!/bin/bash
# 模拟nvidia-smi命令

case "$1" in
    "--query-gpu=name,memory.total,memory.free")
        echo "Simulated NVIDIA A100, 16384 MiB, 15360 MiB"
        echo "Simulated NVIDIA A100, 16384 MiB, 15360 MiB"
        ;;
    "--query-gpu=utilization.gpu,memory.used,memory.total")
        echo "0 %, 1024 MiB, 16384 MiB"
        echo "0 %, 1024 MiB, 16384 MiB"
        ;;
    *)
        echo "Fri Dec  8 10:30:00 2023"
        echo "+-----------------------------------------------------------------------------+"
        echo "| NVIDIA-SMI 535.54.03              Driver Version: 535.54.03    CUDA Version: 12.2     |"
        echo "|-------------------------------+----------------------+----------------------+"
        echo "| GPU  Name                     Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |"
        echo "| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |"
        echo "|                               |                      |               MIG M. |"
        echo "|===============================+======================+======================|"
        echo "|   0  Simulated NVIDIA A100    Off  | 00000000:00:00.0 Off |                    0 |"
        echo "| N/A   35C    P0    50W / 400W |   1024MiB / 16384MiB |      0%      Default |"
        echo "|                               |                      |                  N/A |"
        echo "+-------------------------------+----------------------+----------------------+"
        echo "|   1  Simulated NVIDIA A100    Off  | 00000000:01:00.0 Off |                    0 |"
        echo "| N/A   36C    P0    52W / 400W |   1024MiB / 16384MiB |      0%      Default |"
        echo "|                               |                      |                  N/A |"
        echo "+-------------------------------+----------------------+----------------------+"
        echo ""
        echo "+-----------------------------------------------------------------------------+"
        echo "| Processes:                                                                  |"
        echo "|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |"
        echo "|        ID   ID                                                   Usage      |"
        echo "|=============================================================================|"
        echo "|  No running processes found                                                 |"
        echo "+-----------------------------------------------------------------------------+"
        ;;
esac
EOF

        chmod +x "$YICA_SIMULATION_DIR/nvidia-smi"
        
        # 将模拟脚本添加到PATH
        export PATH="$YICA_SIMULATION_DIR:$PATH"
        
        log_info "GPU模拟环境初始化完成"
        
        # 显示模拟GPU信息
        log_info "模拟GPU设备信息:"
        nvidia-smi
    else
        log_info "GPU模拟已禁用，纯CPU模式"
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
    mkdir -p "$YICA_SIMULATION_DIR"
    
    # 设置权限
    chmod 755 "$YICA_RUNTIME_DIR"
    chmod 755 "$YICA_CONFIG_DIR"
    chmod 755 "$YICA_LOG_DIR"
    chmod 755 "$YICA_CHECKPOINT_DIR"
    chmod 755 "$YICA_MODEL_DIR"
    chmod 755 "$YICA_SIMULATION_DIR"
    
    log_info "目录初始化完成"
}

# 生成CPU优化配置
generate_cpu_config() {
    local config_file="$YICA_CONFIG_DIR/runtime_config_cpu.json"
    local cpu_cores=$(nproc)
    local memory_gb=$(free -g | awk '/^Mem:/{print $2}')
    
    if [ ! -f "$config_file" ]; then
        log_info "生成CPU优化运行时配置..."
        
        cat > "$config_file" << EOF
{
    "runtime": {
        "mode": "$YICA_MODE",
        "gpu_enabled": false,
        "gpu_simulation_enabled": $YICA_GPU_SIMULATION,
        "cpu_only": $YICA_CPU_ONLY,
        "log_level": "$YICA_LOG_LEVEL",
        "monitoring_enabled": $YICA_MONITORING_ENABLED,
        "ml_optimization_enabled": $YICA_ML_OPTIMIZATION_ENABLED
    },
    "cpu_optimization": {
        "thread_count": $cpu_cores,
        "use_openmp": true,
        "use_simd": true,
        "cache_optimization": true,
        "memory_bandwidth_optimization": true,
        "numa_awareness": true
    },
    "gpu_simulation": {
        "enabled": $YICA_GPU_SIMULATION,
        "simulated_gpu_count": 2,
        "simulated_memory_gb": 16,
        "simulated_compute_capability": "8.6",
        "performance_scaling_factor": 0.1,
        "latency_simulation_ms": 2.0
    },
    "performance_monitor": {
        "collection_frequency_hz": 100,
        "sliding_window_size": 50,
        "anomaly_detection_enabled": true,
        "metrics_export_enabled": true,
        "cpu_metrics_enabled": true,
        "memory_metrics_enabled": true
    },
    "ml_optimizer": {
        "model_type": "cpu_optimized_lstm",
        "learning_rate": 0.001,
        "batch_size": 16,
        "sequence_length": 30,
        "hidden_size": 64,
        "num_layers": 2,
        "online_learning_enabled": true,
        "cpu_backend": "openmp"
    },
    "reinforcement_optimizer": {
        "algorithm": "q_learning",
        "learning_rate": 0.1,
        "discount_factor": 0.95,
        "epsilon": 0.1,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
        "cpu_parallelization": true
    },
    "multi_objective_optimizer": {
        "algorithm": "nsga2_cpu",
        "population_size": 30,
        "max_generations": 50,
        "crossover_rate": 0.8,
        "mutation_rate": 0.1,
        "parallel_evaluation": true
    },
    "hardware": {
        "cpu_cores": $cpu_cores,
        "memory_gb": $memory_gb,
        "cache_size_mb": 32,
        "memory_bandwidth_gbps": 128,
        "simulated_cim_array_size": 256,
        "simulated_spm_size_mb": 32
    },
    "optimization_objectives": {
        "performance_weight": 0.5,
        "energy_efficiency_weight": 0.3,
        "latency_weight": 0.2,
        "target_performance_improvement": 0.10,
        "target_energy_reduction": 0.15
    }
}
EOF
        log_info "CPU优化配置已生成: $config_file"
    else
        log_info "使用现有CPU配置: $config_file"
    fi
}

# 启动性能监控 (CPU版本)
start_cpu_performance_monitor() {
    if [ "$YICA_MONITORING_ENABLED" = "true" ]; then
        log_info "启动CPU性能监控器..."
        
        # 启动监控后台进程
        nohup "$YICA_ROOT/build/tests/yica/test_runtime_optimizer" \
            --mode=monitor \
            --config="$YICA_CONFIG_DIR/runtime_config_cpu.json" \
            --log-dir="$YICA_LOG_DIR" \
            --cpu-only \
            > "$YICA_LOG_DIR/cpu_performance_monitor.log" 2>&1 &
        
        echo $! > "$YICA_LOG_DIR/cpu_performance_monitor.pid"
        log_info "CPU性能监控器已启动 (PID: $(cat $YICA_LOG_DIR/cpu_performance_monitor.pid))"
    fi
}

# 启动ML优化器 (CPU版本)
start_cpu_ml_optimizer() {
    if [ "$YICA_ML_OPTIMIZATION_ENABLED" = "true" ]; then
        log_info "启动CPU ML优化器..."
        
        # 启动ML优化器后台进程
        nohup "$YICA_ROOT/build/tests/yica/test_runtime_optimizer" \
            --mode=ml_optimizer \
            --config="$YICA_CONFIG_DIR/runtime_config_cpu.json" \
            --model-dir="$YICA_MODEL_DIR" \
            --checkpoint-dir="$YICA_CHECKPOINT_DIR" \
            --log-dir="$YICA_LOG_DIR" \
            --cpu-only \
            > "$YICA_LOG_DIR/cpu_ml_optimizer.log" 2>&1 &
        
        echo $! > "$YICA_LOG_DIR/cpu_ml_optimizer.pid"
        log_info "CPU ML优化器已启动 (PID: $(cat $YICA_LOG_DIR/cpu_ml_optimizer.pid))"
    fi
}

# 启动主运行时 (CPU版本)
start_yica_cpu_runtime() {
    log_info "启动YICA CPU运行时优化器..."
    
    # 构建启动命令
    local cmd="$YICA_ROOT/build/tests/yica/test_runtime_optimizer"
    local args=(
        "--mode=runtime"
        "--config=$YICA_CONFIG_DIR/runtime_config_cpu.json"
        "--log-dir=$YICA_LOG_DIR"
        "--checkpoint-dir=$YICA_CHECKPOINT_DIR"
        "--model-dir=$YICA_MODEL_DIR"
        "--log-level=$YICA_LOG_LEVEL"
        "--cpu-only"
    )
    
    if [ "$YICA_GPU_SIMULATION" = "true" ]; then
        args+=("--simulate-gpu")
        args+=("--simulation-dir=$YICA_SIMULATION_DIR")
    fi
    
    # 启动主进程
    exec "$cmd" "${args[@]}"
}

# 清理函数
cleanup() {
    log_info "正在清理资源..."
    
    # 停止后台进程
    if [ -f "$YICA_LOG_DIR/cpu_performance_monitor.pid" ]; then
        local pid=$(cat "$YICA_LOG_DIR/cpu_performance_monitor.pid")
        if kill -0 "$pid" 2>/dev/null; then
            log_info "停止CPU性能监控器 (PID: $pid)"
            kill -TERM "$pid"
            wait "$pid" 2>/dev/null || true
        fi
        rm -f "$YICA_LOG_DIR/cpu_performance_monitor.pid"
    fi
    
    if [ -f "$YICA_LOG_DIR/cpu_ml_optimizer.pid" ]; then
        local pid=$(cat "$YICA_LOG_DIR/cpu_ml_optimizer.pid")
        if kill -0 "$pid" 2>/dev/null; then
            log_info "停止CPU ML优化器 (PID: $pid)"
            kill -TERM "$pid"
            wait "$pid" 2>/dev/null || true
        fi
        rm -f "$YICA_LOG_DIR/cpu_ml_optimizer.pid"
    fi
    
    # 清理运行时PID文件
    rm -f "$YICA_LOG_DIR/yica-cpu-runtime.pid"
    
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
        --simulate-gpu)
            YICA_GPU_SIMULATION="true"
            shift
            ;;
        --no-simulation)
            YICA_GPU_SIMULATION="false"
            shift
            ;;
        --log-level=*)
            YICA_LOG_LEVEL="${1#*=}"
            shift
            ;;
        --help|-h)
            echo "YICA CPU运行时优化器启动脚本"
            echo ""
            echo "使用方法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --mode=MODE              运行模式 (runtime|monitor|ml_optimizer)"
            echo "  --config=CONFIG          配置名称 (cpu|custom)"
            echo "  --simulate-gpu           启用GPU模拟"
            echo "  --no-simulation          禁用GPU模拟"
            echo "  --log-level=LEVEL        日志级别 (DEBUG|INFO|WARN|ERROR)"
            echo "  --help, -h               显示此帮助信息"
            echo ""
            echo "环境变量:"
            echo "  YICA_MODE                运行模式"
            echo "  YICA_GPU_SIMULATION      GPU模拟启用状态"
            echo "  YICA_CPU_ONLY            CPU模式启用状态"
            echo "  YICA_LOG_LEVEL           日志级别"
            echo "  OMP_NUM_THREADS          OpenMP线程数"
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
    log_info "启动YICA CPU运行时优化器..."
    log_info "模式: $YICA_MODE"
    log_info "CPU模式: $YICA_CPU_ONLY"
    log_info "GPU模拟: $YICA_GPU_SIMULATION"
    log_info "监控启用: $YICA_MONITORING_ENABLED"
    log_info "ML优化启用: $YICA_ML_OPTIMIZATION_ENABLED"
    
    # 检查CPU环境
    check_cpu_environment
    
    # 初始化环境
    init_directories
    init_gpu_simulation
    generate_cpu_config
    
    # 创建运行时PID文件
    echo $$ > "$YICA_LOG_DIR/yica-cpu-runtime.pid"
    
    # 启动组件
    case "$YICA_MODE" in
        "runtime")
            start_cpu_performance_monitor
            start_cpu_ml_optimizer
            start_yica_cpu_runtime
            ;;
        "monitor")
            start_cpu_performance_monitor
            wait
            ;;
        "ml_optimizer")
            start_cpu_ml_optimizer
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
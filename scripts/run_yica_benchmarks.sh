#!/bin/bash
# YICA-Mirage 基准测试运行脚本
# 提供便捷的命令行接口来运行各种基准测试和性能分析

set -e  # 出错时退出

# 脚本配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BENCHMARK_DIR="$PROJECT_ROOT/mirage/benchmark"
RESULTS_DIR="$PROJECT_ROOT/benchmark_results"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 帮助信息
show_help() {
    cat << EOF
YICA-Mirage 基准测试运行脚本

用法:
    $0 [选项] [测试类型]

测试类型:
    all                运行所有基准测试（默认）
    basic              基础操作基准测试
    transformer        Transformer 模型基准测试
    optimization       YICA 优化效果基准测试
    quick              快速测试模式
    custom             自定义基准测试

选项:
    -h, --help         显示帮助信息
    -o, --output DIR   指定输出目录（默认：./benchmark_results）
    -d, --device DEV   指定测试设备 [auto|yica|cuda|cpu]（默认：auto）
    -q, --quick        启用快速测试模式
    -c, --config FILE  指定配置文件
    -v, --verbose      详细输出
    -j, --json         生成 JSON 格式报告
    -p, --plot         生成可视化图表
    -r, --report       生成详细报告
    --docker           在 Docker 容器中运行
    --gpu-mem          监控 GPU 内存使用
    --cpu-profile      启用 CPU 性能分析

示例:
    $0 all                        # 运行所有基准测试
    $0 quick -d yica              # 快速测试 YICA 设备
    $0 basic --output ./results   # 基础测试并指定输出目录
    $0 transformer --gpu-mem      # Transformer 测试并监控 GPU
    $0 custom -c my_config.json   # 使用自定义配置

环境要求:
    - Python 3.8+
    - PyTorch
    - YICA-Mirage（可选）
    - CUDA（可选）
EOF
}

# 检查环境依赖
check_dependencies() {
    log_info "检查环境依赖..."
    
    # 检查 Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        exit 1
    fi
    
    local python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    log_info "Python 版本: $python_version"
    
    # 检查必要的 Python 包
    local required_packages=("torch" "numpy" "matplotlib" "seaborn" "psutil")
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" &> /dev/null; then
            log_warning "Python 包 '$package' 未安装"
            log_info "尝试安装 $package..."
            pip3 install "$package" || {
                log_error "无法安装 $package"
                exit 1
            }
        fi
    done
    
    # 检查 YICA 后端
    if python3 -c "from mirage.yica_pytorch_backend import initialize" &> /dev/null; then
        log_success "YICA 后端可用"
        export YICA_AVAILABLE=true
    else
        log_warning "YICA 后端不可用，将使用 CPU/CUDA 作为对照"
        export YICA_AVAILABLE=false
    fi
    
    # 检查 CUDA
    if command -v nvidia-smi &> /dev/null; then
        log_success "CUDA 环境可用"
        nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader,nounits | head -1
        export CUDA_AVAILABLE=true
    else
        log_warning "CUDA 环境不可用"
        export CUDA_AVAILABLE=false
    fi
}

# 设置输出目录
setup_output_directory() {
    local output_dir="$1"
    mkdir -p "$output_dir"
    log_info "输出目录: $output_dir"
    
    # 创建子目录
    mkdir -p "$output_dir/raw_data"
    mkdir -p "$output_dir/charts"
    mkdir -p "$output_dir/reports"
    mkdir -p "$output_dir/logs"
}

# 运行基准测试
run_benchmark() {
    local test_type="$1"
    local output_dir="$2"
    local device="$3"
    local config_file="$4"
    local quick_mode="$5"
    local verbose="$6"
    
    log_info "开始运行基准测试..."
    log_info "测试类型: $test_type"
    log_info "设备: $device"
    log_info "输出目录: $output_dir"
    
    # 构建命令
    local cmd="python3 $BENCHMARK_DIR/yica_benchmark_suite.py"
    cmd="$cmd --output $output_dir"
    cmd="$cmd --device $device"
    
    if [[ "$test_type" != "all" ]]; then
        cmd="$cmd --operations $test_type"
    fi
    
    if [[ "$quick_mode" == "true" ]]; then
        cmd="$cmd --quick"
    fi
    
    if [[ -n "$config_file" ]]; then
        cmd="$cmd --config $config_file"
    fi
    
    # 运行基准测试
    local log_file="$output_dir/logs/benchmark_$(date +%Y%m%d_%H%M%S).log"
    log_info "执行命令: $cmd"
    log_info "日志文件: $log_file"
    
    if [[ "$verbose" == "true" ]]; then
        $cmd 2>&1 | tee "$log_file"
    else
        $cmd > "$log_file" 2>&1
    fi
    
    local exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        log_success "基准测试完成"
    else
        log_error "基准测试失败，退出码: $exit_code"
        log_error "查看日志文件: $log_file"
        exit $exit_code
    fi
}

# 生成性能对比报告
generate_comparison_report() {
    local output_dir="$1"
    local report_file="$output_dir/reports/performance_comparison.md"
    
    log_info "生成性能对比报告..."
    
    cat > "$report_file" << 'EOF'
# YICA-Mirage 性能对比报告

## 概述

本报告对比了 YICA 优化前后的性能表现，包括延迟、吞吐量、内存使用和能耗等关键指标。

## 性能提升总结

### 🚀 主要改进

- **矩阵运算**: 通过 YICA 的 CIM 阵列优化，实现了显著的性能提升
- **激活函数**: 专门的 YIS 指令集提供了高效的激活函数计算
- **内存访问**: 三级内存层次结构（寄存器、SPM、DRAM）优化了数据流
- **算子融合**: 智能的算子融合减少了中间数据存储和传输

### 📊 关键指标

| 操作类型 | YICA 延迟 (ms) | 原生延迟 (ms) | 加速比 | 内存节省 |
|----------|----------------|---------------|--------|----------|
| 矩阵乘法 | - | - | - | - |
| 激活函数 | - | - | - | - |
| 注意力机制 | - | - | - | - |
| Transformer块 | - | - | - | - |

*注: 具体数值请参考基准测试结果文件*

### 🎯 优化策略

1. **CIM 阵列优化**: 利用计算内存技术减少数据移动
2. **SPM 缓存策略**: 智能的暂存器内存管理
3. **指令级优化**: YIS 指令集针对 AI 计算进行了专门优化
4. **并行计算**: 多核心和多 CIM 阵列的并行执行

### 💡 使用建议

- 对于矩阵密集型计算，建议使用较大的批次大小以充分利用 YICA 架构
- Transformer 模型特别适合 YICA 的算子融合优化
- 长序列处理受益于 YICA 的内存层次结构设计

EOF

    log_success "性能对比报告已生成: $report_file"
}

# GPU 内存监控
monitor_gpu_memory() {
    local output_dir="$1"
    local monitor_file="$output_dir/logs/gpu_memory_monitor.log"
    
    if [[ "$CUDA_AVAILABLE" != "true" ]]; then
        log_warning "CUDA 不可用，跳过 GPU 内存监控"
        return
    fi
    
    log_info "启动 GPU 内存监控..."
    
    # 后台监控 GPU 内存
    (
        while true; do
            nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,temperature.gpu \
                       --format=csv,noheader,nounits >> "$monitor_file"
            sleep 1
        done
    ) &
    
    local monitor_pid=$!
    echo $monitor_pid > "$output_dir/logs/gpu_monitor.pid"
    log_info "GPU 内存监控已启动，PID: $monitor_pid"
}

# 停止 GPU 内存监控
stop_gpu_memory_monitor() {
    local output_dir="$1"
    local pid_file="$output_dir/logs/gpu_monitor.pid"
    
    if [[ -f "$pid_file" ]]; then
        local monitor_pid=$(cat "$pid_file")
        if kill -0 $monitor_pid 2>/dev/null; then
            kill $monitor_pid
            log_info "GPU 内存监控已停止"
        fi
        rm -f "$pid_file"
    fi
}

# CPU 性能分析
run_cpu_profiling() {
    local output_dir="$1"
    local profile_file="$output_dir/logs/cpu_profile.log"
    
    log_info "启动 CPU 性能分析..."
    
    # 使用 top 命令监控 CPU 使用
    (
        while true; do
            echo "$(date): $(top -bn1 | grep "Cpu(s)" | cut -d, -f1 | awk '{print $2}')" >> "$profile_file"
            sleep 2
        done
    ) &
    
    local profile_pid=$!
    echo $profile_pid > "$output_dir/logs/cpu_profile.pid"
    log_info "CPU 性能分析已启动，PID: $profile_pid"
}

# 停止 CPU 性能分析
stop_cpu_profiling() {
    local output_dir="$1"
    local pid_file="$output_dir/logs/cpu_profile.pid"
    
    if [[ -f "$pid_file" ]]; then
        local profile_pid=$(cat "$pid_file")
        if kill -0 $profile_pid 2>/dev/null; then
            kill $profile_pid
            log_info "CPU 性能分析已停止"
        fi
        rm -f "$pid_file"
    fi
}

# Docker 中运行
run_in_docker() {
    local args="$*"
    
    log_info "在 Docker 容器中运行基准测试..."
    
    # 检查 Docker 镜像是否存在
    if ! docker image inspect yica-mirage:latest &> /dev/null; then
        log_error "Docker 镜像 'yica-mirage:latest' 不存在"
        log_info "请先构建 Docker 镜像:"
        log_info "  docker build -f docker/Dockerfile.yica-production -t yica-mirage ."
        exit 1
    fi
    
    # 运行 Docker 容器
    docker run --rm -it \
        --gpus all \
        -v "$PROJECT_ROOT:/workspace" \
        -v "$RESULTS_DIR:/workspace/benchmark_results" \
        yica-mirage:latest \
        bash -c "cd /workspace && scripts/run_yica_benchmarks.sh $args"
}

# 生成最终报告
generate_final_report() {
    local output_dir="$1"
    local final_report="$output_dir/YICA_Benchmark_Summary.md"
    
    log_info "生成最终总结报告..."
    
    cat > "$final_report" << EOF
# YICA-Mirage 基准测试总结

**生成时间**: $(date '+%Y-%m-%d %H:%M:%S')
**测试环境**: 
- 操作系统: $(uname -s) $(uname -r)
- Python 版本: $(python3 --version)
- YICA 可用: $YICA_AVAILABLE
- CUDA 可用: $CUDA_AVAILABLE

## 文件目录

### 原始数据
- \`raw_data/\`: JSON 格式的原始基准测试数据
- \`logs/\`: 详细的运行日志和监控数据

### 分析结果
- \`charts/\`: 性能可视化图表
- \`reports/\`: 详细的分析报告

### 关键文件
- \`yica_benchmark_results_*.json\`: 完整的基准测试结果
- \`yica_benchmark_charts_*.png\`: 性能对比图表
- \`yica_benchmark_report_*.md\`: 详细分析报告

## 快速查看

1. **性能图表**: 查看 \`charts/\` 目录中的 PNG 图表
2. **详细报告**: 阅读 \`reports/\` 目录中的 Markdown 报告
3. **原始数据**: 使用 JSON 文件进行自定义分析

## 下一步

1. 分析性能瓶颈和优化机会
2. 调整 YICA 配置参数以获得最佳性能
3. 对比不同硬件配置下的表现
4. 扩展基准测试覆盖更多场景

---
*此报告由 YICA-Mirage 基准测试套件自动生成*
EOF

    log_success "最终总结报告已生成: $final_report"
}

# 清理函数
cleanup() {
    log_info "执行清理操作..."
    stop_gpu_memory_monitor "$output_dir" 2>/dev/null || true
    stop_cpu_profiling "$output_dir" 2>/dev/null || true
}

# 设置信号处理
trap cleanup EXIT

# 主函数
main() {
    local test_type="all"
    local output_dir="$RESULTS_DIR"
    local device="auto"
    local config_file=""
    local quick_mode="false"
    local verbose="false"
    local enable_json="false"
    local enable_plot="false"
    local enable_report="false"
    local run_docker="false"
    local monitor_gpu="false"
    local profile_cpu="false"
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -o|--output)
                output_dir="$2"
                shift 2
                ;;
            -d|--device)
                device="$2"
                shift 2
                ;;
            -q|--quick)
                quick_mode="true"
                shift
                ;;
            -c|--config)
                config_file="$2"
                shift 2
                ;;
            -v|--verbose)
                verbose="true"
                shift
                ;;
            -j|--json)
                enable_json="true"
                shift
                ;;
            -p|--plot)
                enable_plot="true"
                shift
                ;;
            -r|--report)
                enable_report="true"
                shift
                ;;
            --docker)
                run_docker="true"
                shift
                ;;
            --gpu-mem)
                monitor_gpu="true"
                shift
                ;;
            --cpu-profile)
                profile_cpu="true"
                shift
                ;;
            all|basic|transformer|optimization|quick|custom)
                test_type="$1"
                shift
                ;;
            *)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Docker 模式处理
    if [[ "$run_docker" == "true" ]]; then
        # 重新构建参数（排除 --docker）
        local docker_args=""
        docker_args="$test_type"
        [[ "$output_dir" != "$RESULTS_DIR" ]] && docker_args="$docker_args -o $output_dir"
        [[ "$device" != "auto" ]] && docker_args="$docker_args -d $device"
        [[ "$quick_mode" == "true" ]] && docker_args="$docker_args -q"
        [[ -n "$config_file" ]] && docker_args="$docker_args -c $config_file"
        [[ "$verbose" == "true" ]] && docker_args="$docker_args -v"
        [[ "$enable_json" == "true" ]] && docker_args="$docker_args -j"
        [[ "$enable_plot" == "true" ]] && docker_args="$docker_args -p"
        [[ "$enable_report" == "true" ]] && docker_args="$docker_args -r"
        [[ "$monitor_gpu" == "true" ]] && docker_args="$docker_args --gpu-mem"
        [[ "$profile_cpu" == "true" ]] && docker_args="$docker_args --cpu-profile"
        
        run_in_docker $docker_args
        return
    fi
    
    log_info "🎯 YICA-Mirage 基准测试开始"
    log_info "配置: 测试类型=$test_type, 设备=$device, 输出=$output_dir"
    
    # 检查环境
    check_dependencies
    
    # 设置输出目录
    setup_output_directory "$output_dir"
    
    # 启动监控（如果需要）
    if [[ "$monitor_gpu" == "true" ]]; then
        monitor_gpu_memory "$output_dir"
    fi
    
    if [[ "$profile_cpu" == "true" ]]; then
        run_cpu_profiling "$output_dir"
    fi
    
    # 运行基准测试
    run_benchmark "$test_type" "$output_dir" "$device" "$config_file" "$quick_mode" "$verbose"
    
    # 生成额外报告
    if [[ "$enable_report" == "true" ]]; then
        generate_comparison_report "$output_dir"
    fi
    
    # 生成最终总结
    generate_final_report "$output_dir"
    
    log_success "🎉 所有基准测试完成！"
    log_success "📁 结果保存在: $output_dir"
    
    # 显示快速访问信息
    echo ""
    echo "📊 快速访问:"
    echo "  - 查看图表: ls $output_dir/charts/"
    echo "  - 阅读报告: ls $output_dir/reports/"
    echo "  - 原始数据: ls $output_dir/raw_data/"
    echo ""
}

# 运行主函数
main "$@" 
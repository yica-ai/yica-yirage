#!/bin/bash
# YICA-Mirage Docker 容器启动脚本

set -e

echo "🚀 启动 YICA-Mirage 生产环境容器..."

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# 检查 YICA 硬件
check_yica_hardware() {
    log_info "检查 YICA 硬件..."
    
    if [ -f "/proc/yica/status" ]; then
        log_info "✅ YICA 硬件检测成功"
        cat /proc/yica/status
    else
        log_warn "⚠️  YICA 硬件未检测到，将使用模拟模式"
        export YICA_SIMULATION_MODE=true
    fi
}

# 检查 CUDA 环境
check_cuda_environment() {
    log_info "检查 CUDA 环境..."
    
    if command -v nvidia-smi &> /dev/null; then
        log_info "✅ NVIDIA GPU 检测成功"
        nvidia-smi --query-gpu=gpu_name,memory.total,memory.used --format=csv,noheader,nounits
    else
        log_warn "⚠️  NVIDIA GPU 未检测到"
    fi
    
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        log_info "✅ CUDA 版本: $CUDA_VERSION"
    else
        log_error "❌ CUDA 编译器未找到"
    fi
}

# 初始化 YICA 环境
initialize_yica() {
    log_info "初始化 YICA 环境..."
    
    # 设置 YICA 配置
    export YICA_CONFIG_FILE="/etc/yica-mirage/yica-mirage.conf"
    export YICA_LOG_LEVEL=${YICA_LOG_LEVEL:-INFO}
    export YICA_NUM_DEVICES=${YICA_NUM_DEVICES:-1}
    export YICA_MEMORY_POOL_SIZE=${YICA_MEMORY_POOL_SIZE:-"16GB"}
    
    # 初始化 YICA 驱动
    if [ -x "/usr/local/bin/yica-init.sh" ]; then
        log_info "执行 YICA 初始化脚本..."
        /usr/local/bin/yica-init.sh
    fi
    
    # 验证 Mirage 安装
    if python -c "import mirage; print(f'Mirage 版本: {mirage.__version__}')" 2>/dev/null; then
        log_info "✅ Mirage 安装验证成功"
    else
        log_error "❌ Mirage 安装验证失败"
        exit 1
    fi
}

# 启动性能监控
start_performance_monitoring() {
    if [ "${ENABLE_MONITORING:-true}" = "true" ]; then
        log_info "启动性能监控..."
        
        # 启动性能监控守护进程
        if [ -x "/usr/local/bin/performance-monitor.sh" ]; then
            /usr/local/bin/performance-monitor.sh &
            MONITOR_PID=$!
            echo $MONITOR_PID > /tmp/performance-monitor.pid
            log_info "✅ 性能监控已启动 (PID: $MONITOR_PID)"
        fi
        
        # 启动 Web 仪表板
        if [ "${ENABLE_DASHBOARD:-false}" = "true" ]; then
            log_info "启动 Web 仪表板..."
            cd /workspace && python -m mirage.yica_performance_monitor_demo --port=8080 &
            DASHBOARD_PID=$!
            echo $DASHBOARD_PID > /tmp/dashboard.pid
            log_info "✅ Web 仪表板已启动 (PID: $DASHBOARD_PID) - http://localhost:8080"
        fi
    fi
}

# 设置开发环境
setup_development_environment() {
    if [ "${DEVELOPMENT_MODE:-false}" = "true" ]; then
        log_info "设置开发环境..."
        
        # 启动 Jupyter Lab
        if [ "${ENABLE_JUPYTER:-true}" = "true" ]; then
            log_info "启动 Jupyter Lab..."
            jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root &
            JUPYTER_PID=$!
            echo $JUPYTER_PID > /tmp/jupyter.pid
            log_info "✅ Jupyter Lab 已启动 (PID: $JUPYTER_PID) - http://localhost:8888"
        fi
        
        # 启动 TensorBoard
        if [ "${ENABLE_TENSORBOARD:-false}" = "true" ]; then
            log_info "启动 TensorBoard..."
            tensorboard --logdir=/workspace/logs --port=6006 --host=0.0.0.0 &
            TB_PID=$!
            echo $TB_PID > /tmp/tensorboard.pid
            log_info "✅ TensorBoard 已启动 (PID: $TB_PID) - http://localhost:6006"
        fi
    fi
}

# 运行预热测试
run_warmup_tests() {
    if [ "${RUN_WARMUP:-true}" = "true" ]; then
        log_info "运行系统预热测试..."
        
        # 简单的 YICA 功能测试
        python -c "
import sys
sys.path.append('/opt/mirage/python')
try:
    from mirage.yica.config import YICAConfig
    from mirage.python.mirage.yica_llama_optimizer import YICALlamaOptimizer
    
    config = YICAConfig()
    print(f'✅ YICA 配置加载成功: {config.num_cim_arrays} CIM 阵列')
    
    # 简单的性能测试
    import torch
    if torch.cuda.is_available():
        device = torch.device('cuda')
        x = torch.randn(1024, 1024, device=device)
        y = torch.matmul(x, x.T)
        print(f'✅ CUDA 矩阵乘法测试成功: {y.shape}')
    
    print('✅ 系统预热测试完成')
    
except Exception as e:
    print(f'❌ 预热测试失败: {e}')
    sys.exit(1)
"
        
        if [ $? -eq 0 ]; then
            log_info "✅ 系统预热测试通过"
        else
            log_error "❌ 系统预热测试失败"
            exit 1
        fi
    fi
}

# 清理函数
cleanup() {
    log_info "正在清理资源..."
    
    # 停止所有后台进程
    if [ -f /tmp/performance-monitor.pid ]; then
        MONITOR_PID=$(cat /tmp/performance-monitor.pid)
        if kill -0 $MONITOR_PID 2>/dev/null; then
            log_info "停止性能监控 (PID: $MONITOR_PID)"
            kill $MONITOR_PID
        fi
        rm -f /tmp/performance-monitor.pid
    fi
    
    if [ -f /tmp/dashboard.pid ]; then
        DASHBOARD_PID=$(cat /tmp/dashboard.pid)
        if kill -0 $DASHBOARD_PID 2>/dev/null; then
            log_info "停止 Web 仪表板 (PID: $DASHBOARD_PID)"
            kill $DASHBOARD_PID
        fi
        rm -f /tmp/dashboard.pid
    fi
    
    if [ -f /tmp/jupyter.pid ]; then
        JUPYTER_PID=$(cat /tmp/jupyter.pid)
        if kill -0 $JUPYTER_PID 2>/dev/null; then
            log_info "停止 Jupyter Lab (PID: $JUPYTER_PID)"
            kill $JUPYTER_PID
        fi
        rm -f /tmp/jupyter.pid
    fi
    
    if [ -f /tmp/tensorboard.pid ]; then
        TB_PID=$(cat /tmp/tensorboard.pid)
        if kill -0 $TB_PID 2>/dev/null; then
            log_info "停止 TensorBoard (PID: $TB_PID)"
            kill $TB_PID
        fi
        rm -f /tmp/tensorboard.pid
    fi
    
    log_info "✅ 资源清理完成"
}

# 信号处理
trap cleanup SIGTERM SIGINT

# 主程序
main() {
    log_info "🌟 YICA-Mirage 生产环境启动"
    log_info "容器 ID: $(hostname)"
    log_info "启动时间: $(date)"
    
    # 执行初始化步骤
    check_yica_hardware
    check_cuda_environment
    initialize_yica
    start_performance_monitoring
    setup_development_environment
    run_warmup_tests
    
    log_info "🎉 YICA-Mirage 环境启动完成!"
    
    # 显示环境信息
    echo ""
    log_info "=== 环境信息 ==="
    log_info "YICA 主目录: $YICA_HOME"
    log_info "Mirage 主目录: $MIRAGE_HOME"
    log_info "工作目录: /workspace"
    log_info "Python 路径: $PYTHONPATH"
    
    echo ""
    log_info "=== 可用服务 ==="
    if [ "${ENABLE_DASHBOARD:-false}" = "true" ]; then
        log_info "🌐 Web 仪表板: http://localhost:8080"
    fi
    if [ "${ENABLE_JUPYTER:-true}" = "true" ] && [ "${DEVELOPMENT_MODE:-false}" = "true" ]; then
        log_info "📓 Jupyter Lab: http://localhost:8888"
    fi
    if [ "${ENABLE_TENSORBOARD:-false}" = "true" ]; then
        log_info "📊 TensorBoard: http://localhost:6006"
    fi
    
    echo ""
    log_info "=== 使用示例 ==="
    log_info "# 运行 YICA 基准测试"
    log_info "python /opt/mirage/python/mirage/demo_yica_comprehensive_benchmark.py"
    log_info ""
    log_info "# 启动分布式训练"
    log_info "python /opt/mirage/python/mirage/demo_yica_distributed_training.py --mode=train"
    log_info ""
    log_info "# 性能监控演示"
    log_info "python /opt/mirage/python/mirage/yica_performance_monitor_demo.py"
    
    echo ""
    log_info "🚀 容器已就绪，请开始使用 YICA-Mirage!"
    
    # 执行传入的命令或启动交互式 shell
    if [ $# -eq 0 ]; then
        # 如果没有传入命令，启动交互式 bash
        exec bash
    else
        # 执行传入的命令
        exec "$@"
    fi
}

# 执行主程序
main "$@" 
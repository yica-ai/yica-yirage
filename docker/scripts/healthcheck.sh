#!/bin/bash
# YICA-Mirage Docker 容器健康检查脚本

set -e

# 返回码定义
EXIT_SUCCESS=0
EXIT_FAILURE=1

# 日志函数
log_info() {
    echo "[HEALTHCHECK] $(date '+%Y-%m-%d %H:%M:%S') INFO: $1"
}

log_error() {
    echo "[HEALTHCHECK] $(date '+%Y-%m-%d %H:%M:%S') ERROR: $1" >&2
}

# 检查基础系统
check_basic_system() {
    # 检查基础命令可用性
    if ! command -v python3 >/dev/null; then
        log_error "Python3 不可用"
        return $EXIT_FAILURE
    fi
    
    # 检查内存使用率
    if command -v free >/dev/null; then
        local mem_usage=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')
        if [ "$mem_usage" -gt 95 ]; then
            log_error "内存使用率过高: ${mem_usage}%"
            return $EXIT_FAILURE
        fi
    fi
    
    return $EXIT_SUCCESS
}

# 检查 YICA 环境
check_yica_environment() {
    # 检查 YICA 状态文件
    if [ ! -f "/proc/yica/status" ]; then
        log_error "YICA 状态文件不存在"
        return $EXIT_FAILURE
    fi
    
    # 检查 YICA 设备
    local device_count=0
    for device in /dev/yica*; do
        if [ -e "$device" ]; then
            device_count=$((device_count + 1))
        fi
    done
    
    if [ "$device_count" -eq 0 ]; then
        log_error "未找到 YICA 设备"
        return $EXIT_FAILURE
    fi
    
    return $EXIT_SUCCESS
}

# 检查 Mirage 库
check_mirage_library() {
    # 测试 Mirage 导入
    if ! python3 -c "import mirage; print('Mirage version:', mirage.__version__)" 2>/dev/null; then
        log_error "Mirage 库导入失败"
        return $EXIT_FAILURE
    fi
    
    # 测试 YICA 后端
    if ! python3 -c "
import sys
sys.path.append('/opt/mirage/python')
try:
    from mirage.yica.config import YICAConfig
    config = YICAConfig()
    print('YICA config loaded successfully')
except Exception as e:
    print(f'YICA config error: {e}')
    sys.exit(1)
" 2>/dev/null; then
        log_error "YICA 配置加载失败"
        return $EXIT_FAILURE
    fi
    
    return $EXIT_SUCCESS
}

# 检查性能监控
check_performance_monitoring() {
    # 检查性能监控进程
    if [ -f "/tmp/performance-monitor.pid" ]; then
        local monitor_pid=$(cat /tmp/performance-monitor.pid)
        if ! kill -0 "$monitor_pid" 2>/dev/null; then
            log_error "性能监控进程不存在 (PID: $monitor_pid)"
            return $EXIT_FAILURE
        fi
    fi
    
    # 检查日志文件
    if [ ! -f "/workspace/logs/yica_performance.log" ]; then
        log_error "性能监控日志文件不存在"
        return $EXIT_FAILURE
    fi
    
    # 检查最近的日志活动（5分钟内）
    local log_file="/workspace/logs/yica_performance.log"
    if [ -f "$log_file" ]; then
        local last_log_time=$(tail -n 1 "$log_file" | awk '{print $1, $2}')
        if [ -n "$last_log_time" ]; then
            local current_time=$(date +%s)
            local log_time=$(date -d "$last_log_time" +%s 2>/dev/null || echo 0)
            local time_diff=$((current_time - log_time))
            
            if [ "$time_diff" -gt 300 ]; then  # 5分钟
                log_error "性能监控日志过期 (最后更新: ${time_diff}秒前)"
                return $EXIT_FAILURE
            fi
        fi
    fi
    
    return $EXIT_SUCCESS
}

# 检查 Web 服务
check_web_services() {
    # 检查 Web 仪表板（如果启用）
    if [ "${ENABLE_DASHBOARD:-false}" = "true" ]; then
        if [ -f "/tmp/dashboard.pid" ]; then
            local dashboard_pid=$(cat /tmp/dashboard.pid)
            if ! kill -0 "$dashboard_pid" 2>/dev/null; then
                log_error "Web 仪表板进程不存在 (PID: $dashboard_pid)"
                return $EXIT_FAILURE
            fi
            
            # 测试 HTTP 连接
            if command -v curl >/dev/null; then
                if ! curl -f -s "http://localhost:8080/health" >/dev/null 2>&1; then
                    log_error "Web 仪表板健康检查失败"
                    return $EXIT_FAILURE
                fi
            fi
        fi
    fi
    
    # 检查 Jupyter Lab（如果启用）
    if [ "${ENABLE_JUPYTER:-true}" = "true" ] && [ "${DEVELOPMENT_MODE:-false}" = "true" ]; then
        if [ -f "/tmp/jupyter.pid" ]; then
            local jupyter_pid=$(cat /tmp/jupyter.pid)
            if ! kill -0 "$jupyter_pid" 2>/dev/null; then
                log_error "Jupyter Lab 进程不存在 (PID: $jupyter_pid)"
                return $EXIT_FAILURE
            fi
        fi
    fi
    
    return $EXIT_SUCCESS
}

# 检查 CUDA 环境
check_cuda_environment() {
    # 检查 NVIDIA 驱动
    if command -v nvidia-smi >/dev/null; then
        if ! nvidia-smi >/dev/null 2>&1; then
            log_error "NVIDIA GPU 不可用"
            return $EXIT_FAILURE
        fi
        
        # 检查 GPU 温度
        local gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -n 1)
        if [ -n "$gpu_temp" ] && [ "$gpu_temp" -gt 85 ]; then
            log_error "GPU 温度过高: ${gpu_temp}°C"
            return $EXIT_FAILURE
        fi
    fi
    
    return $EXIT_SUCCESS
}

# 执行简单的功能测试
run_functional_test() {
    # 执行 YICA 功能测试
    if ! python3 -c "
import sys
import numpy as np
sys.path.append('/opt/mirage/python')

try:
    # 测试基础数值计算
    a = np.random.randn(100, 100)
    b = np.random.randn(100, 100)
    c = np.matmul(a, b)
    assert c.shape == (100, 100), 'Matrix multiplication failed'
    
    # 测试 YICA 配置
    from mirage.yica.config import YICAConfig
    config = YICAConfig()
    assert config.num_cim_arrays > 0, 'Invalid CIM array count'
    
    print('Functional test passed')
    
except Exception as e:
    print(f'Functional test failed: {e}')
    sys.exit(1)
" 2>/dev/null; then
        log_error "功能测试失败"
        return $EXIT_FAILURE
    fi
    
    return $EXIT_SUCCESS
}

# 主健康检查函数
main() {
    log_info "开始容器健康检查..."
    
    local overall_status=$EXIT_SUCCESS
    
    # 执行各项检查
    if ! check_basic_system; then
        log_error "基础系统检查失败"
        overall_status=$EXIT_FAILURE
    fi
    
    if ! check_yica_environment; then
        log_error "YICA 环境检查失败"
        overall_status=$EXIT_FAILURE
    fi
    
    if ! check_mirage_library; then
        log_error "Mirage 库检查失败"
        overall_status=$EXIT_FAILURE
    fi
    
    if ! check_performance_monitoring; then
        log_error "性能监控检查失败"
        overall_status=$EXIT_FAILURE
    fi
    
    if ! check_web_services; then
        log_error "Web 服务检查失败"
        overall_status=$EXIT_FAILURE
    fi
    
    if ! check_cuda_environment; then
        log_error "CUDA 环境检查失败"
        overall_status=$EXIT_FAILURE
    fi
    
    if ! run_functional_test; then
        log_error "功能测试失败"
        overall_status=$EXIT_FAILURE
    fi
    
    # 输出结果
    if [ $overall_status -eq $EXIT_SUCCESS ]; then
        log_info "✅ 容器健康检查通过"
        
        # 输出状态摘要
        echo "=== 容器状态摘要 ==="
        echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "主机名: $(hostname)"
        echo "运行时间: $(uptime)"
        
        # YICA 状态
        if [ -f "/proc/yica/status" ]; then
            echo "YICA 状态: $(grep "Status:" /proc/yica/status | awk '{print $2}')"
        fi
        
        # 内存使用
        if command -v free >/dev/null; then
            echo "内存使用: $(free -h | awk 'NR==2{printf "%s/%s (%.1f%%)", $3,$2,$3*100/$2}')"
        fi
        
        # GPU 状态
        if command -v nvidia-smi >/dev/null; then
            echo "GPU 状态: $(nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -n 1)"
        fi
        
        exit $EXIT_SUCCESS
    else
        log_error "❌ 容器健康检查失败"
        exit $EXIT_FAILURE
    fi
}

# 执行主函数
main "$@" 
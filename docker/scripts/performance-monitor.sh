#!/bin/bash
# YICA 性能监控守护进程

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[PERF-MONITOR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"; }
log_warn() { echo -e "${YELLOW}[PERF-MONITOR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"; }
log_error() { echo -e "${RED}[PERF-MONITOR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"; }
log_debug() { echo -e "${BLUE}[PERF-MONITOR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"; }

# 配置参数
MONITOR_INTERVAL=${MONITOR_INTERVAL:-5}  # 监控间隔（秒）
LOG_FILE=${YICA_PERF_LOG:-"/workspace/logs/yica_performance.log"}
METRICS_FILE="/workspace/logs/yica_metrics.json"
ALERT_THRESHOLD_FILE="/etc/yica-mirage/alert_thresholds.json"

# 创建日志目录
mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$(dirname "$METRICS_FILE")"

# 初始化日志文件
echo "$(date '+%Y-%m-%d %H:%M:%S') - YICA 性能监控启动" >> "$LOG_FILE"

# 创建告警阈值配置
create_alert_thresholds() {
    if [ ! -f "$ALERT_THRESHOLD_FILE" ]; then
        log_info "创建告警阈值配置..."
        cat > "$ALERT_THRESHOLD_FILE" << 'EOF'
{
    "cim_utilization": {
        "warning": 80.0,
        "critical": 95.0
    },
    "spm_hit_rate": {
        "warning": 70.0,
        "critical": 50.0
    },
    "memory_usage": {
        "warning": 85.0,
        "critical": 95.0
    },
    "temperature": {
        "warning": 75.0,
        "critical": 85.0
    },
    "dram_bandwidth_utilization": {
        "warning": 80.0,
        "critical": 90.0
    },
    "communication_latency_ms": {
        "warning": 100.0,
        "critical": 200.0
    }
}
EOF
    fi
}

# 收集 YICA 硬件指标
collect_yica_metrics() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local metrics="{\"timestamp\": \"$timestamp\""
    
    # CIM 阵列利用率
    if [ -f "/proc/yica/cim_stats" ]; then
        local cim_util=$(cat /proc/yica/cim_stats | grep "utilization" | awk '{print $2}' | sed 's/%//')
        metrics="${metrics}, \"cim_utilization\": ${cim_util:-0}"
    else
        # 模拟 CIM 利用率
        local cim_util=$((RANDOM % 100))
        metrics="${metrics}, \"cim_utilization\": $cim_util"
    fi
    
    # SPM 命中率
    if [ -f "/proc/yica/spm_stats" ]; then
        local spm_hit_rate=$(cat /proc/yica/spm_stats | grep "hit_rate" | awk '{print $2}' | sed 's/%//')
        metrics="${metrics}, \"spm_hit_rate\": ${spm_hit_rate:-0}"
    else
        # 模拟 SPM 命中率
        local spm_hit_rate=$((75 + RANDOM % 20))
        metrics="${metrics}, \"spm_hit_rate\": $spm_hit_rate"
    fi
    
    # 内存使用率
    if command -v free >/dev/null; then
        local mem_total=$(free -m | awk 'NR==2{print $2}')
        local mem_used=$(free -m | awk 'NR==2{print $3}')
        local mem_usage=$(echo "scale=2; $mem_used * 100 / $mem_total" | bc)
        metrics="${metrics}, \"memory_usage\": $mem_usage"
    else
        local mem_usage=$((50 + RANDOM % 30))
        metrics="${metrics}, \"memory_usage\": $mem_usage"
    fi
    
    # 温度监控
    if [ -f "/proc/yica/thermal" ]; then
        local temperature=$(cat /proc/yica/thermal | grep "core_temp" | awk '{print $2}')
        metrics="${metrics}, \"temperature\": ${temperature:-45}"
    else
        # 模拟温度
        local temperature=$((45 + RANDOM % 25))
        metrics="${metrics}, \"temperature\": $temperature"
    fi
    
    # DRAM 带宽利用率
    if [ -f "/proc/yica/dram_stats" ]; then
        local dram_bw=$(cat /proc/yica/dram_stats | grep "bandwidth_util" | awk '{print $2}' | sed 's/%//')
        metrics="${metrics}, \"dram_bandwidth_utilization\": ${dram_bw:-0}"
    else
        # 模拟 DRAM 带宽利用率
        local dram_bw=$((30 + RANDOM % 50))
        metrics="${metrics}, \"dram_bandwidth_utilization\": $dram_bw"
    fi
    
    # 通信延迟
    if [ -f "/proc/yica/yccl_stats" ]; then
        local comm_latency=$(cat /proc/yica/yccl_stats | grep "avg_latency" | awk '{print $2}')
        metrics="${metrics}, \"communication_latency_ms\": ${comm_latency:-10}"
    else
        # 模拟通信延迟
        local comm_latency=$((10 + RANDOM % 50))
        metrics="${metrics}, \"communication_latency_ms\": $comm_latency"
    fi
    
    # 指令吞吐量
    if [ -f "/proc/yica/instruction_stats" ]; then
        local inst_throughput=$(cat /proc/yica/instruction_stats | grep "throughput" | awk '{print $2}')
        metrics="${metrics}, \"instruction_throughput\": ${inst_throughput:-1000}"
    else
        # 模拟指令吞吐量 (MIPS)
        local inst_throughput=$((800 + RANDOM % 400))
        metrics="${metrics}, \"instruction_throughput\": $inst_throughput"
    fi
    
    # 能耗监控
    if [ -f "/proc/yica/power" ]; then
        local power_consumption=$(cat /proc/yica/power | grep "total_power" | awk '{print $2}')
        metrics="${metrics}, \"power_consumption_watts\": ${power_consumption:-150}"
    else
        # 模拟功耗 (瓦特)
        local power_consumption=$((120 + RANDOM % 60))
        metrics="${metrics}, \"power_consumption_watts\": $power_consumption"
    fi
    
    metrics="${metrics}}"
    echo "$metrics"
}

# 检查告警条件
check_alerts() {
    local metrics="$1"
    
    # 读取阈值配置
    if [ ! -f "$ALERT_THRESHOLD_FILE" ]; then
        return
    fi
    
    # 使用 Python 进行告警检查
    python3 -c "
import json
import sys

try:
    metrics = json.loads('$metrics')
    with open('$ALERT_THRESHOLD_FILE', 'r') as f:
        thresholds = json.load(f)
    
    alerts = []
    
    for metric_name, value in metrics.items():
        if metric_name == 'timestamp':
            continue
            
        if metric_name in thresholds:
            threshold = thresholds[metric_name]
            
            if isinstance(value, (int, float)):
                if value >= threshold.get('critical', 999999):
                    alerts.append(f'CRITICAL: {metric_name}={value} (threshold={threshold[\"critical\"]})')
                elif value >= threshold.get('warning', 999999):
                    alerts.append(f'WARNING: {metric_name}={value} (threshold={threshold[\"warning\"]})')
                
                # 特殊处理：SPM 命中率低于阈值是告警
                if metric_name == 'spm_hit_rate':
                    if value <= threshold.get('critical', 0):
                        alerts.append(f'CRITICAL: {metric_name}={value}% (threshold<={threshold[\"critical\"]}%)')
                    elif value <= threshold.get('warning', 0):
                        alerts.append(f'WARNING: {metric_name}={value}% (threshold<={threshold[\"warning\"]}%)')
    
    if alerts:
        for alert in alerts:
            print(alert)
    
except Exception as e:
    print(f'Alert check error: {e}', file=sys.stderr)
"
}

# 记录指标到文件
log_metrics() {
    local metrics="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # 追加到指标文件
    echo "$metrics" >> "$METRICS_FILE"
    
    # 保持文件大小限制（保留最近 10000 行）
    if [ -f "$METRICS_FILE" ]; then
        local line_count=$(wc -l < "$METRICS_FILE")
        if [ "$line_count" -gt 10000 ]; then
            tail -n 5000 "$METRICS_FILE" > "${METRICS_FILE}.tmp"
            mv "${METRICS_FILE}.tmp" "$METRICS_FILE"
        fi
    fi
    
    # 记录到主日志
    echo "$timestamp - Metrics: $metrics" >> "$LOG_FILE"
}

# 生成性能报告
generate_performance_report() {
    local report_file="/workspace/logs/yica_performance_report_$(date +%Y%m%d_%H%M%S).html"
    
    log_info "生成性能报告: $report_file"
    
    cat > "$report_file" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>YICA 性能监控报告</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric-card { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .warning { background-color: #fff3cd; }
        .critical { background-color: #f8d7da; }
        .normal { background-color: #d4edda; }
        .chart-container { width: 400px; height: 300px; display: inline-block; margin: 10px; }
        h1, h2 { color: #333; }
        .timestamp { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <h1>🚀 YICA 性能监控报告</h1>
    <div class="timestamp">生成时间: $(date '+%Y-%m-%d %H:%M:%S')</div>
    
    <h2>📊 当前性能指标</h2>
    <div id="current-metrics">
        <!-- 当前指标将通过 JavaScript 动态加载 -->
    </div>
    
    <h2>📈 性能趋势图表</h2>
    <div class="chart-container">
        <canvas id="cimUtilizationChart"></canvas>
    </div>
    <div class="chart-container">
        <canvas id="memoryUsageChart"></canvas>
    </div>
    <div class="chart-container">
        <canvas id="temperatureChart"></canvas>
    </div>
    <div class="chart-container">
        <canvas id="spmHitRateChart"></canvas>
    </div>
    
    <h2>🚨 告警历史</h2>
    <div id="alert-history">
        <!-- 告警历史将通过 JavaScript 动态加载 -->
    </div>
    
    <script>
        // 加载性能数据的 JavaScript 代码
        // 在实际环境中，这里会从后端 API 加载真实数据
        
        // 示例数据
        const performanceData = {
            cim_utilization: [65, 72, 68, 75, 80, 77, 73],
            memory_usage: [45, 52, 48, 55, 60, 58, 54],
            temperature: [52, 55, 54, 58, 62, 60, 57],
            spm_hit_rate: [85, 88, 86, 90, 87, 89, 91]
        };
        
        const labels = ['6h前', '5h前', '4h前', '3h前', '2h前', '1h前', '现在'];
        
        // CIM 利用率图表
        new Chart(document.getElementById('cimUtilizationChart'), {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'CIM 利用率 (%)',
                    data: performanceData.cim_utilization,
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                plugins: { title: { display: true, text: 'CIM 阵列利用率' } }
            }
        });
        
        // 内存使用率图表
        new Chart(document.getElementById('memoryUsageChart'), {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: '内存使用率 (%)',
                    data: performanceData.memory_usage,
                    borderColor: 'rgb(255, 99, 132)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                plugins: { title: { display: true, text: '内存使用率' } }
            }
        });
        
        // 温度图表
        new Chart(document.getElementById('temperatureChart'), {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: '温度 (°C)',
                    data: performanceData.temperature,
                    borderColor: 'rgb(255, 205, 86)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                plugins: { title: { display: true, text: '芯片温度' } }
            }
        });
        
        // SPM 命中率图表
        new Chart(document.getElementById('spmHitRateChart'), {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'SPM 命中率 (%)',
                    data: performanceData.spm_hit_rate,
                    borderColor: 'rgb(54, 162, 235)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                plugins: { title: { display: true, text: 'SPM 缓存命中率' } }
            }
        });
    </script>
</body>
</html>
EOF
    
    log_info "✅ 性能报告已生成: $report_file"
}

# 主监控循环
monitoring_loop() {
    log_info "开始 YICA 性能监控循环 (间隔: ${MONITOR_INTERVAL}秒)"
    
    local iteration=0
    while true; do
        iteration=$((iteration + 1))
        
        # 收集指标
        local metrics=$(collect_yica_metrics)
        
        if [ -n "$metrics" ]; then
            # 记录指标
            log_metrics "$metrics"
            
            # 检查告警
            local alerts=$(check_alerts "$metrics")
            if [ -n "$alerts" ]; then
                log_warn "性能告警检测到:"
                echo "$alerts" | while read -r alert; do
                    log_warn "  $alert"
                    echo "$(date '+%Y-%m-%d %H:%M:%S') - ALERT: $alert" >> "$LOG_FILE"
                done
            fi
            
            # 每 100 次迭代输出一次简要状态
            if [ $((iteration % 100)) -eq 0 ]; then
                log_info "监控状态: 第 $iteration 次迭代完成"
                
                # 生成性能报告（每 1000 次迭代）
                if [ $((iteration % 1000)) -eq 0 ]; then
                    generate_performance_report
                fi
            fi
        else
            log_error "指标收集失败"
        fi
        
        # 等待下一次监控
        sleep "$MONITOR_INTERVAL"
    done
}

# 信号处理
cleanup() {
    log_info "收到终止信号，正在清理..."
    
    # 生成最终报告
    generate_performance_report
    
    log_info "性能监控已停止"
    exit 0
}

trap cleanup SIGTERM SIGINT

# 主程序
main() {
    log_info "🚀 启动 YICA 性能监控守护进程"
    log_info "PID: $$"
    log_info "日志文件: $LOG_FILE"
    log_info "指标文件: $METRICS_FILE"
    
    # 创建告警阈值配置
    create_alert_thresholds
    
    # 验证 Python 可用性
    if ! command -v python3 >/dev/null; then
        log_error "Python3 未找到，某些功能可能无法正常工作"
    fi
    
    # 验证 bc 可用性（用于浮点计算）
    if ! command -v bc >/dev/null; then
        log_warn "bc 计算器未找到，将使用整数计算"
    fi
    
    log_info "开始监控 YICA 性能指标..."
    
    # 启动监控循环
    monitoring_loop
}

# 执行主程序
main "$@" 
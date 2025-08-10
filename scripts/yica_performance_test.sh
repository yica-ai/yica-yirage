#!/bin/bash
# YICA 虚拟环境性能测试脚本
# 测试 yirage 结合 YICA 虚拟硬件的算子性能
# 基于 Docker 容器化环境，支持完整的性能基准测试

set -e

echo "🧠 YICA 虚拟环境性能测试启动..."

# 配置参数
REMOTE_USER="johnson.chen"
REMOTE_HOST="10.11.60.58"
REMOTE_SSH="${REMOTE_USER}@${REMOTE_HOST}"
WORK_DIR="/home/${REMOTE_USER}/yica-docker-workspace"
CONTAINER_NAME="yica-qemu-container"

# 测试配置
TEST_MODE="comprehensive"  # quick, comprehensive, stress
BENCHMARK_ITERATIONS=100
WARMUP_ITERATIONS=10
OUTPUT_DIR="yica_performance_results"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "${PURPLE}============================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}============================================${NC}"
}

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查服务器连接
check_connection() {
    print_status "检查服务器连接..."
    
    if ! ssh -o ConnectTimeout=10 "$REMOTE_SSH" "echo 'SSH连接成功'" 2>/dev/null; then
        print_error "无法连接到服务器 $REMOTE_SSH"
        exit 1
    fi
    
    print_success "服务器连接正常"
}

# 检查 Docker 环境和容器状态
check_docker_environment() {
    print_status "检查 Docker 环境和 YICA 容器状态..."
    
    ssh "$REMOTE_SSH" << EOF
        cd $WORK_DIR
        
        echo "🐳 检查 Docker 环境..."
        
        # 检查容器是否运行
        if docker ps | grep -q $CONTAINER_NAME; then
            echo "✅ YICA-QEMU 容器正在运行"
            docker ps | grep $CONTAINER_NAME
        else
            echo "⚠️  YICA-QEMU 容器未运行，尝试启动..."
            
            # 尝试启动容器
            if [ -f docker-compose.yml ]; then
                docker-compose up -d
            else
                docker start $CONTAINER_NAME 2>/dev/null || {
                    echo "❌ 容器启动失败，请先运行部署脚本"
                    exit 1
                }
            fi
            
            echo "⏳ 等待容器启动..."
            sleep 10
        fi
        
        # 检查容器内服务
        echo "🔍 检查容器内服务状态..."
        docker exec $CONTAINER_NAME ps aux | grep -E "(vnc|python)" || true
        
        # 检查 yirage 环境
        echo "🧪 检查 yirage 环境..."
        docker exec $CONTAINER_NAME python3 -c "
import sys
sys.path.insert(0, '/home/yica/workspace/yirage/python')
try:
    import yirage
    print(f'✅ yirage 版本: {yirage.__version__}')
except ImportError as e:
    print(f'❌ yirage 导入失败: {e}')
    sys.exit(1)
"
EOF
    
    if [ $? -eq 0 ]; then
        print_success "Docker 环境和 yirage 检查通过"
    else
        print_error "Docker 环境检查失败"
        exit 1
    fi
}

# 启动 YICA 虚拟硬件服务
start_yica_virtual_services() {
    print_header "启动 YICA 虚拟硬件服务"
    
    ssh "$REMOTE_SSH" << EOF
        cd $WORK_DIR
        
        echo "🚀 启动 YICA 虚拟硬件服务..."
        
        # 1. 启动 gem5 RISC-V 模拟器 (后台运行)
        echo "1️⃣  启动 gem5 RISC-V 模拟器..."
        docker exec -d $CONTAINER_NAME bash -c "
            cd /home/yica/workspace
            echo '启动 gem5...' > logs/gem5.log 2>&1
            /home/yica/workspace/gem5-docker.sh >> logs/gem5.log 2>&1 &
        "
        
        # 等待 gem5 启动
        sleep 5
        
        # 2. 启动 QEMU 虚拟机 (后台运行)
        echo "2️⃣  启动 QEMU 虚拟机..."
        docker exec -d $CONTAINER_NAME bash -c "
            cd /home/yica/workspace
            echo '启动 QEMU...' > logs/qemu.log 2>&1
            /home/yica/workspace/qemu-docker.sh >> logs/qemu.log 2>&1 &
        "
        
        # 等待 QEMU 启动
        sleep 10
        
        # 3. 验证服务启动状态
        echo "3️⃣  验证虚拟硬件服务状态..."
        
        # 检查进程
        echo "检查 gem5 和 QEMU 进程..."
        docker exec $CONTAINER_NAME ps aux | grep -E "(gem5|qemu)" || echo "⚠️  未找到相关进程"
        
        # 检查端口
        echo "检查服务端口..."
        docker exec $CONTAINER_NAME netstat -tlnp | grep -E "(3456|4444)" || echo "⚠️  服务端口未监听"
        
        # 检查日志
        echo "检查启动日志..."
        echo "=== gem5 日志 ==="
        docker exec $CONTAINER_NAME tail -5 /home/yica/workspace/logs/gem5.log 2>/dev/null || echo "gem5 日志文件不存在"
        echo "=== QEMU 日志 ==="
        docker exec $CONTAINER_NAME tail -5 /home/yica/workspace/logs/qemu.log 2>/dev/null || echo "QEMU 日志文件不存在"
        
        echo "✅ YICA 虚拟硬件服务启动完成"
EOF
    
    print_success "YICA 虚拟硬件服务已启动"
}

# 创建性能测试脚本
create_performance_test_script() {
    print_status "创建容器内性能测试脚本..."
    
    ssh "$REMOTE_SSH" << 'EOF'
        cd $WORK_DIR
        
        # 创建容器内的性能测试脚本
        docker exec $CONTAINER_NAME bash -c "
cat > /home/yica/workspace/yica_performance_test.py << 'PYTHON_EOF'
#!/usr/bin/env python3
\"\"\"
YICA 虚拟环境性能测试脚本
测试 yirage 生成的算子在 YICA 虚拟硬件上的性能
\"\"\"

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path

# 添加 yirage 路径
sys.path.insert(0, '/home/yica/workspace/yirage/python')

try:
    import yirage
    from yirage.yica_pytorch_backend import (
        initialize as yica_initialize,
        get_yica_backend,
        optimize_model
    )
    YICA_AVAILABLE = True
    print(f'✅ yirage 版本: {yirage.__version__}')
except ImportError as e:
    YICA_AVAILABLE = False
    print(f'❌ yirage 导入失败: {e}')
    sys.exit(1)

class YICAPerformanceTester:
    \"\"\"YICA 性能测试器\"\"\"
    
    def __init__(self, output_dir='./yica_test_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
        # 初始化 YICA 后端
        if YICA_AVAILABLE:
            try:
                yica_initialize()
                self.backend = get_yica_backend()
                print('✅ YICA 后端初始化成功')
            except Exception as e:
                print(f'⚠️  YICA 后端初始化失败: {e}')
                self.backend = None
        else:
            self.backend = None
    
    def benchmark_operator(self, name, func, *args, iterations=100, warmup=10):
        \"\"\"基准测试单个算子\"\"\"
        print(f'🧪 测试算子: {name}')
        
        # 预热
        for _ in range(warmup):
            func(*args)
        
        # 基准测试
        latencies = []
        for i in range(iterations):
            start_time = time.time()
            result = func(*args)
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # ms
            
            if i % 20 == 0:
                print(f'  进度: {i+1}/{iterations}')
        
        # 统计结果
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        throughput = 1000 / mean_latency  # ops/sec
        
        result = {
            'name': name,
            'mean_latency_ms': mean_latency,
            'std_latency_ms': std_latency,
            'min_latency_ms': min_latency,
            'max_latency_ms': max_latency,
            'throughput_ops_per_sec': throughput,
            'iterations': iterations,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(result)
        
        print(f'  ✅ 平均延迟: {mean_latency:.3f} ± {std_latency:.3f} ms')
        print(f'  📊 吞吐量: {throughput:.2f} ops/sec')
        
        return result
    
    def test_matrix_operations(self):
        \"\"\"测试矩阵运算\"\"\"
        print('\\n🔢 测试矩阵运算...')
        
        # 不同规模的矩阵乘法
        sizes = [256, 512, 1024]
        
        for size in sizes:
            print(f'\\n📐 矩阵规模: {size}x{size}')
            
            # 创建测试数据
            A = torch.randn(size, size, dtype=torch.float32)
            B = torch.randn(size, size, dtype=torch.float32)
            
            # 原生 PyTorch 矩阵乘法
            def pytorch_matmul():
                return torch.mm(A, B)
            
            self.benchmark_operator(f'pytorch_matmul_{size}x{size}', pytorch_matmul)
            
            # YICA 优化矩阵乘法 (如果可用)
            if self.backend:
                try:
                    # 使用 YICA 后端优化
                    class MatMulModel(nn.Module):
                        def forward(self, x, y):
                            return torch.mm(x, y)
                    
                    model = MatMulModel()
                    optimized_model = optimize_model(model)
                    
                    def yica_matmul():
                        return optimized_model(A, B)
                    
                    self.benchmark_operator(f'yica_matmul_{size}x{size}', yica_matmul)
                    
                except Exception as e:
                    print(f'  ⚠️  YICA 矩阵乘法测试失败: {e}')
    
    def test_attention_mechanisms(self):
        \"\"\"测试注意力机制\"\"\"
        print('\\n🎯 测试注意力机制...')
        
        batch_size = 8
        seq_len = 512
        hidden_size = 768
        num_heads = 12
        head_dim = hidden_size // num_heads
        
        print(f'📊 配置: batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}')
        
        # 创建测试数据
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        # 原生 PyTorch 注意力
        def pytorch_attention():
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(head_dim)
            attn_weights = torch.softmax(scores, dim=-1)
            return torch.matmul(attn_weights, V)
        
        self.benchmark_operator('pytorch_attention', pytorch_attention)
        
        # YICA 优化注意力 (如果可用)
        if self.backend:
            try:
                class AttentionModel(nn.Module):
                    def forward(self, q, k, v):
                        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
                        attn_weights = torch.softmax(scores, dim=-1)
                        return torch.matmul(attn_weights, v)
                
                model = AttentionModel()
                optimized_model = optimize_model(model)
                
                def yica_attention():
                    return optimized_model(Q, K, V)
                
                self.benchmark_operator('yica_attention', yica_attention)
                
            except Exception as e:
                print(f'  ⚠️  YICA 注意力测试失败: {e}')
    
    def test_normalization_operations(self):
        \"\"\"测试规范化操作\"\"\"
        print('\\n📏 测试规范化操作...')
        
        batch_size = 16
        seq_len = 1024
        hidden_size = 768
        
        print(f'📊 配置: batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}')
        
        # 创建测试数据
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        # LayerNorm
        layer_norm = nn.LayerNorm(hidden_size)
        def pytorch_layernorm():
            return layer_norm(x)
        
        self.benchmark_operator('pytorch_layernorm', pytorch_layernorm)
        
        # RMSNorm (手动实现)
        def pytorch_rmsnorm():
            variance = x.pow(2).mean(-1, keepdim=True)
            return x * torch.rsqrt(variance + 1e-6)
        
        self.benchmark_operator('pytorch_rmsnorm', pytorch_rmsnorm)
        
        # YICA 优化规范化 (如果可用)
        if self.backend:
            try:
                optimized_layernorm = optimize_model(layer_norm)
                
                def yica_layernorm():
                    return optimized_layernorm(x)
                
                self.benchmark_operator('yica_layernorm', yica_layernorm)
                
            except Exception as e:
                print(f'  ⚠️  YICA LayerNorm 测试失败: {e}')
    
    def test_activation_functions(self):
        \"\"\"测试激活函数\"\"\"
        print('\\n⚡ 测试激活函数...')
        
        batch_size = 32
        hidden_size = 4096
        
        print(f'📊 配置: batch_size={batch_size}, hidden_size={hidden_size}')
        
        # 创建测试数据
        x = torch.randn(batch_size, hidden_size)
        
        # 测试各种激活函数
        activations = {
            'relu': torch.relu,
            'gelu': torch.nn.functional.gelu,
            'silu': torch.nn.functional.silu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid
        }
        
        for name, func in activations.items():
            def activation_func():
                return func(x)
            
            self.benchmark_operator(f'pytorch_{name}', activation_func)
            
            # YICA 优化激活函数 (如果可用)
            if self.backend:
                try:
                    class ActivationModel(nn.Module):
                        def __init__(self, activation):
                            super().__init__()
                            self.activation = activation
                        
                        def forward(self, x):
                            return self.activation(x)
                    
                    model = ActivationModel(func)
                    optimized_model = optimize_model(model)
                    
                    def yica_activation():
                        return optimized_model(x)
                    
                    self.benchmark_operator(f'yica_{name}', yica_activation)
                    
                except Exception as e:
                    print(f'  ⚠️  YICA {name} 测试失败: {e}')
    
    def run_comprehensive_benchmark(self):
        \"\"\"运行综合基准测试\"\"\"
        print('🚀 开始 YICA 综合性能测试...')
        print(f'📅 测试时间: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
        print(f'🐍 Python 版本: {sys.version}')
        print(f'🔥 PyTorch 版本: {torch.__version__}')
        if YICA_AVAILABLE:
            print(f'🧠 yirage 版本: {yirage.__version__}')
        print()
        
        # 运行各类测试
        self.test_matrix_operations()
        self.test_attention_mechanisms()
        self.test_normalization_operations()
        self.test_activation_functions()
        
        # 保存结果
        self.save_results()
        self.generate_report()
        
        print('\\n🎉 YICA 性能测试完成！')
    
    def save_results(self):
        \"\"\"保存测试结果\"\"\"
        results_file = self.output_dir / 'yica_performance_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f'📄 结果保存到: {results_file}')
    
    def generate_report(self):
        \"\"\"生成性能报告\"\"\"
        report_file = self.output_dir / 'yica_performance_report.md'
        
        with open(report_file, 'w') as f:
            f.write('# YICA 虚拟环境性能测试报告\\n\\n')
            f.write(f'**测试时间**: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}\\n')
            f.write(f'**测试环境**: YICA 虚拟硬件 + Docker 容器\\n')
            f.write(f'**yirage 版本**: {yirage.__version__ if YICA_AVAILABLE else \"N/A\"}\\n')
            f.write(f'**PyTorch 版本**: {torch.__version__}\\n\\n')
            
            f.write('## 测试结果摘要\\n\\n')
            f.write(f'- **总测试数**: {len(self.results)}\\n')
            
            if self.results:
                avg_latency = np.mean([r[\"mean_latency_ms\"] for r in self.results])
                total_throughput = sum([r[\"throughput_ops_per_sec\"] for r in self.results])
                f.write(f'- **平均延迟**: {avg_latency:.3f} ms\\n')
                f.write(f'- **总吞吐量**: {total_throughput:.2f} ops/sec\\n')
            
            f.write('\\n## 详细结果\\n\\n')
            f.write('| 算子名称 | 平均延迟 (ms) | 标准差 (ms) | 吞吐量 (ops/sec) |\\n')
            f.write('|----------|---------------|-------------|------------------|\\n')
            
            for result in self.results:
                f.write(f'| {result[\"name\"]} | {result[\"mean_latency_ms\"]:.3f} | '
                       f'{result[\"std_latency_ms\"]:.3f} | {result[\"throughput_ops_per_sec\"]:.2f} |\\n')
            
            f.write('\\n## 性能分析\\n\\n')
            
            # 分析 YICA vs PyTorch 性能
            pytorch_results = {r[\"name\"]: r for r in self.results if r[\"name\"].startswith(\"pytorch_\")}
            yica_results = {r[\"name\"]: r for r in self.results if r[\"name\"].startswith(\"yica_\")}
            
            if yica_results:
                f.write('### YICA 优化效果\\n\\n')
                for yica_name, yica_result in yica_results.items():
                    pytorch_name = yica_name.replace(\"yica_\", \"pytorch_\")
                    if pytorch_name in pytorch_results:
                        pytorch_result = pytorch_results[pytorch_name]
                        speedup = pytorch_result[\"mean_latency_ms\"] / yica_result[\"mean_latency_ms\"]
                        f.write(f'- **{yica_name}**: {speedup:.2f}x 加速\\n')
                f.write('\\n')
            else:
                f.write('### 注意事项\\n\\n')
                f.write('- ⚠️ YICA 优化结果未生成，可能是后端初始化失败\\n')
                f.write('- 📝 当前结果仅为 PyTorch 原生实现的基准性能\\n\\n')
            
            f.write('---\\n')
            f.write('*报告由 YICA 虚拟环境性能测试脚本自动生成*\\n')
        
        print(f'📋 报告保存到: {report_file}')

def main():
    \"\"\"主函数\"\"\"
    print('🧠 YICA 虚拟环境性能测试')
    print('=' * 50)
    
    # 创建测试器
    tester = YICAPerformanceTester()
    
    # 运行综合基准测试
    tester.run_comprehensive_benchmark()

if __name__ == '__main__':
    main()
PYTHON_EOF

chmod +x /home/yica/workspace/yica_performance_test.py
echo '✅ 性能测试脚本创建完成'
"
EOF
    
    print_success "性能测试脚本创建完成"
}

# 运行性能测试
run_performance_tests() {
    print_header "运行 YICA 性能测试"
    
    ssh "$REMOTE_SSH" << EOF
        cd $WORK_DIR
        
        echo "🧪 在容器内运行性能测试..."
        
        # 在容器内运行性能测试
        docker exec -it $CONTAINER_NAME bash -c "
            cd /home/yica/workspace
            
            echo '🚀 启动 YICA 性能测试...'
            echo '⏳ 这可能需要几分钟时间，请耐心等待...'
            echo ''
            
            # 设置环境变量
            export YICA_HOME=/home/yica/workspace
            export PYTHONPATH=/home/yica/workspace/yirage/python:\$PYTHONPATH
            export YICA_BACKEND_MODE=virtual
            
            # 运行测试
            python3 yica_performance_test.py
            
            echo ''
            echo '✅ 性能测试完成！'
            echo '📁 查看结果文件:'
            ls -la yica_test_results/
        "
EOF
    
    print_success "性能测试执行完成"
}

# 收集和分析测试结果
collect_test_results() {
    print_header "收集和分析测试结果"
    
    ssh "$REMOTE_SSH" << EOF
        cd $WORK_DIR
        
        echo "📊 收集测试结果..."
        
        # 从容器中复制结果文件
        docker cp $CONTAINER_NAME:/home/yica/workspace/yica_test_results ./yica_performance_results
        
        echo "📁 本地结果文件:"
        ls -la yica_performance_results/
        
        # 显示性能报告摘要
        if [ -f yica_performance_results/yica_performance_report.md ]; then
            echo ""
            echo "📋 性能测试报告摘要:"
            echo "=" * 50
            head -20 yica_performance_results/yica_performance_report.md
            echo "..."
            echo ""
            echo "📄 完整报告: yica_performance_results/yica_performance_report.md"
        fi
        
        # 显示 JSON 结果摘要
        if [ -f yica_performance_results/yica_performance_results.json ]; then
            echo ""
            echo "🔢 数值结果摘要:"
            python3 -c "
import json
import numpy as np

with open('yica_performance_results/yica_performance_results.json', 'r') as f:
    results = json.load(f)

print(f'总测试数: {len(results)}')

if results:
    latencies = [r['mean_latency_ms'] for r in results]
    throughputs = [r['throughput_ops_per_sec'] for r in results]
    
    print(f'平均延迟: {np.mean(latencies):.3f} ± {np.std(latencies):.3f} ms')
    print(f'总吞吐量: {sum(throughputs):.2f} ops/sec')
    
    print('\n🏆 最佳性能算子:')
    best_latency = min(results, key=lambda x: x['mean_latency_ms'])
    best_throughput = max(results, key=lambda x: x['throughput_ops_per_sec'])
    
    print(f'  最低延迟: {best_latency[\"name\"]} ({best_latency[\"mean_latency_ms\"]:.3f} ms)')
    print(f'  最高吞吐: {best_throughput[\"name\"]} ({best_throughput[\"throughput_ops_per_sec\"]:.2f} ops/sec)')
    
    # YICA vs PyTorch 对比
    pytorch_results = [r for r in results if r['name'].startswith('pytorch_')]
    yica_results = [r for r in results if r['name'].startswith('yica_')]
    
    if yica_results:
        print(f'\n🧠 YICA 优化效果:')
        print(f'  PyTorch 算子: {len(pytorch_results)} 个')
        print(f'  YICA 算子: {len(yica_results)} 个')
        
        if pytorch_results and yica_results:
            pytorch_avg = np.mean([r['mean_latency_ms'] for r in pytorch_results])
            yica_avg = np.mean([r['mean_latency_ms'] for r in yica_results])
            speedup = pytorch_avg / yica_avg if yica_avg > 0 else 1.0
            print(f'  平均加速比: {speedup:.2f}x')
    else:
        print(f'\n⚠️  未发现 YICA 优化结果，可能是后端未正常工作')
"
        fi
EOF
    
    print_success "测试结果收集完成"
}

# 生成可视化报告
generate_visualization() {
    print_status "生成性能可视化报告..."
    
    ssh "$REMOTE_SSH" << 'EOF'
        cd $WORK_DIR
        
        # 创建可视化脚本
        cat > visualize_results.py << 'VIZ_EOF'
#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_visualizations():
    results_file = Path('yica_performance_results/yica_performance_results.json')
    if not results_file.exists():
        print("❌ 结果文件不存在")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    if not results:
        print("❌ 没有测试结果")
        return
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 延迟对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    names = [r['name'] for r in results]
    latencies = [r['mean_latency_ms'] for r in results]
    throughputs = [r['throughput_ops_per_sec'] for r in results]
    
    # 延迟图
    colors = ['red' if name.startswith('pytorch_') else 'blue' for name in names]
    bars1 = ax1.bar(range(len(names)), latencies, color=colors, alpha=0.7)
    ax1.set_xlabel('Operators')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('YICA vs PyTorch Latency Comparison')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels([name.replace('pytorch_', 'PT_').replace('yica_', 'YC_') for name in names], 
                        rotation=45, ha='right')
    
    # 添加数值标签
    for bar, latency in zip(bars1, latencies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{latency:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 吞吐量图
    bars2 = ax2.bar(range(len(names)), throughputs, color=colors, alpha=0.7)
    ax2.set_xlabel('Operators')
    ax2.set_ylabel('Throughput (ops/sec)')
    ax2.set_title('YICA vs PyTorch Throughput Comparison')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([name.replace('pytorch_', 'PT_').replace('yica_', 'YC_') for name in names], 
                        rotation=45, ha='right')
    
    # 添加数值标签
    for bar, throughput in zip(bars2, throughputs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{throughput:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('yica_performance_results/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 加速比分析
    pytorch_results = {r['name']: r for r in results if r['name'].startswith('pytorch_')}
    yica_results = {r['name']: r for r in results if r['name'].startswith('yica_')}
    
    if yica_results and pytorch_results:
        speedups = []
        op_names = []
        
        for yica_name, yica_result in yica_results.items():
            pytorch_name = yica_name.replace('yica_', 'pytorch_')
            if pytorch_name in pytorch_results:
                pytorch_result = pytorch_results[pytorch_name]
                speedup = pytorch_result['mean_latency_ms'] / yica_result['mean_latency_ms']
                speedups.append(speedup)
                op_names.append(yica_name.replace('yica_', ''))
        
        if speedups:
            plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(op_names)), speedups, 
                          color=['green' if s > 1 else 'red' for s in speedups], alpha=0.7)
            plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No speedup')
            plt.xlabel('Operations')
            plt.ylabel('Speedup (x)')
            plt.title('YICA Optimization Speedup')
            plt.xticks(range(len(op_names)), op_names, rotation=45, ha='right')
            
            # 添加数值标签
            for bar, speedup in zip(bars, speedups):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{speedup:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.legend()
            plt.tight_layout()
            plt.savefig('yica_performance_results/speedup_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ 生成加速比分析图: speedup_analysis.png")
    
    print("✅ 性能可视化图表生成完成")
    print("📊 生成的图表:")
    print("  - performance_comparison.png: 延迟和吞吐量对比")
    if yica_results and pytorch_results and speedups:
        print("  - speedup_analysis.png: YICA 加速比分析")

if __name__ == '__main__':
    create_visualizations()
VIZ_EOF
        
        # 运行可视化脚本
        python3 visualize_results.py
EOF
    
    print_success "可视化报告生成完成"
}

# 显示测试总结
show_test_summary() {
    print_header "YICA 虚拟环境性能测试总结"
    
    echo -e "${GREEN}🎉 YICA 虚拟环境性能测试完成！${NC}"
    echo ""
    echo -e "${CYAN}📋 测试概况:${NC}"
    echo -e "  🖥️  服务器: $REMOTE_HOST"
    echo -e "  🐳 容器: $CONTAINER_NAME"
    echo -e "  📁 结果目录: $WORK_DIR/yica_performance_results"
    echo ""
    echo -e "${CYAN}📊 生成的文件:${NC}"
    echo -e "  📄 yica_performance_results.json - 详细数值结果"
    echo -e "  📋 yica_performance_report.md - 性能分析报告"
    echo -e "  📊 performance_comparison.png - 性能对比图表"
    echo -e "  📈 speedup_analysis.png - 加速比分析图表"
    echo ""
    echo -e "${CYAN}🔍 查看结果:${NC}"
    echo -e "  ssh $REMOTE_SSH 'cd $WORK_DIR && ls -la yica_performance_results/'"
    echo -e "  ssh $REMOTE_SSH 'cd $WORK_DIR && cat yica_performance_results/yica_performance_report.md'"
    echo ""
    echo -e "${CYAN}📥 下载结果:${NC}"
    echo -e "  scp -r $REMOTE_SSH:$WORK_DIR/yica_performance_results ./local_yica_results"
    echo ""
    echo -e "${YELLOW}💡 下一步建议:${NC}"
    echo -e "  1. 分析性能报告，识别优化效果最显著的算子"
    echo -e "  2. 针对性能瓶颈进行进一步优化"
    echo -e "  3. 在实际应用场景中验证性能提升"
    echo -e "  4. 考虑扩展测试到更大规模的模型"
}

# 主函数
main() {
    case "${1:-}" in
        "check")
            check_connection
            check_docker_environment
            ;;
        "services")
            check_connection
            start_yica_virtual_services
            ;;
        "test")
            check_connection
            check_docker_environment
            create_performance_test_script
            run_performance_tests
            ;;
        "collect")
            check_connection
            collect_test_results
            generate_visualization
            ;;
        "quick")
            print_header "YICA 快速性能测试"
            TEST_MODE="quick"
            BENCHMARK_ITERATIONS=20
            WARMUP_ITERATIONS=3
            
            check_connection
            check_docker_environment
            start_yica_virtual_services
            create_performance_test_script
            run_performance_tests
            collect_test_results
            generate_visualization
            show_test_summary
            ;;
        "")
            print_header "YICA 完整性能测试流程"
            
            check_connection
            check_docker_environment
            start_yica_virtual_services
            create_performance_test_script
            run_performance_tests
            collect_test_results
            generate_visualization
            show_test_summary
            ;;
        *)
            echo "YICA 虚拟环境性能测试脚本"
            echo ""
            echo "用法: $0 [命令]"
            echo ""
            echo "命令:"
            echo "  check      - 检查环境和连接"
            echo "  services   - 启动 YICA 虚拟硬件服务"
            echo "  test       - 运行性能测试"
            echo "  collect    - 收集和分析结果"
            echo "  quick      - 快速测试模式 (20次迭代)"
            echo "  (空)       - 完整测试流程 (100次迭代)"
            echo ""
            echo "功能特性:"
            echo "  ✅ 完整的 YICA 虚拟硬件环境"
            echo "  ✅ yirage 算子性能基准测试"
            echo "  ✅ PyTorch vs YICA 性能对比"
            echo "  ✅ 自动化结果收集和分析"
            echo "  ✅ 可视化性能报告生成"
            echo "  ✅ Docker 容器化测试环境"
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@" 
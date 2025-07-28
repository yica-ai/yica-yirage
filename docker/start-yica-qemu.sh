#!/bin/bash
# YICA-QEMU Docker容器启动脚本
# 支持Mac本地验证开发

set -e

echo "🚀 启动YICA-QEMU验证环境..."

# 环境变量设置
export YICA_HOME="/yica-workspace"
export PYTHONPATH="/yica-workspace/yirage/python:$PYTHONPATH"
export LD_LIBRARY_PATH="/yica-workspace/yirage/build:/opt/rocm/lib:$LD_LIBRARY_PATH"

# 创建必要目录
mkdir -p /yica-workspace/qemu-images
mkdir -p /yica-workspace/logs
mkdir -p /tmp/yica-sockets

# 检查YICA核心库
echo "🔍 检查YICA核心库..."
if [ ! -f "/yica-workspace/yirage/build/libyirage_runtime.so" ]; then
    echo "❌ YICA核心库未找到，尝试重新编译..."
    cd /yica-workspace/yirage/build
    make -j$(nproc)
fi

# 验证Python导入
echo "🐍 验证Python环境..."
cd /yica-workspace
python3 -c "
try:
    import yirage
    print('✅ yirage导入成功')
    print(f'版本: {yirage.__version__}')
except ImportError as e:
    print(f'❌ yirage导入失败: {e}')
    print('尝试修复...')
"

# 运行基础测试
echo "🧪 运行基础验证测试..."
cd /yica-workspace/tests
python3 -m pytest yica_basic_benchmarks.py -v --tb=short || echo "⚠️  部分测试失败，继续启动"

# 启动交互式shell或指定命令
if [ "$#" -eq 0 ]; then
    echo "🎯 YICA-QEMU验证环境就绪！"
    echo ""
    echo "可用命令："
    echo "  - 运行YICA测试: cd /yica-workspace/tests && python3 -m pytest"
    echo "  - 运行基准测试: cd /yica-workspace && python3 -m yirage.benchmark.yica_benchmark_suite"
    echo "  - 启动QEMU: /yica-workspace/scripts/start-qemu.sh"
    echo "  - 查看日志: tail -f /yica-workspace/logs/*.log"
    echo ""
    exec /bin/bash
else
    exec "$@"
fi 
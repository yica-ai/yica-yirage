#!/bin/bash

# YICA CPU构建脚本 - 无CUDA依赖
# 用于在没有NVIDIA GPU或CUDA驱动的环境中构建YICA

set -e

echo "========================================"
echo "构建YICA CPU版本 (无CUDA依赖)"
echo "========================================"

# 检查必要的工具
echo "检查构建工具..."

if ! command -v g++ &> /dev/null; then
    echo "错误: g++未找到，请安装g++编译器"
    exit 1
fi

if ! command -v cmake &> /dev/null; then
    echo "错误: cmake未找到，请安装cmake"
    exit 1
fi

echo "✓ 构建工具检查通过"

# 检查OpenMP支持
echo "检查OpenMP支持..."
if g++ -fopenmp -dumpversion &> /dev/null; then
    echo "✓ OpenMP支持已找到"
else
    echo "警告: OpenMP支持未找到，并行性能可能受限"
fi

# 创建构建目录
BUILD_DIR="build-cpu"
echo "创建构建目录: $BUILD_DIR"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# 配置CMake - CPU版本
echo "配置CMake (CPU版本)..."
cmake -S .. -B . \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_FLAGS="-DYICA_CPU_ONLY -DNO_CUDA -fopenmp -mavx2 -ffast-math" \
    -DUSE_CUDA=OFF \
    -DBUILD_YICA_TESTS=ON \
    -DBUILD_YICA_EXAMPLES=ON

# 开始构建
echo "开始构建YICA CPU版本..."
make -j$(nproc) yica_cpu

# 检查构建结果
if [ -f "libyica_cpu.so" ] || [ -f "libyica_cpu.a" ]; then
    echo "✓ YICA CPU库构建成功"
else
    echo "❌ YICA CPU库构建失败"
    exit 1
fi

# 构建测试
if [ "$1" = "--with-tests" ]; then
    echo "构建测试程序..."
    make -j$(nproc) yica_tests
    
    if [ -f "yica_tests" ]; then
        echo "✓ 测试程序构建成功"
        echo "运行基本测试..."
        ./yica_tests
    else
        echo "❌ 测试程序构建失败"
    fi
fi

# 构建示例
if [ "$1" = "--with-examples" ]; then
    echo "构建示例程序..."
    make -j$(nproc) yica_demo
    
    if [ -f "yica_demo" ]; then
        echo "✓ 示例程序构建成功"
    else
        echo "❌ 示例程序构建失败"
    fi
fi

echo "========================================"
echo "YICA CPU构建完成!"
echo "========================================"
echo "构建产物位置:"
echo "  库文件: $PWD/libyica_cpu.*"
if [ -f "yica_tests" ]; then
    echo "  测试程序: $PWD/yica_tests"
fi
if [ -f "yica_demo" ]; then
    echo "  示例程序: $PWD/yica_demo"
fi

echo ""
echo "使用方法:"
echo "  直接链接: g++ your_code.cpp -L$PWD -lyica_cpu -fopenmp"
echo "  运行测试: ./yica_tests"
if [ -f "yica_demo" ]; then
    echo "  运行示例: ./yica_demo"
fi

echo ""
echo "系统信息:"
echo "  编译器: $(g++ --version | head -n1)"
echo "  OpenMP: $(g++ -fopenmp -dumpversion 2>/dev/null || echo '未支持')"
echo "  CPU核心: $(nproc)"
echo "  可用内存: $(free -h | grep '^Mem:' | awk '{print $2}' || echo '未知')"

cd ..
echo "构建脚本执行完成。" 
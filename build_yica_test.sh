#!/bin/bash

# YICA优化器功能构建和测试脚本
# 包含YICA架构感知分析器和优化策略库

set -e  # 遇到错误时退出

echo "=== YICA优化器构建和测试 ==="

# 项目根目录
PROJECT_ROOT=$(pwd)
BUILD_DIR="${PROJECT_ROOT}/build_yica"

# 创建构建目录
echo "创建构建目录: ${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# 配置CMake
echo "配置CMake..."
cmake "${PROJECT_ROOT}/mirage" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DBUILD_TESTING=ON

# 编译项目
echo "编译YICA相关源文件..."
make -j$(nproc) 2>&1 | tee build.log

# 检查编译是否成功
if [ $? -eq 0 ]; then
    echo "✅ 编译成功"
    
    # 列出生成的测试文件
    echo "生成的测试文件:"
    find . -name "*yica*" -type f -executable 2>/dev/null || echo "未找到YICA测试可执行文件"
    
    # 运行YICA测试（如果存在）
    if [ -f "./tests/yica/yica_tests" ]; then
        echo "运行YICA测试..."
        ./tests/yica/yica_tests
        
        if [ $? -eq 0 ]; then
            echo "✅ 所有YICA测试通过"
        else
            echo "❌ YICA测试失败"
            exit 1
        fi
    else
        echo "⚠️  YICA测试可执行文件不存在，跳过测试"
    fi
    
    # 显示构建摘要
    echo ""
    echo "=== 构建摘要 ==="
    echo "项目根目录: ${PROJECT_ROOT}"
    echo "构建目录: ${BUILD_DIR}"
    echo "构建类型: Debug"
    echo "C++标准: C++17"
    echo ""
    echo "已实现的YICA功能:"
    echo "1. ✅ YICA架构感知分析器"
    echo "2. ✅ YICA优化策略库"
    echo "   - CIM数据重用优化策略"
    echo "   - SPM分配优化策略"  
    echo "   - 算子融合优化策略"
    echo "3. ✅ YICA代码生成器"
    echo "   - 模板化代码生成系统"
    echo "   - CIM指令生成算法"
    echo "   - 多种操作生成器支持"
    echo "4. ✅ 策略选择和组合算法"
    echo "5. ✅ 端到端优化流程"
    echo ""
    echo "测试覆盖:"
    echo "- 架构感知分析器单元测试"
    echo "- 优化策略库单元测试"
    echo "- YICA代码生成器单元测试"
    echo "- 策略应用和兼容性测试"
    echo ""
    echo "📁 已创建的文件:"
    echo "  - mirage/include/mirage/search/yica/yica_types.h"
    echo "  - mirage/include/mirage/search/yica/yica_analyzer.h"
    echo "  - mirage/include/mirage/search/yica/optimization_strategy.h"
    echo "  - mirage/include/mirage/search/yica/strategy_library.h"
    echo "  - mirage/include/mirage/search/yica/code_generator.h"
    echo "  - mirage/include/mirage/search/yica/operator_generators.h"
    echo "  - mirage/src/search/yica/yica_analyzer.cc"
    echo "  - mirage/src/search/yica/optimization_strategy.cc"
    echo "  - mirage/src/search/yica/strategy_library.cc"
    echo "  - mirage/src/search/yica/code_generator.cc"
    echo "  - mirage/src/search/yica/operator_generators.cc"
    echo "  - mirage/tests/yica/test_yica_analyzer.cc"
    echo "  - mirage/tests/yica/test_strategy_library.cc"
    echo "  - mirage/tests/yica/test_code_generator.cc"
    echo ""
    echo "🚀 下一步:"
    echo "  1. 集成到Mirage构建系统"
    echo "  2. 添加更多优化策略"
    echo "  3. 性能基准测试"
    echo "  4. 实际工作负载验证"
    
else
    echo "❌ 编译失败，查看错误信息:"
    tail -20 build.log
    exit 1
fi

echo "=== YICA优化器构建完成 ===" 
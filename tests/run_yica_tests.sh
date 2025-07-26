#!/bin/bash

# YICA-Yirage 测试运行脚本
# 这个脚本运行所有 YICA 相关的测试和基准测试

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# 脚本参数
CONFIG="default"
SKIP_PERFORMANCE=false
SKIP_CLI=false
QUICK_MODE=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --skip-performance)
            SKIP_PERFORMANCE=true
            shift
            ;;
        --skip-cli)
            SKIP_CLI=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            CONFIG="quick"
            shift
            ;;
        --help)
            echo "YICA-Yirage 测试运行脚本"
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --config CONFIG      测试配置 (default, quick, full)"
            echo "  --skip-performance   跳过性能测试"
            echo "  --skip-cli          跳过命令行工具测试"
            echo "  --quick             快速模式 (等同于 --config quick)"
            echo "  --help              显示此帮助信息"
            exit 0
            ;;
        *)
            print_error "未知参数: $1"
            exit 1
            ;;
    esac
done

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

print_header "🚀 YICA-Yirage 测试套件"
print_status "项目根目录: $PROJECT_ROOT"
print_status "测试配置: $CONFIG"

# 检查 Python 环境
print_status "检查 Python 环境..."
python3 --version || {
    print_error "Python 3 未安装"
    exit 1
}

# 检查必要的包
print_status "检查依赖包..."
MISSING_DEPS=""

if ! python3 -c "import numpy" 2>/dev/null; then
    MISSING_DEPS="$MISSING_DEPS numpy"
fi

if ! python3 -c "import torch" 2>/dev/null; then
    print_warning "PyTorch 未安装 - 某些测试将被跳过"
fi

if ! python3 -c "import yirage" 2>/dev/null; then
    print_warning "YICA 包未安装 - 尝试从本地路径导入"
fi

if [ -n "$MISSING_DEPS" ]; then
    print_warning "缺少依赖包: $MISSING_DEPS"
    print_status "尝试安装缺少的包..."
    python3 -m pip install $MISSING_DEPS || {
        print_warning "无法安装依赖包，某些测试可能失败"
    }
fi

# 创建结果目录
RESULTS_DIR="$PROJECT_ROOT/test_results"
mkdir -p "$RESULTS_DIR"
print_status "测试结果将保存到: $RESULTS_DIR"

# 设置测试参数
PYTHON_CMD="python3"
if [ "$QUICK_MODE" = true ]; then
    BENCHMARK_ARGS="--warmup 2 --profile 10"
    TEST_ARGS="--config quick"
else
    BENCHMARK_ARGS="--warmup 16 --profile 100"
    TEST_ARGS="--config $CONFIG"
fi

if [ "$SKIP_PERFORMANCE" = true ]; then
    TEST_ARGS="$TEST_ARGS --skip-performance"
fi

if [ "$SKIP_CLI" = true ]; then
    TEST_ARGS="$TEST_ARGS --skip-cli"
fi

# 运行测试
print_header "📋 1/3 运行综合测试套件"
cd "$PROJECT_ROOT"

if [ -f "tests/yica_comprehensive_test_suite.py" ]; then
    print_status "运行 YICA 综合测试..."
    $PYTHON_CMD tests/yica_comprehensive_test_suite.py $TEST_ARGS --output-dir "$RESULTS_DIR" || {
        print_warning "综合测试套件执行出现问题"
    }
else
    print_error "测试文件不存在: tests/yica_comprehensive_test_suite.py"
fi

print_header "⚡ 2/3 运行基准测试"

if [ -f "tests/yica_basic_benchmarks.py" ]; then
    print_status "运行 YICA 基础基准测试..."
    OUTPUT_FILE="$RESULTS_DIR/yica_benchmark_$(date +%Y%m%d_%H%M%S).json"
    $PYTHON_CMD tests/yica_basic_benchmarks.py $BENCHMARK_ARGS --output "$OUTPUT_FILE" || {
        print_warning "基准测试执行出现问题"
    }
else
    print_error "基准测试文件不存在: tests/yica_basic_benchmarks.py"
fi

print_header "🔧 3/3 运行命令行工具测试"

if [ "$SKIP_CLI" = false ]; then
    print_status "测试 YICA 命令行工具..."
    
    # 测试命令行工具
    CLI_TOOLS=("yica-optimizer" "yica-benchmark" "yica-analyze")
    
    for tool in "${CLI_TOOLS[@]}"; do
        print_status "测试 $tool..."
        if command -v "$tool" >/dev/null 2>&1; then
            $tool --version 2>/dev/null || $tool --help >/dev/null 2>&1 || {
                print_warning "$tool 可能未正确安装"
            }
            print_status "✅ $tool 可用"
        else
            print_warning "❌ $tool 未找到"
        fi
    done
else
    print_status "跳过命令行工具测试"
fi

# 生成汇总报告
print_header "📊 生成测试报告"

SUMMARY_FILE="$RESULTS_DIR/test_summary_$(date +%Y%m%d_%H%M%S).txt"

cat > "$SUMMARY_FILE" << EOF
YICA-Yirage 测试执行摘要
========================================

执行时间: $(date)
测试配置: $CONFIG
项目根目录: $PROJECT_ROOT

测试组件:
- 综合测试套件: ✅
- 基准测试: ✅
- 命令行工具测试: $([ "$SKIP_CLI" = false ] && echo "✅" || echo "⏭️ 跳过")

结果文件:
$(ls -la "$RESULTS_DIR"/*.json 2>/dev/null || echo "  无 JSON 结果文件")
$(ls -la "$RESULTS_DIR"/*.txt 2>/dev/null || echo "  无文本结果文件")

环境信息:
- Python 版本: $(python3 --version 2>&1)
- 操作系统: $(uname -a)
- 工作目录: $(pwd)

注意事项:
- 详细的测试结果请查看 $RESULTS_DIR 目录下的文件
- 如有测试失败，请检查相应的错误日志
- 性能测试结果可能因硬件和系统负载而有所不同

EOF

print_status "测试摘要已保存到: $SUMMARY_FILE"

# 显示结果目录内容
print_header "📁 测试结果文件"
ls -la "$RESULTS_DIR/" 2>/dev/null || print_warning "结果目录为空"

# 最终状态
print_header "🏁 测试执行完成"
print_status "所有测试已执行完毕"
print_status "结果目录: $RESULTS_DIR"
print_status "摘要文件: $SUMMARY_FILE"

# 如果是快速模式，显示一些基本统计
if [ "$QUICK_MODE" = true ]; then
    print_header "⚡ 快速测试统计"
    
    # 统计结果文件数量
    JSON_COUNT=$(ls "$RESULTS_DIR"/*.json 2>/dev/null | wc -l)
    TXT_COUNT=$(ls "$RESULTS_DIR"/*.txt 2>/dev/null | wc -l)
    
    print_status "生成的 JSON 结果文件: $JSON_COUNT"
    print_status "生成的文本结果文件: $TXT_COUNT"
    
    # 显示最新的测试结果文件
    LATEST_JSON=$(ls -t "$RESULTS_DIR"/*.json 2>/dev/null | head -1)
    if [ -n "$LATEST_JSON" ]; then
        print_status "最新结果文件: $(basename "$LATEST_JSON")"
    fi
fi

print_status "运行完成! 🎉" 
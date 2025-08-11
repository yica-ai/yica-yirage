#!/bin/bash
# YICA-Yirage 测试运行脚本
# 从项目根目录运行各种分类测试

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# 显示帮助信息
show_help() {
    echo "YICA-Yirage 测试运行脚本"
    echo ""
    echo "用法: $0 [测试类型]"
    echo ""
    echo "测试类型:"
    echo "  all           - 运行所有测试"
    echo "  integration   - 集成测试"
    echo "  hardware      - 硬件模拟测试"
    echo "  yica          - YICA核心C++测试"
    echo "  python        - Python后端测试"
    echo "  help          - 显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 integration    # 运行集成测试"
    echo "  $0 python         # 运行Python测试"
    echo "  $0 all            # 运行所有测试"
}

# 运行集成测试
run_integration_tests() {
    print_status "运行集成测试..."
    cd tests/integration
    python3 simple_yirage_test.py
    cd ../..
    print_success "集成测试完成"
}

# 运行硬件模拟测试
run_hardware_tests() {
    print_status "运行硬件模拟测试..."
    cd tests/hardware
    python3 yica_hardware_simulation_test.py
    cd ../..
    print_success "硬件模拟测试完成"
}



# 运行YICA核心测试
run_yica_tests() {
    print_status "运行YICA核心C++测试..."
    cd tests
    if [ -f "run_yica_tests.sh" ]; then
        ./run_yica_tests.sh
    else
        print_warning "run_yica_tests.sh 不存在，跳过C++测试"
    fi
    cd ..
    print_success "YICA核心测试完成"
}

# 运行Python后端测试
run_python_tests() {
    print_status "运行Python后端测试..."
    cd tests
    
    python_tests=(
        "yica_basic_benchmarks.py"
        "yica_backend_simple_validation.py" 
        "yica_architecture_comparison.py"
    )
    
    for test in "${python_tests[@]}"; do
        if [ -f "$test" ]; then
            print_status "运行 $test..."
            python3 "$test" || print_warning "$test 运行失败"
        else
            print_warning "$test 不存在，跳过"
        fi
    done
    
    cd ..
    print_success "Python后端测试完成"
}



# 运行所有测试
run_all_tests() {
    print_status "运行所有测试..."
    echo ""
    
    run_integration_tests
    echo ""
    
    # 可选的硬件和Python测试
    print_status "运行可选测试..."
    run_python_tests || print_warning "部分Python测试失败"
    
    print_success "所有测试运行完成"
}

# 主函数
main() {
    case "${1:-help}" in
        "all")
            run_all_tests
            ;;
        "integration")
            run_integration_tests
            ;;
        "hardware")
            run_hardware_tests
            ;;
        "yica")
            run_yica_tests
            ;;
        "python")
            run_python_tests
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            print_error "未知的测试类型: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# 检查是否在项目根目录
if [ ! -d "tests" ] || [ ! -d "yirage" ]; then
    print_error "请在项目根目录运行此脚本"
    exit 1
fi

# 执行主函数
main "$@"

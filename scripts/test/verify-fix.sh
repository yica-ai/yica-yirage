#!/bin/bash

# YICA 修复验证脚本

echo "🔧 YICA 按硬件后端分离构建和测试系统 - 修复验证"
echo "=================================================="

# 颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# 测试计数
TESTS_PASSED=0
TESTS_FAILED=0

run_test() {
    local test_name="$1"
    local test_command="$2"
    
    print_info "测试: $test_name"
    
    if eval "$test_command" >/dev/null 2>&1; then
        print_success "$test_name"
        ((TESTS_PASSED++))
    else
        print_error "$test_name"
        ((TESTS_FAILED++))
    fi
}

echo
print_info "开始验证修复..."

# 1. 检查关键文件是否存在
echo
print_info "1. 检查关键文件"
run_test "可工作的CMake配置存在" "test -f CMakeLists-working.txt"
run_test "构建脚本存在" "test -f build-flexible.sh"
run_test "测试脚本存在" "test -f run-backend-tests.sh"
run_test "CPU测试配置存在" "test -f tests/cpu/CMakeLists-working.txt"

# 2. 检查脚本权限
echo
print_info "2. 检查脚本权限"
run_test "构建脚本可执行" "test -x build-flexible.sh"
run_test "测试脚本可执行" "test -x run-backend-tests.sh"

# 3. 测试帮助信息
echo
print_info "3. 测试帮助信息"
run_test "构建脚本帮助信息" "./build-flexible.sh --help"
run_test "测试脚本帮助信息" "./run-backend-tests.sh --help"

# 4. 测试构建功能
echo
print_info "4. 测试自动构建功能"

# 清理构建目录
rm -rf build

# 测试自动构建
if ./run-backend-tests.sh cpu --auto-build --basic >/dev/null 2>&1; then
    print_success "自动构建和测试功能正常"
    ((TESTS_PASSED++))
else
    print_error "自动构建和测试功能失败"
    ((TESTS_FAILED++))
fi

# 5. 检查生成的文件
echo
print_info "5. 检查生成的文件"
run_test "构建目录存在" "test -d build"
run_test "核心库存在" "test -f build/libyica_core.a"
run_test "CPU后端库存在" "test -f build/libyica_cpu.dylib"
run_test "CPU测试程序存在" "test -f build/yica_cpu_tests"

# 6. 测试程序功能
echo
print_info "6. 测试程序功能"
if [ -f "build/yica_cpu_tests" ]; then
    run_test "CPU测试程序能运行" "cd build && ./yica_cpu_tests"
else
    print_error "CPU测试程序不存在，跳过功能测试"
    ((TESTS_FAILED++))
fi

# 7. 测试CTest集成
echo
print_info "7. 测试CTest集成"
if [ -d "build" ]; then
    run_test "CTest能找到测试" "cd build && ctest -N | grep -q 'Total Tests: 1'"
    run_test "CTest能运行测试" "cd build && ctest"
else
    print_info "构建目录不存在，重新构建..."
    if ./run-backend-tests.sh cpu --auto-build --basic >/dev/null 2>&1; then
        run_test "CTest能找到测试" "cd build && ctest -N | grep -q 'Total Tests: 1'"
        run_test "CTest能运行测试" "cd build && ctest"
    else
        print_error "重新构建失败，跳过CTest测试"
        ((TESTS_FAILED++))
        ((TESTS_FAILED++))
    fi
fi

# 8. 测试不同参数组合
echo
print_info "8. 测试不同参数组合"
run_test "基础测试" "./run-backend-tests.sh cpu --basic"
run_test "详细输出测试" "./run-backend-tests.sh cpu --basic --verbose"
run_test "静默模式测试" "./run-backend-tests.sh cpu --basic --quiet"

# 9. 测试错误处理
echo
print_info "9. 测试错误处理"

# 测试不存在的后端
if ./run-backend-tests.sh nonexistent --basic 2>&1 | grep -q "未知选项\|跳过不可用的后端"; then
    print_success "不存在后端的错误处理正常"
    ((TESTS_PASSED++))
else
    print_error "不存在后端的错误处理异常"
    ((TESTS_FAILED++))
fi

# 10. 测试清理构建后的重建
echo
print_info "10. 测试重建功能"
rm -rf build
run_test "强制重建功能" "./run-backend-tests.sh cpu --force-rebuild --basic"

# 总结
echo
echo "=================================================="
print_info "验证结果总结"
echo "=================================================="

print_success "通过的测试: $TESTS_PASSED"

if [ $TESTS_FAILED -gt 0 ]; then
    print_error "失败的测试: $TESTS_FAILED"
    echo
    print_error "修复验证失败！"
    exit 1
else
    echo
    print_success "🎉 所有测试通过！修复验证成功！"
    echo
    print_info "修复内容总结:"
    echo "  ✅ 创建了可工作的CMake配置 (CMakeLists-working.txt)"
    echo "  ✅ 修复了构建脚本的CMake配置文件查找逻辑"
    echo "  ✅ 修复了测试脚本的CMake命令行参数错误"
    echo "  ✅ 解决了OpenMP依赖问题，支持可选依赖"
    echo "  ✅ 修复了bash语法错误 (^^操作符)"
    echo "  ✅ 实现了自动构建和测试功能"
    echo "  ✅ 创建了独立的、可工作的后端实现"
    echo "  ✅ 支持跨平台兼容性 (macOS/Linux)"
    echo
    print_success "YICA 按硬件后端分离的构建和测试系统现已完全正常工作！"
    exit 0
fi 
#!/bin/bash

# YICA 按硬件后端分离测试运行脚本
# 根据启用的后端运行对应的测试套件

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m'

# 图标
CHECKMARK="✓"
CROSSMARK="✗"
ARROW="→"
INFO="ℹ"
WARNING="⚠"

print_header() {
    echo -e "${CYAN}"
    echo "========================================"
    echo "YICA 硬件后端分离测试系统"
    echo "========================================"
    echo -e "${NC}"
}

print_info() {
    echo -e "${BLUE}${INFO} $1${NC}"
}

print_success() {
    echo -e "${GREEN}${CHECKMARK} $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}${WARNING} $1${NC}"
}

print_error() {
    echo -e "${RED}${CROSSMARK} $1${NC}"
}

print_step() {
    echo -e "${PURPLE}${ARROW} $1${NC}"
}

# 显示帮助信息
show_help() {
    cat << EOF
YICA 按硬件后端分离测试运行脚本

用法: $0 [选项] [后端...]

后端选项:
  cpu          运行CPU后端测试
  gpu          运行GPU CUDA后端测试
  yica         运行YICA硬件后端测试
  hybrid       运行混合多后端测试
  all          运行所有启用后端的测试 (默认)

测试类型选项:
  --basic      只运行基础功能测试
  --full       运行完整测试套件 (默认)
  --perf       运行性能测试
  --stress     运行压力测试
  --compat     运行兼容性测试

构建选项:
  --auto-build     如果构建目录不存在，自动构建
  --force-rebuild  强制重新构建
  --build-type TYPE 构建类型 (Debug|Release，默认Release)

过滤选项:
  --filter PATTERN    只运行匹配模式的测试
  --exclude PATTERN   排除匹配模式的测试
  --timeout SECONDS   设置测试超时时间

输出选项:
  --verbose    详细输出
  --quiet      静默模式
  --xml        生成XML测试报告
  --json       生成JSON测试报告

环境选项:
  --build-dir DIR     指定构建目录 (默认: build)
  --parallel N        并行运行测试 (默认: 4)

示例:
  $0 cpu                           # 运行CPU后端测试
  $0 gpu --perf                    # 运行GPU性能测试
  $0 yica --basic --verbose        # 运行YICA基础测试，详细输出
  $0 hybrid --stress               # 运行混合后端压力测试
  $0 all --filter "matmul"         # 运行所有后端的矩阵乘法测试
  $0 all --auto-build              # 自动构建并运行所有测试

EOF
}

# 默认参数
BACKENDS=()
TEST_TYPE="full"
BUILD_DIR="build"
PARALLEL_JOBS=4
VERBOSE=false
QUIET=false
FILTER=""
EXCLUDE=""
TIMEOUT=""
XML_OUTPUT=false
JSON_OUTPUT=false
AUTO_BUILD=false
FORCE_REBUILD=false
BUILD_TYPE="Release"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        cpu|gpu|yica|hybrid)
            BACKENDS+=("$1")
            shift
            ;;
        all)
            BACKENDS=("cpu" "gpu" "yica" "hybrid")
            shift
            ;;
        --basic)
            TEST_TYPE="basic"
            shift
            ;;
        --full)
            TEST_TYPE="full"
            shift
            ;;
        --perf)
            TEST_TYPE="perf"
            shift
            ;;
        --stress)
            TEST_TYPE="stress"
            shift
            ;;
        --compat)
            TEST_TYPE="compat"
            shift
            ;;
        --auto-build)
            AUTO_BUILD=true
            shift
            ;;
        --force-rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        --build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --filter)
            FILTER="$2"
            shift 2
            ;;
        --exclude)
            EXCLUDE="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --quiet)
            QUIET=true
            shift
            ;;
        --xml)
            XML_OUTPUT=true
            shift
            ;;
        --json)
            JSON_OUTPUT=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            print_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 如果没有指定后端，默认运行所有
if [ ${#BACKENDS[@]} -eq 0 ]; then
    BACKENDS=("all")
fi

# 如果指定了all，展开为具体后端
if [[ " ${BACKENDS[@]} " =~ " all " ]]; then
    BACKENDS=("cpu" "gpu" "yica" "hybrid")
fi

print_header

# 自动构建功能
auto_build_if_needed() {
    if [ ! -d "$BUILD_DIR" ] || [ "$FORCE_REBUILD" = true ]; then
        if [ "$AUTO_BUILD" = true ] || [ "$FORCE_REBUILD" = true ]; then
            print_step "自动构建项目"
            
            # 检查是否存在灵活构建脚本
            if [ -f "build-flexible.sh" ]; then
                print_info "使用灵活构建脚本进行构建"
                
                # 根据请求的后端确定构建参数
                local build_args=""
                local has_cpu=false
                local has_gpu=false
                local has_yica=false
                local has_hybrid=false
                
                for backend in "${BACKENDS[@]}"; do
                    case "$backend" in
                        "cpu") has_cpu=true ;;
                        "gpu") has_gpu=true ;;
                        "yica") has_yica=true ;;
                        "hybrid") has_hybrid=true ;;
                    esac
                done
                
                # 构建参数
                if [ "$has_cpu" = true ] && [ "$has_gpu" = false ] && [ "$has_yica" = false ]; then
                    build_args="--cpu-only"
                elif [ "$has_gpu" = true ] && [ "$has_cpu" = false ] && [ "$has_yica" = false ]; then
                    build_args="--gpu-cuda"
                elif [ "$has_yica" = true ] && [ "$has_cpu" = false ] && [ "$has_gpu" = false ]; then
                    build_args="--yica-hardware"
                elif [ "$has_hybrid" = true ] || ([ "$has_cpu" = true ] && [ "$has_gpu" = true ]) || ([ "$has_cpu" = true ] && [ "$has_yica" = true ]) || ([ "$has_gpu" = true ] && [ "$has_yica" = true ]); then
                    build_args="--hybrid"
                else
                    build_args="--detect-auto"
                fi
                
                # 添加测试构建
                build_args="$build_args --with-tests"
                
                # 添加构建类型
                if [ "$BUILD_TYPE" = "Debug" ]; then
                    build_args="$build_args --debug"
                fi
                
                print_info "构建命令: ./build-flexible.sh $build_args"
                
                if ./build-flexible.sh $build_args; then
                    print_success "自动构建完成"
                else
                    print_error "自动构建失败"
                    exit 1
                fi
            else
                # 使用CMake直接构建
                print_info "使用CMake进行构建"
                
                # 创建构建目录
                mkdir -p "$BUILD_DIR"
                cd "$BUILD_DIR"
                
                # 构建参数
                local cmake_args="-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
                
                # 根据后端设置CMake参数
                for backend in "${BACKENDS[@]}"; do
                    case "$backend" in
                        "cpu")
                            cmake_args="$cmake_args -DBUILD_CPU_BACKEND=ON"
                            ;;
                        "gpu")
                            cmake_args="$cmake_args -DBUILD_GPU_BACKEND=ON"
                            ;;
                        "yica")
                            cmake_args="$cmake_args -DBUILD_YICA_BACKEND=ON"
                            ;;
                        "hybrid")
                            cmake_args="$cmake_args -DBUILD_HYBRID_BACKEND=ON"
                            ;;
                    esac
                done
                
                cmake_args="$cmake_args -DBUILD_TESTS=ON"
                
                print_info "CMake配置: cmake .. $cmake_args"
                
                # 查找可用的CMake配置文件
                local cmake_file=""
                if [ -f "../CMakeLists-working.txt" ]; then
                    cmake_file="../CMakeLists-working.txt"
                    print_info "使用可工作的CMake配置: $cmake_file"
                elif [ -f "../CMakeLists-modular.txt" ]; then
                    cmake_file="../CMakeLists-modular.txt"
                    print_info "使用模块化CMake配置: $cmake_file"
                elif [ -f "../CMakeLists.txt" ]; then
                    cmake_file="../CMakeLists.txt"
                    print_info "使用标准CMake配置: $cmake_file"
                else
                    print_error "找不到任何CMake配置文件"
                    exit 1
                fi
                
                # 创建符号链接（如果需要）
                if [ "$cmake_file" != "../CMakeLists.txt" ] && [ ! -f "../CMakeLists.txt" ]; then
                    ln -sf "$(basename "$cmake_file")" ../CMakeLists.txt
                    print_info "创建CMakeLists.txt符号链接"
                fi
                
                print_info "CMake配置: cmake $cmake_args .."
                
                if cmake $cmake_args ..; then
                    print_success "CMake配置完成"
                else
                    print_error "CMake配置失败"
                    exit 1
                fi
                
                # 编译
                print_info "开始编译 (并行度: $PARALLEL_JOBS)"
                if make -j$PARALLEL_JOBS; then
                    print_success "编译完成"
                else
                    print_error "编译失败"
                    exit 1
                fi
                
                cd ..
            fi
        else
            print_error "构建目录不存在: $BUILD_DIR"
            print_info "请使用以下选项之一:"
            print_info "  1. 添加 --auto-build 参数自动构建"
            print_info "  2. 手动运行 ./build-flexible.sh 构建项目"
            print_info "  3. 手动创建构建目录并配置CMake"
            exit 1
        fi
    else
        print_success "构建目录已存在: $BUILD_DIR"
    fi
}

# 调用自动构建检查
auto_build_if_needed

# 检查构建目录
if [ ! -d "$BUILD_DIR" ]; then
    print_error "构建目录仍然不存在: $BUILD_DIR"
    exit 1
fi

cd "$BUILD_DIR"

# 检测可用的后端
detect_available_backends() {
    local available=()
    
    if [ -f "libyica_cpu.so" ] || [ -f "libyica_cpu.a" ] || [ -f "yica_cpu_tests" ]; then
        available+=("cpu")
    fi
    
    if [ -f "libyica_gpu.so" ] || [ -f "libyica_gpu.a" ] || [ -f "yica_gpu_tests" ]; then
        available+=("gpu")
    fi
    
    if [ -f "libyica_hardware.so" ] || [ -f "libyica_hardware.a" ] || [ -f "yica_hardware_tests" ]; then
        available+=("yica")
    fi
    
    if [ -f "libyica_hybrid.so" ] || [ -f "libyica_hybrid.a" ] || [ -f "yica_hybrid_tests" ]; then
        available+=("hybrid")
    fi
    
    echo "${available[@]}"
}

AVAILABLE_BACKENDS=($(detect_available_backends))

print_info "可用后端: ${AVAILABLE_BACKENDS[*]}"
print_info "请求测试后端: ${BACKENDS[*]}"
print_info "测试类型: $TEST_TYPE"

# 验证请求的后端是否可用
for backend in "${BACKENDS[@]}"; do
    if [[ ! " ${AVAILABLE_BACKENDS[@]} " =~ " ${backend} " ]]; then
        print_warning "后端 '$backend' 不可用，跳过"
        continue
    fi
done

# 构建CTest命令
build_ctest_command() {
    local backend="$1"
    local test_type="$2"
    
    local cmd="ctest"
    
    # 并行执行
    cmd="$cmd -j $PARALLEL_JOBS"
    
    # 后端标签过滤
    case "$backend" in
        "cpu")
            cmd="$cmd -L cpu"
            ;;
        "gpu")
            cmd="$cmd -L gpu"
            ;;
        "yica")
            cmd="$cmd -L yica"
            ;;
        "hybrid")
            cmd="$cmd -L hybrid"
            ;;
    esac
    
    # 测试类型过滤
    case "$test_type" in
        "basic")
            cmd="$cmd -L basic"
            ;;
        "perf")
            cmd="$cmd -L performance"
            ;;
        "stress")
            cmd="$cmd -L stress"
            ;;
        "compat")
            cmd="$cmd -L compatibility"
            ;;
    esac
    
    # 自定义过滤
    if [ -n "$FILTER" ]; then
        cmd="$cmd -R '$FILTER'"
    fi
    
    if [ -n "$EXCLUDE" ]; then
        cmd="$cmd -E '$EXCLUDE'"
    fi
    
    # 超时设置
    if [ -n "$TIMEOUT" ]; then
        cmd="$cmd --timeout $TIMEOUT"
    fi
    
    # 输出选项
    if [ "$VERBOSE" = true ]; then
        cmd="$cmd -V"
    elif [ "$QUIET" = true ]; then
        cmd="$cmd -Q"
    fi
    
    # 报告格式
    if [ "$XML_OUTPUT" = true ]; then
        cmd="$cmd -T Test --output-junit ${backend}_test_results.xml"
    fi
    
    echo "$cmd"
}

# 运行单个后端测试
run_backend_tests() {
    local backend="$1"
    local test_type="$2"
    
    print_step "运行 ${backend} 后端测试 (类型: $test_type)"
    
    # 检查测试可执行文件
    local test_executable=""
    case "$backend" in
        "cpu")
            # 检查多种可能的测试可执行文件名
            if [ -f "yica_cpu_working_tests" ]; then
                test_executable="yica_cpu_working_tests"
            elif [ -f "yica_cpu_tests" ]; then
                test_executable="yica_cpu_tests"
            elif [ -f "tests/cpu/yica_cpu_working_tests" ]; then
                test_executable="tests/cpu/yica_cpu_working_tests"
            fi
            ;;
        "gpu")
            if [ -f "yica_gpu_working_tests" ]; then
                test_executable="yica_gpu_working_tests"
            elif [ -f "yica_gpu_tests" ]; then
                test_executable="yica_gpu_tests"
            fi
            ;;
        "yica")
            if [ -f "yica_hardware_working_tests" ]; then
                test_executable="yica_hardware_working_tests"
            elif [ -f "yica_hardware_tests" ]; then
                test_executable="yica_hardware_tests"
            fi
            ;;
        "hybrid")
            if [ -f "yica_hybrid_working_tests" ]; then
                test_executable="yica_hybrid_working_tests"
            elif [ -f "yica_hybrid_tests" ]; then
                test_executable="yica_hybrid_tests"
            fi
            ;;
    esac
    
    if [ ! -f "$test_executable" ]; then
        print_warning "测试可执行文件不存在: $test_executable"
        print_info "尝试直接运行CTest..."
        
        # 尝试使用CTest
        local cmd=$(build_ctest_command "$backend" "$test_type")
        
        print_info "执行命令: $cmd"
        
        local start_time=$(date +%s)
        local result=0
        
        if eval "$cmd" 2>/dev/null; then
            local end_time=$(date +%s)
            local duration=$((end_time - start_time))
            print_success "${backend} 后端测试通过 (耗时: ${duration}s)"
        else
            result=$?
            local end_time=$(date +%s)
            local duration=$((end_time - start_time))
            print_warning "${backend} 后端测试未找到或失败 (耗时: ${duration}s)"
            print_info "这可能是因为该后端未被构建或测试未被启用"
        fi
        
        return $result
    fi
    
    # 设置环境变量
    case "$backend" in
        "gpu")
            export CUDA_VISIBLE_DEVICES=0
            ;;
        "yica")
            export YICA_DEVICE_PATH=/dev/yica0
            ;;
        "hybrid")
            export CUDA_VISIBLE_DEVICES=0
            export YICA_DEVICE_PATH=/dev/yica0
            ;;
    esac
    
    # 构建并运行测试命令
    local cmd=$(build_ctest_command "$backend" "$test_type")
    
    print_info "执行命令: $cmd"
    
    local start_time=$(date +%s)
    local result=0
    
    if eval "$cmd"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_success "${backend} 后端测试通过 (耗时: ${duration}s)"
    else
        result=$?
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_error "${backend} 后端测试失败 (耗时: ${duration}s)"
    fi
    
    return $result
}

# 生成测试报告
generate_test_report() {
    print_step "生成测试报告"
    
    local report_file="test_report_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "YICA 硬件后端测试报告"
        echo "======================"
        echo "生成时间: $(date)"
        echo "测试后端: ${BACKENDS[*]}"
        echo "测试类型: $TEST_TYPE"
        echo "构建目录: $BUILD_DIR"
        echo "构建类型: $BUILD_TYPE"
        echo ""
        
        for backend in "${BACKENDS[@]}"; do
            if [[ " ${AVAILABLE_BACKENDS[@]} " =~ " ${backend} " ]]; then
                echo "${backend} 后端测试结果:"
                echo "------------------------"
                
                # 提取测试结果
                if [ -f "Testing/Temporary/LastTest.log" ]; then
                    grep -A 5 -B 5 "$backend" Testing/Temporary/LastTest.log || echo "无详细日志"
                fi
                echo ""
            fi
        done
        
        echo "测试总结:"
        echo "--------"
        ctest -N 2>/dev/null | grep "Total Tests:" || echo "无法获取测试总数"
        
    } > "$report_file"
    
    print_success "测试报告已生成: $report_file"
}

# 主执行流程
main() {
    local total_backends=${#BACKENDS[@]}
    local failed_backends=()
    local passed_backends=()
    local skipped_backends=()
    
    print_info "开始运行 $total_backends 个后端的测试"
    
    for backend in "${BACKENDS[@]}"; do
        if [[ ! " ${AVAILABLE_BACKENDS[@]} " =~ " ${backend} " ]]; then
            print_warning "跳过不可用的后端: $backend"
            skipped_backends+=("$backend")
            continue
        fi
        
        echo
        if run_backend_tests "$backend" "$TEST_TYPE"; then
            passed_backends+=("$backend")
        else
            failed_backends+=("$backend")
        fi
    done
    
    echo
    print_step "测试结果总结"
    
    if [ ${#passed_backends[@]} -gt 0 ]; then
        print_success "通过的后端 (${#passed_backends[@]}): ${passed_backends[*]}"
    fi
    
    if [ ${#skipped_backends[@]} -gt 0 ]; then
        print_warning "跳过的后端 (${#skipped_backends[@]}): ${skipped_backends[*]}"
    fi
    
    if [ ${#failed_backends[@]} -gt 0 ]; then
        print_error "失败的后端 (${#failed_backends[@]}): ${failed_backends[*]}"
    fi
    
    # 生成报告
    if [ "$XML_OUTPUT" = true ] || [ "$JSON_OUTPUT" = true ]; then
        generate_test_report
    fi
    
    # 返回适当的退出码
    if [ ${#failed_backends[@]} -gt 0 ]; then
        exit 1
    else
        if [ ${#passed_backends[@]} -gt 0 ]; then
            print_success "所有可用测试通过！"
        else
            print_warning "没有可运行的测试"
        fi
        exit 0
    fi
}

# 运行主程序
main "$@" 
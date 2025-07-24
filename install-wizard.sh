#!/bin/bash

# YICA 安装向导 - 帮助用户选择最适合的版本
# 交互式界面，自动检测硬件并推荐最优配置

set -e

# 颜色和样式定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# 图标定义
CHECKMARK="✓"
CROSSMARK="✗"
ARROW="→"
STAR="★"
INFO="ℹ"
WARNING="⚠"

print_header() {
    clear
    echo -e "${CYAN}"
    cat << "EOF"
 ╭─────────────────────────────────────────────────────────────╮
 │                                                             │
 │    █╗   ██╗██╗ ██████╗ █████╗     ██╗███╗   ██╗███████╗     │
 │    ╚██╗ ██╔╝██║██╔════╝██╔══██╗    ██║████╗  ██║██╔════╝    │
 │     ╚████╔╝ ██║██║     ███████║    ██║██╔██╗ ██║███████╗    │
 │      ╚██╔╝  ██║██║     ██╔══██║    ██║██║╚██╗██║╚════██║    │
 │       ██║   ██║╚██████╗██║  ██║    ██║██║ ╚████║███████║    │
 │       ╚═╝   ╚═╝ ╚═════╝╚═╝  ╚═╝    ╚═╝╚═╝  ╚═══╝╚══════╝    │
 │                                                             │
 │             YICA 存算一体架构优化器 - 安装向导               │
 │                                                             │
 ╰─────────────────────────────────────────────────────────────╯
EOF
    echo -e "${NC}"
    echo -e "${WHITE}欢迎使用 YICA 安装向导！我们将帮您选择最适合的版本${NC}"
    echo
}

print_step() {
    echo -e "${BLUE}${ARROW} $1${NC}"
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

print_info() {
    echo -e "${CYAN}${INFO} $1${NC}"
}

print_highlight() {
    echo -e "${PURPLE}${STAR} $1${NC}"
}

# 等待用户输入
wait_for_input() {
    echo
    echo -e "${WHITE}按回车键继续...${NC}"
    read -r
}

# 选择菜单函数
show_menu() {
    local title="$1"
    shift
    local options=("$@")
    
    echo -e "${WHITE}$title${NC}"
    echo
    
    for i in "${!options[@]}"; do
        echo -e "${CYAN}$((i+1))${NC}. ${options[$i]}"
    done
    echo
    echo -e "${CYAN}0${NC}. 退出"
    echo
    echo -n -e "${WHITE}请选择 [0-${#options[@]}]: ${NC}"
}

# 硬件检测函数
detect_system_info() {
    print_step "检测系统信息..."
    
    # 操作系统检测
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="Linux"
        if [ -f /etc/os-release ]; then
            DISTRO=$(grep '^NAME=' /etc/os-release | cut -d'"' -f2)
            VERSION=$(grep '^VERSION=' /etc/os-release | cut -d'"' -f2)
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macOS"
        DISTRO="macOS"
        VERSION=$(sw_vers -productVersion)
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        OS="Windows"
        DISTRO="Windows"
        VERSION=$(cmd.exe /c ver 2>/dev/null | grep -o '[0-9]*\.[0-9]*\.[0-9]*')
    else
        OS="Unknown"
        DISTRO="Unknown"
        VERSION="Unknown"
    fi
    
    # CPU信息
    if command -v nproc &> /dev/null; then
        CPU_CORES=$(nproc)
    elif command -v sysctl &> /dev/null; then
        CPU_CORES=$(sysctl -n hw.ncpu)
    else
        CPU_CORES="Unknown"
    fi
    
    # 内存信息
    if [ -f /proc/meminfo ]; then
        MEMORY_GB=$(awk '/MemTotal/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)
    elif command -v sysctl &> /dev/null; then
        MEMORY_BYTES=$(sysctl -n hw.memsize)
        MEMORY_GB=$(echo "scale=1; $MEMORY_BYTES/1024/1024/1024" | bc)
    else
        MEMORY_GB="Unknown"
    fi
    
    # GPU检测
    GPU_INFO="None"
    HAS_NVIDIA_GPU=false
    HAS_CUDA=false
    CUDA_VERSION=""
    
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            HAS_NVIDIA_GPU=true
            GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        fi
    fi
    
    if command -v nvcc &> /dev/null; then
        HAS_CUDA=true
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    fi
    
    # YICA硬件检测（模拟）
    HAS_YICA_HARDWARE=false
    YICA_VERSION=""
    
    if [ -d "/opt/yica" ] || [ -f "/sys/class/yica/device0" ]; then
        HAS_YICA_HARDWARE=true
        YICA_VERSION="1.0"  # 模拟版本
    fi
    
    print_success "系统信息检测完成"
}

# 显示系统信息
show_system_info() {
    print_header
    print_step "系统信息总览"
    echo
    
    echo -e "${WHITE}操作系统:${NC} $OS ($DISTRO $VERSION)"
    echo -e "${WHITE}CPU核心数:${NC} $CPU_CORES"
    echo -e "${WHITE}内存大小:${NC} ${MEMORY_GB}GB"
    echo
    
    echo -e "${WHITE}GPU信息:${NC}"
    if [ "$HAS_NVIDIA_GPU" = true ]; then
        print_success "NVIDIA GPU: $GPU_INFO"
    else
        print_warning "未检测到NVIDIA GPU"
    fi
    
    echo -e "${WHITE}CUDA支持:${NC}"
    if [ "$HAS_CUDA" = true ]; then
        print_success "CUDA $CUDA_VERSION 已安装"
    else
        print_warning "CUDA未安装"
    fi
    
    echo -e "${WHITE}YICA硬件:${NC}"
    if [ "$HAS_YICA_HARDWARE" = true ]; then
        print_success "YICA硬件 v$YICA_VERSION 已检测"
    else
        print_warning "未检测到YICA硬件"
    fi
    
    wait_for_input
}

# 推荐版本
recommend_version() {
    print_header
    print_step "版本推荐分析"
    echo
    
    # 分析用户需求
    echo -e "${WHITE}基于您的系统配置，我们推荐以下版本：${NC}"
    echo
    
    local recommendations=()
    local reasons=()
    
    # 推荐逻辑
    if [ "$HAS_YICA_HARDWARE" = true ]; then
        recommendations+=("YICA硬件版本")
        reasons+=("您有YICA硬件，可获得最佳性能和能效")
    fi
    
    if [ "$HAS_NVIDIA_GPU" = true ] && [ "$HAS_CUDA" = true ]; then
        recommendations+=("GPU CUDA版本")
        reasons+=("您有NVIDIA GPU和CUDA，可获得优秀的加速性能")
    fi
    
    if [ "$HAS_NVIDIA_GPU" = true ] && [ "$HAS_YICA_HARDWARE" = true ]; then
        recommendations+=("混合多后端版本")
        reasons+=("您同时拥有GPU和YICA硬件，混合版本可自动选择最优后端")
    fi
    
    # 总是推荐CPU版本作为备选
    recommendations+=("CPU版本")
    reasons+=("通用兼容，无特殊硬件要求，适合开发和测试")
    
    # 显示推荐
    for i in "${!recommendations[@]}"; do
        if [ $i -eq 0 ]; then
            print_highlight "${recommendations[$i]} (推荐)"
            echo -e "   ${CYAN}${reasons[$i]}${NC}"
        else
            echo -e "${GREEN}${CHECKMARK}${NC} ${recommendations[$i]}"
            echo -e "   ${reasons[$i]}"
        fi
        echo
    done
    
    wait_for_input
}

# 版本选择菜单
version_selection_menu() {
    while true; do
        print_header
        
        local options=(
            "CPU版本 - 纯CPU，无GPU依赖，通用兼容"
            "GPU CUDA版本 - NVIDIA GPU加速，高性能推理"
            "YICA硬件版本 - 专用YICA硬件，存算一体优化"
            "混合多后端版本 - 支持多种硬件，自动选择最优后端"
            "查看详细对比"
            "自定义构建选项"
        )
        
        show_menu "请选择要安装的YICA版本：" "${options[@]}"
        
        read -r choice
        
        case $choice in
            1)
                install_cpu_version
                break
                ;;
            2)
                if [ "$HAS_CUDA" = true ]; then
                    install_gpu_version
                    break
                else
                    print_warning "您的系统未安装CUDA，无法安装GPU版本"
                    echo "是否要先安装CUDA？(y/n)"
                    read -r install_cuda
                    if [[ $install_cuda =~ ^[Yy]$ ]]; then
                        show_cuda_installation_guide
                    fi
                fi
                ;;
            3)
                if [ "$HAS_YICA_HARDWARE" = true ]; then
                    install_yica_version
                    break
                else
                    print_warning "未检测到YICA硬件"
                    print_info "YICA硬件版本需要专用的YICA硬件支持"
                    wait_for_input
                fi
                ;;
            4)
                install_hybrid_version
                break
                ;;
            5)
                show_version_comparison
                ;;
            6)
                show_custom_build_options
                break
                ;;
            0)
                echo "感谢使用YICA安装向导！"
                exit 0
                ;;
            *)
                print_error "无效选择，请重试"
                sleep 1
                ;;
        esac
    done
}

# 版本对比
show_version_comparison() {
    print_header
    print_step "版本详细对比"
    echo
    
    cat << EOF
╭─────────────────┬─────────────┬─────────────┬─────────────┬─────────────╮
│      特性       │   CPU版本   │  GPU版本    │  YICA版本   │  混合版本   │
├─────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│   硬件要求      │     最低    │   NVIDIA    │   YICA      │    多种     │
│                 │             │    GPU      │   硬件      │             │
├─────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│   安装复杂度    │     简单    │    中等     │    中等     │    复杂     │
├─────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│   推理性能      │    基线     │   3-10x     │   5-20x     │    最优     │
├─────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│   内存使用      │    基线     │  基线+GPU   │   基线×0.6  │   动态      │
├─────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│   功耗消耗      │    基线     │  基线×3-5   │   基线×0.3  │   自适应    │
├─────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│   包大小        │    15MB     │    150MB    │    80MB     │    200MB    │
├─────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│   适用场景      │  开发测试   │  GPU推理    │  专用硬件   │   研究      │
╰─────────────────┴─────────────┴─────────────┴─────────────┴─────────────╯
EOF
    
    echo
    print_info "选择建议："
    echo "• 开发和测试：选择CPU版本"
    echo "• 生产推理且有GPU：选择GPU版本"
    echo "• 有YICA硬件：选择YICA版本"
    echo "• 研究对比：选择混合版本"
    
    wait_for_input
}

# CPU版本安装
install_cpu_version() {
    print_header
    print_step "安装CPU版本"
    echo
    
    print_info "CPU版本特点："
    echo "• 无GPU依赖，通用兼容"
    echo "• 使用OpenMP并行化"
    echo "• 支持AVX2/AVX512优化"
    echo "• 包大小约15MB"
    echo
    
    case $OS in
        "Linux")
            if command -v apt &> /dev/null; then
                print_step "使用APT安装..."
                echo "sudo apt update"
                echo "sudo apt install yica-optimizer-cpu"
            elif command -v yum &> /dev/null; then
                print_step "使用YUM安装..."
                echo "sudo yum install yica-optimizer-cpu"
            else
                print_step "使用Docker安装..."
                echo "docker pull yica/yica-optimizer:cpu-latest"
                echo "docker run -it yica/yica-optimizer:cpu-latest"
            fi
            ;;
        "macOS")
            print_step "使用Homebrew安装..."
            echo "brew tap yica/tap"
            echo "brew install yica-optimizer"
            ;;
        *)
            print_step "使用Docker安装..."
            echo "docker pull yica/yica-optimizer:cpu-latest"
            echo "docker run -it yica/yica-optimizer:cpu-latest"
            ;;
    esac
    
    echo
    print_success "安装命令已显示，请复制执行"
    wait_for_input
}

# GPU版本安装
install_gpu_version() {
    print_header
    print_step "安装GPU版本"
    echo
    
    print_info "GPU版本特点："
    echo "• 需要NVIDIA GPU和CUDA"
    echo "• 高性能推理加速"
    echo "• 支持混合精度"
    echo "• 包大小约150MB"
    echo
    
    if [ "$HAS_CUDA" = true ]; then
        print_success "检测到CUDA $CUDA_VERSION"
        
        case $OS in
            "Linux")
                print_step "安装GPU版本..."
                echo "sudo apt install yica-optimizer-gpu-cuda${CUDA_VERSION//./}"
                ;;
            *)
                print_step "使用Docker安装..."
                echo "docker pull yica/yica-optimizer:gpu-cuda${CUDA_VERSION//./}"
                echo "docker run --gpus all -it yica/yica-optimizer:gpu-cuda${CUDA_VERSION//./}"
                ;;
        esac
    else
        print_warning "未检测到CUDA，请先安装CUDA"
        show_cuda_installation_guide
    fi
    
    wait_for_input
}

# YICA版本安装
install_yica_version() {
    print_header
    print_step "安装YICA硬件版本"
    echo
    
    print_info "YICA硬件版本特点："
    echo "• 专为YICA硬件优化"
    echo "• 存算一体计算"
    echo "• 超低功耗设计"
    echo "• 包大小约80MB"
    echo
    
    print_step "安装YICA版本..."
    echo "sudo apt install yica-optimizer-hardware"
    echo
    
    print_info "注意事项："
    echo "• 需要YICA硬件驱动"
    echo "• 需要YICA SDK"
    echo "• 建议配置YCCL通信库"
    
    wait_for_input
}

# 混合版本安装
install_hybrid_version() {
    print_header
    print_step "安装混合多后端版本"
    echo
    
    print_info "混合版本特点："
    echo "• 支持CPU+GPU+YICA多后端"
    echo "• 自动选择最优后端"
    echo "• 适合研究和对比"
    echo "• 包大小约200MB"
    echo
    
    print_step "安装混合版本..."
    echo "sudo apt install yica-optimizer-hybrid"
    echo
    
    print_warning "混合版本需要较多依赖，安装时间较长"
    
    wait_for_input
}

# CUDA安装指南
show_cuda_installation_guide() {
    print_header
    print_step "CUDA安装指南"
    echo
    
    print_info "安装CUDA的步骤："
    echo
    
    case $OS in
        "Linux")
            echo "1. 下载CUDA Toolkit："
            echo "   wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run"
            echo
            echo "2. 安装CUDA："
            echo "   sudo sh cuda_12.1.1_530.30.02_linux.run"
            echo
            echo "3. 设置环境变量："
            echo "   export PATH=/usr/local/cuda/bin:\$PATH"
            echo "   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
            ;;
        *)
            echo "请访问NVIDIA官网下载适合您系统的CUDA Toolkit："
            echo "https://developer.nvidia.com/cuda-downloads"
            ;;
    esac
    
    wait_for_input
}

# 自定义构建选项
show_custom_build_options() {
    print_header
    print_step "自定义构建选项"
    echo
    
    print_info "如果预构建版本不满足您的需求，可以使用自定义构建："
    echo
    
    echo "1. 自动检测构建："
    echo "   ./build-flexible.sh --detect-auto"
    echo
    echo "2. 指定硬件配置："
    echo "   ./build-flexible.sh --cpu-only"
    echo "   ./build-flexible.sh --gpu-cuda"
    echo "   ./build-flexible.sh --yica-hardware"
    echo
    echo "3. 包含测试和示例："
    echo "   ./build-flexible.sh --detect-auto --with-tests --with-examples"
    echo
    echo "4. 指定安装路径："
    echo "   ./build-flexible.sh --detect-auto --install-prefix /opt/yica"
    echo
    echo "5. 生成不同格式的包："
    echo "   ./build-flexible.sh --detect-auto --package-format deb"
    echo "   ./build-flexible.sh --detect-auto --package-format docker"
    
    wait_for_input
}

# 安装后验证
post_install_verification() {
    print_header
    print_step "安装验证"
    echo
    
    print_info "安装完成后，您可以运行以下命令验证："
    echo
    echo "1. 检查版本："
    echo "   yica-optimizer --version"
    echo
    echo "2. 运行基本测试："
    echo "   yica-optimizer --test"
    echo
    echo "3. 查看硬件信息："
    echo "   yica-optimizer --hardware-info"
    echo
    echo "4. 运行示例："
    echo "   yica-optimizer --example matmul"
    
    wait_for_input
}

# 主菜单
main_menu() {
    while true; do
        print_header
        
        local options=(
            "查看系统信息"
            "获取版本推荐"
            "选择安装版本"
            "查看安装后验证"
            "查看帮助文档"
        )
        
        show_menu "主菜单 - 请选择操作：" "${options[@]}"
        
        read -r choice
        
        case $choice in
            1)
                show_system_info
                ;;
            2)
                recommend_version
                ;;
            3)
                version_selection_menu
                ;;
            4)
                post_install_verification
                ;;
            5)
                show_help_docs
                ;;
            0)
                print_success "感谢使用YICA安装向导！"
                exit 0
                ;;
            *)
                print_error "无效选择，请重试"
                sleep 1
                ;;
        esac
    done
}

# 帮助文档
show_help_docs() {
    print_header
    print_step "帮助文档"
    echo
    
    print_info "相关文档链接："
    echo
    echo "• 官方文档: https://docs.yica.ai/"
    echo "• 快速开始: https://docs.yica.ai/quickstart/"
    echo "• API参考: https://docs.yica.ai/api/"
    echo "• 示例代码: https://github.com/yica-project/examples"
    echo "• 问题反馈: https://github.com/yica-project/yica-optimizer/issues"
    echo
    
    print_info "社区支持："
    echo "• 论坛: https://forum.yica.ai/"
    echo "• QQ群: 123456789"
    echo "• 微信群: 扫描二维码加入"
    
    wait_for_input
}

# 主程序入口
main() {
    # 检查依赖
    if ! command -v curl &> /dev/null && ! command -v wget &> /dev/null; then
        print_error "需要curl或wget来下载软件包"
        exit 1
    fi
    
    # 检测系统信息
    detect_system_info
    
    # 显示主菜单
    main_menu
}

# 运行主程序
main "$@" 
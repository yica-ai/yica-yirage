#!/bin/bash

# YICA CPU Docker运行脚本
# 不依赖GPU驱动的CPU版本运行脚本

set -e

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 默认配置
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yica-cpu.yml"
SERVICE_NAME="yica-runtime-cpu"
SIMULATE_GPU=true
DETACHED=false
BUILD=false
BENCHMARK=false

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 检查依赖
check_dependencies() {
    log_step "检查系统依赖..."
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    
    # 检查Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi
    
    log_info "依赖检查完成 - 纯CPU模式，无需GPU驱动"
}

# 检查系统资源
check_system_resources() {
    log_step "检查系统资源..."
    
    # 检查CPU核心数
    local cpu_cores=$(nproc)
    log_info "可用CPU核心数: $cpu_cores"
    
    if [ "$cpu_cores" -lt 4 ]; then
        log_warn "CPU核心数较少，建议至少4核心以获得最佳性能"
    fi
    
    # 检查内存
    local memory_gb=$(free -g | awk '/^Mem:/{print $2}')
    log_info "可用内存: ${memory_gb}GB"
    
    if [ "$memory_gb" -lt 8 ]; then
        log_warn "内存较少，建议至少8GB内存以获得最佳性能"
    fi
    
    # 检查磁盘空间
    local disk_free_gb=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | sed 's/G//')
    log_info "可用磁盘空间: ${disk_free_gb}GB"
    
    if [ "$disk_free_gb" -lt 10 ]; then
        log_warn "磁盘空间不足，建议至少10GB可用空间"
    fi
}

# 构建镜像
build_images() {
    log_step "构建YICA CPU Docker镜像..."
    
    cd "$PROJECT_ROOT"
    
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$COMPOSE_FILE" build
    else
        docker compose -f "$COMPOSE_FILE" build
    fi
    
    log_info "镜像构建完成"
}

# 启动服务
start_services() {
    log_step "启动YICA CPU运行时服务..."
    
    cd "$PROJECT_ROOT"
    
    local compose_cmd=""
    if command -v docker-compose &> /dev/null; then
        compose_cmd="docker-compose"
    else
        compose_cmd="docker compose"
    fi
    
    # 设置环境变量
    export YICA_GPU_SIMULATION=$SIMULATE_GPU
    export YICA_CPU_ONLY=true
    
    if [ "$DETACHED" = true ]; then
        $compose_cmd -f "$COMPOSE_FILE" up -d
        log_info "CPU服务已在后台启动"
        
        # 显示服务状态
        sleep 5
        $compose_cmd -f "$COMPOSE_FILE" ps
        
        # 显示访问信息
        echo ""
        log_info "服务访问地址:"
        echo "  • YICA Runtime API:    http://localhost:8080"
        echo "  • Performance Monitor: http://localhost:8081"
        echo "  • ML Optimizer API:    http://localhost:8082"
        echo "  • Grafana Dashboard:   http://localhost:3000 (admin/yica2024)"
        echo "  • Jupyter Lab:         http://localhost:8888 (token: yica2024)"
        echo ""
        echo "特性说明:"
        echo "  • 纯CPU运行，无需GPU驱动"
        echo "  • GPU行为模拟: $([ "$SIMULATE_GPU" = "true" ] && echo "启用" || echo "禁用")"
        echo "  • 多线程优化: OpenMP + SIMD"
        echo "  • 机器学习: CPU优化算法"
        echo ""
        echo "查看日志: docker logs yica-runtime-cpu"
        echo "停止服务: $0 stop"
    else
        $compose_cmd -f "$COMPOSE_FILE" up
    fi
}

# 启动基准测试
start_benchmark() {
    log_step "启动CPU基准测试..."
    
    cd "$PROJECT_ROOT"
    
    local compose_cmd=""
    if command -v docker-compose &> /dev/null; then
        compose_cmd="docker-compose"
    else
        compose_cmd="docker compose"
    fi
    
    # 启动基准测试服务
    $compose_cmd -f "$COMPOSE_FILE" --profile benchmark up yica-benchmark-cpu
    
    log_info "基准测试完成，结果保存在yica-cpu-benchmarks卷中"
}

# 停止服务
stop_services() {
    log_step "停止YICA CPU服务..."
    
    cd "$PROJECT_ROOT"
    
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$COMPOSE_FILE" down
    else
        docker compose -f "$COMPOSE_FILE" down
    fi
    
    log_info "服务已停止"
}

# 清理资源
clean_resources() {
    log_step "清理Docker资源..."
    
    cd "$PROJECT_ROOT"
    
    # 停止并删除容器
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$COMPOSE_FILE" down -v --remove-orphans
        docker-compose -f "$COMPOSE_FILE" --profile benchmark down -v --remove-orphans
    else
        docker compose -f "$COMPOSE_FILE" down -v --remove-orphans
        docker compose -f "$COMPOSE_FILE" --profile benchmark down -v --remove-orphans
    fi
    
    # 删除镜像
    docker rmi yica-optimizer:cpu-latest yica-monitor:cpu-latest yica-jupyter:cpu-latest 2>/dev/null || true
    
    # 清理未使用的资源
    docker system prune -f
    
    log_info "清理完成"
}

# 显示状态
show_status() {
    log_step "显示服务状态..."
    
    cd "$PROJECT_ROOT"
    
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$COMPOSE_FILE" ps
    else
        docker compose -f "$COMPOSE_FILE" ps
    fi
    
    # 显示资源使用情况
    echo ""
    log_info "容器资源使用情况:"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" \
        yica-runtime-cpu yica-monitor-cpu yica-jupyter-cpu 2>/dev/null || true
}

# 显示日志
show_logs() {
    local service=${1:-$SERVICE_NAME}
    log_step "显示 $service 服务日志..."
    
    cd "$PROJECT_ROOT"
    
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$COMPOSE_FILE" logs -f "$service"
    else
        docker compose -f "$COMPOSE_FILE" logs -f "$service"
    fi
}

# 进入容器
enter_container() {
    local service=${1:-$SERVICE_NAME}
    log_step "进入 $service 容器..."
    
    local container_name=""
    case $service in
        "yica-runtime-cpu"|"runtime")
            container_name="yica-runtime-cpu"
            ;;
        "monitor")
            container_name="yica-monitor-cpu"
            ;;
        "jupyter")
            container_name="yica-jupyter-cpu"
            ;;
        *)
            container_name="yica-runtime-cpu"
            ;;
    esac
    
    docker exec -it "$container_name" /bin/bash
}

# 运行性能测试
run_performance_test() {
    log_step "运行性能测试..."
    
    # 检查CPU性能
    docker exec yica-runtime-cpu /bin/bash -c "
        echo '=== CPU信息 ==='
        lscpu | grep -E 'Model name|CPU\\(s\\)|Thread|Core'
        echo ''
        echo '=== 内存信息 ==='
        free -h
        echo ''
        echo '=== 模拟GPU信息 ==='
        nvidia-smi 2>/dev/null || echo 'GPU模拟未启用'
        echo ''
        echo '=== YICA运行时状态 ==='
        curl -s http://localhost:8080/health || echo '运行时未响应'
    "
}

# 显示帮助
show_help() {
    echo "YICA CPU Docker运行脚本"
    echo ""
    echo "使用方法: $0 [选项] [命令]"
    echo ""
    echo "命令:"
    echo "  start       启动YICA CPU服务 (默认)"
    echo "  stop        停止YICA CPU服务"
    echo "  restart     重启YICA CPU服务"
    echo "  status      显示服务状态"
    echo "  logs        显示服务日志"
    echo "  shell       进入容器shell"
    echo "  test        运行性能测试"
    echo "  benchmark   运行基准测试"
    echo "  clean       清理Docker资源"
    echo ""
    echo "选项:"
    echo "  -d, --detached        后台运行"
    echo "  -b, --build           重新构建镜像"
    echo "  --no-simulation       禁用GPU模拟"
    echo "  --service=NAME        指定服务名称 (runtime|monitor|jupyter)"
    echo "  -h, --help            显示此帮助信息"
    echo ""
    echo "特性说明:"
    echo "  • 纯CPU运行，无需安装GPU驱动"
    echo "  • 支持GPU行为模拟，兼容GPU代码"
    echo "  • 多线程优化: OpenMP + SIMD"
    echo "  • CPU优化的机器学习算法"
    echo "  • 完整的监控和可视化"
    echo ""
    echo "示例:"
    echo "  $0                        # 启动CPU服务 (前台运行)"
    echo "  $0 -d                     # 后台启动服务"
    echo "  $0 -b start               # 重新构建并启动"
    echo "  $0 --no-simulation start  # 禁用GPU模拟启动"
    echo "  $0 logs                   # 查看运行时日志"
    echo "  $0 shell                  # 进入运行时容器"
    echo "  $0 test                   # 运行性能测试"
    echo "  $0 benchmark              # 运行基准测试"
    echo "  $0 clean                  # 清理资源"
}

# 解析命令行参数
COMMAND="start"
while [[ $# -gt 0 ]]; do
    case $1 in
        start|stop|restart|status|logs|shell|test|benchmark|clean)
            COMMAND="$1"
            shift
            ;;
        -d|--detached)
            DETACHED=true
            shift
            ;;
        -b|--build)
            BUILD=true
            shift
            ;;
        --no-simulation)
            SIMULATE_GPU=false
            shift
            ;;
        --service=*)
            SERVICE_NAME="${1#*=}"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 主函数
main() {
    log_info "YICA CPU Docker运行脚本"
    log_info "项目目录: $PROJECT_ROOT"
    log_info "CPU模式: 纯CPU运行，无需GPU驱动"
    log_info "GPU模拟: $SIMULATE_GPU"
    echo ""
    
    # 检查依赖
    check_dependencies
    
    # 检查系统资源
    check_system_resources
    
    # 执行命令
    case $COMMAND in
        "start")
            if [ "$BUILD" = true ]; then
                build_images
            fi
            start_services
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            stop_services
            sleep 2
            if [ "$BUILD" = true ]; then
                build_images
            fi
            start_services
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs "$SERVICE_NAME"
            ;;
        "shell")
            enter_container "$SERVICE_NAME"
            ;;
        "test")
            run_performance_test
            ;;
        "benchmark")
            start_benchmark
            ;;
        "clean")
            clean_resources
            ;;
        *)
            log_error "未知命令: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@" 
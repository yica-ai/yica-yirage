#!/bin/bash

# YICA GPU Docker运行脚本
# 一键启动支持NVIDIA GPU的YICA运行时优化器

set -e

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 默认配置
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yica-gpu.yml"
SERVICE_NAME="yica-runtime"
GPU_ENABLED=true
DETACHED=false
BUILD=false
CLEAN=false

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
    
    # 检查NVIDIA Docker支持
    if [ "$GPU_ENABLED" = true ]; then
        if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
            log_error "NVIDIA Docker支持未正确配置"
            log_error "请确保已安装nvidia-docker2和NVIDIA Container Toolkit"
            exit 1
        fi
        log_info "NVIDIA Docker支持检查通过"
    fi
    
    log_info "依赖检查完成"
}

# 检查GPU状态
check_gpu_status() {
    if [ "$GPU_ENABLED" = true ]; then
        log_step "检查GPU状态..."
        
        if command -v nvidia-smi &> /dev/null; then
            echo "可用GPU设备:"
            nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
            echo ""
        else
            log_warn "nvidia-smi不可用，无法显示GPU信息"
        fi
    fi
}

# 构建镜像
build_images() {
    log_step "构建YICA Docker镜像..."
    
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
    log_step "启动YICA运行时服务..."
    
    cd "$PROJECT_ROOT"
    
    local compose_cmd=""
    if command -v docker-compose &> /dev/null; then
        compose_cmd="docker-compose"
    else
        compose_cmd="docker compose"
    fi
    
    if [ "$DETACHED" = true ]; then
        $compose_cmd -f "$COMPOSE_FILE" up -d
        log_info "服务已在后台启动"
        
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
        echo "查看日志: docker logs yica-runtime-gpu"
        echo "停止服务: $0 --stop"
    else
        $compose_cmd -f "$COMPOSE_FILE" up
    fi
}

# 停止服务
stop_services() {
    log_step "停止YICA服务..."
    
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
    else
        docker compose -f "$COMPOSE_FILE" down -v --remove-orphans
    fi
    
    # 删除镜像
    docker rmi yica-optimizer:gpu-latest yica-monitor:latest yica-jupyter:gpu-latest 2>/dev/null || true
    
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
        "yica-runtime"|"runtime")
            container_name="yica-runtime-gpu"
            ;;
        "monitor")
            container_name="yica-monitor"
            ;;
        "jupyter")
            container_name="yica-jupyter-gpu"
            ;;
        *)
            container_name="yica-runtime-gpu"
            ;;
    esac
    
    docker exec -it "$container_name" /bin/bash
}

# 显示帮助
show_help() {
    echo "YICA GPU Docker运行脚本"
    echo ""
    echo "使用方法: $0 [选项] [命令]"
    echo ""
    echo "命令:"
    echo "  start     启动YICA服务 (默认)"
    echo "  stop      停止YICA服务"
    echo "  restart   重启YICA服务"
    echo "  status    显示服务状态"
    echo "  logs      显示服务日志"
    echo "  shell     进入容器shell"
    echo "  clean     清理Docker资源"
    echo ""
    echo "选项:"
    echo "  -d, --detached    后台运行"
    echo "  -b, --build       重新构建镜像"
    echo "  --no-gpu          禁用GPU支持"
    echo "  --service=NAME    指定服务名称 (runtime|monitor|jupyter)"
    echo "  -h, --help        显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                      # 启动YICA服务 (前台运行)"
    echo "  $0 -d                   # 后台启动服务"
    echo "  $0 -b start             # 重新构建并启动"
    echo "  $0 logs                 # 查看运行时日志"
    echo "  $0 shell                # 进入运行时容器"
    echo "  $0 --service=jupyter shell  # 进入Jupyter容器"
    echo "  $0 stop                 # 停止服务"
    echo "  $0 clean                # 清理资源"
}

# 解析命令行参数
COMMAND="start"
while [[ $# -gt 0 ]]; do
    case $1 in
        start|stop|restart|status|logs|shell|clean)
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
        --no-gpu)
            GPU_ENABLED=false
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
    log_info "YICA GPU Docker运行脚本"
    log_info "项目目录: $PROJECT_ROOT"
    log_info "GPU支持: $GPU_ENABLED"
    echo ""
    
    # 检查依赖
    check_dependencies
    
    # 检查GPU状态
    check_gpu_status
    
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
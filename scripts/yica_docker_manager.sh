#!/bin/bash
# YICA-QEMU Docker管理脚本
# 简化的用户界面，无sudo权限要求

set -e

# 配置参数
REMOTE_USER="johnson.chen"
REMOTE_HOST="10.11.60.58"
REMOTE_SSH="${REMOTE_USER}@${REMOTE_HOST}"
WORK_DIR="/home/${REMOTE_USER}/yica-docker-workspace"
CONTAINER_NAME="yica-qemu-container"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "${CYAN}================================${NC}"
    echo -e "${CYAN} YICA-QEMU Docker 管理工具${NC}"
    echo -e "${CYAN}================================${NC}"
}

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

# 检查连接状态
check_connection() {
    print_status "检查远程连接..."
    if ssh -o ConnectTimeout=5 "$REMOTE_SSH" "echo '连接成功'" 2>/dev/null; then
        print_success "远程服务器连接正常"
        return 0
    else
        print_error "无法连接到远程服务器 $REMOTE_SSH"
        return 1
    fi
}

# 检查容器状态
check_container_status() {
    print_status "检查容器状态..."
    
    local status=$(ssh "$REMOTE_SSH" "docker ps -q -f name=$CONTAINER_NAME" 2>/dev/null)
    
    if [ -n "$status" ]; then
        print_success "容器正在运行 (ID: $status)"
        
        # 显示端口信息
        echo -e "${BLUE}端口映射:${NC}"
        ssh "$REMOTE_SSH" "docker port $CONTAINER_NAME 2>/dev/null" | sed 's/^/  /'
        
        return 0
    else
        print_warning "容器未运行"
        return 1
    fi
}

# 显示服务状态
show_status() {
    print_header
    
    if ! check_connection; then
        return 1
    fi
    
    if check_container_status; then
        echo ""
        echo -e "${GREEN}🌐 访问地址:${NC}"
        echo -e "  VNC客户端: ${CYAN}vnc://$REMOTE_HOST:5900${NC} (密码: yica)"
        echo -e "  Web VNC:   ${CYAN}http://$REMOTE_HOST:6080${NC} (密码: yica)"
        echo -e "  QEMU监控:  ${CYAN}telnet $REMOTE_HOST 4444${NC}"
        echo ""
        echo -e "${BLUE}🔧 管理命令:${NC}"
        echo -e "  进入容器: ${YELLOW}$0 shell${NC}"
        echo -e "  查看日志: ${YELLOW}$0 logs${NC}"
        echo -e "  重启服务: ${YELLOW}$0 restart${NC}"
    else
        echo ""
        echo -e "${YELLOW}💡 启动服务: ${CYAN}$0 start${NC}"
    fi
}

# 启动服务
start_service() {
    print_status "启动YICA-QEMU Docker服务..."
    
    if ! check_connection; then
        return 1
    fi
    
    ssh "$REMOTE_SSH" << EOF
        cd $WORK_DIR
        
        echo "🐳 启动Docker容器..."
        
        # 停止现有容器
        docker stop $CONTAINER_NAME 2>/dev/null || true
        docker rm $CONTAINER_NAME 2>/dev/null || true
        
        # 启动容器
        if command -v docker-compose >/dev/null 2>&1 && [ -f docker-compose.yml ]; then
            docker-compose up -d
        else
            echo "使用Docker命令启动..."
            docker run -d \\
                --name $CONTAINER_NAME \\
                -p 5900:5900 \\
                -p 6080:6080 \\
                -p 4444:4444 \\
                -p 3456:3456 \\
                -p 2222:2222 \\
                -v \$(pwd)/yirage:/home/yica/workspace/yirage \\
                -v \$(pwd)/image2:/home/yica/workspace/image2 \\
                -v \$(pwd)/logs:/home/yica/workspace/logs \\
                --device /dev/kvm:/dev/kvm \\
                -e YICA_HOME=/home/yica/workspace \\
                -e YICA_BACKEND_MODE=cpu \\
                yica-qemu:latest
        fi
        
        echo "⏳ 等待服务启动..."
        sleep 10
EOF
    
    if check_container_status; then
        print_success "YICA-QEMU服务启动成功！"
        echo ""
        echo -e "${GREEN}🌐 现在可以访问:${NC}"
        echo -e "  Web VNC: ${CYAN}http://$REMOTE_HOST:6080${NC}"
        echo -e "  VNC客户端: ${CYAN}vnc://$REMOTE_HOST:5900${NC}"
    else
        print_error "服务启动失败，请检查日志"
    fi
}

# 停止服务
stop_service() {
    print_status "停止YICA-QEMU Docker服务..."
    
    if ! check_connection; then
        return 1
    fi
    
    ssh "$REMOTE_SSH" << EOF
        cd $WORK_DIR
        
        if command -v docker-compose >/dev/null 2>&1 && [ -f docker-compose.yml ]; then
            docker-compose down
        else
            docker stop $CONTAINER_NAME 2>/dev/null || true
            docker rm $CONTAINER_NAME 2>/dev/null || true
        fi
EOF
    
    print_success "服务已停止"
}

# 重启服务
restart_service() {
    print_status "重启YICA-QEMU Docker服务..."
    stop_service
    sleep 2
    start_service
}

# 查看日志
show_logs() {
    print_status "显示容器日志..."
    
    if ! check_connection; then
        return 1
    fi
    
    ssh "$REMOTE_SSH" "docker logs -f --tail=50 $CONTAINER_NAME"
}

# 进入容器Shell
enter_shell() {
    print_status "进入容器Shell..."
    
    if ! check_connection; then
        return 1
    fi
    
    if ! check_container_status >/dev/null 2>&1; then
        print_error "容器未运行，请先启动服务"
        return 1
    fi
    
    echo -e "${CYAN}进入容器，工作目录: /home/yica/workspace${NC}"
    echo -e "${YELLOW}退出容器: 输入 exit${NC}"
    echo ""
    
    ssh -t "$REMOTE_SSH" "docker exec -it $CONTAINER_NAME bash"
}

# 启动QEMU
start_qemu() {
    print_status "启动QEMU虚拟机..."
    
    if ! check_connection; then
        return 1
    fi
    
    if ! check_container_status >/dev/null 2>&1; then
        print_error "容器未运行，请先启动服务"
        return 1
    fi
    
    ssh "$REMOTE_SSH" "docker exec -d $CONTAINER_NAME /home/yica/workspace/qemu-docker.sh"
    
    print_success "QEMU启动命令已发送"
    print_status "可以通过VNC查看虚拟机: http://$REMOTE_HOST:6080"
}

# 启动gem5
start_gem5() {
    print_status "启动gem5模拟器..."
    
    if ! check_connection; then
        return 1
    fi
    
    if ! check_container_status >/dev/null 2>&1; then
        print_error "容器未运行，请先启动服务"
        return 1
    fi
    
    ssh "$REMOTE_SSH" "docker exec -d $CONTAINER_NAME /home/yica/workspace/gem5-docker.sh"
    
    print_success "gem5启动命令已发送"
}

# 快速部署
quick_deploy() {
    print_header
    print_status "执行快速部署..."
    
    if ! check_connection; then
        return 1
    fi
    
    # 运行完整部署脚本
    local deploy_script="$(dirname "$0")/docker_yica_deployment.sh"
    
    if [ -f "$deploy_script" ]; then
        print_status "运行部署脚本..."
        "$deploy_script"
    else
        print_error "部署脚本不存在: $deploy_script"
        return 1
    fi
}

# 显示帮助信息
show_help() {
    print_header
    echo ""
    echo -e "${BLUE}用法:${NC} $0 [命令]"
    echo ""
    echo -e "${BLUE}🔧 服务管理:${NC}"
    echo -e "  ${CYAN}status${NC}    - 显示服务状态"
    echo -e "  ${CYAN}start${NC}     - 启动Docker服务"
    echo -e "  ${CYAN}stop${NC}      - 停止Docker服务"
    echo -e "  ${CYAN}restart${NC}   - 重启Docker服务"
    echo -e "  ${CYAN}logs${NC}      - 查看服务日志"
    echo ""
    echo -e "${BLUE}🖥️  操作命令:${NC}"
    echo -e "  ${CYAN}shell${NC}     - 进入容器Shell"
    echo -e "  ${CYAN}qemu${NC}      - 启动QEMU虚拟机"
    echo -e "  ${CYAN}gem5${NC}      - 启动gem5模拟器"
    echo ""
    echo -e "${BLUE}🚀 部署命令:${NC}"
    echo -e "  ${CYAN}deploy${NC}    - 执行完整部署"
    echo ""
    echo -e "${BLUE}📖 访问地址:${NC}"
    echo -e "  Web VNC:   ${YELLOW}http://$REMOTE_HOST:6080${NC}"
    echo -e "  VNC客户端: ${YELLOW}vnc://$REMOTE_HOST:5900${NC}"
    echo -e "  QEMU监控:  ${YELLOW}telnet $REMOTE_HOST 4444${NC}"
    echo ""
    echo -e "${BLUE}💡 常用操作流程:${NC}"
    echo -e "  1. ${CYAN}$0 deploy${NC}   # 首次部署"
    echo -e "  2. ${CYAN}$0 start${NC}    # 启动服务"
    echo -e "  3. ${CYAN}$0 shell${NC}    # 进入容器"
    echo -e "  4. ${CYAN}$0 qemu${NC}     # 启动虚拟机"
    echo ""
}

# 主函数
main() {
    case "${1:-status}" in
        "status"|"stat"|"s")
            show_status
            ;;
        "start"|"up")
            start_service
            ;;
        "stop"|"down")
            stop_service
            ;;
        "restart"|"reboot")
            restart_service
            ;;
        "logs"|"log"|"l")
            show_logs
            ;;
        "shell"|"bash"|"sh")
            enter_shell
            ;;
        "qemu"|"vm")
            start_qemu
            ;;
        "gem5"|"g5")
            start_gem5
            ;;
        "deploy"|"install")
            quick_deploy
            ;;
        "help"|"h"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "未知命令: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@" 
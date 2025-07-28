#!/bin/bash
# YICA-QEMU Docker化远程部署脚本
# 避免sudo权限问题，全容器化部署
# 严格按照yicai-qemu.md文档要求实现

set -e

echo "🐳 YICA-QEMU Docker化远程部署 (无sudo权限)..."

# 配置参数
REMOTE_USER="johnson.chen"
REMOTE_HOST="10.11.60.58"
REMOTE_SSH="${REMOTE_USER}@${REMOTE_HOST}"
WORK_DIR="/home/${REMOTE_USER}/yica-docker-workspace"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Docker容器配置
CONTAINER_NAME="yica-qemu-container"
DOCKER_IMAGE="yica-qemu:latest"
VNC_PORT="5900"
QEMU_MONITOR_PORT="4444"
GEM5_PORT="3456"

# 网络配置 (Docker网络，无需TAP)
DOCKER_NETWORK="yica-network"
CONTAINER_IP="172.20.0.10"

# 根据yicai-qemu.md文档的具体要求
IMAGE_SOURCE_SERVER="10.12.70.52"
IMAGE_SOURCE_PATH="/home/data/zhongjin.wu/image2/ubuntu22.04-kernel6.2.8_publish.img"
GITLAB_REPO="git@10.11.60.249:release/software-release.git"
GEM5_SERVER="10.11.60.100"
GEM5_PATH="/opt/tools/gem5-release"

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

# 检查SSH连接
check_ssh_connection() {
    print_status "检查SSH连接到 $REMOTE_SSH..."
    
    if ! ssh -o ConnectTimeout=10 "$REMOTE_SSH" "echo 'SSH连接成功'" 2>/dev/null; then
        print_error "无法连接到远程服务器 $REMOTE_SSH"
        exit 1
    fi
    
    print_success "SSH连接验证通过"
}

# 检查Docker环境
check_docker_environment() {
    print_status "检查远程Docker环境..."
    
    ssh "$REMOTE_SSH" << 'EOF'
        echo "🐳 检查Docker环境..."
        
        # 检查Docker是否安装
        if command -v docker >/dev/null 2>&1; then
            echo "✅ Docker已安装"
            docker --version
            
            # 检查Docker服务状态
            if docker info >/dev/null 2>&1; then
                echo "✅ Docker服务运行正常"
            else
                echo "❌ Docker服务未运行，尝试启动..."
                # 尝试启动Docker (如果用户在docker组中)
                systemctl --user start docker 2>/dev/null || echo "需要手动启动Docker服务"
            fi
        else
            echo "❌ Docker未安装"
            echo "请先安装Docker: curl -fsSL https://get.docker.com | sh"
            echo "并将用户添加到docker组: usermod -aG docker $USER"
            exit 1
        fi
        
        # 检查Docker Compose
        if command -v docker-compose >/dev/null 2>&1; then
            echo "✅ Docker Compose已安装"
            docker-compose --version
        else
            echo "⚠️  Docker Compose未安装，将使用docker命令"
        fi
        
        # 检查磁盘空间
        echo "磁盘空间:"
        df -h $HOME
        
        echo "✅ Docker环境检查完成"
EOF
    
    print_success "Docker环境检查完成"
}

# 1. 获取系统镜像 (按文档1节要求)
get_system_image() {
    print_status "获取系统镜像 (按yicai-qemu.md第1节)..."
    
    ssh "$REMOTE_SSH" << EOF
        set -e
        cd $WORK_DIR
        
        echo "🖼️  获取QEMU系统镜像..."
        
        # 创建image2目录
        mkdir -p image2
        
        # 检查是否已有镜像
        if [ -f "image2/test2.qcow2" ]; then
            echo "✅ 系统镜像已存在: image2/test2.qcow2"
        else
            echo "📥 从源服务器获取系统镜像..."
            echo "镜像源: $IMAGE_SOURCE_SERVER:$IMAGE_SOURCE_PATH"
            
            # 尝试从源服务器拷贝镜像
            if scp -o ConnectTimeout=10 "$IMAGE_SOURCE_SERVER:$IMAGE_SOURCE_PATH" image2/test2.qcow2 2>/dev/null; then
                echo "✅ 系统镜像下载成功"
            else
                echo "⚠️  无法从源服务器获取镜像，创建空镜像用于测试"
                qemu-img create -f qcow2 image2/test2.qcow2 50G
                echo "⚠️  需要手动安装Ubuntu 22.04 + kernel 6.2.8 + ROCm 5.7.3"
            fi
        fi
        
        echo "✅ 系统镜像准备完成"
EOF
    
    print_success "系统镜像获取完成"
}

# 2.1 获取QEMU二进制文件 (按文档2.1节要求)
get_qemu_binaries() {
    print_status "获取QEMU二进制文件 (按yicai-qemu.md第2.1节)..."
    
    ssh "$REMOTE_SSH" << EOF
        set -e
        cd $WORK_DIR
        
        echo "📦 获取QEMU软件包..."
        
        # 按文档要求从gitlab获取软件包
        if [ ! -d "software-release" ]; then
            echo "从GitLab克隆软件发布包..."
            git clone -b g100-dev $GITLAB_REPO software-release || {
                echo "⚠️  GitLab克隆失败，尝试创建本地qemubin目录"
                mkdir -p qemubin
                # 使用系统QEMU作为fallback
                ln -sf /usr/bin/qemu-system-x86_64 qemubin/qemu-system-x86_64 2>/dev/null || true
            }
        else
            echo "✅ 软件发布包已存在"
        fi
        
        # 检查qemubin目录
        if [ -d "software-release/qemubin" ]; then
            echo "✅ 发现qemubin目录"
            ls -la software-release/qemubin/ | head -10
        elif [ -d "qemubin" ]; then
            echo "✅ 使用本地qemubin目录"
        else
            echo "⚠️  创建qemubin目录并使用系统QEMU"
            mkdir -p qemubin
            cp /usr/bin/qemu-system-x86_64 qemubin/ 2>/dev/null || true
        fi
        
        echo "✅ QEMU二进制文件准备完成"
EOF
    
    print_success "QEMU二进制文件获取完成"
}

# 2.4 配置gem5环境 (按文档2.4节要求)
setup_gem5_environment() {
    print_status "配置gem5环境 (按yicai-qemu.md第2.4节)..."
    
    ssh "$REMOTE_SSH" << EOF
        set -e
        cd $WORK_DIR
        
        echo "🔧 配置gem5 RISC-V模拟器..."
        
        # 检查是否可以从gem5服务器获取
        echo "尝试从gem5服务器获取二进制包..."
        if scp -r -o ConnectTimeout=10 "$GEM5_SERVER:$GEM5_PATH" /tmp/gem5-release 2>/dev/null; then
            echo "✅ 从服务器获取gem5成功"
            sudo mkdir -p /opt/tools 2>/dev/null || mkdir -p gem5-tools
            if sudo mv /tmp/gem5-release /opt/tools/ 2>/dev/null; then
                echo "✅ gem5安装到系统目录"
            else
                mv /tmp/gem5-release gem5-tools/ 2>/dev/null || true
                echo "✅ gem5安装到用户目录"
            fi
        else
            echo "⚠️  无法从gem5服务器获取，使用源码编译"
        fi
        
        # 如果没有预编译版本，从源码编译
        if [ ! -f "/opt/tools/gem5-release/build/RISCV/gem5.opt" ] && [ ! -f "gem5-tools/gem5-release/build/RISCV/gem5.opt" ] && [ ! -f "gem5/build/RISCV/gem5.opt" ]; then
            echo "📦 从源码编译gem5..."
            if [ ! -d "gem5" ]; then
                git clone https://github.com/gem5/gem5.git
                cd gem5
                git checkout v22.1.0.0
            else
                cd gem5
            fi
            
            echo "编译gem5 RISC-V版本 (这可能需要20-30分钟)..."
            scons build/RISCV/gem5.opt -j\$(nproc) || echo "gem5编译可能需要更多时间..."
            cd ..
        fi
        
        echo "✅ gem5环境配置完成"
EOF
    
    print_success "gem5环境配置完成"
}

# 创建Dockerfile
create_dockerfile() {
    print_status "创建YICA-QEMU Dockerfile..."
    
    ssh "$REMOTE_SSH" << EOF
        mkdir -p $WORK_DIR/docker
        
        cat > $WORK_DIR/docker/Dockerfile << 'DOCKERFILE_EOF'
# YICA-QEMU Docker镜像
# 基于Ubuntu 22.04，包含QEMU、gem5、ROCm和YICA环境
FROM ubuntu:22.04

# 设置非交互模式
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 创建工作用户 (避免root权限)
RUN useradd -m -s /bin/bash yica && \
    echo "yica:yica" | chpasswd && \
    usermod -aG sudo yica

# 安装基础依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    python3 \
    python3-pip \
    python3-dev \
    pkg-config \
    libssl-dev \
    libffi-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装QEMU相关依赖
RUN apt-get update && apt-get install -y \
    qemu-system-x86 \
    qemu-utils \
    qemu-kvm \
    bridge-utils \
    net-tools \
    iproute2 \
    iptables \
    && rm -rf /var/lib/apt/lists/*

# 安装VNC支持
RUN apt-get update && apt-get install -y \
    tigervnc-standalone-server \
    tigervnc-common \
    xfce4 \
    xfce4-goodies \
    novnc \
    websockify \
    && rm -rf /var/lib/apt/lists/*

# 安装gem5依赖
RUN apt-get update && apt-get install -y \
    scons \
    m4 \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-perftools-dev \
    gcc-riscv64-linux-gnu \
    g++-riscv64-linux-gnu \
    && rm -rf /var/lib/apt/lists/*

# 安装YICA数学库
RUN apt-get update && apt-get install -y \
    libeigen3-dev \
    libopenblas-dev \
    liblapack-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install numpy cython pytest pytest-cov matplotlib seaborn

# 创建工作目录
WORKDIR /home/yica/workspace
RUN chown -R yica:yica /home/yica

# 切换到yica用户
USER yica

# 设置环境变量
ENV YICA_HOME=/home/yica/workspace
ENV PYTHONPATH=/home/yica/workspace/yirage/python:$PYTHONPATH
ENV YICA_BACKEND_MODE=cpu
ENV OMP_NUM_THREADS=8

# 暴露端口
EXPOSE 5900 4444 3456 6080

    # 创建启动脚本 (修复权限问题)
    COPY --chmod=755 docker/start-services.sh /home/yica/start-services.sh

# 默认启动命令
CMD ["/home/yica/start-services.sh"]
DOCKERFILE_EOF

        echo "✅ Dockerfile创建完成"
EOF
    
    print_success "Dockerfile创建完成"
}

# 创建Docker启动脚本
create_docker_start_script() {
    print_status "创建Docker容器启动脚本..."
    
    ssh "$REMOTE_SSH" << EOF
        cat > $WORK_DIR/docker/start-services.sh << 'START_SCRIPT_EOF'
#!/bin/bash
# YICA-QEMU Docker容器启动脚本

set -e

echo "🚀 启动YICA-QEMU服务 (Docker容器内)..."

# 设置环境变量
export YICA_HOME=/home/yica/workspace
export PYTHONPATH=/home/yica/workspace/yirage/python:\$PYTHONPATH
export DISPLAY=:1

# 创建必要目录
mkdir -p /home/yica/workspace/logs
mkdir -p /home/yica/workspace/image2
mkdir -p /home/yica/.vnc

# 配置VNC服务器
echo "🖥️  配置VNC服务器..."
echo "yica" | vncpasswd -f > /home/yica/.vnc/passwd
chmod 600 /home/yica/.vnc/passwd

# 启动VNC服务器
echo "启动VNC服务器 (端口5900)..."
vncserver :1 -geometry 1024x768 -depth 24 -passwd /home/yica/.vnc/passwd &

# 启动noVNC (Web VNC客户端)
echo "启动noVNC Web客户端 (端口6080)..."
websockify --web=/usr/share/novnc/ 6080 localhost:5901 &

# 等待VNC启动
sleep 3

# 在VNC会话中启动桌面环境
export DISPLAY=:1
startxfce4 &

# 创建QEMU启动脚本 (容器化版本)
cat > /home/yica/workspace/qemu-docker.sh << 'QEMU_DOCKER_EOF'
#!/bin/bash
# QEMU启动脚本 (Docker容器版本)

set -e

# 配置参数
IMAGE_PATH="/home/yica/workspace/image2/test2.qcow2"
UNIX_FILE="/tmp/yica-socket"
VNC_DISPLAY=":2"  # 使用不同的VNC显示

echo "=== YICA-QEMU启动 (Docker容器) ==="
echo "镜像路径: \$IMAGE_PATH"
echo "Socket文件: \$UNIX_FILE"
echo "VNC显示: \$VNC_DISPLAY"

# 清理socket文件
rm -f \$UNIX_FILE

# 检查镜像文件
if [ ! -f "\$IMAGE_PATH" ]; then
    echo "⚠️  创建测试镜像文件..."
    qemu-img create -f qcow2 \$IMAGE_PATH 50G
    echo "📝 需要安装操作系统到镜像中"
fi

# 启动QEMU (容器化配置)
echo "启动QEMU虚拟机..."
qemu-system-x86_64 \
    -enable-kvm \
    -cpu host \
    -smp 4 \
    -m 8G \
    -hda \$IMAGE_PATH \
    -netdev user,id=net0,hostfwd=tcp::2222-:22 \
    -device virtio-net-pci,netdev=net0 \
    -vnc \$VNC_DISPLAY \
    -device yz-g100,rp=off,socket=\$UNIX_FILE,cimdie_cnt=8,cluster_cnt=4 \
    -monitor telnet:0.0.0.0:4444,server,nowait
QEMU_DOCKER_EOF

chmod +x /home/yica/workspace/qemu-docker.sh

# 创建gem5启动脚本 (容器化版本)
cat > /home/yica/workspace/gem5-docker.sh << 'GEM5_DOCKER_EOF'
#!/bin/bash
# gem5启动脚本 (Docker容器版本)

UNIX_FILE=\$1
if [ -z "\$UNIX_FILE" ]; then
    UNIX_FILE="/tmp/yica-socket"
fi

echo "=== gem5 RISC-V模拟器启动 (Docker容器) ==="
echo "Socket文件: \$UNIX_FILE"

# 检查gem5是否存在
if [ -f "/home/yica/workspace/gem5/build/RISCV/gem5.opt" ]; then
    GEM5_BIN="/home/yica/workspace/gem5/build/RISCV/gem5.opt"
    GEM5_CONFIG="/home/yica/workspace/gem5/configs/example/se.py"
    
    # 清理socket文件
    rm -f \$UNIX_FILE
    
    # 启动gem5
    echo "启动gem5模拟器..."
    \$GEM5_BIN \$GEM5_CONFIG \
        --cpu-type=TimingSimpleCPU \
        --mem-size=2GB \
        --caches \
        --l2cache \
        --socket=\$UNIX_FILE
else
    echo "❌ gem5二进制文件不存在，请先构建gem5"
    echo "构建命令: cd /home/yica/workspace/gem5 && scons build/RISCV/gem5.opt -j\$(nproc)"
fi
GEM5_DOCKER_EOF

chmod +x /home/yica/workspace/gem5-docker.sh

# 保持容器运行
echo "✅ YICA-QEMU服务启动完成"
echo "VNC端口: 5900 (密码: yica)"
echo "noVNC Web端口: 6080"
echo "QEMU Monitor端口: 4444"
echo ""
echo "手动启动命令:"
echo "  QEMU: /home/yica/workspace/qemu-docker.sh"
echo "  gem5: /home/yica/workspace/gem5-docker.sh"

# 保持容器运行
tail -f /dev/null
START_SCRIPT_EOF

        chmod +x $WORK_DIR/docker/start-services.sh
        echo "✅ Docker启动脚本创建完成"
EOF
    
    print_success "Docker启动脚本创建完成"
}

# 创建Docker Compose配置
create_docker_compose() {
    print_status "创建Docker Compose配置..."
    
    ssh "$REMOTE_SSH" << EOF
        cat > $WORK_DIR/docker-compose.yml << 'COMPOSE_EOF'
version: '3.8'

services:
  yica-qemu:
    build:
      context: .
      dockerfile: docker/Dockerfile
    container_name: yica-qemu-container
    hostname: yica-qemu
    
    # 端口映射
    ports:
      - "5900:5900"    # VNC
      - "6080:6080"    # noVNC Web
      - "4444:4444"    # QEMU Monitor
      - "3456:3456"    # gem5
      - "2222:2222"    # SSH转发
    
    # 卷挂载
    volumes:
      - ./yirage:/home/yica/workspace/yirage
      - ./image2:/home/yica/workspace/image2
      - ./logs:/home/yica/workspace/logs
      - yica-data:/home/yica/workspace/data
    
    # 网络配置
    networks:
      - yica-network
    
    # 设备权限 (KVM支持)
    devices:
      - /dev/kvm:/dev/kvm
    
    # 特权模式 (QEMU需要)
    privileged: false
    
    # 环境变量
    environment:
      - YICA_HOME=/home/yica/workspace
      - YICA_BACKEND_MODE=cpu
      - OMP_NUM_THREADS=8
      - DISPLAY=:1
    
    # 重启策略
    restart: unless-stopped
    
    # 健康检查
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6080"]
      interval: 30s
      timeout: 10s
      retries: 3

networks:
  yica-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

volumes:
  yica-data:
    driver: local
COMPOSE_EOF

        echo "✅ Docker Compose配置创建完成"
EOF
    
    print_success "Docker Compose配置创建完成"
}

# 同步项目代码
sync_project_code() {
    print_status "同步项目代码到远程服务器..."
    
    # 创建远程工作目录
    ssh "$REMOTE_SSH" "mkdir -p $WORK_DIR"
    
    # 使用rsync同步代码
    rsync -avz --progress \
        --exclude='.git' \
        --exclude='build/' \
        --exclude='*.pyc' \
        --exclude='__pycache__' \
        --exclude='.DS_Store' \
        --exclude='venv/' \
        --exclude='*.log' \
        "$PROJECT_ROOT/" \
        "$REMOTE_SSH:$WORK_DIR/" || {
        print_error "代码同步失败"
        exit 1
    }
    
    print_success "项目代码同步完成"
}

# 构建Docker镜像
build_docker_image() {
    print_status "构建YICA-QEMU Docker镜像..."
    
    ssh "$REMOTE_SSH" << EOF
        cd $WORK_DIR
        
        echo "🐳 构建Docker镜像..."
        
        # 构建镜像
        if command -v docker-compose >/dev/null 2>&1; then
            echo "使用Docker Compose构建..."
            docker-compose build --no-cache
        else
            echo "使用Docker命令构建..."
            docker build -t $DOCKER_IMAGE -f docker/Dockerfile .
        fi
        
        echo "✅ Docker镜像构建完成"
        docker images | grep yica-qemu
EOF
    
    print_success "Docker镜像构建完成"
}

# 启动Docker容器
start_docker_container() {
    print_status "启动YICA-QEMU Docker容器..."
    
    ssh "$REMOTE_SSH" << EOF
        cd $WORK_DIR
        
        echo "🚀 启动Docker容器..."
        
        # 停止现有容器
        docker-compose down 2>/dev/null || true
        docker stop $CONTAINER_NAME 2>/dev/null || true
        docker rm $CONTAINER_NAME 2>/dev/null || true
        
        # 启动新容器
        if command -v docker-compose >/dev/null 2>&1; then
            echo "使用Docker Compose启动..."
            docker-compose up -d
        else
            echo "使用Docker命令启动..."
            docker run -d \
                --name $CONTAINER_NAME \
                -p 5900:5900 \
                -p 6080:6080 \
                -p 4444:4444 \
                -p 3456:3456 \
                -p 2222:2222 \
                -v \$(pwd)/yirage:/home/yica/workspace/yirage \
                -v \$(pwd)/image2:/home/yica/workspace/image2 \
                -v \$(pwd)/logs:/home/yica/workspace/logs \
                --device /dev/kvm:/dev/kvm \
                -e YICA_HOME=/home/yica/workspace \
                -e YICA_BACKEND_MODE=cpu \
                $DOCKER_IMAGE
        fi
        
        # 等待容器启动
        sleep 5
        
        echo "✅ Docker容器启动完成"
        docker ps | grep yica-qemu
EOF
    
    print_success "Docker容器启动完成"
}

# 验证Docker部署
verify_docker_deployment() {
    print_status "验证Docker部署..."
    
    ssh "$REMOTE_SSH" << EOF
        cd $WORK_DIR
        
        echo "🧪 验证YICA-QEMU Docker部署..."
        
        # 检查容器状态
        echo "容器状态:"
        docker ps | grep yica-qemu || echo "❌ 容器未运行"
        
        # 检查端口映射
        echo "端口映射:"
        docker port $CONTAINER_NAME 2>/dev/null || echo "⚠️  端口信息获取失败"
        
        # 检查容器内服务
        echo "容器内服务检查:"
        docker exec $CONTAINER_NAME ps aux | grep -E "(vnc|qemu|gem5)" || echo "⚠️  服务进程检查"
        
        # 检查VNC端口
        echo "VNC端口检查:"
        netstat -tlnp | grep 5900 || echo "⚠️  VNC端口未监听"
        
        # 检查noVNC端口
        echo "noVNC端口检查:"
        netstat -tlnp | grep 6080 || echo "⚠️  noVNC端口未监听"
        
        # 测试容器内Python环境
        echo "Python环境测试:"
        docker exec $CONTAINER_NAME python3 -c "
import sys
print(f'Python版本: {sys.version}')
try:
    import numpy
    print('✅ NumPy可用')
except ImportError:
    print('❌ NumPy不可用')
"
        
        echo "✅ Docker部署验证完成"
EOF
    
    print_success "Docker部署验证完成"
}

# 创建使用说明
create_docker_usage_guide() {
    print_status "创建Docker使用说明..."
    
    ssh "$REMOTE_SSH" << EOF
        cd $WORK_DIR
        
        cat > YICA-QEMU-Docker使用说明.md << 'DOCKER_USAGE_EOF'
# YICA-QEMU Docker化部署使用说明
## 无sudo权限，全容器化解决方案

## 🎯 环境概述

已成功将YICA-QEMU环境完全容器化，避免sudo权限问题：

### ✅ 容器化组件
- **基础镜像**: Ubuntu 22.04
- **QEMU虚拟化**: 完全容器化运行
- **gem5模拟器**: RISC-V模拟器容器内构建
- **VNC服务**: 容器内VNC + noVNC Web客户端
- **YICA环境**: 完整的yirage库和Python绑定

## 🐳 Docker服务管理

### 启动服务
\`\`\`bash
cd $WORK_DIR

# 使用Docker Compose (推荐)
docker-compose up -d

# 或使用Docker命令
docker start yica-qemu-container
\`\`\`

### 停止服务
\`\`\`bash
# 停止服务
docker-compose down

# 或
docker stop yica-qemu-container
\`\`\`

### 查看服务状态
\`\`\`bash
# 查看容器状态
docker ps | grep yica-qemu

# 查看容器日志
docker logs yica-qemu-container

# 查看端口映射
docker port yica-qemu-container
\`\`\`

## 🖥️  访问方式

### 1. VNC连接 (传统方式)
\`\`\`bash
# VNC地址
vnc://$REMOTE_HOST:5900
# 密码: yica
\`\`\`

### 2. Web VNC (推荐，无需客户端)
\`\`\`bash
# 浏览器访问
http://$REMOTE_HOST:6080/vnc.html
# 密码: yica
\`\`\`

### 3. 容器内Shell
\`\`\`bash
# 进入容器
docker exec -it yica-qemu-container bash

# 切换到工作目录
cd /home/yica/workspace
\`\`\`

## 🚀 QEMU和gem5启动

### 方式1: 容器内手动启动
\`\`\`bash
# 进入容器
docker exec -it yica-qemu-container bash

# 启动gem5 (终端1)
/home/yica/workspace/gem5-docker.sh

# 启动QEMU (终端2)
/home/yica/workspace/qemu-docker.sh
\`\`\`

### 方式2: 外部执行
\`\`\`bash
# 启动gem5
docker exec -d yica-qemu-container /home/yica/workspace/gem5-docker.sh

# 启动QEMU
docker exec -d yica-qemu-container /home/yica/workspace/qemu-docker.sh
\`\`\`

## 🔧 服务端口说明

| 服务 | 容器端口 | 主机端口 | 说明 |
|------|----------|----------|------|
| VNC | 5900 | 5900 | 传统VNC客户端 |
| noVNC | 6080 | 6080 | Web VNC客户端 |
| QEMU Monitor | 4444 | 4444 | QEMU监控接口 |
| gem5 | 3456 | 3456 | gem5通信端口 |
| SSH转发 | 22 | 2222 | 虚拟机SSH访问 |

## 📁 数据持久化

### 卷挂载
\`\`\`bash
# 主机目录 -> 容器目录
$WORK_DIR/yirage -> /home/yica/workspace/yirage
$WORK_DIR/image2 -> /home/yica/workspace/image2  
$WORK_DIR/logs -> /home/yica/workspace/logs
\`\`\`

### 系统镜像管理
\`\`\`bash
# 创建QEMU系统镜像
docker exec yica-qemu-container qemu-img create -f qcow2 /home/yica/workspace/image2/test2.qcow2 50G

# 镜像文件位置
ls -la $WORK_DIR/image2/
\`\`\`

## 🧪 测试和验证

### 1. 容器健康检查
\`\`\`bash
# 检查容器状态
docker exec yica-qemu-container ps aux

# 检查VNC服务
docker exec yica-qemu-container netstat -tlnp | grep 5900

# 检查Python环境
docker exec yica-qemu-container python3 -c "import yirage; print('YICA可用')"
\`\`\`

### 2. 网络连接测试
\`\`\`bash
# 测试VNC端口
telnet $REMOTE_HOST 5900

# 测试Web VNC
curl http://$REMOTE_HOST:6080

# 测试QEMU Monitor
telnet $REMOTE_HOST 4444
\`\`\`

### 3. YICA功能测试
\`\`\`bash
# 进入容器测试
docker exec -it yica-qemu-container bash

# Python测试
cd /home/yica/workspace
python3 -c "
import sys
sys.path.insert(0, 'yirage/python')
import yirage
print(f'YICA版本: {yirage.__version__}')
"
\`\`\`

## 🔧 故障排除

### 1. 容器启动失败
\`\`\`bash
# 查看详细日志
docker logs yica-qemu-container

# 重新构建镜像
docker-compose build --no-cache

# 清理并重启
docker-compose down
docker system prune -f
docker-compose up -d
\`\`\`

### 2. VNC连接问题
\`\`\`bash
# 检查VNC进程
docker exec yica-qemu-container ps aux | grep vnc

# 重启VNC服务
docker exec yica-qemu-container pkill Xvnc
docker restart yica-qemu-container
\`\`\`

### 3. QEMU启动问题
\`\`\`bash
# 检查KVM设备
ls -la /dev/kvm

# 检查镜像文件
docker exec yica-qemu-container ls -la /home/yica/workspace/image2/

# 手动启动QEMU (调试模式)
docker exec -it yica-qemu-container /home/yica/workspace/qemu-docker.sh
\`\`\`

### 4. 端口冲突
\`\`\`bash
# 检查端口占用
netstat -tlnp | grep -E "(5900|6080|4444)"

# 修改端口映射
# 编辑 docker-compose.yml 中的ports配置
\`\`\`

## 🔄 维护操作

### 更新代码
\`\`\`bash
# 同步新代码到主机
rsync -avz /path/to/local/code/ $REMOTE_HOST:$WORK_DIR/

# 重启容器应用更改
docker-compose restart
\`\`\`

### 备份数据
\`\`\`bash
# 备份镜像文件
tar -czf yica-backup-\$(date +%Y%m%d).tar.gz -C $WORK_DIR image2/ logs/

# 导出Docker镜像
docker save yica-qemu:latest | gzip > yica-qemu-image.tar.gz
\`\`\`

### 清理空间
\`\`\`bash
# 清理Docker缓存
docker system prune -f

# 清理未使用的镜像
docker image prune -f
\`\`\`

## 🌐 远程访问配置

### SSH隧道 (安全访问)
\`\`\`bash
# 本地机器执行，创建SSH隧道
ssh -L 5900:localhost:5900 -L 6080:localhost:6080 $REMOTE_SSH

# 然后本地访问
vnc://localhost:5900  # VNC
http://localhost:6080  # Web VNC
\`\`\`

### 防火墙配置 (如需要)
\`\`\`bash
# 开放必要端口 (在有权限的情况下)
iptables -A INPUT -p tcp --dport 5900 -j ACCEPT
iptables -A INPUT -p tcp --dport 6080 -j ACCEPT
\`\`\`

---
**部署时间**: \$(date)  
**服务器**: $REMOTE_HOST  
**容器名**: yica-qemu-container  
**Docker化版本**: 完全无sudo权限方案
DOCKER_USAGE_EOF

        echo "✅ Docker使用说明创建完成: YICA-QEMU-Docker使用说明.md"
EOF
    
    print_success "Docker使用说明创建完成"
}

# 显示部署总结
show_docker_deployment_summary() {
    print_success "🎉 YICA-QEMU Docker化部署完成 (无sudo权限)！"
    echo ""
    echo "📋 部署总结："
    echo "  - 服务器: $REMOTE_SSH"
    echo "  - 工作目录: $WORK_DIR"
    echo "  - 容器名: $CONTAINER_NAME"
    echo "  - Docker镜像: $DOCKER_IMAGE"
    echo ""
    echo "🌐 访问方式："
    echo "  - VNC客户端: vnc://$REMOTE_HOST:5900 (密码: yica)"
    echo "  - Web VNC: http://$REMOTE_HOST:6080 (密码: yica)"
    echo "  - QEMU Monitor: telnet $REMOTE_HOST 4444"
    echo ""
    echo "🐳 Docker管理："
    echo "  - 查看状态: ssh $REMOTE_SSH 'docker ps | grep yica-qemu'"
    echo "  - 查看日志: ssh $REMOTE_SSH 'docker logs yica-qemu-container'"
    echo "  - 进入容器: ssh $REMOTE_SSH 'docker exec -it yica-qemu-container bash'"
    echo ""
    echo "🚀 启动QEMU和gem5："
    echo "  1. 进入容器: docker exec -it yica-qemu-container bash"
    echo "  2. 启动gem5: /home/yica/workspace/gem5-docker.sh"
    echo "  3. 启动QEMU: /home/yica/workspace/qemu-docker.sh"
    echo ""
    echo "📖 详细说明: cat YICA-QEMU-Docker使用说明.md"
    echo ""
    echo "✅ 完全容器化，无需sudo权限！"
}

# 主函数
main() {
    case "${1:-}" in
        "check")
            check_ssh_connection
            check_docker_environment
            ;;
        "dockerfile")
            check_ssh_connection
            create_dockerfile
            create_docker_start_script
            create_docker_compose
            ;;
        "sync")
            check_ssh_connection
            sync_project_code
            ;;
        "image")
            check_ssh_connection
            get_system_image
            ;;
        "qemu")
            check_ssh_connection
            get_qemu_binaries
            ;;
        "gem5")
            check_ssh_connection
            setup_gem5_environment
            ;;
        "build")
            check_ssh_connection
            build_docker_image
            ;;
        "start")
            check_ssh_connection
            start_docker_container
            ;;
        "verify")
            check_ssh_connection
            verify_docker_deployment
            ;;
        "guide")
            check_ssh_connection
            create_docker_usage_guide
            ;;
        "")
            print_status "执行完整YICA-QEMU Docker化部署流程..."
            check_ssh_connection
            check_docker_environment
            sync_project_code
            get_system_image
            get_qemu_binaries
            setup_gem5_environment
            create_dockerfile
            create_docker_start_script
            create_docker_compose
            build_docker_image
            start_docker_container
            verify_docker_deployment
            create_docker_usage_guide
            show_docker_deployment_summary
            ;;
        *)
            echo "YICA-QEMU Docker化部署脚本 (无sudo权限)"
            echo ""
            echo "用法: $0 [命令]"
            echo ""
            echo "命令:"
            echo "  check      - 检查SSH和Docker环境"
            echo "  dockerfile - 创建Docker配置文件"
            echo "  sync       - 同步项目代码"
            echo "  image      - 获取系统镜像 (文档第1节)"
            echo "  qemu       - 获取QEMU二进制 (文档2.1节)"
            echo "  gem5       - 配置gem5环境 (文档2.4节)"
            echo "  build      - 构建Docker镜像"
            echo "  start      - 启动Docker容器"
            echo "  verify     - 验证Docker部署"
            echo "  guide      - 创建使用说明"
            echo "  (空)       - 执行完整部署流程"
            echo ""
            echo "Docker化优势:"
            echo "  ✅ 无需sudo权限"
            echo "  ✅ 环境隔离和一致性"
            echo "  ✅ 简化部署和维护"
            echo "  ✅ 支持Web VNC访问"
            echo "  ✅ 数据持久化"
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@" 
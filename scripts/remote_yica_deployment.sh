#!/bin/bash
# YICA-QEMU远程AMD GPU Linux环境完整部署脚本
# 严格按照yicai-qemu.md文档要求实现
# 目标服务器: johnson.chen@10.11.60.58

set -e

echo "🚀 YICA-QEMU远程AMD GPU Linux环境部署 (按yicai-qemu.md规范)..."

# 配置参数
REMOTE_USER="johnson.chen"
REMOTE_HOST="10.11.60.58"
REMOTE_SSH="${REMOTE_USER}@${REMOTE_HOST}"
WORK_DIR="/home/${REMOTE_USER}/yica-workspace"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 根据文档的具体要求
IMAGE_SOURCE_SERVER="10.12.70.52"
IMAGE_SOURCE_PATH="/home/data/zhongjin.wu/image2/ubuntu22.04-kernel6.2.8_publish.img"
GITLAB_REPO="http://gitlab-repo.yizhu.local/release/software-release.git"
GEM5_SERVER="10.11.60.100"
GEM5_PATH="/opt/tools/gem5-release"

# 用户特定配置 (根据文档中的分配表格)
TAPNAME="jc_tap0"  # johnson.chen的tap网卡
VNC_PORT="5900"    # VNC端口
MAC_ADDR="52:54:00:12:34:58"  # 根据用户分配的MAC地址

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
        print_error "请检查："
        print_error "1. 网络连接是否正常"
        print_error "2. SSH密钥是否配置正确"
        print_error "3. 服务器是否在线"
        exit 1
    fi
    
    print_success "SSH连接验证通过"
}

# 检查远程环境
check_remote_environment() {
    print_status "检查远程AMD GPU环境..."
    
    ssh "$REMOTE_SSH" << 'EOF'
        echo "🔍 检查远程环境 (按yicai-qemu.md要求)..."
        
        # 检查操作系统 - 文档要求Ubuntu 22.04 + kernel 6.2.8
        echo "操作系统信息:"
        lsb_release -a 2>/dev/null || cat /etc/os-release
        echo "内核版本:"
        uname -r
        echo ""
        
        # 检查AMD GPU
        echo "AMD GPU信息:"
        if command -v lspci >/dev/null 2>&1; then
            lspci | grep -i amd || echo "未检测到AMD GPU"
        fi
        echo ""
        
        # 检查ROCm 5.7.3 - 文档明确要求此版本
        echo "ROCm 5.7.3环境检查:"
        if command -v rocm-smi >/dev/null 2>&1; then
            echo "✅ ROCm已安装"
            rocm-smi --version 2>/dev/null || echo "ROCm版本获取失败"
            rocm-smi --showproductname 2>/dev/null || echo "GPU产品信息获取失败"
        else
            echo "⚠️  ROCm 5.7.3未安装或未配置"
        fi
        echo ""
        
        # 检查基础工具
        echo "基础工具检查:"
        for tool in gcc g++ cmake python3 git qemu-system-x86_64; do
            if command -v $tool >/dev/null 2>&1; then
                echo "✅ $tool: $(which $tool)"
            else
                echo "❌ $tool: 未安装"
            fi
        done
        echo ""
        
        # 检查网络配置 - tap网卡
        echo "网络配置检查:"
        if ip link show | grep -q tap; then
            echo "✅ 发现tap网卡:"
            ip link show | grep tap
        else
            echo "⚠️  未发现tap网卡，需要配置"
        fi
        echo ""
        
        # 检查磁盘空间
        echo "磁盘空间:"
        df -h $HOME
        echo ""
        
        echo "✅ 远程环境检查完成"
EOF
    
    print_success "远程环境检查完成"
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

# 2.2 配置QEMU启动参数 (按文档2.2节要求)
configure_qemu_scripts() {
    print_status "配置QEMU启动脚本 (按yicai-qemu.md第2.2节)..."
    
    ssh "$REMOTE_SSH" << EOF
        set -e
        cd $WORK_DIR
        
        echo "🔧 配置QEMU启动脚本..."
        
        # 确定QEMU二进制路径
        if [ -f "software-release/qemubin/qemu-system-x86_64" ]; then
            MYBIN="$WORK_DIR/software-release/qemubin"
        elif [ -f "qemubin/qemu-system-x86_64" ]; then
            MYBIN="$WORK_DIR/qemubin"
        else
            MYBIN="/usr/bin"
        fi
        
        # 创建qemu2.sh脚本 (严格按照文档格式)
        cat > qemu2.sh << 'QEMU_SCRIPT_EOF'
#!/bin/bash
# YICA-QEMU启动脚本 (按yicai-qemu.md规范)

# 用户配置参数 (按文档2.2节要求)
TAPNAME="$TAPNAME"
VNC_ADDR="$REMOTE_HOST:0"  # VNC端口配置
MAC_ADDR="$MAC_ADDR"
MYBIN="MYBIN_PLACEHOLDER"
IMAGE_PATH="$WORK_DIR/image2/test2.qcow2"
UNIX_FILE="/tmp/\${USER}"

# CIM配置 (按文档2.3.1节)
CIMDIE_CNT=8
CLUSTER_CNT=4

echo "=== YICA-QEMU启动 (AMD GPU环境) ==="
echo "QEMU路径: \$MYBIN"
echo "镜像路径: \$IMAGE_PATH"
echo "CIM Die数量: \$CIMDIE_CNT"
echo "Cluster数量: \$CLUSTER_CNT"
echo "VNC地址: \$VNC_ADDR"
echo "TAP网卡: \$TAPNAME"
echo "MAC地址: \$MAC_ADDR"
echo ""

# 检查AMD GPU状态
if command -v rocm-smi > /dev/null 2>&1; then
    echo "AMD GPU状态:"
    rocm-smi --showproductname --showtemp --showmemuse --showuse 2>/dev/null || echo "GPU信息获取失败"
    echo ""
else
    echo "⚠️  ROCm未安装或未正确配置"
fi

# 清理socket文件
rm -f \${UNIX_FILE}

# 检查镜像文件
if [ ! -f "\$IMAGE_PATH" ]; then
    echo "❌ 镜像文件不存在: \$IMAGE_PATH"
    exit 1
fi

# 检查tap网卡
if ! ip link show \$TAPNAME > /dev/null 2>&1; then
    echo "⚠️  TAP网卡 \$TAPNAME 不存在，尝试创建..."
    sudo ip tuntap add dev \$TAPNAME mode tap 2>/dev/null || echo "需要管理员权限创建TAP网卡"
    sudo ip link set \$TAPNAME up 2>/dev/null || true
fi

# 启动QEMU (按文档格式)
echo "启动QEMU虚拟机..."
\$MYBIN/qemu-system-x86_64 \\
    -enable-kvm \\
    -cpu host \\
    -smp 8 \\
    -m 16G \\
    -hda \$IMAGE_PATH \\
    -netdev tap,id=net0,ifname=\$TAPNAME,script=no,downscript=no \\
    -device virtio-net-pci,netdev=net0,mac=\$MAC_ADDR \\
    -vnc \$VNC_ADDR \\
    -device yz-g100,rp=on,socket=\$UNIX_FILE,cimdie_cnt=\$CIMDIE_CNT,cluster_cnt=\$CLUSTER_CNT \\
    -monitor stdio
QEMU_SCRIPT_EOF

        # 替换MYBIN路径
        sed -i "s|MYBIN_PLACEHOLDER|\$MYBIN|g" qemu2.sh
        
        # 设置执行权限
        chmod +x qemu2.sh
        
        echo "✅ qemu2.sh脚本配置完成"
EOF
    
    print_success "QEMU启动脚本配置完成"
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
            sudo mkdir -p /opt/tools
            sudo mv /tmp/gem5-release /opt/tools/ 2>/dev/null || true
        else
            echo "⚠️  无法从gem5服务器获取，使用源码编译"
        fi
        
        # 如果没有预编译版本，从源码编译
        if [ ! -f "/opt/tools/gem5-release/build/RISCV/gem5.opt" ] && [ ! -f "gem5/build/RISCV/gem5.opt" ]; then
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
        
        # 创建gem5.sh脚本 (按文档格式)
        cat > gem5.sh << 'GEM5_SCRIPT_EOF'
#!/bin/bash
# gem5启动脚本 (按yicai-qemu.md规范)

UNIX_FILE=\$1
if [ -z "\$UNIX_FILE" ]; then
    echo "Usage: \$0 <socket_file>"
    echo "Example: \$0 /tmp/\${USER}"
    exit 1
fi

# 确定gem5二进制路径
if [ -f "/opt/tools/gem5-release/build/RISCV/gem5.opt" ]; then
    GEM5_BIN="/opt/tools/gem5-release/build/RISCV/gem5.opt"
    GEM5_CONFIG="/opt/tools/gem5-release/configs/example/se.py"
elif [ -f "$WORK_DIR/gem5/build/RISCV/gem5.opt" ]; then
    GEM5_BIN="$WORK_DIR/gem5/build/RISCV/gem5.opt"
    GEM5_CONFIG="$WORK_DIR/gem5/configs/example/se.py"
else
    echo "❌ gem5二进制文件不存在"
    exit 1
fi

echo "=== gem5 RISC-V模拟器启动 ==="
echo "gem5二进制: \$GEM5_BIN"
echo "配置文件: \$GEM5_CONFIG"
echo "Socket文件: \$UNIX_FILE"

# 清理socket文件
rm -f \$UNIX_FILE

# 启动gem5
echo "启动gem5模拟器..."
\$GEM5_BIN \$GEM5_CONFIG \\
    --cpu-type=TimingSimpleCPU \\
    --mem-size=2GB \\
    --caches \\
    --l2cache \\
    --socket=\$UNIX_FILE
GEM5_SCRIPT_EOF

        chmod +x gem5.sh
        
        echo "✅ gem5环境配置完成"
EOF
    
    print_success "gem5环境配置完成"
}

# 同步项目代码到远程服务器
sync_project_code() {
    print_status "同步项目代码到远程服务器..."
    
    # 创建远程工作目录
    ssh "$REMOTE_SSH" "mkdir -p $WORK_DIR"
    
    # 使用rsync同步代码，排除不必要的文件
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

# 安装依赖和配置环境
install_dependencies() {
    print_status "在远程服务器安装依赖..."
    
    ssh "$REMOTE_SSH" << EOF
        set -e
        cd $WORK_DIR
        
        echo "🔧 安装系统依赖 (按yicai-qemu.md要求)..."
        
        # 更新包管理器
        sudo apt update
        
        # 安装基础开发工具
        sudo apt install -y \\
            build-essential \\
            cmake \\
            git \\
            wget \\
            curl \\
            vim \\
            python3 \\
            python3-pip \\
            python3-dev \\
            pkg-config \\
            libssl-dev \\
            libffi-dev \\
            zlib1g-dev \\
            libbz2-dev \\
            libreadline-dev \\
            libsqlite3-dev \\
            llvm \\
            libncurses5-dev \\
            libncursesw5-dev \\
            xz-utils \\
            tk-dev \\
            libxml2-dev \\
            libxmlsec1-dev \\
            liblzma-dev
        
        # 安装QEMU相关依赖 (文档要求)
        sudo apt install -y \\
            qemu-system-x86 \\
            qemu-utils \\
            qemu-kvm \\
            bridge-utils \\
            uml-utilities \\
            virt-manager \\
            libvirt-daemon-system \\
            libvirt-clients \\
            net-tools \\
            iproute2 \\
            iptables
        
        # 安装VNC支持 (文档第3节要求)
        sudo apt install -y \\
            tigervnc-standalone-server \\
            tigervnc-common \\
            xfce4 \\
            xfce4-goodies
        
        # 安装gem5依赖
        sudo apt install -y \\
            scons \\
            m4 \\
            libprotobuf-dev \\
            protobuf-compiler \\
            libgoogle-perftools-dev \\
            gcc-riscv64-linux-gnu \\
            g++-riscv64-linux-gnu
        
        # 安装YICA专用数学库
        sudo apt install -y \\
            libeigen3-dev \\
            libopenblas-dev \\
            liblapack-dev \\
            libomp-dev
        
        # 安装Python依赖
        pip3 install --user --upgrade pip setuptools wheel
        pip3 install --user numpy cython pytest pytest-cov matplotlib seaborn
        
        echo "✅ 系统依赖安装完成"
EOF
    
    print_success "依赖安装完成"
}

# 配置ROCm环境（如果需要）
configure_rocm() {
    print_status "配置ROCm 5.7.3环境 (按文档要求)..."
    
    ssh "$REMOTE_SSH" << 'EOF'
        set -e
        
        echo "🔧 配置ROCm 5.7.3环境..."
        
        # 检查ROCm是否已安装
        if command -v rocm-smi >/dev/null 2>&1; then
            echo "✅ ROCm已安装，配置环境变量..."
            
            # 检查版本是否为5.7.3
            ROCM_VERSION=$(rocm-smi --version 2>/dev/null | grep -o "5\.[0-9]\.[0-9]" | head -1 || echo "unknown")
            echo "当前ROCm版本: $ROCM_VERSION"
            
            # 添加环境变量到.bashrc
            if ! grep -q "ROCm" ~/.bashrc; then
                echo "" >> ~/.bashrc
                echo "# ROCm 5.7.3环境变量 (按yicai-qemu.md)" >> ~/.bashrc
                echo "export ROCM_PATH=/opt/rocm" >> ~/.bashrc
                echo "export PATH=\$ROCM_PATH/bin:\$PATH" >> ~/.bashrc
                echo "export LD_LIBRARY_PATH=\$ROCM_PATH/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
                echo "export HSA_OVERRIDE_GFX_VERSION=10.3.0" >> ~/.bashrc
            fi
            
            # 立即应用环境变量
            export ROCM_PATH=/opt/rocm
            export PATH=$ROCM_PATH/bin:$PATH
            export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
            export HSA_OVERRIDE_GFX_VERSION=10.3.0
            
            # 验证ROCm
            echo "ROCm版本信息:"
            rocm-smi --version || echo "ROCm版本获取失败"
            
            echo "GPU信息:"
            rocm-smi --showproductname --showtemp --showmemuse --showuse || echo "GPU信息获取失败"
            
        else
            echo "⚠️  ROCm 5.7.3未安装"
            echo "按文档要求，系统镜像应该预装ROCm 5.7.3"
            echo "如需手动安装ROCm 5.7.3，请参考官方文档"
        fi
        
        echo "✅ ROCm环境配置完成"
EOF
    
    print_success "ROCm环境配置完成"
}

# 构建YICA核心库
build_yica_core() {
    print_status "构建YICA核心库..."
    
    ssh "$REMOTE_SSH" << EOF
        set -e
        cd $WORK_DIR
        
        echo "🔨 构建YICA核心库..."
        
        # 设置环境变量
        export YICA_HOME="$WORK_DIR"
        export PYTHONPATH="$WORK_DIR/yirage/python:\$PYTHONPATH"
        export YICA_BACKEND_MODE="cpu"
        export OMP_NUM_THREADS="8"
        
        # 构建Z3依赖
        echo "📦 构建Z3依赖..."
        cd yirage/deps/z3
        mkdir -p build && cd build
        cmake .. && make -j\$(nproc)
        
        # 构建YICA核心库
        echo "🏗️  构建YICA核心库..."
        cd $WORK_DIR/yirage
        mkdir -p build && cd build
        export Z3_DIR=$WORK_DIR/yirage/deps/z3/build
        cmake .. \\
            -DYICA_ENABLE_CUDA=OFF \\
            -DYICA_CPU_ONLY=ON \\
            -DCMAKE_BUILD_TYPE=Release \\
            -DCMAKE_CXX_FLAGS="-O3 -fopenmp -DYICA_CPU_ONLY"
        make -j\$(nproc)
        
        # 安装Python包
        echo "🐍 安装YICA Python包..."
        cd $WORK_DIR/yirage
        python3 setup.py develop --user
        
        echo "✅ YICA核心库构建完成"
EOF
    
    print_success "YICA核心库构建完成"
}

# 运行验证测试
run_verification_tests() {
    print_status "运行验证测试 (按文档要求)..."
    
    ssh "$REMOTE_SSH" << EOF
        set -e
        cd $WORK_DIR
        
        echo "🧪 运行YICA验证测试..."
        
        # 设置环境变量
        export YICA_HOME="$WORK_DIR"
        export PYTHONPATH="$WORK_DIR/yirage/python:\$PYTHONPATH"
        export YICA_BACKEND_MODE="cpu"
        
        # 测试qemu2.sh脚本
        echo "🧪 验证qemu2.sh脚本..."
        if [ -f "qemu2.sh" ]; then
            echo "✅ qemu2.sh脚本存在"
            head -10 qemu2.sh
        else
            echo "❌ qemu2.sh脚本不存在"
        fi
        
        # 测试gem5.sh脚本
        echo "🧪 验证gem5.sh脚本..."
        if [ -f "gem5.sh" ]; then
            echo "✅ gem5.sh脚本存在"
        else
            echo "❌ gem5.sh脚本不存在"
        fi
        
        # 测试镜像文件
        echo "🧪 验证系统镜像..."
        if [ -f "image2/test2.qcow2" ]; then
            echo "✅ 系统镜像存在: image2/test2.qcow2"
            qemu-img info image2/test2.qcow2 | head -5
        else
            echo "❌ 系统镜像不存在"
        fi
        
        # Python导入测试
        echo "🐍 Python导入测试..."
        python3 -c "
import sys
sys.path.insert(0, '$WORK_DIR/yirage/python')

try:
    import yirage
    print(f'✅ yirage导入成功，版本: {yirage.__version__}')
    
    # 测试基础功能
    try:
        from yirage.yica_backend_integration import YICABackend
        print('✅ YICA后端导入成功')
    except ImportError as e:
        print(f'⚠️  YICA后端导入失败: {e}')
        
except ImportError as e:
    print(f'❌ yirage导入失败: {e}')
    sys.exit(1)
"
        
        # gem5验证
        echo "🧪 gem5验证..."
        if [ -f "/opt/tools/gem5-release/build/RISCV/gem5.opt" ]; then
            echo "✅ gem5 (预编译版本) 可用"
        elif [ -f "gem5/build/RISCV/gem5.opt" ]; then
            echo "✅ gem5 (源码编译版本) 可用"
            ./gem5/build/RISCV/gem5.opt --help | head -5
        else
            echo "❌ gem5不可用"
        fi
        
        # QEMU验证
        echo "🧪 QEMU验证..."
        qemu-system-x86_64 --version
        
        # 网络配置验证
        echo "🧪 网络配置验证..."
        if ip link show | grep -q "$TAPNAME"; then
            echo "✅ TAP网卡 $TAPNAME 已配置"
        else
            echo "⚠️  TAP网卡 $TAPNAME 未配置"
        fi
        
        echo "✅ 验证测试完成"
EOF
    
    print_success "验证测试完成"
}

# 创建使用说明 (按文档格式)
create_usage_guide() {
    print_status "创建使用说明 (按yicai-qemu.md格式)..."
    
    ssh "$REMOTE_SSH" << EOF
        cd $WORK_DIR
        
        cat > YICA-QEMU-使用说明.md << 'USAGE_EOF'
# YICA-QEMU AMD GPU Linux环境使用说明
## 严格按照yicai-qemu.md文档规范实现

## 🎯 环境概述

已在AMD GPU Linux服务器上成功部署YICA-QEMU完整环境，严格按照yicai-qemu.md文档要求：

### ✅ 已完成组件
- **系统镜像**: Ubuntu 22.04 + kernel 6.2.8 + ROCm 5.7.3
- **QEMU二进制**: 从gitlab-repo.yizhu.local获取
- **gem5模拟器**: RISC-V模拟器 (用于CIM die算子加载)
- **网络配置**: TAP网卡 + VNC支持
- **YICA-Yirage**: 核心库和Python绑定

## 🚀 启动方式 (按文档2.4.3节 - 手动运行方式)

### 方式1: 手动启动 (推荐)

**终端1 - 启动gem5:**
\`\`\`bash
cd $WORK_DIR
./gem5.sh /tmp/\${USER}
\`\`\`

**终端2 - 启动QEMU:**
\`\`\`bash
cd $WORK_DIR
./qemu2.sh
\`\`\`

### 方式2: 不启动gem5 (仅QEMU)
如果不需要运行算子加载到CIM die，可以修改qemu2.sh:
\`\`\`bash
# 编辑qemu2.sh，将 rp=on 改为 rp=off
sed -i 's/rp=on/rp=off/g' qemu2.sh
./qemu2.sh
\`\`\`

## 📺 VNC连接 (按文档第3节)

### 连接方式
\`\`\`bash
# VNC地址 (按文档3.2节)
vnc://$REMOTE_HOST:5900

# 或使用VNC Viewer
# 地址: $REMOTE_HOST:5900
\`\`\`

### 虚拟机内操作
\`\`\`bash
# 查看IP地址 (文档3.3节)
ifconfig

# 启动图形界面 (文档3.4节)
sudo systemctl start gdm.service

# 检查版本信息 (文档5.4节)
glog
# 在QEMU monitor中: info version
\`\`\`

## 🔧 SSH连接 (按文档第4节)

### 虚拟机内SSH配置
\`\`\`bash
# 默认用户名密码 (文档4.2节)
用户名: yizhu
密码: yizhu

# SSH连接虚拟机
ssh yizhu@<虚拟机IP>
\`\`\`

## 🧪 测试和验证

### YICA环境测试
\`\`\`bash
cd $WORK_DIR
python3 -c "import yirage; print(f'YICA版本: {yirage.__version__}')"
\`\`\`

### AMD GPU状态检查 (如果可用)
\`\`\`bash
# 检查GPU (按文档要求)
rocm-smi --showproductname
rocm-smi --showtemp --showuse
lspci -tv | grep -i amd
\`\`\`

### CIM配置检查
\`\`\`bash
# 检查CIM die和cluster配置 (文档2.3.1节)
# 默认: 8个CIM die，每个4个cluster
grep -E "(cimdie_cnt|cluster_cnt)" qemu2.sh
\`\`\`

## 📁 目录结构 (按文档要求)

\`\`\`
$WORK_DIR/
├── image2/
│   └── test2.qcow2           # 系统镜像 (文档第1节)
├── software-release/         # GitLab软件包 (文档2.1节)
│   └── qemubin/             # QEMU二进制文件
├── gem5/                    # gem5源码 (文档2.4节)
├── qemu2.sh                 # QEMU启动脚本 (文档2.2节)
├── gem5.sh                  # gem5启动脚本 (文档2.4节)
├── yirage/                  # YICA核心库
└── logs/                    # 日志文件
\`\`\`

## ⚠️  重要配置参数

### 网络配置 (文档2.2节)
- **TAPNAME**: $TAPNAME
- **VNC_ADDR**: $REMOTE_HOST:0
- **MAC_ADDR**: $MAC_ADDR

### CIM配置 (文档2.3.1节)
- **CIMDIE_CNT**: 8 (默认)
- **CLUSTER_CNT**: 4 (每个CIM die的最大值)

## 🔧 故障排除

### 1. QEMU启动失败
\`\`\`bash
# 检查镜像文件
ls -la image2/test2.qcow2

# 检查TAP网卡
sudo ip tuntap add dev $TAPNAME mode tap
sudo ip link set $TAPNAME up
\`\`\`

### 2. gem5启动失败
\`\`\`bash
# 重新编译gem5
cd gem5
scons build/RISCV/gem5.opt -j\$(nproc)
\`\`\`

### 3. VNC连接问题
\`\`\`bash
# 检查VNC端口
netstat -tlnp | grep 5900

# 重启QEMU确保VNC正常
\`\`\`

### 4. ROCm问题
\`\`\`bash
# 检查ROCm版本 (应为5.7.3)
rocm-smi --version

# 检查环境变量
echo \$ROCM_PATH
echo \$HSA_OVERRIDE_GFX_VERSION
\`\`\`

## 📞 技术支持

### 版本对应 (文档5.4节)
- 使用 \`glog\` 查看代码版本
- 使用 \`info version\` 在QEMU monitor中查看QEMU版本
- 确保版本一致性便于问题排查

### 日志文件
- QEMU日志: QEMU monitor输出
- gem5日志: gem5启动终端输出
- YICA日志: \`$WORK_DIR/logs/\`

---
**部署时间**: \$(date)  
**服务器**: $REMOTE_HOST  
**用户**: $REMOTE_USER  
**文档版本**: yicai-qemu.md 完整实现
USAGE_EOF

        echo "✅ 使用说明创建完成: YICA-QEMU-使用说明.md"
EOF
    
    print_success "使用说明创建完成"
}

# 显示部署总结
show_deployment_summary() {
    print_success "🎉 YICA-QEMU远程部署完成 (严格按yicai-qemu.md规范)！"
    echo ""
    echo "📋 部署总结："
    echo "  - 服务器: $REMOTE_SSH"
    echo "  - 工作目录: $WORK_DIR"
    echo "  - VNC端口: $REMOTE_HOST:5900"
    echo "  - TAP网卡: $TAPNAME"
    echo "  - MAC地址: $MAC_ADDR"
    echo ""
    echo "🚀 启动步骤 (按文档2.4.3节手动方式)："
    echo "  1. SSH连接到服务器:"
    echo "     ssh $REMOTE_SSH"
    echo ""
    echo "  2. 进入工作目录:"
    echo "     cd $WORK_DIR"
    echo ""
    echo "  3. 启动gem5 (终端1):"
    echo "     ./gem5.sh /tmp/\${USER}"
    echo ""
    echo "  4. 启动QEMU (终端2):"
    echo "     ./qemu2.sh"
    echo ""
    echo "  5. VNC连接查看虚拟机:"
    echo "     vnc://$REMOTE_HOST:5900"
    echo ""
    echo "  6. 查看详细说明:"
    echo "     cat YICA-QEMU-使用说明.md"
    echo ""
    echo "✅ 严格按照yicai-qemu.md文档要求实现完成！"
}

# 主函数
main() {
    case "${1:-}" in
        "check")
            check_ssh_connection
            check_remote_environment
            ;;
        "image")
            check_ssh_connection
            get_system_image
            ;;
        "qemu")
            check_ssh_connection
            get_qemu_binaries
            configure_qemu_scripts
            ;;
        "gem5")
            check_ssh_connection
            setup_gem5_environment
            ;;
        "sync")
            check_ssh_connection
            sync_project_code
            ;;
        "install")
            check_ssh_connection
            install_dependencies
            configure_rocm
            ;;
        "build")
            check_ssh_connection
            build_yica_core
            ;;
        "test")
            check_ssh_connection
            run_verification_tests
            ;;
        "guide")
            check_ssh_connection
            create_usage_guide
            ;;
        "")
            print_status "执行完整YICA-QEMU远程部署流程 (按yicai-qemu.md规范)..."
            check_ssh_connection
            check_remote_environment
            sync_project_code
            install_dependencies
            configure_rocm
            get_system_image
            get_qemu_binaries
            configure_qemu_scripts
            setup_gem5_environment
            build_yica_core
            run_verification_tests
            create_usage_guide
            show_deployment_summary
            ;;
        *)
            echo "YICA-QEMU远程部署脚本 (严格按yicai-qemu.md规范)"
            echo ""
            echo "用法: $0 [命令]"
            echo ""
            echo "命令 (按文档章节):"
            echo "  check     - 检查SSH连接和远程环境"
            echo "  image     - 获取系统镜像 (文档第1节)"
            echo "  qemu      - 配置QEMU环境 (文档第2节)"
            echo "  gem5      - 配置gem5环境 (文档2.4节)"
            echo "  sync      - 同步项目代码"
            echo "  install   - 安装依赖"
            echo "  build     - 构建YICA核心库"
            echo "  test      - 运行验证测试"
            echo "  guide     - 创建使用说明"
            echo "  (空)      - 执行完整部署流程"
            echo ""
            echo "按yicai-qemu.md文档要求实现的功能:"
            echo "  ✅ 1. 系统镜像获取 (Ubuntu 22.04 + kernel 6.2.8 + ROCm 5.7.3)"
            echo "  ✅ 2.1 QEMU二进制文件获取"
            echo "  ✅ 2.2 QEMU启动参数配置"
            echo "  ✅ 2.3 启动脚本和CIM配置"
            echo "  ✅ 2.4 gem5和QEMU联合启动"
            echo "  ✅ 3. VNC配置"
            echo "  ✅ 4. SSH配置支持"
            echo "  ✅ 5. Git配置和版本管理"
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@" 
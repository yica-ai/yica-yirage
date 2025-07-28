# YICA-QEMU Docker化部署使用文档

## 📋 概述

本文档描述了如何在远程服务器上通过Docker容器化部署YICA-QEMU环境，完全避免sudo权限问题。该方案将整个YICA-QEMU环境打包到Docker容器中，提供VNC远程桌面访问。

## 🎯 部署目标

- **服务器**: `johnson.chen@10.11.60.58`
- **环境**: Ubuntu 22.04 + Docker + YICA-QEMU
- **访问方式**: VNC (传统客户端 + Web浏览器)
- **权限**: 无需sudo，完全用户权限部署

## 🚀 快速部署

### 1. 一键部署命令

```bash
# 完整部署流程 (约15-20分钟)
./scripts/docker_yica_deployment.sh

# 或分步执行
./scripts/docker_yica_deployment.sh check    # 检查环境
./scripts/docker_yica_deployment.sh sync     # 同步代码
./scripts/docker_yica_deployment.sh build    # 构建镜像
./scripts/docker_yica_deployment.sh start    # 启动容器
./scripts/docker_yica_deployment.sh verify   # 验证部署
```

### 2. 部署结果验证

```bash
# 检查容器状态
./scripts/yica_docker_manager.sh status

# 预期输出
✅ 容器正在运行 (ID: xxxxxxxx)
🌐 访问地址:
  Web VNC:   http://10.11.60.58:6080 (密码: yica)
  VNC客户端: vnc://10.11.60.58:5900 (密码: yica)
```

## 🌐 访问方式

### 方式1: Web VNC (推荐)

1. **打开浏览器**，访问: `http://10.11.60.58:6080`
2. **点击连接按钮**
3. **输入VNC密码**: `yica`
4. **开始使用** Ubuntu桌面环境

### 方式2: VNC客户端

1. **安装VNC客户端** (如RealVNC Viewer、TightVNC等)
2. **连接地址**: `10.11.60.58:5900`
3. **输入密码**: `yica`
4. **连接成功** 进入桌面

### 方式3: SSH进入容器

```bash
# 通过管理脚本进入
./scripts/yica_docker_manager.sh shell

# 或直接Docker命令
ssh johnson.chen@10.11.60.58 "docker exec -it yica-qemu-container bash"
```

## 🔧 管理操作

### 容器管理

```bash
# 查看容器状态
./scripts/yica_docker_manager.sh status

# 进入容器shell
./scripts/yica_docker_manager.sh shell

# 查看容器日志
./scripts/yica_docker_manager.sh logs

# 重启容器
./scripts/yica_docker_manager.sh restart

# 停止容器
./scripts/yica_docker_manager.sh stop

# 启动容器
./scripts/yica_docker_manager.sh start
```

### 服务管理

```bash
# 重启VNC服务
./scripts/yica_docker_manager.sh shell
# 在容器内执行:
vncserver -kill :1
vncserver :1 -geometry 1024x768 -depth 24

# 重启noVNC服务
pkill websockify
websockify --web=/usr/share/novnc/ 6080 localhost:5901 &
```

## 🧪 YICA环境使用

### 启动YICA-QEMU

在VNC桌面环境中：

```bash
# 打开终端
cd /home/yica/workspace

# 方式1: 启动完整YICA环境 (gem5 + QEMU)
# 终端1 - 启动gem5
./gem5.sh /tmp/yica

# 终端2 - 启动QEMU
./qemu2.sh

# 方式2: 仅启动QEMU (不使用gem5)
# 修改qemu2.sh中的rp参数
sed -i 's/rp=on/rp=off/g' qemu2.sh
./qemu2.sh
```

### YICA Python环境

```bash
# 进入容器
./scripts/yica_docker_manager.sh shell

# 测试YICA环境
cd /home/yica/workspace
python3 -c "
import sys
sys.path.insert(0, '/home/yica/workspace/yirage/python')
import yirage
print(f'YICA版本: {yirage.__version__}')
"

# 运行YICA示例
cd yirage/demo
python3 demo_gated_mlp.py
```

## 📁 目录结构

```
/home/johnson.chen/yica-docker-workspace/
├── docker/
│   ├── Dockerfile              # Docker构建文件
│   ├── docker-compose.yml      # Docker Compose配置
│   └── start-services.sh       # 容器启动脚本
├── image2/
│   └── test2.qcow2            # QEMU系统镜像
├── software-release/           # GitLab软件包
│   └── qemubin/               # QEMU二进制文件
├── qemu2.sh                   # QEMU启动脚本
├── gem5.sh                    # gem5启动脚本
└── yirage/                    # YICA核心库源码
```

## 🔧 端口映射

| 服务 | 容器端口 | 主机端口 | 说明 |
|------|----------|----------|------|
| VNC Server | 5901 | 5900 | 传统VNC客户端连接 |
| noVNC Web | 6080 | 6080 | Web浏览器VNC访问 |
| QEMU Monitor | 4444 | 4444 | QEMU监控接口 |
| gem5 Interface | 3456 | 3456 | gem5通信端口 |
| SSH | 22 | 2222 | 容器SSH访问 |

## ⚠️ 故障排除

### 1. 容器无法启动

```bash
# 检查Docker服务
ssh johnson.chen@10.11.60.58 "docker info"

# 重新构建镜像
./scripts/docker_yica_deployment.sh build

# 查看详细错误
./scripts/yica_docker_manager.sh logs
```

### 2. VNC连接失败

```bash
# 检查VNC端口
curl -I http://10.11.60.58:6080

# 重启VNC服务
./scripts/yica_docker_manager.sh shell
vncserver -kill :1
vncserver :1 -geometry 1024x768 -depth 24
```

### 3. 网络访问问题

```bash
# 检查端口映射
ssh johnson.chen@10.11.60.58 "docker port yica-qemu-container"

# 检查防火墙
ssh johnson.chen@10.11.60.58 "sudo ufw status"
```

### 4. 系统镜像问题

```bash
# 重新获取镜像
./scripts/docker_yica_deployment.sh image

# 检查镜像文件
ssh johnson.chen@10.11.60.58 "ls -la /home/johnson.chen/yica-docker-workspace/image2/"
```

### 5. QEMU启动失败

```bash
# 进入容器检查
./scripts/yica_docker_manager.sh shell

# 检查QEMU二进制
ls -la /home/yica/workspace/software-release/qemubin/

# 手动启动QEMU测试
cd /home/yica/workspace
./qemu2.sh
```

## 📊 性能优化

### 1. 容器资源限制

```bash
# 修改docker-compose.yml
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 8G
    reservations:
      cpus: '2.0'
      memory: 4G
```

### 2. VNC显示优化

```bash
# 修改VNC分辨率
vncserver -kill :1
vncserver :1 -geometry 1920x1080 -depth 24

# 启用硬件加速 (如果支持)
export LIBGL_ALWAYS_SOFTWARE=1
```

## 🔐 安全配置

### 1. 修改VNC密码

```bash
# 进入容器
./scripts/yica_docker_manager.sh shell

# 设置新密码
vncpasswd
# 输入新密码并确认

# 重启VNC服务
vncserver -kill :1
vncserver :1 -geometry 1024x768 -depth 24
```

### 2. 网络安全

```bash
# 限制访问IP (可选)
# 修改docker-compose.yml中的端口映射
ports:
  - "127.0.0.1:5900:5900"  # 仅本地访问
  - "10.11.60.58:6080:6080"  # 指定IP访问
```

## 📝 日志管理

### 查看日志

```bash
# 容器总体日志
./scripts/yica_docker_manager.sh logs

# VNC服务日志
./scripts/yica_docker_manager.sh shell
tail -f ~/.vnc/*.log

# QEMU运行日志
./scripts/yica_docker_manager.sh shell
cd /home/yica/workspace
./qemu2.sh 2>&1 | tee qemu.log
```

### 日志轮转

```bash
# 清理Docker日志
ssh johnson.chen@10.11.60.58 "docker logs --tail 100 yica-qemu-container"

# 清理VNC日志
./scripts/yica_docker_manager.sh shell
rm -f ~/.vnc/*.log
```

## 🔄 更新和维护

### 更新YICA代码

```bash
# 同步最新代码
./scripts/docker_yica_deployment.sh sync

# 重新构建镜像
./scripts/docker_yica_deployment.sh build

# 重启容器
./scripts/yica_docker_manager.sh restart
```

### 系统维护

```bash
# 清理Docker资源
ssh johnson.chen@10.11.60.58 "docker system prune -f"

# 更新系统包 (在容器内)
./scripts/yica_docker_manager.sh shell
apt update && apt upgrade -y
```

## 📞 技术支持

### 版本信息

```bash
# Docker版本
ssh johnson.chen@10.11.60.58 "docker --version"

# 容器信息
./scripts/yica_docker_manager.sh shell
cat /etc/os-release
python3 --version
```

### 联系方式

- **部署脚本**: `scripts/docker_yica_deployment.sh`
- **管理脚本**: `scripts/yica_docker_manager.sh`
- **配置文件**: `docker/docker-compose.yml`

---

**文档版本**: v1.0  
**创建时间**: 2025-01-28  
**适用环境**: Ubuntu 22.04 + Docker + YICA-QEMU  
**部署方式**: Docker容器化 (无sudo权限) 
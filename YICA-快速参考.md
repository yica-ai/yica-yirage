# YICA-QEMU Docker 快速参考

## 🚀 一键部署
```bash
./scripts/docker_yica_deployment.sh
```

## 🌐 访问地址
- **Web VNC**: http://10.11.60.58:6080 (密码: yica)
- **VNC客户端**: 10.11.60.58:5900 (密码: yica)

## 🔧 常用命令

### 容器管理
```bash
./scripts/yica_docker_manager.sh status    # 查看状态
./scripts/yica_docker_manager.sh shell     # 进入容器
./scripts/yica_docker_manager.sh logs      # 查看日志
./scripts/yica_docker_manager.sh restart   # 重启容器
```

### 部署管理
```bash
./scripts/docker_yica_deployment.sh check   # 检查环境
./scripts/docker_yica_deployment.sh build   # 构建镜像
./scripts/docker_yica_deployment.sh start   # 启动容器
./scripts/docker_yica_deployment.sh verify  # 验证部署
```

## 🧪 YICA使用

### 在VNC桌面中启动QEMU
```bash
cd /home/yica/workspace
./qemu2.sh
```

### Python环境测试
```bash
python3 -c "import sys; sys.path.insert(0, '/home/yica/workspace/yirage/python'); import yirage; print(f'YICA版本: {yirage.__version__}')"
```

## ⚠️ 故障排除
```bash
# 重启VNC服务
./scripts/yica_docker_manager.sh shell
vncserver -kill :1
vncserver :1 -geometry 1024x768 -depth 24

# 检查端口
curl -I http://10.11.60.58:6080
```

---
**服务器**: johnson.chen@10.11.60.58  
**工作目录**: /home/johnson.chen/yica-docker-workspace 
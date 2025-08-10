# 部署运维指南

本目录包含YICA/YiRage的部署、运维和环境配置相关文档。

## 📖 文档列表

### 部署方案
- **[Docker部署](docker-deployment.md)** - 使用Docker容器化部署YICA-QEMU环境
- **[QEMU设置](qemu-setup.md)** - QEMU虚拟化环境的详细配置
- **[部署报告](deployment-report.md)** - Docker化部署的实施报告

## 🚀 部署选项

### 1. Docker部署 (推荐)
**适用场景**: 开发、测试、演示环境
- ✅ 无需sudo权限
- ✅ 环境隔离
- ✅ 快速部署
- ✅ 易于维护

### 2. 原生部署
**适用场景**: 生产环境、高性能需求
- ✅ 最佳性能
- ✅ 直接硬件访问
- ⚠️ 需要系统管理权限
- ⚠️ 依赖管理复杂

### 3. 混合部署
**适用场景**: 多环境支持
- ✅ 灵活配置
- ✅ 分层管理
- ⚠️ 复杂度较高

## 🌐 部署架构

### Docker化架构
```
┌─────────────────────────────────┐
│         主机服务器               │
│    (johnson.chen@10.11.60.58)  │
│                                 │
│ ┌─────────────────────────────┐ │
│ │      Docker容器             │ │
│ │  (yica-qemu-container)     │ │
│ │                            │ │
│ │ ┌─────────────────────────┐ │ │
│ │ │    Ubuntu 22.04         │ │ │
│ │ │  + YICA-QEMU环境        │ │ │
│ │ │  + VNC服务器            │ │ │
│ │ │  + noVNC Web界面        │ │ │
│ │ └─────────────────────────┘ │ │
│ └─────────────────────────────┘ │
│                                 │
│ 端口映射:                       │
│ 5900 → VNC服务                  │
│ 6080 → Web VNC                  │
│ 4444 → QEMU监控                 │
└─────────────────────────────────┘
```

### 网络配置
| 服务 | 容器端口 | 主机端口 | 协议 | 说明 |
|------|----------|----------|------|------|
| VNC Server | 5901 | 5900 | TCP | 传统VNC客户端 |
| noVNC Web | 6080 | 6080 | HTTP | Web浏览器访问 |
| QEMU Monitor | 4444 | 4444 | TCP | QEMU控制接口 |
| gem5 Interface | 3456 | 3456 | TCP | gem5通信端口 |
| SSH | 22 | 2222 | TCP | 容器SSH访问 |

## 🔧 快速部署

### 一键部署
```bash
# 完整部署流程
./scripts/docker_yica_deployment.sh

# 分步执行
./scripts/docker_yica_deployment.sh check    # 环境检查
./scripts/docker_yica_deployment.sh sync     # 代码同步
./scripts/docker_yica_deployment.sh build    # 镜像构建
./scripts/docker_yica_deployment.sh start    # 容器启动
./scripts/docker_yica_deployment.sh verify   # 部署验证
```

### 快速访问
```bash
# Web访问 (推荐)
http://10.11.60.58:6080

# VNC客户端
vnc://10.11.60.58:5900

# SSH访问
ssh -p 2222 yica@10.11.60.58
```

## 🛠️ 管理操作

### 容器管理
```bash
# 状态检查
./scripts/yica_docker_manager.sh status

# 容器操作
./scripts/yica_docker_manager.sh start     # 启动
./scripts/yica_docker_manager.sh stop      # 停止
./scripts/yica_docker_manager.sh restart   # 重启
./scripts/yica_docker_manager.sh shell     # 进入shell

# 日志查看
./scripts/yica_docker_manager.sh logs
```

### 服务管理
```bash
# VNC服务重启
vncserver -kill :1
vncserver :1 -geometry 1024x768 -depth 24

# noVNC服务重启
pkill websockify
websockify --web=/usr/share/novnc/ 6080 localhost:5901 &
```

## 📊 系统要求

### 最低要求
- **CPU**: 2核心
- **内存**: 4GB RAM
- **存储**: 20GB可用空间
- **网络**: 互联网连接（用于下载依赖）

### 推荐配置
- **CPU**: 4核心或更多
- **内存**: 8GB RAM或更多
- **存储**: 50GB SSD
- **GPU**: 可选，用于CUDA后端

### 软件要求
- **Docker**: 20.10+
- **Docker Compose**: 1.29+
- **Python**: 3.8+
- **Git**: 2.20+

## ⚠️ 故障排除

### 常见问题

#### 1. 容器无法启动
```bash
# 检查Docker状态
docker info

# 查看详细错误
./scripts/yica_docker_manager.sh logs

# 重新构建镜像
./scripts/docker_yica_deployment.sh build
```

#### 2. VNC连接失败
```bash
# 检查端口状态
curl -I http://10.11.60.58:6080

# 重启VNC服务
./scripts/yica_docker_manager.sh shell
vncserver -kill :1
vncserver :1 -geometry 1024x768 -depth 24
```

#### 3. 性能问题
```bash
# 检查资源使用
docker stats yica-qemu-container

# 调整资源限制
# 编辑 docker-compose.yml 中的资源配置
```

### 日志管理
```bash
# 容器日志
./scripts/yica_docker_manager.sh logs

# VNC日志
tail -f ~/.vnc/*.log

# QEMU日志
./qemu2.sh 2>&1 | tee qemu.log
```

## 🔐 安全配置

### 访问控制
- 修改默认VNC密码
- 限制网络访问范围
- 使用防火墙规则

### 数据保护
- 定期备份重要数据
- 使用安全的传输协议
- 监控系统访问日志

## 📈 性能优化

### 容器优化
```yaml
# docker-compose.yml
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 8G
    reservations:
      cpus: '2.0'
      memory: 4G
```

### 网络优化
- 使用本地网络存储
- 优化网络带宽配置
- 减少网络延迟

## 🔗 相关文档

- [架构设计](../architecture/) - 了解系统架构
- [开发指南](../development/) - 开发环境配置
- [快速入门](../getting-started/) - 基础概念
- [API文档](../api/) - 编程接口

---

*部署文档将根据实际使用情况持续更新和完善。如有问题，请参考故障排除章节或联系维护团队。*

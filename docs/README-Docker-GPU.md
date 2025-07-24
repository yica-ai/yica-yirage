# YICA GPU Docker 使用指南

本文档介绍如何使用Docker在支持NVIDIA GPU的环境中运行YICA运行时优化器。

## 前置要求

### 硬件要求
- NVIDIA GPU (支持CUDA 11.8+)
- 至少8GB GPU显存 (推荐16GB+)
- 至少16GB系统内存
- 至少50GB可用磁盘空间

### 软件要求
- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA驱动 (版本470+)
- NVIDIA Container Toolkit

## 环境准备

### 1. 安装NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# CentOS/RHEL
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo

sudo yum install -y nvidia-docker2
sudo systemctl restart docker
```

### 2. 验证GPU支持

```bash
# 测试NVIDIA Docker支持
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

## 快速开始

### 1. 一键启动 (推荐)

```bash
# 后台启动所有服务
./docker/run-yica-gpu.sh -d

# 前台启动 (查看实时日志)
./docker/run-yica-gpu.sh
```

### 2. 使用Docker Compose

```bash
# 构建并启动
cd docker
docker-compose -f docker-compose.yica-gpu.yml up -d

# 查看状态
docker-compose -f docker-compose.yica-gpu.yml ps

# 查看日志
docker-compose -f docker-compose.yica-gpu.yml logs -f yica-runtime
```

## 服务访问

启动成功后，可以通过以下地址访问各种服务：

| 服务 | 地址 | 用途 |
|------|------|------|
| YICA Runtime API | http://localhost:8080 | 运行时优化器API |
| Performance Monitor | http://localhost:8081 | 性能监控接口 |
| ML Optimizer API | http://localhost:8082 | 机器学习优化器API |
| Grafana Dashboard | http://localhost:3000 | 性能监控仪表板 |
| Jupyter Lab | http://localhost:8888 | 开发和实验环境 |

### 默认认证信息

- **Grafana**: admin / yica2024
- **Jupyter**: token: yica2024

## 配置说明

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `YICA_MODE` | runtime | 运行模式 (runtime/monitor/ml_optimizer) |
| `YICA_GPU_ENABLED` | true | 是否启用GPU加速 |
| `YICA_LOG_LEVEL` | INFO | 日志级别 (DEBUG/INFO/WARN/ERROR) |
| `YICA_MONITORING_ENABLED` | true | 是否启用性能监控 |
| `YICA_ML_OPTIMIZATION_ENABLED` | true | 是否启用ML优化 |
| `CUDA_VISIBLE_DEVICES` | all | 可见的GPU设备 |

### 运行时配置

默认配置文件位于容器内的 `/workspace/yica-runtime/configs/runtime_config.json`，包含：

```json
{
    "runtime": {
        "mode": "runtime",
        "gpu_enabled": true,
        "log_level": "INFO",
        "monitoring_enabled": true,
        "ml_optimization_enabled": true
    },
    "performance_monitor": {
        "collection_frequency_hz": 1000,
        "sliding_window_size": 100,
        "anomaly_detection_enabled": true
    },
    "ml_optimizer": {
        "model_type": "lstm",
        "learning_rate": 0.001,
        "batch_size": 32,
        "hidden_size": 128,
        "online_learning_enabled": true
    },
    "hardware": {
        "cim_array_size": 256,
        "spm_size_mb": 32,
        "memory_bandwidth_gbps": 1024,
        "compute_units": 64
    }
}
```

## 常用操作

### 查看GPU使用情况

```bash
# 在容器内查看GPU状态
docker exec -it yica-runtime-gpu nvidia-smi

# 持续监控GPU使用
docker exec -it yica-runtime-gpu watch -n 1 nvidia-smi
```

### 查看服务日志

```bash
# 查看运行时日志
./docker/run-yica-gpu.sh logs

# 查看特定服务日志
./docker/run-yica-gpu.sh --service=monitor logs
./docker/run-yica-gpu.sh --service=jupyter logs

# 使用docker-compose查看
docker-compose -f docker/docker-compose.yica-gpu.yml logs -f yica-runtime
```

### 进入容器调试

```bash
# 进入运行时容器
./docker/run-yica-gpu.sh shell

# 进入Jupyter容器
./docker/run-yica-gpu.sh --service=jupyter shell

# 直接使用docker
docker exec -it yica-runtime-gpu /bin/bash
```

### 性能监控

```bash
# 查看实时性能指标
curl http://localhost:8081/metrics

# 查看GPU利用率
curl http://localhost:8081/gpu/utilization

# 查看内存使用情况
curl http://localhost:8081/memory/usage
```

## 开发和调试

### 挂载本地代码

如需在开发过程中实时修改代码，可以挂载本地目录：

```yaml
# 在docker-compose.yica-gpu.yml中添加
volumes:
  - ./mirage:/workspace/yica-optimizer/mirage
  - ./good-kernels:/workspace/yica-optimizer/good-kernels
```

### 自定义配置

```bash
# 创建自定义配置目录
mkdir -p ./docker/yica-configs

# 复制并修改配置文件
cp docker/yica-configs/runtime_config.json ./docker/yica-configs/my_config.json

# 使用自定义配置启动
YICA_CONFIG=my_config ./docker/run-yica-gpu.sh -d
```

### 调试模式

```bash
# 启用调试日志
YICA_LOG_LEVEL=DEBUG ./docker/run-yica-gpu.sh

# 禁用GPU进行CPU调试
./docker/run-yica-gpu.sh --no-gpu
```

## 故障排除

### GPU相关问题

**问题**: `nvidia-smi` 在容器内不可用
```bash
# 检查主机GPU状态
nvidia-smi

# 检查NVIDIA Docker支持
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# 重启Docker服务
sudo systemctl restart docker
```

**问题**: GPU内存不足
```bash
# 限制GPU内存使用
export CUDA_VISIBLE_DEVICES=0  # 只使用第一块GPU

# 检查GPU内存使用
docker exec -it yica-runtime-gpu nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### 性能问题

**问题**: 容器启动缓慢
```bash
# 增加共享内存大小
docker run --shm-size=2g ...

# 或在docker-compose中设置
shm_size: 2gb
```

**问题**: 网络连接问题
```bash
# 检查端口占用
netstat -tulpn | grep -E '8080|8081|8082|3000|8888'

# 修改端口映射
vim docker/docker-compose.yica-gpu.yml
```

### 日志和监控

```bash
# 查看详细启动日志
./docker/run-yica-gpu.sh 2>&1 | tee yica-startup.log

# 检查容器健康状态
docker inspect yica-runtime-gpu | jq '.[0].State.Health'

# 查看资源使用情况
docker stats yica-runtime-gpu
```

## 性能优化

### GPU优化

```bash
# 设置GPU性能模式
nvidia-smi -pm 1

# 设置最大时钟频率
nvidia-smi -ac 877,1911  # 根据GPU型号调整
```

### 容器优化

```yaml
# 在docker-compose中优化资源配置
deploy:
  resources:
    limits:
      cpus: '8.0'
      memory: 16G
    reservations:
      cpus: '4.0'
      memory: 8G
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

### 存储优化

```bash
# 使用SSD存储卷
docker volume create --driver local \
  --opt type=none \
  --opt o=bind \
  --opt device=/path/to/ssd/storage \
  yica-fast-storage
```

## 生产部署

### 安全配置

```yaml
# 生产环境安全设置
security_opt:
  - no-new-privileges:true
  - seccomp:unconfined
user: "1000:1000"
read_only: true
tmpfs:
  - /tmp
  - /var/tmp
```

### 监控和日志

```yaml
# 配置日志轮转
logging:
  driver: "json-file"
  options:
    max-size: "100m"
    max-file: "5"
    compress: "true"
```

### 高可用配置

```bash
# 使用Docker Swarm进行集群部署
docker swarm init
docker stack deploy -c docker-compose.yica-gpu.yml yica-stack
```

## 参考资料

- [NVIDIA Container Toolkit文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Docker GPU支持文档](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- [CUDA Docker镜像](https://hub.docker.com/r/nvidia/cuda)
- [YICA架构文档](./YICA_ARCH.md)
- [Mirage框架文档](./mirage/README.md) 
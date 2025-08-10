# YICA 虚拟环境性能测试指南

## 🎯 概述

本指南介绍如何在服务器上测试 **yirage 1.0.2** 结合 **YICA 虚拟硬件环境** 的算子性能。通过 Docker 容器化部署，实现了完整的 YICA 存算一体架构仿真环境，可以测试 yirage 生成的算子在 YICA 硬件上的实际性能表现。

## 🏗️ 环境架构

```
┌─────────────────────────────────────────────────────────────┐
│                    本地开发环境                               │
├─────────────────────────────────────────────────────────────┤
│  • YZ-optimizer-bin 项目                                     │
│  • 性能测试脚本                                               │
│  • SSH 连接到远程服务器                                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ SSH
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                远程服务器 (10.11.60.58)                     │
├─────────────────────────────────────────────────────────────┤
│  Docker 容器: yica-qemu-container                          │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  • Ubuntu 22.04 + yirage 1.0.2                        │ │
│  │  • QEMU 虚拟机 (YICA-G100 设备仿真)                    │ │
│  │  • gem5 RISC-V 模拟器                                  │ │
│  │  • YICA 虚拟硬件服务                                   │ │
│  │  • 性能测试环境                                         │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 方式1: 一键快速测试 (推荐)

```bash
# 在本地项目根目录执行
./scripts/start_yica_test.sh
```

这个脚本会自动完成：
1. ✅ 检查服务器连接
2. ✅ 启动 Docker 容器和 YICA 虚拟硬件
3. ✅ 运行快速性能测试 (4个核心算子)
4. ✅ 收集和显示测试结果

**预期输出示例:**
```
🚀 YICA 性能测试快速启动...
🔗 连接服务器: johnson.chen@10.11.60.58
📁 工作目录: /home/johnson.chen/yica-docker-workspace

[步骤] 1. 检查服务器连接...
[成功] 服务器连接正常

[步骤] 2. 检查和启动 Docker 环境...
🐳 检查 Docker 容器状态...
✅ YICA 容器正在运行
🧪 检查 yirage 环境...
✅ yirage 版本: 1.0.2
[成功] Docker 环境就绪

[步骤] 3. 启动 YICA 虚拟硬件服务...
🔧 启动 gem5 和 QEMU 虚拟硬件...
[成功] YICA 虚拟硬件服务已启动

[步骤] 4. 创建并运行性能测试...
🧠 YICA 快速性能测试
========================================
📅 时间: 2024-12-XX XX:XX:XX
🐍 Python: 3.11.x
🔥 PyTorch: 2.x.x
✅ yirage 版本: 1.0.2

🔢 测试矩阵乘法...
🧪 测试: 矩阵乘法_512x512
  ✅ 延迟: 12.345 ± 1.234 ms
  📊 吞吐: 81.0 ops/sec

🎯 测试注意力机制...
🧪 测试: 注意力机制
  ✅ 延迟: 45.678 ± 2.345 ms
  📊 吞吐: 21.9 ops/sec

📏 测试 RMSNorm...
🧪 测试: RMSNorm
  ✅ 延迟: 8.901 ± 0.567 ms
  📊 吞吐: 112.3 ops/sec

⚡ 测试 GELU 激活...
🧪 测试: GELU激活
  ✅ 延迟: 3.456 ± 0.234 ms
  📊 吞吐: 289.4 ops/sec

🎉 快速测试完成！

[成功] 性能测试执行完成

[步骤] 5. 获取测试结果...
📊 测试结果摘要:
==================
# YICA 快速性能测试报告

**测试时间**: 2024-12-XX XX:XX:XX
**yirage 版本**: 1.0.2
**测试环境**: YICA 虚拟硬件环境

## 测试结果

| 算子 | 平均延迟 (ms) | 吞吐量 (ops/sec) |
|------|---------------|------------------|
| 矩阵乘法_512x512 | 12.345 | 81.00 |
| 注意力机制 | 45.678 | 21.90 |
| RMSNorm | 8.901 | 112.30 |
| GELU激活 | 3.456 | 289.40 |

**平均延迟**: 17.595 ms
**总吞吐量**: 504.60 ops/sec

✅ YICA 后端可用，测试结果包含 YICA 优化效果

🎉 YICA 性能测试完成！
```

### 方式2: 完整性能测试

```bash
# 运行完整的基准测试套件 (100次迭代)
./scripts/yica_performance_test.sh

# 或者快速测试模式 (20次迭代)
./scripts/yica_performance_test.sh quick
```

## 📊 测试内容

### 核心算子测试

| 测试类别 | 具体算子 | 测试规模 | YICA 优化特性 |
|----------|----------|----------|---------------|
| **矩阵运算** | 矩阵乘法 | 256×256 到 1024×1024 | CIM 阵列并行计算 |
| **注意力机制** | Multi-Head Attention | batch=8, seq=512, hidden=768 | SPM 内存优化 + 算子融合 |
| **规范化** | LayerNorm, RMSNorm | 不同 batch 和序列长度 | 存算一体规范化 |
| **激活函数** | ReLU, GELU, SiLU, Tanh | 大规模张量激活 | 内存带宽优化 |

### 性能指标

- **延迟 (Latency)**: 单次操作执行时间 (毫秒)
- **吞吐量 (Throughput)**: 每秒操作数 (ops/sec)
- **GFLOPS**: 每秒浮点运算次数 (10^9 FLOPS)
- **内存使用**: 峰值内存占用 (MB)
- **加速比**: YICA vs PyTorch 性能提升倍数

## 🔧 YICA 虚拟硬件架构

### YICA-G100 规格仿真

```
YICA-G100 存算一体处理器 (虚拟)
├── CIM Dies: 8个
├── 每Die的Clusters: 4个  
├── 每Cluster的CIM阵列: 16个
├── SPM大小: 2GB/Die (总计16GB)
├── 峰值算力: 200 TOPS
├── 能效优化: 0.3-1.0 pJ/操作
└── 精度支持: INT8/FP16/混合精度
```

### 虚拟硬件服务

1. **gem5 RISC-V 模拟器**
   - 端口: 3456
   - 功能: RISC-V 指令集仿真
   - 配置: TimingSimpleCPU + 2GB 内存

2. **QEMU 虚拟机**
   - 端口: 4444 (Monitor)
   - 功能: YICA-G100 设备仿真
   - 配置: 8个 CIM Dies, 4个 Clusters

3. **VNC 图形界面**
   - VNC: 端口 5900
   - Web VNC: 端口 6080
   - 密码: yica

## 📈 性能分析

### YICA 优化效果

基于历史测试数据，YICA 架构在以下算子上表现出显著优化：

| 算子类型 | 预期加速比 | 优化机制 |
|----------|------------|----------|
| **矩阵乘法** | 2.0-3.0x | CIM 阵列并行 + 数据局部性 |
| **注意力机制** | 2.5-3.5x | SPM 内存层次 + 算子融合 |
| **RMSNorm** | 1.5-2.5x | 存算一体归约 + 内存优化 |
| **激活函数** | 1.2-2.0x | 向量化计算 + 带宽优化 |

### 性能瓶颈分析

1. **内存带宽限制**
   - 大规模矩阵运算受内存带宽影响
   - YICA SPM 层次结构可有效缓解

2. **计算并行度**
   - 小规模操作无法充分利用 CIM 阵列
   - 批处理和算子融合可提升并行度

3. **数据传输开销**
   - 主机-设备数据传输成本
   - YICA 就近计算减少数据移动

## 🛠️ 高级使用

### 自定义测试配置

```bash
# 修改测试参数
vim scripts/yica_performance_test.sh

# 关键配置
BENCHMARK_ITERATIONS=100  # 测试迭代次数
WARMUP_ITERATIONS=10      # 预热次数
TEST_MODE="comprehensive" # 测试模式: quick/comprehensive/stress
```

### 手动操作容器

```bash
# 连接到服务器
ssh johnson.chen@10.11.60.58

# 进入工作目录
cd /home/johnson.chen/yica-docker-workspace

# 进入容器
docker exec -it yica-qemu-container bash

# 容器内操作
cd /home/yica/workspace
export PYTHONPATH=/home/yica/workspace/yirage/python:$PYTHONPATH

# 手动运行测试
python3 quick_yica_test.py
```

### 查看虚拟硬件状态

```bash
# 检查 gem5 状态
docker exec yica-qemu-container ps aux | grep gem5

# 检查 QEMU 状态  
docker exec yica-qemu-container ps aux | grep qemu

# 查看服务日志
docker exec yica-qemu-container tail -f /home/yica/workspace/logs/gem5.log
docker exec yica-qemu-container tail -f /home/yica/workspace/logs/qemu.log

# 检查端口监听
docker exec yica-qemu-container netstat -tlnp | grep -E "(3456|4444)"
```

## 📁 结果文件

### 文件结构

```
yica_quick_results/
├── results.json          # 详细数值结果
├── report.md             # 性能分析报告
└── (可选扩展文件)

yica_performance_results/  # 完整测试结果
├── yica_performance_results.json
├── yica_performance_report.md
├── performance_comparison.png
└── speedup_analysis.png
```

### 下载结果到本地

```bash
# 下载快速测试结果
scp -r johnson.chen@10.11.60.58:/home/johnson.chen/yica-docker-workspace/yica_quick_results ./local_results

# 下载完整测试结果
scp -r johnson.chen@10.11.60.58:/home/johnson.chen/yica-docker-workspace/yica_performance_results ./full_results
```

## 🐛 故障排除

### 常见问题

#### 1. 服务器连接失败
```bash
# 检查网络连接
ping 10.11.60.58

# 检查 SSH 配置
ssh -v johnson.chen@10.11.60.58

# 确认用户权限
ssh johnson.chen@10.11.60.58 "whoami && pwd"
```

#### 2. Docker 容器未运行
```bash
# 手动启动容器
ssh johnson.chen@10.11.60.58
cd /home/johnson.chen/yica-docker-workspace
docker-compose up -d

# 或运行部署脚本
./scripts/docker_yica_deployment.sh
```

#### 3. yirage 导入失败
```bash
# 检查 Python 路径
docker exec yica-qemu-container python3 -c "import sys; print(sys.path)"

# 检查 yirage 文件
docker exec yica-qemu-container ls -la /home/yica/workspace/yirage/python/

# 重新安装 yirage
docker exec yica-qemu-container bash -c "
cd /home/yica/workspace/yirage
pip install -e . -v
"
```

#### 4. 虚拟硬件服务启动失败
```bash
# 检查 KVM 支持
ls -la /dev/kvm

# 手动启动服务
docker exec -it yica-qemu-container bash
/home/yica/workspace/gem5-docker.sh
/home/yica/workspace/qemu-docker.sh
```

### 调试模式

```bash
# 启用详细输出
./scripts/start_yica_test.sh 2>&1 | tee yica_test_debug.log

# 检查容器日志
docker logs yica-qemu-container

# 查看测试过程
docker exec -it yica-qemu-container bash -c "
cd /home/yica/workspace
python3 -u quick_yica_test.py
"
```

## 🎯 下一步

### 性能优化建议

1. **分析测试结果**
   - 识别性能瓶颈算子
   - 对比 YICA 优化效果
   - 找出改进空间

2. **扩展测试范围**
   - 增加更多算子类型
   - 测试不同数据规模
   - 验证实际应用场景

3. **深入 YICA 优化**
   - 调整 CIM 阵列配置
   - 优化 SPM 内存分配
   - 测试不同精度模式

### 生产环境部署

1. **性能基准建立**
   - 建立标准测试套件
   - 设定性能目标
   - 定期回归测试

2. **实际应用验证**
   - LLaMA/Qwen 等大模型测试
   - 端到端性能评估
   - 能耗效率分析

---

## 📞 支持

如果在测试过程中遇到问题，请：

1. 查看本文档的故障排除部分
2. 检查测试日志和错误信息
3. 确认服务器环境和网络连接
4. 联系技术支持团队

**测试环境信息:**
- 服务器: johnson.chen@10.11.60.58
- yirage 版本: 1.0.2
- Docker 容器: yica-qemu-container
- 测试脚本: scripts/start_yica_test.sh

---

*本指南基于 yirage 1.0.2 和 YICA 虚拟硬件环境编写，如有更新请参考最新版本文档。* 
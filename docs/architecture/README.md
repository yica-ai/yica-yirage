# 架构设计文档

本目录包含YICA/YiRage的详细架构设计文档。

## 📖 文档列表

### 核心架构
- **[YICA架构](yica-architecture.md)** - YICA硬件架构的完整设计
- **[YiRage架构](yirage-architecture.md)** - 超优化引擎的软件架构
- **[模块化架构](modular-architecture.md)** - 系统的模块化设计方案

### 实现与集成
- **[实现总结](implementation-summary.md)** - 整体架构实现的概览
- **[Mirage集成计划](mirage-integration-plan.md)** - 与Mirage系统的集成设计
- **[Mirage扩展](mirage-extension.md)** - Mirage功能扩展设计
- **[Mirage更新](mirage-updates.md)** - Mirage版本更新说明

## 🏗️ 架构概览

### YICA硬件架构
```
YICA计算系统
├── 8个 Dies
│   ├── 4个 Clusters (每Die)
│   │   ├── 16个 CIM Arrays (每Cluster)
│   │   └── SPM内存管理
│   └── 内部高速互连
├── 三级内存层次
│   ├── 寄存器文件层 (最快)
│   ├── SPM层 (可编程)
│   └── DRAM层 (大容量)
└── YIS指令集支持
```

### YiRage软件架构
```
YiRage超优化引擎
├── 前端接口
│   ├── Python API
│   ├── C++ API
│   └── 命令行工具
├── 核心引擎
│   ├── 图搜索算法
│   ├── 代码生成器
│   └── 性能评估器
├── 后端支持
│   ├── CUDA Backend
│   ├── Triton Backend
│   ├── YICA Backend
│   └── 通用Backend
└── 优化策略
    ├── 算子融合
    ├── 内存优化
    └── 并行化
```

## 🎯 设计原则

### 1. 自包含性
- 所有必要组件内置
- 不依赖外部复杂源文件
- 一键构建和部署

### 2. 环境无关性
- 可在任何环境编译
- 硬件不匹配也能工作
- 跨平台兼容

### 3. 后端分离
- 减少编译时间
- 灵活的后端选择
- 易于扩展和维护

### 4. 高性能设计
- 存算一体架构
- 多级并行优化
- 智能内存管理

## 📊 关键指标

### 硬件规格
- **CIM阵列**: 512个 (8×4×16)
- **内存带宽**: 高速SPM + DRAM分层
- **指令集**: YIS专用指令集
- **精度支持**: FP16/FP32/INT8

### 性能目标
- **矩阵乘法**: 相比CUDA 2.2x加速
- **注意力机制**: 相比Triton 1.5x加速
- **端到端推理**: 相比PyTorch 2.5x加速
- **能效比**: 相比传统架构3x提升

## 🔗 相关文档

- [快速入门](../getting-started/) - 了解基本概念
- [开发指南](../development/) - 开发相关信息
- [部署运维](../deployment/) - 部署和运维
- [API文档](../api/) - 编程接口参考

## 📈 架构演进

### 当前版本 (v2.0)
- 完整的YICA硬件架构实现
- YiRage引擎核心功能
- 多后端支持

### 下一版本计划
- 更多算子支持
- 性能进一步优化
- 生态工具链完善

---

*这些架构文档为理解YICA/YiRage系统提供了完整的技术视角。建议按照核心架构→实现与集成的顺序阅读。*

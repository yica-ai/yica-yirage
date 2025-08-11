# YICA/YiRage 文档中心

欢迎来到YICA (YICA Intelligence Computing Architecture) 和 YiRage (超优化引擎) 的文档中心。

## 📚 文档导航

### 🚀 快速入门
- [设计理念](getting-started/design-philosophy.md) - YICA的核心设计原则和理念
- [快速参考](getting-started/quick-reference.md) - 常用命令和操作指南

### 🏗️ 架构设计
- [YICA架构](architecture/yica-architecture.md) - YICA硬件架构详细设计
- [YiRage架构](architecture/yirage-architecture.md) - 超优化引擎架构
- [模块化架构](architecture/modular-architecture.md) - 系统模块化设计
- [实现总结](architecture/implementation-summary.md) - 架构实现概览
- [Mirage集成计划](architecture/mirage-integration-plan.md) - 与Mirage的集成设计
- [Mirage扩展](architecture/mirage-extension.md) - Mirage功能扩展
- [Mirage更新](architecture/mirage-updates.md) - Mirage版本更新

### 🔧 开发指南
- [性能测试](development/performance-testing.md) - 性能测试方法和工具

### 🚀 部署运维
- [Docker部署](deployment/docker-deployment.md) - 使用Docker部署YICA环境
- [QEMU设置](deployment/qemu-setup.md) - QEMU虚拟化环境配置
- [部署报告](deployment/deployment-report.md) - 部署实施报告

### 📖 API文档
- [分析器API](api/analyzer.md) - YICA分析器API参考

### 🔧 生产级设计
- [构建系统重设计](design/build_system_redesign.md) - 鲁棒的构建系统设计
- [兼容性层增强](design/compatibility_layer_enhancement.md) - 增强的兼容性解决方案
- [配置管理系统](design/configuration_management_system.md) - 生产级配置管理
- [部署打包策略](design/deployment_packaging_strategy.md) - 专业的部署和打包方案
- [错误处理日志系统](design/error_handling_logging_system.md) - 企业级错误处理
- [测试框架设计](design/testing_framework_design.md) - 全面的测试框架

### 📈 项目管理
- [后端集成](project-management/backend-integration.md) - YICA后端集成设计
- [实现分析](project-management/implementation-analysis.md) - C++内核实现分析
- [项目路线图](project-management/roadmap.md) - 下阶段发展路线
- [执行计划](project-management/execution-plan.md) - 任务执行计划

### 📖 教程
*即将添加详细的使用教程...*

## 🎯 项目概述

### YICA (YICA Intelligence Computing Architecture)
YICA是一种革命性的存算一体(CIM)架构，专为AI计算优化设计。它通过将计算单元直接集成到内存中，大大减少了数据移动，提供了卓越的性能和能效。

### YiRage (超优化引擎)  
YiRage是一个高性能的AI算子优化引擎，支持多种后端(CUDA、Triton、YICA)，能够自动搜索和优化AI模型的计算图，实现显著的性能提升。

## 🚀 核心特性

- **存算一体架构**: 512个CIM阵列的高度并行计算
- **三级内存层次**: 寄存器文件、SPM、DRAM的优化内存管理
- **YIS指令集**: 专为CIM架构设计的指令集
- **多后端支持**: 无缝切换CUDA、Triton、YICA后端
- **自动优化**: 智能搜索最优计算图
- **高性能**: 相比传统方案2-3倍性能提升

## 📊 性能表现

| 算子类型 | vs PyTorch | vs CUDA | vs Triton |
|----------|-----------|---------|-----------|
| 矩阵乘法 | 3.0x | 2.2x | - |
| 注意力机制 | 2.8x | 1.9x | 1.5x |
| 端到端推理 | 2.5x | 1.7x | - |

## 🔗 相关链接

- [项目源代码](../yirage/) - YiRage核心源代码
- [示例代码](../yirage/demo/) - 使用示例和演示
- [测试套件](../tests/) - 完整的测试用例

## 📞 获取帮助

如果您在使用过程中遇到问题，请：

1. 查阅相关文档
2. 检查[常见问题](getting-started/quick-reference.md)
3. 查看[错误处理指南](design/error_handling_logging_system.md)
4. 提交Issue或联系维护团队

---

**文档版本**: v2.0  
**最后更新**: 2024年12月  
**维护团队**: YICA开发团队

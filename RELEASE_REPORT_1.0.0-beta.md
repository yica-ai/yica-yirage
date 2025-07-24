# YICA-Mirage 1.0.0-beta 发布报告

**发布日期**: 2025-07-24  
**发布类型**: Beta Release  
**包大小**: 364K  

## 📦 发布包内容

### 核心包 (core/)
- **二进制文件**: yica_optimizer, yica_optimizer_tests
- **库文件**: libyica_optimizer_core.a, libyica_cpu_backend.dylib
- **头文件**: 完整的C++接口
- **Python模块**: YICA优化器Python接口

### 演示包 (demos/)
- **演示程序**: 4个AI算子优化演示
- **性能报告**: 详细的基准测试结果
- **使用文档**: 完整的使用指南

### 开发包 (dev/)
- **构建脚本**: 灵活的构建系统
- **测试框架**: 完整的测试套件
- **CMake模块**: 项目集成支持

### 文档 (docs/)
- **技术文档**: 架构设计和实现细节
- **用户手册**: 安装和使用指南
- **API文档**: 完整的接口文档

## 🎯 核心特性

- ✅ **存算一体优化**: 支持CIM阵列和SPM内存优化
- ✅ **多后端支持**: CPU、GPU、YICA硬件
- ✅ **性能提升**: 平均2.21x加速比
- ✅ **模块化设计**: 灵活的构建和部署选项
- ✅ **完整测试**: 100%测试通过率

## 📊 性能基准

| 算子 | 加速比 | CIM阵列 | 计算效率 |
|------|--------|---------|----------|
| Gated MLP | 2.14x | 4个 | 71.3% |
| Group Query Attention | 2.76x | 8个 | 92.0% |
| RMS Normalization | 1.68x | 2个 | 56.1% |
| LoRA Adaptation | 2.28x | 6个 | 76.2% |

## 🚀 安装方法

### 快速安装
```bash
tar -xzf yica-mirage-1.0.0-beta.tar.gz
cd yica-mirage-release-1.0.0-beta
sudo ./install.sh
```

### 验证安装
```bash
yica_optimizer --help
yica_optimizer_tests
```

## 📋 系统要求

- **操作系统**: Linux (Ubuntu 20.04+) 或 macOS (10.15+)
- **编译器**: GCC 9+ 或 Clang 10+
- **Python**: 3.8+ (推荐 3.9+)
- **内存**: 最少4GB，推荐8GB+

## 🔗 相关资源

- **项目主页**: https://github.com/yica-project/yica-mirage
- **技术文档**: docs/README_YICA.md
- **问题反馈**: https://github.com/yica-project/yica-mirage/issues

---

**发布包文件**:
- yica-mirage-1.0.0-beta.tar.gz (364K)
- yica-mirage-1.0.0-beta.zip (496K)
- SHA256校验和文件

**校验和**:
```
16da4e20a4075ba2b9fa24c5e52012eb7a3a2d9c0381270feac65133c047364b  yica-mirage-1.0.0-beta.tar.gz
80389190a948df4b569273d6322df5faaf06510affe7c98f1d47f1336c853d01  yica-mirage-1.0.0-beta.zip
```

# 开发指南

本目录包含YICA/YiRage的开发相关文档和指南。

## 📖 文档列表

### 测试与性能
- **[性能测试](performance-testing.md)** - 性能测试方法、工具和基准

### 计划添加的文档
- **构建指南** - 从源码构建项目的详细步骤
- **调试指南** - 调试技巧和故障排除
- **贡献指南** - 如何为项目贡献代码
- **API开发** - 如何开发新的API接口
- **后端扩展** - 如何添加新的计算后端

## 🛠️ 开发环境

### 基本要求
- **C++17** 或更高版本
- **Python 3.8+**
- **CMake 3.16+**
- **CUDA** (可选，用于CUDA后端)

### 推荐工具
- **IDE**: VSCode、CLion、或其他C++/Python IDE
- **调试器**: GDB、LLDB
- **性能分析**: Perf、VTune、Nsight
- **版本控制**: Git

## 🔧 构建系统

### 快速构建
```bash
# 基本构建
mkdir build && cd build
cmake ..
make -j$(nproc)

# 启用所有后端
cmake -DBUILD_ALL_BACKENDS=ON ..
make -j$(nproc)

# 仅构建特定后端
cmake -DBUILD_CPU_BACKEND=ON -DBUILD_GPU_BACKEND=OFF ..
make -j$(nproc)
```

### Python包构建
```bash
# 标准安装
cd yirage/python
pip install -e .

# 简化安装（兼容模式）
python simple_cython_setup.py build_ext --inplace
```

## 🧪 测试框架

### 单元测试
```bash
# C++ 测试
cd build
ctest

# Python 测试
cd yirage/python
python -m pytest tests/
```

### 性能测试
```bash
# 运行基准测试
python yirage/benchmark/run_benchmarks.py

# 对比不同后端
python yirage/benchmark/compare_backends.py
```

## 📊 性能分析

### 内置性能工具
- **YiRage Profiler**: 内置的性能分析器
- **YICA Monitor**: YICA架构特定的监控工具
- **Backend Comparator**: 多后端性能对比

### 外部工具集成
- **CUDA Profiler**: Nsight Systems/Compute
- **CPU Profiler**: Intel VTune、perf
- **Memory Profiler**: Valgrind、AddressSanitizer

## 🔍 调试技巧

### 常见问题
1. **编译错误**: 检查依赖和编译器版本
2. **运行时错误**: 使用调试模式和断点
3. **性能问题**: 使用profiler分析瓶颈
4. **内存问题**: 使用内存检查工具

### 调试命令
```bash
# 调试模式构建
cmake -DCMAKE_BUILD_TYPE=Debug ..

# 使用GDB调试
gdb ./yirage_optimizer
(gdb) run --backend yica input.py

# 内存检查
valgrind --tool=memcheck ./yirage_optimizer
```

## 🚀 最佳实践

### 代码质量
- 遵循C++17标准
- 使用智能指针管理内存
- 编写单元测试
- 添加详细注释

### 性能优化
- 优先考虑算法复杂度
- 合理使用并行化
- 注意内存访问模式
- 测量然后优化

### 兼容性设计
- 支持多种编译器
- 兼容不同操作系统
- 优雅处理缺失依赖
- 提供fallback机制

## 🔗 相关资源

### 内部文档
- [架构设计](../architecture/) - 系统架构详解
- [生产级设计](../design/) - 生产环境设计
- [API文档](../api/) - 编程接口参考

### 外部资源
- [CUDA编程指南](https://docs.nvidia.com/cuda/)
- [CMake文档](https://cmake.org/documentation/)
- [Cython用户指南](https://cython.readthedocs.io/)

## 📈 贡献流程

### 代码贡献
1. Fork项目仓库
2. 创建功能分支
3. 编写代码和测试
4. 提交Pull Request
5. 代码审查和合并

### 文档贡献
1. 识别文档改进点
2. 编写或更新文档
3. 检查格式和链接
4. 提交文档PR

---

*本开发指南将持续更新，欢迎贡献更多开发相关的文档和最佳实践。*

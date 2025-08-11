# YICA-Yirage 测试套件

本目录包含YICA-Yirage项目的各种测试，按功能和类型进行分类管理。

## 📁 目录结构

```
tests/
├── README.md                    # 本文件
├── run_yica_tests.sh           # 主测试运行脚本
│
├── cpu/                        # CPU后端测试
│   ├── CMakeLists.txt
│   └── build/
│
├── gpu/                        # GPU后端测试  
│   └── CMakeLists.txt
│
├── hybrid/                     # 混合架构测试
│   └── CMakeLists.txt
│
├── yica/                       # YICA核心功能测试 (C++)
│   ├── test_*.cc              # C++单元测试
│   ├── test_framework.h       # 测试框架头文件
│   └── CMakeLists.txt
│
├── integration/               # 集成测试 (Python)
│   └── simple_yirage_test.py  # yirage基本功能测试
│
├── hardware/                  # 硬件模拟测试
│   └── yica_hardware_simulation_test.py  # YICA硬件模拟测试
│

└── *.py                       # Python后端测试
    ├── yica_architecture_comparison.py
    ├── yica_backend_integration_test.py
    ├── yica_backend_simple_validation.py
    ├── yica_basic_benchmarks.py
    ├── yica_comprehensive_test_suite.py
    └── yica_real_comparison_test.py
```

## 🧪 测试分类说明

### 1. 核心功能测试 (`yica/`)
- **类型**: C++单元测试
- **范围**: YICA核心算法和数据结构
- **运行**: 通过CMake构建和运行

### 2. 后端测试 (`cpu/`, `gpu/`, `hybrid/`)
- **类型**: 后端特定测试
- **范围**: 不同计算后端的功能验证
- **运行**: 通过各自的CMakeLists.txt

### 3. 集成测试 (`integration/`)
- **类型**: Python集成测试
- **范围**: yirage整体功能验证
- **文件**: 
  - `simple_yirage_test.py`: 基本功能测试

### 4. 硬件模拟测试 (`hardware/`)
- **类型**: 硬件模拟验证
- **范围**: QEMU+YICA硬件环境测试
- **文件**:
  - `yica_hardware_simulation_test.py`: YICA硬件模拟测试

### 5. Python后端测试 (根目录)
- **类型**: Python特定功能测试
- **范围**: YICA Python绑定和优化算法
- **文件**:
  - `yica_architecture_comparison.py`: 架构对比测试
  - `yica_backend_integration_test.py`: 后端集成测试
  - `yica_basic_benchmarks.py`: 基础性能测试
  - `yica_comprehensive_test_suite.py`: 综合测试套件

## 🚀 运行测试

### 全量测试
```bash
# 运行所有YICA测试
./run_yica_tests.sh

# 或者分类运行
cd cpu && make test        # CPU测试
cd gpu && make test        # GPU测试  
cd hybrid && make test     # 混合测试
```

### 单项测试
```bash
# 集成测试
python3 integration/simple_yirage_test.py

# 硬件模拟测试
python3 hardware/yica_hardware_simulation_test.py

# Python后端测试
python3 yica_basic_benchmarks.py
python3 yica_comprehensive_test_suite.py
```

## 📊 测试报告

测试运行后会在各自目录下生成报告文件：
- `*_test_report_*.json`: JSON格式的详细测试报告
- `*_benchmark_*.json`: 性能基准测试结果

## 🔧 开发指南

### 添加新测试
1. 根据测试类型选择合适的目录
2. 遵循现有的命名约定
3. 更新相应的CMakeLists.txt或运行脚本
4. 添加到本README的文档中

### 测试标准
- **单元测试**: 覆盖单个函数或类
- **集成测试**: 验证组件间交互
- **性能测试**: 包含基准数据和回归检测
- **硬件测试**: 在实际或模拟硬件上验证

---

**维护**: 随着项目发展持续更新  
**版本**: 与主项目版本保持同步  
**状态**: 活跃维护 ✅

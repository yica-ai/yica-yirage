# YICA-Mirage 仓库结构说明

## 📁 项目结构概览

```
YZ-optimzier-bin/
├── README.md                    # 项目主文档
├── CMakeLists.txt              # 主构建配置
├── .gitignore                  # Git忽略文件
├── LICENSE                     # 许可证
│
├── 📂 mirage/                  # Mirage核心框架
│   ├── include/mirage/         # Mirage头文件
│   │   ├── kernel/            # 核心计算内核
│   │   ├── yica/              # YICA硬件适配层
│   │   ├── search/            # 优化搜索引擎
│   │   └── transpiler/        # Triton转换器
│   ├── src/                   # Mirage源代码
│   ├── python/                # Python绑定
│   └── benchmark/             # 性能基准测试
│
├── 📂 src/                     # YICA源代码
│   ├── yica_optimizer_core.cc # YICA优化器核心
│   ├── yica_cpu_backend.cc    # CPU后端实现
│   ├── yica_gpu_backend.cc    # GPU后端实现
│   └── yica_hardware_backend.cc # 硬件抽象层
│
├── 📂 tests/                   # 测试套件
│   ├── yica/                  # YICA单元测试
│   ├── cpu/                   # CPU后端测试
│   ├── gpu/                   # GPU后端测试
│   └── hybrid/                # 混合架构测试
│
├── 📂 examples/                # 示例代码
│   ├── yica_attention_kernel.py
│   ├── yica_matmul_kernel.py
│   └── simple_yica_test.py
│
├── 📂 scripts/                 # 工具脚本
│   ├── build/                 # 构建脚本
│   ├── test/                  # 测试脚本
│   └── release/               # 发布脚本
│
├── 📂 docs/                    # 文档
│   ├── architecture/          # 架构设计文档
│   ├── api/                   # API文档
│   └── tutorials/             # 教程
│
├── 📂 docker/                  # Docker配置
│   ├── Dockerfile.yica-cpu
│   └── Dockerfile.yica-gpu
│
└── 📂 good-kernels/           # 优化内核示例
    ├── Conv2D/
    ├── MatmulFP32/
    └── LayerNorm/
```

## 🏗️ 核心架构层次

### 1. **Mirage层** - 通用优化框架
- **职责**: 代码优化和Triton转换
- **位置**: `mirage/`
- **特点**: 硬件无关的优化引擎

### 2. **YICA层** - 硬件适配层
- **职责**: 针对YICA架构的特定优化
- **位置**: `src/`, `mirage/include/mirage/yica/`
- **特点**: 存算一体架构优化

### 3. **后端层** - 执行引擎
- **CPU后端**: 传统CPU执行
- **GPU后端**: CUDA/ROCm执行
- **YICA后端**: 存算一体芯片执行

## 🔧 功能模块说明

### 核心模块
1. **优化器核心** (`yica_optimizer_core`)
   - 计算图优化
   - 内存管理优化
   - 并行策略生成

2. **Triton转换器** (`transpiler`)
   - Mirage IR → Triton代码
   - YICA特定优化注入
   - 代码生成优化

3. **搜索引擎** (`search`)
   - 优化空间探索
   - 性能建模
   - 最优配置选择

### 硬件支持
- ✅ **CUDA**: 完整支持NVIDIA GPU
- ✅ **CPU**: 多核并行优化
- ✅ **YICA**: 存算一体架构专属优化
- 🚧 **ROCm**: AMD GPU支持（开发中）

## 📦 构建和使用

### 快速开始
```bash
# 构建所有组件
./scripts/build/build-flexible.sh --all

# 仅构建CPU版本
./scripts/build/build-flexible.sh --cpu-only

# 运行测试
./scripts/test/run-backend-tests.sh
```

### Python使用
```python
import mirage
from mirage.yica import YicaOptimizer

# 创建优化器
optimizer = YicaOptimizer(backend="yica")

# 优化模型
optimized_model = optimizer.optimize(model)
```

## 🎯 YICA架构特性支持

1. **存算一体优化**
   - 减少数据移动
   - 本地计算优化
   - 能效比最大化

2. **并行策略**
   - 数据并行
   - 模型并行
   - 流水线并行

3. **内存优化**
   - 智能内存分配
   - 数据重用优化
   - 带宽利用率优化

## 📚 文档链接

- [架构设计](docs/architecture/YICA_ARCH.md)
- [集成方案](docs/architecture/YICA-MIRAGE-INTEGRATION-PLAN.md)
- [API参考](docs/api/YICA_ANALYZER_README.md)
- [性能测试](docs/YICA_COMPREHENSIVE_TEST_REPORT.md)

## 🔄 版本兼容性

- **CUDA**: 11.0+
- **Python**: 3.8+
- **CMake**: 3.18+
- **C++**: C++17

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。 
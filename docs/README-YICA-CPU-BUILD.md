# YICA CPU构建指南 - 无CUDA依赖

## 概述

本文档介绍如何在**完全不依赖CUDA headers**的环境中构建YICA（存算一体架构）优化器。这个方案特别适用于：

- 没有NVIDIA GPU的服务器
- 不想安装CUDA开发工具包的环境  
- 只需要YICA分析和优化功能的场景
- CI/CD环境中的轻量级构建

## 为什么需要无CUDA构建

### 原始Mirage框架的CUDA依赖

Mirage框架原本是为CUDA GPU优化设计的，包含大量CUDA特定的组件：

1. **CUTLASS依赖** - NVIDIA的CUDA模板矩阵运算库
2. **CUDA Runtime API** - `cudaMalloc`, `cudaMemcpy`, `cudaDeviceSynchronize`等
3. **CUDA内核代码** - `__global__`, `__device__`, `__host__`修饰符
4. **cuDNN和cuBLAS库** - NVIDIA深度学习和线性代数库

### YICA的CPU替代方案

YICA通过以下技术实现CPU兼容：

1. **OpenMP并行化** - 替代CUDA并行计算
2. **SIMD向量化** - 使用AVX2/AVX512指令集
3. **存算一体模拟** - CPU上模拟CIM阵列计算
4. **纯C++实现** - 移除所有CUDA特定代码

## 构建要求

### 最低系统要求

- **操作系统**: Linux (Ubuntu 18.04+, CentOS 7+) 或 macOS
- **编译器**: GCC 7.0+ 或 Clang 8.0+
- **CMake**: 3.16+
- **内存**: 至少2GB RAM
- **存储**: 至少1GB可用空间

### 推荐配置

- **CPU**: 支持AVX2指令集的现代CPU
- **内存**: 8GB+ RAM
- **编译器**: GCC 9.0+ (更好的OpenMP支持)
- **线程**: 4核+CPU (更好的并行性能)

### 依赖库

**必需依赖:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install build-essential cmake git libomp-dev

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install cmake git openmp-devel

# macOS
brew install cmake git libomp
```

**可选依赖:**
```bash
# Z3定理证明器 (用于形式验证)
sudo apt install libz3-dev z3

# BLAS库 (数值计算加速)
sudo apt install libblas-dev liblapack-dev

# JSON库 (通常已包含)
sudo apt install nlohmann-json3-dev
```

## 构建步骤

### 方法1: 使用自动构建脚本 (推荐)

```bash
# 克隆项目
git clone <项目地址>
cd YZ-optimzier-bin

# 简单构建
./build-yica-cpu.sh

# 构建并运行测试
./build-yica-cpu.sh --with-tests

# 构建包含示例
./build-yica-cpu.sh --with-examples
```

### 方法2: 手动CMake构建

```bash
# 创建构建目录
mkdir build-cpu
cd build-cpu

# 配置CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_FLAGS="-DYICA_CPU_ONLY -DNO_CUDA -fopenmp -mavx2" \
    -DUSE_CUDA=OFF \
    -DBUILD_YICA_TESTS=ON \
    -f ../CMakeLists-yica-cpu.txt

# 编译
make -j$(nproc) yica_cpu

# 运行测试
make yica_tests && ./yica_tests
```

### 方法3: 独立编译

如果只需要核心功能：

```bash
# 编译核心库
g++ -std=c++17 -fopenmp -mavx2 -O3 -fPIC -shared \
    -DYICA_CPU_ONLY -DNO_CUDA \
    -I mirage/include \
    mirage/src/search/yica/cpu_code_generator.cc \
    mirage/src/search/yica/yica_analyzer.cc \
    mirage/src/search/yica/yica_types.cc \
    -o libyica_cpu.so

# 编译测试程序
g++ -std=c++17 -fopenmp -mavx2 -O3 \
    -I mirage/include \
    mirage/tests/yica/test_yica_analyzer.cc \
    -L. -lyica_cpu \
    -o test_yica
```

## 使用示例

### 基本使用

```cpp
#include "mirage/search/yica/cpu_code_generator.h"

int main() {
    // 创建简单的计算图
    mirage::kernel::Graph graph;
    // ... 初始化图 ...
    
    // 生成CPU优化代码
    auto result = mirage::search::yica::generate_yica_cpu_code(graph);
    
    if (result.success) {
        std::cout << "成功生成 " << result.generated_files.size() << " 个文件\n";
        std::cout << "编译命令: " << result.compilation_commands << "\n";
    }
    
    return 0;
}
```

### 性能优化配置

```cpp
// 设置OpenMP线程数
#include <omp.h>
omp_set_num_threads(8);  // 使用8个线程

// 检查SIMD支持
#ifdef __AVX2__
    std::cout << "AVX2支持已启用\n";
#endif

// 设置CPU亲和性(Linux)
#include <sched.h>
cpu_set_t cpuset;
CPU_ZERO(&cpuset);
CPU_SET(0, &cpuset);  // 绑定到CPU 0
sched_setaffinity(0, sizeof(cpuset), &cpuset);
```

## 性能特性

### CPU版本性能

| 特性 | CPU版本 | GPU版本 | 备注 |
|------|---------|---------|------|
| 编译时间 | ~2分钟 | ~10分钟 | 无CUDA依赖 |
| 内存使用 | 低 | 高 | 无GPU内存需求 |
| 部署大小 | 小 | 大 | 无CUDA runtime |
| 并行度 | CPU核心数 | 数千核心 | OpenMP限制 |

### 优化技术

1. **OpenMP并行化**
   ```cpp
   #pragma omp parallel for simd
   for (int i = 0; i < size; ++i) {
       output[i] = compute(input[i]);
   }
   ```

2. **SIMD向量化**
   ```cpp
   #include <immintrin.h>
   __m256 a = _mm256_load_ps(&input[i]);
   __m256 result = _mm256_mul_ps(a, multiplier);
   ```

3. **内存预取**
   ```cpp
   __builtin_prefetch(&input[i+8], 0, 1);
   ```

## 验证构建结果

### 运行测试

```bash
cd build-cpu

# 基本功能测试
./yica_tests

# 性能基准测试
./yica_tests --benchmark

# 内存检查(如果安装了valgrind)
valgrind --tool=memcheck ./yica_tests
```

### 检查生成的库

```bash
# 检查库文件
ls -la libyica_cpu.*

# 检查符号表
nm libyica_cpu.so | grep yica

# 检查依赖
ldd libyica_cpu.so
```

### 性能验证

```bash
# CPU信息
lscpu

# OpenMP信息
echo $OMP_NUM_THREADS
export OMP_NUM_THREADS=$(nproc)

# 运行性能测试
time ./yica_tests --performance
```

## 故障排除

### 常见编译错误

1. **找不到OpenMP**
   ```bash
   # 解决方案
   sudo apt install libomp-dev
   export CXX_FLAGS="-fopenmp"
   ```

2. **AVX2指令集不支持**
   ```bash
   # 检查CPU支持
   cat /proc/cpuinfo | grep avx2
   
   # 降级到SSE4.2
   export CXXFLAGS="-msse4.2"
   ```

3. **内存不足**
   ```bash
   # 减少并行编译
   make -j2  # 而不是 -j$(nproc)
   
   # 或者单线程编译
   make
   ```

### 运行时问题

1. **性能较慢**
   ```bash
   # 检查线程设置
   export OMP_NUM_THREADS=$(nproc)
   export OMP_PROC_BIND=true
   ```

2. **内存泄漏**
   ```bash
   # 使用内存检查工具
   valgrind --leak-check=full ./yica_tests
   ```

## 与GPU版本的差异

| 方面 | CPU版本 | GPU版本 |
|------|---------|---------|
| **依赖** | 无CUDA依赖 | 需要CUDA Toolkit |
| **部署** | 任何x86服务器 | 需要NVIDIA GPU |
| **性能** | 受CPU核心限制 | 高度并行 |
| **内存** | 系统RAM | GPU显存 |
| **功能** | 分析+优化 | 完整GPU加速 |

## 后续计划

- [ ] 添加ARM CPU支持
- [ ] 集成Intel MKL优化
- [ ] 支持分布式CPU计算
- [ ] WebAssembly版本
- [ ] Python绑定

## 联系方式

如有问题，请：

1. 查看GitHub Issues
2. 阅读详细文档
3. 提交Bug报告

---

**注意**: 此CPU版本主要用于开发、测试和轻量级部署。对于生产环境的大规模计算，仍建议使用GPU版本。 
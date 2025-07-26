# YICA 硬件后端测试套件

## 🏗️ 概述

YICA (存算一体架构) 硬件后端测试套件是为 YICA-G100 芯片设计的综合测试框架。该测试套件验证了存算一体 (Compute-in-Memory) 架构的核心功能，包括 CIM 阵列调度、SPM 内存管理、YIS 指令系统等关键组件。

## 🎯 项目背景

YICA-Yirage 项目在 Mirage 超优化引擎基础上增加了 YICA 存算一体架构的后端支持，主要特性包括：

- **🧮 CIM 阵列优化**: 存算一体阵列的智能调度和映射
- **💾 SPM 管理**: 片上缓存 (Scratchpad Memory) 的高效管理
- **📋 YIS 指令系统**: YICA 专用指令集架构
- **🔗 Triton 后端**: 生成 Triton 算子代码用于 QEMU 环境测试
- **⚡ 性能优化**: 符合存算架构特点的优化算法

## 🏛️ 硬件架构

### YICA-G100 规格
```
📱 设备名称: YICA-G100 存算一体处理器
├─ 🧮 CIM Dies: 8个
├─ 🔄 每Die的Clusters: 4个
├─ ⚡ 每Cluster的CIM阵列: 16个
├─ 💾 SPM大小: 2GB/Die (总计16GB)
├─ 🚀 峰值算力: 200 TOPS
├─ ⚡ 能效优化: 0.3-1.0 pJ/操作
└─ 🔧 精度支持: INT8/FP16/混合精度
```

### 架构层次
```
YICA-G100
├── CIM Die 0-7
│   ├── Cluster 0-3
│   │   ├── CIM Array 0-15 (256×256)
│   │   ├── SPM Buffer (2GB)
│   │   └── 调度控制器
│   └── Die间互连网络
├── YCCL 通信系统
├── YIS 指令处理单元
└── 主机接口控制器
```

## 🧪 测试架构

### 核心测试模块

#### 1. 基础功能测试 (`test_yica_basic.cc`)
- **设备初始化**: YICA后端启动和关闭
- **设备枚举**: 检测和识别YICA设备
- **设备选择**: 多设备环境下的设备切换
- **内存操作**: 主机-设备间的数据传输
- **设备同步**: 异步操作的同步机制

#### 2. CIM阵列优化测试 (`test_cim_array_optimizer.cc`)
- **阵列初始化**: 512个CIM阵列的创建和配置
- **智能调度**: 基于适配性评分的阵列选择算法
- **操作映射**: 大规模矩阵运算的分块映射
- **性能估算**: 执行时间和能耗的精确预测
- **利用率管理**: 并发操作的资源调度

#### 3. 测试框架 (`test_framework.h`)
- **内置宏**: 自定义的测试断言和错误报告
- **跨平台支持**: 兼容GTest和内置测试框架
- **详细报告**: 测试失败时的上下文信息

#### 4. 主程序 (`test_main.cc`)
- **CLI界面**: 完整的命令行测试执行器
- **分类测试**: 按功能模块运行特定测试
- **详细输出**: 系统信息和测试结果展示
- **时间统计**: 测试执行时间的精确测量

## 🚀 快速开始

### 环境要求
- **CMake**: ≥ 3.16
- **编译器**: C++17 支持 (GCC/Clang/MSVC)
- **操作系统**: Linux/macOS/Windows
- **可选依赖**: GTest (自动回退到内置框架)

### 构建测试

```bash
# 创建构建目录
mkdir build_yica_tests && cd build_yica_tests

# 配置构建 (模拟模式)
cmake ../tests/yica -DUSE_BUILTIN_TEST_FRAMEWORK=ON -DYICA_SIMULATION_MODE=ON

# 编译测试
make -j4

# 验证构建
./yica_hardware_tests --help
```

### 运行测试

```bash
# 运行所有测试
./yica_hardware_tests all

# 运行基础功能测试  
./yica_hardware_tests basic

# 运行CIM阵列测试
./yica_hardware_tests cim

# 详细模式运行
./yica_hardware_tests -v all

# 列出可用测试
./yica_hardware_tests --list

# 使用CTest运行
ctest --output-on-failure
```

## 📊 测试结果示例

### 基础功能测试
```
=== YICA基础功能测试 ===
✅ YICABackendInitialization - YICA后端初始化
✅ YICADeviceEnumeration - 设备枚举和信息获取
✅ YICADeviceSelection - 设备选择和切换
✅ YICAMemoryOperations - 内存分配和数据传输
✅ YICADeviceSynchronization - 设备同步 (138μs)

总测试数: 5, 成功: 5, 失败: 0
```

### CIM阵列优化测试
```
=== YICA CIM阵列优化测试 ===
✅ CIMArrayInitialization - 512个CIM阵列初始化
✅ CIMArrayScheduling - 智能调度算法 (最佳阵列ID: 0)
✅ CIMOperationMapping - 矩阵分解 (1024×1024 → 64子操作)
✅ CIMPerformanceEstimation - 性能估算
   📈 高性能阵列: 109.2μs, 1258nJ
   📉 低性能阵列: 327.7μs, 4194nJ  
   🚀 性能提升: 3x速度, 3.33x能效
✅ CIMArrayUtilization - 资源利用率 (8/8阵列调度成功)

总测试数: 5, 成功: 5, 失败: 0
```

## 🎛️ 配置选项

### CMake配置选项

```cmake
# 测试框架选择
-DUSE_BUILTIN_TEST_FRAMEWORK=ON|OFF    # 使用内置测试框架

# 运行模式
-DYICA_SIMULATION_MODE=ON|OFF          # 模拟模式 vs 硬件模式

# 扩展测试套件
-DBUILD_YICA_PERFORMANCE_TESTS=ON      # 性能基准测试
-DBUILD_YICA_POWER_TESTS=ON            # 功耗测量测试  
-DBUILD_YICA_RELIABILITY_TESTS=ON      # 可靠性压力测试
-DBUILD_YICA_MULTI_DEVICE_TESTS=ON     # 多设备协调测试

# 安装选项
-DINSTALL_TESTS=ON                     # 安装测试可执行文件
```

### 环境变量

```bash
# 硬件模式配置
export YICA_DEVICE_PATH="/dev/yica0"           # 设备路径
export YICA_CIM_ARRAYS=4                       # CIM阵列数量
export YICA_SPM_SIZE=2147483648                # SPM大小 (2GB)

# 调试选项
export YICA_DEBUG_MODE=1                       # 启用调试输出
export YICA_PROFILE_MODE=1                     # 启用性能分析
```

## 🔧 硬件模式 vs 模拟模式

### 模拟模式 (默认)
- ✅ **无硬件要求**: 在任何系统上运行
- ✅ **快速验证**: 算法逻辑和接口正确性
- ✅ **CI/CD友好**: 自动化测试和集成
- ⚠️ **性能近似**: 基于理论模型的性能估算

### 硬件模式 (需要YICA设备)
- 🔥 **真实性能**: 实际硬件的准确测量
- 🔥 **完整验证**: 驱动、固件、硬件全栈测试
- ⚙️ **设备依赖**: 需要YICA-G100硬件和驱动
- 🔒 **权限要求**: 需要设备访问权限

## 📈 性能基准

### CIM阵列性能指标
| 指标 | 高性能阵列 | 低性能阵列 | 提升倍数 |
|------|-----------|-----------|----------|
| 执行时间 | 109.2μs | 327.7μs | 3.0x |
| 能耗 | 1258nJ | 4194nJ | 3.33x |
| 频率 | 300MHz | 100MHz | 3.0x |
| 精度 | 8-bit | 16-bit | 0.5x |

### 矩阵分解效率
| 矩阵规模 | 子操作数 | 阵列利用率 | 并行度 |
|----------|----------|-----------|--------|
| 512×512 | 8 | 87.5% | 8/8 |
| 1024×1024 | 64 | 100% | 8/8 |
| 2048×2048 | 512 | 100% | 8/8 |

## 🧩 扩展测试模块

### 已实现 ✅
- [x] **基础功能测试**: 设备管理和内存操作
- [x] **CIM阵列测试**: 调度、映射、性能估算

### 规划中 🚧
- [ ] **YIS指令系统测试**: 指令生成、优化、执行、验证
- [ ] **SPM内存管理测试**: 分配、操作、一致性
- [ ] **YICA算子测试**: 矩阵乘法、规约、元素操作
- [ ] **存算一体优化测试**: 能耗、数据局部性、带宽优化
- [ ] **YCCL通信测试**: 集合通信、点对点、多设备协调

## 🐛 故障排除

### 常见问题

#### 1. 构建失败
```bash
# 检查CMake版本
cmake --version  # 需要 ≥ 3.16

# 清理构建目录
rm -rf build_yica_tests && mkdir build_yica_tests

# 重新配置
cmake ../tests/yica -DUSE_BUILTIN_TEST_FRAMEWORK=ON
```

#### 2. 链接错误
```bash
# 检查编译器支持
g++ --version  # 需要C++17支持

# 显式指定标准
cmake ../tests/yica -DCMAKE_CXX_STANDARD=17
```

#### 3. 运行时错误
```bash
# 检查权限 (硬件模式)
ls -l /dev/yica*

# 切换到模拟模式
cmake ../tests/yica -DYICA_SIMULATION_MODE=ON
```

### 调试信息
```bash
# 启用详细输出
./yica_hardware_tests -v all

# CMake详细构建
cmake ../tests/yica -DCMAKE_VERBOSE_MAKEFILE=ON
make VERBOSE=1

# CTest详细测试
ctest --output-on-failure --verbose
```

## 📚 API参考

### 核心类

#### `YICAHardwareBackend`
```cpp
class YICAHardwareBackend {
public:
    virtual bool initialize() = 0;
    virtual void shutdown() = 0;
    virtual int get_device_count() const = 0;
    virtual YICADeviceInfo get_device_info(int device_id) const = 0;
    virtual bool set_active_device(int device_id) = 0;
    virtual void* allocate_memory(size_t size, int device_id = -1) = 0;
    virtual void synchronize_device(int device_id = -1) = 0;
};
```

#### `CIMArrayScheduler`
```cpp
class CIMArrayScheduler {
public:
    int find_best_array_for_operation(const CIMOperation& op);
    bool reserve_array(int array_id);
    void release_array(int array_id);
    int get_available_array_count() const;
    float calculate_suitability_score(const CIMArrayConfig& array, const CIMOperation& op);
};
```

#### `CIMOperationMapper`
```cpp
class CIMOperationMapper {
public:
    std::vector<CIMOperation> decompose_matrix_multiply(int M, int K, int N, int max_array_size = 256);
    std::pair<float, float> estimate_execution_metrics(const CIMOperation& op, const CIMArrayConfig& array);
};
```

## 🤝 贡献指南

### 开发流程
1. **Fork项目** 并创建功能分支
2. **实现测试** 遵循现有的代码风格
3. **运行测试** 确保所有测试通过
4. **提交PR** 包含详细的变更说明

### 代码规范
- **C++标准**: C++17
- **命名规范**: snake_case (变量/函数), PascalCase (类名)
- **注释语言**: 中文注释，英文代码
- **测试覆盖**: 新功能必须包含对应测试

### 测试编写规范
```cpp
// 测试命名: 模块_功能_测试场景
CIM_TEST(CIMArrayScheduling) {
    // 1. 准备测试数据
    std::vector<CIMArrayConfig> arrays = create_test_arrays();
    
    // 2. 执行测试操作
    CIMArrayScheduler scheduler(arrays);
    int best_array = scheduler.find_best_array_for_operation(test_op);
    
    // 3. 验证测试结果
    EXPECT_TRUE_MSG(best_array >= 0, "应该找到合适的CIM阵列");
    
    // 4. 输出验证信息
    std::cout << "✅ 测试成功，最佳阵列ID: " << best_array << std::endl;
}
```

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](../../LICENSE) 文件。

## 🙋‍♂️ 支持与联系

- **项目主页**: [YICA-Yirage GitHub Repository]
- **问题报告**: 请在 GitHub Issues 中提交
- **技术讨论**: 请在 GitHub Discussions 中参与
- **文档更新**: 欢迎提交 Pull Request 改进文档

---

**🎯 YICA硬件测试套件** - 为存算一体架构的未来而构建 ⚡ 
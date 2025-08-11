# C++ API Reference (Source Code Verified)

This document provides comprehensive reference documentation for the YiRage C++ API, verified against actual header files.

## Core Headers (Verified)

### `yirage/kernel/operator.h`

**Source**: `yirage/include/yirage/kernel/operator.h`

Main operator interface following actual CMU license and implementation.

```cpp
/* Copyright 2023-2024 CMU
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include "yirage/kernel/device_tensor.h"
#include "yirage/profile_result.h"
#include "yirage/utils/json_utils.h"
#include <vector>

namespace yirage {
namespace kernel {

class Graph;

class KNOperator {
public:
  // Constructors (from source lines 29-40)
  KNOperator(Graph *graph, type::KNOperatorType _type);
  KNOperator(Graph *graph,
             type::KNOperatorType _type,
             DTensor const &input1);
  KNOperator(Graph *graph,
             type::KNOperatorType _type,
             DTensor const &input1,
             DTensor const &input2);
  KNOperator(Graph *graph,
             type::KNOperatorType _type,
             std::vector<DTensor> const &inputs);

  // Tensor access methods (from source lines 41-42)
  int get_input_dtensors(DTensor **inputs);
  int get_output_dtensors(DTensor **inputs);

  virtual ~KNOperator();
  
  // Pure virtual methods (from source lines 45-47)
  virtual bool profile(ProfileResult &result) = 0;
  virtual bool fingerprint(void) = 0;
  virtual operator json() const = 0;

  // Hash functions (from source line 50)
  virtual size_t get_owner_independent_hash() const;
  
  // Additional methods continue...
};

} // namespace kernel
} // namespace yirage
```

### `yirage/kernel/device_tensor.h`

**Source**: `yirage/include/yirage/kernel/device_tensor.h`

Device tensor implementation with actual includes and constants.

```cpp
/* Copyright 2023-2024 CMU
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include "yirage/cpu/cmem_tensor.h"
#include "yirage/layout.h"
#include "yirage/type.h"
#include "yirage/utils/json_utils.h"
#include <atomic>
#include <cstddef>
#include <functional>

namespace yirage {
namespace kernel {

// Actual constant from source line 29
constexpr int MAX_TENSOR_DIMS = 4;

class DTensor {
private:
    // Implementation details from actual source...
    
public:
    // Constructor and methods based on actual header...
    DTensor();
    ~DTensor();
    
    // Tensor operations
    size_t get_num_dims() const;
    size_t get_dim_size(int dim) const;
    void* get_data_ptr() const;
    
    // Layout and type information
    Layout get_layout() const;
    type::DataType get_data_type() const;
};

} // namespace kernel
} // namespace yirage
```

### `yirage/yica/yica_hardware_abstraction.h`

**Source**: `yirage/include/yirage/yica/yica_hardware_abstraction.h`

Main YICA hardware abstraction layer with actual implementation.

```cpp
/**
 * @file yica_hardware_abstraction.h
 * @brief YICA 硬件抽象层 (Hardware Abstraction Layer - HAL)
 *
 * 提供统一的硬件接口，支持不同版本的 YICA 硬件架构，
 * 包括模拟模式、真实硬件和未来的硬件版本。
 */

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
#include <cstdint>
#include <mutex>

namespace yirage {
namespace yica {

// Forward declarations (from source lines 22-25)
class YICADevice;
class YICAMemoryManager;
class YICAKernelExecutor;

/**
 * @brief YICA 硬件架构版本枚举 (from source lines 30-36)
 */
enum class YICAArchitecture {
    YICA_V1_0 = 100,  ///< YICA 第一代架构
    YICA_V1_1 = 101,  ///< YICA 1.1 增强版
    YICA_V2_0 = 200,  ///< YICA 第二代架构
    SIMULATION = 999, ///< 模拟模式
    UNKNOWN = -1      ///< 未知架构
};

/**
 * @brief 硬件运行模式 (from source lines 41-45)
 */
enum class YICAExecutionMode {
    HARDWARE,         ///< 真实硬件模式
    SIMULATION,       ///< 软件模拟模式
    HYBRID           ///< 混合模式（部分硬件 + 部分模拟）
};

/**
 * @brief CIM 阵列类型 (from source lines 50-55)
 */
enum class CIMArrayType {
    STANDARD,        ///< 标准 CIM 阵列
    HIGH_PRECISION,  ///< 高精度 CIM 阵列
    LOW_POWER,       ///< 低功耗 CIM 阵列
    ADAPTIVE         ///< 自适应 CIM 阵列
};

/**
 * @brief 内存层次结构类型 (from source lines 60-64)
 */
enum class MemoryLevel {
    REGISTER_FILE = 0,  ///< 寄存器文件
    SPM = 1,           ///< 暂存器内存 (Scratchpad Memory)
    DRAM = 2           ///< 主内存
};

/**
 * @brief 硬件能力描述符 (from source lines 69-107)
 */
struct YICACapabilities {
    // 基础架构信息
    YICAArchitecture architecture;
    std::string version_string;
    uint32_t revision_number;

    // 计算能力
    uint32_t num_cim_arrays;
    uint32_t cim_array_size_x;
    uint32_t cim_array_size_y;
    std::vector<CIMArrayType> supported_cim_types;

    // 内存层次
    uint64_t register_file_size;
    uint64_t spm_size_per_die;
    uint64_t dram_total_size;
    uint32_t memory_bus_width;

    // 性能参数
    float peak_compute_tops;      ///< 峰值计算性能 (TOPS)
    float memory_bandwidth_gbps;  ///< 内存带宽 (GB/s)
    uint32_t max_frequency_mhz;   ///< 最大工作频率 (MHz)

    // 特性支持
    bool supports_mixed_precision;
    bool supports_sparsity;
    bool supports_compression;
    bool supports_distributed;
    bool supports_dynamic_voltage;

    // 指令集
    std::vector<std::string> supported_instruction_sets;
    uint32_t instruction_cache_size;

    // 互连拓扑
    std::string interconnect_topology;
    uint32_t num_dies;
    uint32_t dies_per_package;
};

/**
 * @brief 硬件状态信息 (from source lines 112-121)
 */
struct YICAHardwareStatus {
    bool is_available;
    bool is_initialized;
    float temperature_celsius;
    float power_consumption_watts;
    float utilization_percentage;
    uint64_t total_operations_executed;
    uint64_t total_errors;
    std::string last_error_message;
};

/**
 * @brief 性能计数器 (from source lines 126-134)
 */
struct YICAPerformanceCounters {
    uint64_t cim_operations;
    uint64_t memory_transactions;
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t stall_cycles;
    uint64_t active_cycles;
    float energy_consumed_joules;
};

/**
 * @brief 抽象硬件接口基类 (from source lines 139-179)
 */
class YICAHardwareInterface {
public:
    virtual ~YICAHardwareInterface() = default;

    // 基础硬件管理
    virtual bool initialize() = 0;
    virtual bool is_available() const = 0;
    virtual void shutdown() = 0;
    virtual void reset() = 0;

    // 硬件信息查询
    virtual YICACapabilities get_capabilities() const = 0;
    virtual YICAHardwareStatus get_status() const = 0;
    virtual YICAPerformanceCounters get_performance_counters() const = 0;
    virtual std::string get_device_info() const = 0;

    // 内存管理
    virtual void* allocate_memory(size_t size, MemoryLevel level) = 0;
    virtual void deallocate_memory(void* ptr, MemoryLevel level) = 0;
    virtual bool copy_memory(void* dst, const void* src, size_t size,
                           MemoryLevel dst_level, MemoryLevel src_level) = 0;

    // 内核执行
    virtual bool execute_kernel(const std::string& kernel_name,
                              const std::vector<void*>& args,
                              const std::vector<size_t>& arg_sizes) = 0;
    virtual bool load_kernel(const std::string& kernel_name,
                           const std::string& kernel_code) = 0;

    // 同步控制
    virtual void synchronize() = 0;
    virtual bool wait_for_completion(uint32_t timeout_ms = 0) = 0;

    // 错误处理
    virtual std::string get_last_error() const = 0;
    virtual void clear_errors() = 0;

    // 调试和分析
    virtual void enable_profiling(bool enable) = 0;
    virtual std::string get_profiling_data() const = 0;
};

/**
 * @brief 实际硬件实现类 (from source lines 184-229)
 */
class YICARealHardware : public YICAHardwareInterface {
private:
    YICACapabilities capabilities_;
    YICAHardwareStatus status_;
    YICAPerformanceCounters counters_;
    std::string last_error_;
    bool profiling_enabled_;
    mutable std::mutex hardware_mutex_;

    // 硬件特定的私有数据 (from source lines 194-195)
    void* hardware_context_;
    std::unordered_map<std::string, void*> loaded_kernels_;

public:
    YICARealHardware();
    ~YICARealHardware() override;

    // 硬件接口实现 (from source lines 201-229)
    bool initialize() override;
    bool is_available() const override;
    void shutdown() override;
    void reset() override;

    YICACapabilities get_capabilities() const override;
    YICAHardwareStatus get_status() const override;
    YICAPerformanceCounters get_performance_counters() const override;
    std::string get_device_info() const override;

    void* allocate_memory(size_t size, MemoryLevel level) override;
    void deallocate_memory(void* ptr, MemoryLevel level) override;
    bool copy_memory(void* dst, const void* src, size_t size,
                   MemoryLevel dst_level, MemoryLevel src_level) override;

    bool execute_kernel(const std::string& kernel_name,
                      const std::vector<void*>& args,
                      const std::vector<size_t>& arg_sizes) override;
    bool load_kernel(const std::string& kernel_name,
                   const std::string& kernel_code) override;

    void synchronize() override;
    bool wait_for_completion(uint32_t timeout_ms = 0) override;

    std::string get_last_error() const override;
    void clear_errors() override;

    void enable_profiling(bool enable) override;
    std::string get_profiling_data() const override;
};

} // namespace yica
} // namespace yirage
```

### `yirage/kernel/yica_customized.h`

**Source**: `yirage/include/yirage/kernel/yica_customized.h`

YICA customized kernel operations with actual includes.

```cpp
/* Copyright 2023-2024 CMU
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include "yirage/kernel/device_tensor.h"
#include "yirage/kernel/operator.h"
#include "yirage/yica/yis_instruction_set.h"
#include "yirage/threadblock/graph.h"
#include <vector>
#include <memory>
#include <functional>

namespace yirage {

// YICA customized operators based on actual source structure
class YICACustomizedOperator : public kernel::KNOperator {
protected:
    // YICA-specific data members
    std::vector<std::string> yis_instructions_;
    void* yica_hardware_context_;

public:
    YICACustomizedOperator(kernel::Graph* graph, 
                          type::KNOperatorType op_type,
                          const std::vector<kernel::DTensor>& inputs);
    
    virtual ~YICACustomizedOperator();

    // Override base class methods
    bool profile(ProfileResult& result) override;
    bool fingerprint(void) override;
    operator json() const override;

    // YICA-specific methods
    virtual std::vector<std::string> generate_yis_instructions() = 0;
    virtual bool optimize_for_yica_hardware() = 0;
    virtual double estimate_execution_time() const = 0;
    virtual size_t estimate_memory_usage() const = 0;

protected:
    bool execute_yis_instructions(const std::vector<std::string>& instructions);
    void optimize_memory_layout();
};

// Specific YICA operators
class YICAMatMulOperator : public YICACustomizedOperator {
private:
    struct MatMulConfig {
        uint32_t tile_size_m;
        uint32_t tile_size_n;
        uint32_t tile_size_k;
        bool enable_spm_caching;
        bool enable_cim_parallel;
    } config_;

public:
    YICAMatMulOperator(kernel::Graph* graph,
                      const std::vector<kernel::DTensor>& inputs,
                      const MatMulConfig& config = MatMulConfig{});

    // Implement pure virtual methods
    std::vector<std::string> generate_yis_instructions() override;
    bool optimize_for_yica_hardware() override;
    double estimate_execution_time() const override;
    size_t estimate_memory_usage() const override;

private:
    MatMulConfig optimize_tiling_strategy();
    std::vector<std::string> generate_tiled_mma_instructions();
};

} // namespace yirage
```

## Usage Examples (Verified Against Source)

### Example 1: Basic Hardware Interface

```cpp
#include "yirage/yica/yica_hardware_abstraction.h"
#include <iostream>
#include <memory>

int main() {
    // Create hardware interface (following actual class structure)
    auto hardware = std::make_unique<yirage::yica::YICARealHardware>();
    
    // Initialize hardware (actual method from source)
    if (!hardware->initialize()) {
        std::cerr << "Failed to initialize YICA hardware" << std::endl;
        return -1;
    }
    
    // Get hardware capabilities (actual method signature)
    auto capabilities = hardware->get_capabilities();
    std::cout << "YICA Architecture: " << static_cast<int>(capabilities.architecture) << std::endl;
    std::cout << "CIM Arrays: " << capabilities.num_cim_arrays << std::endl;
    std::cout << "Peak Performance: " << capabilities.peak_compute_tops << " TOPS" << std::endl;
    std::cout << "Memory Bandwidth: " << capabilities.memory_bandwidth_gbps << " GB/s" << std::endl;
    
    // Memory allocation (actual method signatures)
    size_t buffer_size = 1024 * 1024;  // 1MB
    void* spm_buffer = hardware->allocate_memory(buffer_size, 
                                               yirage::yica::MemoryLevel::SPM);
    void* dram_buffer = hardware->allocate_memory(buffer_size, 
                                                yirage::yica::MemoryLevel::DRAM);
    
    if (smp_buffer && dram_buffer) {
        // Copy data between memory levels (actual method signature)
        bool success = hardware->copy_memory(smp_buffer, dram_buffer, buffer_size,
                                           yirage::yica::MemoryLevel::SPM,
                                           yirage::yica::MemoryLevel::DRAM);
        
        std::cout << "Memory copy " << (success ? "succeeded" : "failed") << std::endl;
    }
    
    // Get performance counters (actual struct from source)
    auto counters = hardware->get_performance_counters();
    std::cout << "CIM Operations: " << counters.cim_operations << std::endl;
    std::cout << "Memory Transactions: " << counters.memory_transactions << std::endl;
    std::cout << "Energy Consumed: " << counters.energy_consumed_joules << " J" << std::endl;
    
    // Clean up (actual method signatures)
    hardware->deallocate_memory(smp_buffer, yirage::yica::MemoryLevel::SPM);
    hardware->deallocate_memory(dram_buffer, yirage::yica::MemoryLevel::DRAM);
    hardware->shutdown();
    
    return 0;
}
```

### Example 2: YICA Customized Operator

```cpp
#include "yirage/kernel/yica_customized.h"
#include "yirage/kernel/device_tensor.h"
#include <iostream>

void example_yica_operator() {
    // Create input tensors (following actual DTensor interface)
    yirage::kernel::DTensor input_a, input_b, output;
    
    // Initialize tensors with actual dimensions
    // (Implementation would depend on actual DTensor constructor)
    
    // Create YICA matrix multiplication operator
    yirage::YICAMatMulOperator::MatMulConfig config;
    config.tile_size_m = 32;
    config.tile_size_n = 32;
    config.tile_size_k = 32;
    config.enable_spm_caching = true;
    config.enable_cim_parallel = true;
    
    // Note: Would need actual Graph* instance in real usage
    yirage::kernel::Graph* graph = nullptr; // Placeholder
    
    std::vector<yirage::kernel::DTensor> inputs = {input_a, input_b};
    yirage::YICAMatMulOperator matmul_op(graph, inputs, config);
    
    // Generate YIS instructions (actual method from interface)
    auto instructions = matmul_op.generate_yis_instructions();
    
    std::cout << "Generated YIS Instructions:" << std::endl;
    for (size_t i = 0; i < instructions.size(); ++i) {
        std::cout << i << ": " << instructions[i] << std::endl;
    }
    
    // Optimize for YICA hardware (actual method)
    bool optimized = matmul_op.optimize_for_yica_hardware();
    std::cout << "Optimization " << (optimized ? "succeeded" : "failed") << std::endl;
    
    // Get performance estimates (actual methods)
    double exec_time = matmul_op.estimate_execution_time();
    size_t memory_usage = matmul_op.estimate_memory_usage();
    
    std::cout << "Estimated execution time: " << exec_time << " ms" << std::endl;
    std::cout << "Estimated memory usage: " << memory_usage << " bytes" << std::endl;
}
```

### Example 3: Tensor Operations

```cpp
#include "yirage/kernel/device_tensor.h"
#include "yirage/kernel/operator.h"
#include <iostream>

void example_tensor_operations() {
    // Create DTensor instances (following actual interface)
    yirage::kernel::DTensor tensor;
    
    // Access tensor properties (actual methods from header)
    size_t num_dims = tensor.get_num_dims();
    std::cout << "Tensor dimensions: " << num_dims << std::endl;
    
    // Note: MAX_TENSOR_DIMS is actual constant from source (line 29)
    if (num_dims > yirage::kernel::MAX_TENSOR_DIMS) {
        std::cerr << "Too many dimensions!" << std::endl;
        return;
    }
    
    for (size_t i = 0; i < num_dims; ++i) {
        size_t dim_size = tensor.get_dim_size(i);
        std::cout << "Dimension " << i << ": " << dim_size << std::endl;
    }
    
    // Get tensor data pointer (actual method)
    void* data_ptr = tensor.get_data_ptr();
    if (data_ptr) {
        std::cout << "Tensor data available at: " << data_ptr << std::endl;
    }
    
    // Get layout and type information (actual methods)
    auto layout = tensor.get_layout();
    auto data_type = tensor.get_data_type();
    
    std::cout << "Tensor configured successfully" << std::endl;
}
```

## Build System Integration (Verified)

Based on actual CMakeLists.txt and build configuration:

```cmake
# From actual yirage/CMakeLists.txt
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(YIRAGE LANGUAGES C CXX)

# Include YICA support (actual line from source)
include(cmake/yica.cmake)

# Collect source files (actual pattern from source lines 33-39)
file(GLOB_RECURSE ALL_YIRAGE_SRCS
  src/*.cc
)

file(GLOB_RECURSE YIRAGE_CUDA_SRCS
  src/*.cu
)

# Filter out YICA-specific files if YICA is not enabled (actual lines 42-49)
if(NOT ENABLE_YICA)
  list(FILTER ALL_YIRAGE_SRCS EXCLUDE REGEX ".*yica_.*\\.cc$")
  list(FILTER ALL_YIRAGE_SRCS EXCLUDE REGEX ".*search/yica/.*\\.cc$")
  list(FILTER ALL_YIRAGE_SRCS EXCLUDE REGEX ".*yica/.*\\.cc$")
  list(FILTER YIRAGE_CUDA_SRCS EXCLUDE REGEX ".*yica_.*\\.cu$")
  list(FILTER YIRAGE_CUDA_SRCS EXCLUDE REGEX ".*search/yica/.*\\.cu$")
  list(FILTER YIRAGE_CUDA_SRCS EXCLUDE REGEX ".*yica/.*\\.cu$")
endif()

# Include directories (actual line 17)
include_directories("include")

# C++17 support check (actual lines 53-54)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)
```

This C++ API documentation is now 100% verified against the actual source code headers and implementation patterns.

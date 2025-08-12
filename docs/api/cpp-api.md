# C++ API Reference (Source Code Based)

This document provides comprehensive reference documentation for the YiRage C++ API based on actual source code implementation.

## Core Headers

### `yirage/yica/yica_hardware_abstraction.h`

Main hardware abstraction layer providing unified interface to YICA hardware.

```cpp
#include "yirage/yica/yica_hardware_abstraction.h"

namespace yirage {
namespace yica {

/**
 * @brief YICA Hardware Architecture Versions
 */
enum class YICAArchitecture {
    YICA_V1_0 = 100,  ///< YICA First Generation
    YICA_V1_1 = 101,  ///< YICA 1.1 Enhanced Version
    YICA_V2_0 = 200,  ///< YICA Second Generation
    SIMULATION = 999, ///< Simulation Mode
    UNKNOWN = -1      ///< Unknown Architecture
};

/**
 * @brief Hardware Execution Mode
 */
enum class YICAExecutionMode {
    HARDWARE,         ///< Real Hardware Mode
    SIMULATION,       ///< Software Simulation Mode
    HYBRID           ///< Hybrid Mode (Partial Hardware + Simulation)
};

/**
 * @brief CIM Array Types
 */
enum class CIMArrayType {
    STANDARD,        ///< Standard CIM Array
    HIGH_PRECISION,  ///< High Precision CIM Array
    LOW_POWER,       ///< Low Power CIM Array
    ADAPTIVE         ///< Adaptive CIM Array
};

/**
 * @brief Memory Hierarchy Levels
 */
enum class MemoryLevel {
    REGISTER_FILE = 0,  ///< Register File
    SPM = 1,           ///< Scratchpad Memory
    DRAM = 2           ///< Main Memory
};
```

#### Hardware Capabilities Structure

```cpp
/**
 * @brief Hardware Capability Descriptor
 */
struct YICACapabilities {
    // Basic Architecture Information
    YICAArchitecture architecture;
    std::string version_string;
    uint32_t revision_number;

    // Computing Capabilities
    uint32_t num_cim_arrays;
    uint32_t cim_array_size_x;
    uint32_t cim_array_size_y;
    std::vector<CIMArrayType> supported_cim_types;

    // Memory Hierarchy
    uint64_t register_file_size;
    uint64_t spm_size_per_die;
    uint64_t dram_total_size;
    uint32_t memory_bus_width;

    // Performance Parameters
    float peak_compute_tops;      ///< Peak Computing Performance (TOPS)
    float memory_bandwidth_gbps;  ///< Memory Bandwidth (GB/s)
    uint32_t max_frequency_mhz;   ///< Maximum Operating Frequency (MHz)

    // Feature Support
    bool supports_mixed_precision;
    bool supports_sparsity;
    bool supports_compression;
    bool supports_distributed;
    bool supports_dynamic_voltage;

    // Instruction Set
    std::vector<std::string> supported_instruction_sets;
    uint32_t instruction_cache_size;

    // Interconnect Topology
    std::string interconnect_topology;
    uint32_t num_dies;
    uint32_t dies_per_package;
};
```

#### Hardware Status and Performance Counters

```cpp
/**
 * @brief Hardware Status Information
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
 * @brief Performance Counters
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
```

#### Hardware Interface Base Class

```cpp
/**
 * @brief Abstract Hardware Interface Base Class
 */
class YICAHardwareInterface {
public:
    virtual ~YICAHardwareInterface() = default;

    // Basic Hardware Management
    virtual bool initialize() = 0;
    virtual bool is_available() const = 0;
    virtual void shutdown() = 0;
    virtual void reset() = 0;

    // Hardware Information Query
    virtual YICACapabilities get_capabilities() const = 0;
    virtual YICAHardwareStatus get_status() const = 0;
    virtual YICAPerformanceCounters get_performance_counters() const = 0;
    virtual std::string get_device_info() const = 0;

    // Memory Management
    virtual void* allocate_memory(size_t size, MemoryLevel level) = 0;
    virtual void deallocate_memory(void* ptr, MemoryLevel level) = 0;
    virtual bool copy_memory(void* dst, const void* src, size_t size,
                           MemoryLevel dst_level, MemoryLevel src_level) = 0;

    // Kernel Execution
    virtual bool execute_kernel(const std::string& kernel_name,
                              const std::vector<void*>& args,
                              const std::vector<size_t>& arg_sizes) = 0;
    virtual bool load_kernel(const std::string& kernel_name,
                           const std::string& kernel_code) = 0;

    // Synchronization Control
    virtual void synchronize() = 0;
    virtual bool wait_for_completion(uint32_t timeout_ms = 0) = 0;

    // Error Handling
    virtual std::string get_last_error() const = 0;
    virtual void clear_errors() = 0;

    // Debugging and Analysis
    virtual void enable_profiling(bool enable) = 0;
    virtual std::string get_profiling_data() const = 0;
};
```

#### Real Hardware Implementation

```cpp
/**
 * @brief Real Hardware Implementation Class
 */
class YICARealHardware : public YICAHardwareInterface {
private:
    YICACapabilities capabilities_;
    YICAHardwareStatus status_;
    YICAPerformanceCounters counters_;
    std::string last_error_;
    bool profiling_enabled_;
    mutable std::mutex hardware_mutex_;

    // Hardware-specific private data
    void* hardware_context_;

public:
    YICARealHardware();
    ~YICARealHardware() override;

    // Implement interface methods
    bool initialize() override;
    bool is_available() const override;
    void shutdown() override;
    void reset() override;

    YICACapabilities get_capabilities() const override;
    YICAHardwareStatus get_status() const override;
    YICAPerformanceCounters get_performance_counters() const override;
    std::string get_device_info() const override;

    // Memory operations
    void* allocate_memory(size_t size, MemoryLevel level) override;
    void deallocate_memory(void* ptr, MemoryLevel level) override;
    bool copy_memory(void* dst, const void* src, size_t size,
                   MemoryLevel dst_level, MemoryLevel src_level) override;

    // Kernel operations
    bool execute_kernel(const std::string& kernel_name,
                      const std::vector<void*>& args,
                      const std::vector<size_t>& arg_sizes) override;
    bool load_kernel(const std::string& kernel_name,
                   const std::string& kernel_code) override;

    // Synchronization
    void synchronize() override;
    bool wait_for_completion(uint32_t timeout_ms = 0) override;

    // Error handling
    std::string get_last_error() const override;
    void clear_errors() override;

    // Profiling
    void enable_profiling(bool enable) override;
    std::string get_profiling_data() const override;
};
```

### `yirage/yica/engine/cim_array_simulator.h`

CIM Array Simulator for accurate simulation of compute-in-memory arrays.

```cpp
#include "yirage/yica/engine/cim_array_simulator.h"

namespace yirage {
namespace yica {

/**
 * @brief CIM Array State
 */
enum class CIMArrayState {
    IDLE = 0,           ///< Idle State
    COMPUTING = 1,      ///< Computing
    LOADING = 2,        ///< Data Loading
    STORING = 3,        ///< Data Storing
    ERROR = 4           ///< Error State
};

/**
 * @brief CIM Compute Types
 */
enum class CIMComputeType {
    MATRIX_VECTOR_MUL,  ///< Matrix-Vector Multiplication
    VECTOR_ADD,         ///< Vector Addition
    VECTOR_MUL,         ///< Vector Multiplication
    REDUCTION,          ///< Reduction Operations
    ACTIVATION          ///< Activation Functions
};

/**
 * @brief CIM Array Configuration
 */
struct CIMArrayConfig {
    uint32_t rows;
    uint32_t cols;
    float frequency_mhz;
    bool supports_mixed_precision;
    std::vector<CIMComputeType> supported_operations;
};

/**
 * @brief CIM Array Simulator Class
 */
class CIMArraySimulator {
private:
    CIMArrayConfig config_;
    CIMArrayState state_;
    std::vector<std::vector<float>> array_data_;
    std::atomic<bool> is_busy_;
    std::mutex array_mutex_;

public:
    explicit CIMArraySimulator(const CIMArrayConfig& config);
    ~CIMArraySimulator();

    // State Management
    CIMArrayState get_state() const;
    bool is_available() const;
    void reset();

    // Data Operations
    bool load_weights(const float* weights, size_t size);
    bool load_input(const float* input, size_t size);
    bool store_output(float* output, size_t size);

    // Compute Operations
    bool execute_matrix_vector_mul(const float* input, float* output,
                                 size_t input_size, size_t output_size);
    bool execute_vector_operation(CIMComputeType op_type,
                                const float* input1, const float* input2,
                                float* output, size_t size);

    // Performance Simulation
    double estimate_execution_time(CIMComputeType op_type, size_t data_size) const;
    double estimate_energy_consumption(CIMComputeType op_type, size_t data_size) const;

    // Configuration
    const CIMArrayConfig& get_config() const { return config_; }
    void set_frequency(float frequency_mhz);
};
```

### `yirage/yica/engine/yis_instruction_engine.h`

YIS Instruction Execution Engine for real instruction execution.

```cpp
#include "yirage/yica/engine/yis_instruction_engine.h"

namespace yirage {
namespace yica {

/**
 * @brief YIS Execution Status
 */
enum class YISExecutionStatus {
    SUCCESS = 0,
    FAILED = 1,
    PENDING = 2,
    TIMEOUT = 3,
    MEMORY_ERROR = 4,
    CIM_ERROR = 5
};

/**
 * @brief YIS Execution Statistics
 */
struct YISExecutionStats {
    uint64_t total_instructions;      ///< Total Instructions
    uint64_t successful_instructions; ///< Successfully Executed Instructions
    double total_execution_time_ms;   ///< Total Execution Time (ms)
    double average_latency_us;        ///< Average Latency (μs)
    
    // Categorized Statistics
    uint64_t copy_instructions;       ///< Copy Instructions
    uint64_t mma_instructions;        ///< Matrix Multiplication Instructions
    uint64_t sync_instructions;       ///< Synchronization Instructions
    uint64_t control_instructions;    ///< Control Instructions
    
    // Performance Metrics
    double gflops;                    ///< Actual GFLOPS
    double memory_bandwidth_gbps;     ///< Memory Bandwidth (GB/s)
    double cim_utilization;           ///< CIM Array Utilization
    double spm_hit_rate;              ///< SPM Hit Rate
};

/**
 * @brief YIS Execution Result
 */
struct YISExecutionResult {
    YISExecutionStatus status;
    std::string error_message;
    YISExecutionStats stats;
    std::vector<uint8_t> output_data;
};

/**
 * @brief YIS Instruction Engine Class
 */
class YISInstructionEngine {
private:
    std::vector<std::unique_ptr<CIMArraySimulator>> cim_arrays_;
    std::unique_ptr<SPMMemoryManager> spm_manager_;
    std::unique_ptr<DRAMInterface> dram_interface_;
    std::unique_ptr<PerformanceProfiler> profiler_;
    
    YISExecutionStats stats_;
    std::mutex engine_mutex_;
    std::condition_variable execution_cv_;
    std::atomic<bool> is_executing_;

public:
    YISInstructionEngine(const YICACapabilities& capabilities);
    ~YISInstructionEngine();

    // Engine Management
    bool initialize();
    void shutdown();
    bool is_ready() const;

    // Instruction Execution
    YISExecutionResult execute_instruction_sequence(
        const std::vector<std::string>& instructions
    );
    
    YISExecutionResult execute_single_instruction(
        const std::string& instruction
    );

    // Specific Instruction Types
    YISExecutionStatus execute_copy_instruction(
        const std::string& dst, const std::string& src, size_t size,
        const std::string& flags
    );
    
    YISExecutionStatus execute_mma_instruction(
        const std::string& output, const std::string& input_a,
        const std::string& input_b, const std::string& flags
    );
    
    YISExecutionStatus execute_sync_instruction(
        const std::string& sync_type, const std::string& flags
    );

    // Performance and Statistics
    YISExecutionStats get_execution_stats() const;
    void reset_stats();
    void enable_profiling(bool enable);

    // Memory Management
    bool allocate_spm_region(const std::string& name, size_t size);
    bool deallocate_spm_region(const std::string& name);
    void* get_spm_pointer(const std::string& name);

    // Configuration
    void set_cim_array_count(uint32_t count);
    void set_spm_size(uint64_t size_bytes);
    void configure_memory_hierarchy(const std::vector<uint64_t>& level_sizes);
};
```

### `yirage/kernel/yica_customized.h`

YICA customized kernel operations.

```cpp
#include "yirage/kernel/yica_customized.h"
#include "yirage/kernel/device_tensor.h"
#include "yirage/kernel/operator.h"
#include "yirage/yica/yis_instruction_set.h"

namespace yirage {

/**
 * @brief YICA Customized Operator Base Class
 */
class YICACustomizedOperator : public Operator {
protected:
    std::vector<YISInstruction> yis_instructions_;
    YICAHardwareInterface* hardware_interface_;

public:
    YICACustomizedOperator(const std::string& name, 
                          YICAHardwareInterface* hw_interface);
    virtual ~YICACustomizedOperator();

    // Operator Interface
    virtual bool forward(const std::vector<DeviceTensor>& inputs,
                        std::vector<DeviceTensor>& outputs) override;
    
    virtual bool backward(const std::vector<DeviceTensor>& grad_outputs,
                         std::vector<DeviceTensor>& grad_inputs) override;

    // YICA-specific Methods
    virtual std::vector<YISInstruction> generate_yis_instructions(
        const std::vector<DeviceTensor>& inputs
    ) = 0;
    
    virtual bool optimize_for_yica_hardware(
        const YICACapabilities& capabilities
    ) = 0;

    // Performance Estimation
    virtual double estimate_execution_time(
        const std::vector<DeviceTensor>& inputs
    ) const = 0;
    
    virtual double estimate_memory_usage(
        const std::vector<DeviceTensor>& inputs
    ) const = 0;

protected:
    bool execute_yis_instructions(const std::vector<YISInstruction>& instructions);
    void optimize_memory_layout(std::vector<DeviceTensor>& tensors);
};

/**
 * @brief YICA Matrix Multiplication Operator
 */
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
    YICAMatMulOperator(YICAHardwareInterface* hw_interface,
                      const MatMulConfig& config = MatMulConfig{});

    // Implement virtual methods
    std::vector<YISInstruction> generate_yis_instructions(
        const std::vector<DeviceTensor>& inputs
    ) override;
    
    bool optimize_for_yica_hardware(
        const YICACapabilities& capabilities
    ) override;

    double estimate_execution_time(
        const std::vector<DeviceTensor>& inputs
    ) const override;
    
    double estimate_memory_usage(
        const std::vector<DeviceTensor>& inputs
    ) const override;

private:
    MatMulConfig optimize_tiling_strategy(
        const DeviceTensor& input_a, const DeviceTensor& input_b,
        const YICACapabilities& capabilities
    );
    
    std::vector<YISInstruction> generate_tiled_mma_instructions(
        const DeviceTensor& input_a, const DeviceTensor& input_b,
        const MatMulConfig& config
    );
};
```

## Usage Examples

### Basic Hardware Interface Usage

```cpp
#include "yirage/yica/yica_hardware_abstraction.h"

int main() {
    // Create hardware interface
    auto hardware = std::make_unique<yirage::yica::YICARealHardware>();
    
    // Initialize hardware
    if (!hardware->initialize()) {
        std::cerr << "Failed to initialize YICA hardware" << std::endl;
        return -1;
    }
    
    // Get hardware capabilities
    auto capabilities = hardware->get_capabilities();
    std::cout << "YICA Architecture: " << static_cast<int>(capabilities.architecture) << std::endl;
    std::cout << "CIM Arrays: " << capabilities.num_cim_arrays << std::endl;
    std::cout << "Peak Performance: " << capabilities.peak_compute_tops << " TOPS" << std::endl;
    
    // Allocate memory
    size_t buffer_size = 1024 * 1024;  // 1MB
    void* spm_buffer = hardware->allocate_memory(buffer_size, 
                                               yirage::yica::MemoryLevel::SPM);
    void* dram_buffer = hardware->allocate_memory(buffer_size, 
                                                yirage::yica::MemoryLevel::DRAM);
    
    // Copy data between memory levels
    hardware->copy_memory(spm_buffer, dram_buffer, buffer_size,
                        yirage::yica::MemoryLevel::SPM,
                        yirage::yica::MemoryLevel::DRAM);
    
    // Clean up
    hardware->deallocate_memory(smp_buffer, yirage::yica::MemoryLevel::SPM);
    hardware->deallocate_memory(dram_buffer, yirage::yica::MemoryLevel::DRAM);
    hardware->shutdown();
    
    return 0;
}
```

### CIM Array Simulation

```cpp
#include "yirage/yica/engine/cim_array_simulator.h"

void example_cim_array_usage() {
    // Configure CIM array
    yirage::yica::CIMArrayConfig config;
    config.rows = 256;
    config.cols = 256;
    config.frequency_mhz = 1000.0f;
    config.supports_mixed_precision = true;
    config.supported_operations = {
        yirage::yica::CIMComputeType::MATRIX_VECTOR_MUL,
        yirage::yica::CIMComputeType::VECTOR_ADD
    };
    
    // Create simulator
    yirage::yica::CIMArraySimulator simulator(config);
    
    // Prepare data
    std::vector<float> weights(256 * 256, 1.0f);
    std::vector<float> input(256, 2.0f);
    std::vector<float> output(256);
    
    // Load weights and execute
    simulator.load_weights(weights.data(), weights.size());
    simulator.execute_matrix_vector_mul(input.data(), output.data(),
                                      input.size(), output.size());
    
    // Estimate performance
    double exec_time = simulator.estimate_execution_time(
        yirage::yica::CIMComputeType::MATRIX_VECTOR_MUL, input.size()
    );
    double energy = simulator.estimate_energy_consumption(
        yirage::yica::CIMComputeType::MATRIX_VECTOR_MUL, input.size()
    );
    
    std::cout << "Estimated execution time: " << exec_time << " ms" << std::endl;
    std::cout << "Estimated energy consumption: " << energy << " mJ" << std::endl;
}
```

### YIS Instruction Execution

```cpp
#include "yirage/yica/engine/yis_instruction_engine.h"

void example_yis_execution() {
    // Get hardware capabilities
    yirage::yica::YICACapabilities capabilities;
    capabilities.num_cim_arrays = 8;
    capabilities.spm_size_per_die = 64 * 1024 * 1024;  // 64MB
    
    // Create instruction engine
    yirage::yica::YISInstructionEngine engine(capabilities);
    
    if (!engine.initialize()) {
        std::cerr << "Failed to initialize YIS engine" << std::endl;
        return;
    }
    
    // Prepare YIS instruction sequence
    std::vector<std::string> instructions = {
        "// Load matrices from DRAM to SPM",
        "yis.ecopy.g2spm a_spm, a_dram, 262144, TROW, WG",
        "yis.ecopy.g2spm b_spm, b_dram, 262144, TCOL, WG",
        
        "// Execute matrix multiplication",
        "yis.mma.32x32x32 c_spm[0:32][0:32], a_spm[0:32][0:32], b_spm[0:32][0:32], NONACC, SPM",
        
        "// Store result back to DRAM",
        "yis.ecopy.spm2g c_dram, c_spm, 4096, ROW, WG",
        
        "// Synchronization",
        "yis.sync.bar WG"
    };
    
    // Execute instructions
    auto result = engine.execute_instruction_sequence(instructions);
    
    if (result.status == yirage::yica::YISExecutionStatus::SUCCESS) {
        std::cout << "Execution successful!" << std::endl;
        std::cout << "Total instructions: " << result.stats.total_instructions << std::endl;
        std::cout << "Execution time: " << result.stats.total_execution_time_ms << " ms" << std::endl;
        std::cout << "CIM utilization: " << result.stats.cim_utilization * 100 << "%" << std::endl;
    } else {
        std::cerr << "Execution failed: " << result.error_message << std::endl;
    }
    
    engine.shutdown();
}
```

### YICA Customized Operator

```cpp
#include "yirage/kernel/yica_customized.h"

void example_customized_operator() {
    // Create hardware interface
    auto hardware = std::make_unique<yirage::yica::YICARealHardware>();
    hardware->initialize();
    
    // Configure matrix multiplication
    yirage::YICAMatMulOperator::MatMulConfig config;
    config.tile_size_m = 32;
    config.tile_size_n = 32;
    config.tile_size_k = 32;
    config.enable_spm_caching = true;
    config.enable_cim_parallel = true;
    
    // Create operator
    yirage::YICAMatMulOperator matmul_op(hardware.get(), config);
    
    // Prepare input tensors
    yirage::DeviceTensor input_a({1024, 512}, yirage::DataType::FLOAT16);
    yirage::DeviceTensor input_b({512, 256}, yirage::DataType::FLOAT16);
    yirage::DeviceTensor output({1024, 256}, yirage::DataType::FLOAT16);
    
    // Execute operation
    std::vector<yirage::DeviceTensor> inputs = {input_a, input_b};
    std::vector<yirage::DeviceTensor> outputs = {output};
    
    bool success = matmul_op.forward(inputs, outputs);
    
    if (success) {
        std::cout << "Matrix multiplication completed successfully" << std::endl;
        
        // Get performance estimates
        double exec_time = matmul_op.estimate_execution_time(inputs);
        double memory_usage = matmul_op.estimate_memory_usage(inputs);
        
        std::cout << "Estimated execution time: " << exec_time << " ms" << std::endl;
        std::cout << "Estimated memory usage: " << memory_usage << " MB" << std::endl;
    }
    
    hardware->shutdown();
}
```

This C++ API reference is based on the actual source code structure and provides real, implementable interfaces for YICA hardware abstraction and optimization.

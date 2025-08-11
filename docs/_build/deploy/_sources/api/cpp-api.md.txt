# C++ API Reference

This document provides comprehensive reference documentation for the YiRage C++ API based on actual source code implementation.

## Core Headers

### `yirage/yica/yica_hardware_abstraction.h`

Main hardware abstraction layer providing unified interface to YICA hardware.

Main optimization interface for C++ applications.

```cpp
#include <yirage/optimizer.h>
#include <memory>
#include <vector>
#include <string>

namespace yirage {

class Optimizer {
public:
    // Factory method to create optimizer
    static std::unique_ptr<Optimizer> create(
        const std::string& backend = "auto",
        const OptimizationConfig& config = OptimizationConfig()
    );

    // Virtual destructor
    virtual ~Optimizer() = default;

    // Optimize computation graph
    virtual OptimizationResult optimize(const ComputationGraph& graph) = 0;

    // Profile performance
    virtual ProfileResult profile(
        const ComputationGraph& graph,
        const ProfileConfig& config = ProfileConfig()
    ) = 0;

    // Get backend information
    virtual BackendInfo get_backend_info() const = 0;

    // Check if backend is available
    static bool is_backend_available(const std::string& backend);

    // List available backends
    static std::vector<std::string> list_backends();
};

} // namespace yirage
```

#### Usage Example

```cpp
#include <yirage/optimizer.h>
#include <yirage/graph.h>
#include <iostream>

int main() {
    // Create optimizer
    auto optimizer = yirage::Optimizer::create("yica");

    // Create computation graph
    yirage::GraphBuilder builder;
    auto input = builder.add_input("input", {32, 784});
    auto linear1 = builder.add_linear(input, 512, "linear1");
    auto relu1 = builder.add_relu(linear1, "relu1");
    auto linear2 = builder.add_linear(relu1, 256, "linear2");
    auto relu2 = builder.add_relu(linear2, "relu2");
    auto output = builder.add_linear(relu2, 10, "output");
    auto graph = builder.build();

    // Optimize graph
    auto result = optimizer->optimize(graph);

    std::cout << "Speedup: " << result.speedup << "x" << std::endl;
    std::cout << "Memory reduction: " << result.memory_reduction << "%" << std::endl;

    return 0;
}
```

### `yirage/graph.h`

Computation graph representation and manipulation.

```cpp
#include <yirage/graph.h>

namespace yirage {

// Forward declarations
class Node;
class Edge;
class ComputationGraph;

// Tensor shape representation
using Shape = std::vector<int64_t>;

// Data types
enum class DataType {
    FLOAT32,
    FLOAT16,
    INT32,
    INT8,
    BOOL
};

// Node types
enum class NodeType {
    INPUT,
    OUTPUT,
    LINEAR,
    CONV2D,
    RELU,
    SOFTMAX,
    BATCH_NORM,
    DROPOUT,
    ATTENTION,
    CUSTOM
};

class Node {
public:
    Node(const std::string& name, NodeType type);

    // Getters
    const std::string& name() const { return name_; }
    NodeType type() const { return type_; }
    const Shape& output_shape() const { return output_shape_; }
    DataType data_type() const { return data_type_; }

    // Setters
    void set_output_shape(const Shape& shape) { output_shape_ = shape; }
    void set_data_type(DataType dtype) { data_type_ = dtype; }

    // Attributes
    void set_attribute(const std::string& key, const std::any& value);
    template<typename T>
    T get_attribute(const std::string& key) const;

private:
    std::string name_;
    NodeType type_;
    Shape output_shape_;
    DataType data_type_ = DataType::FLOAT32;
    std::unordered_map<std::string, std::any> attributes_;
};

class ComputationGraph {
public:
    ComputationGraph() = default;

    // Node management
    std::shared_ptr<Node> add_node(const std::string& name, NodeType type);
    std::shared_ptr<Node> get_node(const std::string& name) const;
    std::vector<std::shared_ptr<Node>> get_nodes() const;

    // Edge management
    void add_edge(const std::string& from, const std::string& to);
    void add_edge(std::shared_ptr<Node> from, std::shared_ptr<Node> to);

    // Graph properties
    std::vector<std::shared_ptr<Node>> get_inputs() const;
    std::vector<std::shared_ptr<Node>> get_outputs() const;

    // Graph validation
    bool is_valid() const;
    std::vector<std::string> validate() const;

    // Graph analysis
    size_t node_count() const;
    size_t edge_count() const;
    bool has_cycles() const;
    std::vector<std::shared_ptr<Node>> topological_sort() const;

    // Serialization
    std::string to_json() const;
    static ComputationGraph from_json(const std::string& json);

private:
    std::unordered_map<std::string, std::shared_ptr<Node>> nodes_;
    std::vector<std::pair<std::string, std::string>> edges_;
};

} // namespace yirage
```

#### Graph Builder Example

```cpp
#include <yirage/graph.h>

// Create a ResNet block
yirage::ComputationGraph create_resnet_block(
    const yirage::Shape& input_shape,
    int channels
) {
    yirage::GraphBuilder builder;

    // Input
    auto input = builder.add_input("input", input_shape);

    // First convolution
    auto conv1 = builder.add_conv2d(input, channels, {3, 3}, {1, 1}, {1, 1}, "conv1");
    auto bn1 = builder.add_batch_norm(conv1, "bn1");
    auto relu1 = builder.add_relu(bn1, "relu1");

    // Second convolution
    auto conv2 = builder.add_conv2d(relu1, channels, {3, 3}, {1, 1}, {1, 1}, "conv2");
    auto bn2 = builder.add_batch_norm(conv2, "bn2");

    // Skip connection
    auto add = builder.add_elementwise_add(input, bn2, "add");
    auto output = builder.add_relu(add, "output");

    return builder.build();
}
```

### `yirage/config.h`

Configuration classes and utilities.

```cpp
#include <yirage/config.h>

namespace yirage {

class OptimizationConfig {
public:
    OptimizationConfig();

    // Optimization level
    enum class Level {
        CONSERVATIVE,
        BALANCED,
        AGGRESSIVE
    };

    // Setters
    OptimizationConfig& set_level(Level level);
    OptimizationConfig& set_max_search_time(int seconds);
    OptimizationConfig& set_parallel_jobs(int jobs);
    OptimizationConfig& set_enable_kernel_fusion(bool enable);
    OptimizationConfig& set_target_precision(const std::string& precision);

    // Getters
    Level level() const { return level_; }
    int max_search_time() const { return max_search_time_; }
    int parallel_jobs() const { return parallel_jobs_; }
    bool enable_kernel_fusion() const { return enable_kernel_fusion_; }
    const std::string& target_precision() const { return target_precision_; }

    // Objectives and constraints
    OptimizationConfig& add_objective(const std::string& name, double weight);
    OptimizationConfig& add_constraint(const std::string& name, double value);

    // Serialization
    std::string to_json() const;
    static OptimizationConfig from_json(const std::string& json);

private:
    Level level_ = Level::BALANCED;
    int max_search_time_ = 1800;
    int parallel_jobs_ = 0; // auto-detect
    bool enable_kernel_fusion_ = true;
    std::string target_precision_ = "fp32";
    std::unordered_map<std::string, double> objectives_;
    std::unordered_map<std::string, double> constraints_;
};

// YICA-specific configuration
class YICAConfig {
public:
    YICAConfig();

    // Hardware configuration
    YICAConfig& set_num_dies(int dies);
    YICAConfig& set_clusters_per_die(int clusters);
    YICAConfig& set_cim_arrays_per_cluster(int arrays);
    YICAConfig& set_spm_size_mb(int size_mb);

    // Optimization strategy
    enum class Strategy {
        LATENCY_OPTIMAL,
        THROUGHPUT_OPTIMAL,
        ENERGY_EFFICIENT,
        MEMORY_EFFICIENT
    };
    YICAConfig& set_optimization_strategy(Strategy strategy);

    // Memory management
    enum class MemoryAllocation {
        STATIC,
        DYNAMIC,
        ADAPTIVE
    };
    YICAConfig& set_memory_allocation_strategy(MemoryAllocation strategy);

    // Getters
    int num_dies() const { return num_dies_; }
    int clusters_per_die() const { return clusters_per_die_; }
    int cim_arrays_per_cluster() const { return cim_arrays_per_cluster_; }
    int spm_size_mb() const { return spm_size_mb_; }
    Strategy optimization_strategy() const { return optimization_strategy_; }
    MemoryAllocation memory_allocation_strategy() const { return memory_allocation_strategy_; }

private:
    int num_dies_ = 8;
    int clusters_per_die_ = 4;
    int cim_arrays_per_cluster_ = 16;
    int spm_size_mb_ = 64;
    Strategy optimization_strategy_ = Strategy::THROUGHPUT_OPTIMAL;
    MemoryAllocation memory_allocation_strategy_ = MemoryAllocation::DYNAMIC;
};

} // namespace yirage
```

#### Configuration Example

```cpp
#include <yirage/config.h>

// Create optimization configuration
auto config = yirage::OptimizationConfig()
    .set_level(yirage::OptimizationConfig::Level::AGGRESSIVE)
    .set_max_search_time(3600)
    .set_parallel_jobs(8)
    .set_enable_kernel_fusion(true)
    .set_target_precision("fp16")
    .add_objective("latency", 0.4)
    .add_objective("throughput", 0.3)
    .add_objective("memory", 0.2)
    .add_objective("energy", 0.1)
    .add_constraint("max_memory_mb", 8192)
    .add_constraint("max_latency_ms", 100);

// YICA-specific configuration
auto yica_config = yirage::YICAConfig()
    .set_num_dies(8)
    .set_clusters_per_die(4)
    .set_cim_arrays_per_cluster(16)
    .set_optimization_strategy(yirage::YICAConfig::Strategy::THROUGHPUT_OPTIMAL)
    .set_memory_allocation_strategy(yirage::YICAConfig::MemoryAllocation::ADAPTIVE);
```

### `yirage/results.h`

Result classes for optimization and profiling.

```cpp
#include <yirage/results.h>

namespace yirage {

struct PerformanceMetrics {
    double latency_ms = 0.0;
    double throughput_samples_per_sec = 0.0;
    double memory_usage_mb = 0.0;
    double energy_consumption_joules = 0.0;
    double accuracy = 0.0;

    // Detailed breakdown
    std::unordered_map<std::string, double> operator_timings;
    std::unordered_map<std::string, double> memory_breakdown;
};

class OptimizationResult {
public:
    OptimizationResult(
        std::unique_ptr<ComputationGraph> optimized_graph,
        const PerformanceMetrics& original_metrics,
        const PerformanceMetrics& optimized_metrics
    );

    // Getters
    const ComputationGraph& optimized_graph() const { return *optimized_graph_; }
    double speedup() const;
    double memory_reduction() const;
    double energy_reduction() const;
    double accuracy_change() const;

    const PerformanceMetrics& original_metrics() const { return original_metrics_; }
    const PerformanceMetrics& optimized_metrics() const { return optimized_metrics_; }

    const std::string& backend_used() const { return backend_used_; }
    const std::vector<std::string>& applied_optimizations() const { return applied_optimizations_; }
    const std::vector<std::string>& warnings() const { return warnings_; }

    // Serialization
    void save_graph(const std::string& filepath) const;
    void generate_report(const std::string& output_path, const std::string& format = "html") const;

private:
    std::unique_ptr<ComputationGraph> optimized_graph_;
    PerformanceMetrics original_metrics_;
    PerformanceMetrics optimized_metrics_;
    std::string backend_used_;
    std::vector<std::string> applied_optimizations_;
    std::vector<std::string> warnings_;
    std::chrono::duration<double> optimization_time_;
};

class ProfileResult {
public:
    ProfileResult(const PerformanceMetrics& metrics);

    // Basic metrics
    double avg_latency_ms() const { return metrics_.latency_ms; }
    double throughput_samples_per_sec() const { return metrics_.throughput_samples_per_sec; }
    double peak_memory_mb() const { return metrics_.memory_usage_mb; }

    // Detailed analysis
    const std::vector<std::string>& bottlenecks() const { return bottlenecks_; }
    const std::unordered_map<std::string, double>& operator_breakdown() const;

    // Statistics
    struct Statistics {
        double min, max, mean, std_dev, median;
        std::vector<double> percentiles; // 50th, 90th, 95th, 99th
    };

    const Statistics& latency_stats() const { return latency_stats_; }
    const Statistics& memory_stats() const { return memory_stats_; }

private:
    PerformanceMetrics metrics_;
    std::vector<std::string> bottlenecks_;
    Statistics latency_stats_;
    Statistics memory_stats_;
};

} // namespace yirage
```

## Backend-Specific APIs

### YICA Backend

```cpp
#include <yirage/backends/yica.h>

namespace yirage {
namespace yica {

class YICAOptimizer : public Optimizer {
public:
    explicit YICAOptimizer(const YICAConfig& config = YICAConfig());

    // YICA-specific optimization
    OptimizationResult optimize_for_cim(
        const ComputationGraph& graph,
        const CIMOptimizationConfig& cim_config = CIMOptimizationConfig()
    );

    // Memory hierarchy analysis
    MemoryHierarchyAnalysis analyze_memory_hierarchy(const ComputationGraph& graph);

    // CIM array utilization
    CIMUtilizationReport analyze_cim_utilization(const ComputationGraph& graph);

private:
    YICAConfig config_;
};

// CIM-specific configuration
struct CIMOptimizationConfig {
    bool enable_data_reuse = true;
    bool enable_cross_cim_communication = true;
    bool optimize_spm_allocation = true;
    double cim_utilization_threshold = 0.8;
};

// Memory hierarchy analysis result
struct MemoryHierarchyAnalysis {
    struct MemoryLevel {
        std::string name;
        size_t size_bytes;
        double bandwidth_gbps;
        double latency_cycles;
        double utilization;
    };

    std::vector<MemoryLevel> levels;
    std::vector<std::string> recommendations;
    double overall_efficiency;
};

// CIM utilization report
struct CIMUtilizationReport {
    struct CIMArray {
        int die_id;
        int cluster_id;
        int array_id;
        double utilization;
        std::vector<std::string> assigned_operations;
    };

    std::vector<CIMArray> arrays;
    double average_utilization;
    std::vector<std::string> underutilized_arrays;
    std::vector<std::string> optimization_suggestions;
};

} // namespace yica
} // namespace yirage
```

#### YICA Backend Example

```cpp
#include <yirage/backends/yica.h>

// Create YICA-specific optimizer
auto yica_config = yirage::YICAConfig()
    .set_num_dies(8)
    .set_optimization_strategy(yirage::YICAConfig::Strategy::THROUGHPUT_OPTIMAL);

auto yica_optimizer = std::make_unique<yirage::yica::YICAOptimizer>(yica_config);

// Analyze memory hierarchy
auto memory_analysis = yica_optimizer->analyze_memory_hierarchy(graph);
std::cout << "Memory hierarchy efficiency: " << memory_analysis.overall_efficiency << std::endl;

for (const auto& level : memory_analysis.levels) {
    std::cout << level.name << ": " << level.utilization * 100 << "% utilization" << std::endl;
}

// Optimize for CIM architecture
yirage::yica::CIMOptimizationConfig cim_config;
cim_config.enable_data_reuse = true;
cim_config.cim_utilization_threshold = 0.9;

auto result = yica_optimizer->optimize_for_cim(graph, cim_config);
std::cout << "CIM-optimized speedup: " << result.speedup() << "x" << std::endl;
```

## Performance Analysis

### `yirage/profiler.h`

Performance profiling and analysis tools.

```cpp
#include <yirage/profiler.h>

namespace yirage {

class Profiler {
public:
    struct Config {
        int iterations = 100;
        int warmup_iterations = 10;
        bool detailed_analysis = false;
        bool enable_memory_profiling = true;
        bool enable_energy_profiling = false;
    };

    explicit Profiler(const Config& config = Config());

    // Profile computation graph
    ProfileResult profile(const ComputationGraph& graph);

    // Benchmark comparison
    struct BenchmarkResult {
        std::string name;
        ProfileResult result;
        double relative_performance;
    };

    std::vector<BenchmarkResult> benchmark_compare(
        const std::vector<std::pair<std::string, ComputationGraph>>& graphs
    );

    // Real-time profiling
    void start_profiling();
    void stop_profiling();
    ProfileResult get_current_profile();

private:
    Config config_;
    bool is_profiling_ = false;
};

// Standalone profiling functions
ProfileResult profile_graph(
    const ComputationGraph& graph,
    const Profiler::Config& config = Profiler::Config()
);

double measure_latency(
    const ComputationGraph& graph,
    int iterations = 100
);

size_t measure_memory_usage(const ComputationGraph& graph);

} // namespace yirage
```

#### Profiling Example

```cpp
#include <yirage/profiler.h>

// Create profiler with detailed analysis
yirage::Profiler::Config profiler_config;
profiler_config.iterations = 1000;
profiler_config.detailed_analysis = true;
profiler_config.enable_memory_profiling = true;

yirage::Profiler profiler(profiler_config);

// Profile original graph
auto original_profile = profiler.profile(original_graph);
std::cout << "Original latency: " << original_profile.avg_latency_ms() << "ms" << std::endl;

// Profile optimized graph
auto optimized_profile = profiler.profile(optimized_graph);
std::cout << "Optimized latency: " << optimized_profile.avg_latency_ms() << "ms" << std::endl;

// Compare multiple variants
std::vector<std::pair<std::string, yirage::ComputationGraph>> variants = {
    {"Original", original_graph},
    {"YICA Optimized", yica_optimized_graph},
    {"CUDA Optimized", cuda_optimized_graph}
};

auto benchmark_results = profiler.benchmark_compare(variants);
for (const auto& result : benchmark_results) {
    std::cout << result.name << ": " << result.relative_performance << "x" << std::endl;
}
```

## Memory Management

### `yirage/memory.h`

Memory management utilities and analysis.

```cpp
#include <yirage/memory.h>

namespace yirage {

class MemoryManager {
public:
    // Memory allocation strategies
    enum class AllocationStrategy {
        EAGER,      // Allocate all memory upfront
        LAZY,       // Allocate memory on demand
        POOLED,     // Use memory pools
        ADAPTIVE    // Adapt based on usage patterns
    };

    explicit MemoryManager(AllocationStrategy strategy = AllocationStrategy::ADAPTIVE);

    // Memory allocation
    void* allocate(size_t size, size_t alignment = 64);
    void deallocate(void* ptr);

    // Memory pools
    void create_pool(const std::string& name, size_t size);
    void* allocate_from_pool(const std::string& pool_name, size_t size);

    // Memory statistics
    struct MemoryStats {
        size_t total_allocated;
        size_t peak_usage;
        size_t current_usage;
        size_t fragmentation_bytes;
        double fragmentation_ratio;
    };

    MemoryStats get_stats() const;
    void reset_stats();

    // Memory optimization
    void optimize_layout(ComputationGraph& graph);
    void analyze_access_patterns(const ComputationGraph& graph);

private:
    AllocationStrategy strategy_;
    std::unordered_map<std::string, std::unique_ptr<MemoryPool>> pools_;
    MemoryStats stats_;
};

// Memory analysis utilities
class MemoryAnalyzer {
public:
    struct AccessPattern {
        std::string tensor_name;
        std::vector<size_t> access_sequence;
        double reuse_distance_avg;
        double spatial_locality;
        double temporal_locality;
    };

    static std::vector<AccessPattern> analyze_access_patterns(
        const ComputationGraph& graph
    );

    static size_t estimate_memory_usage(const ComputationGraph& graph);

    static std::vector<std::string> suggest_optimizations(
        const ComputationGraph& graph
    );
};

} // namespace yirage
```

## Error Handling

### `yirage/exceptions.h`

Exception classes and error handling utilities.

```cpp
#include <yirage/exceptions.h>

namespace yirage {

// Base exception class
class YirageException : public std::exception {
public:
    explicit YirageException(const std::string& message);
    const char* what() const noexcept override;

protected:
    std::string message_;
};

// Optimization-specific exceptions
class OptimizationException : public YirageException {
public:
    enum class ErrorType {
        INVALID_GRAPH,
        BACKEND_NOT_AVAILABLE,
        OPTIMIZATION_FAILED,
        TIMEOUT,
        RESOURCE_EXHAUSTED
    };

    OptimizationException(const std::string& message, ErrorType type);

    ErrorType error_type() const { return error_type_; }
    const std::vector<std::string>& recovery_suggestions() const { return recovery_suggestions_; }

    void add_recovery_suggestion(const std::string& suggestion);

private:
    ErrorType error_type_;
    std::vector<std::string> recovery_suggestions_;
};

// Backend-specific exceptions
class BackendException : public YirageException {
public:
    BackendException(const std::string& message, const std::string& backend_name);

    const std::string& backend_name() const { return backend_name_; }

private:
    std::string backend_name_;
};

// Validation exceptions
class ValidationException : public YirageException {
public:
    ValidationException(const std::string& message, const std::vector<std::string>& errors);

    const std::vector<std::string>& validation_errors() const { return validation_errors_; }

private:
    std::vector<std::string> validation_errors_;
};

// Error handling utilities
class ErrorHandler {
public:
    static void set_error_callback(std::function<void(const YirageException&)> callback);
    static void handle_error(const YirageException& exception);

    // Recovery strategies
    static bool try_recovery(const OptimizationException& exception);
    static std::unique_ptr<Optimizer> create_fallback_optimizer(const std::string& failed_backend);
};

} // namespace yirage
```

#### Error Handling Example

```cpp
#include <yirage/exceptions.h>

try {
    auto optimizer = yirage::Optimizer::create("yica");
    auto result = optimizer->optimize(graph);
} catch (const yirage::OptimizationException& e) {
    std::cerr << "Optimization failed: " << e.what() << std::endl;
    std::cerr << "Error type: " << static_cast<int>(e.error_type()) << std::endl;

    std::cerr << "Recovery suggestions:" << std::endl;
    for (const auto& suggestion : e.recovery_suggestions()) {
        std::cerr << "  - " << suggestion << std::endl;
    }

    // Try recovery
    if (yirage::ErrorHandler::try_recovery(e)) {
        // Retry with different configuration
        auto fallback_optimizer = yirage::ErrorHandler::create_fallback_optimizer("yica");
        auto result = fallback_optimizer->optimize(graph);
    }
} catch (const yirage::BackendException& e) {
    std::cerr << "Backend error (" << e.backend_name() << "): " << e.what() << std::endl;

    // Try different backend
    auto cuda_optimizer = yirage::Optimizer::create("cuda");
    auto result = cuda_optimizer->optimize(graph);
}
```

## Build Integration

### CMake Integration

```cmake
# Find YiRage package
find_package(YiRage REQUIRED)

# Create executable
add_executable(my_optimizer main.cpp)

# Link YiRage libraries
target_link_libraries(my_optimizer
    YiRage::Core
    YiRage::YICA
    YiRage::CUDA
)

# Set C++ standard
set_target_properties(my_optimizer PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Optional: Enable specific backends
if(YiRage_YICA_FOUND)
    target_compile_definitions(my_optimizer PRIVATE YIRAGE_ENABLE_YICA)
endif()

if(YiRage_CUDA_FOUND)
    target_compile_definitions(my_optimizer PRIVATE YIRAGE_ENABLE_CUDA)
endif()
```

### pkg-config Integration

```bash
# Compile with pkg-config
g++ -std=c++17 main.cpp $(pkg-config --cflags --libs yirage) -o my_optimizer

# Check available features
pkg-config --modversion yirage
pkg-config --variable=features yirage
```

This comprehensive C++ API reference provides detailed documentation for all major classes, methods, and features of YiRage's C++ interface, with extensive code examples and usage patterns.

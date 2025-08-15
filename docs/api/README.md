# API Documentation

This directory contains comprehensive API reference documentation and usage examples for YICA/YiRage.

## 📖 Documentation Overview

### Current Documentation
- **[Analyzer API](analyzer.md)** - YICA analyzer API reference and usage guide
- **[Python API](python-api.md)** - Complete YiRage Python interface reference
- **[C++ API](cpp-api.md)** - YICA C++ kernel API documentation

### Planned Documentation
- **REST API** - Web service interface documentation
- **Usage Examples** - Various API usage examples and tutorials

## 🔌 API Overview

### Python API

YiRage provides rich Python interfaces supporting various use cases:

```python
import yirage

# Basic optimization
optimizer = yirage.Optimizer()
result = optimizer.superoptimize(backend="yica")

# Advanced configuration
config = yirage.YICAConfig(
    optimization_strategy="throughput_optimal",
    enable_kernel_fusion=True,
    memory_optimization=True
)
result = optimizer.superoptimize(backend="yica", yica_config=config)
```

### C++ API

Low-level C++ interface provides maximum performance and flexibility:

```cpp
#include "yirage/yica_optimizer.h"

// Create optimizer
auto optimizer = yirage::YICAOptimizer::create();

// Execute optimization
auto result = optimizer->optimize(input_graph);

// Access results
std::cout << "Speedup: " << result.speedup << "x" << std::endl;
```

### REST API

Web service interface supports remote invocation:

```bash
# Submit optimization task
curl -X POST http://localhost:8080/api/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "backend": "yica", 
    "model": "...",
    "optimization_level": "aggressive"
  }'

# Query optimization status
curl http://localhost:8080/api/status/{task_id}

# Retrieve results
curl http://localhost:8080/api/results/{task_id}
```

## 📚 API Categories

### 1. Core Optimization APIs

#### Graph Optimization
- **Function**: Computation graph search and optimization
- **Input**: Model definition, optimization parameters
- **Output**: Optimized computation graph

```python
# Graph-level optimization
result = optimizer.optimize_graph(
    graph=computation_graph,
    objectives=["latency", "memory"],
    constraints={"max_memory_mb": 8192}
)
```

#### Operator Optimization
- **Function**: Single operator optimization
- **Input**: Operator definition, input specifications
- **Output**: Optimized operator implementation

```python
# Operator-level optimization
optimized_op = optimizer.optimize_operator(
    operator="attention",
    input_shapes=[(1024, 768), (1024, 768)],
    backend="yica"
)
```

#### End-to-End Optimization
- **Function**: Complete model optimization
- **Input**: Full model, sample inputs
- **Output**: Optimized model with performance metrics

```python
# End-to-end optimization
optimized_model = optimizer.optimize_model(
    model=pytorch_model,
    sample_inputs=sample_data,
    optimization_level="aggressive"
)
```

### 2. Backend Management APIs

#### Backend Selection
- **Function**: Choose and configure optimization backends
- **Supported**: CUDA, Triton, YICA, CPU

```python
# List available backends
backends = yirage.list_backends()

# Select backend
optimizer = yirage.Optimizer(backend="yica")

# Multi-backend comparison
results = yirage.compare_backends(
    model=model,
    backends=["cuda", "triton", "yica"]
)
```

#### Backend Configuration
- **Function**: Backend-specific configuration options
- **Features**: Hardware settings, optimization parameters

```python
# YICA backend configuration
yica_config = yirage.YICAConfig(
    num_dies=8,
    clusters_per_die=4,
    cim_arrays_per_cluster=16,
    optimization_strategy="energy_efficient"
)

# CUDA backend configuration
cuda_config = yirage.CUDAConfig(
    compute_capability="8.6",
    max_shared_memory_kb=96,
    enable_tensor_cores=True
)
```

#### Backend Information
- **Function**: Query backend capabilities and status
- **Output**: Hardware specs, availability, performance characteristics

```python
# Query backend capabilities
capabilities = yirage.get_backend_info("yica")
print(f"Max memory: {capabilities.max_memory_gb}GB")
print(f"Compute units: {capabilities.compute_units}")

# Check backend availability
is_available = yirage.is_backend_available("yica")
```

### 3. Performance Analysis APIs

#### Performance Profiling
- **Function**: Detailed performance analysis and profiling
- **Output**: Timing, memory usage, bottleneck analysis

```python
# Profile model performance
profile = yirage.profile_model(
    model=model,
    inputs=sample_inputs,
    backend="yica",
    detailed=True
)

print(f"Total time: {profile.total_time_ms}ms")
print(f"Memory peak: {profile.peak_memory_mb}MB")
print(f"Bottlenecks: {profile.bottlenecks}")
```

#### Benchmark Testing
- **Function**: Standardized performance benchmarks
- **Output**: Comparative performance metrics

```python
# Run benchmark suite
benchmark_results = yirage.run_benchmarks(
    model=model,
    backends=["pytorch", "yica"],
    iterations=100
)

# Custom benchmark
custom_benchmark = yirage.Benchmark(
    name="custom_attention",
    model=attention_model,
    inputs=attention_inputs
)
results = custom_benchmark.run(backends=["yica"])
```

#### Performance Comparison
- **Function**: Multi-backend performance comparison
- **Output**: Detailed comparison reports

```python
# Compare performance across backends
comparison = yirage.compare_performance(
    model=model,
    backends=["cuda", "triton", "yica"],
    metrics=["latency", "throughput", "energy"]
)

# Generate comparison report
report = yirage.generate_report(
    comparison_results=comparison,
    output_format="html",
    include_charts=True
)
```

### 4. Configuration Management APIs

#### Configuration Loading
- **Function**: Load configuration from files or environment
- **Formats**: JSON, YAML, environment variables

```python
# Load from file
config = yirage.load_config("config.json")

# Load from environment
config = yirage.load_config_from_env(prefix="YICA_")

# Merge configurations
merged_config = yirage.merge_configs([file_config, env_config])
```

#### Configuration Validation
- **Function**: Validate configuration parameters
- **Output**: Validation results and error messages

```python
# Validate configuration
validation_result = yirage.validate_config(config)

if not validation_result.is_valid:
    print("Configuration errors:")
    for error in validation_result.errors:
        print(f"  - {error}")
```

#### Dynamic Configuration
- **Function**: Runtime configuration updates
- **Features**: Hot reloading, parameter tuning

```python
# Update configuration at runtime
optimizer.update_config({
    "optimization_level": "aggressive",
    "memory_limit_mb": 16384
})

# Enable auto-tuning
optimizer.enable_auto_tuning(
    parameters=["batch_size", "memory_allocation"],
    target_metric="throughput"
)
```

### 5. Debugging and Logging APIs

#### Logging Management
- **Function**: Control logging levels and output
- **Features**: Structured logging, multiple outputs

```python
# Configure logging
yirage.configure_logging(
    level="DEBUG",
    output_file="yirage.log",
    format="structured",
    enable_profiling=True
)

# Context-specific logging
with yirage.log_context("optimization"):
    result = optimizer.optimize(model)
```

#### Debug Information
- **Function**: Access detailed debugging information
- **Output**: Optimization traces, intermediate results

```python
# Enable debug mode
optimizer = yirage.Optimizer(debug_mode=True)
result = optimizer.optimize(model)

# Access debug information
debug_info = result.debug_info
print("Optimization steps:", debug_info.optimization_steps)
print("Intermediate graphs:", debug_info.intermediate_graphs)
```

#### Error Handling
- **Function**: Comprehensive error handling and recovery
- **Features**: Error classification, recovery suggestions

```python
try:
    result = optimizer.optimize(model)
except yirage.OptimizationError as e:
    print(f"Optimization failed: {e.message}")
    print(f"Error type: {e.error_type}")
    print(f"Suggestions: {e.recovery_suggestions}")
```

## 🛠️ Advanced Usage Patterns

### Batch Processing
```python
# Batch optimization
batch_processor = yirage.BatchProcessor(
    backend="yica",
    parallel_jobs=4,
    output_dir="optimized_models"
)

results = batch_processor.optimize_batch([
    "model1.py", "model2.py", "model3.py"
])
```

### Pipeline Integration
```python
# Create optimization pipeline
pipeline = yirage.Pipeline([
    yirage.stages.ModelLoader(),
    yirage.stages.GraphOptimizer(backend="yica"),
    yirage.stages.PerformanceValidator(),
    yirage.stages.ModelExporter()
])

# Execute pipeline
result = pipeline.run(input_model="model.py")
```

### Custom Extensions
```python
# Register custom optimization strategy
@yirage.register_strategy("my_custom_strategy")
class CustomStrategy(yirage.OptimizationStrategy):
    def optimize(self, graph):
        # Custom optimization logic
        return optimized_graph

# Use custom strategy
optimizer = yirage.Optimizer(strategy="my_custom_strategy")
```

## 📋 API Reference Quick Links

### Python API
- **Core Classes** (planned documentation)
- **Optimization Methods** (planned documentation)
- **Backend Interfaces** (planned documentation)
- **Configuration System** (planned documentation)
- **Utilities** (planned documentation)

### C++ API
- **Core Headers** (planned documentation)
- **Optimization Engine** (planned documentation)
- **Memory Management** (planned documentation)
- **Performance Tools** (planned documentation)

### REST API
- **Authentication** (planned documentation)
- **Optimization Endpoints** (planned documentation)
- **Status and Monitoring** (planned documentation)
- **Error Handling** (planned documentation)

## 🔗 Related Documentation

- **Getting Started** - Basic concepts and setup
- **Architecture** - System architecture overview
- **Usage Guide** - Comprehensive usage examples
- **Development** - Development environment setup

## 📞 API Support

### Getting Help with APIs
1. Check the specific API documentation
2. Review [usage examples](../USAGE.md)
3. Search [GitHub issues](https://github.com/yica-ai/yica-yirage/issues)
4. Join our [Discord community](https://discord.gg/yica-yirage)

### Reporting API Issues
When reporting API-related issues, please include:
- API version (`yirage.api_version`)
- Code example reproducing the issue
- Expected vs actual behavior
- System information (`yirage.system_info()`)

---

*The API documentation is continuously updated. For the latest information, check the source code documentation and examples.*
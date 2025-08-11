# YICA/YiRage Usage Guide

This comprehensive guide covers everything you need to know about using YICA/YiRage effectively.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- CMake 3.16 or higher
- C++17 compatible compiler
- 8GB RAM minimum (16GB recommended)

### Basic Installation
```bash
# Clone the repository
git clone https://github.com/your-org/yica-yirage.git
cd yica-yirage

# Build the project
mkdir build && cd build
cmake ..
make -j$(nproc)

# Install Python package
cd ../yirage/python
pip install -e .
```

### Docker Quick Start
```bash
# One-click deployment
./scripts/docker_yica_deployment.sh

# Access via web VNC
# URL: http://localhost:6080 (password: yica)
```

## 📖 Basic Usage

### Python API

#### Simple Optimization
```python
import yirage

# Create optimizer
optimizer = yirage.Optimizer()

# Basic optimization
result = optimizer.superoptimize(
    backend="yica",
    input_code=your_model
)

print(f"Performance improvement: {result.speedup}x")
```

#### Advanced Configuration
```python
import yirage

# Configure optimization parameters
config = yirage.OptimizationConfig(
    target_architecture="yica_v2",
    objectives=["latency", "energy_efficiency", "memory_efficiency"],
    objective_weights=[0.4, 0.3, 0.3],
    search_algorithms=["genetic", "bayesian"],
    max_search_time=3600,  # 1 hour
    convergence_threshold=0.01
)

# Create optimizer with configuration
optimizer = yirage.YirageOptimizer(config)

# Perform optimization
result = optimizer.optimize(
    input_code=attention_kernel,
    baseline_performance=torch_baseline,
    validation_data=test_inputs
)

# Display results
print(f"Speedup: {result.speedup}x")
print(f"Energy reduction: {result.energy_reduction}%")
print(f"Memory efficiency: {result.memory_efficiency}%")
```

### Command Line Interface

#### Basic Commands
```bash
# Basic optimization
yirage optimize --input model.py --target yica --output optimized_model.py

# Advanced configuration
yirage optimize \
  --input llama_attention.py \
  --target yica \
  --arch-config yica_v2.json \
  --objectives "latency,energy,memory" \
  --weights "0.4,0.3,0.3" \
  --search-budget 5000 \
  --output optimized_attention.py \
  --report optimization_report.html

# Batch optimization
yirage batch-optimize \
  --model-dir ./models \
  --target yica \
  --parallel-jobs 8 \
  --output-dir ./optimized_models
```

#### Performance Analysis
```bash
# Profile performance
yirage profile --input model.py --backend yica

# Compare backends
yirage compare \
  --input model.py \
  --backends cuda,triton,yica \
  --output comparison_report.html

# Benchmark against baselines
yirage benchmark \
  --input model.py \
  --baseline pytorch \
  --iterations 100
```

## 🏗️ Architecture-Specific Usage

### YICA Backend Configuration

#### Hardware Configuration
```python
# Configure YICA hardware settings
yica_config = yirage.YICAConfig(
    num_dies=8,
    clusters_per_die=4,
    cim_arrays_per_cluster=16,
    spm_size_mb=64,
    enable_cross_die_communication=True,
    optimization_strategy="throughput_optimal"
)

optimizer = yirage.Optimizer(backend="yica", yica_config=yica_config)
```

#### Memory Management
```python
# Configure memory hierarchy
memory_config = yirage.MemoryConfig(
    register_file_size=32,  # KB
    spm_levels=3,
    spm_sizes=[256, 1024, 4096],  # KB per level
    dram_bandwidth_gbps=1000,
    enable_memory_compression=True
)

yica_config = yirage.YICAConfig(memory_config=memory_config)
```

### Multi-Backend Optimization

#### Backend Comparison
```python
# Test multiple backends
results = {}
backends = ["cuda", "triton", "yica"]

for backend in backends:
    optimizer = yirage.Optimizer(backend=backend)
    result = optimizer.optimize(model)
    results[backend] = result

# Find best backend
best_backend = max(results.keys(), key=lambda k: results[k].speedup)
print(f"Best backend: {best_backend} ({results[best_backend].speedup}x speedup)")
```

#### Adaptive Backend Selection
```python
# Automatic backend selection
optimizer = yirage.AdaptiveOptimizer()
result = optimizer.optimize(
    model,
    hardware_constraints={"memory_gb": 16, "compute_capability": "sm_80"},
    performance_requirements={"max_latency_ms": 10}
)

print(f"Selected backend: {result.selected_backend}")
print(f"Performance: {result.performance_metrics}")
```

## 🎯 Use Cases and Examples

### Large Language Model Optimization

#### LLaMA Attention Optimization
```python
import torch
import yirage

# Define LLaMA attention module
class LLaMAAttention(torch.nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = torch.nn.Linear(dim, dim, bias=False)
        self.wk = torch.nn.Linear(dim, dim, bias=False)
        self.wv = torch.nn.Linear(dim, dim, bias=False)
        self.wo = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # Attention computation
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        # ... attention logic
        return output

# Optimize with YiRage
attention = LLaMAAttention(4096, 32)
optimizer = yirage.Optimizer(backend="yica")

optimized_attention = optimizer.optimize(
    attention,
    sample_input=torch.randn(1, 2048, 4096),
    optimization_level="aggressive"
)

# Benchmark performance
baseline_time = benchmark_model(attention, test_input)
optimized_time = benchmark_model(optimized_attention, test_input)
speedup = baseline_time / optimized_time

print(f"Attention optimization speedup: {speedup:.2f}x")
```

### Computer Vision Model Optimization

#### ResNet Block Optimization
```python
import torchvision
import yirage

# Load ResNet model
model = torchvision.models.resnet50(pretrained=True)

# Configure for vision workloads
vision_config = yirage.OptimizationConfig(
    target_architecture="yica_v2",
    workload_type="computer_vision",
    batch_size_range=[1, 32],
    input_resolution=[224, 224],
    optimization_objectives=["throughput", "memory_efficiency"]
)

# Optimize model
optimizer = yirage.Optimizer(config=vision_config)
optimized_model = optimizer.optimize(
    model,
    sample_input=torch.randn(8, 3, 224, 224)
)

# Evaluate performance
throughput_improvement = evaluate_throughput(model, optimized_model)
print(f"Throughput improvement: {throughput_improvement:.2f}x")
```

### Custom Operator Development

#### Creating Custom Operators
```python
import yirage

# Define custom operator
@yirage.custom_operator
def fused_gelu_dropout(x, dropout_rate=0.1):
    """Fused GELU activation with dropout"""
    # GELU activation
    gelu_out = 0.5 * x * (1 + torch.tanh(
        torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
    ))
    
    # Dropout
    if dropout_rate > 0:
        dropout_mask = torch.rand_like(gelu_out) > dropout_rate
        gelu_out = gelu_out * dropout_mask / (1 - dropout_rate)
    
    return gelu_out

# Optimize custom operator
optimizer = yirage.Optimizer(backend="yica")
optimized_op = optimizer.optimize(
    fused_gelu_dropout,
    sample_input=torch.randn(1024, 4096),
    fusion_opportunities=["activation_dropout"]
)

# Use optimized operator
result = optimized_op(input_tensor, dropout_rate=0.1)
```

## 🔧 Configuration and Tuning

### Performance Tuning

#### Optimization Objectives
```python
# Configure optimization objectives
config = yirage.OptimizationConfig(
    objectives={
        "latency": 0.4,        # 40% weight on latency
        "throughput": 0.3,     # 30% weight on throughput
        "energy": 0.2,         # 20% weight on energy efficiency
        "memory": 0.1          # 10% weight on memory usage
    },
    constraints={
        "max_memory_mb": 8192,
        "max_latency_ms": 100,
        "min_accuracy": 0.99
    }
)
```

#### Search Algorithm Configuration
```python
# Configure search algorithms
search_config = yirage.SearchConfig(
    algorithms=["genetic", "simulated_annealing", "bayesian"],
    genetic_config={
        "population_size": 100,
        "generations": 500,
        "mutation_rate": 0.1,
        "crossover_rate": 0.8
    },
    bayesian_config={
        "acquisition_function": "expected_improvement",
        "kernel": "matern52",
        "n_initial_points": 20
    },
    parallel_jobs=8,
    max_search_time_hours=2
)
```

### Debugging and Profiling

#### Enable Debug Mode
```python
# Enable detailed logging
yirage.set_log_level("DEBUG")
yirage.enable_profiling(detailed=True)

# Run optimization with debugging
optimizer = yirage.Optimizer(debug_mode=True)
result = optimizer.optimize(model)

# Access debug information
print("Optimization trace:", result.debug_info.optimization_trace)
print("Performance breakdown:", result.debug_info.performance_breakdown)
print("Memory usage:", result.debug_info.memory_usage)
```

#### Performance Profiling
```bash
# Profile with detailed metrics
yirage profile \
  --input model.py \
  --backend yica \
  --detailed-metrics \
  --output-format json \
  --output profile_results.json

# Visualize profiling results
yirage visualize --profile profile_results.json --output profile_report.html
```

## 🚨 Troubleshooting

### Common Issues

#### Installation Problems
```bash
# Check system requirements
yirage system-check

# Verify installation
yirage --version
python -c "import yirage; print(yirage.__version__)"

# Rebuild if needed
cd build && make clean && make -j$(nproc)
```

#### Runtime Issues
```bash
# Check backend availability
yirage backends --list

# Verify hardware compatibility
yirage hardware-check --backend yica

# Run diagnostics
yirage diagnose --verbose
```

#### Performance Issues
```python
# Enable performance debugging
yirage.set_debug_level("PERFORMANCE")

# Check for common bottlenecks
performance_report = yirage.analyze_performance(model)
print("Bottlenecks:", performance_report.bottlenecks)
print("Recommendations:", performance_report.recommendations)
```

### Getting Help

#### Community Resources
- **Documentation**: [https://yica-yirage.readthedocs.io/](https://yica-yirage.readthedocs.io/)
- **GitHub Issues**: [https://github.com/your-org/yica-yirage/issues](https://github.com/your-org/yica-yirage/issues)
- **Discord Community**: [https://discord.gg/yica-yirage](https://discord.gg/yica-yirage)

#### Support Channels
- **Bug Reports**: Use GitHub Issues with the `bug` label
- **Feature Requests**: Use GitHub Issues with the `enhancement` label
- **Questions**: Use GitHub Discussions or Discord

## 📚 Advanced Topics

### Custom Backend Development
See **Backend Development Guide** (planned documentation) for creating custom optimization backends.

### Integration with Other Frameworks
See **Integration Guide** (planned documentation) for integrating YiRage with PyTorch, TensorFlow, and other frameworks.

### Production Deployment
See **Production Deployment Guide** (planned documentation) for deploying optimized models in production environments.

---

**Need Help?** Check our [FAQ](getting-started/quick-reference.md) or [contact support](mailto:support@yica-yirage.com).
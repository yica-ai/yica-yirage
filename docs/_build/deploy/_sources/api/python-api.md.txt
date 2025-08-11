# Python API Reference

This document provides comprehensive reference documentation for the YiRage Python API based on the actual source code implementation.

## Module Structure

YiRage is organized into several key modules:

```python
import yirage

# Core modules (always available)
from yirage import core, global_config, utils
from yirage.version import __version__  # "1.0.1"

# YICA-specific modules (conditional availability)
from yirage import yica_advanced           # YICA_ADVANCED_AVAILABLE
from yirage import yica_performance_monitor # YICA_MONITOR_AVAILABLE  
from yirage import yica_real_optimizer     # YICA_OPTIMIZER_AVAILABLE

# Optional modules
from yirage import yica_auto_tuner, yica_distributed, yica_llama_optimizer
from yirage import yica_pytorch_backend, visualizer, profiler, triton_profiler
```

## Core Classes

### `yirage.core.YICACore`

The main YICA core interface providing unified access to hardware abstraction and optimization.

#### Constructor

```python
class YICACore:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize YICA Core Interface
        
        Args:
            config: Configuration dictionary with options:
                - backend_mode (str): 'cpu', 'gpu', 'yica' (default: 'cpu')
                - num_cim_arrays (int): Number of CIM arrays (default: 8)
                - spm_size (int): SPM size in bytes (default: 128MB)
        """
```

**Example:**
```python
import yirage

# Basic initialization
core = yirage.core.YICACore()

# Advanced configuration
config = {
    'backend_mode': 'yica',
    'num_cim_arrays': 16,
    'spm_size': 256 * 1024 * 1024,  # 256MB
}
core = yirage.core.YICACore(config)

print(f"Backend type: {core.backend_type}")  # 'native' or 'fallback'
print(f"CIM arrays: {core.num_cim_arrays}")
```

#### Methods

##### `get_yica_info() -> Dict[str, Any]`

Get comprehensive YICA system information.

```python
info = core.get_yica_info()
print(f"Version: {info['version']}")
print(f"Available backends: {info['available_backends']}")
print(f"Hardware support: {info['hardware_support']}")
```

### `yirage.yica_real_optimizer.YICAKernelOptimizer`

Real YICA kernel optimizer implementation (not simulation).

#### Constructor

```python
@dataclass 
class YICAHardwareConfig:
    """YICA Hardware Configuration"""
    num_cim_arrays: int = 4
    cim_array_size: Tuple[int, int] = (256, 256)
    spm_size_kb: int = 512
    memory_bandwidth_gbps: float = 1000.0
    compute_capability: float = 25.0  # TOPS per CIM array
    enable_mixed_precision: bool = True
    enable_data_compression: bool = True

class YICAKernelOptimizer:
    def __init__(self, hardware_config: YICAHardwareConfig):
        """Initialize YICA kernel optimizer with hardware configuration"""
```

**Example:**
```python
from yirage.yica_real_optimizer import YICAKernelOptimizer, YICAHardwareConfig

# Create hardware configuration
hw_config = YICAHardwareConfig(
    num_cim_arrays=8,
    cim_array_size=(512, 512),
    spm_size_kb=1024,
    memory_bandwidth_gbps=2000.0,
    compute_capability=50.0
)

# Initialize optimizer
optimizer = YICAKernelOptimizer(hw_config)
```

#### Methods

##### `optimize_matrix_multiplication(graph, input_shapes: List[Tuple[int, ...]]) -> Any`

Optimize matrix multiplication operations using YICA CIM arrays.

```python
# Example: Optimize a matrix multiplication
input_shapes = [(1024, 512), (512, 256)]  # A: 1024x512, B: 512x256
result = optimizer.optimize_matrix_multiplication(graph, input_shapes)

print(f"CIM parallelization strategy: {result.cim_strategy}")
print(f"SPM allocation plan: {result.spm_allocation}")
```

##### `optimize_attention_mechanism(graph, config: Dict) -> Any`

Optimize attention mechanisms for transformer models.

```python
attention_config = {
    'sequence_length': 2048,
    'hidden_dim': 4096,
    'num_heads': 32,
    'head_dim': 128
}
result = optimizer.optimize_attention_mechanism(graph, attention_config)
```

### `yirage.yica_backend_integration.YICAMatMulKernel`

YICA matrix multiplication kernel implementation based on YIS instruction set.

#### Constructor

```python
from yirage.yica_backend_integration import YICAMatMulKernel, YICAKernelConfig, YISInstructionType

config = YICAKernelConfig(
    yis_instruction_type=YISInstructionType.YISMMA,
    use_spm=True,
    enable_cim_parallel=True
)
kernel = YICAMatMulKernel(config)
```

#### Methods

##### `estimate_performance(A: torch.Tensor, B: torch.Tensor) -> Dict[str, float]`

Estimate performance metrics for matrix multiplication.

```python
import torch

A = torch.randn(1024, 512, dtype=torch.float16)
B = torch.randn(512, 256, dtype=torch.float16)

perf_metrics = kernel.estimate_performance(A, B)
print(f"Estimated latency: {perf_metrics['estimated_latency_ms']:.2f} ms")
print(f"SPM utilization: {perf_metrics['smp_utilization']:.2f}")
print(f"CIM efficiency: {perf_metrics['cim_efficiency']:.2f}")
```

##### `generate_yis_instructions(A: torch.Tensor, B: torch.Tensor) -> List[str]`

Generate YIS (YICA Instruction Set) assembly instructions.

```python
instructions = kernel.generate_yis_instructions(A, B)
for i, instruction in enumerate(instructions):
    print(f"{i:2d}: {instruction}")

# Example output:
# 0: // Load Matrix A (1024x512) from DRAM to SPM
# 1: yis.ecopy.g2spm a_spm, a_dram, 1048576, TROW, WG
# 2: // Load Matrix B (512x256) from DRAM to SPM  
# 3: yis.ecopy.g2spm b_spm, b_dram, 262144, TCOL, WG
# 4: yis.mma.32x32x32 c_spm[0:32][0:32], a_spm[0:32][0:32], b_spm[0:32][0:32], NONACC, SPM
# ...
```

##### `execute(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor`

Execute the matrix multiplication on YICA hardware.

```python
# Execute matrix multiplication
C = kernel.execute(A, B)
print(f"Result shape: {C.shape}")  # torch.Size([1024, 256])

# Verify correctness
expected = torch.matmul(A, B)
torch.allclose(C, expected, atol=1e-3)  # True
```

## YIS Instruction Set Integration

### `yirage.yica_backend_integration.YISInstructionType`

YIS instruction types supported by YICA architecture.

```python
from yirage.yica_backend_integration import YISInstructionType

class YISInstructionType(Enum):
    YISECOPY = "external_copy"     # External copy instruction
    YISICOPY = "internal_copy"     # Internal copy instruction  
    YISMMA = "matrix_multiply"     # Matrix multiplication instruction
    YISSYNC = "synchronization"    # Synchronization instruction
    YISCONTROL = "control_flow"    # Control flow instruction
```

### Memory Hierarchy Types

```python
from yirage.yica_backend_integration import YICAMemoryType

class YICAMemoryType(Enum):
    REGISTER_FILE = "register"     # Register file (fastest)
    SPM_LEVEL1 = "spm_l1"         # SPM Level 1 cache
    SPM_LEVEL2 = "spm_l2"         # SPM Level 2 cache  
    DRAM = "dram"                 # Main DRAM memory
```

## Advanced Features

### Auto-Tuning

```python
from yirage.yica_auto_tuner import YICAAutoTuner, YICAConfig

# Configure auto-tuner
yica_config = YICAConfig(
    cim_array_count=8,
    spm_size_per_die_mb=64,
    memory_bandwidth_gbps=1000,
    target_precision="fp16"
)

auto_tuner = YICAAutoTuner(yica_config)

# Auto-tune for specific workload
model = torch.nn.Linear(4096, 4096)
tuned_config = auto_tuner.tune_for_model(model)
print(f"Optimal tile size: {tuned_config.optimal_tile_size}")
```

### Performance Monitoring

```python
from yirage.yica_performance_monitor import YICAPerformanceMonitor

# Enable performance monitoring
monitor = YICAPerformanceMonitor()
monitor.start_monitoring()

# Run operations
result = kernel.execute(A, B)

# Get performance metrics
metrics = monitor.stop_monitoring()
print(f"CIM utilization: {metrics.cim_utilization:.2f}%")
print(f"SPM hit rate: {metrics.spm_hit_rate:.2f}%")
print(f"Memory bandwidth: {metrics.memory_bandwidth_gbps:.1f} GB/s")
```

### PyTorch Backend Integration

```python
from yirage.yica_pytorch_backend import YICAPyTorchBackend

# Register YICA as PyTorch backend
backend = YICAPyTorchBackend()
torch.utils.rename_privateuse1_backend("yica")

# Use YICA device in PyTorch
device = torch.device("yica:0")
model = model.to(device)
input_tensor = input_tensor.to(device)

# Operations automatically use YICA optimization
output = model(input_tensor)
```

## Error Handling

### Exception Classes

```python
from yirage.yica_backend_integration import YICAError, YICAHardwareError

try:
    result = kernel.execute(A, B)
except YICAHardwareError as e:
    print(f"Hardware error: {e}")
    print(f"Error code: {e.error_code}")
    print(f"Recovery suggestions: {e.recovery_suggestions}")
except YICAError as e:
    print(f"General YICA error: {e}")
```

## Utility Functions

### System Information

```python
# Check module availability
print(f"YICA Core Available: {yirage.YICA_CORE_AVAILABLE}")
print(f"YICA Advanced Available: {yirage.YICA_ADVANCED_AVAILABLE}")
print(f"YICA Monitor Available: {yirage.YICA_MONITOR_AVAILABLE}")
print(f"YICA Optimizer Available: {yirage.YICA_OPTIMIZER_AVAILABLE}")

# Check dependencies
print(f"PyTorch Available: {yirage.TORCH_AVAILABLE}")
print(f"NumPy Available: {yirage.NUMPY_AVAILABLE}")
print(f"Z3 Available: {yirage.Z3_AVAILABLE}")
```

### Create Optimizer Instance

```python
def create_yica_optimizer(config=None):
    """Create a YICA optimizer instance with automatic fallback"""
    if yirage.YICA_OPTIMIZER_AVAILABLE:
        from yirage.yica_real_optimizer import YICAKernelOptimizer, YICAHardwareConfig
        hw_config = YICAHardwareConfig(**(config or {}))
        return YICAKernelOptimizer(hw_config)
    else:
        raise RuntimeError("YICA optimizer not available")

# Usage
optimizer = yirage.create_yica_optimizer({
    'num_cim_arrays': 8,
    'spm_size_kb': 1024
})
```

#### Constructor

```python
class Optimizer:
    def __init__(
        self,
        backend: str = "auto",
        config: Optional[OptimizationConfig] = None,
        debug_mode: bool = False,
        log_level: str = "INFO"
    )
```

**Parameters:**
- `backend` (str): Backend to use ("yica", "cuda", "triton", "auto")
- `config` (OptimizationConfig, optional): Optimization configuration
- `debug_mode` (bool): Enable debug mode for detailed logging
- `log_level` (str): Logging level ("DEBUG", "INFO", "WARNING", "ERROR")

**Example:**
```python
import yirage

# Basic optimizer
optimizer = yirage.Optimizer(backend="yica")

# Advanced optimizer with configuration
config = yirage.OptimizationConfig(
    optimization_level="aggressive",
    max_search_time=3600,
    enable_kernel_fusion=True
)
optimizer = yirage.Optimizer(backend="yica", config=config, debug_mode=True)
```

#### Methods

##### `optimize(model, **kwargs) -> OptimizationResult`

Optimize a model using the configured backend.

**Parameters:**
- `model`: Model to optimize (PyTorch Module, TensorFlow Model, or ONNX Graph)
- `sample_input`: Sample input for shape inference
- `optimization_level` (str): "conservative", "balanced", "aggressive"
- `objectives` (List[str]): Optimization objectives ["latency", "throughput", "memory", "energy"]
- `constraints` (Dict): Optimization constraints

**Returns:**
- `OptimizationResult`: Optimization results and metadata

**Example:**
```python
import torch
import yirage

# Create model
model = torch.nn.Sequential(
    torch.nn.Linear(784, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10)
)

# Optimize model
optimizer = yirage.Optimizer(backend="yica")
result = optimizer.optimize(
    model,
    sample_input=torch.randn(32, 784),
    optimization_level="aggressive",
    objectives=["latency", "memory"],
    constraints={"max_memory_mb": 4096}
)

print(f"Speedup: {result.speedup:.2f}x")
print(f"Memory reduction: {result.memory_reduction:.1f}%")
print(f"Optimized model: {result.optimized_model}")
```

##### `profile(model, **kwargs) -> ProfileResult`

Profile a model's performance characteristics.

**Parameters:**
- `model`: Model to profile
- `sample_input`: Input for profiling
- `iterations` (int): Number of profiling iterations (default: 100)
- `warmup_iterations` (int): Warmup iterations (default: 10)
- `detailed` (bool): Enable detailed profiling (default: False)

**Returns:**
- `ProfileResult`: Detailed profiling information

**Example:**
```python
# Profile model performance
profile_result = optimizer.profile(
    model,
    sample_input=torch.randn(32, 784),
    iterations=1000,
    detailed=True
)

print(f"Average latency: {profile_result.avg_latency_ms:.2f}ms")
print(f"Throughput: {profile_result.throughput_samples_per_sec:.0f} samples/sec")
print(f"Memory usage: {profile_result.peak_memory_mb:.1f}MB")
print(f"Bottlenecks: {profile_result.bottlenecks}")
```

##### `compare_backends(model, backends, **kwargs) -> ComparisonResult`

Compare performance across multiple backends.

**Parameters:**
- `model`: Model to compare
- `backends` (List[str]): List of backends to compare
- `sample_input`: Input for comparison
- `iterations` (int): Number of iterations per backend

**Returns:**
- `ComparisonResult`: Performance comparison results

**Example:**
```python
# Compare backends
comparison = optimizer.compare_backends(
    model,
    backends=["cuda", "triton", "yica"],
    sample_input=torch.randn(32, 784),
    iterations=100
)

for backend, metrics in comparison.results.items():
    print(f"{backend}: {metrics.latency_ms:.2f}ms, {metrics.throughput:.0f} samples/sec")

# Get best backend
best_backend = comparison.best_backend("latency")
print(f"Best backend for latency: {best_backend}")
```

### `yirage.OptimizationConfig`

Configuration class for optimization parameters.

#### Constructor

```python
class OptimizationConfig:
    def __init__(
        self,
        optimization_level: str = "balanced",
        max_search_time: int = 1800,
        parallel_jobs: int = None,
        enable_kernel_fusion: bool = True,
        enable_memory_optimization: bool = True,
        target_precision: str = "fp32",
        **kwargs
    )
```

**Parameters:**
- `optimization_level` (str): "conservative", "balanced", "aggressive"
- `max_search_time` (int): Maximum optimization time in seconds
- `parallel_jobs` (int): Number of parallel optimization jobs
- `enable_kernel_fusion` (bool): Enable kernel fusion optimizations
- `enable_memory_optimization` (bool): Enable memory optimizations
- `target_precision` (str): Target precision ("fp16", "fp32", "int8")

**Example:**
```python
# Create optimization configuration
config = yirage.OptimizationConfig(
    optimization_level="aggressive",
    max_search_time=7200,  # 2 hours
    parallel_jobs=8,
    enable_kernel_fusion=True,
    enable_memory_optimization=True,
    target_precision="fp16",
    
    # Backend-specific options
    yica_config={
        "num_dies": 8,
        "optimization_strategy": "throughput_optimal"
    },
    
    # Optimization objectives
    objectives={
        "latency": 0.4,
        "throughput": 0.3,
        "memory": 0.2,
        "energy": 0.1
    },
    
    # Constraints
    constraints={
        "max_memory_mb": 8192,
        "max_latency_ms": 100,
        "min_accuracy": 0.99
    }
)
```

#### Methods

##### `to_dict() -> Dict`

Convert configuration to dictionary.

##### `from_dict(config_dict: Dict) -> OptimizationConfig`

Create configuration from dictionary.

##### `save(filepath: str)`

Save configuration to file.

##### `load(filepath: str) -> OptimizationConfig`

Load configuration from file.

### `yirage.OptimizationResult`

Result class containing optimization outcomes.

#### Attributes

```python
class OptimizationResult:
    optimized_model: Any              # Optimized model
    speedup: float                    # Performance speedup ratio
    memory_reduction: float           # Memory usage reduction percentage
    energy_reduction: float           # Energy consumption reduction
    accuracy_change: float            # Accuracy change (positive = improvement)
    optimization_time: float          # Time taken for optimization
    backend_used: str                 # Backend used for optimization
    
    # Detailed metrics
    original_metrics: PerformanceMetrics
    optimized_metrics: PerformanceMetrics
    
    # Optimization metadata
    optimization_strategy: str
    applied_optimizations: List[str]
    warnings: List[str]
    debug_info: Optional[DebugInfo]
```

#### Methods

##### `save_model(filepath: str)`

Save optimized model to file.

##### `generate_report(output_path: str, format: str = "html")`

Generate optimization report.

**Example:**
```python
# Use optimization result
result = optimizer.optimize(model)

# Access metrics
print(f"Original latency: {result.original_metrics.latency_ms:.2f}ms")
print(f"Optimized latency: {result.optimized_metrics.latency_ms:.2f}ms")
print(f"Speedup: {result.speedup:.2f}x")

# Save optimized model
result.save_model("optimized_model.pth")

# Generate report
result.generate_report("optimization_report.html", format="html")

# Access applied optimizations
print("Applied optimizations:")
for opt in result.applied_optimizations:
    print(f"  - {opt}")
```

## Backend-Specific APIs

### YICA Backend

#### `yirage.YICAConfig`

Configuration specific to YICA backend.

```python
class YICAConfig:
    def __init__(
        self,
        num_dies: int = 8,
        clusters_per_die: int = 4,
        cim_arrays_per_cluster: int = 16,
        spm_size_mb: int = 64,
        optimization_strategy: str = "balanced",
        enable_cross_die_communication: bool = True,
        memory_allocation_strategy: str = "dynamic"
    )
```

**Example:**
```python
# YICA-specific configuration
yica_config = yirage.YICAConfig(
    num_dies=8,
    clusters_per_die=4,
    cim_arrays_per_cluster=16,
    spm_size_mb=128,
    optimization_strategy="throughput_optimal",
    enable_cross_die_communication=True
)

# Use with optimizer
config = yirage.OptimizationConfig(yica_config=yica_config)
optimizer = yirage.Optimizer(backend="yica", config=config)
```

#### `yirage.YICAOptimizer`

Specialized optimizer for YICA backend.

```python
class YICAOptimizer(Optimizer):
    def __init__(self, yica_config: YICAConfig = None):
        super().__init__(backend="yica")
        self.yica_config = yica_config or YICAConfig()
    
    def optimize_for_cim(self, model, **kwargs):
        """Optimize specifically for compute-in-memory architecture."""
        pass
    
    def analyze_memory_hierarchy(self, model):
        """Analyze memory access patterns for YICA hierarchy."""
        pass
```

### CUDA Backend

#### `yirage.CUDAConfig`

```python
class CUDAConfig:
    def __init__(
        self,
        compute_capability: str = "auto",
        max_shared_memory_kb: int = 96,
        enable_tensor_cores: bool = True,
        enable_mixed_precision: bool = False,
        optimization_flags: List[str] = None
    )
```

## Utility Functions

### Performance Analysis

#### `yirage.benchmark_model(model, input_data, **kwargs) -> BenchmarkResult`

Benchmark model performance.

```python
def benchmark_model(
    model,
    input_data,
    iterations: int = 100,
    warmup_iterations: int = 10,
    device: str = "auto"
) -> BenchmarkResult:
    """
    Benchmark model performance.
    
    Args:
        model: Model to benchmark
        input_data: Input data for benchmarking
        iterations: Number of benchmark iterations
        warmup_iterations: Number of warmup iterations
        device: Device to run benchmark on
    
    Returns:
        BenchmarkResult with timing and memory statistics
    """
```

**Example:**
```python
# Benchmark original model
original_benchmark = yirage.benchmark_model(
    original_model,
    test_input,
    iterations=1000
)

# Benchmark optimized model
optimized_benchmark = yirage.benchmark_model(
    optimized_model,
    test_input,
    iterations=1000
)

# Compare results
speedup = original_benchmark.avg_latency / optimized_benchmark.avg_latency
print(f"Measured speedup: {speedup:.2f}x")
```

#### `yirage.profile_memory_usage(model, input_data) -> MemoryProfile`

Profile memory usage patterns.

```python
def profile_memory_usage(model, input_data) -> MemoryProfile:
    """
    Profile memory usage during model execution.
    
    Returns:
        MemoryProfile with detailed memory statistics
    """
```

### Model Validation

#### `yirage.validate_optimization(original_model, optimized_model, test_data, **kwargs) -> ValidationResult`

Validate optimization correctness.

```python
def validate_optimization(
    original_model,
    optimized_model,
    test_data,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    max_samples: int = 1000
) -> ValidationResult:
    """
    Validate that optimization preserves model correctness.
    
    Args:
        original_model: Original model
        optimized_model: Optimized model
        test_data: Test dataset
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        max_samples: Maximum number of samples to test
    
    Returns:
        ValidationResult with correctness statistics
    """
```

**Example:**
```python
# Validate optimization
validation = yirage.validate_optimization(
    original_model,
    result.optimized_model,
    test_dataset,
    rtol=1e-4,
    atol=1e-6
)

print(f"Validation passed: {validation.is_valid}")
print(f"Max difference: {validation.max_difference:.6f}")
print(f"Mean difference: {validation.mean_difference:.6f}")
print(f"Samples tested: {validation.samples_tested}")
```

## Advanced Features

### Custom Optimization Strategies

#### Creating Custom Strategies

```python
from yirage.strategies import OptimizationStrategy

class CustomStrategy(OptimizationStrategy):
    def __init__(self, **kwargs):
        super().__init__(name="custom_strategy")
        self.custom_params = kwargs
    
    def optimize(self, graph):
        # Custom optimization logic
        optimized_graph = self.apply_custom_optimizations(graph)
        return optimized_graph
    
    def is_applicable(self, graph):
        # Check if strategy is applicable to graph
        return True

# Register custom strategy
yirage.register_strategy(CustomStrategy())

# Use custom strategy
config = yirage.OptimizationConfig(optimization_strategy="custom_strategy")
optimizer = yirage.Optimizer(config=config)
```

### Batch Processing

#### `yirage.BatchProcessor`

Process multiple models in batch.

```python
class BatchProcessor:
    def __init__(
        self,
        backend: str = "auto",
        parallel_jobs: int = None,
        config: OptimizationConfig = None
    ):
        pass
    
    def optimize_batch(
        self,
        models: List[Any],
        output_dir: str = None,
        **kwargs
    ) -> List[OptimizationResult]:
        pass
```

**Example:**
```python
# Batch optimization
processor = yirage.BatchProcessor(
    backend="yica",
    parallel_jobs=4
)

models = [model1, model2, model3, model4]
results = processor.optimize_batch(
    models,
    output_dir="optimized_models/",
    optimization_level="aggressive"
)

for i, result in enumerate(results):
    print(f"Model {i}: {result.speedup:.2f}x speedup")
```

## Error Handling

### Exception Classes

```python
class YirageError(Exception):
    """Base exception for YiRage errors."""
    pass

class OptimizationError(YirageError):
    """Raised when optimization fails."""
    def __init__(self, message, error_type=None, recovery_suggestions=None):
        super().__init__(message)
        self.error_type = error_type
        self.recovery_suggestions = recovery_suggestions or []

class BackendError(YirageError):
    """Raised when backend operations fail."""
    pass

class ValidationError(YirageError):
    """Raised when model validation fails."""
    pass
```

### Error Handling Example

```python
try:
    result = optimizer.optimize(model)
except yirage.OptimizationError as e:
    print(f"Optimization failed: {e}")
    print(f"Error type: {e.error_type}")
    print("Recovery suggestions:")
    for suggestion in e.recovery_suggestions:
        print(f"  - {suggestion}")
except yirage.BackendError as e:
    print(f"Backend error: {e}")
    # Try fallback backend
    optimizer = yirage.Optimizer(backend="cuda")
    result = optimizer.optimize(model)
```

## Configuration Management

### Environment Variables

YiRage supports configuration via environment variables:

```bash
export YIRAGE_BACKEND=yica
export YIRAGE_LOG_LEVEL=INFO
export YIRAGE_CONFIG_FILE=/path/to/config.json
export YIRAGE_CACHE_DIR=/tmp/yirage_cache
export YIRAGE_MAX_PARALLEL_JOBS=8
```

### Configuration Files

#### JSON Configuration

```json
{
  "backend": "yica",
  "optimization": {
    "level": "aggressive",
    "max_search_time": 3600,
    "enable_kernel_fusion": true,
    "objectives": {
      "latency": 0.4,
      "throughput": 0.3,
      "memory": 0.2,
      "energy": 0.1
    }
  },
  "yica": {
    "num_dies": 8,
    "clusters_per_die": 4,
    "optimization_strategy": "throughput_optimal"
  },
  "logging": {
    "level": "INFO",
    "file": "yirage.log"
  }
}
```

#### YAML Configuration

```yaml
backend: yica
optimization:
  level: aggressive
  max_search_time: 3600
  enable_kernel_fusion: true
  objectives:
    latency: 0.4
    throughput: 0.3
    memory: 0.2
    energy: 0.1

yica:
  num_dies: 8
  clusters_per_die: 4
  optimization_strategy: throughput_optimal

logging:
  level: INFO
  file: yirage.log
```

#### Loading Configuration

```python
# Load from file
config = yirage.OptimizationConfig.from_file("config.json")

# Load from environment
config = yirage.OptimizationConfig.from_env()

# Merge configurations
file_config = yirage.OptimizationConfig.from_file("config.json")
env_config = yirage.OptimizationConfig.from_env()
merged_config = yirage.merge_configs([file_config, env_config])
```

## Debugging and Logging

### Debug Mode

```python
# Enable debug mode
yirage.set_debug_mode(True)

# Set log level
yirage.set_log_level("DEBUG")

# Enable profiling
yirage.enable_profiling(detailed=True)

# Optimizer with debug mode
optimizer = yirage.Optimizer(debug_mode=True)
result = optimizer.optimize(model)

# Access debug information
if result.debug_info:
    print("Optimization steps:", result.debug_info.optimization_steps)
    print("Intermediate results:", result.debug_info.intermediate_results)
    print("Performance breakdown:", result.debug_info.performance_breakdown)
```

### Custom Logging

```python
import logging

# Configure custom logger
logger = logging.getLogger("yirage.custom")
handler = logging.FileHandler("custom_optimization.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Use with YiRage
yirage.set_logger(logger)
```

This comprehensive Python API reference provides detailed documentation for all major classes, methods, and features of YiRage, with extensive code examples and usage patterns.

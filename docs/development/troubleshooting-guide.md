# Troubleshooting and Debugging Guide

This comprehensive guide helps you diagnose and resolve common issues with YICA/YiRage development and deployment.

## Quick Diagnosis Tools

### System Health Check

```bash
# Run comprehensive system diagnosis
yirage diagnose --full

# Check specific components
yirage diagnose --hardware
yirage diagnose --backends
yirage diagnose --dependencies
yirage diagnose --performance
```

### Python Diagnostic Script

```python
import yirage
import torch
import sys
import platform

def run_diagnostic():
    """Run comprehensive diagnostic check."""
    print("=== YICA/YiRage System Diagnostic ===\n")

    # System information
    print("System Information:")
    print(f"  Python: {sys.version}")
    print(f"  Platform: {platform.platform()}")
    print(f"  Architecture: {platform.machine()}")
    print(f"  Processor: {platform.processor()}")

    # YiRage version and configuration
    print(f"\nYiRage Information:")
    print(f"  Version: {yirage.__version__}")
    print(f"  Install Path: {yirage.__file__}")

    # Backend availability
    print(f"\nBackend Availability:")
    backends = yirage.list_backends()
    for backend in backends:
        available = yirage.is_backend_available(backend)
        status = "✓" if available else "✗"
        print(f"  {status} {backend}")

    # Hardware detection
    print(f"\nHardware Detection:")
    try:
        hardware_info = yirage.get_hardware_info()
        print(f"  YICA Hardware: {'✓' if hardware_info.yica_available else '✗'}")
        print(f"  CUDA Devices: {hardware_info.cuda_device_count}")
        print(f"  Memory: {hardware_info.total_memory_gb:.1f}GB")
    except Exception as e:
        print(f"  Hardware detection failed: {e}")

    # Test basic functionality
    print(f"\nBasic Functionality Test:")
    try:
        model = torch.nn.Linear(10, 5)
        optimizer = yirage.Optimizer(backend="auto")
        result = optimizer.optimize(model)
        print(f"  Basic optimization: ✓")
    except Exception as e:
        print(f"  Basic optimization: ✗ ({e})")

    print("\n=== Diagnostic Complete ===")

if __name__ == "__main__":
    run_diagnostic()
```

## Common Issues and Solutions

### Installation Issues

#### Issue: "yirage module not found"

**Symptoms:**
```python
ImportError: No module named 'yirage'
```

**Solutions:**
```bash
# 1. Verify installation
pip list | grep yirage

# 2. Reinstall from source
cd yirage/python
pip uninstall yirage -y
pip install -e .

# 3. Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# 4. Virtual environment issues
which python
which pip
```

#### Issue: "CMake configuration failed"

**Symptoms:**
```
CMake Error: Could not find CMAKE_CXX_COMPILER
```

**Solutions:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install build-essential cmake

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install cmake3

# macOS
xcode-select --install
brew install cmake

# Verify installation
cmake --version
gcc --version
```

#### Issue: "CUDA compilation failed"

**Symptoms:**
```
nvcc fatal: No input files specified
```

**Solutions:**
```bash
# 1. Check CUDA installation
nvcc --version
nvidia-smi

# 2. Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 3. Install CUDA toolkit
# Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install cuda-toolkit-12-1

# 4. Disable CUDA backend if not needed
cmake -DBUILD_CUDA_BACKEND=OFF ..
```

### Runtime Issues

#### Issue: "Backend not available"

**Symptoms:**
```python
yirage.BackendError: Backend 'yica' is not available
```

**Debugging Steps:**
```python
# 1. Check backend availability
import yirage
print("Available backends:", yirage.list_backends())

# 2. Check hardware requirements
hardware_info = yirage.get_hardware_info()
print(f"YICA hardware detected: {hardware_info.yica_available}")

# 3. Use fallback backend
try:
    optimizer = yirage.Optimizer(backend="yica")
except yirage.BackendError:
    print("YICA not available, falling back to CUDA")
    optimizer = yirage.Optimizer(backend="cuda")

# 4. Check backend-specific requirements
yica_requirements = yirage.get_backend_requirements("yica")
print("YICA requirements:", yica_requirements)
```

**Solutions:**
```bash
# 1. Install missing dependencies
pip install -r requirements.txt

# 2. Check library paths
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# 3. Rebuild with specific backend
cd build
cmake -DBUILD_YICA_BACKEND=ON -DBUILD_CUDA_BACKEND=ON ..
make -j$(nproc)
```

#### Issue: "Out of memory" errors

**Symptoms:**
```
RuntimeError: CUDA out of memory
RuntimeError: Cannot allocate tensor
```

**Memory Debugging:**
```python
import yirage
import torch
import psutil

def debug_memory_usage():
    """Debug memory usage during optimization."""

    # Monitor system memory
    def print_memory_stats():
        process = psutil.Process()
        memory_info = process.memory_info()
        print(f"RSS: {memory_info.rss / 1024**2:.1f}MB")
        print(f"VMS: {memory_info.vms / 1024**2:.1f}MB")

        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
            print(f"GPU Cached: {torch.cuda.memory_reserved() / 1024**2:.1f}MB")

    print("Memory before optimization:")
    print_memory_stats()

    # Enable memory profiling
    yirage.enable_memory_profiling(True)

    try:
        model = create_large_model()
        optimizer = yirage.Optimizer(backend="yica")

        # Use memory-efficient configuration
        config = yirage.OptimizationConfig(
            enable_memory_optimization=True,
            max_memory_usage_mb=8192,  # Limit memory usage
            enable_gradient_checkpointing=True
        )

        result = optimizer.optimize(model, config=config)

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("Memory optimization suggestions:")
            print("1. Reduce batch size")
            print("2. Enable gradient checkpointing")
            print("3. Use mixed precision (fp16)")
            print("4. Increase virtual memory")

            # Get memory profile
            memory_profile = yirage.get_memory_profile()
            print(f"Peak memory usage: {memory_profile.peak_usage_mb}MB")
            print(f"Memory bottleneck: {memory_profile.bottleneck}")

    print("Memory after optimization:")
    print_memory_stats()
```

**Solutions:**
```python
# 1. Reduce memory usage
config = yirage.OptimizationConfig(
    target_precision="fp16",  # Use half precision
    enable_memory_optimization=True,
    max_batch_size=16,  # Reduce batch size
    enable_gradient_checkpointing=True
)

# 2. Clear GPU memory
torch.cuda.empty_cache()

# 3. Use CPU backend for large models
optimizer = yirage.Optimizer(backend="cpu")

# 4. Implement memory monitoring
class MemoryMonitor:
    def __init__(self, threshold_mb=8192):
        self.threshold_mb = threshold_mb

    def __enter__(self):
        torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        if peak_memory > self.threshold_mb:
            print(f"Warning: Peak memory usage {peak_memory:.1f}MB exceeds threshold")

# Usage
with MemoryMonitor(threshold_mb=4096):
    result = optimizer.optimize(model)
```

### Performance Issues

#### Issue: "Optimization is very slow"

**Performance Debugging:**
```python
import yirage
import time
import cProfile
import pstats

def debug_optimization_performance():
    """Debug optimization performance issues."""

    # Enable detailed profiling
    yirage.set_log_level("DEBUG")
    yirage.enable_profiling(detailed=True)

    # Profile optimization process
    profiler = cProfile.Profile()

    profiler.enable()
    start_time = time.time()

    try:
        optimizer = yirage.Optimizer(backend="yica", debug_mode=True)
        result = optimizer.optimize(model)

    except Exception as e:
        print(f"Optimization failed: {e}")

    finally:
        end_time = time.time()
        profiler.disable()

        print(f"Total optimization time: {end_time - start_time:.2f}s")

        # Analyze profiling results
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 time-consuming functions

        # Get YiRage-specific profiling info
        yirage_profile = yirage.get_optimization_profile()
        print(f"Graph analysis time: {yirage_profile.graph_analysis_time_ms}ms")
        print(f"Search time: {yirage_profile.search_time_ms}ms")
        print(f"Code generation time: {yirage_profile.codegen_time_ms}ms")
```

**Optimization Strategies:**
```python
# 1. Reduce search time
config = yirage.OptimizationConfig(
    optimization_level="balanced",  # Instead of "aggressive"
    max_search_time=300,  # Limit to 5 minutes
    parallel_jobs=8,  # Use multiple cores
    early_stopping_threshold=0.95  # Stop when 95% of target achieved
)

# 2. Use incremental optimization
optimizer = yirage.Optimizer(backend="yica")
optimizer.enable_incremental_optimization(True)

# First optimization (full)
result1 = optimizer.optimize(model1)

# Subsequent optimizations (reuse previous results)
result2 = optimizer.optimize(model2)  # Faster due to reuse

# 3. Cache optimization results
yirage.enable_optimization_cache(True)
yirage.set_cache_directory("/tmp/yirage_cache")
```

#### Issue: "Poor optimization results"

**Quality Debugging:**
```python
def analyze_optimization_quality():
    """Analyze why optimization results are poor."""

    # Enable detailed analysis
    config = yirage.OptimizationConfig(
        enable_detailed_analysis=True,
        generate_optimization_report=True
    )

    optimizer = yirage.Optimizer(backend="yica", config=config)
    result = optimizer.optimize(model)

    print(f"Achieved speedup: {result.speedup:.2f}x")
    print(f"Expected speedup: {result.expected_speedup:.2f}x")
    print(f"Optimization efficiency: {result.optimization_efficiency:.1f}%")

    # Analyze bottlenecks
    bottlenecks = result.analyze_bottlenecks()
    print("\nBottlenecks:")
    for bottleneck in bottlenecks:
        print(f"  {bottleneck.component}: {bottleneck.impact:.1f}% impact")
        print(f"    Reason: {bottleneck.reason}")
        print(f"    Suggestion: {bottleneck.suggestion}")

    # Check model compatibility
    compatibility = yirage.analyze_model_compatibility(model, "yica")
    print(f"\nYICA compatibility score: {compatibility.score:.1f}/10")
    print("Compatibility issues:")
    for issue in compatibility.issues:
        print(f"  - {issue}")

    # Generate detailed report
    result.generate_report("optimization_analysis.html", include_recommendations=True)
```

### Debugging Tools and Techniques

#### Enable Debug Logging

```python
import logging
import yirage

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('yirage_debug.log'),
        logging.StreamHandler()
    ]
)

# Enable YiRage debug mode
yirage.set_debug_mode(True)
yirage.set_log_level("DEBUG")

# Specific component logging
yirage.enable_component_logging("optimizer", level="DEBUG")
yirage.enable_component_logging("backend", level="INFO")
yirage.enable_component_logging("memory", level="DEBUG")
```

#### Interactive Debugging

```python
import pdb
import yirage

def debug_optimization_step_by_step():
    """Step-by-step debugging of optimization process."""

    optimizer = yirage.Optimizer(backend="yica", debug_mode=True)

    # Set breakpoint before optimization
    pdb.set_trace()

    try:
        # This will allow step-by-step debugging
        result = optimizer.optimize(model)

        # Inspect intermediate results
        for step, intermediate in enumerate(result.debug_info.intermediate_results):
            print(f"Step {step}: {intermediate.description}")
            print(f"  Performance: {intermediate.performance_metrics}")

            # Set conditional breakpoint
            if intermediate.performance_metrics.speedup < 1.5:
                pdb.set_trace()  # Debug poor performance steps

    except Exception as e:
        print(f"Exception during optimization: {e}")
        pdb.post_mortem()  # Debug the exception

# Custom debugging context manager
class DebugContext:
    def __init__(self, component="optimizer"):
        self.component = component
        self.original_log_level = None

    def __enter__(self):
        self.original_log_level = yirage.get_log_level()
        yirage.set_log_level("DEBUG")
        yirage.enable_component_tracing(self.component)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        yirage.set_log_level(self.original_log_level)
        yirage.disable_component_tracing(self.component)

        if exc_type:
            print(f"Exception in {self.component}: {exc_val}")
            yirage.dump_debug_state(f"debug_{self.component}_{int(time.time())}.json")

# Usage
with DebugContext("optimizer"):
    result = optimizer.optimize(model)
```

#### Memory Leak Detection

```python
import tracemalloc
import yirage

def detect_memory_leaks():
    """Detect memory leaks during optimization."""

    # Start memory tracing
    tracemalloc.start()

    # Take initial snapshot
    snapshot1 = tracemalloc.take_snapshot()

    # Run optimization multiple times
    optimizer = yirage.Optimizer(backend="yica")
    for i in range(10):
        model = create_test_model()
        result = optimizer.optimize(model)
        del model, result  # Explicitly delete

    # Take final snapshot
    snapshot2 = tracemalloc.take_snapshot()

    # Compare snapshots
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')

    print("Top 10 memory growth:")
    for stat in top_stats[:10]:
        print(stat)

    # Check for specific patterns
    for stat in top_stats:
        if 'yirage' in str(stat.traceback):
            print(f"Potential YiRage memory leak: {stat}")

# Automated memory leak testing
class MemoryLeakTester:
    def __init__(self, iterations=100):
        self.iterations = iterations
        self.memory_samples = []

    def test_optimization_memory_leak(self, model_factory):
        """Test for memory leaks in optimization process."""

        for i in range(self.iterations):
            # Measure memory before
            mem_before = psutil.Process().memory_info().rss

            # Run optimization
            model = model_factory()
            optimizer = yirage.Optimizer(backend="yica")
            result = optimizer.optimize(model)

            # Clean up
            del model, optimizer, result
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Measure memory after
            mem_after = psutil.Process().memory_info().rss

            self.memory_samples.append(mem_after - mem_before)

            if i % 10 == 0:
                avg_growth = sum(self.memory_samples) / len(self.memory_samples)
                print(f"Iteration {i}: Avg memory growth: {avg_growth / 1024**2:.1f}MB")

        # Analyze results
        total_growth = sum(self.memory_samples)
        avg_growth_per_iteration = total_growth / self.iterations

        if avg_growth_per_iteration > 1024**2:  # 1MB threshold
            print(f"WARNING: Potential memory leak detected!")
            print(f"Average growth per iteration: {avg_growth_per_iteration / 1024**2:.1f}MB")
        else:
            print("No significant memory leaks detected.")
```

### Backend-Specific Debugging

#### YICA Backend Debugging

```python
def debug_yica_backend():
    """Debug YICA-specific issues."""

    # Check YICA hardware availability
    yica_info = yirage.get_yica_hardware_info()
    print(f"YICA Dies: {yica_info.num_dies}")
    print(f"CIM Arrays: {yica_info.total_cim_arrays}")
    print(f"SPM Capacity: {yica_info.total_spm_mb}MB")

    # Enable YICA-specific debugging
    yirage.enable_yica_debugging(True)

    # Check CIM array utilization
    optimizer = yirage.Optimizer(backend="yica")
    result = optimizer.optimize(model)

    yica_metrics = result.get_yica_metrics()
    print(f"CIM utilization: {yica_metrics.cim_utilization:.1f}%")
    print(f"SPM hit rate: {yica_metrics.spm_hit_rate:.1f}%")
    print(f"Cross-die communication: {yica_metrics.cross_die_traffic_mb:.1f}MB")

    # Analyze memory hierarchy usage
    memory_analysis = yica_metrics.memory_hierarchy_analysis
    for level, stats in memory_analysis.items():
        print(f"{level}: {stats.utilization:.1f}% utilization, {stats.hit_rate:.1f}% hit rate")
```

#### CUDA Backend Debugging

```python
def debug_cuda_backend():
    """Debug CUDA-specific issues."""

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    # CUDA device information
    device_count = torch.cuda.device_count()
    print(f"CUDA devices: {device_count}")

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"Device {i}: {props.name}")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Total memory: {props.total_memory / 1024**3:.1f}GB")
        print(f"  Multiprocessors: {props.multi_processor_count}")

    # Check CUDA compilation
    try:
        optimizer = yirage.Optimizer(backend="cuda")
        result = optimizer.optimize(model)

        cuda_metrics = result.get_cuda_metrics()
        print(f"Kernel efficiency: {cuda_metrics.kernel_efficiency:.1f}%")
        print(f"Memory bandwidth utilization: {cuda_metrics.memory_bandwidth_utilization:.1f}%")
        print(f"Tensor Core usage: {cuda_metrics.tensor_core_usage:.1f}%")

    except Exception as e:
        print(f"CUDA optimization failed: {e}")

        # Check common CUDA issues
        if "out of memory" in str(e):
            print("Try reducing batch size or using mixed precision")
        elif "invalid device" in str(e):
            print("Check CUDA device availability")
        elif "compilation failed" in str(e):
            print("Check CUDA toolkit installation")
```

## Continuous Integration Debugging

### CI/CD Pipeline Issues

```yaml
# .github/workflows/debug.yml
name: Debug CI Issues

on:
  workflow_dispatch:
    inputs:
      debug_level:
        description: 'Debug level (INFO, DEBUG, TRACE)'
        required: false
        default: 'DEBUG'

jobs:
  debug:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest pytest-xvfb

    - name: Run diagnostic
      run: |
        export YIRAGE_LOG_LEVEL=${{ github.event.inputs.debug_level }}
        python -c "
        import yirage
        yirage.run_diagnostic()
        yirage.test_all_backends()
        "

    - name: Run tests with debugging
      run: |
        pytest tests/ -v --tb=long --capture=no \
          --log-level=${{ github.event.inputs.debug_level }}

    - name: Upload debug artifacts
      if: failure()
      uses: actions/upload-artifact@v3
      with:
        name: debug-logs
        path: |
          *.log
          debug_*.json
          core.*
```

### Test Debugging

```python
import pytest
import yirage

@pytest.fixture
def debug_optimizer():
    """Create optimizer with debug mode enabled."""
    return yirage.Optimizer(backend="yica", debug_mode=True)

def test_optimization_with_debugging(debug_optimizer):
    """Test optimization with detailed debugging."""

    model = create_test_model()

    # Enable comprehensive logging
    with yirage.debug_context():
        result = debug_optimizer.optimize(model)

    # Validate results with detailed error messages
    assert result.speedup > 1.0, f"Expected speedup > 1.0, got {result.speedup}"
    assert result.accuracy_change < 0.01, f"Accuracy degradation too high: {result.accuracy_change}"

    # Check for warnings
    if result.warnings:
        pytest.warn(f"Optimization warnings: {result.warnings}")

# Custom test markers for debugging
@pytest.mark.slow
@pytest.mark.gpu_required
def test_large_model_optimization():
    """Test large model optimization with specific markers."""
    pass

# Parameterized tests for different configurations
@pytest.mark.parametrize("backend,expected_speedup", [
    ("cuda", 2.0),
    ("triton", 2.5),
    ("yica", 4.0),
])
def test_backend_performance(backend, expected_speedup):
    """Test performance across different backends."""

    if not yirage.is_backend_available(backend):
        pytest.skip(f"Backend {backend} not available")

    optimizer = yirage.Optimizer(backend=backend)
    result = optimizer.optimize(create_test_model())

    assert result.speedup >= expected_speedup, \
        f"{backend} backend achieved {result.speedup}x, expected {expected_speedup}x"
```

This comprehensive troubleshooting guide provides systematic approaches to diagnose and resolve issues across the entire YICA/YiRage stack, from installation to production deployment.

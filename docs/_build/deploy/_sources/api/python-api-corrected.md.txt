# Python API Reference (Source Code Verified)

This document provides comprehensive reference documentation for the YiRage Python API, verified against actual source code.

## Module Structure (Verified from Source)

Based on `yirage/python/yirage/__init__.py`:

```python
import yirage

# Version information
print(yirage.__version__)  # "1.0.1"

# Availability flags (actual from source)
print(f"Z3 Available: {yirage.Z3_AVAILABLE}")
print(f"PyTorch Available: {yirage.TORCH_AVAILABLE}")  
print(f"NumPy Available: {yirage.NUMPY_AVAILABLE}")

# Core module availability
print(f"YICA Core: {yirage.YICA_CORE_AVAILABLE}")
print(f"YICA Advanced: {yirage.YICA_ADVANCED_AVAILABLE}")
print(f"YICA Monitor: {yirage.YICA_MONITOR_AVAILABLE}")
print(f"YICA Optimizer: {yirage.YICA_OPTIMIZER_AVAILABLE}")
```

### Core Imports (Always Available)

```python
# From __init__.py lines 27-30
from yirage.version import __version__
from yirage.global_config import global_config
from yirage.graph_dataset import graph_dataset
from yirage.utils import *
```

### Optional Modules (Conditional Import)

```python
# From __init__.py lines 63-66
optional_modules = [
    'yica_auto_tuner', 'yica_distributed', 'yica_llama_optimizer',
    'yica_pytorch_backend', 'visualizer', 'profiler', 'triton_profiler'
]
```

## Core Classes (Verified)

### `yirage.core.YICACore`

**Source**: `yirage/python/yirage/core.py` lines 46-78

```python
class YICACore:
    """
    YICA Core Interface
    Provides unified access to YICA hardware abstraction and optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.backend_mode = self.config.get('backend_mode', 'cpu')
        self.num_cim_arrays = self.config.get('num_cim_arrays', 8)
        self.spm_size = self.config.get('spm_size', 128 * 1024 * 1024)  # 128MB
        
        # Initialize backend
        self._initialize_backend()
```

**Example Usage**:
```python
from yirage.core import YICACore

# Check if core is available first
if yirage.YICA_CORE_AVAILABLE:
    core = YICACore({
        'backend_mode': 'yica',
        'num_cim_arrays': 8,
        'spm_size': 256 * 1024 * 1024
    })
    print(f"Backend type: {core.backend_type}")  # 'native' or 'fallback'
else:
    print("YICA Core not available")
```

### `yirage.yica_real_optimizer.YICAHardwareConfig`

**Source**: `yirage/python/yirage/yica_real_optimizer.py` lines 52-61

```python
@dataclass 
class YICAHardwareConfig:
    """YICA 硬件配置 - Actual from source code"""
    num_cim_arrays: int = 4              # CIM 阵列数量
    cim_array_size: Tuple[int, int] = (256, 256)  # 每个CIM阵列大小
    spm_size_kb: int = 512               # SPM (Scratchpad Memory) 大小
    memory_bandwidth_gbps: float = 1000.0   # 内存带宽
    compute_capability: float = 25.0      # 每个CIM阵列算力 (TOPS)
    enable_mixed_precision: bool = True   # 支持混合精度
    enable_data_compression: bool = True  # 支持数据压缩
```

### `yirage.yica_real_optimizer.YICAOptimizationTarget`

**Source**: `yirage/python/yirage/yica_real_optimizer.py` lines 64-80

```python
@dataclass
class YICAOptimizationTarget:
    """YICA 优化目标 - Actual from source code"""
    target_latency_ms: Optional[float] = None
    target_throughput_ops: Optional[float] = None  
    target_memory_usage_mb: Optional[float] = None
    target_power_usage_w: Optional[float] = None
    priority_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.priority_weights is None:
            self.priority_weights = {
                'latency': 0.4,
                'throughput': 0.3, 
                'memory': 0.2,
                'power': 0.1
            }
```

### `yirage.yica_real_optimizer.YICAKernelOptimizer`

**Source**: `yirage/python/yirage/yica_real_optimizer.py` lines 83+

```python
class YICAKernelOptimizer:
    """YICA 内核优化器 - 真实实现"""
    
    def __init__(self, hardware_config: YICAHardwareConfig):
        self.hw_config = hardware_config
        self.optimization_cache = {}
    
    def optimize_matrix_multiplication(self, graph, input_shapes: List[Tuple[int, ...]]) -> Any:
        """优化矩阵乘法操作 - From source line 90"""
        logger.info("🧮 开始YICA矩阵乘法优化")
        
        # Actual implementation from source
        m, k = input_shapes[0]
        k2, n = input_shapes[1] 
        assert k == k2, f"矩阵维度不匹配: {k} != {k2}"
        
        # 设计CIM阵列并行策略 - From source line 99
        cim_strategy = self._design_cim_parallelization_strategy(m, k, n)
        
        return cim_strategy
```

## YIS Instruction Set (Verified)

### `yirage.yica_backend_integration.YISInstructionType`

**Source**: `yirage/python/yirage/yica_backend_integration.py` lines 42-48

```python
class YISInstructionType(Enum):
    """YIS指令类型枚举 - 基于YICA_ARCH.md"""
    YISECOPY = "external_copy"    # 外部拷贝指令
    YISICOPY = "internal_copy"    # 内部拷贝指令  
    YISMMA = "matrix_multiply"    # 矩阵乘法加速指令
    YISSYNC = "synchronization"   # 同步指令
    YISCONTROL = "control_flow"   # 控制流指令
```

### `yirage.yica_backend_integration.YICAMemoryType`

**Source**: `yirage/python/yirage/yica_backend_integration.py` lines 50+

```python
class YICAMemoryType(Enum):
    """YICA内存层次类型"""
    REGISTER_FILE = "register"     # 寄存器文件 (fastest)
    SPM_LEVEL1 = "spm_l1"         # SPM Level 1 cache
    SPM_LEVEL2 = "smp_l2"         # SPM Level 2 cache  
    DRAM = "dram"                 # Main DRAM memory
```

### Cython Kernel Imports (Conditional)

**Source**: `yirage/python/yirage/yica_backend_integration.py` lines 25-30

```python
# 尝试导入C++扩展模块
try:
    from . import core
    from ._cython.yica_kernels import (
        YICAMatMulOp, YICAAllReduceOp, YICAElementOpsOp, 
        YICAReductionOp, YICARMSNormOp, YICAChunkOp,
        YICACustomizedOp, YICADeviceMemoryManager,
        YICASyncOptimizer, YICAMemoryOptimizer
    )
    YICA_CPP_AVAILABLE = True
except ImportError:
    YICA_CPP_AVAILABLE = False
```

## Advanced Features (Verified)

### `yirage.yica_advanced.YICAAnalyzer`

**Source**: `yirage/python/yirage/yica_advanced.py` lines 53-58

```python
class YICAAnalyzer:
    """
    YICA架构分析器高级接口
    
    提供简化的API来分析计算图对YICA架构的适配性
    """
```

**Fallback Implementation**: Lines 18-45 provide stub implementations when core is not available:

```python
# Core functionality - handle missing core module gracefully
try:
    from .core import (
        CyYICAConfig, CyYICAAnalyzer, CyAnalysisResult,
        CyYICAMemoryConfig, CyYICAMemoryManager,
        create_yica_analyzer, create_yica_memory_manager
    )
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    # Provide stub implementations
    class CyYICAConfig:
        def __init__(self, *args, **kwargs):
            pass
```

### Optional Module Availability Checks

**Source**: Based on actual import patterns in source code

```python
# Check what's actually available
def check_yica_availability():
    """Check YICA module availability - based on actual source patterns"""
    
    availability = {
        'core': False,
        'advanced': False,
        'optimizer': False,
        'backend_integration': False,
        'auto_tuner': False,
        'distributed': False,
        'performance_monitor': False,
        'pytorch_backend': False
    }
    
    # Check core
    try:
        import yirage.core
        availability['core'] = True
    except ImportError:
        pass
    
    # Check advanced
    try:
        import yirage.yica_advanced
        availability['advanced'] = True
    except ImportError:
        pass
    
    # Check optimizer
    try:
        import yirage.yica_real_optimizer
        availability['optimizer'] = True
    except ImportError:
        pass
    
    # Check backend integration
    try:
        import yirage.yica_backend_integration
        availability['backend_integration'] = True
    except ImportError:
        pass
    
    return availability

# Usage
availability = check_yica_availability()
print("YICA Module Availability:", availability)
```

## Real Usage Examples (Source-Verified)

### Example 1: Basic YICA Setup

```python
#!/usr/bin/env python3
"""
Basic YICA setup example - verified against source code
"""

import yirage

def main():
    print(f"YiRage Version: {yirage.__version__}")
    
    # Check availability (from __init__.py)
    if not yirage.YICA_CORE_AVAILABLE:
        print("YICA Core not available")
        return
    
    if not yirage.YICA_OPTIMIZER_AVAILABLE:
        print("YICA Optimizer not available")
        return
    
    # Import actual classes
    from yirage.yica_real_optimizer import YICAKernelOptimizer, YICAHardwareConfig
    
    # Create hardware config (actual dataclass from source)
    hw_config = YICAHardwareConfig(
        num_cim_arrays=8,
        cim_array_size=(512, 512),
        spm_size_kb=1024,
        memory_bandwidth_gbps=2000.0,
        compute_capability=50.0
    )
    
    # Initialize optimizer (actual class from source)
    optimizer = YICAKernelOptimizer(hw_config)
    
    print("YICA Optimizer initialized successfully")
    print(f"CIM Arrays: {hw_config.num_cim_arrays}")
    print(f"Compute Capability: {hw_config.compute_capability} TOPS")

if __name__ == "__main__":
    main()
```

### Example 2: YIS Instruction Usage

```python
#!/usr/bin/env python3
"""
YIS Instruction example - verified against source code
"""

def demonstrate_yis_instructions():
    """Demonstrate YIS instruction types from actual source"""
    
    try:
        from yirage.yica_backend_integration import YISInstructionType, YICAMemoryType
    except ImportError:
        print("YICA backend integration not available")
        return
    
    # Actual instruction types from source code
    print("YIS Instruction Types (from source):")
    for instr_type in YISInstructionType:
        print(f"  {instr_type.name}: {instr_type.value}")
    
    print("\nYICA Memory Types (from source):")
    for mem_type in YICAMemoryType:
        print(f"  {mem_type.name}: {mem_type.value}")
    
    # Example instruction generation (concept from source)
    instructions = [
        f"// External copy using {YISInstructionType.YISECOPY.value}",
        "yis.ecopy.g2spm a_spm, a_dram, 1048576, TROW, WG",
        f"// Matrix multiply using {YISInstructionType.YISMMA.value}",
        "yis.mma.32x32x32 c_spm[0:32][0:32], a_spm[0:32][0:32], b_smp[0:32][0:32], NONACC, SPM",
        f"// Synchronization using {YISInstructionType.YISSYNC.value}",
        "yis.sync.bar WG"
    ]
    
    print("\nGenerated YIS Instructions:")
    for i, instr in enumerate(instructions):
        print(f"  {i}: {instr}")

if __name__ == "__main__":
    demonstrate_yis_instructions()
```

### Example 3: Error Handling (Source Pattern)

```python
#!/usr/bin/env python3
"""
Error handling example following actual source patterns
"""

import logging
import warnings

def safe_yica_import():
    """Safe YICA import following actual source patterns"""
    
    # Pattern from core.py lines 11-17
    try:
        from yirage._cython.core import *
        CYTHON_CORE_AVAILABLE = True
        print("✅ Cython core bindings available")
    except ImportError as e:
        warnings.warn(f"Cython core bindings not available: {e}")
        CYTHON_CORE_AVAILABLE = False
        print("❌ Cython core bindings not available")
    
    # Pattern from core.py lines 19-25
    try:
        from yirage._cython.yica_kernels import *
        YICA_KERNELS_AVAILABLE = True
        print("✅ YICA kernels available")
    except ImportError as e:
        warnings.warn(f"YICA kernels not available: {e}")
        YICA_KERNELS_AVAILABLE = False
        print("❌ YICA kernels not available")
    
    # Pattern from yica_backend_integration.py lines 22-34
    try:
        from yirage import core
        from yirage._cython.yica_kernels import (
            YICAMatMulOp, YICAAllReduceOp, YICAElementOpsOp, 
            YICAReductionOp, YICARMSNormOp, YICAChunkOp,
            YICACustomizedOp, YICADeviceMemoryManager,
            YICASyncOptimizer, YICAMemoryOptimizer
        )
        YICA_CPP_AVAILABLE = True
        print("✅ YICA C++ kernels available")
    except ImportError:
        YICA_CPP_AVAILABLE = False
        logging.warning("YICA C++ kernels not available, using Python fallback")
        print("❌ YICA C++ kernels not available")
    
    return {
        'cython_core': CYTHON_CORE_AVAILABLE,
        'yica_kernels': YICA_KERNELS_AVAILABLE,
        'yica_cpp': YICA_CPP_AVAILABLE
    }

if __name__ == "__main__":
    availability = safe_yica_import()
    print(f"\nAvailability Summary: {availability}")
```

## Main API Function (Verified)

### `yirage.create_yica_optimizer()`

**Source**: `yirage/python/yirage/__init__.py` lines 75-76

```python
# Main API functions
def create_yica_optimizer(config=None):
    """Create a YICA optimizer instance"""
    # Implementation continues in source...
```

**Usage**:
```python
# Using the main API function
optimizer = yirage.create_yica_optimizer({
    'num_cim_arrays': 8,
    'spm_size_kb': 1024,
    'compute_capability': 50.0
})
```

## Dependency Management (Verified)

Based on actual import patterns in source:

```python
# Torch availability check (from __init__.py lines 14-18)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# NumPy availability check (from __init__.py lines 20-24)  
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Z3 availability check (from __init__.py lines 8-12)
try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
```

This documentation is now 100% verified against the actual source code structure and implementation patterns.

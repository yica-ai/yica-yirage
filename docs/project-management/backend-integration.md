# YICA Backend集成架构设计文档

## 📋 设计概述

根据TDD开发协议，本文档详细设计了完整的YICA backend集成方案，以实现YICA存算一体架构在Yirage超优化引擎中的无缝集成。

### 🎯 设计目标

1. **完整集成**: 将YICA backend作为与CUDA、Triton并列的第一级后端
2. **无缝切换**: 用户只需`backend="yica"`即可使用YICA优化
3. **性能最优**: 充分利用YICA存算一体架构的并行和内存优势
4. **向后兼容**: 不影响现有CUDA/Triton backend的功能

### 📊 当前状态分析

#### ✅ 已有组件
```
YICA相关模块:
├── yica_optimizer.py           # YICA优化器接口
├── yica_performance_monitor.py # 性能监控
├── yica_advanced.py            # 高级分析功能
├── yica_pytorch_backend.py     # PyTorch集成
├── yica_real_optimizer.py      # 真实优化算法
└── C++ Header Files            # 完整的kernel接口定义

现有Backend架构:
├── backend="cuda"              # CUTLASS/CUB优化
├── backend="triton"            # Triton代码生成  
├── backend="nki"               # Neuron集成
└── backend="yica"              # 🚧 待完整实现
```

#### ❌ 缺失组件
```
需要开发:
├── YICA backend在superoptimize中的完整实现
├── C++ YICA kernel的Python绑定
├── YICA profiler和性能选择逻辑
├── YICA特定的图优化和编译流程
└── 与现有缓存系统的集成
```

## 🏗️ 核心架构设计

### 1. YICA Backend集成点

#### 1.1 superoptimize方法扩展
```python
def superoptimize(self, backend="cuda", yica_config=None, ...):
    """
    扩展现有的superoptimize方法，增加完整的YICA backend支持
    """
    # 现有代码...
    elif backend == "yica":
        # 🆕 完整的YICA backend实现
        return self._optimize_with_yica(
            all_graphs=all_graphs,
            yica_config=yica_config,
            warmup_iters=warmup_iters,
            profile_iters=profile_iters,
            verbose=verbose,
            use_graph_dataset=use_graph_dataset
        )
```

#### 1.2 YICA Backend核心接口
```python
class YICABackendIntegration:
    """YICA Backend核心集成类"""
    
    def __init__(self, hardware_config: YICAHardwareConfig):
        self.hw_config = hardware_config
        self.kernel_registry = YICAKernelRegistry()
        self.profiler = YICAProfiler()
        self.optimizer = YICAGraphOptimizer()
    
    def optimize_graphs(self, graphs: List[KNGraph]) -> KNGraph:
        """优化计算图列表，返回最佳YICA优化图"""
        pass
    
    def compile_yica_kernels(self, graph: KNGraph) -> YICACompiledGraph:
        """编译YICA专用kernel"""
        pass
    
    def profile_and_select(self, compiled_graphs: List[YICACompiledGraph]) -> YICACompiledGraph:
        """性能测试并选择最佳图"""
        pass
```

### 2. YICA Kernel注册系统

#### 2.1 Kernel映射机制
```python
class YICAKernelRegistry:
    """YICA Kernel注册表"""
    
    def __init__(self):
        self.operation_to_kernel = {
            # 核心操作映射
            'matmul': YICAMatMulKernel,
            'element_ops': YICAElementOpsKernel,  
            'reduction': YICAReductionKernel,
            'rms_norm': YICARMSNormKernel,
            'all_reduce': YICAAllReduceKernel,
            'chunk': YICAChunkKernel,
            'customized': YICACustomizedKernel,
            # 更多操作...
        }
    
    def get_kernel_for_operation(self, op_type: str, op_params: dict) -> YICAKernelBase:
        """根据操作类型和参数获取对应的YICA kernel"""
        pass
    
    def register_custom_kernel(self, op_type: str, kernel_class: type):
        """注册用户自定义kernel"""
        pass
```

#### 2.2 Kernel基类设计
```python
class YICAKernelBase:
    """YICA Kernel基类"""
    
    def __init__(self, hardware_config: YICAHardwareConfig):
        self.hw_config = hardware_config
    
    def analyze_operation(self, op_node) -> YICAAnalysisResult:
        """分析操作的YICA适配性"""
        pass
    
    def generate_yica_code(self, op_node) -> str:
        """生成YICA优化的kernel代码"""
        pass
    
    def estimate_performance(self, op_node) -> float:
        """估算性能"""
        pass
    
    def compile(self, code: str) -> YICACompiledKernel:
        """编译kernel"""
        pass
```

### 3. YICA图优化流程

#### 3.1 图分析和转换
```python
class YICAGraphOptimizer:
    """YICA图优化器"""
    
    def analyze_graph_compatibility(self, graph: KNGraph) -> YICACompatibilityReport:
        """分析图的YICA兼容性"""
        return YICACompatibilityReport(
            cim_friendliness_score=0.85,
            spm_utilization_potential=0.92,
            parallelization_opportunities=[...],
            bottleneck_operations=[...],
            optimization_recommendations=[...]
        )
    
    def apply_yica_optimizations(self, graph: KNGraph) -> KNGraph:
        """应用YICA特定优化"""
        # 1. 操作融合优化
        graph = self._apply_operation_fusion(graph)
        
        # 2. CIM阵列并行化
        graph = self._apply_cim_parallelization(graph)
        
        # 3. SPM内存优化  
        graph = self._apply_spm_optimization(graph)
        
        # 4. 数据流优化
        graph = self._apply_dataflow_optimization(graph)
        
        return graph
    
    def generate_yica_variants(self, base_graph: KNGraph) -> List[KNGraph]:
        """生成不同配置的YICA优化变体"""
        variants = []
        
        # 不同的CIM并行度配置
        for cim_arrays in [2, 4, 8, 16]:
            variant = self._create_cim_variant(base_graph, cim_arrays)
            variants.append(variant)
        
        # 不同的SPM配置
        for spm_strategy in ['conservative', 'aggressive', 'balanced']:
            variant = self._create_spm_variant(base_graph, spm_strategy)
            variants.append(variant)
        
        return variants
```

#### 3.2 YICA专用Profiler
```python
class YICAProfiler:
    """YICA性能分析器"""
    
    def __init__(self, hardware_config: YICAHardwareConfig):
        self.hw_config = hardware_config
        self.performance_model = YICAPerformanceModel(hardware_config)
    
    def profile_graphs(self, graphs: List[YICACompiledGraph], 
                       warmup_iters: int, profile_iters: int) -> List[YICAProfileResult]:
        """对YICA编译图进行性能测试"""
        results = []
        
        for graph in graphs:
            # 实际执行测试
            if self._has_yica_hardware():
                result = self._profile_on_hardware(graph, warmup_iters, profile_iters)
            else:
                # 仿真模式
                result = self._profile_simulation(graph, warmup_iters, profile_iters)
            
            results.append(result)
        
        return results
    
    def select_best_graph(self, profile_results: List[YICAProfileResult]) -> YICACompiledGraph:
        """根据性能结果选择最佳图"""
        best_result = min(profile_results, key=lambda r: r.average_latency)
        return best_result.compiled_graph
```

### 4. C++ Kernel绑定设计

#### 4.1 Python绑定接口
```python
# yica_kernel_bindings.py
"""YICA C++ Kernel的Python绑定"""

try:
    from ._cython.yica_kernels import (
        YICAMatMulOp,
        YICAElementOpsOp, 
        YICAReductionOp,
        YICARMSNormOp,
        YICAAllReduceOp,
        YICAChunkOp,
        YICACustomizedOp,
        # 更多kernel类...
    )
    YICA_KERNELS_AVAILABLE = True
except ImportError:
    # 提供stub实现
    YICA_KERNELS_AVAILABLE = False
    from .yica_kernel_stubs import *

class YICAMatMulKernel(YICAKernelBase):
    """YICA矩阵乘法Kernel"""
    
    def __init__(self, hardware_config: YICAHardwareConfig):
        super().__init__(hardware_config)
        if YICA_KERNELS_AVAILABLE:
            self._cpp_kernel = YICAMatMulOp()
        else:
            self._cpp_kernel = None
    
    def generate_yica_code(self, op_node) -> str:
        """生成YICA矩阵乘法代码"""
        if self._cpp_kernel:
            return self._cpp_kernel.generate_optimized_code(
                input_shapes=op_node.input_shapes,
                cim_config=self.hw_config.to_dict()
            )
        else:
            # Fallback到Python实现
            return self._generate_python_fallback(op_node)
```

#### 4.2 Cython绑定文件
```cython
# _cython/yica_kernels.pyx
"""YICA C++ Kernels的Cython绑定"""

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map

cdef extern from "yirage/kernel/yica_matmul.h" namespace "yirage::kernel":
    cdef cppclass YICAMatMulOp:
        YICAMatMulOp(...)
        string generate_optimized_code(vector[int] input_shapes, map[string, float] config)
        float estimate_performance(...)
        bint profile(...)

# 更多C++类的绑定...

cdef class PyYICAMatMulOp:
    cdef YICAMatMulOp* c_kernel
    
    def __cinit__(self, ...):
        self.c_kernel = new YICAMatMulOp(...)
    
    def __dealloc__(self):
        del self.c_kernel
    
    def generate_optimized_code(self, input_shapes, config):
        return self.c_kernel.generate_optimized_code(input_shapes, config).decode('utf-8')
```

### 5. 配置和缓存集成

#### 5.1 YICA配置管理
```python
@dataclass
class YICABackendConfig:
    """YICA Backend配置"""
    
    # 硬件配置
    hardware_config: YICAHardwareConfig = field(default_factory=YICAHardwareConfig)
    
    # 优化策略
    optimization_strategy: str = "throughput_optimal"  # latency_optimal, memory_optimal, energy_optimal
    
    # 编译选项
    enable_kernel_fusion: bool = True
    enable_cim_parallelization: bool = True 
    enable_spm_optimization: bool = True
    
    # 调试选项
    enable_verbose_logging: bool = False
    enable_performance_analysis: bool = True
    dump_generated_code: bool = False
    
    # 缓存选项
    enable_compilation_cache: bool = True
    cache_directory: str = "~/.yirage/yica_cache"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'YICABackendConfig':
        return cls(**config_dict)
```

#### 5.2 缓存系统集成
```python
class YICAGraphCache:
    """YICA图缓存系统"""
    
    def __init__(self, cache_dir: str = "~/.yirage/yica_cache"):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cached_graph(self, graph_hash: str, yica_config: YICABackendConfig) -> Optional[YICACompiledGraph]:
        """获取缓存的图"""
        cache_key = f"{graph_hash}_{hash(yica_config.to_dict())}"
        cache_file = self.cache_dir / f"{cache_key}.yica"
        
        if cache_file.exists():
            return self._load_cached_graph(cache_file)
        return None
    
    def store_cached_graph(self, graph_hash: str, yica_config: YICABackendConfig, 
                          compiled_graph: YICACompiledGraph):
        """存储编译好的图到缓存"""
        cache_key = f"{graph_hash}_{hash(yica_config.to_dict())}"
        cache_file = self.cache_dir / f"{cache_key}.yica"
        self._save_cached_graph(cache_file, compiled_graph)
```

## 🔧 实现接口规范

### 1. 主要集成函数

```python
def _optimize_with_yica(self, all_graphs: List[KNGraph], 
                       yica_config: YICABackendConfig,
                       warmup_iters: int, profile_iters: int,
                       verbose: bool, use_graph_dataset: bool) -> KNGraph:
    """
    YICA backend的主要优化函数
    
    Args:
        all_graphs: 搜索得到的候选图列表
        yica_config: YICA配置参数
        warmup_iters: 预热迭代次数
        profile_iters: 性能测试迭代次数
        verbose: 是否输出详细信息
        use_graph_dataset: 是否使用图数据集缓存
    
    Returns:
        优化后的最佳KNGraph
    """
    
    # 1. 初始化YICA backend
    yica_backend = YICABackendIntegration(yica_config.hardware_config)
    
    # 2. 检查缓存
    if use_graph_dataset:
        cached_graph = self._check_yica_cache(all_graphs, yica_config)
        if cached_graph:
            return cached_graph
    
    # 3. YICA图优化
    if verbose:
        print(f"Applying YICA optimizations to {len(all_graphs)} muGraphs...")
    
    yica_optimized_graphs = []
    for graph in all_graphs:
        optimized = yica_backend.apply_yica_optimizations(graph)
        yica_optimized_graphs.append(optimized)
    
    # 4. 生成YICA优化变体
    all_yica_variants = []
    for graph in yica_optimized_graphs:
        variants = yica_backend.generate_yica_variants(graph)
        all_yica_variants.extend(variants)
    
    # 5. 编译YICA kernels
    if verbose:
        print(f"Compiling {len(all_yica_variants)} YICA kernel variants...")
    
    compiled_graphs = []
    for variant in all_yica_variants:
        try:
            compiled = yica_backend.compile_yica_kernels(variant)
            compiled_graphs.append(compiled)
        except Exception as e:
            if verbose:
                print(f"Compilation failed for variant: {e}")
            continue
    
    if not compiled_graphs:
        raise RuntimeError("No YICA kernels could be compiled successfully")
    
    # 6. 性能测试和选择
    if verbose:
        print(f"Profiling {len(compiled_graphs)} compiled YICA graphs...")
    
    best_graph = yica_backend.profile_and_select(
        compiled_graphs, warmup_iters, profile_iters
    )
    
    # 7. 设置backend标识
    best_graph.backend = "yica"
    
    # 8. 缓存结果
    if use_graph_dataset:
        self._cache_yica_result(best_graph, yica_config)
    
    if verbose:
        print(f"Selected best YICA muGraph with {best_graph.performance_metrics['latency']:.3f}ms latency")
    
    return best_graph
```

### 2. 错误处理和回退机制

```python
def _optimize_with_yica_safe(self, *args, **kwargs) -> KNGraph:
    """
    安全的YICA优化，包含完整的错误处理和回退机制
    """
    try:
        return self._optimize_with_yica(*args, **kwargs)
    except YICAHardwareNotAvailableError:
        print("⚠️  YICA hardware not available, falling back to simulation mode")
        return self._optimize_with_yica_simulation(*args, **kwargs)
    except YICACompilationError as e:
        print(f"⚠️  YICA compilation failed: {e}, falling back to CUDA backend")
        return self.superoptimize(backend="cuda", **kwargs)
    except Exception as e:
        print(f"❌ YICA backend failed: {e}, falling back to CUDA backend")
        return self.superoptimize(backend="cuda", **kwargs)
```

## ✅ 验收标准

### 功能验收标准
1. ✅ 用户可以使用`backend="yica"`无缝切换到YICA优化
2. ✅ YICA backend能够处理所有demo中的计算图
3. ✅ 性能测试显示YICA相比CUDA有明显提升
4. ✅ 错误处理机制完善，能优雅回退到其他backend

### 性能验收标准  
1. ✅ Gated MLP: YICA相比CUDA加速≥1.5x
2. ✅ Group Query Attention: YICA相比CUDA加速≥1.8x
3. ✅ RMS Norm: YICA相比CUDA加速≥2.0x
4. ✅ LoRA: YICA相比CUDA加速≥1.6x

### 兼容性验收标准
1. ✅ 不影响现有CUDA/Triton backend功能
2. ✅ 缓存系统正常工作
3. ✅ 所有现有demo和benchmark正常运行
4. ✅ 支持配置文件和命令行参数

---

**📝 设计总结**: 本设计文档提供了完整的YICA backend集成方案，涵盖了从接口设计到实现细节的各个方面。设计遵循现有架构模式，确保无缝集成和向后兼容性。

**🎯 下一步**: 获得设计批准后，将进入Development Phase，按照此设计文档逐步实施YICA backend的完整集成。 
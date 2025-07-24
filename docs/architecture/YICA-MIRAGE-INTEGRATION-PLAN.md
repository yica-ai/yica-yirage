# YICA与Mirage集成方案

## 项目目标

**借助Mirage已有的代码基础，增加YICA特定的优化逻辑，为YICA架构提供后端支持，最终输出优化后的Triton代码。**

## 技术路线

```
Mirage计算图 → YICA优化器 → Mirage搜索引擎 → Triton Transpiler → 优化的Triton代码
```

## 核心组件设计

### 1. YICA后端集成到Mirage

在Mirage现有的后端系统中添加YICA支持：

```python
# mirage/python/mirage/kernel.py 
def superoptimize(self, backend="yica", **kwargs):
    # 现有代码支持 "cuda", "nki", "triton"
    # 新增 "yica" 后端支持
    
    if backend == "yica":
        from .yica_optimizer import YICAMirageOptimizer
        
        # 1. YICA特定的搜索和优化
        yica_optimizer = YICAMirageOptimizer(self.cygraph)
        optimized_graphs = yica_optimizer.optimize_for_yica(
            yica_config=kwargs.get('yica_config'),
            optimization_objectives=kwargs.get('objectives', ['latency'])
        )
        
        # 2. 选择最优图
        best_graph = yica_optimizer.select_best_graph(optimized_graphs)
        
        # 3. 使用Triton transpiler生成最终代码
        return self._generate_triton_from_yica_optimized(best_graph)
```

### 2. YICA优化器组件

创建专门的YICA优化逻辑：

```python
# mirage/python/mirage/yica_optimizer.py

class YICAMirageOptimizer:
    """
    YICA架构专用的Mirage优化器
    利用Mirage的搜索引擎，增加YICA特定的优化策略
    """
    
    def __init__(self, mirage_graph):
        self.mirage_graph = mirage_graph
        self.yica_analyzer = YICAArchitectureAnalyzer()
        self.search_space = YICASearchSpace()
        
    def optimize_for_yica(self, yica_config, optimization_objectives):
        """
        针对YICA架构进行专门优化
        """
        # 1. 分析计算图的YICA适配性
        analysis = self.yica_analyzer.analyze_graph(
            self.mirage_graph, yica_config
        )
        
        # 2. 生成YICA特定的搜索空间
        search_candidates = self.search_space.generate_yica_optimized_variants(
            self.mirage_graph, analysis
        )
        
        # 3. 应用YICA特定的优化策略
        optimized_graphs = []
        for candidate in search_candidates:
            # CIM阵列并行化优化
            cim_optimized = self._optimize_for_cim_parallelism(candidate)
            
            # SPM内存层次优化
            spm_optimized = self._optimize_spm_memory_hierarchy(cim_optimized)
            
            # 存算一体计算模式优化
            pim_optimized = self._optimize_pim_computation(spm_optimized)
            
            optimized_graphs.append(pim_optimized)
            
        return optimized_graphs
```

### 3. YICA架构分析器

```python
# mirage/python/mirage/yica_analyzer.py

class YICAArchitectureAnalyzer:
    """
    分析Mirage计算图对YICA架构的适配性
    """
    
    def analyze_graph(self, mirage_graph, yica_config):
        analysis = YICAAnalysisResult()
        
        # 分析计算密集度
        analysis.compute_intensity = self._analyze_compute_intensity(mirage_graph)
        
        # 分析内存访问模式
        analysis.memory_pattern = self._analyze_memory_access_pattern(mirage_graph)
        
        # 分析CIM友好度
        analysis.cim_friendliness = self._analyze_cim_friendliness(mirage_graph)
        
        # 分析并行化潜力
        analysis.parallelization_potential = self._analyze_parallelization(mirage_graph)
        
        return analysis
    
    def _analyze_cim_friendliness(self, graph):
        """
        分析操作对CIM阵列的友好程度
        """
        cim_friendly_ops = ['matmul', 'conv2d', 'elementwise_mul', 'elementwise_add']
        friendly_score = 0
        
        for op in graph.operators:
            if op.op_type in cim_friendly_ops:
                friendly_score += self._get_op_weight(op)
                
        return friendly_score / len(graph.operators)
```

### 4. YICA搜索空间定义

```python
# mirage/python/mirage/yica_search_space.py

class YICASearchSpace:
    """
    定义YICA架构特定的优化搜索空间
    """
    
    def generate_yica_optimized_variants(self, graph, analysis):
        variants = []
        
        # 1. CIM阵列映射策略
        cim_mappings = self._generate_cim_mapping_strategies(graph, analysis)
        
        # 2. SPM数据布局策略  
        spm_layouts = self._generate_spm_layout_strategies(graph, analysis)
        
        # 3. 计算调度策略
        scheduling_strategies = self._generate_scheduling_strategies(graph, analysis)
        
        # 组合不同策略生成候选方案
        for cim_map in cim_mappings:
            for spm_layout in spm_layouts:
                for schedule in scheduling_strategies:
                    variant = self._create_variant(graph, cim_map, spm_layout, schedule)
                    variants.append(variant)
                    
        return variants
    
    def _generate_cim_mapping_strategies(self, graph, analysis):
        """
        生成不同的CIM阵列映射策略
        """
        strategies = []
        
        # 策略1: 最大并行化 - 尽可能多的CIM阵列同时工作
        strategies.append(CIMStrategy.MAX_PARALLELISM)
        
        # 策略2: 负载均衡 - 平衡各CIM阵列的工作负载
        strategies.append(CIMStrategy.LOAD_BALANCED)
        
        # 策略3: 内存优化 - 优化SPM访问模式
        strategies.append(CIMStrategy.MEMORY_OPTIMIZED)
        
        return strategies
```

### 5. 扩展Mirage的Triton Transpiler

```cpp
// mirage/src/triton_transpiler/yica_transpile.cc

class YICATritonTranspiler : public TritonTranspiler {
    /*
    扩展现有的Triton transpiler，增加YICA特定的代码生成
    */
    
public:
    TritonTranspileResult transpile_yica_optimized_graph(
        kernel::Graph const* graph,
        YICAOptimizationResult const& yica_result) {
            
        // 1. 基于YICA优化结果调整Triton代码生成
        auto adjusted_config = adjust_triton_config_for_yica(yica_result);
        
        // 2. 生成YICA特定的内核优化
        auto yica_kernels = generate_yica_optimized_kernels(graph, yica_result);
        
        // 3. 调用基础Triton transpiler
        auto base_result = TritonTranspiler::transpile_ugraph();
        
        // 4. 注入YICA优化代码
        auto enhanced_code = inject_yica_optimizations(base_result.code, yica_kernels);
        
        TritonTranspileResult result;
        result.code = enhanced_code;
        result.output_shapes = base_result.output_shapes;
        
        return result;
    }
    
private:
    std::string inject_yica_optimizations(
        std::string const& base_triton_code,
        YICAKernelOptimizations const& yica_opts) {
            
        std::stringstream enhanced_code;
        
        // 添加YICA特定的导入
        enhanced_code << "# YICA-optimized Triton kernels\n";
        enhanced_code << "import triton\n";
        enhanced_code << "import triton.language as tl\n";
        enhanced_code << "from yica_runtime import CIMArray, SPMManager\n\n";
        
        // 注入CIM阵列优化
        enhanced_code << generate_cim_optimization_code(yica_opts);
        
        // 注入SPM内存优化
        enhanced_code << generate_spm_optimization_code(yica_opts);
        
        // 添加原始Triton代码
        enhanced_code << base_triton_code;
        
        return enhanced_code.str();
    }
};
```

### 6. YICA运行时支持

```python
# mirage/include/mirage/triton_transpiler/runtime/yica_runtime.py

class CIMArray:
    """
    CIM阵列运行时模拟
    在Triton中模拟YICA的存算一体计算
    """
    
    @staticmethod
    @triton.jit
    def cim_matmul(a_ptr, b_ptr, c_ptr, M, N, K, cim_id):
        """
        CIM阵列矩阵乘法
        针对YICA架构优化的矩阵乘法实现
        """
        # 获取CIM阵列特定的线程块配置
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        # YICA特定的内存访问模式
        BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
        
        # 考虑CIM阵列的并行度
        cim_offset = cim_id * (M * N // 4)  # 假设4个CIM阵列
        
        # 优化的内存加载模式
        a_block = tl.load(a_ptr + cim_offset, mask=..., other=0.0)
        b_block = tl.load(b_ptr + cim_offset, mask=..., other=0.0)
        
        # 存算一体计算
        c_block = tl.dot(a_block, b_block)
        
        # 写回结果
        tl.store(c_ptr + cim_offset, c_block, mask=...)

class SPMManager:
    """
    SPM内存管理器
    优化Triton中的共享内存使用模式
    """
    
    @staticmethod
    @triton.jit  
    def optimized_spm_access(data_ptr, spm_size, access_pattern):
        """
        优化的SPM访问模式
        根据YICA分析结果优化内存访问
        """
        # 实现YICA特定的内存访问优化
        pass
```

## 集成流程

### 步骤1: 扩展Mirage后端支持

```bash
# 在mirage/python/mirage/kernel.py中添加YICA后端
# 在mirage/python/mirage/目录下添加YICA相关模块：
# - yica_optimizer.py
# - yica_analyzer.py  
# - yica_search_space.py
```

### 步骤2: 扩展Triton Transpiler

```bash
# 在mirage/src/triton_transpiler/目录下添加：
# - yica_transpile.cc
# - yica_transpile.h

# 在mirage/include/mirage/triton_transpiler/runtime/目录下添加：
# - yica_runtime.py
```

### 步骤3: 集成到构建系统

```cmake
# 在mirage/CMakeLists.txt中添加YICA组件
set(YICA_TRANSPILER_SRCS
    src/triton_transpiler/yica_transpile.cc
)

# 添加到目标
target_sources(mirage_runtime PRIVATE ${YICA_TRANSPILER_SRCS})
```

### 步骤4: Python绑定

```python
# 在mirage/python/mirage/_cython/core.pyx中添加YICA函数绑定
def generate_yica_triton_program(CyKNGraph input_graph, *, 
                                int target_cc,
                                dict yica_config) -> dict:
    # 调用C++的YICA Triton transpiler
    pass
```

## 使用示例

```python
import mirage as mi

# 创建计算图
graph = mi.new_kernel_graph()
X = graph.new_input(dims=(1024, 1024), dtype=mi.float16)
W = graph.new_input(dims=(1024, 1024), dtype=mi.float16) 
Y = graph.matmul(X, W)
graph.mark_output(Y)

# 使用YICA后端优化
yica_config = {
    'num_cim_arrays': 4,
    'spm_size_kb': 256,
    'optimization_objectives': ['latency', 'energy_efficiency']
}

optimized_graph = graph.superoptimize(
    backend="yica",
    yica_config=yica_config
)

# 生成优化后的Triton代码
triton_code = mi.generate_triton_program(
    optimized_graph.cygraph,
    target_cc=90
)["code"]

# 保存生成的代码
with open("yica_optimized_kernel.py", "w") as f:
    f.write(triton_code)
```

## 技术优势

1. **复用Mirage生态** - 利用成熟的搜索引擎和优化框架
2. **YICA特化优化** - 针对存算一体架构的专门优化策略  
3. **Triton代码输出** - 生成高性能的Triton GPU内核
4. **渐进式集成** - 可以逐步扩展和完善YICA支持
5. **后端兼容性** - 与现有的CUDA/NKI后端并存

这个方案更符合你们的实际需求：**借助Mirage → 增加YICA优化 → 输出Triton代码**。 
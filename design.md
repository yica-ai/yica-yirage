# YZ自动优化器设计文档 - 基于Mirage改造的YICA算子超优化方案

## 项目背景与目标

基于亿铸科技存算一体AI大算力芯片(YICA)架构，通过改造Mirage超优化器来实现：

1. **Mirage-YICA适配** - 改造Mirage以支持YICA架构的算子生成和优化
2. **YICA后端集成** - 开发YICA专用的编译器后端和代码生成器
3. **算子超优化** - 利用Mirage的搜索空间探索能力自动发现YICA最优算子实现

## 核心设计理念

### 1. Mirage-YICA架构适配层

改造Mirage以支持YICA存算一体架构的算子生成和优化：

```python
class MirageYICAAdapter:
    """
    Mirage到YICA架构的适配器
    改造Mirage原有的GPU代码生成，转为YICA指令生成
    """
    
    def __init__(self, yica_config: YICAConfig):
        self.yica_config = yica_config
        self.yica_transpiler = YICATranspiler()
        self.memory_hierarchy = YICAMemoryHierarchy(yica_config)
        
    def adapt_mirage_graph(self, mirage_graph: 'mirage.Graph') -> YICAOptimizedGraph:
        """
        将Mirage计算图适配到YICA架构
        """
        # 1. 分析计算图的内存访问模式
        memory_pattern = self._analyze_memory_pattern(mirage_graph)
        
        # 2. 映射到YICA的存储层次结构
        memory_mapping = self._map_to_yica_memory(memory_pattern)
        
        # 3. 生成YICA指令序列
        yica_instructions = self._generate_yica_instructions(mirage_graph, memory_mapping)
        
        # 4. 优化指令调度
        optimized_instructions = self._optimize_instruction_schedule(yica_instructions)
        
        return YICAOptimizedGraph(
            instructions=optimized_instructions,
            memory_mapping=memory_mapping,
            performance_estimate=self._estimate_performance(optimized_instructions)
        )
    
    def _analyze_memory_pattern(self, graph: 'mirage.Graph') -> MemoryAccessPattern:
        """
        分析Mirage图的内存访问模式
        """
        patterns = []
        for op in graph.operators:
            if op.op_type in ['matmul', 'attention']:
                # 矩阵运算，适合CIM阵列
                patterns.append(MemoryPattern.CIM_FRIENDLY)
            elif op.op_type in ['elementwise', 'activation']:
                # 元素级运算，适合SPM
                patterns.append(MemoryPattern.SPM_FRIENDLY)
            elif op.op_type in ['reduction', 'norm']:
                # 归约运算，需要跨CIM通信
                patterns.append(MemoryPattern.REDUCTION_HEAVY)
        
        return MemoryAccessPattern(patterns)
    
    def _map_to_yica_memory(self, pattern: MemoryAccessPattern) -> YICAMemoryMapping:
        """
        映射到YICA的存储层次结构
        SPM -> CIM Die -> DRAM
        """
        mapping = YICAMemoryMapping()
        
        for op_pattern in pattern.patterns:
            if op_pattern == MemoryPattern.CIM_FRIENDLY:
                # 数据直接存储在CIM阵列中
                mapping.add_cim_storage(size=self._estimate_cim_usage(op_pattern))
            elif op_pattern == MemoryPattern.SPM_FRIENDLY:
                # 使用SPM作为临时存储
                mapping.add_spm_storage(size=self._estimate_spm_usage(op_pattern))
            elif op_pattern == MemoryPattern.REDUCTION_HEAVY:
                # 需要DRAM支持大规模数据移动
                mapping.add_dram_access(size=self._estimate_dram_usage(op_pattern))
        
        return mapping
    
    def _generate_yica_instructions(self, 
                                  graph: 'mirage.Graph', 
                                  memory_mapping: YICAMemoryMapping) -> List[YICAInstruction]:
        """
        生成YICA指令序列
        """
        instructions = []
        
        for op in graph.operators:
            if op.op_type == 'matmul':
                # 生成CIM矩阵乘法指令
                instr = YICAInstruction(
                    type='CIM_MATMUL',
                    operands=self._map_operands(op.input_tensors, memory_mapping),
                    output=self._map_output(op.output_tensors[0], memory_mapping),
                    config=CIMConfig(
                        array_size=self.yica_config.cim_array_size,
                        precision=op.precision,
                        parallel_factor=self._calculate_parallelism(op)
                    )
                )
                instructions.append(instr)
                
            elif op.op_type == 'elementwise':
                # 生成SPM元素级运算指令
                instr = YICAInstruction(
                    type='SPM_ELEMENTWISE',
                    operands=self._map_operands(op.input_tensors, memory_mapping),
                    output=self._map_output(op.output_tensors[0], memory_mapping),
                    config=SPMConfig(
                        vector_width=self.yica_config.spm_vector_width,
                        operation=op.elementwise_type
                    )
                )
                instructions.append(instr)
                
            elif op.op_type == 'reduction':
                # 生成跨CIM通信和归约指令
                instr = YICAInstruction(
                    type='CIM_REDUCTION',
                    operands=self._map_operands(op.input_tensors, memory_mapping),
                    output=self._map_output(op.output_tensors[0], memory_mapping),
                    config=ReductionConfig(
                        reduction_tree=self._build_reduction_tree(op),
                        communication_pattern=self._optimize_communication(op)
                    )
                )
                instructions.append(instr)
        
        return instructions
```

### 2. YICA后端编译器

开发专门针对YICA架构的Mirage后端编译器：

```python
class YICAMirageBackend:
    """
    YICA架构的Mirage编译器后端
    替换原有的Triton后端，直接生成YICA指令
    """
    
    def __init__(self, yica_config: YICAConfig):
        self.yica_config = yica_config
        self.adapter = MirageYICAAdapter(yica_config)
        self.code_generator = YICACodeGenerator()
        self.performance_model = YICAPerformanceModel()
        
    def compile_mirage_graph(self, 
                           mirage_graph: 'mirage.Graph',
                           optimization_objectives: List[str]) -> YICACompiledKernel:
        """
        编译Mirage图为YICA可执行代码
        """
        # 1. 适配Mirage图到YICA架构
        yica_graph = self.adapter.adapt_mirage_graph(mirage_graph)
        
        # 2. 执行YICA特定的优化
        optimized_graph = self._optimize_for_yica(yica_graph, optimization_objectives)
        
        # 3. 生成YICA汇编代码
        assembly_code = self.code_generator.generate_assembly(optimized_graph)
        
        # 4. 生成运行时配置
        runtime_config = self._generate_runtime_config(optimized_graph)
        
        return YICACompiledKernel(
            assembly_code=assembly_code,
            runtime_config=runtime_config,
            performance_profile=optimized_graph.performance_estimate
        )
    
    def _optimize_for_yica(self, 
                          yica_graph: YICAOptimizedGraph,
                          objectives: List[str]) -> YICAOptimizedGraph:
        """
        执行YICA架构特定的优化
        """
        optimizations = []
        
        if "latency" in objectives:
            # 延迟优化：最小化CIM间通信
            optimizations.append(self._minimize_cim_communication)
            
        if "memory_efficiency" in objectives:
            # 内存效率优化：最大化SPM利用率
            optimizations.append(self._maximize_spm_utilization)
            
        if "energy_efficiency" in objectives:
            # 能效优化：减少DRAM访问
            optimizations.append(self._minimize_dram_access)
            
        if "throughput" in objectives:
            # 吞吐量优化：最大化CIM阵列并行度
            optimizations.append(self._maximize_cim_parallelism)
        
        # 应用所有优化
        optimized_graph = yica_graph
        for optimization in optimizations:
            optimized_graph = optimization(optimized_graph)
            
        return optimized_graph
    
    def _minimize_cim_communication(self, graph: YICAOptimizedGraph) -> YICAOptimizedGraph:
        """
        最小化CIM间通信开销
        """
        # 分析数据依赖关系
        dependency_graph = self._build_dependency_graph(graph)
        
        # 重新调度指令以减少通信
        reordered_instructions = self._reorder_for_locality(
            graph.instructions, dependency_graph
        )
        
        # 插入数据预取指令
        optimized_instructions = self._insert_prefetch_instructions(
            reordered_instructions
        )
        
        return graph.update_instructions(optimized_instructions)
    
    def _maximize_spm_utilization(self, graph: YICAOptimizedGraph) -> YICAOptimizedGraph:
        """
        最大化SPM利用率
        """
        # 分析SPM使用模式
        spm_usage_pattern = self._analyze_spm_usage(graph)
        
        # 重新分配SPM空间
        optimized_mapping = self._optimize_spm_allocation(
            graph.memory_mapping, spm_usage_pattern
        )
        
        # 插入数据重用指令
        reuse_optimized_instructions = self._optimize_data_reuse(
            graph.instructions, optimized_mapping
        )
        
        return graph.update_memory_mapping(optimized_mapping).update_instructions(
            reuse_optimized_instructions
        )
    
    def _minimize_dram_access(self, graph: YICAOptimizedGraph) -> YICAOptimizedGraph:
        """
        减少DRAM访问以提高能效
        """
        # 识别可以缓存的数据
        cacheable_data = self._identify_cacheable_data(graph)
        
        # 插入缓存管理指令
        cache_managed_instructions = self._insert_cache_management(
            graph.instructions, cacheable_data
        )
        
        # 合并DRAM访问
        batched_instructions = self._batch_dram_accesses(cache_managed_instructions)
        
        return graph.update_instructions(batched_instructions)
    
    def _maximize_cim_parallelism(self, graph: YICAOptimizedGraph) -> YICAOptimizedGraph:
        """
        最大化CIM阵列并行度
        """
        # 分析可并行执行的指令
        parallel_groups = self._identify_parallel_instructions(graph.instructions)
        
        # 重新调度以最大化并行度
        parallel_scheduled = self._schedule_for_parallelism(
            graph.instructions, parallel_groups
        )
        
        # 插入同步指令
        synchronized_instructions = self._insert_synchronization(parallel_scheduled)
        
        return graph.update_instructions(synchronized_instructions)
```

### 3. Mirage超优化搜索引擎

改造Mirage的搜索引擎以支持YICA架构的搜索空间：

```python
class YICAMirageSearchEngine:
    """
    针对YICA架构改造的Mirage搜索引擎
    """
    
    def __init__(self, yica_backend: YICAMirageBackend):
        self.yica_backend = yica_backend
        self.search_space = YICASearchSpace()
        self.performance_model = YICAPerformanceModel()
        
    def superoptimize_for_yica(self, 
                             mirage_graph: 'mirage.Graph',
                             objectives: List[str],
                             search_budget: int = 1000) -> YICAOptimizedKernel:
        """
        针对YICA架构执行超优化搜索
        """
        # 1. 定义YICA特定的搜索空间
        search_space = self._define_yica_search_space(mirage_graph)
        
        # 2. 初始化搜索状态
        search_state = SearchState(
            best_kernel=None,
            best_performance=float('inf'),
            explored_configs=set(),
            iteration=0
        )
        
        # 3. 执行多轮搜索
        for iteration in range(search_budget):
            # 生成候选配置
            candidate_config = self._sample_configuration(search_space, search_state)
            
            # 编译候选配置
            candidate_kernel = self._compile_configuration(
                mirage_graph, candidate_config, objectives
            )
            
            # 评估性能
            performance = self._evaluate_performance(candidate_kernel, objectives)
            
            # 更新最优解
            if performance < search_state.best_performance:
                search_state.best_kernel = candidate_kernel
                search_state.best_performance = performance
                
            # 更新搜索策略
            search_state = self._update_search_strategy(search_state, candidate_config, performance)
            
        return search_state.best_kernel
    
    def _define_yica_search_space(self, graph: 'mirage.Graph') -> YICASearchSpace:
        """
        定义YICA架构特定的搜索空间
        """
        return YICASearchSpace(
            # CIM阵列配置
            cim_array_configs=[
                CIMArrayConfig(rows=256, cols=256, precision="int8"),
                CIMArrayConfig(rows=512, cols=512, precision="int4"),
                CIMArrayConfig(rows=1024, cols=1024, precision="int2"),
            ],
            
            # SPM分配策略
            spm_allocation_strategies=[
                SPMStrategy.TEMPORAL_REUSE,      # 时间重用优先
                SPMStrategy.SPATIAL_LOCALITY,    # 空间局部性优先
                SPMStrategy.BALANCED,            # 平衡策略
            ],
            
            # 数据切分模式
            data_tiling_patterns=[
                TilingPattern(tile_sizes=[64, 64], order="row_major"),
                TilingPattern(tile_sizes=[128, 128], order="col_major"),
                TilingPattern(tile_sizes=[256, 256], order="zigzag"),
            ],
            
            # 并行化策略
            parallelization_schemes=[
                ParallelScheme.DATA_PARALLEL,    # 数据并行
                ParallelScheme.MODEL_PARALLEL,   # 模型并行
                ParallelScheme.PIPELINE_PARALLEL, # 流水线并行
                ParallelScheme.HYBRID,           # 混合并行
            ],
            
            # 通信模式
            communication_patterns=[
                CommPattern.RING_ALLREDUCE,      # 环形全归约
                CommPattern.TREE_REDUCE,         # 树状归约
                CommPattern.BUTTERFLY,           # 蝶式通信
                CommPattern.DIRECT_SEND,         # 直接发送
            ]
        )
    
    def _sample_configuration(self, 
                            search_space: YICASearchSpace, 
                            search_state: SearchState) -> YICAConfiguration:
        """
        根据搜索状态采样配置
        """
        if search_state.iteration < 100:
            # 前期使用随机采样
            return self._random_sample(search_space)
        elif search_state.iteration < 500:
            # 中期使用基于梯度的采样
            return self._gradient_based_sample(search_space, search_state)
        else:
            # 后期使用局部搜索
            return self._local_search_sample(search_space, search_state)
    
    def _compile_configuration(self, 
                             graph: 'mirage.Graph',
                             config: YICAConfiguration,
                             objectives: List[str]) -> YICACompiledKernel:
        """
        编译特定配置的内核
        """
        # 应用配置到Mirage图
        configured_graph = self._apply_configuration(graph, config)
        
        # 使用YICA后端编译
        compiled_kernel = self.yica_backend.compile_mirage_graph(
            configured_graph, objectives
        )
        
        return compiled_kernel
    
    def _evaluate_performance(self, 
                            kernel: YICACompiledKernel,
                            objectives: List[str]) -> float:
        """
        评估内核性能
        """
        performance_metrics = self.performance_model.evaluate(kernel)
        
        # 多目标优化的加权评分
        score = 0.0
        for objective in objectives:
            if objective == "latency":
                score += performance_metrics.latency * 0.4
            elif objective == "memory_efficiency":
                score += (1.0 - performance_metrics.memory_utilization) * 0.3
            elif objective == "energy_efficiency":
                score += performance_metrics.energy_consumption * 0.2
            elif objective == "throughput":
                score += (1.0 / performance_metrics.throughput) * 0.1
        
        return score
```

## 使用示例：LLM模型优化

### 1. 使用改造后的Mirage优化Llama模型

```python
class YICALlamaOptimizer:
    """
    使用改造后的Mirage优化Llama模型
    """
    
    def __init__(self, yica_config: YICAConfig):
        self.yica_backend = YICAMirageBackend(yica_config)
        self.search_engine = YICAMirageSearchEngine(self.yica_backend)
        
    def optimize_llama_attention(self, 
                               hidden_size: int = 4096,
                               num_heads: int = 32,
                               seq_len: int = 2048) -> YICAOptimizedKernel:
        """
        优化Llama的Attention层
        """
        # 1. 创建Attention的Mirage图
        import mirage as mi
        graph = mi.new_kernel_graph()
        
        head_dim = hidden_size // num_heads
        Q = graph.new_input(dims=(1, num_heads, seq_len, head_dim), dtype=mi.float16)
        K = graph.new_input(dims=(1, num_heads, seq_len, head_dim), dtype=mi.float16)
        V = graph.new_input(dims=(1, num_heads, seq_len, head_dim), dtype=mi.float16)
        
        # Flash Attention计算图
        scores = graph.matmul(Q, K, trans_b=True)
        scores = graph.mul_scalar(scores, 1.0 / math.sqrt(head_dim))
        attn_weights = graph.softmax(scores, dim=-1)
        output = graph.matmul(attn_weights, V)
        
        graph.mark_output(output)
        
        # 2. 使用YICA搜索引擎超优化
        optimized_kernel = self.search_engine.superoptimize_for_yica(
            mirage_graph=graph,
            objectives=["latency", "memory_efficiency", "energy_efficiency"],
            search_budget=1000
        )
        
        return optimized_kernel
    
    def optimize_llama_mlp(self, 
                         hidden_size: int = 4096,
                         intermediate_size: int = 11008) -> YICAOptimizedKernel:
        """
        优化Llama的MLP层（SwiGLU）
        """
        import mirage as mi
        graph = mi.new_kernel_graph()
        
        X = graph.new_input(dims=(1, 2048, hidden_size), dtype=mi.float16)
        W_gate = graph.new_input(dims=(hidden_size, intermediate_size), dtype=mi.float16)
        W_up = graph.new_input(dims=(hidden_size, intermediate_size), dtype=mi.float16)
        W_down = graph.new_input(dims=(intermediate_size, hidden_size), dtype=mi.float16)
        
        # SwiGLU: silu(X @ W_gate) * (X @ W_up) @ W_down
        gate_proj = graph.matmul(X, W_gate)
        up_proj = graph.matmul(X, W_up)
        gate_activated = graph.silu(gate_proj)
        fused = graph.mul(gate_activated, up_proj)
        output = graph.matmul(fused, W_down)
        
        graph.mark_output(output)
        
        # 使用YICA超优化
        optimized_kernel = self.search_engine.superoptimize_for_yica(
            mirage_graph=graph,
            objectives=["throughput", "energy_efficiency"],
            search_budget=800
        )
        
        return optimized_kernel
    
    def optimize_llama_rms_norm(self, hidden_size: int = 4096) -> YICAOptimizedKernel:
        """
        优化Llama的RMSNorm层
        """
        import mirage as mi
        graph = mi.new_kernel_graph()
        
        X = graph.new_input(dims=(1, 2048, hidden_size), dtype=mi.float16)
        weight = graph.new_input(dims=(hidden_size,), dtype=mi.float16)
        
        # RMSNorm计算
        variance = graph.reduction(graph.square(X), dim=-1, op="mean")
        normalized = graph.div(X, graph.sqrt(graph.add(variance, 1e-6)))
        output = graph.mul(normalized, weight)
        
        graph.mark_output(output)
        
        # 使用YICA超优化
        optimized_kernel = self.search_engine.superoptimize_for_yica(
            mirage_graph=graph,
            objectives=["latency", "memory_efficiency"],
            search_budget=500
        )
        
        return optimized_kernel
```

## 技术实现路线图

### Phase 1: Mirage-YICA适配层 (4周)
1. **Mirage架构分析和改造**
   - 分析Mirage现有架构和代码生成流程
   - 设计YICA适配接口
   - 实现内存模式分析器

2. **YICA指令生成器**
   - CIM阵列指令生成
   - SPM操作指令生成
   - 跨CIM通信指令生成

### Phase 2: YICA后端编译器 (6周)
1. **编译器后端开发**
   - YICA汇编代码生成器
   - 性能模型集成
   - 优化pass开发

2. **搜索引擎改造**
   - YICA搜索空间定义
   - 多目标优化算法
   - 搜索策略优化

### Phase 3: 集成测试和优化 (4周)  
1. **LLM模型集成**
   - Llama模型各层优化
   - 端到端性能测试
   - 正确性验证

2. **性能调优和工具**
   - 性能基准测试框架
   - 调试和分析工具
   - 用户接口开发

## 预期效果

### 1. YICA架构专用优化
- **存算一体优势**: 充分利用CIM阵列的并行计算能力
- **内存层次优化**: 最大化SPM利用率，减少DRAM访问
- **通信优化**: 最小化CIM间数据传输开销

### 2. Mirage超优化能力
- **自动搜索**: 自动发现YICA架构的最优算子配置
- **多目标优化**: 同时优化延迟、吞吐量、能效和内存效率
- **算子融合**: 自动识别和优化算子融合机会

### 3. 性能提升目标
- **矩阵乘法**: 相比传统GPU实现3x加速
- **Attention**: 内存使用减少60%，计算效率提升2.5x
- **LayerNorm**: 能效提升4x，延迟减少70%
- **端到端LLM**: 推理速度提升2x，能耗降低50%

---

**设计方案总结**

本方案通过改造Mirage超优化器来适配YICA存算一体架构：

1. **Mirage-YICA适配** - 将Mirage的计算图转换为YICA指令
2. **YICA后端编译器** - 专门针对YICA架构的代码生成和优化
3. **超优化搜索引擎** - 自动发现YICA架构的最优配置
4. **LLM专用优化** - 针对大语言模型的关键算子深度优化

该方案将Mirage的超优化能力与YICA的存算一体优势相结合，为AI模型在YICA架构上的高效执行提供强有力的编译器支持。

# YICA 架构感知分析器设计规范

## 功能概述
YICA架构感知分析器是YICA优化器的第一个核心组件，负责分析计算图对YICA存算一体架构的适配性。

## 模块设计

### 1. YICAArchitectureAnalyzer 类设计

```cpp
class YICAArchitectureAnalyzer {
public:
    struct YICAConfig {
        size_t cim_array_rows = 256;
        size_t cim_array_cols = 256;
        size_t spm_size_per_die = 2 * 1024 * 1024;  // 2MB
        size_t dram_bandwidth = 1024;  // GB/s
        size_t num_cim_dies = 16;
        float cim_frequency = 1000.0f;  // MHz
    };
    
    struct AnalysisResult {
        float cim_friendliness_score;      // CIM操作友好度 [0-1]
        float memory_locality_score;      // 内存局部性评分 [0-1]
        float parallelization_potential;  // 并行化潜力 [0-1]
        std::vector<std::string> bottlenecks;  // 性能瓶颈列表
        std::map<std::string, float> optimization_suggestions;  // 优化建议
    };
    
    YICAArchitectureAnalyzer(const YICAConfig& config);
    
    // 分析kernel graph的YICA适配性
    AnalysisResult analyze_computation_pattern(const kernel::Graph& graph);
    
    // 识别CIM友好的操作
    std::vector<kernel::DTensor*> identify_cim_operations(const kernel::Graph& graph);
    
    // 分析内存访问模式
    float analyze_memory_access_pattern(const kernel::Graph& graph);
    
    // 发现并行化机会
    std::vector<ParallelizationOpportunity> find_parallel_patterns(const kernel::Graph& graph);
    
private:
    YICAConfig config_;
    
    // 计算操作的CIM友好度
    float calculate_cim_friendliness(const kernel::DTensor* op);
    
    // 估计内存访问成本
    float estimate_memory_cost(const kernel::DTensor* tensor);
};
```

### 2. 数据结构定义

```cpp
struct ParallelizationOpportunity {
    enum Type {
        DATA_PARALLEL,     // 数据并行
        MODEL_PARALLEL,    // 模型并行
        PIPELINE_PARALLEL  // 流水线并行
    };
    
    Type type;
    std::vector<kernel::DTensor*> involved_tensors;
    float efficiency_score;  // 并行效率评分
    size_t recommended_parallelism;  // 推荐并行度
};
```

### 3. 实现算法规范

#### 3.1 CIM友好度计算算法
```
算法：calculate_cim_friendliness
输入：kernel::DTensor* op
输出：float (0-1之间的分数)

步骤：
1. 判断操作类型：
   - MATMUL -> 基础分数0.9 (CIM阵列优化)
   - ELEMENTWISE -> 基础分数0.7 (SPM向量单元优化)
   - REDUCTION -> 基础分数0.6 (部分CIM支持)
   - 其他 -> 基础分数0.3

2. 考虑数据规模：
   - 如果输入张量大小 <= SPM容量 -> 分数 * 1.2
   - 如果输入张量大小 > DRAM容量 -> 分数 * 0.8

3. 考虑数据类型：
   - FP16/BF16 -> 分数 * 1.1 (CIM阵列优化)
   - INT8/INT4 -> 分数 * 1.2 (量化计算优化)
   - FP32 -> 分数 * 0.9

返回：min(1.0, 调整后分数)
```

#### 3.2 内存访问模式分析算法
```
算法：analyze_memory_access_pattern
输入：kernel::Graph& graph
输出：float (内存局部性评分 0-1)

步骤：
1. 遍历所有tensor操作
2. 对每个tensor计算：
   - 重用距离 (reuse_distance)
   - 访问顺序 (access_pattern)
   - 数据生命周期 (lifetime)

3. 计算SPM适配性：
   spm_score = (适合SPM的数据量) / (总数据量)

4. 计算访问局部性：
   locality_score = (连续访问的数据量) / (总访问量)

5. 综合评分：
   return 0.6 * spm_score + 0.4 * locality_score
```

## 测试计划

### 单元测试用例
1. **test_cim_friendliness_calculation**
   - 测试不同操作类型的CIM友好度计算
   - 验证数据规模和类型对评分的影响

2. **test_memory_pattern_analysis**
   - 测试内存访问模式的正确识别
   - 验证SPM适配性评估的准确性

3. **test_parallelization_opportunities**
   - 测试并行化机会的发现算法
   - 验证并行效率评分的合理性

### 集成测试用例
1. **test_typical_transformer_block**
   - 使用典型的Transformer块测试完整分析流程
   - 验证分析结果的合理性和一致性

2. **test_conv_network_analysis**
   - 使用卷积网络测试YICA适配性分析
   - 验证CIM友好操作的正确识别

## 性能指标
- 分析延迟：< 100ms (对于包含50个操作的图)
- 内存占用：< 10MB
- 分析准确性：与手工分析的一致性 > 90%

## 接口设计

### C++接口
```cpp
namespace mirage {
namespace search {
namespace yica {

class YICAArchitectureAnalyzer; // 主分析器类

} // namespace yica
} // namespace search  
} // namespace mirage
```

### Python接口
```python
class YICAAnalyzer:
    def __init__(self, arch_config: dict)
    def analyze_graph(self, graph) -> dict
    def get_optimization_suggestions(self, analysis_result: dict) -> list
```

## 文件组织
```
mirage/include/mirage/search/yica/
├── yica_analyzer.h          # 主分析器头文件
├── yica_config.h           # YICA配置定义
└── yica_types.h            # YICA相关数据类型

mirage/src/search/yica/
├── yica_analyzer.cc        # 主分析器实现
├── cim_analysis.cc         # CIM友好度分析
└── memory_analysis.cc      # 内存模式分析
```

这是第一个功能的完整设计规范，接下来将进入开发阶段。

# YICA 优化策略库设计规范

## 功能概述
YICA优化策略库是Yirage优化器的第二个核心组件，提供针对YICA存算一体架构的专门优化策略集合。

## 核心设计理念
- **架构特化**：每种策略都深度适配YICA架构特性
- **可组合性**：策略可以组合使用，产生更好的效果
- **自适应性**：根据工作负载特征自动选择最佳策略
- **可扩展性**：支持新策略的动态添加

## 模块设计

### 1. OptimizationStrategy 基类设计

```cpp
class OptimizationStrategy {
public:
    enum StrategyType {
        CIM_DATA_REUSE,           // CIM数据重用优化
        SPM_ALLOCATION,           // SPM分配优化
        CROSS_CIM_COMMUNICATION,  // 跨CIM通信优化
        MEMORY_ACCESS_PATTERN,    // 内存访问模式优化
        OPERATOR_FUSION,          // 算子融合优化
        PARALLELIZATION          // 并行化优化
    };
    
    struct OptimizationResult {
        bool success;
        float improvement_score;      // 改进评分 [0-1]
        std::string description;      // 优化描述
        std::map<std::string, float> metrics;  // 具体指标改进
    };
    
    virtual ~OptimizationStrategy() = default;
    virtual bool is_applicable(const AnalysisResult& analysis) const = 0;
    virtual OptimizationResult apply(kernel::Graph& graph, const YICAConfig& config) = 0;
    virtual StrategyType get_type() const = 0;
    virtual std::string get_name() const = 0;
};
```

### 2. CIMDataReuseStrategy 类设计

```cpp
class CIMDataReuseStrategy : public OptimizationStrategy {
public:
    struct ReusePattern {
        std::vector<kernel::DTensor*> reusable_tensors;
        float reuse_factor;           // 重用因子
        size_t memory_saving;         // 内存节省量（字节）
    };
    
    CIMDataReuseStrategy();
    bool is_applicable(const AnalysisResult& analysis) const override;
    OptimizationResult apply(kernel::Graph& graph, const YICAConfig& config) override;
    StrategyType get_type() const override { return CIM_DATA_REUSE; }
    std::string get_name() const override { return "CIM Data Reuse Optimization"; }

private:
    std::vector<ReusePattern> identify_reuse_opportunities(const kernel::Graph& graph);
    void implement_data_reuse(kernel::Graph& graph, const ReusePattern& pattern);
};
```

### 3. SPMAllocationStrategy 类设计

```cpp
class SPMAllocationStrategy : public OptimizationStrategy {
public:
    struct AllocationPlan {
        std::map<kernel::DTensor*, size_t> tensor_allocation;  // 张量->SPM位置映射
        float spm_utilization;        // SPM利用率
        float access_efficiency;      // 访问效率提升
    };
    
    SPMAllocationStrategy();
    bool is_applicable(const AnalysisResult& analysis) const override;
    OptimizationResult apply(kernel::Graph& graph, const YICAConfig& config) override;
    StrategyType get_type() const override { return SPM_ALLOCATION; }
    std::string get_name() const override { return "SPM Allocation Optimization"; }

private:
    AllocationPlan generate_allocation_plan(const kernel::Graph& graph, const YICAConfig& config);
    void implement_spm_allocation(kernel::Graph& graph, const AllocationPlan& plan);
};
```

### 4. OperatorFusionStrategy 类设计

```cpp
class OperatorFusionStrategy : public OptimizationStrategy {
public:
    struct FusionGroup {
        std::vector<kernel::KNOperator*> operators;
        float fusion_benefit;         // 融合收益评分
        std::string fusion_type;      // 融合类型描述
    };
    
    OperatorFusionStrategy();
    bool is_applicable(const AnalysisResult& analysis) const override;
    OptimizationResult apply(kernel::Graph& graph, const YICAConfig& config) override;
    StrategyType get_type() const override { return OPERATOR_FUSION; }
    std::string get_name() const override { return "Operator Fusion Optimization"; }

private:
    std::vector<FusionGroup> identify_fusion_opportunities(const kernel::Graph& graph);
    void implement_operator_fusion(kernel::Graph& graph, const FusionGroup& group);
};
```

### 5. YICAOptimizationStrategyLibrary 主类设计

```cpp
class YICAOptimizationStrategyLibrary {
public:
    using StrategyPtr = std::unique_ptr<OptimizationStrategy>;
    
    struct StrategySelection {
        std::vector<StrategyPtr> selected_strategies;
        float expected_improvement;   // 预期改进程度
        std::string selection_rationale;  // 选择理由
    };
    
    YICAOptimizationStrategyLibrary();
    ~YICAOptimizationStrategyLibrary();
    
    // 策略管理
    void register_strategy(StrategyPtr strategy);
    void unregister_strategy(OptimizationStrategy::StrategyType type);
    std::vector<OptimizationStrategy*> get_all_strategies() const;
    
    // 策略选择
    StrategySelection select_strategies(const AnalysisResult& analysis) const;
    std::vector<OptimizationStrategy*> get_applicable_strategies(const AnalysisResult& analysis) const;
    
    // 策略应用
    std::vector<OptimizationStrategy::OptimizationResult> 
        apply_strategies(kernel::Graph& graph, 
                        const YICAConfig& config,
                        const std::vector<OptimizationStrategy*>& strategies) const;
    
    // 策略组合优化
    StrategySelection optimize_strategy_combination(const AnalysisResult& analysis) const;

private:
    std::map<OptimizationStrategy::StrategyType, StrategyPtr> strategies_;
    
    // 策略评估和选择算法
    float evaluate_strategy_combination(const std::vector<OptimizationStrategy*>& strategies,
                                       const AnalysisResult& analysis) const;
    bool are_strategies_compatible(OptimizationStrategy* s1, OptimizationStrategy* s2) const;
};
```

## 算法设计

### 1. 策略选择算法

```cpp
// 基于贪心算法的策略选择
std::vector<OptimizationStrategy*> select_greedy_strategies(const AnalysisResult& analysis) {
    std::vector<OptimizationStrategy*> selected;
    std::vector<OptimizationStrategy*> candidates = get_applicable_strategies(analysis);
    
    // 按预期收益排序
    std::sort(candidates.begin(), candidates.end(), 
              [&](auto* a, auto* b) {
                  return estimate_benefit(a, analysis) > estimate_benefit(b, analysis);
              });
    
    // 贪心选择兼容的策略
    for (auto* strategy : candidates) {
        if (is_compatible_with_selected(strategy, selected)) {
            selected.push_back(strategy);
        }
    }
    
    return selected;
}
```

### 2. 数据重用优化算法

```cpp
// 基于生命周期分析的数据重用优化
std::vector<ReusePattern> analyze_data_reuse(const kernel::Graph& graph) {
    std::vector<ReusePattern> patterns;
    
    // 构建张量生命周期图
    auto lifetime_graph = build_tensor_lifetime_graph(graph);
    
    // 识别重用机会
    for (auto& tensor : graph.get_all_tensors()) {
        auto reuse_ops = find_reuse_operations(tensor, lifetime_graph);
        if (reuse_ops.size() > 1) {
            ReusePattern pattern;
            pattern.reusable_tensors = {tensor};
            pattern.reuse_factor = static_cast<float>(reuse_ops.size());
            patterns.push_back(pattern);
        }
    }
    
    return patterns;
}
```

### 3. SPM分配优化算法

```cpp
// 基于最优装箱问题的SPM分配算法
AllocationPlan optimize_spm_allocation(const kernel::Graph& graph, const YICAConfig& config) {
    AllocationPlan plan;
    
    // 获取所有需要分配的张量
    auto tensors = get_allocatable_tensors(graph);
    
    // 按访问频率和大小排序
    std::sort(tensors.begin(), tensors.end(), 
              [](const auto& a, const auto& b) {
                  float score_a = a.access_frequency / a.size;
                  float score_b = b.access_frequency / b.size;
                  return score_a > score_b;
              });
    
    // 贪心分配到SPM
    size_t spm_used = 0;
    for (auto& tensor : tensors) {
        if (smp_used + tensor.size <= config.spm_size_per_die) {
            plan.tensor_allocation[tensor.ptr] = spm_used;
            spm_used += tensor.size;
        }
    }
    
    plan.smp_utilization = static_cast<float>(smp_used) / config.spm_size_per_die;
    return plan;
}
```

## 性能目标

- **策略选择延迟**: < 50ms
- **策略应用延迟**: < 200ms  
- **内存占用**: < 50MB
- **策略有效性**: > 80%的情况下产生性能改进
- **策略组合效果**: 多策略组合比单策略效果提升 > 20%

## 扩展性设计

### 1. 插件机制
```cpp
// 支持动态加载新的优化策略
class StrategyPlugin {
public:
    virtual StrategyPtr create_strategy() = 0;
    virtual std::string get_strategy_name() const = 0;
};

void YICAOptimizationStrategyLibrary::load_plugin(const std::string& plugin_path) {
    // 动态加载策略插件
}
```

### 2. 配置驱动
```cpp
// 支持通过配置文件定制策略行为
struct StrategyConfig {
    std::map<std::string, float> parameters;
    std::vector<std::string> enabled_strategies;
    std::map<std::string, std::string> strategy_options;
};
```

**下一步**: 等待设计确认，然后开始实现 YICAOptimizationStrategyLibrary 的核心功能。

# Yirage AI内核优化器 - 详细设计文档

## 项目概述
Yirage是一个结合YICA存算一体架构和Mirage优化框架的AI内核优化器，通过深度架构感知和智能代码生成技术，实现对深度学习工作负载的极致优化。

## 核心功能模块

### 功能1: YICA架构感知分析器 ✅
**目标**: 分析计算图对YICA架构的适配性和优化潜力

#### 核心组件
- **YICAArchitectureAnalyzer类**: 主要分析引擎
- **AnalysisResult结构**: 存储分析结果  
- **YICAConfig结构**: YICA架构配置

#### 核心算法
- **CIM友好度评估**: 基于操作类型、数据局部性和并行度的综合评分算法
- **内存访问模式分析**: 识别SPM利用机会和内存瓶颈
- **并行化机会发现**: 分析CIM阵列级并行化潜力
- **性能瓶颈识别**: 定位计算和内存访问瓶颈

#### 性能目标
- 分析延迟: < 100ms (1000个操作的图)
- 准确度: CIM友好度预测误差 < 10%
- 内存占用: < 50MB
- 并发支持: 支持多线程并行分析

### 功能2: YICA优化策略库 ✅
**目标**: 提供针对YICA架构的优化策略集合

#### 核心组件
- **OptimizationStrategy基类**: 策略接口定义
- **CIMDataReuseStrategy**: CIM数据重用优化
- **SPMAllocationStrategy**: SPM分配优化  
- **OperatorFusionStrategy**: 算子融合优化
- **YICAOptimizationStrategyLibrary**: 策略库主类

#### 核心算法  
- **贪心策略选择**: 基于收益预估的策略选择算法
- **基于生命周期分析的数据重用**: 通过数据依赖图分析优化数据重用
- **基于最优装箱的SPM分配**: 使用动态规划优化SPM空间分配

#### 性能目标
- 策略选择: < 50ms
- 策略应用: < 200ms  
- 内存开销: < 50MB
- 策略有效性: > 80%的情况下产生性能提升

### 功能3: YICA代码生成器 ✅
**目标**: 生成针对YICA架构优化的高性能代码

#### 核心组件
- **YICACodeGenerator主类**: 代码生成器核心
- **CodeTemplateManager**: 模板管理系统
- **CIMCodeGenAlgorithm**: CIM指令生成算法
- **OperatorGenerator**: 操作生成器基类及具体实现

#### 核心算法
- **计算图到代码映射**: 将优化后的计算图转换为YICA原生代码
- **CIM阵列代码生成**: 生成CIM阵列的并行计算代码
- **SPM分配代码生成**: 生成SPM内存管理代码
- **并行化代码生成**: 生成多CIM阵列协同工作代码

#### 性能目标
- 代码生成延迟: < 500ms
- 生成代码性能提升: 15-30% vs 基准实现
- SPM利用率: > 85%
- 编译成功率: > 95%

### 功能4: YICA运行时优化器 (新增设计)
**目标**: 在运行时动态调整和优化YICA架构的执行策略

#### 核心组件

**4.1 运行时监控系统**
- **PerformanceMonitor类**: 性能指标收集器
  - CIM阵列利用率监控
  - SPM命中率统计
  - 内存带宽使用监控
  - 计算吞吐量测量
  - 功耗监控

- **ProfilingManager类**: 性能剖析管理器
  - 热点操作识别
  - 数据流分析
  - 瓶颈定位
  - 性能基线建立

**4.2 动态优化引擎**
- **RuntimeOptimizer类**: 运行时优化核心
  - 实时策略调整
  - 负载均衡优化
  - 资源重分配
  - 执行路径优化

- **AdaptiveScheduler类**: 自适应调度器
  - 基于实时负载的CIM阵列调度
  - SPM分配动态调整
  - 操作融合实时决策
  - 数据预取优化

**4.3 机器学习辅助优化**
- **MLOptimizer类**: 机器学习优化器
  - 性能预测模型
  - 策略推荐系统
  - 参数自动调优
  - 异常检测和处理

- **OnlineLearning类**: 在线学习系统
  - 实时模型更新
  - 经验回放机制
  - 多任务适应
  - 个性化优化

#### 核心算法

**4.1 实时性能监控算法**
```cpp
class PerformanceMetrics {
    float cim_utilization;           // CIM阵列利用率 [0,1]
    float spm_hit_rate;             // SPM命中率 [0,1]
    float memory_bandwidth_usage;    // 内存带宽使用率 [0,1]
    float compute_throughput;        // 计算吞吐量 TOPS
    float power_consumption;         // 功耗 W
    float latency_ms;               // 延迟 ms
    std::vector<float> per_array_util; // 每个CIM阵列利用率
};

// 滑动窗口性能监控
class SlidingWindowMonitor {
    void update_metrics(const PerformanceMetrics& metrics);
    PerformanceMetrics get_average_metrics(int window_size_ms);
    bool detect_performance_anomaly();
    std::vector<std::string> identify_bottlenecks();
};
```

**4.2 自适应优化决策算法**
```cpp
// 基于强化学习的优化决策
class ReinforcementOptimizer {
    // 状态空间: 当前性能指标 + 系统配置
    struct State {
        PerformanceMetrics current_metrics;
        YICAConfig current_config;
        WorkloadCharacteristics workload;
    };
    
    // 动作空间: 可执行的优化动作
    enum class OptimizationAction {
        INCREASE_CIM_FREQUENCY,    // 提高CIM频率
        DECREASE_CIM_FREQUENCY,    // 降低CIM频率
        REDISTRIBUTE_SPM,          // 重新分配SPM
        CHANGE_FUSION_STRATEGY,    // 改变融合策略
        ADJUST_PARALLELISM,        // 调整并行度
        ENABLE_PREFETCH,           // 启用预取
        DISABLE_PREFETCH           // 禁用预取
    };
    
    // Q-learning决策
    OptimizationAction select_action(const State& state);
    void update_q_value(const State& state, OptimizationAction action, float reward);
};
```

**4.3 多目标优化算法**
```cpp
// 基于帕累托前沿的多目标优化
class MultiObjectiveOptimizer {
    struct OptimizationObjective {
        float performance_weight = 0.6f;  // 性能权重
        float energy_weight = 0.3f;      // 能效权重
        float latency_weight = 0.1f;     // 延迟权重
    };
    
    // 帕累托最优解集
    std::vector<YICAConfig> pareto_front;
    
    // NSGA-II算法实现
    std::vector<YICAConfig> evolve_solutions(
        const std::vector<YICAConfig>& population,
        const PerformanceMetrics& current_metrics);
    
    YICAConfig select_best_config(
        const OptimizationObjective& objective,
        const std::vector<YICAConfig>& candidates);
};
```

**4.4 预测模型算法**
```cpp
// 基于LSTM的性能预测模型
class PerformancePredictionModel {
    // 时序特征提取
    struct TimeSeriesFeatures {
        std::vector<float> cim_utilization_history;
        std::vector<float> memory_access_pattern;
        std::vector<float> workload_characteristics;
        int sequence_length = 50;  // 历史窗口长度
    };
    
    // 预测未来性能
    PerformanceMetrics predict_performance(
        const TimeSeriesFeatures& features,
        const YICAConfig& proposed_config);
    
    // 在线模型更新
    void update_model(
        const TimeSeriesFeatures& features,
        const PerformanceMetrics& actual_performance);
};
```

#### 数据结构设计

**4.1 运行时状态管理**
```cpp
struct RuntimeState {
    PerformanceMetrics current_metrics;     // 当前性能指标
    YICAConfig active_config;              // 当前配置
    std::vector<OptimizationAction> action_history;  // 动作历史
    std::chrono::time_point<std::chrono::steady_clock> last_update; // 最后更新时间
    
    // 性能历史记录
    std::deque<PerformanceMetrics> metrics_history;
    static constexpr size_t MAX_HISTORY_SIZE = 1000;
};

struct WorkloadCharacteristics {
    float compute_intensity;         // 计算密集度
    float memory_intensity;          // 内存密集度
    float data_reuse_factor;        // 数据重用因子
    std::vector<float> op_distribution; // 操作分布
    float batch_size_variability;   // 批大小变化
};
```

**4.2 优化上下文**
```cpp
struct OptimizationContext {
    RuntimeState runtime_state;             // 运行时状态
    WorkloadCharacteristics workload;       // 工作负载特征
    OptimizationObjective objective;        // 优化目标
    std::vector<YICAConfig> candidate_configs; // 候选配置
    float optimization_budget_ms;           // 优化时间预算
    bool enable_ml_optimization;            // 是否启用ML优化
};
```

#### 接口设计

**4.1 主要接口**
```cpp
class YICARuntime {
public:
    // 初始化运行时系统
    bool initialize(const YICAConfig& initial_config);
    
    // 启动性能监控
    void start_monitoring();
    void stop_monitoring();
    
    // 执行优化
    OptimizationResult optimize(const OptimizationContext& context);
    
    // 应用配置更改
    bool apply_configuration(const YICAConfig& new_config);
    
    // 获取当前状态
    RuntimeState get_current_state() const;
    PerformanceMetrics get_current_metrics() const;
    
    // 事件回调
    void set_performance_callback(std::function<void(const PerformanceMetrics&)> callback);
    void set_anomaly_callback(std::function<void(const std::string&)> callback);
};
```

**4.2 配置管理接口**
```cpp
class RuntimeConfigManager {
public:
    // 配置验证
    bool validate_config(const YICAConfig& config);
    
    // 配置转换
    YICAConfig merge_configs(const YICAConfig& base, const YICAConfig& update);
    
    // 配置回滚
    bool rollback_to_previous_config();
    
    // 配置历史管理
    std::vector<YICAConfig> get_config_history(int max_count = 10);
    void save_config_checkpoint(const std::string& name);
    bool restore_config_checkpoint(const std::string& name);
};
```

#### 性能目标

**4.1 实时性要求**
- 性能监控开销: < 2% 系统开销
- 优化决策延迟: < 10ms (简单调整), < 100ms (复杂优化)
- 配置切换延迟: < 5ms
- 监控采样频率: 1000Hz

**4.2 优化效果**
- 性能提升: 5-15% vs 静态优化
- 能效提升: 10-20% 在相同性能下
- 适应时间: < 1000ms 适应新工作负载
- 稳定性: > 99.9% 运行时间无异常

**4.3 资源消耗**
- 内存开销: < 100MB
- CPU开销: < 5% 单核使用率
- 存储开销: < 500MB (模型和历史数据)

#### 实现架构

**4.1 模块层次结构**
```
YICARuntime (顶层接口)
├── PerformanceMonitor (性能监控)
│   ├── MetricsCollector
│   ├── SlidingWindowMonitor  
│   └── AnomalyDetector
├── RuntimeOptimizer (运行时优化)
│   ├── ReinforcementOptimizer
│   ├── MultiObjectiveOptimizer
│   └── AdaptiveScheduler
├── MLOptimizer (机器学习优化)
│   ├── PerformancePredictionModel
│   ├── OnlineLearning
│   └── ParameterTuner
└── RuntimeConfigManager (配置管理)
    ├── ConfigValidator
    ├── ConfigHistory
    └── CheckpointManager
```

**4.2 数据流设计**
```
硬件性能计数器 → MetricsCollector → SlidingWindowMonitor 
                                        ↓
WorkloadProfiler → RuntimeOptimizer ← PerformanceMetrics
       ↓                ↓
MLOptimizer ← OptimizationContext → AdaptiveScheduler
       ↓                              ↓
ParameterTuner                   ConfigValidator
       ↓                              ↓
YICAConfig ← RuntimeConfigManager → HardwareController
```

#### 集成方案

**4.1 与现有功能的集成**
- **架构感知分析器**: 提供工作负载特征分析
- **优化策略库**: 提供备选优化策略
- **代码生成器**: 生成运行时可切换的代码变体

**4.2 扩展接口**
- 支持自定义性能指标
- 支持用户定义的优化目标
- 支持第三方优化插件
- 支持分布式优化协调

#### 错误处理和恢复

**4.1 异常情况处理**
- 性能监控失败: 使用备用监控机制
- 优化失败: 回滚到已知良好配置
- 硬件故障: 自动降级和资源重分配
- 模型预测错误: 切换到启发式方法

**4.2 容错机制**
- 配置验证和安全检查
- 渐进式配置更改
- 性能回归检测和自动回滚
- 关键路径保护

#### 测试和验证

**4.1 功能测试**
- 单元测试: 每个组件的独立测试
- 集成测试: 端到端优化流程测试
- 性能测试: 优化效果验证
- 压力测试: 高负载下的稳定性测试

**4.2 验证基准**
- 与静态优化的性能对比
- 多种工作负载的适应性测试
- 长时间运行的稳定性验证
- 不同硬件配置的兼容性测试

### 总体架构集成

四个功能模块形成完整的优化管道:

```
输入计算图 → YICA架构感知分析器 → 优化策略库 → YICA代码生成器 
                     ↓                ↓              ↓
              分析结果缓存      策略选择历史    生成代码库
                     ↓                ↓              ↓
                 YICA运行时优化器 ← 性能反馈 ← 执行引擎
                     ↓
               动态配置调整 → 持续性能优化
```

这个设计确保了从静态分析到动态运行时优化的完整闭环，实现了真正的自适应AI内核优化系统。

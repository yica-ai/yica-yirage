# yirage-YICA扩展：目标函数支持的搜索优化方案

## 概述

在现有yirage超优化器基础上，增加YICA架构特定的目标函数支持，通过扩展yirage的搜索配置和评估机制来实现YICA优化。

## 核心设计

### 1. YICA目标函数框架

```cpp
// yirage/include/yirage/search/yica_objective.h
#pragma once

#include "yirage/search/config.h"
#include "yirage/kernel/graph.h"
#include "yirage/threadblock/graph.h"

namespace yirage {
namespace search {
namespace yica {

enum class YICAObjectiveType {
  LATENCY,
  MEMORY_EFFICIENCY,
  ENERGY_EFFICIENCY,
  THROUGHPUT,
  COMBINED
};

struct YICAArchConfig {
  size_t cim_array_rows = 256;
  size_t cim_array_cols = 256;
  size_t spm_size_per_die = 2 * 1024 * 1024;  // 2MB
  size_t dram_bandwidth = 1024;  // GB/s
  size_t cim_frequency = 1000;   // MHz
  size_t num_cim_dies = 16;
  
  // 存算一体特定参数
  float cim_energy_per_op = 0.1;    // pJ per operation
  float spm_energy_per_access = 1.0; // pJ per access
  float dram_energy_per_access = 100.0; // pJ per access
  float communication_latency = 10.0; // ns
};

struct YICAPerformanceMetrics {
  float latency_ms = 0.0;
  float memory_utilization = 0.0;
  float energy_consumption_mj = 0.0;
  float throughput_ops = 0.0;
  float cim_utilization = 0.0;
  float spm_hit_rate = 0.0;
  float communication_overhead = 0.0;
  
  float combined_score(const std::vector<float>& weights) const;
};

class YICAObjectiveFunction {
public:
  YICAObjectiveFunction(YICAArchConfig config, YICAObjectiveType type);
  
  // 评估内核图的性能
  YICAPerformanceMetrics evaluate(const kernel::Graph& graph) const;
  
  // 评估线程块图的性能
  YICAPerformanceMetrics evaluate(const threadblock::Graph& graph) const;
  
  // 获取目标函数值（越小越好）
  float getObjectiveValue(const YICAPerformanceMetrics& metrics) const;
  
private:
  YICAArchConfig arch_config_;
  YICAObjectiveType objective_type_;
  
  // 分析内存访问模式
  float analyzeMemoryPattern(const kernel::Graph& graph) const;
  
  // 估计CIM阵列利用率
  float estimateCIMUtilization(const kernel::Graph& graph) const;
  
  // 估计通信开销
  float estimateCommunicationOverhead(const threadblock::Graph& graph) const;
  
  // 估计能耗
  float estimateEnergyConsumption(const kernel::Graph& graph) const;
};

}}} // namespace yirage::search::yica
```

### 2. 扩展yirage搜索配置

```cpp
// 在 yirage/include/yirage/search/config.h 中添加
struct YICAGeneratorConfig : public GeneratorConfig {
  // YICA特定配置
  yica::YICAArchConfig yica_arch_config;
  yica::YICAObjectiveType objective_type = yica::YICAObjectiveType::COMBINED;
  std::vector<float> objective_weights = {0.4, 0.3, 0.2, 0.1}; // latency, memory, energy, throughput
  
  // YICA搜索空间
  std::vector<size_t> cim_tile_sizes = {64, 128, 256};
  std::vector<size_t> spm_allocation_strategies = {0, 1, 2}; // temporal_reuse, spatial_locality, balanced
  std::vector<size_t> data_layout_patterns = {0, 1, 2}; // row_major, col_major, tiled
  
  // 搜索预算分配
  float yica_search_weight = 0.3; // 30%搜索预算用于YICA优化
  size_t max_yica_configurations = 100;
  
  static YICAGeneratorConfig get_yica_default_config();
  void enable_yica_optimization();
  void set_yica_objectives(const std::vector<yica::YICAObjectiveType>& objectives);
};
```

### 3. YICA性能评估器

```cpp
// yirage/src/search/yica_evaluator.cc
#include "yirage/search/yica_objective.h"
#include <cmath>

namespace yirage {
namespace search {
namespace yica {

YICAPerformanceMetrics YICAObjectiveFunction::evaluate(const kernel::Graph& graph) const {
  YICAPerformanceMetrics metrics;
  
  // 1. 分析算子类型和计算复杂度
  size_t total_ops = 0;
  size_t matmul_ops = 0;
  size_t elementwise_ops = 0;
  size_t reduction_ops = 0;
  
  for (auto* op : graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_MATMUL_OP) {
      matmul_ops++;
      // 估计矩阵乘法的计算量
      auto input_shape = op->input_tensors[0].dim;
      total_ops += input_shape[0] * input_shape[1] * op->input_tensors[1].dim[1];
    } else if (op->op_type == type::KNOperatorType::KN_ADD_OP ||
               op->op_type == type::KNOperatorType::KN_MUL_OP) {
      elementwise_ops++;
      auto input_shape = op->input_tensors[0].dim;
      size_t elements = 1;
      for (int i = 0; i < op->input_tensors[0].num_dims; i++) {
        elements *= input_shape[i];
      }
      total_ops += elements;
    }
  }
  
  // 2. 估计CIM阵列利用率
  float cim_peak_ops = arch_config_.cim_array_rows * arch_config_.cim_array_cols * arch_config_.cim_frequency;
  metrics.cim_utilization = std::min(1.0f, static_cast<float>(matmul_ops * cim_peak_ops) / total_ops);
  
  // 3. 估计内存利用率
  metrics.memory_utilization = analyzeMemoryPattern(graph);
  
  // 4. 估计延迟
  float compute_latency = total_ops / (cim_peak_ops * metrics.cim_utilization);
  float memory_latency = estimateMemoryLatency(graph);
  metrics.latency_ms = std::max(compute_latency, memory_latency) * 1000.0f;
  
  // 5. 估计能耗
  metrics.energy_consumption_mj = estimateEnergyConsumption(graph);
  
  // 6. 估计吞吐量
  metrics.throughput_ops = total_ops / (metrics.latency_ms / 1000.0f);
  
  return metrics;
}

float YICAObjectiveFunction::analyzeMemoryPattern(const kernel::Graph& graph) const {
  size_t total_memory_accesses = 0;
  size_t spm_friendly_accesses = 0;
  
  for (auto* op : graph.operators) {
    for (auto& tensor : op->input_tensors) {
      size_t tensor_size = 1;
      for (int i = 0; i < tensor.num_dims; i++) {
        tensor_size *= tensor.dim[i];
      }
      tensor_size *= sizeof(float); // 假设float16
      
      total_memory_accesses += tensor_size;
      
      // 判断是否适合SPM
      if (tensor_size <= arch_config_.spm_size_per_die) {
        spm_friendly_accesses += tensor_size;
      }
    }
  }
  
  return static_cast<float>(spm_friendly_accesses) / total_memory_accesses;
}

float YICAObjectiveFunction::estimateEnergyConsumption(const kernel::Graph& graph) const {
  float total_energy = 0.0f;
  
  for (auto* op : graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_MATMUL_OP) {
      // CIM阵列矩阵乘法能耗
      auto input_shape = op->input_tensors[0].dim;
      size_t ops = input_shape[0] * input_shape[1] * op->input_tensors[1].dim[1];
      total_energy += ops * arch_config_.cim_energy_per_op;
    } else {
      // SPM或DRAM访问能耗
      for (auto& tensor : op->input_tensors) {
        size_t tensor_size = 1;
        for (int i = 0; i < tensor.num_dims; i++) {
          tensor_size *= tensor.dim[i];
        }
        
        if (tensor_size <= arch_config_.spm_size_per_die) {
          total_energy += tensor_size * arch_config_.spm_energy_per_access;
        } else {
          total_energy += tensor_size * arch_config_.dram_energy_per_access;
        }
      }
    }
  }
  
  return total_energy / 1e9; // 转换为mJ
}

}}} // namespace yirage::search::yica
```

### 4. 集成到yirage搜索引擎

```cpp
// 修改 yirage/src/search/search.cc
#include "yirage/search/yica_objective.h"

// 在 KernelGraphGenerator 类中添加
class KernelGraphGenerator {
private:
  std::unique_ptr<yica::YICAObjectiveFunction> yica_evaluator_;
  YICAGeneratorConfig yica_config_;
  
public:
  void setYICAConfig(const YICAGeneratorConfig& config) {
    yica_config_ = config;
    yica_evaluator_ = std::make_unique<yica::YICAObjectiveFunction>(
      config.yica_arch_config, config.objective_type);
  }
  
  // 增强的验证函数，考虑YICA目标函数
  bool verifyWithYICAObjective(const kernel::Graph& graph) {
    // 首先进行原有的正确性验证
    if (!verify(graph)) {
      return false;
    }
    
    // 然后进行YICA性能评估
    auto metrics = yica_evaluator_->evaluate(graph);
    float objective_value = yica_evaluator_->getObjectiveValue(metrics);
    
    // 设定阈值或者用于排序
    return objective_value < yica_config_.performance_threshold;
  }
};

// 修改搜索循环以支持YICA评估
void KernelGraphGenerator::generate_next_operator(/*参数*/) {
  // ... 原有逻辑 ...
  
  // 在验证阶段增加YICA评估
  if (yica_config_.enable_yica_optimization) {
    if (verifyWithYICAObjective(*c.kn_graph)) {
      verified.push_back(SerializedSearchContext(c));
      return;
    }
  } else {
    // 原有验证逻辑
    if (verify(c)) {
      verified.push_back(SerializedSearchContext(c));
      return;
    }
  }
  
  // ... 继续搜索逻辑 ...
}
```

### 5. Python接口扩展

```python
# yirage/python/yirage/yica_optimizer.py
class YICAOptimizer:
    """
    YICA架构专用的yirage优化器
    """
    
    def __init__(self, 
                 arch_config=None,
                 objectives=["latency", "memory_efficiency"],
                 objective_weights=None):
        self.arch_config = arch_config or self._get_default_arch_config()
        self.objectives = objectives
        self.objective_weights = objective_weights or [0.5, 0.3, 0.1, 0.1]
        
    def superoptimize_for_yica(self, 
                              graph,
                              search_budget=1000,
                              config="yica_default"):
        """
        针对YICA架构执行超优化
        """
        # 设置YICA搜索配置
        yica_config = self._create_yica_config(config)
        
        # 调用C++层的YICA搜索
        from . import _cython_yirage
        
        optimized_graphs = _cython_yirage.yica_search(
            graph.cygraph,
            yica_config.to_dict(),
            search_budget
        )
        
        # 使用YICA目标函数评估和选择最佳图
        best_graph = self._select_best_yica_graph(optimized_graphs)
        
        return best_graph
    
    def _create_yica_config(self, config_name):
        """创建YICA搜索配置"""
        from . import YICAGeneratorConfig
        
        config = YICAGeneratorConfig.get_yica_default_config()
        config.enable_yica_optimization()
        config.set_yica_objectives(self.objectives)
        config.yica_arch_config = self.arch_config
        config.objective_weights = self.objective_weights
        
        return config
    
    def _select_best_yica_graph(self, graphs):
        """基于YICA目标函数选择最佳图"""
        best_graph = None
        best_score = float('inf')
        
        for graph in graphs:
            metrics = self._evaluate_yica_performance(graph)
            score = self._calculate_combined_score(metrics)
            
            if score < best_score:
                best_score = score
                best_graph = graph
        
        return best_graph

# 扩展现有的Graph类
class Graph:
    def superoptimize_yica(self, 
                          arch_config=None,
                          objectives=["latency", "memory_efficiency"],
                          search_budget=1000):
        """
        使用YICA优化器进行超优化
        """
        optimizer = YICAOptimizer(arch_config, objectives)
        return optimizer.superoptimize_for_yica(self, search_budget)
```

### 6. 使用示例

```python
import yirage as mi

# 创建计算图
graph = mi.new_kernel_graph()
A = graph.new_input(dims=(1024, 1024), dtype=mi.float16)
B = graph.new_input(dims=(1024, 1024), dtype=mi.float16)
C = graph.matmul(A, B)
graph.mark_output(C)

# YICA架构配置
yica_config = mi.YICAArchConfig(
    cim_array_rows=512,
    cim_array_cols=512,
    spm_size_per_die=4*1024*1024,  # 4MB
    num_cim_dies=32
)

# 使用YICA优化器
optimized_graph = graph.superoptimize_yica(
    arch_config=yica_config,
    objectives=["latency", "energy_efficiency"],
    search_budget=2000
)

print(f"优化后的性能提升: {optimized_graph.performance_improvement}")
```

## 实施计划

### 阶段1: 核心框架 (2周)
1. 实现YICAObjectiveFunction类
2. 扩展GeneratorConfig支持YICA配置
3. 基础性能评估算法

### 阶段2: 搜索集成 (3周)
1. 修改搜索引擎支持YICA目标函数
2. 实现YICA特定的搜索策略
3. 性能评估器优化

### 阶段3: Python接口和测试 (2周)
1. Python接口实现
2. 单元测试和集成测试
3. 性能基准测试

## 预期效果

通过在yirage中增加YICA目标函数支持，预期实现：

1. **精确的YICA性能建模**: 考虑存算一体架构的特殊性
2. **多目标优化**: 同时优化延迟、内存效率、能耗和吞吐量
3. **无缝集成**: 在现有yirage框架基础上最小化修改
4. **显著性能提升**: 相比通用GPU优化，在YICA上实现2-3x性能提升

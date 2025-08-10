# API文档

本目录包含YICA/YiRage的API参考文档和使用示例。

## 📖 文档列表

### 当前文档
- **[分析器API](analyzer.md)** - YICA分析器的API参考和使用指南

### 计划添加的文档
- **Python API** - YiRage Python接口完整参考
- **C++ API** - YICA C++内核API文档  
- **REST API** - Web服务接口文档
- **使用示例** - 各种API的使用示例

## 🔌 API概览

### Python API
YiRage提供了丰富的Python接口，支持多种使用场景：

```python
import yirage

# 基本优化
optimizer = yirage.Optimizer()
result = optimizer.superoptimize(backend="yica")

# 高级配置
config = yirage.YICAConfig(
    optimization_strategy="throughput_optimal",
    enable_kernel_fusion=True
)
result = optimizer.superoptimize(backend="yica", yica_config=config)
```

### C++ API
底层C++接口提供最高性能和灵活性：

```cpp
#include "yirage/yica_optimizer.h"

// 创建优化器
auto optimizer = yirage::YICAOptimizer::create();

// 执行优化
auto result = optimizer->optimize(input_graph);
```

### REST API
Web服务接口支持远程调用：

```bash
# 提交优化任务
curl -X POST http://localhost:8080/api/optimize \
  -H "Content-Type: application/json" \
  -d '{"backend": "yica", "model": "..."}'
```

## 📚 API分类

### 1. 核心优化API
- **图优化**: 计算图的搜索和优化
- **算子优化**: 单个算子的优化
- **端到端优化**: 完整模型的优化

### 2. 后端管理API
- **后端选择**: CUDA、Triton、YICA后端切换
- **后端配置**: 后端特定的配置选项
- **后端信息**: 后端能力和状态查询

### 3. 性能分析API
- **性能测试**: Profiling和基准测试
- **性能监控**: 实时性能指标
- **性能对比**: 多后端性能比较

### 4. 配置管理API
- **配置加载**: 从文件或环境变量加载配置
- **配置验证**: 配置有效性检查
- **配置更新**: 动态更新配置

### 5. 调试工具API
- **日志管理**: 日志级别和输出控制
- **错误诊断**: 错误信息和堆栈跟踪
- **调试信息**: 中间结果和状态查询

## 🚀 快速开始

### Python环境设置
```bash
# 安装YiRage
pip install yica-yirage

# 或从源码安装
cd yirage/python
pip install -e .
```

### 基本使用示例
```python
import yirage
import torch

# 创建一个简单的模型
model = torch.nn.Linear(1024, 1024)
input_tensor = torch.randn(32, 1024)

# 使用YiRage优化
optimizer = yirage.Optimizer()
optimized_model = optimizer.optimize(model, input_tensor)

# 性能对比
with yirage.profiler():
    # 原始模型
    original_output = model(input_tensor)
    
    # 优化后模型
    optimized_output = optimized_model(input_tensor)
```

### C++集成示例
```cpp
#include "yirage/core.h"

int main() {
    // 初始化YiRage
    yirage::initialize();
    
    // 创建计算图
    auto graph = yirage::Graph::create();
    
    // 添加算子
    auto matmul = graph->add_operator("matmul", {1024, 1024});
    
    // 优化图
    auto optimizer = yirage::Optimizer::create();
    auto optimized_graph = optimizer->optimize(graph);
    
    // 执行计算
    auto result = optimized_graph->execute();
    
    return 0;
}
```

## 📖 详细文档

### 参数说明
所有API函数都提供详细的参数说明：
- **必需参数**: 必须提供的参数
- **可选参数**: 有默认值的参数
- **类型说明**: 参数的数据类型
- **取值范围**: 参数的有效取值

### 返回值说明
详细说明每个API的返回值：
- **返回类型**: 返回值的数据类型
- **返回结构**: 复杂返回值的结构说明
- **错误处理**: 异常情况的处理方式

### 使用注意事项
- **线程安全**: API的线程安全性说明
- **内存管理**: 内存分配和释放注意事项
- **性能考虑**: 性能相关的使用建议

## 🔧 配置选项

### YiRage配置
```python
yirage_config = {
    "backend": "yica",                    # 后端选择
    "optimization_level": "O3",           # 优化级别
    "enable_profiling": True,             # 启用性能分析
    "cache_directory": "~/.yirage/cache", # 缓存目录
    "verbose": False                      # 详细输出
}
```

### YICA特定配置
```python
yica_config = {
    "hardware_config": {
        "cim_arrays": 512,               # CIM阵列数量
        "spm_size": "16MB",              # SPM大小
        "dram_bandwidth": "1TB/s"        # DRAM带宽
    },
    "optimization_strategy": "throughput_optimal",
    "enable_kernel_fusion": True,
    "enable_cim_parallelization": True
}
```

## ❗ 错误处理

### 常见错误类型
- **YirageError**: YiRage通用错误
- **BackendError**: 后端相关错误
- **ConfigError**: 配置错误
- **OptimizationError**: 优化过程错误

### 错误处理示例
```python
try:
    result = optimizer.superoptimize(backend="yica")
except yirage.BackendError as e:
    print(f"后端错误: {e}")
    # 回退到其他后端
    result = optimizer.superoptimize(backend="cuda")
except yirage.OptimizationError as e:
    print(f"优化失败: {e}")
    # 使用默认配置重试
    result = optimizer.superoptimize(backend="yica", use_default_config=True)
```

## 🔗 相关资源

### 内部文档
- [架构设计](../architecture/) - 系统架构
- [开发指南](../development/) - 开发环境
- [部署运维](../deployment/) - 部署配置

### 示例代码
- [基础示例](../../yirage/demo/) - 简单使用示例
- [高级示例](../../yirage/examples/) - 复杂应用场景
- [基准测试](../../yirage/benchmark/) - 性能测试代码

### 外部资源
- [PyTorch文档](https://pytorch.org/docs/)
- [CUDA编程指南](https://docs.nvidia.com/cuda/)
- [Triton文档](https://triton-lang.org/)

---

*API文档将随着功能的增加而持续更新。欢迎贡献API使用示例和最佳实践。*

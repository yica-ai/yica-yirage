#pragma once

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <cstdint>

namespace yica {
namespace analyzer {

/**
 * YICA 架构配置参数
 * 描述具体 YICA 硬件的配置信息
 */
struct YICAArchConfig {
    // CIM 阵列配置
    size_t cim_array_rows = 256;
    size_t cim_array_cols = 256;
    size_t num_cim_dies = 16;
    float cim_frequency_mhz = 1000.0f;
    
    // 内存层次配置
    size_t spm_size_per_die = 2 * 1024 * 1024;  // 2MB SPM per die
    size_t dram_size_gb = 64;                   // 64GB DRAM
    float dram_bandwidth_gbs = 1024.0f;         // 1TB/s bandwidth
    
    // 通信配置
    float inter_cim_latency_ns = 10.0f;         // 跨CIM延迟
    float spm_access_latency_cycles = 2.0f;     // SPM访问延迟
    float dram_access_latency_ns = 100.0f;      // DRAM访问延迟
    
    // 能耗参数
    float cim_energy_per_op_pj = 0.1f;          // CIM操作能耗
    float spm_energy_per_access_pj = 1.0f;      // SPM访问能耗
    float dram_energy_per_access_pj = 100.0f;   // DRAM访问能耗
    
    // 精度支持
    std::vector<std::string> supported_dtypes = {"fp32", "fp16", "bf16", "int8", "int4"};
    
    // 获取默认配置
    static YICAArchConfig get_default_config();
    
    // 验证配置有效性
    bool is_valid() const;
};

/**
 * 计算图算子表示
 * 简化的算子抽象，用于分析
 */
struct OperatorNode {
    enum OpType {
        MATMUL,           // 矩阵乘法
        CONV2D,           // 2D卷积
        ELEMENTWISE,      // 元素级运算(Add, Mul, ReLU等)
        REDUCTION,        // 归约运算(Sum, Max, Mean等)
        TRANSPOSE,        // 转置
        RESHAPE,          // 重塑
        SOFTMAX,          // Softmax
        LAYERNORM,        // LayerNorm
        ATTENTION,        // Attention
        UNKNOWN           // 未知类型
    };
    
    OpType op_type;
    std::string op_name;
    
    // 输入输出张量描述
    struct TensorDesc {
        std::vector<int64_t> shape;
        std::string dtype;
        size_t size_bytes;
        
        size_t total_elements() const;
        float get_dtype_size() const;
    };
    
    std::vector<TensorDesc> input_tensors;
    std::vector<TensorDesc> output_tensors;
    
    // 计算复杂度估计
    int64_t flops = 0;
    int64_t memory_accesses = 0;
    
    // 是否为CIM友好的操作
    bool is_cim_friendly() const;
    
    // 获取操作的数据重用因子
    float get_data_reuse_factor() const;
};

/**
 * 计算图表示
 */
struct ComputeGraph {
    std::vector<OperatorNode> operators;
    std::map<std::string, int> tensor_to_op_map;  // 张量名 -> 操作索引映射
    
    // 图统计信息
    int64_t total_flops() const;
    int64_t total_memory_accesses() const;
    size_t total_parameters() const;
    
    // 获取关键路径
    std::vector<int> get_critical_path() const;
};

/**
 * 并行化机会描述
 */
struct ParallelizationOpportunity {
    enum Type {
        DATA_PARALLEL,     // 数据并行
        MODEL_PARALLEL,    // 模型并行
        PIPELINE_PARALLEL, // 流水线并行
        CIM_PARALLEL       // CIM阵列级并行
    };
    
    Type type;
    std::vector<int> involved_ops;      // 涉及的算子索引
    float efficiency_score;             // 并行效率评分 [0-1]
    size_t recommended_parallelism;     // 推荐并行度
    std::string description;            // 并行化描述
};

/**
 * YICA 分析结果
 */
struct YICAAnalysisResult {
    // 核心评分指标 [0-1]
    float cim_friendliness_score = 0.0f;      // CIM操作友好度
    float memory_locality_score = 0.0f;       // 内存局部性评分
    float parallelization_potential = 0.0f;   // 并行化潜力
    float energy_efficiency_score = 0.0f;     // 能效评分
    float overall_yica_suitability = 0.0f;    // 总体YICA适配性
    
    // 详细分析数据
    float cim_utilization_estimate = 0.0f;    // CIM阵列利用率估计
    float spm_hit_rate_estimate = 0.0f;       // SPM命中率估计
    float compute_memory_ratio = 0.0f;        // 计算访存比
    
    // 瓶颈分析
    std::vector<std::string> bottlenecks;     // 性能瓶颈列表
    std::vector<std::string> optimization_suggestions;  // 优化建议
    
    // 并行化分析
    std::vector<ParallelizationOpportunity> parallel_opportunities;
    
    // 算子级别分析
    std::vector<float> per_op_cim_scores;     // 每个算子的CIM友好度
    std::vector<std::string> per_op_bottlenecks; // 每个算子的瓶颈
    
    // 性能预测
    float estimated_latency_ms = 0.0f;        // 预估延迟
    float estimated_throughput_ops = 0.0f;    // 预估吞吐量
    float estimated_energy_mj = 0.0f;         // 预估能耗
    
    // 生成综合报告
    std::string generate_report() const;
    
    // 获取关键优化建议
    std::vector<std::string> get_top_optimizations(size_t top_k = 5) const;
};

/**
 * YICA 架构感知分析器主类
 */
class YICAArchitectureAnalyzer {
public:
    explicit YICAArchitectureAnalyzer(const YICAArchConfig& config);
    ~YICAArchitectureAnalyzer() = default;
    
    // 主要分析接口
    YICAAnalysisResult analyze_computation_pattern(const ComputeGraph& graph) const;
    
    // 分步分析接口
    float calculate_cim_friendliness(const ComputeGraph& graph) const;
    float analyze_memory_access_pattern(const ComputeGraph& graph) const;
    std::vector<ParallelizationOpportunity> find_parallel_patterns(const ComputeGraph& graph) const;
    std::vector<std::string> identify_bottlenecks(const ComputeGraph& graph) const;
    
    // 算子级别分析
    float calculate_op_cim_friendliness(const OperatorNode& op) const;
    float estimate_op_memory_cost(const OperatorNode& op) const;
    float estimate_op_energy_cost(const OperatorNode& op) const;
    
    // 性能预测
    float predict_execution_latency(const ComputeGraph& graph) const;
    float predict_throughput(const ComputeGraph& graph) const;
    float predict_energy_consumption(const ComputeGraph& graph) const;
    
    // 配置管理
    void update_arch_config(const YICAArchConfig& new_config);
    const YICAArchConfig& get_arch_config() const { return arch_config_; }
    
    // 分析统计
    struct AnalysisStats {
        size_t total_analyses = 0;
        double avg_analysis_time_ms = 0.0;
        size_t cache_hits = 0;
        size_t cache_misses = 0;
    };
    AnalysisStats get_stats() const { return stats_; }
    void reset_stats();

private:
    YICAArchConfig arch_config_;
    mutable AnalysisStats stats_;
    
    // 分析结果缓存
    mutable std::map<std::string, YICAAnalysisResult> analysis_cache_;
    
    // 内部分析方法
    float calculate_spm_utilization(const ComputeGraph& graph) const;
    float calculate_cim_array_efficiency(const ComputeGraph& graph) const;
    float calculate_communication_overhead(const ComputeGraph& graph) const;
    
    // 并行化分析辅助方法
    std::vector<ParallelizationOpportunity> analyze_data_parallelism(const ComputeGraph& graph) const;
    std::vector<ParallelizationOpportunity> analyze_model_parallelism(const ComputeGraph& graph) const;
    std::vector<ParallelizationOpportunity> analyze_pipeline_parallelism(const ComputeGraph& graph) const;
    std::vector<ParallelizationOpportunity> analyze_cim_parallelism(const ComputeGraph& graph) const;
    
    // 性能建模
    float model_matmul_performance(const OperatorNode& op) const;
    float model_conv_performance(const OperatorNode& op) const;
    float model_elementwise_performance(const OperatorNode& op) const;
    float model_memory_transfer_cost(size_t data_size, bool is_spm_to_spm) const;
    
    // 缓存管理
    std::string generate_cache_key(const ComputeGraph& graph) const;
    bool try_get_cached_result(const std::string& key, YICAAnalysisResult& result) const;
    void cache_result(const std::string& key, const YICAAnalysisResult& result) const;
    
    // 实用工具方法
    static float normalize_score(float raw_score, float min_val, float max_val);
    static std::vector<float> calculate_weighted_average(const std::vector<float>& scores, 
                                                        const std::vector<float>& weights);
};

/**
 * 工厂类，用于创建不同类型的分析器
 */
class YICAAnalyzerFactory {
public:
    enum AnalyzerType {
        STANDARD,          // 标准分析器
        FAST,             // 快速分析器(精度稍低但速度快)
        DETAILED,         // 详细分析器(精度高但速度慢)
        ENERGY_FOCUSED,   // 能效优先分析器
        PERFORMANCE_FOCUSED // 性能优先分析器
    };
    
    static std::unique_ptr<YICAArchitectureAnalyzer> create_analyzer(
        AnalyzerType type, 
        const YICAArchConfig& config = YICAArchConfig::get_default_config()
    );
    
    static YICAArchConfig get_optimized_config_for_workload(const ComputeGraph& graph);
};

// 便捷函数
namespace utils {
    // 从文件加载计算图
    ComputeGraph load_graph_from_file(const std::string& file_path);
    
    // 将分析结果保存到文件
    void save_analysis_result(const YICAAnalysisResult& result, const std::string& file_path);
    
    // 生成性能对比报告
    std::string generate_comparison_report(const std::vector<YICAAnalysisResult>& results,
                                         const std::vector<std::string>& labels);
    
    // 算子类型字符串转换
    std::string op_type_to_string(OperatorNode::OpType type);
    OperatorNode::OpType string_to_op_type(const std::string& type_str);
    
    // 性能评分归一化
    float normalize_performance_score(float raw_score, OperatorNode::OpType op_type);
}

} // namespace analyzer
} // namespace yica 
#include "yica_analyzer.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <sstream>
#include <iomanip>

namespace yica {
namespace analyzer {

// ============================================================================
// YICAArchConfig 实现
// ============================================================================

YICAArchConfig YICAArchConfig::get_default_config() {
    YICAArchConfig config;
    
    // 标准 YICA v2 配置
    config.cim_array_rows = 512;
    config.cim_array_cols = 512;
    config.num_cim_dies = 32;
    config.cim_frequency_mhz = 1200.0f;
    
    config.spm_size_per_die = 4 * 1024 * 1024;  // 4MB per die
    config.dram_size_gb = 128;
    config.dram_bandwidth_gbs = 2048.0f;        // 2TB/s
    
    config.inter_cim_latency_ns = 8.0f;
    config.spm_access_latency_cycles = 1.5f;
    config.dram_access_latency_ns = 80.0f;
    
    config.cim_energy_per_op_pj = 0.08f;
    config.spm_energy_per_access_pj = 0.8f;
    config.dram_energy_per_access_pj = 80.0f;
    
    return config;
}

bool YICAArchConfig::is_valid() const {
    return cim_array_rows > 0 && cim_array_cols > 0 && 
           num_cim_dies > 0 && cim_frequency_mhz > 0 &&
           spm_size_per_die > 0 && dram_size_gb > 0 &&
           dram_bandwidth_gbs > 0;
}

// ============================================================================
// OperatorNode 实现
// ============================================================================

size_t OperatorNode::TensorDesc::total_elements() const {
    return std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
}

float OperatorNode::TensorDesc::get_dtype_size() const {
    if (dtype == "fp32") return 4.0f;
    if (dtype == "fp16" || dtype == "bf16") return 2.0f;
    if (dtype == "int8") return 1.0f;
    if (dtype == "int4") return 0.5f;
    if (dtype == "fp64") return 8.0f;
    return 4.0f; // 默认为fp32
}

bool OperatorNode::is_cim_friendly() const {
    switch (op_type) {
        case MATMUL:
        case CONV2D:
            return true;  // 矩阵运算对CIM最友好
        case ELEMENTWISE:
            return false; // 元素级运算适合SPM
        case REDUCTION:
            return false; // 归约需要跨CIM通信
        case ATTENTION:
            return true;  // 包含大量矩阵运算
        default:
            return false;
    }
}

float OperatorNode::get_data_reuse_factor() const {
    if (input_tensors.empty()) return 1.0f;
    
    int64_t total_ops = flops;
    int64_t total_data = 0;
    
    for (const auto& tensor : input_tensors) {
        total_data += tensor.total_elements() * static_cast<int64_t>(tensor.get_dtype_size());
    }
    
    if (total_data == 0) return 1.0f;
    return static_cast<float>(total_ops) / static_cast<float>(total_data);
}

// ============================================================================
// ComputeGraph 实现
// ============================================================================

int64_t ComputeGraph::total_flops() const {
    return std::accumulate(operators.begin(), operators.end(), 0LL,
                          [](int64_t sum, const OperatorNode& op) {
                              return sum + op.flops;
                          });
}

int64_t ComputeGraph::total_memory_accesses() const {
    return std::accumulate(operators.begin(), operators.end(), 0LL,
                          [](int64_t sum, const OperatorNode& op) {
                              return sum + op.memory_accesses;
                          });
}

size_t ComputeGraph::total_parameters() const {
    size_t total = 0;
    for (const auto& op : operators) {
        for (const auto& tensor : op.input_tensors) {
            total += tensor.total_elements();
        }
    }
    return total;
}

std::vector<int> ComputeGraph::get_critical_path() const {
    // 简化的关键路径分析：找到FLOP最多的操作序列
    std::vector<int> critical_path;
    
    for (size_t i = 0; i < operators.size(); ++i) {
        if (operators[i].flops > 0) {
            critical_path.push_back(static_cast<int>(i));
        }
    }
    
    // 按FLOP数量排序
    std::sort(critical_path.begin(), critical_path.end(),
              [this](int a, int b) {
                  return operators[a].flops > operators[b].flops;
              });
    
    return critical_path;
}

// ============================================================================
// YICAArchitectureAnalyzer 实现
// ============================================================================

YICAArchitectureAnalyzer::YICAArchitectureAnalyzer(const YICAArchConfig& config)
    : arch_config_(config) {
    if (!arch_config_.is_valid()) {
        throw std::invalid_argument("Invalid YICA architecture configuration");
    }
}

YICAAnalysisResult YICAArchitectureAnalyzer::analyze_computation_pattern(const ComputeGraph& graph) const {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 检查缓存
    std::string cache_key = generate_cache_key(graph);
    YICAAnalysisResult cached_result;
    if (try_get_cached_result(cache_key, cached_result)) {
        ++stats_.cache_hits;
        return cached_result;
    }
    ++stats_.cache_misses;
    
    YICAAnalysisResult result;
    
    // 1. 计算核心评分指标
    result.cim_friendliness_score = calculate_cim_friendliness(graph);
    result.memory_locality_score = analyze_memory_access_pattern(graph);
    result.parallelization_potential = 0.0f; // 稍后计算
    result.energy_efficiency_score = 0.0f;   // 稍后计算
    
    // 2. 并行化分析
    result.parallel_opportunities = find_parallel_patterns(graph);
    if (!result.parallel_opportunities.empty()) {
        // 计算平均并行化潜力
        float avg_parallel_score = 0.0f;
        for (const auto& opp : result.parallel_opportunities) {
            avg_parallel_score += opp.efficiency_score;
        }
        result.parallelization_potential = avg_parallel_score / result.parallel_opportunities.size();
    }
    
    // 3. 详细性能分析
    result.cim_utilization_estimate = calculate_cim_array_efficiency(graph);
    result.spm_hit_rate_estimate = calculate_spm_utilization(graph);
    result.compute_memory_ratio = static_cast<float>(graph.total_flops()) / 
                                 std::max(1LL, graph.total_memory_accesses());
    
    // 4. 瓶颈识别
    result.bottlenecks = identify_bottlenecks(graph);
    
    // 5. 性能预测
    result.estimated_latency_ms = predict_execution_latency(graph);
    result.estimated_throughput_ops = predict_throughput(graph);
    result.estimated_energy_mj = predict_energy_consumption(graph);
    
    // 6. 能效评分
    if (result.estimated_energy_mj > 0) {
        result.energy_efficiency_score = result.estimated_throughput_ops / result.estimated_energy_mj;
        result.energy_efficiency_score = normalize_score(result.energy_efficiency_score, 0.0f, 1000.0f);
    }
    
    // 7. 算子级别分析
    result.per_op_cim_scores.reserve(graph.operators.size());
    result.per_op_bottlenecks.reserve(graph.operators.size());
    
    for (const auto& op : graph.operators) {
        result.per_op_cim_scores.push_back(calculate_op_cim_friendliness(op));
        
        std::string op_bottleneck = "none";
        if (op.memory_accesses > op.flops) {
            op_bottleneck = "memory_bound";
        } else if (op.flops > arch_config_.cim_array_rows * arch_config_.cim_array_cols * 1000) {
            op_bottleneck = "compute_bound";
        }
        result.per_op_bottlenecks.push_back(op_bottleneck);
    }
    
    // 8. 综合适配性评分
    std::vector<float> scores = {
        result.cim_friendliness_score,
        result.memory_locality_score,
        result.parallelization_potential,
        result.energy_efficiency_score
    };
    std::vector<float> weights = {0.35f, 0.25f, 0.25f, 0.15f}; // 权重可调
    
    result.overall_yica_suitability = 0.0f;
    for (size_t i = 0; i < scores.size(); ++i) {
        result.overall_yica_suitability += scores[i] * weights[i];
    }
    
    // 9. 生成优化建议
    result.optimization_suggestions = generate_optimization_suggestions(result);
    
    // 缓存结果
    cache_result(cache_key, result);
    
    // 更新统计
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    ++stats_.total_analyses;
    stats_.avg_analysis_time_ms = (stats_.avg_analysis_time_ms * (stats_.total_analyses - 1) + 
                                  duration.count() / 1000.0) / stats_.total_analyses;
    
    return result;
}

float YICAArchitectureAnalyzer::calculate_cim_friendliness(const ComputeGraph& graph) const {
    if (graph.operators.empty()) return 0.0f;
    
    float total_weighted_score = 0.0f;
    int64_t total_flops = 0;
    
    for (const auto& op : graph.operators) {
        float op_score = calculate_op_cim_friendliness(op);
        total_weighted_score += op_score * op.flops;
        total_flops += op.flops;
    }
    
    return total_flops > 0 ? total_weighted_score / total_flops : 0.0f;
}

float YICAArchitectureAnalyzer::calculate_op_cim_friendliness(const OperatorNode& op) const {
    float base_score = 0.0f;
    
    // 基础评分根据算子类型
    switch (op.op_type) {
        case OperatorNode::MATMUL:
            base_score = 0.95f;  // 矩阵乘法最适合CIM
            break;
        case OperatorNode::CONV2D:
            base_score = 0.90f;  // 卷积也很适合
            break;
        case OperatorNode::ATTENTION:
            base_score = 0.85f;  // 注意力机制包含矩阵运算
            break;
        case OperatorNode::ELEMENTWISE:
            base_score = 0.30f;  // 元素级运算不太适合CIM
            break;
        case OperatorNode::REDUCTION:
            base_score = 0.40f;  // 归约运算部分适合
            break;
        case OperatorNode::SOFTMAX:
            base_score = 0.35f;  // Softmax有归约部分
            break;
        case OperatorNode::LAYERNORM:
            base_score = 0.45f;  // LayerNorm有一些矩阵运算
            break;
        default:
            base_score = 0.20f;
    }
    
    // 根据数据大小调整评分
    if (!op.input_tensors.empty()) {
        size_t total_elements = 0;
        for (const auto& tensor : op.input_tensors) {
            total_elements += tensor.total_elements();
        }
        
        size_t spm_capacity = arch_config_.spm_size_per_die / 4; // 假设用1/4 SPM
        if (total_elements * 4 <= spm_capacity) {  // 假设fp32
            base_score *= 1.2f;  // 能放入SPM的运算更友好
        } else if (total_elements * 4 > arch_config_.spm_size_per_die * 4) {
            base_score *= 0.8f;  // 太大的数据会影响效率
        }
    }
    
    // 根据数据重用因子调整
    float reuse_factor = op.get_data_reuse_factor();
    if (reuse_factor > 2.0f) {
        base_score *= 1.1f;  // 高重用率对CIM有利
    }
    
    return std::min(1.0f, base_score);
}

float YICAArchitectureAnalyzer::analyze_memory_access_pattern(const ComputeGraph& graph) const {
    if (graph.operators.empty()) return 0.0f;
    
    size_t total_memory_accesses = 0;
    size_t spm_friendly_accesses = 0;
    size_t sequential_accesses = 0;
    
    for (const auto& op : graph.operators) {
        for (const auto& tensor : op.input_tensors) {
            size_t tensor_size = tensor.total_elements() * static_cast<size_t>(tensor.get_dtype_size());
            total_memory_accesses += tensor_size;
            
            // 判断是否适合SPM
            if (tensor_size <= arch_config_.spm_size_per_die / 2) {  // 保守估计使用一半SPM
                spm_friendly_accesses += tensor_size;
            }
            
            // 假设矩阵运算有更好的访问局部性
            if (op.op_type == OperatorNode::MATMUL || op.op_type == OperatorNode::CONV2D) {
                sequential_accesses += tensor_size;
            }
        }
    }
    
    if (total_memory_accesses == 0) return 0.0f;
    
    float spm_ratio = static_cast<float>(spm_friendly_accesses) / total_memory_accesses;
    float locality_ratio = static_cast<float>(sequential_accesses) / total_memory_accesses;
    
    // 综合评分：70%看SPM适配性，30%看访问局部性
    return 0.7f * spm_ratio + 0.3f * locality_ratio;
}

std::vector<ParallelizationOpportunity> YICAArchitectureAnalyzer::find_parallel_patterns(const ComputeGraph& graph) const {
    std::vector<ParallelizationOpportunity> opportunities;
    
    // 分析数据并行机会
    auto data_parallel_opps = analyze_data_parallelism(graph);
    opportunities.insert(opportunities.end(), data_parallel_opps.begin(), data_parallel_opps.end());
    
    // 分析模型并行机会
    auto model_parallel_opps = analyze_model_parallelism(graph);
    opportunities.insert(opportunities.end(), model_parallel_opps.begin(), model_parallel_opps.end());
    
    // 分析CIM并行机会
    auto cim_parallel_opps = analyze_cim_parallelism(graph);
    opportunities.insert(opportunities.end(), cim_parallel_opps.begin(), cim_parallel_opps.end());
    
    return opportunities;
}

std::vector<ParallelizationOpportunity> YICAArchitectureAnalyzer::analyze_data_parallelism(const ComputeGraph& graph) const {
    std::vector<ParallelizationOpportunity> opportunities;
    
    for (size_t i = 0; i < graph.operators.size(); ++i) {
        const auto& op = graph.operators[i];
        
        // 检查是否适合数据并行
        if (op.op_type == OperatorNode::MATMUL || op.op_type == OperatorNode::CONV2D ||
            op.op_type == OperatorNode::ELEMENTWISE) {
            
            ParallelizationOpportunity opp;
            opp.type = ParallelizationOpportunity::DATA_PARALLEL;
            opp.involved_ops = {static_cast<int>(i)};
            
            // 估算并行效率
            if (!op.input_tensors.empty() && !op.input_tensors[0].shape.empty()) {
                int64_t batch_size = op.input_tensors[0].shape[0];
                opp.recommended_parallelism = std::min(static_cast<size_t>(batch_size), 
                                                      arch_config_.num_cim_dies);
                opp.efficiency_score = std::min(1.0f, static_cast<float>(batch_size) / 
                                               arch_config_.num_cim_dies);
            } else {
                opp.recommended_parallelism = arch_config_.num_cim_dies;
                opp.efficiency_score = 0.8f;  // 默认效率
            }
            
            opp.description = "Data parallel across batch dimension";
            opportunities.push_back(opp);
        }
    }
    
    return opportunities;
}

std::vector<ParallelizationOpportunity> YICAArchitectureAnalyzer::analyze_cim_parallelism(const ComputeGraph& graph) const {
    std::vector<ParallelizationOpportunity> opportunities;
    
    for (size_t i = 0; i < graph.operators.size(); ++i) {
        const auto& op = graph.operators[i];
        
        if (op.op_type == OperatorNode::MATMUL) {
            ParallelizationOpportunity opp;
            opp.type = ParallelizationOpportunity::CIM_PARALLEL;
            opp.involved_ops = {static_cast<int>(i)};
            
            // 对于矩阵乘法，可以跨多个CIM阵列并行
            if (op.input_tensors.size() >= 2) {
                auto& A = op.input_tensors[0];
                auto& B = op.input_tensors[1];
                
                if (A.shape.size() >= 2 && B.shape.size() >= 2) {
                    int64_t M = A.shape[A.shape.size()-2];
                    int64_t N = B.shape[B.shape.size()-1];
                    
                    // 估算需要的CIM阵列数量
                    size_t needed_arrays = std::max(1UL, 
                        static_cast<size_t>((M * N) / (arch_config_.cim_array_rows * arch_config_.cim_array_cols)));
                    
                    opp.recommended_parallelism = std::min(needed_arrays, arch_config_.num_cim_dies);
                    opp.efficiency_score = std::min(1.0f, static_cast<float>(needed_arrays) / arch_config_.num_cim_dies);
                    opp.description = "Matrix multiplication across multiple CIM arrays";
                    
                    opportunities.push_back(opp);
                }
            }
        }
    }
    
    return opportunities;
}

// 继续实现其他方法...
std::vector<std::string> YICAArchitectureAnalyzer::identify_bottlenecks(const ComputeGraph& graph) const {
    std::vector<std::string> bottlenecks;
    
    // 计算总的计算量和内存访问量
    int64_t total_flops = graph.total_flops();
    int64_t total_memory = graph.total_memory_accesses();
    
    // 估算理论峰值性能
    float peak_compute_ops_per_sec = arch_config_.num_cim_dies * 
                                    arch_config_.cim_array_rows * 
                                    arch_config_.cim_array_cols * 
                                    arch_config_.cim_frequency_mhz * 1e6;
    
    float peak_memory_bw_bytes_per_sec = arch_config_.dram_bandwidth_gbs * 1e9;
    
    // 计算理论执行时间
    float compute_time = total_flops / peak_compute_ops_per_sec;
    float memory_time = total_memory / peak_memory_bw_bytes_per_sec;
    
    if (memory_time > compute_time * 1.5f) {
        bottlenecks.push_back("memory_bandwidth_bound");
    }
    
    if (compute_time > memory_time * 1.5f) {
        bottlenecks.push_back("compute_bound");
    }
    
    // 分析CIM利用率
    float cim_utilization = calculate_cim_array_efficiency(graph);
    if (cim_utilization < 0.5f) {
        bottlenecks.push_back("low_cim_utilization");
    }
    
    // 分析SPM命中率
    float spm_hit_rate = calculate_spm_utilization(graph);
    if (spm_hit_rate < 0.6f) {
        bottlenecks.push_back("poor_spm_locality");
    }
    
    // 分析通信开销
    float comm_overhead = calculate_communication_overhead(graph);
    if (comm_overhead > 0.3f) {
        bottlenecks.push_back("high_communication_overhead");
    }
    
    if (bottlenecks.empty()) {
        bottlenecks.push_back("well_optimized");
    }
    
    return bottlenecks;
}

std::vector<std::string> YICAArchitectureAnalyzer::generate_optimization_suggestions(const YICAAnalysisResult& result) const {
    std::vector<std::string> suggestions;
    
    // 基于瓶颈分析给出建议
    for (const auto& bottleneck : result.bottlenecks) {
        if (bottleneck == "memory_bandwidth_bound") {
            suggestions.push_back("考虑数据压缩或使用更高带宽的存储");
            suggestions.push_back("优化数据访问模式以提高缓存命中率");
        } else if (bottleneck == "low_cim_utilization") {
            suggestions.push_back("增加算子融合以提高CIM阵列利用率");
            suggestions.push_back("考虑矩阵分块策略优化");
        } else if (bottleneck == "poor_spm_locality") {
            suggestions.push_back("重新组织数据布局以提高SPM命中率");
            suggestions.push_back("考虑数据预取策略");
        }
    }
    
    // 基于评分给出建议
    if (result.cim_friendliness_score < 0.6f) {
        suggestions.push_back("考虑算子重写以更好利用CIM阵列");
    }
    
    if (result.parallelization_potential > 0.7f) {
        suggestions.push_back("可以进一步提高并行度");
    }
    
    if (result.energy_efficiency_score < 0.5f) {
        suggestions.push_back("考虑降低精度以提高能效");
        suggestions.push_back("优化数据移动以减少能耗");
    }
    
    return suggestions;
}

// 工具函数实现
float YICAArchitectureAnalyzer::normalize_score(float raw_score, float min_val, float max_val) {
    if (max_val <= min_val) return 0.0f;
    return std::max(0.0f, std::min(1.0f, (raw_score - min_val) / (max_val - min_val)));
}

std::string YICAArchitectureAnalyzer::generate_cache_key(const ComputeGraph& graph) const {
    std::stringstream ss;
    ss << graph.operators.size() << "_" << graph.total_flops() << "_" << graph.total_memory_accesses();
    return ss.str();
}

bool YICAArchitectureAnalyzer::try_get_cached_result(const std::string& key, YICAAnalysisResult& result) const {
    auto it = analysis_cache_.find(key);
    if (it != analysis_cache_.end()) {
        result = it->second;
        return true;
    }
    return false;
}

void YICAArchitectureAnalyzer::cache_result(const std::string& key, const YICAAnalysisResult& result) const {
    // 简单的LRU缓存：保持最多100个结果
    if (analysis_cache_.size() >= 100) {
        auto oldest = analysis_cache_.begin();
        analysis_cache_.erase(oldest);
    }
    analysis_cache_[key] = result;
}

// ============================================================================
// YICAAnalysisResult 实现
// ============================================================================

std::string YICAAnalysisResult::generate_report() const {
    std::stringstream report;
    
    report << "YICA Architecture Analysis Report\n";
    report << "================================\n\n";
    
    report << "Overall YICA Suitability: " << std::fixed << std::setprecision(2) 
           << overall_yica_suitability * 100 << "%\n\n";
    
    report << "Core Metrics:\n";
    report << "  CIM Friendliness: " << cim_friendliness_score * 100 << "%\n";
    report << "  Memory Locality: " << memory_locality_score * 100 << "%\n";
    report << "  Parallelization Potential: " << parallelization_potential * 100 << "%\n";
    report << "  Energy Efficiency: " << energy_efficiency_score * 100 << "%\n\n";
    
    report << "Performance Estimates:\n";
    report << "  Estimated Latency: " << estimated_latency_ms << " ms\n";
    report << "  Estimated Throughput: " << estimated_throughput_ops << " ops/sec\n";
    report << "  Estimated Energy: " << estimated_energy_mj << " mJ\n\n";
    
    if (!bottlenecks.empty()) {
        report << "Identified Bottlenecks:\n";
        for (const auto& bottleneck : bottlenecks) {
            report << "  - " << bottleneck << "\n";
        }
        report << "\n";
    }
    
    if (!optimization_suggestions.empty()) {
        report << "Optimization Suggestions:\n";
        for (const auto& suggestion : optimization_suggestions) {
            report << "  - " << suggestion << "\n";
        }
        report << "\n";
    }
    
    return report.str();
}

std::vector<std::string> YICAAnalysisResult::get_top_optimizations(size_t top_k) const {
    std::vector<std::string> top_optimizations;
    
    size_t limit = std::min(top_k, optimization_suggestions.size());
    for (size_t i = 0; i < limit; ++i) {
        top_optimizations.push_back(optimization_suggestions[i]);
    }
    
    return top_optimizations;
}

} // namespace analyzer
} // namespace yica 
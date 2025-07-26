/**
 * YICA CIM阵列优化器测试
 * 测试存算一体(CIM)阵列的操作、映射和调度优化
 */

#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>

#ifdef USE_BUILTIN_TEST
#include "test_framework.h"
#define CIM_TEST(name) BUILTIN_TEST(name)
#else
#include <gtest/gtest.h>
#define CIM_TEST(name) TEST(CIMArrayOptimizer, name)
#define EXPECT_TRUE_MSG(condition, msg) EXPECT_TRUE(condition) << msg
#define EXPECT_EQ_MSG(expected, actual, msg) EXPECT_EQ(expected, actual) << msg
#define EXPECT_GT_MSG(val1, val2, msg) EXPECT_GT(val1, val2) << msg
#endif

// CIM阵列配置
struct CIMArrayConfig {
    int array_id;
    int rows;
    int cols;
    float precision_bits;  // 计算精度（位数）
    float energy_per_op;   // 每次操作的能耗（pJ）
    float max_frequency_mhz;
    bool supports_mixed_precision;
};

// CIM操作类型
enum class CIMOperationType {
    MATRIX_MULTIPLY,
    VECTOR_ADD,
    ELEMENT_WISE_MULTIPLY,
    CONVOLUTION,
    ACTIVATION_FUNCTION
};

// 为CIMOperationType添加输出操作符
std::ostream& operator<<(std::ostream& os, const CIMOperationType& type) {
    switch (type) {
        case CIMOperationType::MATRIX_MULTIPLY:
            return os << "MATRIX_MULTIPLY";
        case CIMOperationType::VECTOR_ADD:
            return os << "VECTOR_ADD";
        case CIMOperationType::ELEMENT_WISE_MULTIPLY:
            return os << "ELEMENT_WISE_MULTIPLY";
        case CIMOperationType::CONVOLUTION:
            return os << "CONVOLUTION";
        case CIMOperationType::ACTIVATION_FUNCTION:
            return os << "ACTIVATION_FUNCTION";
        default:
            return os << "UNKNOWN";
    }
}

// CIM阵列操作描述
struct CIMOperation {
    CIMOperationType type;
    std::vector<int> input_shapes;
    std::vector<int> output_shapes;
    float computational_intensity;  // 计算强度
    float memory_bandwidth_requirement;  // 内存带宽需求
    int estimated_cycles;
};

// CIM阵列调度器
class CIMArrayScheduler {
private:
    std::vector<CIMArrayConfig> arrays_;
    std::vector<bool> array_busy_;
    
public:
    CIMArrayScheduler(const std::vector<CIMArrayConfig>& arrays) 
        : arrays_(arrays), array_busy_(arrays.size(), false) {}
    
    // 找到最适合执行操作的CIM阵列
    int find_best_array_for_operation(const CIMOperation& op) {
        int best_array = -1;
        float best_score = -1.0f;
        
        for (size_t i = 0; i < arrays_.size(); ++i) {
            if (array_busy_[i]) continue;
            
            float score = calculate_suitability_score(arrays_[i], op);
            if (score > best_score) {
                best_score = score;
                best_array = static_cast<int>(i);
            }
        }
        
        return best_array;
    }
    
    // 计算CIM阵列对操作的适合度分数
    float calculate_suitability_score(const CIMArrayConfig& array, const CIMOperation& op) {
        float size_score = 1.0f;
        float energy_score = 1.0f / (array.energy_per_op + 0.001f);  // 能耗越低分数越高
        float frequency_score = array.max_frequency_mhz / 1000.0f;    // 频率越高分数越高
        
        // 根据操作类型调整分数
        switch (op.type) {
            case CIMOperationType::MATRIX_MULTIPLY:
                size_score = std::min(static_cast<float>(array.rows * array.cols) / 1024.0f, 2.0f);
                break;
            case CIMOperationType::VECTOR_ADD:
                size_score = std::min(static_cast<float>(array.cols) / 512.0f, 2.0f);
                break;
            case CIMOperationType::CONVOLUTION:
                size_score = std::min(static_cast<float>(array.rows * array.cols) / 512.0f, 2.0f);
                break;
            default:
                break;
        }
        
        return size_score * energy_score * frequency_score;
    }
    
    // 预留CIM阵列
    bool reserve_array(int array_id) {
        if (array_id < 0 || array_id >= static_cast<int>(arrays_.size())) {
            return false;
        }
        if (array_busy_[array_id]) {
            return false;
        }
        array_busy_[array_id] = true;
        return true;
    }
    
    // 释放CIM阵列
    void release_array(int array_id) {
        if (array_id >= 0 && array_id < static_cast<int>(arrays_.size())) {
            array_busy_[array_id] = false;
        }
    }
    
    // 获取可用阵列数量
    int get_available_array_count() const {
        return std::count(array_busy_.begin(), array_busy_.end(), false);
    }
    
    // 获取总阵列数量
    int get_total_array_count() const {
        return static_cast<int>(arrays_.size());
    }
};

// CIM映射优化器
class CIMOperationMapper {
public:
    // 将大矩阵乘法分解为CIM阵列友好的子操作
    std::vector<CIMOperation> decompose_matrix_multiply(int M, int K, int N, int max_array_size = 256) {
        std::vector<CIMOperation> sub_operations;
        
        // 将矩阵分块以适应CIM阵列大小
        int block_m = std::min(M, max_array_size);
        int block_k = std::min(K, max_array_size);
        int block_n = std::min(N, max_array_size);
        
        for (int i = 0; i < M; i += block_m) {
            for (int j = 0; j < N; j += block_n) {
                for (int k = 0; k < K; k += block_k) {
                    int actual_m = std::min(block_m, M - i);
                    int actual_k = std::min(block_k, K - k);
                    int actual_n = std::min(block_n, N - j);
                    
                    CIMOperation sub_op;
                    sub_op.type = CIMOperationType::MATRIX_MULTIPLY;
                    sub_op.input_shapes = {actual_m, actual_k, actual_k, actual_n};
                    sub_op.output_shapes = {actual_m, actual_n};
                    sub_op.computational_intensity = 2.0f * actual_m * actual_k * actual_n;
                    sub_op.memory_bandwidth_requirement = (actual_m * actual_k + actual_k * actual_n + actual_m * actual_n) * sizeof(float);
                    sub_op.estimated_cycles = actual_m * actual_k * actual_n / 64;  // 假设每64个操作需要1个周期
                    
                    sub_operations.push_back(sub_op);
                }
            }
        }
        
        return sub_operations;
    }
    
    // 估算操作在CIM阵列上的执行时间和能耗
    std::pair<float, float> estimate_execution_metrics(const CIMOperation& op, const CIMArrayConfig& array) {
        // 执行时间估算（微秒）
        float execution_time_us = static_cast<float>(op.estimated_cycles) / (array.max_frequency_mhz);
        
        // 能耗估算（nJ）
        float total_ops = op.computational_intensity;
        float energy_consumption_nj = total_ops * array.energy_per_op / 1000.0f;  // pJ转nJ
        
        return {execution_time_us, energy_consumption_nj};
    }
};

// 测试用例

CIM_TEST(CIMArrayInitialization) {
    // 创建模拟的YICA-G100 CIM阵列配置
    std::vector<CIMArrayConfig> arrays;
    
    // 8个CIM Die，每个Die 4个Cluster，每个Cluster 16个CIM阵列
    for (int die = 0; die < 8; ++die) {
        for (int cluster = 0; cluster < 4; ++cluster) {
            for (int array = 0; array < 16; ++array) {
                CIMArrayConfig config;
                config.array_id = die * 64 + cluster * 16 + array;  // 全局ID
                config.rows = 256;
                config.cols = 256;
                config.precision_bits = 8.0f;  // INT8精度
                config.energy_per_op = 0.5f;   // 0.5pJ每操作
                config.max_frequency_mhz = 200.0f;  // 200MHz
                config.supports_mixed_precision = true;
                
                arrays.push_back(config);
            }
        }
    }
    
    EXPECT_EQ_MSG(512, arrays.size(), "应该创建512个CIM阵列（8 Dies × 4 Clusters × 16 Arrays）");
    
    CIMArrayScheduler scheduler(arrays);
    EXPECT_EQ_MSG(512, scheduler.get_total_array_count(), "调度器应该管理512个CIM阵列");
    EXPECT_EQ_MSG(512, scheduler.get_available_array_count(), "初始时所有阵列都应该可用");
    
    std::cout << "✅ 成功初始化 " << arrays.size() << " 个CIM阵列" << std::endl;
}

CIM_TEST(CIMArrayScheduling) {
    // 创建简化的CIM阵列配置用于测试
    std::vector<CIMArrayConfig> arrays;
    for (int i = 0; i < 16; ++i) {
        CIMArrayConfig config;
        config.array_id = i;
        config.rows = 256;
        config.cols = 256;
        config.precision_bits = 8.0f;
        config.energy_per_op = 0.5f + i * 0.1f;  // 不同阵列有不同能耗
        config.max_frequency_mhz = 200.0f + i * 10.0f;  // 不同频率
        config.supports_mixed_precision = true;
        arrays.push_back(config);
    }
    
    CIMArrayScheduler scheduler(arrays);
    
    // 创建测试操作
    CIMOperation matmul_op;
    matmul_op.type = CIMOperationType::MATRIX_MULTIPLY;
    matmul_op.input_shapes = {128, 128, 128, 128};
    matmul_op.output_shapes = {128, 128};
    matmul_op.computational_intensity = 2.0f * 128 * 128 * 128;
    matmul_op.memory_bandwidth_requirement = 3 * 128 * 128 * sizeof(float);
    matmul_op.estimated_cycles = 128 * 128 * 128 / 64;
    
    // 测试阵列选择
    int best_array = scheduler.find_best_array_for_operation(matmul_op);
    EXPECT_TRUE_MSG(best_array >= 0, "应该找到合适的CIM阵列");
    EXPECT_TRUE_MSG(best_array < 16, "选择的阵列ID应该在有效范围内");
    
    // 测试阵列预留
    bool reserve_result = scheduler.reserve_array(best_array);
    EXPECT_TRUE_MSG(reserve_result, "应该能够成功预留阵列");
    EXPECT_EQ_MSG(15, scheduler.get_available_array_count(), "预留后可用阵列数应该减1");
    
    // 测试重复预留（应该失败）
    bool duplicate_reserve = scheduler.reserve_array(best_array);
    EXPECT_TRUE_MSG(!duplicate_reserve, "重复预留同一阵列应该失败");
    
    // 测试阵列释放
    scheduler.release_array(best_array);
    EXPECT_EQ_MSG(16, scheduler.get_available_array_count(), "释放后可用阵列数应该恢复");
    
    std::cout << "✅ CIM阵列调度测试成功，最佳阵列ID: " << best_array << std::endl;
}

CIM_TEST(CIMOperationMapping) {
    CIMOperationMapper mapper;
    
    // 测试大矩阵分解
    int M = 1024, K = 1024, N = 1024;
    auto sub_operations = mapper.decompose_matrix_multiply(M, K, N, 256);
    
    // 验证分解结果
    EXPECT_GT_MSG(sub_operations.size(), 1, "大矩阵应该被分解为多个子操作");
    
    // 计算理论子操作数量：ceil(M/256) * ceil(K/256) * ceil(N/256)
    int expected_blocks = ((M + 255) / 256) * ((K + 255) / 256) * ((N + 255) / 256);
    EXPECT_EQ_MSG(expected_blocks, sub_operations.size(), "子操作数量应该匹配理论值");
    
    // 验证每个子操作的有效性
    for (const auto& sub_op : sub_operations) {
        EXPECT_EQ_MSG(CIMOperationType::MATRIX_MULTIPLY, sub_op.type, "子操作类型应该是矩阵乘法");
        EXPECT_TRUE_MSG(sub_op.input_shapes.size() == 4, "矩阵乘法应该有4个输入维度");
        EXPECT_TRUE_MSG(sub_op.output_shapes.size() == 2, "矩阵乘法应该有2个输出维度");
        EXPECT_GT_MSG(sub_op.computational_intensity, 0.0f, "计算强度应该大于0");
        EXPECT_GT_MSG(sub_op.estimated_cycles, 0, "估算周期数应该大于0");
    }
    
    std::cout << "✅ 矩阵分解测试成功，生成 " << sub_operations.size() << " 个子操作" << std::endl;
}

CIM_TEST(CIMPerformanceEstimation) {
    CIMOperationMapper mapper;
    
    // 创建高性能CIM阵列配置
    CIMArrayConfig high_perf_array;
    high_perf_array.array_id = 0;
    high_perf_array.rows = 256;
    high_perf_array.cols = 256;
    high_perf_array.precision_bits = 8.0f;
    high_perf_array.energy_per_op = 0.3f;  // 低能耗
    high_perf_array.max_frequency_mhz = 300.0f;  // 高频率
    high_perf_array.supports_mixed_precision = true;
    
    // 创建低性能CIM阵列配置
    CIMArrayConfig low_perf_array;
    low_perf_array.array_id = 1;
    low_perf_array.rows = 128;
    low_perf_array.cols = 128;
    low_perf_array.precision_bits = 16.0f;
    low_perf_array.energy_per_op = 1.0f;  // 高能耗
    low_perf_array.max_frequency_mhz = 100.0f;  // 低频率
    low_perf_array.supports_mixed_precision = false;
    
    // 创建测试操作
    CIMOperation test_op;
    test_op.type = CIMOperationType::MATRIX_MULTIPLY;
    test_op.computational_intensity = 2.0f * 128 * 128 * 128;  // 128x128 矩阵乘法
    test_op.estimated_cycles = 128 * 128 * 128 / 64;
    
    // 比较性能估算
    auto [high_time, high_energy] = mapper.estimate_execution_metrics(test_op, high_perf_array);
    auto [low_time, low_energy] = mapper.estimate_execution_metrics(test_op, low_perf_array);
    
    // 高性能阵列应该更快
    EXPECT_GT_MSG(low_time, high_time, "低性能阵列执行时间应该更长");
    
    // 高性能阵列能耗应该更低（每操作能耗更低）
    EXPECT_GT_MSG(low_energy, high_energy, "低性能阵列能耗应该更高");
    
    std::cout << "✅ 性能估算测试成功" << std::endl;
    std::cout << "  高性能阵列: " << high_time << "μs, " << high_energy << "nJ" << std::endl;
    std::cout << "  低性能阵列: " << low_time << "μs, " << low_energy << "nJ" << std::endl;
    std::cout << "  性能提升: " << (low_time / high_time) << "x 速度, " 
              << (low_energy / high_energy) << "x 能效" << std::endl;
}

CIM_TEST(CIMArrayUtilization) {
    std::vector<CIMArrayConfig> arrays;
    for (int i = 0; i < 8; ++i) {
        CIMArrayConfig config;
        config.array_id = i;
        config.rows = 256;
        config.cols = 256;
        config.precision_bits = 8.0f;
        config.energy_per_op = 0.5f;
        config.max_frequency_mhz = 200.0f;
        config.supports_mixed_precision = true;
        arrays.push_back(config);
    }
    
    CIMArrayScheduler scheduler(arrays);
    CIMOperationMapper mapper;
    
    // 模拟批量操作调度
    std::vector<CIMOperation> batch_operations;
    for (int i = 0; i < 12; ++i) {  // 12个操作，8个阵列
        CIMOperation op;
        op.type = CIMOperationType::MATRIX_MULTIPLY;
        op.input_shapes = {64, 64, 64, 64};
        op.output_shapes = {64, 64};
        op.computational_intensity = 2.0f * 64 * 64 * 64;
        op.estimated_cycles = 64 * 64 * 64 / 64;
        batch_operations.push_back(op);
    }
    
    // 尝试调度所有操作
    std::vector<int> assigned_arrays;
    int successful_assignments = 0;
    
    for (const auto& op : batch_operations) {
        int array_id = scheduler.find_best_array_for_operation(op);
        if (array_id >= 0 && scheduler.reserve_array(array_id)) {
            assigned_arrays.push_back(array_id);
            successful_assignments++;
        }
    }
    
    // 验证调度结果
    EXPECT_EQ_MSG(8, successful_assignments, "应该成功调度8个操作（受阵列数量限制）");
    EXPECT_EQ_MSG(0, scheduler.get_available_array_count(), "所有阵列都应该被占用");
    
    // 释放所有阵列
    for (int array_id : assigned_arrays) {
        scheduler.release_array(array_id);
    }
    
    EXPECT_EQ_MSG(8, scheduler.get_available_array_count(), "释放后所有阵列都应该可用");
    
    std::cout << "✅ CIM阵列利用率测试成功，调度了 " << successful_assignments << " 个操作" << std::endl;
}

#ifdef USE_BUILTIN_TEST
// 运行CIM阵列测试的函数
int run_cim_array_tests() {
    std::cout << "\n=== YICA CIM阵列优化测试 ===" << std::endl;
    
    int failed_tests = 0;
    
    try {
        std::cout << "\n[TEST] CIMArrayInitialization" << std::endl;
        YICABasic_CIMArrayInitialization_Test test1;
        test1.TestBody();
        std::cout << "✅ PASSED" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "❌ FAILED: " << e.what() << std::endl;
        failed_tests++;
    }
    
    try {
        std::cout << "\n[TEST] CIMArrayScheduling" << std::endl;
        YICABasic_CIMArrayScheduling_Test test2;
        test2.TestBody();
        std::cout << "✅ PASSED" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "❌ FAILED: " << e.what() << std::endl;
        failed_tests++;
    }
    
    try {
        std::cout << "\n[TEST] CIMOperationMapping" << std::endl;
        YICABasic_CIMOperationMapping_Test test3;
        test3.TestBody();
        std::cout << "✅ PASSED" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "❌ FAILED: " << e.what() << std::endl;
        failed_tests++;
    }
    
    try {
        std::cout << "\n[TEST] CIMPerformanceEstimation" << std::endl;
        YICABasic_CIMPerformanceEstimation_Test test4;
        test4.TestBody();
        std::cout << "✅ PASSED" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "❌ FAILED: " << e.what() << std::endl;
        failed_tests++;
    }
    
    try {
        std::cout << "\n[TEST] CIMArrayUtilization" << std::endl;
        YICABasic_CIMArrayUtilization_Test test5;
        test5.TestBody();
        std::cout << "✅ PASSED" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "❌ FAILED: " << e.what() << std::endl;
        failed_tests++;
    }
    
    std::cout << "\n=== CIM阵列测试总结 ===" << std::endl;
    std::cout << "总测试数: 5" << std::endl;
    std::cout << "成功: " << (5 - failed_tests) << std::endl;
    std::cout << "失败: " << failed_tests << std::endl;
    
    return failed_tests;
}
#endif 
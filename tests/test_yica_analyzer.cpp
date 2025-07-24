#include "yica_analyzer.h"
#include <iostream>
#include <cassert>
#include <chrono>

using namespace yica::analyzer;

// 创建测试用的矩阵乘法算子
OperatorNode create_matmul_op(int64_t M, int64_t N, int64_t K, const std::string& dtype = "fp16") {
    OperatorNode op;
    op.op_type = OperatorNode::MATMUL;
    op.op_name = "test_matmul";
    
    // 输入张量A: M x K
    OperatorNode::TensorDesc tensor_A;
    tensor_A.shape = {M, K};
    tensor_A.dtype = dtype;
    tensor_A.size_bytes = M * K * (dtype == "fp16" ? 2 : 4);
    
    // 输入张量B: K x N
    OperatorNode::TensorDesc tensor_B;
    tensor_B.shape = {K, N};
    tensor_B.dtype = dtype;
    tensor_B.size_bytes = K * N * (dtype == "fp16" ? 2 : 4);
    
    // 输出张量C: M x N
    OperatorNode::TensorDesc tensor_C;
    tensor_C.shape = {M, N};
    tensor_C.dtype = dtype;
    tensor_C.size_bytes = M * N * (dtype == "fp16" ? 2 : 4);
    
    op.input_tensors = {tensor_A, tensor_B};
    op.output_tensors = {tensor_C};
    
    // 估算FLOPS: 2 * M * N * K (乘法和加法)
    op.flops = 2 * M * N * K;
    
    // 估算内存访问: 读取A和B，写入C
    op.memory_accesses = tensor_A.size_bytes + tensor_B.size_bytes + tensor_C.size_bytes;
    
    return op;
}

// 测试基本配置
void test_yica_config() {
    std::cout << "Testing YICA Configuration..." << std::endl;
    
    // 测试默认配置
    auto config = YICAArchConfig::get_default_config();
    assert(config.is_valid());
    assert(config.cim_array_rows == 512);
    assert(config.cim_array_cols == 512);
    assert(config.num_cim_dies == 32);
    
    std::cout << "✓ YICA Configuration tests passed!" << std::endl;
}

// 测试算子创建和基本属性
void test_operator_creation() {
    std::cout << "Testing Operator Creation..." << std::endl;
    
    // 测试矩阵乘法算子
    auto matmul_op = create_matmul_op(1024, 1024, 1024, "fp16");
    assert(matmul_op.is_cim_friendly());
    assert(matmul_op.flops > 0);
    assert(matmul_op.input_tensors.size() == 2);
    assert(matmul_op.output_tensors.size() == 1);
    
    std::cout << "✓ Operator creation tests passed!" << std::endl;
}

// 测试YICA分析器
void test_yica_analyzer() {
    std::cout << "Testing YICA Architecture Analyzer..." << std::endl;
    
    auto config = YICAArchConfig::get_default_config();
    YICAArchitectureAnalyzer analyzer(config);
    
    // 创建测试计算图
    ComputeGraph graph;
    graph.operators.push_back(create_matmul_op(2048, 2048, 2048, "fp16"));
    
    // 执行分析
    auto start_time = std::chrono::high_resolution_clock::now();
    auto result = analyzer.analyze_computation_pattern(graph);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Analysis time: " << duration.count() << " ms" << std::endl;
    
    // 验证分析结果
    assert(result.cim_friendliness_score >= 0.0f && result.cim_friendliness_score <= 1.0f);
    assert(result.memory_locality_score >= 0.0f && result.memory_locality_score <= 1.0f);
    assert(result.overall_yica_suitability >= 0.0f && result.overall_yica_suitability <= 1.0f);
    
    // 打印分析结果
    std::cout << "Overall YICA Suitability: " << result.overall_yica_suitability * 100 << "%" << std::endl;
    std::cout << "CIM Friendliness: " << result.cim_friendliness_score * 100 << "%" << std::endl;
    std::cout << "Memory Locality: " << result.memory_locality_score * 100 << "%" << std::endl;
    
    std::cout << "✓ YICA analyzer tests passed!" << std::endl;
}

int main() {
    std::cout << "YICA Architecture Analyzer Tests\n";
    std::cout << "================================\n" << std::endl;
    
    try {
        test_yica_config();
        test_operator_creation();
        test_yica_analyzer();
        
        std::cout << "\n🎉 All tests passed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 
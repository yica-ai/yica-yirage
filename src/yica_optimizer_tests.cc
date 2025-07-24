
// YICA 转换优化工具测试
#include <iostream>
#include <cassert>
#include <string>

// 声明外部接口
extern "C" {
    void yica_optimizer_init();
    const char* yica_optimize_code(const char* input_code, const char* target_backend);
    bool yica_is_backend_available(const char* backend);
}

int main() {
    std::cout << "运行YICA转换优化工具测试..." << std::endl;
    
    // 初始化测试
    yica_optimizer_init();
    
    // 测试1: 基础优化功能
    {
        const char* input = "void test() { }";
        const char* result = yica_optimize_code(input, "cpu");
        assert(result != nullptr);
        std::cout << "✓ 基础优化功能测试通过" << std::endl;
    }
    
    // 测试2: 后端可用性检查
    {
        assert(yica_is_backend_available("cpu"));
        assert(yica_is_backend_available("gpu"));
        assert(yica_is_backend_available("yica"));
        std::cout << "✓ 后端可用性测试通过" << std::endl;
    }
    
    // 测试3: 不同后端转换
    {
        const char* input = "void example() { }";
        
        const char* cpu_result = yica_optimize_code(input, "cpu");
        const char* gpu_result = yica_optimize_code(input, "gpu");
        const char* yica_result = yica_optimize_code(input, "yica");
        
        assert(cpu_result != nullptr);
        assert(gpu_result != nullptr);
        assert(yica_result != nullptr);
        
        std::cout << "✓ 多后端转换测试通过" << std::endl;
    }
    
    std::cout << "\n✅ 所有测试通过！" << std::endl;
    std::cout << "YICA转换优化工具功能正常" << std::endl;
    
    return 0;
}
